"""
Dataset adapter that pairs `human_rgb_pick_place` demo videos with `ur5e_pick_place` agent
trajectories (Multi-Task-LFD-Training-Framework's on-disk dataset layout/pkl format) and produces
the exact `(context, traj)` batch contract osvi-awda's `InverseImitation` model / SDTW training
loop expects (see `scripts/train_transformer.py::forward` and
`hem/datasets/agent_teacher_dataset.py::AgentTeacherDataset`).

Only the keys actually consumed by the waypoints=True training path are populated with real data
(`images`, `traj_points`, `head_label`, `setting_name`, and the demo `video`); a few keys the model
code reads but never uses on this path (`states`, `actions`) are filled with cheap placeholders —
see docs/02_training_adaptation.md for the full trace of which keys are load-bearing.

`traj['projection_matrix']` (from `_make_agent_sample`) is real, calibrated camera geometry, built
per-sample by `mtlfd_adaptation.camera_projection.build_sample_projection` - load-bearing whenever
the experiment config sets `image_waypoints: True` (see `scripts/train_transformer.py::
compute_loss_trajectory`), harmless/unused otherwise. `context['projection_matrix']` (from
`_make_context`) stays an identity placeholder since it's only read when `policy.vis.context_only`
is set, which no experiment config here uses.
"""
import glob
import itertools
import os
import random

import numpy as np
from torch.utils.data import Dataset

from hem.datasets.util import crop as crop_fn
from hem.datasets.util import randomize_video, resize
from mtlfd_adaptation.camera_projection import build_sample_projection
from mtlfd_adaptation.camera_projection_real import build_sample_projection as \
    build_sample_projection_real
from mtlfd_adaptation.trajectory_bridge import load_traj

ALL_PICK_PLACE_TASKS = list(range(16))

# raw trajectory steps to skip before treating a frame as "o1" (the conditioning image) - see
# _make_agent_sample's comment; 1 is sufficient per empirical check (object bounding boxes stop
# moving entirely by t=1 across every sampled trajectory).
AGENT_SETTLE_STEPS = 1


class MTLFDAgentTeacherDataset(Dataset):
    def __init__(self, root_dir, task_name='pick_place', agent_name='ur5e', demo_name='human_rgb',
                 mode='train', test_tasks=(12, 13, 14, 15), train_tasks=None,
                 traj_per_task=None, demo_per_task=None,
                 T_context=10, T_pair=1, agent_context=0, height=100, width=180,
                 demo_height=None, demo_width=None,
                 crop=(0, 0, 0, 0), demo_crop=(0, 0, 0, 0),
                 rand_flip=False, flip_sync=True, color_jitter=None, rand_crop=None,
                 rand_translate=None, sample_sides=True, extra_samp_bound=0.4,
                 waypoints=True, grasp=True, is_real=False, **_ignored):
        assert mode in ('train', 'val', 'test'), f'unsupported mode {mode!r}'
        self.root_dir = root_dir
        self.task_name = task_name
        self.agent_name = agent_name
        self.demo_name = demo_name
        self.mode = mode
        self.is_real = is_real
        self.T_context = T_context
        self.T_pair = T_pair
        self.agent_context = agent_context
        self.height, self.width = height, width
        self.demo_height = demo_height or height
        self.demo_width = demo_width or width
        self.crop = tuple(crop)
        self.demo_crop = tuple(demo_crop)
        self.rand_flip = rand_flip
        self.flip_sync = flip_sync
        self.color_jitter = color_jitter
        self.rand_crop = rand_crop
        self.rand_translate = np.array(rand_translate) if rand_translate is not None else np.array([0, 0])
        self.sample_sides = sample_sides
        self.extra_samp_bound = extra_samp_bound
        self.waypoints = waypoints
        self.grasp = grasp

        test_tasks = sorted(set(test_tasks))
        if train_tasks is None:
            train_tasks = [t for t in ALL_PICK_PLACE_TASKS if t not in test_tasks]
        self.train_tasks, self.test_tasks = sorted(train_tasks), test_tasks
        # mirrors AgentTeacherDataset's mosaic branch: 'train' -> train_tasks, anything else
        # (matches Trainer's own use of mode='test' for its "val" loader) -> held-out test_tasks.
        subtasks = self.train_tasks if mode == 'train' else self.test_tasks

        agent_dir = os.path.join(root_dir, task_name, f'{agent_name}_{task_name}')
        demo_dir = os.path.join(root_dir, task_name, f'{demo_name}_{task_name}')

        self.agent_files, self.demo_files = [], []
        self.pairs = []
        # task_ids[i] = which task_id self.pairs[i] belongs to - parallel array, used by
        # hem.models.trainer.PerTaskBatchSampler (config: samples_per_task) to build batches with
        # an exact, guaranteed-per-batch task count, rather than relying on shuffle=True's
        # population-level balance (see mtlfd_adaptation session notes: every task already has
        # identical agent/demo file counts, so plain shuffling is balanced IN EXPECTATION - this
        # exists for exact per-batch control instead, e.g. to test whether that reduces
        # gradient-noise-driven asymmetries between tasks/objects).
        self.task_ids = []
        for subtask in subtasks:
            sub = f'task_{subtask:02d}'
            a_files = sorted(glob.glob(os.path.join(agent_dir, sub, '*.pkl')))
            d_files = sorted(glob.glob(os.path.join(demo_dir, sub, '*.pkl')))
            assert a_files, f'no agent trajectories found in {os.path.join(agent_dir, sub)}'
            assert d_files, f'no demo trajectories found in {os.path.join(demo_dir, sub)}'
            if traj_per_task:
                a_files = a_files[:traj_per_task]
            if demo_per_task:
                d_files = d_files[:demo_per_task]
            a_start, d_start = len(self.agent_files), len(self.demo_files)
            self.agent_files.extend(a_files)
            self.demo_files.extend(d_files)
            a_inds = range(a_start, a_start + len(a_files))
            d_inds = range(d_start, d_start + len(d_files))
            self.pairs.extend(itertools.product(a_inds, d_inds))
            self.task_ids.extend([subtask] * (len(a_files) * len(d_files)))

        # camera_projection[_real].CANVAS_SIZE is a hardcoded constant derived from the relevant
        # camera's known resolution - verify it still matches the actual stored images once, here,
        # rather than silently building a wrong per-sample projection_matrix if the dataset is ever
        # regenerated at a different resolution.
        if self.is_real:
            from mtlfd_adaptation.camera_projection_real import CANVAS_SIZE
        else:
            from mtlfd_adaptation.camera_projection import CANVAS_SIZE
        sample_traj, _ = load_traj(self.agent_files[0], is_real=self.is_real)
        actual_shape = sample_traj.get(0)['obs']['image'].shape[:2]
        assert tuple(actual_shape) == CANVAS_SIZE, (
            f'camera_projection{"_real" if self.is_real else ""}.CANVAS_SIZE={CANVAS_SIZE} does '
            f'not match actual agent image shape {actual_shape} - the hardcoded camera '
            f'intrinsic/extrinsic were derived assuming CANVAS_SIZE; projection_matrix will be '
            f'wrong until updated.')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        a_i, d_i = self.pairs[index % len(self.pairs)]
        force_flip = None
        if self.flip_sync and self.rand_flip:
            force_flip = [(-1 if random.random() > 0.5 else 1),
                          (-1 if random.random() > 0.5 else 1)]

        agent_traj, _ = load_traj(self.agent_files[a_i], is_real=self.is_real)
        demo_traj, _ = load_traj(self.demo_files[d_i])

        context = self._make_context(demo_traj, force_flip=force_flip)
        traj = self._make_agent_sample(agent_traj, force_flip=force_flip)
        return context, traj

    # ------------------------------------------------------------------
    # demo (human_rgb) -> context video, mirrors AgentDemonstrations._make_context
    # ------------------------------------------------------------------
    def _make_context(self, traj, force_flip=None):
        clip = lambda x: int(max(0, min(x, len(traj) - 1)))
        per_bracket = max(len(traj) / self.T_context, 1)
        frames = []
        for i in range(self.T_context):
            n = clip(np.random.randint(int(i * per_bracket), int((i + 1) * per_bracket)))
            if self.sample_sides and i == self.T_context - 1:
                n = len(traj) - 1
            elif self.sample_sides and i == 0:
                n = 0
            img = crop_fn(traj.get(n)['obs']['image'], self.demo_crop)
            img = resize(img, (self.demo_width, self.demo_height))
            frames.append(img[None])
        frames = np.concatenate(frames, 0)
        frames, _ = randomize_video(frames, self.color_jitter, None, self.rand_crop, 0,
                                     self.rand_translate, True, rand_flip=self.rand_flip,
                                     force_flip=force_flip)

        return {
            'video': np.transpose(frames, (0, 3, 1, 2)).astype(np.float32),
            'projection_matrix': np.eye(4, dtype=np.float32),
            'fname': traj.fname,
        }

    # ------------------------------------------------------------------
    # agent (ur5e) -> images + attributed-waypoint supervision, mirrors
    # AgentDemonstrations._get_pairs's waypoints=True branch
    # ------------------------------------------------------------------
    def _make_agent_sample(self, traj, force_flip=None):
        elements = [x for x in traj]
        n = len(elements)
        # waypoints=True always anchors the sampled window near the trajectory start (o1 = first
        # usable frame), matching AgentDemonstrations._get_pairs (`if self.waypoints: start = 0`) -
        # except raw index 0 itself is skipped: objects placed at env.reset() are still settling
        # onto the table in that very first recorded frame (confirmed empirically - obj_bb centers
        # shift ~5-10px between t=0 and t=1 across sampled trajectories, then stay pixel-identical
        # from t=1 onward), so o1=elements[0] would condition the model on a transient scene. This
        # mirrors test_mtlfd_rollout.py's own stabilize() step, which exists for the same reason on
        # the live-rollout side.
        start = AGENT_SETTLE_STEPS if self.waypoints else np.random.randint(0, max(1, n - self.T_pair))
        chosen_t = [min(j + start, n - 1) for j in range(self.T_pair + 1)]

        images = []
        for t in chosen_t:
            img = crop_fn(elements[t]['obs']['image'], self.crop)
            img = resize(img, (self.width, self.height))
            images.append(img[None])
        images = np.concatenate(images, 0)
        images, stats = randomize_video(images, self.color_jitter, None, self.rand_crop, 0,
                                         self.rand_translate, True, rand_flip=self.rand_flip,
                                         force_flip=force_flip)
        proj_fn = build_sample_projection_real if self.is_real else build_sample_projection
        projection_matrix = proj_fn(self.crop, stats, (self.height, self.width))

        images = np.transpose(images, (0, 3, 1, 2)).astype(np.float32)

        # grasp attribute mining: 'grasp' isn't in ur5e_pick_place's obs, so use the same
        # gripper-command threshold osvi-awda's own pick-place branch uses (action[-1] > 0.01).
        # elements[i]['action'] is the action that CAUSED the transition INTO elements[i]['obs']
        # (traj_bridge/the Trajectory saver append the new obs together with the action that
        # produced it - elements[0] has no action at all, confirming this), not the action about
        # to be taken from it. So the command "issued while at position i" lives at elements[i+1],
        # one raw step later. Pairing action[i] with poses[i] (as hem/datasets/agent_dataset.py's
        # identical sim-data formula does) mislabels the grasp position as the frame AFTER the
        # gripper has already started closing rather than the frame it was commanded from -
        # measured as a consistent ~1.7cm downward offset on real ur5e_pick_place trajectories
        # (sim's baseline doesn't show this since a simulated gripper closes in ~1 physics step,
        # no visible transient). Shifting the flags back by one raw index fixes this.
        grasp_frames = [elements[i + 1]['action'][-1] > 0.01 for i in range(n - 1)] + [False]

        out_inds = np.linspace(0, n - 1, num=50, endpoint=True, dtype=int)
        # real obs['ee_aa'] is 3-dim axis-angle ONLY (no position), unlike sim's 6-dim
        # [pos(3), axis_angle(3)] - real position lives in its own obs['eef_pos'] field instead.
        pos_key = 'eef_pos' if self.is_real else 'ee_aa'
        poses = np.stack([
            elements[i]['obs'][pos_key] if self.is_real else elements[i]['obs'][pos_key][:3]
            for i in out_inds
        ]).astype(np.float32)
        grasps = np.stack([grasp_frames[i] for i in out_inds]).astype(np.int32)
        traj_points = np.concatenate((poses, grasps[:, None] * 0.2), axis=-1).astype(np.float32)

        grasp_ind = np.where(grasps)[0]
        if len(grasp_ind) and self.grasp:
            grasp_point = poses[grasp_ind[0]]
        else:
            grasp_point = np.zeros(3, dtype=np.float32)

        T = images.shape[0]
        return {
            'images': images,
            # unused by the waypoints=True forward path (see module docstring) - cheap placeholders
            'states': np.zeros((T, 1), dtype=np.float32),
            'actions': np.zeros((max(T - 1, 0), 7), dtype=np.float32),
            'projection_matrix': projection_matrix,
            # real, load-bearing fields:
            'traj_points': traj_points,
            'grasp_point': grasp_point,
            'head_label': 0,
            'setting_name': traj.setting_name,
        }
