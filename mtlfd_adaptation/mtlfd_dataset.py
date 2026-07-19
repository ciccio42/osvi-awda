"""
Dataset adapter that pairs `human_rgb_pick_place` demo videos with `ur5e_pick_place` agent
trajectories (Multi-Task-LFD-Training-Framework's on-disk dataset layout/pkl format) and produces
the exact `(context, traj)` batch contract osvi-awda's `InverseImitation` model / SDTW training
loop expects (see `scripts/train_transformer.py::forward` and
`hem/datasets/agent_teacher_dataset.py::AgentTeacherDataset`).

Only the keys actually consumed by the waypoints=True training path are populated with real data
(`images`, `traj_points`, `head_label`, `setting_name`, and the demo `video`); a few keys the model
code reads but never uses on this path (`states`, `actions`, `grasp_point`, `projection_matrix`)
are filled with cheap placeholders — see docs/02_training_adaptation.md for the full trace of which
keys are load-bearing.
"""
import glob
import itertools
import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset

from hem.datasets.util import crop as crop_fn
from hem.datasets.util import randomize_video, resize
from mtlfd_adaptation import debug_utils
from mtlfd_adaptation.trajectory_bridge import load_traj
from PIL import Image

ALL_PICK_PLACE_TASKS = list(range(16))


class MTLFDAgentTeacherDataset(Dataset):
    def __init__(self, root_dir, task_name='pick_place', agent_name='ur5e', demo_name='human_rgb',
                 mode='train', test_tasks=(12, 13, 14, 15), train_tasks=None,
                 traj_per_task=None, demo_per_task=None,
                 T_context=10, T_pair=1, agent_context=0, height=100, width=180,
                 demo_height=None, demo_width=None,
                 crop=(0, 0, 0, 0), demo_crop=(0, 0, 0, 0),
                 rand_flip=False, flip_sync=True, color_jitter=None, rand_crop=None,
                 rand_translate=None, sample_sides=True, extra_samp_bound=0.4,
                 waypoints=True, grasp=True, **_ignored):
        assert mode in ('train', 'val', 'test'), f'unsupported mode {mode!r}'
        self.root_dir = root_dir
        self.task_name = task_name
        self.agent_name = agent_name
        self.demo_name = demo_name
        self.mode = mode
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

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        a_i, d_i = self.pairs[index % len(self.pairs)]
        force_flip = None
        if self.flip_sync and self.rand_flip:
            force_flip = [(-1 if random.random() > 0.5 else 1),
                          (-1 if random.random() > 0.5 else 1)]

        agent_traj, _ = load_traj(self.agent_files[a_i])
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
        orig_frames = frames.astype(np.uint8).copy()  # pre-augmentation, for debug comparison
        frames, _ = randomize_video(frames, self.color_jitter, None, self.rand_crop, 0,
                                     self.rand_translate, True, rand_flip=self.rand_flip,
                                     force_flip=force_flip)
        # save original vs augmented frames side by side, paired by index
        for i, (orig, aug) in enumerate(zip(orig_frames, frames)):
            cv2.imwrite(f"context_frame_{i}_orig.png", cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"context_frame_{i}_aug.png",
                        debug_utils.unnormalize(np.transpose(aug, (2, 0, 1))))

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
        # waypoints=True always anchors the sampled window at the trajectory start (o1 = first
        # frame), matching AgentDemonstrations._get_pairs (`if self.waypoints: start = 0`).
        start = 0 if self.waypoints else np.random.randint(0, max(1, n - self.T_pair))
        chosen_t = [min(j + start, n - 1) for j in range(self.T_pair + 1)]

        images = []
        for t in chosen_t:
            img = crop_fn(elements[t]['obs']['image'], self.crop)
            img = resize(img, (self.width, self.height))
            images.append(img[None])
        images = np.concatenate(images, 0)
        images, _ = randomize_video(images, self.color_jitter, None, self.rand_crop, 0,
                                     self.rand_translate, True, rand_flip=self.rand_flip,
                                     force_flip=force_flip)
        # for i, img in enumerate(images):
        #     pil_img = Image.fromarray(img.astype(np.uint8))
        #     pil_img.save(f"agent_frame_{i}.png")
        
        images = np.transpose(images, (0, 3, 1, 2)).astype(np.float32)

        # grasp attribute mining: 'grasp' isn't in ur5e_pick_place's obs, so use the same
        # gripper-command threshold osvi-awda's own pick-place branch uses (action[-1] > 0.01).
        grasp_frames = [False] + [elements[i]['action'][-1] > 0.01 for i in range(1, n)]

        out_inds = np.linspace(0, n - 1, num=50, endpoint=True, dtype=int)
        poses = np.stack([elements[i]['obs']['ee_aa'][:3] for i in out_inds]).astype(np.float32)
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
            'projection_matrix': np.eye(4, dtype=np.float32),
            # real, load-bearing fields:
            'traj_points': traj_points,
            'grasp_point': grasp_point,
            'head_label': 0,
            'setting_name': traj.setting_name,
        }
