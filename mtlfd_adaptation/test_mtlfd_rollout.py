"""
Rollout / evaluation harness for an OSVI-AWDA checkpoint trained on
Multi-Task-LFD-Training-Framework's pick_place data (train_mtlfd.py), driving a *live* UR5e
pick_place robosuite simulation from that framework. See docs/03_test_adaptation.md.

Unlike Multi-Task-LFD-Training-Framework's own test/multi_task_test/test_any_task.py (which
re-queries a BC policy every timestep), OSVI-AWDA infers its full attributed-waypoint plan ONCE
from (v, o1) and then executes it open-loop via 4 hand-crafted motor primitives (paper section
IV-A / Appendix VII): free-space motion, grasping (depth-camera object localization), carrying,
dropping.

Requires an env with multi_task_il / multi_task_robosuite_env / multi_task_test importable (see
docs/03) - run via mtlfd_adaptation/test_mtlfd.sh.

Usage:
    python -u mtlfd_adaptation/test_mtlfd_rollout.py <checkpoint_dir> --saved_step 20000 \
        --episodes 10 --task_ids 12 13 14 15 --results_dir <out_dir>
"""
import argparse
import copy
import glob
import json
import os
import random
import re
import sys

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import yaml  # noqa: E402
import debugpy
# NOTE: these third-party/pip-installed packages must be imported BEFORE osvi-awda's own repo
# root goes on sys.path below - osvi-awda has its own top-level `robosuite/` directory (an old
# vendored fork used by its own scripts) that would otherwise shadow the real, pip-installed
# `robosuite` package (the one with UR5e/OSC_POSE support) once REPO_ROOT takes priority. Once
# imported here, the correct module is cached in sys.modules and safe to use/re-import anywhere
# else (including inside mtlfd_adaptation.robosuite_camera_utils).
import robosuite.utils.transform_utils as T  # noqa: E402
from robosuite import load_controller_config  # noqa: E402
from robosuite.utils import RandomizationError  # noqa: E402
from multi_task_robosuite_env import get_env  # noqa: E402
from multi_task_test.utils import check_bin, check_pick, check_reach  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hem.models.inverse_module import InverseImitation  # noqa: E402
from hem.datasets.util import crop as crop_fn  # noqa: E402
from hem.datasets.util import randomize_video, resize  # noqa: E402
from mtlfd_adaptation import debug_utils  # noqa: E402
from mtlfd_adaptation.camera_projection import (  # noqa: E402
    NO_AUGMENTATION_STATS, build_sample_projection, project_normalized_depth_to_world,
    project_world_to_final_pixel,
)
from mtlfd_adaptation.mtlfd_dataset import AGENT_SETTLE_STEPS, MTLFDAgentTeacherDataset  # noqa: E402
from mtlfd_adaptation.trajectory_bridge import load_traj  # noqa: E402
from mtlfd_adaptation.robosuite_camera_utils import (  # noqa: E402
    get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map, pixel_to_world,
)

def seed_everything(seed=42):
    """Global seeding, ported verbatim from Multi-Task-LFD-Training-Framework's
    test/multi_task_test/test_any_task.py so both harnesses are reproducible the same way."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _task_id_from_path(path):
    """Extract the task_NN directory component's integer id from an agent/demo pkl path."""
    m = re.search(r'task_(\d+)', path)
    assert m, f'could not find task_NN in path {path!r}'
    return int(m.group(1))


def build_training_pairs_by_task(ds_cfg):
    """Build the checkpoint's OWN mode='train' (demo_file, agent_file) pairs, grouped by task_id.

    Reconstructs MTLFDAgentTeacherDataset exactly as train_mtlfd.py would have (same ds_cfg, same
    mode='train'), so the returned (demo, agent) pairs are genuine training-set couples - not just
    any two files that happen to live under the same task_NN directory (which can include files
    beyond traj_per_task/demo_per_task limits, or - for checkpoints whose test_tasks is a subset of
    train_tasks - files that are technically fine but weren't independently verified as members).
    """
    dataset = MTLFDAgentTeacherDataset(mode='train', **ds_cfg)
    pairs_by_task = {}
    for a_i, d_i in dataset.pairs:
        task_id = _task_id_from_path(dataset.agent_files[a_i])
        pairs_by_task.setdefault(task_id, []).append((dataset.agent_files[a_i], dataset.demo_files[d_i]))
    return pairs_by_task


GRASP_THRESHOLD = 0.1     # predicted grasp attribute (0..0.2 scale) above this -> "grasp on"
CONTROLLER_PATH = os.path.join(
    os.path.dirname(REPO_ROOT), 'Multi-Task-LFD-Training-Framework',
    'tasks', 'multi_task_robosuite_env', 'controllers', 'config', 'osc_pose.json')
BIN_NAMES = ['bin_box_1', 'bin_box_2', 'bin_box_3', 'bin_box_4']
DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])   # gripper-facing-down orientation, world frame


# ----------------------------------------------------------------------------
# low-level control
# ----------------------------------------------------------------------------
def current_pose(env):
    eef_id = env.robots[0].eef_site_id
    pos = np.array(env.sim.data.site_xpos[eef_id])
    quat = T.mat2quat(np.reshape(env.sim.data.site_xmat[eef_id], (3, 3)))
    return pos, quat


class EpisodeDone(Exception):
    """Raised by move_to when the env reports done=True (horizon/success), so every primitive
    call site (however deeply nested) stops issuing further env.step() calls - robosuite raises a
    hard ValueError on any step() after termination, so this must be checked eagerly, not just
    inside move_to's own loop."""


def move_to(env, target_pos, target_quat=DOWN_QUAT, gripper=-1.0, pos_tol=0.008, max_iters=50,
            max_step=0.02, on_step=None):
    """Closed-loop step-towards-target (mirrors test/multi_task_test/primitive.py's
    reaching_primitive, generalized to full 3D + explicit target orientation each call)."""
    target_aa = T.quat2axisangle(target_quat)
    obs = None
    for t in range(max_iters):
        pos, _ = current_pose(env)
        delta = np.clip(target_pos - pos, -max_step, max_step)
        action = np.concatenate([pos + delta, target_aa, [gripper]])
        obs, reward, done, info = env.step(action)
        if on_step is not None:
            on_step(obs)
        if np.linalg.norm(target_pos - pos) < pos_tol:
            break
        if done:
            raise EpisodeDone()
    return obs


def stabilize(env, n_steps=20):
    """Hold the current end-effector pose and step the sim so objects placed at env.reset() (which
    can start mid-air or with residual velocity from randomization) settle onto the table before
    start_pos and the initial camera frame are read - otherwise the policy's plan is conditioned
    on a transient, not-yet-physically-settled scene."""
    pos, quat = current_pose(env)
    action = np.concatenate([pos, T.quat2axisangle(quat), [-1.0]])
    obs = None
    for _ in range(n_steps):
        obs, reward, done, info = env.step(action)
        if done:
            raise EpisodeDone()
    return obs


def set_objects_from_training_trajectory(env, agent_file, timestep=None, debug_path=None):
    """Teleports env's colored-box objects (NOT the robot arm or bin) to the exact layout recorded
    in a real training trajectory pkl, so a live rollout can be run inside a replica of a real
    training scenario - isolates "is live rendering itself the problem" from "is an unseen/random
    object layout the problem" (see test_training_pointing_accuracy.py, which measures accuracy
    directly on training pkls with no live env at all, for the other half of that comparison).

    Call right after env.reset() (before stabilize()) - reset's own random placement gets
    overwritten here, then stabilize() lets physics settle the newly-placed objects normally.

    debug_path: if given, saves the trajectory's OWN stored camera_front frame at `timestep`
    (i.e. exactly what the frame looked like at data-collection time, before any live re-rendering)
    as a PNG - the reference to compare the live teleported scene against (see
    verify_scene_replay.py, which did this comparison ad hoc; this bakes the same idea into the
    normal rollout debug output).

    Ground truth comes from the trajectory's raw mujoco state snapshot
    (`AliasedTrajectory.get_raw_state`, exactly `sim.get_state().flatten()` at collection time -
    see Multi-Task-LFD-Training-Framework/tasks/collect_data/rollout_trajectory.py's own use of
    `sim.set_state_from_flattened` for the same replay purpose), NOT obj_bb (pixel-space only) or
    per-object `{name}_pos` obs fields - this dataset variant doesn't store those (verified
    empirically against ur5e_pick_place/task_*/*.pkl; some other trajectory variants in this repo
    do, which is what rollout_trajectory.py's own init_env() instead relies on).

    The flattened state's layout (validated: rs[1:7]/rs[7:13] match obs['joint_pos']/
    obs['gripper_qpos'] exactly, for the SAME timestep, to float precision):
        [0]                          time
        [1:7]                        6 UR5e arm joint qpos
        [7:13]                       6 gripper joint qpos
        [13 : 13+7*len(env.objects)] one [x, y, z, qw, qx, qy, qz] block per object, in the SAME
                                      order env.objects was built in (new_pp.py's
                                      object_seq/obj_names order) - valid here because the
                                      trajectory and this live env are built from the exact same
                                      env class/object_set (UR5e_PickPlaceDistractor,
                                      object_set=2).
    """
    agent_traj, _ = load_traj(agent_file)
    t = AGENT_SETTLE_STEPS if timestep is None else timestep
    raw_state = agent_traj.get_raw_state(t)

    if debug_path is not None:
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        Image.fromarray(agent_traj.get(t)['obs']['image']).save(debug_path)

    n_obj = len(env.objects)
    obj_qpos = np.asarray(raw_state[13:13 + 7 * n_obj], dtype=np.float64).reshape(n_obj, 7)
    for obj, qpos in zip(env.objects, obj_qpos):
        env.sim.data.set_joint_qpos(obj.joints[-1], qpos)
    env.sim.forward()


# ----------------------------------------------------------------------------
# depth-based object localization (paper Appendix VII-A)
# ----------------------------------------------------------------------------
def _save_localize_debug_image(debug_path, rgb, real_depth, valid, mask, centroid, max_depth):
    """Single side-by-side PNG: RGB (left) + depth (right, JET-colormapped over [0, max_depth],
    invalid/background pixels forced black), both annotated with the detected centroid (if any) so
    a failed detection (no centroid) is visually distinguishable from a successful one."""
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()

    depth_vis = np.clip(real_depth, 0, max_depth) / max(max_depth, 1e-6)
    depth_vis = (depth_vis * 255).astype(np.uint8)
    depth_bgr = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    depth_bgr[~valid] = 0
    if mask is not None:
        # thin white outline around the "above floor plane" mask actually used for connected
        # components, so it's clear WHICH blob (if any) got picked below.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(depth_bgr, contours, -1, (255, 255, 255), 1)

    for panel in (rgb_bgr, depth_bgr):
        if centroid is not None:
            row, col = int(round(centroid[0])), int(round(centroid[1]))
            cv2.drawMarker(panel, (col, row), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        else:
            cv2.putText(panel, 'NO DETECTION', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 255), 1, cv2.LINE_AA)

    combined = np.concatenate([rgb_bgr, depth_bgr], axis=1)
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    cv2.imwrite(debug_path, combined)


def localize_grasp_target(env, camera_name='robot0_eye_in_hand', height=200, width=360,
                           floor_margin=0.01, max_depth=1.0, debug_path=None):
    """Paper Appendix VII-A object-localization procedure: mask out background by depth, estimate
    the floor plane as the median remaining depth, mask pixels >1cm above the floor, take
    connected components, pick the one closest to image center, backproject its centroid.

    debug_path: if given, saves a single RGB+depth composite (see _save_localize_debug_image) at
    this path, regardless of whether a target was actually found."""
    get_obs = getattr(env, '_get_observations', None) or getattr(env, '_get_observation')
    obs = get_obs()
    rgb = obs[f'{camera_name}_image']
    real_depth = obs[f'{camera_name}_depth']
    # real_depth = get_real_depth_map(env.sim, depth_norm)
    if real_depth.ndim == 3:
        real_depth = real_depth[:, :, 0]

    valid = real_depth < max_depth
    if not np.any(valid):
        if debug_path:
            _save_localize_debug_image(debug_path, rgb, real_depth, valid, None, None, max_depth)
        return None
    floor_depth = np.median(real_depth[valid])

    # height above floor plane, approximated as (floor_depth - real_depth) since the eye-in-hand
    # camera looks roughly straight down at the table during the grasp approach.
    above_floor = (floor_depth - real_depth) > floor_margin
    mask = (valid & above_floor).astype(np.uint8)
    if mask.sum() == 0:
        if debug_path:
            _save_localize_debug_image(debug_path, rgb, real_depth, valid, mask, None, max_depth)
        return None

    n_labels, labels = cv2.connectedComponents(mask)
    if n_labels <= 1:
        if debug_path:
            _save_localize_debug_image(debug_path, rgb, real_depth, valid, mask, None, max_depth)
        return None
    center = np.array([height / 2, width / 2])
    best_label, best_dist, best_centroid = None, None, None
    for label in range(1, n_labels):
        ys, xs = np.where(labels == label)
        if len(ys) < 4:
            continue
        centroid = np.array([ys.mean(), xs.mean()])
        dist = np.linalg.norm(centroid - center)
        if best_dist is None or dist < best_dist:
            best_dist, best_label, best_centroid = dist, label, centroid
    if best_label is None:
        if debug_path:
            _save_localize_debug_image(debug_path, rgb, real_depth, valid, mask, None, max_depth)
        return None

    if debug_path:
        _save_localize_debug_image(debug_path, rgb, real_depth, valid, mask, best_centroid, max_depth)

    row, col = int(round(best_centroid[0])), int(round(best_centroid[1]))
    K = get_camera_intrinsic_matrix(env.sim, camera_name, height, width)
    Rt = get_camera_extrinsic_matrix(env.sim, camera_name)
    return pixel_to_world(row, col, real_depth[row, col], K, Rt)


# ----------------------------------------------------------------------------
# 4 motor primitives (paper section IV-A)
# ----------------------------------------------------------------------------
def free_space_primitive(env, target_pos, holding, on_step=None):
    gripper = 1.0 if holding else -1.0
    approach = target_pos.copy()
    approach[2] = max(approach[2], current_pose(env)[0][2])
    move_to(env, approach, gripper=gripper, on_step=on_step)
    move_to(env, target_pos, gripper=gripper, on_step=on_step)


def grasp_primitive(env, hint_pos, on_step=None, debug_path=None):
    hover = hint_pos.copy()
    hover[2] += 0.10
    move_to(env, hover, gripper=-1.0, on_step=on_step)
    target = localize_grasp_target(env, debug_path=debug_path)
    # target = None
    if target is None:
        target = hint_pos
        target -= 0.05
    approach = target.copy()
    approach[2] += 0.05
    move_to(env, approach, gripper=-1.0, on_step=on_step)
    descend = target.copy()
    descend[2] = 0.75
    move_to(env, descend, gripper=-1.0, on_step=on_step)
    move_to(env, descend, gripper=1.0, max_iters=10, on_step=on_step)   # close
    lift = descend.copy()
    lift[2] += 0.15
    move_to(env, lift, gripper=1.0, on_step=on_step)


def drop_primitive(env, target_pos, on_step=None, debug_path=None, n_open_steps=10):
    # Was opening the gripper at whatever pose the PREVIOUS waypoint's free_space_primitive left
    # the arm at (never actually visiting target_pos, the predicted "in bin, release" waypoint) -
    # mirror grasp_primitive/free_space_primitive's approach-then-act pattern instead, so the
    # object is actually carried down into the bin before being released.
    move_to(env, target_pos, gripper=1.0, on_step=on_step)
    pos, quat = current_pose(env)
    action = np.concatenate([pos, T.quat2axisangle(quat), [-1.0]])
    # Hold the open-gripper command for several steps rather than firing it once - a single
    # env.step() isn't always enough simulated time for the gripper joint to actually reach fully
    # open (same reasoning as grasp_primitive's own close call, `move_to(..., max_iters=10)`), and
    # unlike that call target_pos here already equals current_pose (we just arrived via move_to
    # above), so move_to's position-tolerance early-break would exit after a single iteration -
    # loop directly instead so all n_open_steps commands actually get sent.
    obs = None
    for _ in range(n_open_steps):
        obs, reward, done, info = env.step(action)
        if on_step is not None:
            on_step(obs)
        if done:
            raise EpisodeDone()
    if debug_path is not None and obs is not None:
        # snapshot once the gripper has had time to fully open and the object has settled into
        # the bin, so you can visually confirm the place actually landed (vs. the fix above just
        # confirming the arm reached target_pos).
        debug_utils.save_rollout_frame(debug_path, cv2.cvtColor(obs['camera_front_image'], cv2.COLOR_RGB2BGR))


# ----------------------------------------------------------------------------
# demo/context + task-inference forward pass
# ----------------------------------------------------------------------------
def preprocess_frame(img, crop, height, width):
    img = crop_fn(img, crop)
    img = resize(img, (width, height))
    frames, _ = randomize_video(img[None], None, None, None, 0, np.array([0, 0]), True,
                                 rand_flip=False)
    return np.transpose(frames, (0, 3, 1, 2)).astype(np.float32)[0]


def make_demo_context(traj, T_context, height, width, crop, sample_sides=True):
    """Deterministic (no augmentation) version of MTLFDAgentTeacherDataset._make_context, for
    evaluation - matches training-time preprocessing modulo the random augmentation that's
    disabled at eval time anyway (see AgentTeacherDataset's own mode != 'train' handling)."""
    clip = lambda x: int(max(0, min(x, len(traj) - 1)))
    per_bracket = max(len(traj) / T_context, 1)
    frames = []
    for i in range(T_context):
        n = clip(int((i + 0.5) * per_bracket))
        if sample_sides and i == T_context - 1:
            n = len(traj) - 1
        elif sample_sides and i == 0:
            n = 0
        frames.append(preprocess_frame(traj.get(n)['obs']['image'], crop, height, width))
    return np.stack(frames, axis=0)


def load_model(model_dir, saved_step, device):
    with open(os.path.join(model_dir, 'config.yaml')) as f:
        config = yaml.safe_load(f)
    model = InverseImitation(**config['policy'])
    ckpt_path = os.path.join(model_dir, f'model_save-{saved_step}.pt')
    # weights_only=False: these checkpoints are torch.save()'d whole nn.Module objects (osvi-awda's
    # own hem/models/trainer.py convention), not plain state_dicts - torch>=2.6 defaults
    # weights_only=True which can't unpickle that. Safe here since we only ever load checkpoints
    # this same pipeline produced.
    loaded = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = loaded.state_dict() if hasattr(loaded, 'state_dict') else loaded
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model, config


def predict_waypoints(model, demo_video, o1_img, device):
    """demo_video: [T_context,C,H,W] np; o1_img: [C,H,W] np. Returns [5,4] relative waypoints."""
    context = torch.from_numpy(demo_video)[None].to(device)
    images = torch.from_numpy(np.stack([o1_img, o1_img]))[None].to(device)
    states = torch.zeros((1, 2, 1), device=device)
    ents = torch.zeros((1,), dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(states, images, context, ret_dist=False, ents=ents)
    waypoints = out['waypoints'][0].cpu().numpy()   # [15,4]
    return waypoints[-5:]   # last consecutive block = the "5-waypoint" trajectory (see docs/03)


def draw_waypoints_on_frame(img, abs_positions, grasp_attrs, crop, height, width, indices=None,
                             scale=1):
    """Draws the given waypoint indices (default: all, connected in plan order by a blue line) on
    img (HWC uint8 BGR, modified in place AND returned). img is assumed to be a `scale`x upscale
    of a (height, width) frame that was itself preprocess_frame(...)'d with `crop` - i.e. the same
    convention project_world_to_final_pixel's docstring describes for o1/predicted_plan.png -
    pixel coords are computed at native (height, width) resolution then scaled up to match.
    Red = grasp-on waypoint, green = grasp-off."""
    if indices is None:
        indices = range(len(abs_positions))
    prev_px = None
    for idx in indices:
        proj = project_world_to_final_pixel(abs_positions[idx], crop, (height, width))
        if proj is None:
            continue
        row, col = proj
        px = (int(col * scale), int(row * scale))
        color = (0, 0, 255) if grasp_attrs[idx] > GRASP_THRESHOLD else (0, 255, 0)
        if 0 <= px[0] < img.shape[1] and 0 <= px[1] < img.shape[0]:
            cv2.circle(img, px, max(3, 2 * scale), color, -1)
            if prev_px is not None:
                cv2.line(img, prev_px, px, (255, 0, 0), max(1, scale // 2))
            prev_px = px
    return img


ROLLOUT_VIDEO_SCALE = 2   # upscale factor for the right (rollout) panel of rollout.mp4 - chosen so
                          # it exactly matches a 2x2 demo-frame grid's size.


def save_rollout_video(out_path, demo_video, plan_img, frame_log, waypoint_idx_log, abs_positions,
                        grasp_attrs, crop, height, width, scale=ROLLOUT_VIDEO_SCALE, fps=10):
    """Writes <out_path> (mp4): left panel is a static 2x2 grid of 4 of the T_context human-demo
    context frames (the conditioning `v` the plan was inferred from); right panel is the robot
    rollout. The very first video frame's right panel is `plan_img` (the full predicted plan - ALL
    waypoints, as also saved standalone to predicted_plan.png); every following frame is the
    corresponding live rollout frame (preprocessed identically to o1) annotated with ONLY the
    single waypoint being pursued at that simulation step (waypoint_idx_log[i]), since that's the
    one actually being "reached" then."""
    right_h, right_w = height * scale, width * scale

    demo_idx = np.linspace(0, demo_video.shape[0] - 1, 4, dtype=int)
    left_panel = debug_utils.build_frame_grid(demo_video[demo_idx], cols=2)
    left_panel = cv2.resize(left_panel, (right_w, right_h), interpolation=cv2.INTER_NEAREST)
    sep = np.full((right_h, 4, 3), 255, dtype=np.uint8)

    def compose(right_panel_upscaled):
        return np.concatenate([left_panel, sep, right_panel_upscaled], axis=1)

    def upscale(img_native_hwc_bgr):
        return cv2.resize(img_native_hwc_bgr, (right_w, right_h), interpolation=cv2.INTER_NEAREST)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    first = compose(upscale(plan_img))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                              (first.shape[1], first.shape[0]))
    writer.write(first)
    for step_obs, wp_idx in zip(frame_log, waypoint_idx_log):
        frame = preprocess_frame(step_obs['camera_front_image'], crop, height, width)
        frame = upscale(debug_utils.unnormalize(frame, normalized=True))
        draw_waypoints_on_frame(frame, abs_positions, grasp_attrs, crop, height, width,
                                 indices=[wp_idx], scale=scale)
        writer.write(compose(frame))
    writer.release()


# ----------------------------------------------------------------------------
# episode rollout
# ----------------------------------------------------------------------------
def run_episode(env, model, config, demo_file, height, width, crop, device, debug_dir=None,
                 replay_training_scene=False, training_agent_file=None, seed=None):
    demo_traj, _ = load_traj(demo_file)
    ds_cfg = config['dataset']
    demo_video = make_demo_context(demo_traj, ds_cfg.get('T_context', 10), height, width,
                                    tuple(ds_cfg.get('demo_crop', (0, 0, 0, 0))))

    # Same reset-retry/reseed pattern as get_expert_trajectory (expert_pick_place.py): reseed
    # np.random before each attempt (env._reset_internal's object-placement sampler draws from it),
    # bumping by `tries` so a RandomizationError doesn't just redraw the same failing layout.
    tries = 0
    while True:
        try:
            if seed is not None:
                np.random.seed(seed + tries)
            obs = env.reset()
            break
        except RandomizationError:
            tries += 1
    if replay_training_scene:
        assert training_agent_file is not None, 'replay_training_scene=True needs training_agent_file'
        scene_debug_path = os.path.join(debug_dir, 'training_scene_frame.png') if debug_dir else None
        set_objects_from_training_trajectory(env, training_agent_file, debug_path=scene_debug_path)
    obs = stabilize(env)
    start_pos, _ = current_pose(env)
    o1 = preprocess_frame(obs['camera_front_image'], crop, height, width)

    waypoints = predict_waypoints(model, demo_video, o1, device)
    if config.get('image_waypoints', False):
        # waypoints[:, :3] is (normalized image u, v, depth) - the policy's raw output space when
        # trained with image_waypoints=True (see camera_projection.py / compute_loss_trajectory) -
        # not a relative-to-start displacement. Project through the same calibrated camera
        # geometry training used, with no random augmentation (preprocess_frame/make_demo_context
        # only apply the deterministic crop+resize at eval time).
        projection = build_sample_projection(crop, NO_AUGMENTATION_STATS, (height, width))
        abs_positions = project_normalized_depth_to_world(waypoints[:, :3], projection)
    else:
        abs_positions = waypoints[:, :3] + start_pos[None]
    grasp_flags = waypoints[:, 3] > GRASP_THRESHOLD
    # Shift the grasp flag one waypoint step earlier: checkpoints trained before
    # mtlfd_dataset.py's grasp_frames fix learned grasp_point/traj_points against a label that was
    # one raw-trajectory step LATE (paired with the position after the gripper had already started
    # closing rather than the position it was commanded from - see that fix's commit message), so
    # their predicted grasp_flags are biased one step late in the same direction. There's no
    # waypoint to pull into the freed-up last slot, so it defaults to open (the trajectory should
    # end with the gripper released, e.g. after the drop-off).
    grasp_flags = np.concatenate([grasp_flags[1:], [False]])

    # NOT get_camera_intrinsic_matrix(env.sim, 'camera_front', height, width) for any of this
    # module's projections - o1/plan_img/rollout frames are all the CROPPED+resized display frame,
    # not a native camera capture, and that call ignores the crop entirely (wrong by ~1.3-1.8x
    # focal length, wrong principal point - see project_world_to_final_pixel's docstring).
    plan_img = draw_waypoints_on_frame(debug_utils.unnormalize(o1, normalized=True), abs_positions,
                                        waypoints[:, 3], crop, height, width)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        # human video conditioning (v) fed to the model alongside o1 - lets you check what
        # demonstration the predicted plan below was actually conditioned on.
        debug_utils.save_frame_grid(demo_video, os.path.join(debug_dir, 'demo_context.png'))
        cv2.imwrite(os.path.join(debug_dir, 'predicted_plan.png'), plan_img)

    frame_log = []
    waypoint_idx_log = []
    current_waypoint_idx = [0]   # 1-elem list so on_step's closure can read the live value

    def on_step(step_obs):
        if debug_dir is not None and len(frame_log) % 3 == 0:
            debug_utils.save_rollout_frame(
                os.path.join(debug_dir, f'frame_{len(frame_log):04d}.png'),
                cv2.cvtColor(step_obs['camera_front_image'], cv2.COLOR_RGB2BGR))
        frame_log.append(step_obs)
        waypoint_idx_log.append(current_waypoint_idx[0])

    holding = False
    grasp_call_idx = 0
    drop_call_idx = 0
    try:
        for i, (pos, grasp_flag) in enumerate(zip(abs_positions, grasp_flags)):
            current_waypoint_idx[0] = i
            if grasp_flag and not holding:
                grasp_debug_path = (os.path.join(debug_dir, f'grasp_localize_{grasp_call_idx:02d}.png')
                                     if debug_dir else None)
                grasp_primitive(env, pos, on_step=on_step, debug_path=grasp_debug_path)
                grasp_call_idx += 1
                holding = True
            elif not grasp_flag and holding:
                drop_debug_path = (os.path.join(debug_dir, f'drop_place_{drop_call_idx:02d}.png')
                                    if debug_dir else None)
                drop_primitive(env, pos, on_step=on_step, debug_path=drop_debug_path)
                drop_call_idx += 1
                holding = False
            else:
                free_space_primitive(env, pos, holding, on_step=on_step)
    except EpisodeDone:
        pass   # env hit horizon/termination mid-plan (common on early/undertrained checkpoints);
               # score whatever was achieved up to that point rather than crashing the episode.

    if debug_dir and frame_log:
        save_rollout_video(os.path.join(debug_dir, 'rollout.mp4'), demo_video, plan_img, frame_log,
                            waypoint_idx_log, abs_positions, waypoints[:, 3], crop, height, width)

    final_obs = frame_log[-1] if frame_log else obs
    target_obj = env.objects[env.object_id].name.lower()
    obj_key, delta_key = f'{target_obj}_pos', f'{target_obj}_to_robot0_eef_pos'
    start_z = obs[obj_key][2]
    target_box_id = int(obs.get('target-box-id', -1))
    reached = picked = False
    place_wrong = 0.0

    # Same 3 checks as above (reach/pick/correct-bin), but run against every OTHER object in the
    # scene instead of the target - answers "did the robot go for/lift the wrong object, or drop
    # some other object into the bin meant for the target" rather than just "did it mishandle the
    # target." Each non-target object gets its OWN reached/picked state (check_pick's `reached`
    # arg must be that SAME object's reach state, not shared across objects, or a reach on one
    # distractor would wrongly count as a "pick" on a different one it was never near).
    other_names = [obj.name.lower() for obj in env.objects if obj.name.lower() != target_obj]
    other_start_z = {name: obs.get(f'{name}_pos', np.zeros(3))[2] for name in other_names}
    other_reached = {name: False for name in other_names}
    other_picked = {name: False for name in other_names}
    other_placed_in_target_bin = {name: False for name in other_names}
    target_bin_name = BIN_NAMES[target_box_id] if 0 <= target_box_id < len(BIN_NAMES) else None

    for step_obs in frame_log:
        reached = check_reach(0.03, step_obs.get(delta_key, np.ones(2)), reached)
        picked = check_pick(0.05, step_obs.get(obj_key, np.zeros(3))[2], start_z, reached, picked)
        for i, bin_name in enumerate(BIN_NAMES):
            if i != step_obs.get('target-box-id', -1):
                place_wrong = max(place_wrong, float(check_bin(
                    0.03, step_obs.get(f'{bin_name}_pos', np.zeros(3)),
                    step_obs.get(obj_key, np.zeros(3)), bool(place_wrong))))

        for name in other_names:
            o_key, d_key = f'{name}_pos', f'{name}_to_robot0_eef_pos'
            other_reached[name] = check_reach(0.03, step_obs.get(d_key, np.ones(2)), other_reached[name])
            other_picked[name] = check_pick(0.05, step_obs.get(o_key, np.zeros(3))[2],
                                             other_start_z[name], other_reached[name], other_picked[name])
            if target_bin_name is not None:
                other_placed_in_target_bin[name] = check_bin(
                    0.03, step_obs.get(f'{target_bin_name}_pos', np.zeros(3)),
                    step_obs.get(o_key, np.zeros(3)), other_placed_in_target_bin[name])
    success = bool(env._check_success()) if hasattr(env, '_check_success') else False

    reached_wrong_object = any(other_reached.values())
    picked_wrong_object = any(other_picked.values())
    placed_wrong_object_in_correct_bin = any(other_placed_in_target_bin.values())

    return {
        'task_id': int(final_obs.get('target-box-id', -1)) + 4 * int(final_obs.get('target-object', -1)),
        'target_object': target_obj,
        'reached': bool(reached),
        'picked': bool(picked),
        'place_wrong': bool(place_wrong),
        'reached_wrong_object': bool(reached_wrong_object),
        'picked_wrong_object': bool(picked_wrong_object),
        'placed_wrong_object_in_correct_bin': bool(placed_wrong_object_in_correct_bin),
        'wrong_object_details': {
            name: {'reached': bool(other_reached[name]), 'picked': bool(other_picked[name]),
                   'placed_in_target_bin': bool(other_placed_in_target_bin[name])}
            for name in other_names
        },
        'success': success,
        'n_primitive_steps': len(frame_log),
        'demo_file': demo_file,
        'replay_training_scene': replay_training_scene,
        'training_agent_file': training_agent_file,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('--saved_step', type=int, required=True)
    parser.add_argument('--episodes', type=int, default=4)
    parser.add_argument('--task_ids', type=int, nargs='+', default=[12, 13, 14, 15])
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds_file', type=str, default=None,
                         help='read per-episode seeds (one int per line) from this file instead of '
                              'generating them from --seed - lets a specific seeds.txt (e.g. from a '
                              'prior run) be replayed exactly; must have >= len(task_ids)*episodes lines')
    parser.add_argument('--debugpy', action='store_true', help='wait for debugger attach on port 5678')
    parser.add_argument('--replay_training_scenes', action='store_true',
                         help='after each env.reset(), teleport objects to match a random real '
                              'training trajectory\'s layout (set_objects_from_training_trajectory) '
                              'instead of leaving the env\'s own random reset placement - isolates '
                              'whether live rendering itself hurts accuracy, independent of the '
                              'object layout being unseen/random (see test_training_pointing_'
                              'accuracy.py for the layout-free, sim-free version of this check)')
    parser.add_argument('--training_pairs', action='store_true',
                         help='sample demo_file/training_agent_file as a genuine (demo, agent) '
                              'couple from the checkpoint\'s own mode=\'train\' dataset pairs '
                              '(MTLFDAgentTeacherDataset.pairs, rebuilt from config[\'dataset\']) '
                              'instead of picking each independently at random - implies '
                              '--replay_training_scenes (the scene is teleported to match the '
                              'paired agent trajectory)')
    args = parser.parse_args()
    if args.training_pairs:
        args.replay_training_scenes = True

    if args.debugpy:
        print("Waiting for debugger attach on port 5678...")
        debugpy.listen(('0.0.0.0', 5678))
        debugpy.wait_for_client()
        print("Debugger attached.")

    seed_everything(args.seed)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    model, config = load_model(args.model_dir, args.saved_step, device)
    ds_cfg = config['dataset']
    height, width = ds_cfg.get('height', 100), ds_cfg.get('width', 180)
    crop = tuple(ds_cfg.get('crop', (0, 0, 0, 0)))

    demo_root = os.path.join(ds_cfg['root_dir'], ds_cfg.get('task_name', 'pick_place'),
                              f"{ds_cfg.get('demo_name', 'human_rgb')}_{ds_cfg.get('task_name', 'pick_place')}")
    agent_root = os.path.join(ds_cfg['root_dir'], ds_cfg.get('task_name', 'pick_place'),
                               f"{ds_cfg.get('agent_name', 'ur5e')}_{ds_cfg.get('task_name', 'pick_place')}")

    results_dir = args.results_dir or os.path.join(args.model_dir, f'results_pick_place/step-{args.saved_step}')
    os.makedirs(results_dir, exist_ok=True)

    controller_config = load_controller_config(custom_fpath=CONTROLLER_PATH)
    action_ranges = np.array([[-0.05, 0.25], [-0.45, 0.5], [0.82, 1.2], [-5, 5], [-5, 5], [-5, 5]])

    training_pairs_by_task = build_training_pairs_by_task(ds_cfg) if args.training_pairs else None

    # Same per-run seed-list generation as test_any_task.py's _proc (one random.getrandbits(32) draw
    # per episode, off a freshly reseeded stream) - written to seeds.txt for reproducibility, and
    # used both to reseed each task's env creation and each episode's env.reset(). --seeds_file
    # overrides this with a fixed list (e.g. a prior run's own seeds.txt), for exact replay.
    n_total_episodes = len(args.task_ids) * args.episodes
    if args.seeds_file:
        with open(args.seeds_file) as f:
            episode_seeds = [int(line) for line in f if line.strip()]
        assert len(episode_seeds) >= n_total_episodes, (
            f'--seeds_file {args.seeds_file!r} has {len(episode_seeds)} seeds, need at least '
            f'{n_total_episodes} (= {len(args.task_ids)} task_ids * {args.episodes} episodes)')
        episode_seeds = episode_seeds[:n_total_episodes]
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        episode_seeds = [random.getrandbits(32) for _ in range(n_total_episodes)]
    with open(os.path.join(results_dir, 'seeds.txt'), 'w') as f:
        for s in episode_seeds:
            f.write(f'{s}\n')

    all_results = []
    ep_counter = 0
    for task_id in args.task_ids:
        task_pairs = None
        if args.training_pairs:
            task_pairs = training_pairs_by_task.get(task_id)
            assert task_pairs, (f'no mode="train" (demo, agent) pairs found for task_id {task_id} '
                                 f'in the checkpoint\'s own dataset config - this task is likely '
                                 f'held out (test_tasks), so --training_pairs cannot be used for it')
        else:
            demo_files = sorted(glob.glob(os.path.join(demo_root, f'task_{task_id:02d}', '*.pkl')))
            assert demo_files, f'no demo files found for task_id {task_id} in {demo_root}'
            agent_files = None
            if args.replay_training_scenes:
                agent_files = sorted(glob.glob(os.path.join(agent_root, f'task_{task_id:02d}', '*.pkl')))
                assert agent_files, f'no agent trajectories found for task_id {task_id} in {agent_root}'
        # Seed env construction itself (object-placement sampler runs at __init__/reset time) with
        # this task's first episode seed, same as get_expert_trajectory's np.random.seed(env_seed)
        # right before its own creation retry loop.
        np.random.seed(episode_seeds[ep_counter])
        while True:
            try:
                env = get_env('UR5e_PickPlaceDistractor', controller_configs=controller_config,
                               task_id=task_id, has_renderer=False, has_offscreen_renderer=True,
                               reward_shaping=False, use_camera_obs=True, ranges=action_ranges,
                               render_gpu_device_id=args.gpu_id, render_camera='camera_front',
                               object_set=2)
                break
            except RandomizationError:
                continue
        for ep in range(args.episodes):
            ep_seed = episode_seeds[ep_counter]
            np.random.seed(ep_seed)
            if args.training_pairs:
                training_agent_file, demo_file = task_pairs[np.random.randint(len(task_pairs))]
            else:
                demo_file = demo_files[np.random.randint(len(demo_files))]
                training_agent_file = (agent_files[np.random.randint(len(agent_files))]
                                        if args.replay_training_scenes else None)
            debug_dir = os.path.join(results_dir, 'debug_images', f'task{task_id:02d}_ep{ep:02d}')
            result = run_episode(env, model, config, demo_file, height, width, crop, device,
                                  debug_dir=debug_dir,
                                  replay_training_scene=args.replay_training_scenes,
                                  training_agent_file=training_agent_file, seed=ep_seed)
            result['episode'] = ep_counter
            all_results.append(result)
            with open(os.path.join(debug_dir, 'result.json'), 'w') as f:
                json.dump(result, f, indent=2)
            print(f'[mtlfd-test] task {task_id} ep {ep}: {result}', flush=True)
            ep_counter += 1
        env.close()

    summary = {
        'n_episodes': len(all_results),
        'success_rate': float(np.mean([r['success'] for r in all_results])) if all_results else 0.0,
        'reached_rate': float(np.mean([r['reached'] for r in all_results])) if all_results else 0.0,
        'picked_rate': float(np.mean([r['picked'] for r in all_results])) if all_results else 0.0,
        'reached_wrong_object_rate': float(np.mean(
            [r['reached_wrong_object'] for r in all_results])) if all_results else 0.0,
        'picked_wrong_object_rate': float(np.mean(
            [r['picked_wrong_object'] for r in all_results])) if all_results else 0.0,
        'placed_wrong_object_in_correct_bin_rate': float(np.mean(
            [r['placed_wrong_object_in_correct_bin'] for r in all_results])) if all_results else 0.0,
    }
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump({'summary': summary, 'episodes': all_results}, f, indent=2)
    print('[mtlfd-test] SUMMARY:', summary, flush=True)


if __name__ == '__main__':
    main()
