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
import sys

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

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
from mtlfd_adaptation.trajectory_bridge import load_traj  # noqa: E402
from mtlfd_adaptation.robosuite_camera_utils import (  # noqa: E402
    get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map, pixel_to_world,
    project_world_to_pixel,
)

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


# ----------------------------------------------------------------------------
# depth-based object localization (paper Appendix VII-A)
# ----------------------------------------------------------------------------
def localize_grasp_target(env, camera_name='robot0_eye_in_hand', height=200, width=360,
                           floor_margin=0.01, max_depth=1.0):
    """Paper Appendix VII-A object-localization procedure: mask out background by depth, estimate
    the floor plane as the median remaining depth, mask pixels >1cm above the floor, take
    connected components, pick the one closest to image center, backproject its centroid."""
    get_obs = getattr(env, '_get_observations', None) or getattr(env, '_get_observation')
    obs = get_obs()
    depth_norm = obs[f'{camera_name}_depth']
    real_depth = get_real_depth_map(env.sim, depth_norm)
    if real_depth.ndim == 3:
        real_depth = real_depth[:, :, 0]

    valid = real_depth < max_depth
    if not np.any(valid):
        return None
    floor_depth = np.median(real_depth[valid])

    # height above floor plane, approximated as (floor_depth - real_depth) since the eye-in-hand
    # camera looks roughly straight down at the table during the grasp approach.
    above_floor = (floor_depth - real_depth) > floor_margin
    mask = (valid & above_floor).astype(np.uint8)
    if mask.sum() == 0:
        return None

    n_labels, labels = cv2.connectedComponents(mask)
    if n_labels <= 1:
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
        return None

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


def grasp_primitive(env, hint_pos, on_step=None):
    hover = hint_pos.copy()
    hover[2] += 0.15
    move_to(env, hover, gripper=-1.0, on_step=on_step)
    target = localize_grasp_target(env)
    if target is None:
        target = hint_pos
    approach = target.copy()
    approach[2] += 0.15
    move_to(env, approach, gripper=-1.0, on_step=on_step)
    descend = target.copy()
    descend[2] += 0.03
    move_to(env, descend, gripper=-1.0, on_step=on_step)
    move_to(env, descend, gripper=1.0, max_iters=10, on_step=on_step)   # close
    lift = descend.copy()
    lift[2] += 0.15
    move_to(env, lift, gripper=1.0, on_step=on_step)


def drop_primitive(env, on_step=None):
    pos, quat = current_pose(env)
    obs, reward, done, info = env.step(np.concatenate([pos, T.quat2axisangle(quat), [-1.0]]))
    if on_step is not None:
        on_step(obs)
    if done:
        raise EpisodeDone()


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


# ----------------------------------------------------------------------------
# episode rollout
# ----------------------------------------------------------------------------
def run_episode(env, model, config, demo_file, height, width, crop, device, debug_dir=None):
    demo_traj, _ = load_traj(demo_file)
    ds_cfg = config['dataset']
    demo_video = make_demo_context(demo_traj, ds_cfg.get('T_context', 10), height, width,
                                    tuple(ds_cfg.get('demo_crop', (0, 0, 0, 0))))

    obs = env.reset()
    start_pos, _ = current_pose(env)
    o1 = preprocess_frame(obs['camera_front_image'], crop, height, width)

    waypoints = predict_waypoints(model, demo_video, o1, device)
    abs_positions = waypoints[:, :3] + start_pos[None]
    grasp_flags = waypoints[:, 3] > GRASP_THRESHOLD

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        # human video conditioning (v) fed to the model alongside o1 - lets you check what
        # demonstration the predicted plan below was actually conditioned on.
        debug_utils.save_frame_grid(demo_video, os.path.join(debug_dir, 'demo_context.png'))

        img = debug_utils.unnormalize(o1, normalized=True)
        K = get_camera_intrinsic_matrix(env.sim, 'camera_front', height, width)
        Rt = get_camera_extrinsic_matrix(env.sim, 'camera_front')
        prev_px = None
        for pos, grasp_attr in zip(abs_positions, waypoints[:, 3]):
            proj = project_world_to_pixel(pos, K, Rt)
            if proj is None:
                continue
            row, col = proj
            color = (0, 0, 255) if grasp_attr > GRASP_THRESHOLD else (0, 255, 0)
            if 0 <= row < img.shape[0] and 0 <= col < img.shape[1]:
                cv2.circle(img, (int(col), int(row)), 3, color, -1)
                if prev_px is not None:
                    cv2.line(img, prev_px, (int(col), int(row)), (255, 0, 0), 1)
                prev_px = (int(col), int(row))
        cv2.imwrite(os.path.join(debug_dir, 'predicted_plan.png'), img)

    frame_log = []

    def on_step(step_obs):
        if debug_dir is not None and len(frame_log) % 3 == 0:
            debug_utils.save_rollout_frame(
                os.path.join(debug_dir, f'frame_{len(frame_log):04d}.png'),
                cv2.cvtColor(step_obs['camera_front_image'], cv2.COLOR_RGB2BGR))
        frame_log.append(step_obs)

    holding = False
    try:
        for pos, grasp_flag in zip(abs_positions, grasp_flags):
            if grasp_flag and not holding:
                grasp_primitive(env, pos, on_step=on_step)
                holding = True
            elif not grasp_flag and holding:
                drop_primitive(env, on_step=on_step)
                holding = False
            else:
                free_space_primitive(env, pos, holding, on_step=on_step)
    except EpisodeDone:
        pass   # env hit horizon/termination mid-plan (common on early/undertrained checkpoints);
               # score whatever was achieved up to that point rather than crashing the episode.

    final_obs = frame_log[-1] if frame_log else obs
    target_obj = env.objects[env.object_id].name.lower()
    obj_key, delta_key = f'{target_obj}_pos', f'{target_obj}_to_robot0_eef_pos'
    start_z = obs[obj_key][2]
    reached = picked = False
    place_wrong = 0.0
    for step_obs in frame_log:
        reached = check_reach(0.03, step_obs.get(delta_key, np.ones(2)), reached)
        picked = check_pick(0.05, step_obs.get(obj_key, np.zeros(3))[2], start_z, reached, picked)
        for i, bin_name in enumerate(BIN_NAMES):
            if i != step_obs.get('target-box-id', -1):
                place_wrong = max(place_wrong, float(check_bin(
                    0.03, step_obs.get(f'{bin_name}_pos', np.zeros(3)),
                    step_obs.get(obj_key, np.zeros(3)), bool(place_wrong))))
    success = bool(env._check_success()) if hasattr(env, '_check_success') else False

    return {
        'task_id': int(final_obs.get('target-box-id', -1)) + 4 * int(final_obs.get('target-object', -1)),
        'target_object': target_obj,
        'reached': bool(reached),
        'picked': bool(picked),
        'place_wrong': bool(place_wrong),
        'success': success,
        'n_primitive_steps': len(frame_log),
        'demo_file': demo_file,
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
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    model, config = load_model(args.model_dir, args.saved_step, device)
    ds_cfg = config['dataset']
    height, width = ds_cfg.get('height', 100), ds_cfg.get('width', 180)
    crop = tuple(ds_cfg.get('crop', (0, 0, 0, 0)))

    demo_root = os.path.join(ds_cfg['root_dir'], ds_cfg.get('task_name', 'pick_place'),
                              f"{ds_cfg.get('demo_name', 'human_rgb')}_{ds_cfg.get('task_name', 'pick_place')}")

    results_dir = args.results_dir or os.path.join(args.model_dir, f'results_pick_place/step-{args.saved_step}')
    os.makedirs(results_dir, exist_ok=True)

    controller_config = load_controller_config(custom_fpath=CONTROLLER_PATH)
    action_ranges = np.array([[-0.05, 0.25], [-0.45, 0.5], [0.82, 1.2], [-5, 5], [-5, 5], [-5, 5]])

    all_results = []
    ep_counter = 0
    for task_id in args.task_ids:
        demo_files = sorted(glob.glob(os.path.join(demo_root, f'task_{task_id:02d}', '*.pkl')))
        assert demo_files, f'no demo files found for task_id {task_id} in {demo_root}'
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
            demo_file = demo_files[np.random.randint(len(demo_files))]
            debug_dir = os.path.join(results_dir, 'debug_images', f'task{task_id:02d}_ep{ep:02d}')
            result = run_episode(env, model, config, demo_file, height, width, crop, device,
                                  debug_dir=debug_dir)
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
    }
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump({'summary': summary, 'episodes': all_results}, f, indent=2)
    print('[mtlfd-test] SUMMARY:', summary, flush=True)


if __name__ == '__main__':
    main()
