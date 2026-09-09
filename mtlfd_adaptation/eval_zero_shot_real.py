"""
Offline zero-shot (no finetuning) / finetuned-checkpoint evaluator for
real_eye_in_hand_ur5e_pick_place: scores a checkpoint's predicted waypoint plan against
pre-recorded real trajectories, no live robot/sim rollout involved.

No such offline harness existed before this: test_mtlfd_rollout.py's preprocess_frame /
make_demo_context / predict_waypoints / load_model are the right dataset-agnostic building blocks
to reuse, but that module also does module-level imports of `robosuite`,
`multi_task_robosuite_env`, `multi_task_test` (needed for its LIVE rollout machinery only) - to
keep this evaluator import-light and independent of the live-sim stack, the small, genuinely
dataset-agnostic functions are reproduced here verbatim instead of imported.

Real trajectories have no sim-only ground-truth fields (`target-object`, `obj_bb['bin']`) - the
picked object / drop bin are inferred from the trajectory itself (nearest obj_bb box to the real
eef position at the grasp frame / at the final frame), then scored as raw-canvas pixel distance
between the predicted final waypoint and that inferred target's `obj_bb` center. Grasp timing is
scored by comparing the predicted grasp waypoint index to the trajectory's own recorded gripper-
close frame (action[-1] > 0.01 - see mtlfd_dataset.py's grasp-mining comment for why this, not
raw obs, is the ground truth).

Usage:
    python -u mtlfd_adaptation/eval_zero_shot_real.py <model_dir> --saved_step 290000 \
        --task_ids 0 5 10 15 --results_dir <out_dir>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hem.models.inverse_module import InverseImitation  # noqa: E402
from hem.datasets.util import crop as crop_fn  # noqa: E402
from hem.datasets.util import randomize_video, resize  # noqa: E402
from mtlfd_adaptation import debug_utils  # noqa: E402
from mtlfd_adaptation.camera_projection_real import (  # noqa: E402
    CANVAS_SIZE, NO_AUGMENTATION_STATS, build_sample_projection,
    project_normalized_depth_to_world, project_world_to_pixel,
)
from mtlfd_adaptation.trajectory_bridge import load_traj  # noqa: E402

OBJECT_KEYS = ('greenbox', 'yellowbox', 'bluebox', 'redbox')
BIN_KEYS = ('bin_0', 'bin_1', 'bin_2', 'bin_3')


def preprocess_frame(img, crop, height, width):
    """Deterministic (no augmentation) crop+resize - mirrors
    test_mtlfd_rollout.py::preprocess_frame exactly."""
    img = crop_fn(img, crop)
    img = resize(img, (width, height))
    frames, _ = randomize_video(img[None], None, None, None, 0, np.array([0, 0]), True,
                                 rand_flip=False)
    return np.transpose(frames, (0, 3, 1, 2)).astype(np.float32)[0]


def make_demo_context(traj, T_context, height, width, crop, sample_sides=True):
    """Mirrors test_mtlfd_rollout.py::make_demo_context exactly."""
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
    """Mirrors test_mtlfd_rollout.py::load_model exactly."""
    with open(os.path.join(model_dir, 'config.yaml')) as f:
        config = yaml.safe_load(f)
    model = InverseImitation(**config['policy'])
    ckpt_path = os.path.join(model_dir, f'model_save-{saved_step}.pt')
    # weights_only=False: see the matching comment in test_mtlfd_rollout.py::load_model.
    loaded = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = loaded.state_dict() if hasattr(loaded, 'state_dict') else loaded
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model, config


def predict_waypoints(model, demo_video, o1_img, device):
    """Mirrors test_mtlfd_rollout.py::predict_waypoints exactly. Returns [5,4]."""
    context = torch.from_numpy(demo_video)[None].to(device)
    images = torch.from_numpy(np.stack([o1_img, o1_img]))[None].to(device)
    states = torch.zeros((1, 2, 1), device=device)
    ents = torch.zeros((1,), dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(states, images, context, ret_dist=False, ents=ents)
    waypoints = out['waypoints'][0].cpu().numpy()
    return waypoints[-5:]


def real_traj_points(traj, waypoints=50):
    """Ground-truth (pos, grasp) samples for a real trajectory - mirrors
    mtlfd_dataset.py::_make_agent_sample's real-data branch (eef_pos position, action[-1]>0.01
    grasp mining), for the save_waypoint_overlay 'gt_waypoints' comparison."""
    elements = [traj.get(t) for t in range(len(traj))]
    n = len(elements)
    grasp_frames = [elements[i + 1]['action'][-1] > 0.01 for i in range(n - 1)] + [False]
    out_inds = np.linspace(0, n - 1, num=waypoints, endpoint=True, dtype=int)
    poses = np.stack([elements[i]['obs']['eef_pos'] for i in out_inds]).astype(np.float32)
    grasps = np.stack([grasp_frames[i] for i in out_inds]).astype(np.int32)
    return np.concatenate((poses, grasps[:, None] * 0.2), axis=-1).astype(np.float32), grasp_frames


def infer_target_object_and_bin(traj, grasp_frames, full_projection):
    """No sim-only target labels exist on real data - infer the picked object as the obj_bb box
    nearest the real eef position (projected to raw canvas pixels) at the grasp frame, and the
    drop bin as the nearest bin at the trajectory's final frame."""
    n = len(traj)
    grasp_ind = next((i for i, g in enumerate(grasp_frames) if g), n - 1)
    obs_grasp = traj.get(grasp_ind)['obs']
    obs_final = traj.get(n - 1)['obs']

    def nearest(obs, keys):
        row, col = project_world_to_pixel(obs['eef_pos'][None], full_projection, CANVAS_SIZE)
        pt = np.array([row[0], col[0]])
        bb = obs['obj_bb']['camera_front']
        dists = {k: np.linalg.norm(pt - np.array(bb[k]['center'])[::-1]) for k in keys if k in bb}
        best = min(dists, key=dists.get)
        return best, bb[best]['center'], dists[best]

    obj_name, obj_center, _ = nearest(obs_grasp, OBJECT_KEYS)
    bin_name, bin_center, _ = nearest(obs_final, BIN_KEYS)
    return obj_name, obj_center, bin_name, bin_center


def evaluate_trajectory(model, agent_file, demo_file, config, device, results_dir, tag):
    ds_cfg = config['dataset']
    height, width, crop = ds_cfg['height'], ds_cfg['width'], ds_cfg['crop']
    demo_crop = ds_cfg.get('demo_crop', crop)
    T_context = ds_cfg['T_context']

    agent_traj, _ = load_traj(agent_file, is_real=True)
    demo_traj, _ = load_traj(demo_file)

    from mtlfd_adaptation.mtlfd_dataset import AGENT_SETTLE_STEPS
    o1_img = preprocess_frame(agent_traj.get(AGENT_SETTLE_STEPS)['obs']['image'], crop,
                               height, width)
    demo_video = make_demo_context(demo_traj, T_context, height, width, demo_crop)

    pred = predict_waypoints(model, demo_video, o1_img, device)  # [5,4]: (u,v,depth,grasp_attr)
    gt_traj_points, grasp_frames = real_traj_points(agent_traj)

    eval_projection = build_sample_projection(crop, NO_AUGMENTATION_STATS, (height, width))
    full_projection = build_sample_projection([0, 0, 0, 0], NO_AUGMENTATION_STATS, CANVAS_SIZE)

    obj_name, obj_center, bin_name, bin_center = infer_target_object_and_bin(
        agent_traj, grasp_frames, full_projection)

    pred_world = project_normalized_depth_to_world(pred[:, :3], eval_projection)
    pred_row, pred_col = project_world_to_pixel(pred_world, full_projection, CANVAS_SIZE)

    pred_grasp_idx = int(np.argmax(pred[:, 3]))
    gt_grasp_idx = next((i for i, g in enumerate(grasp_frames) if g), None)
    gt_grasp_frac = (gt_grasp_idx / max(len(grasp_frames) - 1, 1)) if gt_grasp_idx is not None \
        else None

    pick_err = float(np.linalg.norm(
        np.array([pred_row[pred_grasp_idx], pred_col[pred_grasp_idx]])
        - np.array(obj_center)[::-1]))
    drop_err = float(np.linalg.norm(
        np.array([pred_row[-1], pred_col[-1]]) - np.array(bin_center)[::-1]))

    out_path = os.path.join(results_dir, 'overlays', f'{tag}.png')
    debug_utils.save_waypoint_overlay(out_path, o1_img, pred, gt_traj_points, eval_projection,
                                       image_waypoints=True)

    return {
        'agent_file': agent_file, 'demo_file': demo_file,
        'inferred_target_object': obj_name, 'inferred_target_bin': bin_name,
        'pick_pixel_error': pick_err, 'drop_pixel_error': drop_err,
        'pred_grasp_waypoint_idx': pred_grasp_idx,
        'gt_grasp_frac_of_traj': gt_grasp_frac,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model_dir', type=str, help='checkpoint dir containing config.yaml + '
                                                 'model_save-<step>.pt')
    p.add_argument('--saved_step', type=int, required=True)
    p.add_argument('--task_ids', type=int, nargs='+', default=[0, 5, 10, 15])
    p.add_argument('--agent_name', type=str, default='real_eye_in_hand_ur5e')
    p.add_argument('--demo_name', type=str, default='human_rgb')
    p.add_argument('--episodes_per_task', type=int, default=None,
                    help='cap the number of agent trajectories evaluated per task (default: all)')
    p.add_argument('--results_dir', type=str, required=True)
    p.add_argument('--device', type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    model, config = load_model(args.model_dir, args.saved_step, device)

    root_dir = os.path.expandvars(config['dataset']['root_dir'])
    all_results = []
    for task_id in args.task_ids:
        sub = f'task_{task_id:02d}'
        agent_files = sorted(glob.glob(os.path.join(
            root_dir, 'pick_place', f'{args.agent_name}_pick_place', sub, '*.pkl')))
        demo_files = sorted(glob.glob(os.path.join(
            root_dir, 'pick_place', f'{args.demo_name}_pick_place', sub, '*.pkl')))
        assert agent_files, f'no real agent trajectories found for {sub}'
        assert demo_files, f'no demo trajectories found for {sub}'
        demo_file = demo_files[0]  # one fixed demo per task, same convention as val pairs files
        if args.episodes_per_task:
            agent_files = agent_files[:args.episodes_per_task]

        for agent_file in agent_files:
            tag = f'{sub}_{os.path.splitext(os.path.basename(agent_file))[0]}'
            try:
                result = evaluate_trajectory(model, agent_file, demo_file, config, device,
                                              args.results_dir, tag)
            except Exception as e:
                print(f'[eval_zero_shot_real] {tag} failed: {e}', flush=True)
                continue
            result['task_id'] = task_id
            all_results.append(result)
            print(f'[eval_zero_shot_real] {tag}: pick_err={result["pick_pixel_error"]:.1f}px '
                  f'drop_err={result["drop_pixel_error"]:.1f}px '
                  f'target={result["inferred_target_object"]}', flush=True)

    summary_path = os.path.join(args.results_dir, 'results.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    if all_results:
        pick_errs = [r['pick_pixel_error'] for r in all_results]
        drop_errs = [r['drop_pixel_error'] for r in all_results]
        print(f'\n[eval_zero_shot_real] n={len(all_results)} '
              f'mean_pick_err={np.mean(pick_errs):.1f}px mean_drop_err={np.mean(drop_errs):.1f}px')
    print(f'[eval_zero_shot_real] wrote {summary_path} and overlays/ under {args.results_dir}')


if __name__ == '__main__':
    main()
