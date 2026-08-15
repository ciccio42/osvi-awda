"""
Diagnostic: how much does the predicted waypoint plan change when ONLY the human-demo context
video (v) changes, holding the live env state (o1, object/bin positions) fixed? Answers "is a bad
predicted plan actually caused by the demo picked for that episode" (test_mtlfd_rollout.py picks
one uniformly at random per episode from every human_rgb_pick_place/task_XX/*.pkl demo).

Resets the env ONCE for a given task_id (one fixed o1 / fixed object+bin layout), then re-runs
predict_waypoints against several different demo files for that same task, holding o1 fixed. If
the predicted plan barely moves across demos, the demo isn't what's driving a bad plan you saw
(look at o1 / the model itself instead); if it swings a lot, the demo is a real factor.

Requires the same live robosuite/multi_task_robosuite_env sim as test_mtlfd_rollout.py - run via
mtlfd_adaptation/_analysis_scratch/run_demo_sensitivity.sh, not directly on a login node.

Usage:
    python -u mtlfd_adaptation/_analysis_scratch/test_demo_sensitivity.py <model_dir> \
        --saved_step 430000 --task_id 12 --n_demos 8 --out_dir <dir>
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    # APPEND, not insert(0, ...): REPO_ROOT contains its own vendored `robosuite/` (an old fork -
    # see test_mtlfd_rollout.py's own top-of-file comment), which would shadow the real
    # pip-installed `robosuite` package the moment that module's `from robosuite import
    # load_controller_config` line runs below, if REPO_ROOT came before site-packages on
    # sys.path. Appending keeps site-packages' resolution priority intact.
    sys.path.append(REPO_ROOT)

# Reuse test_mtlfd_rollout.py's own setup wholesale (model loading, env construction, context/
# waypoint prediction, preprocessing) instead of reimplementing it - keeps this diagnostic exactly
# consistent with what a real rollout episode does.
from mtlfd_adaptation.test_mtlfd_rollout import (  # noqa: E402
    CONTROLLER_PATH, GRASP_THRESHOLD, current_pose, get_env, load_controller_config, load_model,
    make_demo_context, predict_waypoints, preprocess_frame, stabilize,
)
from mtlfd_adaptation import debug_utils  # noqa: E402
from mtlfd_adaptation.camera_projection import (  # noqa: E402
    NO_AUGMENTATION_STATS, build_sample_projection, project_normalized_depth_to_world,
    project_world_to_final_pixel,
)
from mtlfd_adaptation.trajectory_bridge import load_traj  # noqa: E402

OVERLAY_SCALE = 3
# distinct BGR colors, one per demo (cycles if n_demos > len(this))
COLORS = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 200), (200, 0, 200),
          (0, 140, 255), (140, 0, 255), (0, 0, 0)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('--saved_step', type=int, required=True)
    parser.add_argument('--task_id', type=int, default=12)
    parser.add_argument('--n_demos', type=int, default=8)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0, help='selects WHICH demo files are sampled, not the env layout')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    model, config = load_model(args.model_dir, args.saved_step, device)
    ds_cfg = config['dataset']
    height, width = ds_cfg.get('height', 100), ds_cfg.get('width', 180)
    crop = tuple(ds_cfg.get('crop', (0, 0, 0, 0)))
    demo_crop = tuple(ds_cfg.get('demo_crop', (0, 0, 0, 0)))
    T_context = ds_cfg.get('T_context', 10)
    image_waypoints = config.get('image_waypoints', False)

    demo_root = os.path.join(ds_cfg['root_dir'], ds_cfg.get('task_name', 'pick_place'),
                              f"{ds_cfg.get('demo_name', 'human_rgb')}_{ds_cfg.get('task_name', 'pick_place')}")
    demo_files = sorted(glob.glob(os.path.join(demo_root, f'task_{args.task_id:02d}', '*.pkl')))
    assert demo_files, f'no demo files found for task_id {args.task_id} in {demo_root}'
    rng = np.random.RandomState(args.seed)
    if len(demo_files) <= args.n_demos:
        chosen = demo_files
    else:
        chosen = list(rng.choice(demo_files, size=args.n_demos, replace=False))

    out_dir = (args.out_dir or os.path.join(
        args.model_dir, f'demo_sensitivity/step-{args.saved_step}_task{args.task_id:02d}'))
    os.makedirs(out_dir, exist_ok=True)

    controller_config = load_controller_config(custom_fpath=CONTROLLER_PATH)
    action_ranges = np.array([[-0.05, 0.25], [-0.45, 0.5], [0.82, 1.2], [-5, 5], [-5, 5], [-5, 5]])

    # ONE env, ONE reset - o1 / object+bin layout stays fixed for every demo tried below, which is
    # the whole point (isolate the demo's effect from env-layout randomness).
    env = get_env('UR5e_PickPlaceDistractor', controller_configs=controller_config,
                   task_id=args.task_id, has_renderer=False, has_offscreen_renderer=True,
                   reward_shaping=False, use_camera_obs=True, ranges=action_ranges,
                   render_gpu_device_id=args.gpu_id, render_camera='camera_front', object_set=2)
    obs = env.reset()
    obs = stabilize(env)
    start_pos, _ = current_pose(env)
    o1 = preprocess_frame(obs['camera_front_image'], crop, height, width)
    # ground truth for "is the model actually pointing at the right object" - env.object_id / the
    # object name is authoritative (obj color<->index mapping is object_set-dependent, see
    # new_pp.py's object_to_id - don't infer this from eyeballing the rendered image).
    target_obj_name = env.objects[env.object_id].name.lower()
    target_box_id = int(obs.get('target-box-id', -1))
    print(f'[demo-sensitivity] task {args.task_id}: true target object = {target_obj_name!r}, '
          f'target bin = {target_box_id}', flush=True)
    env.close()   # done with the sim - everything below is pure model inference

    o1_img = debug_utils.unnormalize(o1, normalized=True)
    projection = build_sample_projection(crop, NO_AUGMENTATION_STATS, (height, width))

    overlay = cv2.resize(o1_img, (width * OVERLAY_SCALE, height * OVERLAY_SCALE),
                          interpolation=cv2.INTER_LINEAR)
    results = []
    for i, demo_file in enumerate(chosen):
        demo_traj, _ = load_traj(demo_file)
        demo_video = make_demo_context(demo_traj, T_context, height, width, demo_crop)
        waypoints = predict_waypoints(model, demo_video, o1, device)
        if image_waypoints:
            abs_positions = project_normalized_depth_to_world(waypoints[:, :3], projection)
        else:
            abs_positions = waypoints[:, :3] + start_pos[None]
        results.append({
            'demo_file': demo_file,
            'waypoints_xyz': abs_positions.tolist(),
            'grasp_attr': waypoints[:, 3].tolist(),
        })

        color = COLORS[i % len(COLORS)]
        prev_px = None
        for pos in abs_positions:
            proj = project_world_to_final_pixel(pos, crop, (height, width))
            if proj is None:
                continue
            row, col = proj
            px = (int(col * OVERLAY_SCALE), int(row * OVERLAY_SCALE))
            if 0 <= px[0] < overlay.shape[1] and 0 <= px[1] < overlay.shape[0]:
                cv2.circle(overlay, px, 4, color, -1)
                if prev_px is not None:
                    cv2.line(overlay, prev_px, px, color, 1)
                prev_px = px
        debug_utils.save_frame_grid(demo_video, os.path.join(out_dir, f'demo_{i:02d}_context.png'))

    legend_h = 14 * len(chosen) + 6
    canvas = cv2.copyMakeBorder(overlay, 0, legend_h, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    for i, demo_file in enumerate(chosen):
        color = COLORS[i % len(COLORS)]
        cv2.putText(canvas, f'{i}: {os.path.basename(demo_file)}',
                    (2, overlay.shape[0] + 12 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
                    cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, 'o1.png'), o1_img)
    cv2.imwrite(os.path.join(out_dir, 'all_plans_overlay.png'), canvas)

    all_xyz = np.array([r['waypoints_xyz'] for r in results])   # [n_demos, 5, 3]
    # per-waypoint spread across demos: std of each xyz component -> combined into one
    # "typical deviation" number per waypoint (euclidean norm of the per-axis std), plus the raw
    # min-max range per axis for context.
    per_waypoint_std_cm = np.linalg.norm(all_xyz.std(axis=0), axis=-1) * 100        # [5]
    per_waypoint_range_cm = (all_xyz.max(axis=0) - all_xyz.min(axis=0)) * 100       # [5,3]

    summary = {
        'model_dir': args.model_dir,
        'saved_step': args.saved_step,
        'task_id': args.task_id,
        'target_object': target_obj_name,
        'target_box_id': target_box_id,
        'n_demos': len(chosen),
        'demo_files': chosen,
        'per_waypoint_position_std_cm': per_waypoint_std_cm.tolist(),
        'per_waypoint_range_cm_xyz': per_waypoint_range_cm.tolist(),
        'mean_waypoint_position_std_cm': float(per_waypoint_std_cm.mean()),
        'max_waypoint_position_std_cm': float(per_waypoint_std_cm.max()),
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump({'summary': summary, 'per_demo': results}, f, indent=2)

    print(f'[demo-sensitivity] task {args.task_id}, {len(chosen)} demos, step {args.saved_step}')
    print(f'  per-waypoint position std across demos (cm): '
          f'{[round(v, 2) for v in per_waypoint_std_cm.tolist()]}')
    print(f"  mean={summary['mean_waypoint_position_std_cm']:.2f}cm  "
          f"max={summary['max_waypoint_position_std_cm']:.2f}cm")
    print(f'  wrote {out_dir}/summary.json, all_plans_overlay.png, o1.png, demo_XX_context.png')


if __name__ == '__main__':
    main()
