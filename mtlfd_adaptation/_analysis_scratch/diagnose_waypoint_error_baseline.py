"""
Same diagnostic as diagnose_waypoint_error.py, but for the ORIGINAL osvi-awda baseline
checkpoints (panda/sawyer, dataset/type: 'agent teacher' via hem.datasets.agent_teacher_dataset
.AgentTeacherDataset) rather than the Multi-Task-LFD-Framework ur5e adaptation
(MTLFDAgentTeacherDataset). e.g. checkpoints under
/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/osvi_awda_baseline/*/bc_inv_ckpt-*.

The two dataset classes were verified to share the exact same __getitem__ contract InverseImitation
/compute_loss_trajectory expects - (context, traj) where context={'video','projection_matrix',
'fname'} and traj={'images','states','traj_points','projection_matrix','head_label',...} built the
same way (AgentDemonstrations._get_pairs / TeacherDemonstrations._make_context are what
MTLFDAgentTeacherDataset's own _make_agent_sample/_make_context were adapted from), including the
same projection_matrix convention ([u*z,v*z,z,1] -> world[x,y,z,1]) and the same traj_points grasp
encoding (grasp * 0.2). So the cm-distance METRICS below (which stay entirely in world space via
mtlfd_adaptation.camera_projection.project_normalized_depth_to_world) are valid unmodified - only
the dataset construction (agent_dir/teacher_dir instead of root_dir/task_name/agent_name/demo_name,
and dropping the 'type' config key that AgentDemonstrations/TeacherDemonstrations don't accept)
differs.

--plot-dir, however, needed a DIFFERENT camera model than debug_utils.save_waypoint_overlay (which
assumes intrinsics are baked into projection_matrix, true for the mtlfd/ur5e case but NOT for
baseline - see mtlfd_adaptation/baseline_camera_projection.py's module docstring for the full
story: baseline's projection_matrix is a pure extrinsic with no FOV/intrinsics, reconstructed after
the original calibration file was lost, and the real per-pixel ground truth (obs['eef_point'])
comes from a totally separate intrinsics pipeline this module reverse-engineers and validates to
0px error against real stored labels).

Usage:
  python mtlfd_adaptation/_analysis_scratch/diagnose_waypoint_error_baseline.py <model_dir> \
      <saved_step> [--split test] [--n-samples 40] [--plot-dir DIR]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

REPO_ROOT = "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "mtlfd_adaptation"))

from hem.models.inverse_module import InverseImitation  # noqa: E402
import yaml  # noqa: E402

from hem.datasets.agent_teacher_dataset import AgentTeacherDataset  # noqa: E402
from camera_projection import project_normalized_depth_to_world  # noqa: E402
from debug_utils import unnormalize, _draw_path  # noqa: E402
from baseline_camera_projection import (  # noqa: E402
    project_world_to_pixel_baseline, adjust_pixel_for_dataset_crop, BASELINE_CANVAS_SIZE)


def save_waypoint_overlay_baseline(out_path, o1_img_chw, pred_uvz, gt_traj_points, projection,
                                    crop, normalized=True):
    """Baseline analog of debug_utils.save_waypoint_overlay: both predicted and GT waypoints are
    routed through WORLD coordinates first (pred via project_normalized_depth_to_world using the
    sample's own extrinsic-only projection_matrix - the same transform the SDTW loss trains
    against, so this is exact, not an approximation), then world->pixel via the validated
    baseline_camera_projection camera model, then adjusted for this dataset's own deterministic
    crop+resize. Avoids ever having to interpret predicted (u, v) as pixel coordinates directly,
    which - unlike the mtlfd/ur5e case - isn't well-defined for baseline (see module docstring)."""
    img = unnormalize(o1_img_chw, normalized)
    h, w = img.shape[:2]

    gt_world = gt_traj_points[:, :3]
    pred_world = project_normalized_depth_to_world(pred_uvz[:, :3], projection)

    gt_row_raw, gt_col_raw = project_world_to_pixel_baseline(gt_world)
    pred_row_raw, pred_col_raw = project_world_to_pixel_baseline(pred_world)

    gt_row, gt_col = adjust_pixel_for_dataset_crop(gt_row_raw, gt_col_raw, crop, BASELINE_CANVAS_SIZE, (h, w))
    pred_row, pred_col = adjust_pixel_for_dataset_crop(pred_row_raw, pred_col_raw, crop, BASELINE_CANVAS_SIZE, (h, w))

    _draw_path(img, gt_row, gt_col, (0, 255, 0))
    _draw_path(img, pred_row, pred_col, (0, 0, 255))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)


def load_model(model_dir, saved_step, device):
    with open(os.path.join(model_dir, 'config.yaml')) as f:
        config = yaml.safe_load(f)
    model = InverseImitation(**config['policy'])
    ckpt_path = os.path.join(model_dir, f'model_save-{saved_step}.pt')
    loaded = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = loaded.state_dict() if hasattr(loaded, 'state_dict') else loaded
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model, config


def predict_waypoints(model, demo_video, o1_img, device):
    context = torch.from_numpy(demo_video)[None].to(device)
    images = torch.from_numpy(np.stack([o1_img, o1_img]))[None].to(device)
    states = torch.zeros((1, 2, 1), device=device)
    ents = torch.zeros((1,), dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(states, images, context, ret_dist=False, ents=ents)
    waypoints = out['waypoints'][0].cpu().numpy()   # [15,4]
    return waypoints[-5:]


def nearest_dists(points, path):
    """points: [k,3], path: [n,3] -> [k] min euclidean distance from each point to the path."""
    d = np.linalg.norm(points[:, None, :] - path[None, :, :], axis=-1)  # [k,n]
    return d.min(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('saved_step', type=int)
    parser.add_argument('--n-samples', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split', choices=['train', 'test'], default='test',
                         help="'test' uses the checkpoint's own config['dataset']['test_tasks'] "
                              "(held-out generalization set); 'train' uses train_tasks.")
    parser.add_argument('--plot-dir', type=str, default=None,
                         help='if set, dump a predicted(red)-vs-GT(green) waypoint overlay PNG '
                              'per sample (on the o1 observation frame given to the model) into '
                              'this directory.')
    args = parser.parse_args()
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    model, config = load_model(args.model_dir, args.saved_step, device)
    if not config.get('image_waypoints', False):
        print("WARNING: config['image_waypoints'] is False - this checkpoint doesn't use the "
              "image-plane+depth waypoint formulation this script assumes.")

    ds_cfg = dict(config['dataset'])
    ds_cfg.pop('type', None)   # not a kwarg AgentDemonstrations/TeacherDemonstrations accept
    # disable randomness so predicted-vs-GT comparisons aren't muddied by augmentation noise; the
    # fixed `crop` (a deterministic config value, not a random augmentation) is kept as-is.
    ds_cfg.update(rand_flip=False, rand_crop=None, rand_translate=None, color_jitter=None)
    dataset = AgentTeacherDataset(mode=args.split, **ds_cfg)
    task_pool = dataset.train_tasks if args.split == 'train' else dataset.test_tasks
    print(f'split={args.split}  dataset: {len(dataset)} (context, traj) pairs across tasks={task_pool}')

    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(dataset), size=min(args.n_samples, len(dataset)), replace=False)

    per_sample_mean = []
    per_sample_max = []
    per_sample_last = []   # distance from the FINAL predicted waypoint to the GT path's end point
    grasp_agreement = []

    for idx in indices:
        context, traj = dataset[idx]
        demo_video = context['video'].astype(np.float32)
        o1 = traj['images'][0]
        projection = traj['projection_matrix']
        gt_world = traj['traj_points'][:, :3].astype(np.float32)     # [50,3], absolute world coords
        gt_grasp = traj['traj_points'][:, 3] > 0.1                    # matches grasps*0.2 encoding

        pred = predict_waypoints(model, demo_video, o1, device)       # [5,4]: u,v,depth,grasp
        pred_world = project_normalized_depth_to_world(pred[:, :3], projection)  # [5,3]

        if args.plot_dir:
            save_waypoint_overlay_baseline(
                os.path.join(args.plot_dir, f'sample_{idx:04d}.png'),
                o1, pred, traj['traj_points'].astype(np.float32), projection,
                crop=ds_cfg['crop'])

        dists = nearest_dists(pred_world, gt_world)
        per_sample_mean.append(dists.mean())
        per_sample_max.append(dists.max())
        per_sample_last.append(float(np.linalg.norm(pred_world[-1] - gt_world[-1])))

        GRASP_THRESHOLD = 0.1   # matches test_mtlfd_rollout.py's own constant (0..0.2 scale)
        pred_grasp = pred[:, 3] > GRASP_THRESHOLD
        grasp_agreement.append(float(pred_grasp.any() == gt_grasp.any()))

        print(f'sample {idx:4d} (task pool={task_pool}): '
              f'nearest-dist mean={dists.mean()*100:5.2f}cm max={dists.max()*100:5.2f}cm  '
              f'final-waypoint-vs-path-end={per_sample_last[-1]*100:5.2f}cm  '
              f'pred_grasp_any={bool(pred_grasp.any())} gt_grasp_any={bool(gt_grasp.any())}  '
              f'raw_pred_grasp_channel={np.array2string(pred[:, 3], precision=3)}')

    per_sample_mean = np.array(per_sample_mean)
    per_sample_max = np.array(per_sample_max)
    per_sample_last = np.array(per_sample_last)

    print(f"\n=== AGGREGATE over {len(indices)} {args.split}-split samples (tasks={task_pool}) ===")
    print(f'nearest-neighbor dist to GT path (cm): mean={per_sample_mean.mean()*100:.2f} '
          f'median={np.median(per_sample_mean)*100:.2f} p90={np.percentile(per_sample_mean, 90)*100:.2f}')
    print(f'worst single waypoint per sample (cm):  mean={per_sample_max.mean()*100:.2f} '
          f'median={np.median(per_sample_max)*100:.2f} p90={np.percentile(per_sample_max, 90)*100:.2f}')
    print(f'final predicted waypoint vs path end (cm): mean={per_sample_last.mean()*100:.2f} '
          f'median={np.median(per_sample_last)*100:.2f} '
          f'(this is the closest analog to "does the plan end near the target/drop location")')
    print(f'grasp-intent agreement (predicted any-grasp == GT any-grasp): '
          f'{np.mean(grasp_agreement)*100:.0f}%')
    if args.plot_dir:
        print(f'\nwrote {len(indices)} waypoint overlay PNGs (pred=red, GT=green) to {args.plot_dir}')


if __name__ == '__main__':
    main()
