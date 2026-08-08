"""
Isolates WHERE the trajectory-plan-estimation problem is: does the network already mispredict
waypoint positions on data drawn from its OWN training distribution (no live sim, no rollout
primitives, no stabilize()/hover-offset/depth-relocalization involved) - or is it accurate there,
which would point the problem at test-time execution/generalization instead (live env reset
randomness, camera-augmentation-state mismatch, closed-loop grasp primitive, etc)?

Method: pull real (context, traj) pairs straight from MTLFDAgentTeacherDataset in mode='train'
(the checkpoint's own train_tasks, augmentation disabled for a clean read), run the model forward
exactly like test_mtlfd_rollout.py's predict_waypoints(), decode the 5 predicted waypoints to
world coordinates via the SAME per-sample projection_matrix the SDTW loss trained against, and
compare each predicted point to the ground-truth traj_points (also world-frame, 50 dense points
covering the same episode) via nearest-neighbor distance - a soft-DTW-friendly metric (elastic
alignment means a "correct" prediction doesn't have to hit a specific time-index, only needs to
land ON the true path somewhere).

Usage:
  python mtlfd_adaptation/_analysis_scratch/diagnose_waypoint_error.py <model_dir> <saved_step> \
      [--n-samples 20]
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "mtlfd_adaptation"))

from hem.models.inverse_module import InverseImitation  # noqa: E402
import yaml  # noqa: E402

from mtlfd_dataset import MTLFDAgentTeacherDataset  # noqa: E402
from camera_projection import project_normalized_depth_to_world  # noqa: E402
from debug_utils import save_waypoint_overlay  # noqa: E402


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
    parser.add_argument('--split', choices=['train', 'test'], default='train',
                         help="'test' uses the checkpoint's own config['dataset']['test_tasks'] "
                              "(held-out generalization set); 'train' matches the original "
                              "training-distribution read this script started as.")
    parser.add_argument('--plot-dir', type=str, default=None,
                         help='if set, dump a predicted(red)-vs-GT(green) waypoint overlay PNG '
                              'per sample (on the o1 observation frame given to the model) into '
                              'this directory, via the same debug_utils.save_waypoint_overlay '
                              'used by train_mtlfd.py.')
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
    # disable randomness so predicted-vs-GT comparisons aren't muddied by augmentation noise; the
    # fixed `crop` (a deterministic config value, not a random augmentation) is kept as-is.
    ds_cfg.update(rand_flip=False, rand_crop=None, rand_translate=None, color_jitter=None)
    dataset = MTLFDAgentTeacherDataset(mode=args.split, **ds_cfg)
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
            # pred: raw model output pre-projection, traj['traj_points']: world xyz + grasp -
            # exactly the convention save_waypoint_overlay expects (same as train_mtlfd.py's own
            # debug dump inside make_debug_forward).
            save_waypoint_overlay(
                os.path.join(args.plot_dir, f'sample_{idx:04d}.png'),
                o1, pred, traj['traj_points'].astype(np.float32), projection,
                image_waypoints=config.get('image_waypoints', False))

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

    print('\nINTERPRETATION:')
    print(' - small numbers here (~1-3cm, well under the 3cm reach threshold) despite 0% rollout')
    print('   success would point the problem at TEST-TIME EXECUTION/GENERALIZATION (live env')
    print('   reset randomness the model has not seen, stabilize()/hover-offset behavior, or a')
    print('   train/eval augmentation-state mismatch) rather than the learned mapping itself.')
    print(' - large numbers here (>5-10cm) on data drawn from the TRAINING distribution would mean')
    print('   the network has not actually learned to localize precisely despite low aggregate')
    print('   SDTW loss - i.e. soft-DTW\'s elastic alignment can hide real positional error behind')
    print('   a deceptively low loss number (it rewards plausible SHAPE/timing more than exact')
    print('   anchor position). That would point at the training objective/data, not the harness.')


if __name__ == '__main__':
    main()
