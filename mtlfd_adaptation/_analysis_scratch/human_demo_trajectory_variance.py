"""
Estimate how spatially consistent the REAL human demo videos (human_rgb_pick_place) are, per
target color, to test whether the model's per-color regression-variance asymmetry (see
training_pointing_accuracy analysis: greenbox std=2.5px vs yellow/blue std=5-7px) traces back to
the human demonstrations themselves being more/less consistent for that color.

human_rgb_pick_place pkls carry ONLY RGB frames (no obj_bb/raw_state/action - real camera video,
not sim), so there is no ground-truth object position to read. Instead: the target color's cube is
localized per-frame via calibrated RGB-channel-dominance thresholding + largest-connected-component
(calibrated and visually verified against a sample frame - see conversation), giving an approximate
pixel-space trajectory of the target object across each demo. Cross-demo variance of that
trajectory (at 10 normalized-time checkpoints) is then compared across the 4 colors.

Usage: python -u mtlfd_adaptation/_analysis_scratch/human_demo_trajectory_variance.py
"""
import glob
import os
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from mtlfd_adaptation.trajectory_bridge import load_traj  # noqa: E402

DEMO_ROOT = '/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset/pick_place/human_rgb_pick_place'
# object_to_id order (new_pp.py): greenbox=0, yellowbox=1, bluebox=2, redbox=3
COLOR_BY_OBJECT_ID = ['greenbox', 'yellowbox', 'bluebox', 'redbox']
N_CHECKPOINTS = 10
MIN_AREA, MAX_AREA = 30, 2000


def color_mask(img_f32, color):
    R, G, B = img_f32[..., 0], img_f32[..., 1], img_f32[..., 2]
    if color == 'redbox':
        return (R > 110) & (R - G > 50) & (R - B > 50)
    if color == 'yellowbox':
        return (R > 110) & (G > 110) & (B < 50)
    if color == 'greenbox':
        return (G > 90) & (G - R > 40) & (G - B > 40)
    if color == 'bluebox':
        return (B > 90) & (B - R > 30) & (B - G > 30)
    raise ValueError(color)


def largest_blob_centroid(mask):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    candidates = [(stats[i, cv2.CC_STAT_AREA], centroids[i]) for i in range(1, n)
                  if MIN_AREA <= stats[i, cv2.CC_STAT_AREA] <= MAX_AREA]
    if not candidates:
        return None
    _, (cx, cy) = max(candidates, key=lambda x: x[0])
    return np.array([cy, cx])


def track_trajectory(pkl_path, color):
    traj, _ = load_traj(pkl_path)
    T = len(traj)
    centroids = []
    for t in range(T):
        img = traj.get(t)['obs']['camera_front_image'].astype(np.float32)
        centroids.append(largest_blob_centroid(color_mask(img, color)))
    # keep only detected frames, resample to N_CHECKPOINTS evenly spaced points via linear interp
    valid_t = [t for t, c in enumerate(centroids) if c is not None]
    if len(valid_t) < max(3, T // 4):
        return None  # too much occlusion/failed detection to trust this demo
    valid_pts = np.array([centroids[t] for t in valid_t])
    sample_t = np.linspace(valid_t[0], valid_t[-1], N_CHECKPOINTS)
    path = np.stack([
        np.interp(sample_t, valid_t, valid_pts[:, 0]),
        np.interp(sample_t, valid_t, valid_pts[:, 1]),
    ], axis=1)  # (N_CHECKPOINTS, 2)
    return path


def main():
    for object_id, color in enumerate(COLOR_BY_OBJECT_ID):
        task_ids = [object_id * 4 + b for b in range(4)]
        pkl_files = []
        for tid in task_ids:
            pkl_files.extend(sorted(glob.glob(os.path.join(DEMO_ROOT, f'task_{tid:02d}', 'traj*.pkl'))))

        paths = []
        n_failed = 0
        for pkl_path in pkl_files:
            path = track_trajectory(pkl_path, color)
            if path is None:
                n_failed += 1
            else:
                paths.append(path)

        if not paths:
            print(f'{color:>10}: no usable trajectories tracked')
            continue
        paths = np.stack(paths, axis=0)  # (n_demos, N_CHECKPOINTS, 2)

        # cross-demo spatial std at each checkpoint (row, col), averaged over both axes and checkpoints
        std_per_checkpoint = paths.std(axis=0)  # (N_CHECKPOINTS, 2)
        mean_std = std_per_checkpoint.mean()
        start_std = std_per_checkpoint[0].mean()
        end_std = std_per_checkpoint[-1].mean()

        print(f'{color:>10}: n_tracked={len(paths)}/{len(pkl_files)} (failed={n_failed})  '
              f'mean_path_std={mean_std:.1f}px  start_std={start_std:.1f}px  end_std={end_std:.1f}px')


if __name__ == '__main__':
    main()
