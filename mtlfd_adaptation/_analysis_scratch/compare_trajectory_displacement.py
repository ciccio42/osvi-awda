"""
Compares raw-trajectory step count / per-step end-effector displacement between the paper's own
baseline dataset (dataset/panda, the agent role fed into traj_points) and our UR5e adaptation's
own agent dataset (ur5e_pick_place), to test the hypothesis that our trajectories simply have many
more raw steps than baseline's, which could matter for training difficulty (either directly, or
indirectly via mtlfd_dataset.py::_make_agent_sample's fixed 50-point downsampling of traj_points
distorting/under-sampling bursty motion more severely on longer raw trajectories).

Both roles expose a comparable field: obs['ee_aa'][:3] (position part of the axis-angle EE pose) -
this matches exactly what mtlfd_adaptation/_analysis_scratch/plot_ur5e_waypoints.py already uses
to build traj_points-equivalent ground truth, so the "downsampled" numbers here are apples-to-
apples with what compute_loss_trajectory actually trains against.

Usage: python mtlfd_adaptation/_analysis_scratch/compare_trajectory_displacement.py
"""
import glob
import os
import pickle
import random
import sys

import numpy as np

REPO_ROOT = "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "mtlfd_adaptation"))
from trajectory_bridge import load_traj  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "mtlfd_adaptation", "_analysis_scratch", "displacement_compare_out")
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLE = 200
SEED = 0
N_INTERP = 50  # matches mtlfd_dataset.py::_make_agent_sample's out_inds = linspace(..., num=50, ...)


def panda_positions(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    traj = d['traj']
    n = len(traj)
    poses = np.stack([traj.get(i)['obs']['ee_aa'][:3] for i in range(n)])
    return poses


def ur5e_positions(path):
    traj, _ = load_traj(path)
    n = len(traj)
    poses = np.stack([traj.get(i)['obs']['ee_aa'][:3] for i in range(n)])
    return poses


def downsample(poses, num=N_INTERP):
    n = len(poses)
    out_inds = np.linspace(0, n - 1, num=num, endpoint=True, dtype=int)
    return poses[out_inds]


def step_stats(poses):
    """Returns (n_raw, path_length, per-step displacement array)."""
    diffs = np.linalg.norm(np.diff(poses, axis=0), axis=1)
    return len(poses), float(diffs.sum()), diffs


def analyze(name, paths, load_fn):
    n_raws, path_lens, all_raw_steps = [], [], []
    ds_path_lens, all_ds_steps = [], []
    burstiness = []  # fraction of raw path length carried by the single largest raw step
    for p in paths:
        try:
            poses = load_fn(p)
        except Exception as e:
            print(f'  [skip] {p}: {e}')
            continue
        if len(poses) < 3:
            continue
        n_raw, path_len, raw_steps = step_stats(poses)
        n_raws.append(n_raw)
        path_lens.append(path_len)
        all_raw_steps.append(raw_steps)
        burstiness.append(raw_steps.max() / path_len if path_len > 0 else 0.0)

        ds_poses = downsample(poses)
        _, ds_path_len, ds_steps = step_stats(ds_poses)
        ds_path_lens.append(ds_path_len)
        all_ds_steps.append(ds_steps)

    n_raws = np.array(n_raws)
    path_lens = np.array(path_lens)
    ds_path_lens = np.array(ds_path_lens)
    burstiness = np.array(burstiness)
    raw_steps_cat = np.concatenate(all_raw_steps)
    ds_steps_cat = np.concatenate(all_ds_steps)

    print(f'\n=== {name} (n_trajectories={len(n_raws)}) ===')
    print(f'raw step count (T):        mean={n_raws.mean():.1f}  median={np.median(n_raws):.0f}  '
          f'std={n_raws.std():.1f}  min={n_raws.min()}  max={n_raws.max()}')
    print(f'total path length (m):     mean={path_lens.mean():.4f}  median={np.median(path_lens):.4f}  '
          f'std={path_lens.std():.4f}')
    print(f'raw per-step disp (m):     mean={raw_steps_cat.mean():.5f}  median={np.median(raw_steps_cat):.5f}  '
          f'std={raw_steps_cat.std():.5f}  p95={np.percentile(raw_steps_cat, 95):.5f}')
    print(f'downsampled-to-{N_INTERP} per-step disp (m) [= actual traj_points training target]:')
    print(f'                            mean={ds_steps_cat.mean():.5f}  median={np.median(ds_steps_cat):.5f}  '
          f'std={ds_steps_cat.std():.5f}  p95={np.percentile(ds_steps_cat, 95):.5f}')
    print(f'path length lost by downsampling: mean={(1 - ds_path_lens / path_lens).mean() * 100:.1f}% '
          f'(shortcuts corners the raw path took between kept indices)')
    print(f'burstiness (largest single raw step / total path length): mean={burstiness.mean():.3f}  '
          f'median={np.median(burstiness):.3f}  max={burstiness.max():.3f}')

    return {
        'name': name, 'n_raws': n_raws, 'path_lens': path_lens, 'ds_path_lens': ds_path_lens,
        'raw_steps': raw_steps_cat, 'ds_steps': ds_steps_cat, 'burstiness': burstiness,
    }


def main():
    random.seed(SEED)

    panda_paths = sorted(glob.glob(os.path.join(REPO_ROOT, 'dataset', 'panda', '*.pkl')))
    ur5e_paths = sorted(glob.glob(
        '/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset/pick_place/ur5e_pick_place/task_*/*.pkl'))
    print(f'found {len(panda_paths)} panda (baseline) trajectories, '
          f'{len(ur5e_paths)} ur5e (ours) trajectories')

    panda_sample = random.sample(panda_paths, min(N_SAMPLE, len(panda_paths)))
    ur5e_sample = random.sample(ur5e_paths, min(N_SAMPLE, len(ur5e_paths)))

    baseline = analyze('BASELINE (panda, agent role)', panda_sample, panda_positions)
    ours = analyze('OURS (ur5e_pick_place, agent role)', ur5e_sample, ur5e_positions)

    print('\n=== HEAD-TO-HEAD ===')
    print(f'raw step count ratio (ours/baseline):        {ours["n_raws"].mean() / baseline["n_raws"].mean():.2f}x')
    print(f'total path length ratio (ours/baseline):     {ours["path_lens"].mean() / baseline["path_lens"].mean():.2f}x')
    print(f'raw per-step displacement ratio (ours/base): {ours["raw_steps"].mean() / baseline["raw_steps"].mean():.2f}x '
          '(< 1 means our steps are finer-grained/smaller motion per raw step)')
    print(f'downsampled per-step displacement ratio:     '
          f'{ours["ds_steps"].mean() / baseline["ds_steps"].mean():.2f}x '
          '(this is what compute_loss_trajectory actually sees as ground-truth step size)')
    print(f'burstiness ratio (ours/baseline):            {ours["burstiness"].mean() / baseline["burstiness"].mean():.2f}x '
          '(> 1 means our motion is more concentrated in a few raw steps, i.e. more info lost by fixed-50 downsampling)')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))

        axes[0, 0].hist(baseline['n_raws'], bins=30, alpha=0.6, label='baseline (panda)')
        axes[0, 0].hist(ours['n_raws'], bins=30, alpha=0.6, label='ours (ur5e)')
        axes[0, 0].set_title('raw trajectory length (steps)')
        axes[0, 0].legend()

        axes[0, 1].hist(baseline['raw_steps'], bins=60, alpha=0.6, label='baseline', range=(0, 0.05), density=True)
        axes[0, 1].hist(ours['raw_steps'], bins=60, alpha=0.6, label='ours', range=(0, 0.05), density=True)
        axes[0, 1].set_title('raw per-step displacement (m)')
        axes[0, 1].legend()

        axes[1, 0].hist(baseline['ds_steps'], bins=60, alpha=0.6, label='baseline', range=(0, 0.08), density=True)
        axes[1, 0].hist(ours['ds_steps'], bins=60, alpha=0.6, label='ours', range=(0, 0.08), density=True)
        axes[1, 0].set_title(f'downsampled-to-{N_INTERP} per-step displacement (m)\n[= actual traj_points training target]')
        axes[1, 0].legend()

        axes[1, 1].hist(baseline['burstiness'], bins=30, alpha=0.6, label='baseline', range=(0, 1))
        axes[1, 1].hist(ours['burstiness'], bins=30, alpha=0.6, label='ours', range=(0, 1))
        axes[1, 1].set_title('burstiness (largest raw step / total path length)')
        axes[1, 1].legend()

        fig.tight_layout()
        out_png = os.path.join(OUT_DIR, 'displacement_comparison.png')
        fig.savefig(out_png, dpi=130)
        print(f'\nwrote {out_png}')
    except ImportError:
        print('\n(matplotlib not available - skipped plot)')

    print('\nDONE')


if __name__ == '__main__':
    main()
