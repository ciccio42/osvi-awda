"""
One-off utility: render every human_rgb_pick_place demo trajectory pkl to an h264 mp4, saved under
a parallel video/task_<ID>/ tree (e.g. task_00/traj005.pkl -> video/task_00/video_005.mp4) rather
than alongside the pkl, so the dataset's own task_XX dirs stay untouched.

Encodes via a direct ffmpeg subprocess (rawvideo frames piped to libx264, yuv420p) rather than
cv2.VideoWriter, since cv2's bundled fourccs don't reliably produce real h264 - system ffmpeg here
is confirmed built with --enable-libx264.

Usage: python -u mtlfd_adaptation/_analysis_scratch/dump_human_demo_videos.py [root_dir] [--fps N]
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from mtlfd_adaptation.trajectory_bridge import load_traj  # noqa: E402

DEFAULT_ROOT = '/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset/pick_place/human_rgb_pick_place'


def write_h264_mp4(frames, out_path, fps=10):
    h, w = frames[0].shape[:2]
    cmd = ['ffmpeg', '-y', '-loglevel', 'error',
           '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
           '-r', str(fps), '-i', '-',
           '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f'ffmpeg exited {ret} while writing {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', nargs='?', default=DEFAULT_ROOT)
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()

    task_dirs = sorted(glob.glob(os.path.join(args.root_dir, 'task_*')))
    assert task_dirs, f'no task_* dirs found in {args.root_dir}'

    n_done, n_skipped, n_failed = 0, 0, 0
    for task_dir in task_dirs:
        out_dir = os.path.join(args.root_dir, 'video', os.path.basename(task_dir))
        os.makedirs(out_dir, exist_ok=True)
        pkl_files = sorted(glob.glob(os.path.join(task_dir, 'traj*.pkl')))
        for pkl_path in pkl_files:
            traj_cnt = os.path.basename(pkl_path)[len('traj'):-len('.pkl')]
            out_path = os.path.join(out_dir, f'video_{traj_cnt}.mp4')
            try:
                traj, _ = load_traj(pkl_path)
                frames = [traj.get(t)['obs']['image'] for t in range(len(traj))]
                write_h264_mp4(frames, out_path, fps=args.fps)
                n_done += 1
                print(f'[dump-video] wrote {out_path} ({len(frames)} frames)', flush=True)
            except Exception as e:  # noqa: BLE001 - keep going across the whole dataset
                n_failed += 1
                print(f'[dump-video] FAILED {pkl_path}: {e}', flush=True)

    print(f'[dump-video] done: {n_done} written, {n_failed} failed, root={args.root_dir}')


if __name__ == '__main__':
    main()
