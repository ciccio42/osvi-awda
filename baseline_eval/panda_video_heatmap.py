"""
Generates per-episode annotated videos + initial frames, per-task heatmaps, and per-task XYZ
trajectory plots for osvi-awda's own natively-collected Panda pick-place trajectories
(dataset/panda/traj*.pkl, produced by scripts/collect_demonstrations.py - see
/home/rsofnc000/.claude/plans/sequential-wibbling-penguin.md section 2).

Adapted from open_x_embodiment/datasets/ur5e_pick_place_delta_all/test.py's approach (per-episode
annotated frames -> video, heat_map() density plot, per-axis trajectory subplots), with real
(non-cosmetic) changes for this repo's different data schema - see plan section 3:
  - Single camera only (`obs['image']`), no wrist/gripper camera.
  - No per-timestep language-instruction string; task label is synthesized from the task index
    (task_id = traj_index // PER_TASK_GROUP -> object_id = task_id // 4, bin_id = task_id % 4).
  - eef_pos is in ABSOLUTE robot-base-frame meters (not centered near the origin like the
    reference dataset), so the heatmap uses a min-max mapping from the real Panda workspace
    bounds (hem/robosuite/custom_ik_wrapper.py's `ranges`) instead of an origin-centered one, and
    matplotlib's own `extent=` handles real-world (non-centered) axis labels directly rather than
    porting the reference's manual centered-pixel-tick math.
  - No hard-coded spawn-region-box overlay - the pickled obs doesn't carry literal bin XY
    placement (only the target bin *index*, via `target-box-id`), so drawing that box would be a
    guess; omitted rather than guessed.

Usage:
    python -m baseline_eval.panda_video_heatmap --dataset_dir dataset/panda \
        --out_dir baseline_eval/outputs/panda --num_workers 32
"""
import argparse
import glob
import os
import pickle
import re

import cv2
import imageio
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image, ImageDraw, ImageFont

import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from hem.datasets.savers.trajectory import Trajectory  # noqa: F401 - registers class for unpickling

ITEM_NAMES = ["Milk", "Bread", "Cereal", "Can"]  # PandaPickPlaceDistractor's object order
# Panda end-effector workspace bounds (position dims only), from the normalize_action() ranges in
# hem/robosuite/custom_ik_wrapper.py - the only place this repo records the real workspace extent.
X_RANGE = (0.44, 0.74)
Y_RANGE = (-0.33, 0.5)
MARGIN_M = 0.05


def task_label(task_id, per_task_group):
    object_id, bin_id = task_id // 4, task_id % 4
    item = ITEM_NAMES[object_id] if object_id < len(ITEM_NAMES) else f"obj{object_id}"
    return f"{item}_bin{bin_id}"


def natural_key(path):
    m = re.search(r"traj(\d+)\.pkl$", path)
    return int(m.group(1)) if m else -1


def annotate_frame(image_rgb, label, action, eef_pos):
    img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    w, h = img.size

    # font.getsize() (not draw.textbbox(), which needs a TrueType font in newer Pillow, or
    # draw.textsize()/font.getbbox(), unavailable in this env's Pillow 9.0.1) works for the
    # default bitmap font here.
    text_w, text_h = font.getsize(label)
    pos = ((w - text_w) // 2, h - text_h - 5)
    draw.rectangle([pos, (pos[0] + text_w, pos[1] + text_h)], fill=(0, 0, 0))
    draw.text(pos, label, fill=(255, 255, 255), font=font)

    action_str = "RESET" if action is None else np.array2string(np.asarray(action).round(2), precision=2)
    gripper_str = "n/a" if action is None else f"{float(action[-1]):.2f}"
    draw.text((5, 5), f"action: {action_str}", fill="yellow")
    draw.text((5, 18), f"eef_pos: {np.array2string(np.asarray(eef_pos).round(3), precision=3)}", fill="yellow")
    draw.text((5, 31), f"gripper_cmd: {gripper_str}", fill="yellow")
    return np.array(img)


def process_episode(args):
    pkl_path, out_dir, per_task_group = args
    n = natural_key(pkl_path)
    task_id = n // per_task_group
    label = task_label(task_id, per_task_group)

    with open(pkl_path, "rb") as f:
        loaded = pickle.load(f)
    traj = loaded["traj"]

    frames, eef_traj = [], []
    for t in range(len(traj)):
        step = traj[t]
        obs = step["obs"]
        image = obs["image"]  # lazily JPEG-decoded on first access, 240x320x3 uint8 RGB
        eef_pos = obs["eef_pos"]
        eef_traj.append(np.asarray(eef_pos[:3], dtype=np.float32))
        frames.append(annotate_frame(image, label, step["action"], eef_pos))

    video_dir = os.path.join(out_dir, "videos", label)
    os.makedirs(video_dir, exist_ok=True)
    imageio.mimwrite(os.path.join(video_dir, f"episode_{n:04d}.mp4"), frames, fps=10, codec="libx264")

    init_dir = os.path.join(out_dir, "initial_frames", label)
    os.makedirs(init_dir, exist_ok=True)
    Image.fromarray(traj[0]["obs"]["image"]).save(
        os.path.join(init_dir, f"episode_{n:04d}_initial.png"))

    return label, n, eef_traj


def heat_map(episodes_xy, out_path, title, x_range=X_RANGE, y_range=Y_RANGE, margin=MARGIN_M):
    """episodes_xy: dict of episode_idx -> list of [x,y,z] (only x,y used), all in absolute
    robot-base-frame meters. Min-max-maps into a canvas from the known workspace bounds (+margin)
    instead of the reference script's origin-centered convention, since eef_pos here is never
    near (0,0) - see module docstring."""
    px_resolution_cm = 0.5
    x_lo, x_hi = (x_range[0] - margin) * 100, (x_range[1] + margin) * 100  # meters -> cm
    y_lo, y_hi = (y_range[0] - margin) * 100, (y_range[1] + margin) * 100
    h_px = max(int((x_hi - x_lo) / px_resolution_cm), 1)
    w_px = max(int((y_hi - y_lo) / px_resolution_cm), 1)
    canvas = np.zeros((h_px, w_px))

    for traj in episodes_xy.values():
        traj = np.asarray(traj)[:, :2] * 100  # meters -> cm
        rows = ((traj[:, 0] - x_lo) / px_resolution_cm).astype(np.int32)
        cols = ((traj[:, 1] - y_lo) / px_resolution_cm).astype(np.int32)
        valid = (rows >= 0) & (rows < h_px) & (cols >= 0) & (cols < w_px)
        for r, c in zip(rows[valid], cols[valid]):
            canvas[r, c] += 1

    fig, ax = plt.subplots(figsize=(8, 10))
    plt.title(title.replace("_", " ").title())
    plt.xlabel("Y (cm, robot base frame)")
    plt.ylabel("X (cm, robot base frame)")
    norm = mcolors.LogNorm(vmin=1, vmax=max(canvas.max(), 1))
    im = ax.imshow(canvas, cmap="plasma", origin="upper", norm=norm,
                    extent=[y_lo, y_hi, x_hi, x_lo])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Trajectory density (log scale)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def trajectory_plot(episodes_xyz, out_path, title):
    axis_labels = ["x", "y", "z"]
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"Trajectory for task: {title}", fontsize=16)
    for dim in range(3):
        ax = axs[dim]
        for traj in episodes_xyz.values():
            traj_np = np.stack(traj)
            ax.plot(traj_np[:, dim], alpha=0.6)
        ax.set_ylabel(f"{axis_labels[dim]} position (m)")
        ax.grid(True)
    axs[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=os.path.join(REPO_ROOT, "dataset", "panda"))
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "baseline_eval", "outputs", "panda"))
    parser.add_argument("--per_task_group", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N files (for a pilot run)")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dataset_dir, "traj*.pkl")), key=natural_key)
    if args.limit:
        files = files[: args.limit]
    print(f"processing {len(files)} episodes from {args.dataset_dir}")

    import multiprocessing
    work = [(f, args.out_dir, args.per_task_group) for f in files]
    with multiprocessing.Pool(args.num_workers) as pool:
        results = pool.map(process_episode, work)

    by_label = {}
    for label, n, eef_traj in results:
        by_label.setdefault(label, {})[n] = eef_traj

    heatmap_dir = os.path.join(args.out_dir, "heatmaps")
    plot_dir = os.path.join(args.out_dir, "trajectory_plots")
    all_episodes = {}
    for label, episodes in by_label.items():
        heat_map(episodes, os.path.join(heatmap_dir, f"{label}_heatmap.png"), label)
        trajectory_plot(episodes, os.path.join(plot_dir, f"{label}_trajectories_combined.png"), label)
        all_episodes.update({f"{label}_{n}": traj for n, traj in episodes.items()})
    heat_map(all_episodes, os.path.join(heatmap_dir, "all_panda_heatmap.png"),
              "training_trajectory_distribution")

    print("done.")


if __name__ == "__main__":
    main()
