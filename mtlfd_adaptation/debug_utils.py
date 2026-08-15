"""Debug-image dumping helpers used by train_mtlfd.py and test_mtlfd_rollout.py.

Images coming out of MTLFDAgentTeacherDataset / the model pipeline are float32 CHW tensors
normalized with hem.datasets.util.MEAN/STD (or raw 0..255 CHW). unnormalize() handles both.
"""
import os

import cv2
import numpy as np

from hem.datasets.util import MEAN, STD
from mtlfd_adaptation.camera_projection import project_world_to_pixel


def unnormalize(img_chw, normalized=True):
    """CHW float array (numpy or torch) -> HWC uint8 BGR (for cv2.imwrite)."""
    if hasattr(img_chw, 'detach'):
        img_chw = img_chw.detach().cpu().numpy()
    img = np.transpose(img_chw, (1, 2, 0))
    if normalized:
        img = (img * STD.reshape(1, 1, 3) + MEAN.reshape(1, 1, 3)) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def build_frame_grid(frames_chw, normalized=True, cols=None):
    """frames_chw: [T, C, H, W] -> single row/grid HWC uint8 BGR array (no file I/O - see
    save_frame_grid for the disk-writing wrapper this was factored out of, used e.g. to build a
    demo-context panel for test_mtlfd_rollout.py's rollout video)."""
    imgs = [unnormalize(f, normalized) for f in frames_chw]
    cols = cols or len(imgs)
    rows = [imgs[i:i + cols] for i in range(0, len(imgs), cols)]
    rows = [np.concatenate(r, axis=1) for r in rows]
    max_w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, max_w - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
    return np.concatenate(rows, axis=0)


def save_frame_grid(frames_chw, out_path, normalized=True, cols=None):
    """frames_chw: [T, C, H, W] -> single row/grid PNG."""
    grid = build_frame_grid(frames_chw, normalized, cols)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, grid)


def save_batch_debug_images(out_dir, context, traj, index=0, tag='sample'):
    """Dumps one batch item's demo context frames + agent frames as PNG grids."""
    os.makedirs(out_dir, exist_ok=True)
    save_frame_grid(context['video'][index], os.path.join(out_dir, f'{tag}_demo_context.png'))
    save_frame_grid(traj['images'][index], os.path.join(out_dir, f'{tag}_agent_frames.png'))


def _draw_path(img, rows, cols, color):
    prev = None
    h, w = img.shape[:2]
    for row, col in zip(rows, cols):
        if not (np.isfinite(row) and np.isfinite(col)):
            prev = None   # break the connecting line across a behind-camera point
            continue
        pt = (int(col), int(row))
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(img, pt, 3, color, -1)
        if prev is not None:
            cv2.line(img, prev, pt, color, 1)
        prev = pt


def save_waypoint_overlay(out_path, o1_img_chw, pred_waypoints, gt_waypoints, projection_matrix,
                           normalized=True, image_waypoints=True):
    """pred_waypoints / gt_waypoints: [W, 4] (position dims + grasp attribute in [0, 0.2]).
    Draws predicted waypoints in red, ground truth in green, on the o1 frame.

    gt_waypoints[:, :3] is always absolute WORLD xyz - ground-truth traj_points never go through
    the image_waypoints transform (only predictions do - see compute_loss_trajectory), so it's
    always projected world->pixel via projection_matrix.

    pred_waypoints[:, :3] depends on image_waypoints:
      True  -> normalized (u, v, depth) in the SAME normalized-image convention projection_matrix
               was built in - already in image space, so plotted directly via the (u,v)->pixel
               formula, no matrix multiply needed (using projection_matrix here would be wrong:
               that's what the earlier, buggy version of this function did, and what
               project_normalized_depth_to_world exists to correctly undo).
      False -> a displacement relative to the trajectory's start position. There's no separate
               "start_pos" available here, so it's approximated as gt_waypoints[0, :3] (matches
               mtlfd_dataset.py's own convention that traj_points[0] IS the start position), then
               projected world->pixel like GT.
    """
    img = unnormalize(o1_img_chw, normalized)
    h, w = img.shape[:2]

    gt_row, gt_col = project_world_to_pixel(gt_waypoints[:, :3], projection_matrix, (h, w))

    if image_waypoints:
        pred_row = h * (1 - pred_waypoints[:, 1]) / 2
        pred_col = (pred_waypoints[:, 0] + 1) * w / 2
    else:
        start_pos = gt_waypoints[0, :3]
        pred_world = pred_waypoints[:, :3] + start_pos[None]
        pred_row, pred_col = project_world_to_pixel(pred_world, projection_matrix, (h, w))

    _draw_path(img, gt_row, gt_col, (0, 255, 0))
    _draw_path(img, pred_row, pred_col, (0, 0, 255))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)


def save_rollout_frame(out_path, frame_hwc_uint8_bgr, annotations=None):
    img = frame_hwc_uint8_bgr.copy()
    if annotations:
        for i, text in enumerate(annotations):
            cv2.putText(img, text, (5, 15 + 15 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 255), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)
