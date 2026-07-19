"""Debug-image dumping helpers used by train_mtlfd.py and test_mtlfd_rollout.py.

Images coming out of MTLFDAgentTeacherDataset / the model pipeline are float32 CHW tensors
normalized with hem.datasets.util.MEAN/STD (or raw 0..255 CHW). unnormalize() handles both.
"""
import os

import cv2
import numpy as np

from hem.datasets.util import MEAN, STD


def unnormalize(img_chw, normalized=True):
    """CHW float array (numpy or torch) -> HWC uint8 BGR (for cv2.imwrite)."""
    if hasattr(img_chw, 'detach'):
        img_chw = img_chw.detach().cpu().numpy()
    img = np.transpose(img_chw, (1, 2, 0))
    if normalized:
        img = (img * STD.reshape(1, 1, 3) + MEAN.reshape(1, 1, 3)) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def save_frame_grid(frames_chw, out_path, normalized=True, cols=None):
    """frames_chw: [T, C, H, W] -> single row/grid PNG."""
    imgs = [unnormalize(f, normalized) for f in frames_chw]
    cols = cols or len(imgs)
    rows = [imgs[i:i + cols] for i in range(0, len(imgs), cols)]
    rows = [np.concatenate(r, axis=1) for r in rows]
    max_w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, max_w - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
    grid = np.concatenate(rows, axis=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, grid)


def save_batch_debug_images(out_dir, context, traj, index=0, tag='sample'):
    """Dumps one batch item's demo context frames + agent frames as PNG grids."""
    os.makedirs(out_dir, exist_ok=True)
    save_frame_grid(context['video'][index], os.path.join(out_dir, f'{tag}_demo_context.png'))
    save_frame_grid(traj['images'][index], os.path.join(out_dir, f'{tag}_agent_frames.png'))


def project_points(points_xyz, projection_matrix, img_w, img_h):
    """Rough 2D projection of Nx3 world points for visualization only (uses the repo's bundled
    generic camera calibration when a real one isn't available - see docs/04)."""
    n = points_xyz.shape[0]
    hom = np.concatenate([points_xyz, np.ones((n, 1), dtype=np.float32)], axis=-1)
    proj = hom @ projection_matrix.T
    proj = proj[:, :3] / np.clip(proj[:, 2:3], 1e-6, None)
    px = ((proj[:, 0] * 0.5 + 0.5) * img_w).astype(int)
    py = ((proj[:, 1] * 0.5 + 0.5) * img_h).astype(int)
    return np.stack([px, py], axis=-1)


def save_waypoint_overlay(out_path, o1_img_chw, pred_waypoints, gt_waypoints, projection_matrix,
                           normalized=True):
    """pred_waypoints / gt_waypoints: [W, 4] (xyz + grasp attribute in [0, 0.2]).
    Draws predicted waypoints in red, ground truth in green, on the o1 frame."""
    img = unnormalize(o1_img_chw, normalized)
    h, w = img.shape[:2]
    for pts, color in ((gt_waypoints, (0, 255, 0)), (pred_waypoints, (0, 0, 255))):
        xy = project_points(pts[:, :3], projection_matrix, w, h)
        for i, (x, y) in enumerate(xy):
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (int(x), int(y)), 3, color, -1)
            if i > 0:
                x0, y0 = xy[i - 1]
                cv2.line(img, (int(x0), int(y0)), (int(x), int(y)), color, 1)
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
