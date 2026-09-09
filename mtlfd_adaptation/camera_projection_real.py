"""
Real-camera counterpart to `camera_projection.py`, for the `real_eye_in_hand_ur5e_pick_place`
dataset's `camera_front_image` stream (ZED front camera, `zed_front` topic - see
`mtlfd_adaptation/ur5e_zed_front_projection.yaml`'s header comments for the full derivation:
camera intrinsics -> `image_norm_to_camera`, combined with the `camera -> ArUco -> table_0 ->
base_link` static-transform chain -> `T_base_camera`, giving `projection_matrix =
T_base_camera @ image_norm_to_camera`).

Same API and the exact same `[u*z, v*z, z, 1] -> world [x, y, z, 1]` convention as
`camera_projection.py` (see that module's docstring), so `build_sample_projection`,
`project_normalized_depth_to_world`, `project_world_to_pixel`, and `project_world_to_final_pixel`
are reused UNCHANGED - only `CANVAS_SIZE` and `BASE_PROJECTION` differ, and here `BASE_PROJECTION`
is loaded directly from the yaml's precomputed matrix rather than derived from intrinsics/pose,
since the yaml already provides the final matrix.

Verified empirically (mtlfd_adaptation session notes / scratchpad test_projection.py): forward-
projecting real trajectories' `obs['eef_pos']` (base_link frame) through this matrix's inverse and
drawing the result on the matching `camera_front_image` frame lands exactly on the gripper across
every sampled timestep of a real trajectory.
"""
import os

import numpy as np
import yaml

from utils.projection_utils import compute_crop_adjustment, embed_mat

_YAML_PATH = os.path.join(os.path.dirname(__file__), 'ur5e_zed_front_projection.yaml')
with open(_YAML_PATH) as _f:
    _CFG = yaml.safe_load(_f)

# (rows, cols) of camera_front_image as stored in the real dataset pkls (ZED VGA resolution),
# before any dataset-side crop/resize - the size compute_crop_adjustment needs as "size_before".
CANVAS_SIZE = (_CFG['camera']['image_height'], _CFG['camera']['image_width'])

BASE_PROJECTION = np.array(_CFG['projection_matrix'], dtype=np.float32)


def _adjust_augmentations(stats, size):
    """Identical to camera_projection.py::_adjust_augmentations - see that module for rationale
    on why this is reimplemented locally rather than imported from hem/datasets/agent_dataset.py."""
    random_crop_adjust = compute_crop_adjustment(stats['crop'], size)
    random_trans_adjust = np.eye(4)
    trans = 2 * stats['trans'] / np.flip(size)
    trans[1] = -trans[1]
    random_trans_adjust[:3, 2] = embed_mat(trans, 3)
    return (np.linalg.inv(random_crop_adjust) @ np.linalg.inv(random_trans_adjust)
            @ embed_mat(stats['flip']))


def build_sample_projection(crop, stats, final_size):
    """Per-sample `projection_matrix` for compute_loss_trajectory's image_waypoints=True branch -
    see camera_projection.py::build_sample_projection for the full parameter contract. Identical
    logic, just built from this module's real-camera BASE_PROJECTION/CANVAS_SIZE."""
    crop_adjust = compute_crop_adjustment(crop, CANVAS_SIZE)
    projection = BASE_PROJECTION.astype(np.float64) @ np.linalg.inv(crop_adjust)
    projection = projection @ _adjust_augmentations(stats, final_size)
    return projection.astype(np.float32)


# stats for a no-op augmentation - see camera_projection.py::NO_AUGMENTATION_STATS.
NO_AUGMENTATION_STATS = {'crop': [0, 0, 0, 0], 'trans': np.array([0.0, 0.0]), 'flip': np.eye(3)}


def project_normalized_depth_to_world(uvz, projection):
    """Identical to camera_projection.py::project_normalized_depth_to_world."""
    uvz = np.asarray(uvz, dtype=np.float64)
    hom = np.concatenate([uvz[..., :2] * uvz[..., 2:3], uvz[..., 2:3],
                           np.ones_like(uvz[..., :1])], axis=-1)
    world_hom = hom @ np.asarray(projection, dtype=np.float64).T
    return world_hom[..., :3]


def project_world_to_pixel(world_xyz, projection, final_size):
    """Identical to camera_projection.py::project_world_to_pixel."""
    H, W = final_size
    world_xyz = np.asarray(world_xyz, dtype=np.float64)
    ones = np.ones(world_xyz.shape[:-1] + (1,), dtype=np.float64)
    world_hom = np.concatenate([world_xyz, ones], axis=-1)
    uvz1 = world_hom @ np.linalg.inv(np.asarray(projection, dtype=np.float64)).T
    z = uvz1[..., 2]
    with np.errstate(invalid='ignore', divide='ignore'):
        u, v = uvz1[..., 0] / z, uvz1[..., 1] / z
    row = np.where(z > 1e-6, H * (1 - v) / 2, np.nan)
    col = np.where(z > 1e-6, (u + 1) * W / 2, np.nan)
    return row, col


def project_world_to_final_pixel(world_xyz, crop, final_size):
    """Identical to camera_projection.py::project_world_to_final_pixel."""
    projection = build_sample_projection(crop, NO_AUGMENTATION_STATS, final_size)
    row, col = project_world_to_pixel(np.asarray(world_xyz, dtype=np.float64), projection, final_size)
    if not np.isfinite(row):
        return None
    return float(row), float(col)
