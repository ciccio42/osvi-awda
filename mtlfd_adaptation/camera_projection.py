"""
Static camera_front projection for the ur5e_pick_place robosuite scene, enabling
image_waypoints=True training: the policy predicts (normalized image x, normalized image y,
depth) per waypoint, and `scripts/train_transformer.py::compute_loss_trajectory` projects that
through `traj['projection_matrix']` into world coordinates for the SoftDTW loss, exactly mirroring
osvi-awda's baseline (panda/sawyer) formulation instead of directly regressing metric 3D from RGB.

`camera_front` is a fixed, non-articulated scene camera (declared once in
Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/config/PickPlaceDistractor.yaml,
never gripper-mounted or moved), so a single constant matrix (`BASE_PROJECTION`) is valid for
every frame in the whole dataset - no live MuJoCo sim needed in the dataloader.

Validated (see mtlfd_adaptation/_analysis_scratch/validate_camera_proj.py):
- the closed-form intrinsic/extrinsic built here from the yaml's static pose+fovy match the live
  sim's actual `sim.model.cam_fovy` / `sim.data.cam_xpos` / `sim.data.cam_xmat` for camera_front to
  ~1e-7 (float precision), confirming the static-camera assumption and the yaml numbers.
- projecting real trajectory `ee_aa[:3]` points through this exact camera model reproduces the
  dataset's own stored `obs['eef_point']` field to sub-pixel accuracy (no extra flips needed -
  `robosuite_camera_utils.project_world_to_pixel` already matches this convention).
- BASE_PROJECTION itself round-trips synthetic world points to ~1e-16 (float noise): project to
  (row, col, depth), convert to this module's normalized-homogeneous form, multiply by
  BASE_PROJECTION, recover the original world point.

The 4x4 BASE_PROJECTION maps [u*z, v*z, z, 1] -> [world_x, world_y, world_z, 1], where (u, v) is
osvi-awda's own normalized image convention ([-1, 1], +x right, +y up - see
utils/projection_utils.py::image_point_to_pixels) and z is metric depth in meters along the
camera's optical axis - exactly the convention `compute_loss_trajectory`'s image_waypoints branch
assumes for its `hom_im_coords`/`projection` multiplication.
"""
import numpy as np

from utils.projection_utils import compute_crop_adjustment, embed_mat

# From PickPlaceDistractor.yaml: camera_heights/camera_widths, camera_poses.camera_front
# (position, quaternion in w,x,y,z order), camera_attribs.fovy.
_CAMERA_HEIGHT, _CAMERA_WIDTH = 200, 360
_CAM_POS = np.array([0.45, -0.002826249197217832, 1.27])
_CAM_QUAT_WXYZ = np.array([0.6620018964346217, 0.26169506249574287,
                            0.25790267731943883, 0.6532651777140575])
_FOVY_DEG = 60.0

# MuJoCo's native camera frame (x right, y up, z out of screen) -> vision/pinhole convention
# (x right, y down, z into the scene) - matches robosuite_camera_utils.py's _AXIS_CORRECTION.
_AXIS_CORRECTION = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# (rows, cols) of camera_front_image as stored in the dataset pkls, before any dataset-side
# crop/resize - the size compute_crop_adjustment needs as "size_before".
CANVAS_SIZE = (_CAMERA_HEIGHT, _CAMERA_WIDTH)


def _quat_to_rotmat(w, x, y, z):
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
    ])


def _build_base_projection():
    H, W = _CAMERA_HEIGHT, _CAMERA_WIDTH
    R = _quat_to_rotmat(*_CAM_QUAT_WXYZ)
    f = 0.5 * H / np.tan(_FOVY_DEG * np.pi / 360)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])

    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = _CAM_POS
    E = pose @ _AXIS_CORRECTION  # camera -> world (vision convention)

    # [u*z, v*z, z, 1] -> [col*z, row*z, z, 1], from col=(u+1)*W/2, row=H*(1-v)/2
    P1 = np.array([
        [W / 2, 0, W / 2, 0],
        [0, -H / 2, H / 2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    # [col*z, row*z, z, 1] -> [X_cam, Y_cam, Z_cam, 1] (inverse pinhole intrinsic)
    P2 = np.eye(4)
    P2[:3, :3] = np.linalg.inv(K)

    return E @ P2 @ P1


BASE_PROJECTION = _build_base_projection().astype(np.float32)


def _adjust_augmentations(stats, size):
    """Mirrors hem/datasets/agent_dataset.py::adjust_augmentations (reimplemented locally rather
    than imported, since that module pulls in unrelated baseline-only dependencies - a
    module-level pickle load of `utils/transformation_mats_square.pkl` and
    envs.metaworld_env_data/envs.mosaic_env_data - at import time that this dataset never needs).
    """
    random_crop_adjust = compute_crop_adjustment(stats['crop'], size)
    random_trans_adjust = np.eye(4)
    trans = 2 * stats['trans'] / np.flip(size)
    trans[1] = -trans[1]
    random_trans_adjust[:3, 2] = embed_mat(trans, 3)
    return (np.linalg.inv(random_crop_adjust) @ np.linalg.inv(random_trans_adjust)
            @ embed_mat(stats['flip']))


def build_sample_projection(crop, stats, final_size):
    """Per-sample `projection_matrix` for compute_loss_trajectory's image_waypoints=True branch:
    maps [u*z, v*z, z, 1] in the FINAL (post deterministic-crop + resize + random-augmentation)
    image's normalized coords to world [x, y, z, 1].

    `crop` - the dataset's deterministic (rt, rb, cl, cr) crop tuple, applied to the raw
        CANVAS_SIZE image (matches `hem.datasets.util.crop`'s convention).
    `stats` - the dict returned as the second element of `randomize_video(...)` for this exact
        sample (`{'crop': [r1,r2,c1,c2], 'trans': ndarray(2,), 'flip': 3x3 ndarray}`) - capture it
        instead of discarding it as `_`.
    `final_size` - (height, width) of the image the network actually sees (i.e. (self.height,
        self.width) in MTLFDAgentTeacherDataset).
    """
    crop_adjust = compute_crop_adjustment(crop, CANVAS_SIZE)
    projection = BASE_PROJECTION.astype(np.float64) @ np.linalg.inv(crop_adjust)
    projection = projection @ _adjust_augmentations(stats, final_size)
    return projection.astype(np.float32)


# stats for a no-op augmentation (crop=[0,0,0,0], trans=[0,0], flip=identity) - _adjust_augmentations
# reduces to the identity matrix for these, verified in
# mtlfd_adaptation/_analysis_scratch/validate_camera_proj.py. Use with build_sample_projection at
# inference time (test_mtlfd_rollout.py), where preprocess_frame/make_demo_context apply only the
# deterministic crop+resize, no random augmentation.
NO_AUGMENTATION_STATS = {'crop': [0, 0, 0, 0], 'trans': np.array([0.0, 0.0]), 'flip': np.eye(3)}


def project_normalized_depth_to_world(uvz, projection):
    """Inverse of the transform compute_loss_trajectory's image_waypoints branch applies to a
    policy's raw waypoint output: given (..., 3) array of (u, v, z) in this module's normalized
    [-1,1]-image + metric-depth convention (i.e. `out['waypoints'][..., :3]` when the policy was
    trained with image_waypoints=True), returns (..., 3) world xyz via the same `projection`
    matrix `build_sample_projection` produces."""
    uvz = np.asarray(uvz, dtype=np.float64)
    hom = np.concatenate([uvz[..., :2] * uvz[..., 2:3], uvz[..., 2:3],
                           np.ones_like(uvz[..., :1])], axis=-1)
    world_hom = hom @ np.asarray(projection, dtype=np.float64).T
    return world_hom[..., :3]


def project_world_to_pixel(world_xyz, projection, final_size):
    """Vectorized world XYZ -> (row, col) pixel coords, given an ALREADY-BUILT `projection` matrix
    (whatever build_sample_projection(crop, stats, final_size) produced - stats may be a real
    per-sample augmentation, not necessarily NO_AUGMENTATION_STATS). This is the forward direction
    of project_normalized_depth_to_world/build_sample_projection: `projection` maps
    [u*z, v*z, z, 1] -> world [x, y, z, 1], so getting back to pixels means INVERTING it, not
    applying it directly - a point-blank `hom @ projection.T` (as an earlier, buggy version of
    debug_utils.py::project_points did) silently produces meaningless coordinates, since it
    multiplies by the matrix in the wrong direction entirely.

    world_xyz: (..., 3) array. Returns (row, col), each (...) shaped, NaN where the point is
    behind the camera (non-positive depth) - vectorized callers should treat NaN as "don't draw".
    """
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
    """Single-point convenience wrapper around project_world_to_pixel for the common eval-time
    case: build the projection fresh from `crop` with NO_AUGMENTATION_STATS (no random
    crop/flip/translate - matches preprocess_frame's deterministic-only preprocessing), for
    debug visualization overlays where the background image is `preprocess_frame(...)`'s output.
    Returns None if the point is behind the camera (non-positive depth).

    Do NOT substitute a freshly-computed `get_camera_intrinsic_matrix(sim, 'camera_front', height,
    width)` here - that computes intrinsics as if the camera natively captured at the final
    (height, width) resolution, which ignores the crop entirely and is wrong by a large,
    non-uniform factor (verified: ~1.3-1.8x off in focal length alone, plus a wrong principal
    point, for this project's crop=(20,25,80,75)) - it silently pushes projected points outside
    the visible frame rather than erroring, which is why earlier debug images that used it showed
    no plan overlay at all.
    """
    projection = build_sample_projection(crop, NO_AUGMENTATION_STATS, final_size)
    row, col = project_world_to_pixel(np.asarray(world_xyz, dtype=np.float64), projection, final_size)
    if not np.isfinite(row):
        return None
    return float(row), float(col)
