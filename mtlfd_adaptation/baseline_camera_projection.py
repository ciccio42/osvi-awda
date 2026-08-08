"""
Static camera model for the ORIGINAL osvi-awda baseline dataset (dataset/panda/*.pkl, PandaPick
PlaceDistractor's 'frontview' camera - fixed/non-articulated, same reasoning as
camera_projection.py's BASE_PROJECTION for the ur5e/mtlfd case), used ONLY for building
`--plot-dir` visualization overlays in diagnose_waypoint_error_baseline.py.

Why this can't reuse camera_projection.py's project_world_to_pixel: the baseline's own
`projection_matrix` (traj_to_base_matrix -> INVERSE_MATS['ost'], loaded per-sample by
AgentTeacherDataset) is a PURE rigid extrinsic with NO camera intrinsics baked in - correct for
compute_loss_trajectory's loss math (see baseline_data/gen_projection_matrices.py's docstring:
predicted (u,v) are raw perspective ray ratios x_cam/z_cam, y_cam/z_cam, not K-normalized
[-1,1] coords), but insufficient on its own to place a world point at a real screen pixel.

The ACTUAL ground-truth pixel labels stored in the dataset (obs['eef_point']) come from a
completely separate function - hem/robosuite/custom_ik_wrapper.py::project_point - which builds
its own intrinsics from the live sim's cam_fovy. FOVY/EXTRINSIC below were extracted from a live
PandaPickPlaceDistractor env once (mtlfd_adaptation/_analysis_scratch/
derive_baseline_camera_intrinsics.py) and the resulting closed-form projection was verified to
reproduce real stored obs['eef_point'] labels to EXACTLY 0px error (6/6 sampled frames) - see that
script's output for the validation transcript.

Note project_point's own quirks, reproduced here bug-for-bug since matching its actual behavior
(not some "corrected" version) is the goal:
  - Its intrinsics are built for a 320x320 canvas regardless of the dataset's real 240-row images
    (post_proc_obs never overrides project_point's frame_width/frame_height defaults).
  - Its own crop=[80,0] default is already baked into the returned pixel (folded into the
    principal point below, effectively cy=80 instead of 160), landing directly in the SAME
    (240, 320) space the dataset's raw stored obs['image'] uses - i.e. BASELINE_CANVAS_SIZE below.
  - Its row/col are effectively swapped+sign-flipped relative to a textbook pinhole formula - see
    the closed form in project_world_to_pixel_baseline's docstring for the exact (validated) math.
"""
import numpy as np

BASELINE_FOVY_DEG = 45.0
_CAM_POS = np.array([1.6, 0.10000000000000003, 1.75])
# raw sim.data.cam_xmat for 'frontview', reshaped (3,3) - NOT axis-corrected (unlike
# camera_projection.py's ur5e recipe): project_point uses this matrix as-is (transposed) and that
# is what reproduced real eef_point labels exactly.
_CAM_ROT = np.array([
    [-1.66533454e-16, 4.24914842e-01, 9.05233327e-01],
    [-9.99998732e-01, -1.44172250e-03, 6.76741862e-04],
    [1.59265292e-03, -9.05232179e-01, 4.24914303e-01],
])
_PROJECT_POINT_CANVAS = 320   # project_point's hardcoded intrinsics canvas (both H and W)
_F = 0.5 * _PROJECT_POINT_CANVAS / np.tan(BASELINE_FOVY_DEG * np.pi / 360)
_CROP_ROW_OFFSET = 80   # project_point's own default crop=[80,0], folded into cy below

# (rows, cols) of the RAW dataset-stored obs['image'] - i.e. AFTER project_point's internal
# crop=[80,0] (320 - 80 = 240), BEFORE any dataset-side (AgentDemonstrations) crop/resize.
BASELINE_CANVAS_SIZE = (240, 320)


def project_world_to_pixel_baseline(world_xyz):
    """world_xyz: (..., 3) -> (row, col), each (...) shaped, in BASELINE_CANVAS_SIZE pixel space
    (NaN where behind the camera). Validated closed form (0px error against 6 real
    obs['eef_point'] samples spanning one full episode):
        [Xcam, Ycam, Zcam] = _CAM_ROT.T @ (world_xyz - _CAM_POS)
        row = _CROP_ROW_OFFSET/... - _F * Ycam / Zcam   (= 80 - f*Ycam/Zcam)
        col = _PROJECT_POINT_CANVAS/2 + _F * Xcam / Zcam   (= 160 + f*Xcam/Zcam)
    """
    world_xyz = np.asarray(world_xyz, dtype=np.float64)
    cam_frame = (world_xyz - _CAM_POS) @ _CAM_ROT   # (..., 3): R.T @ v done as v @ R
    x_cam, y_cam, z_cam = cam_frame[..., 0], cam_frame[..., 1], cam_frame[..., 2]
    cy = _PROJECT_POINT_CANVAS / 2 - _CROP_ROW_OFFSET   # = 80
    cx = _PROJECT_POINT_CANVAS / 2                        # = 160
    # _CAM_ROT is the RAW (un-axis-corrected) mujoco camera matrix - native mujoco camera frame
    # has +z pointing OUT of the screen (see camera_projection.py's _AXIS_CORRECTION comment,
    # deliberately NOT applied here since project_point doesn't apply it either), so points in
    # front of the camera have NEGATIVE z_cam, not positive - verified against real eef_point data.
    in_front = z_cam < -1e-6
    with np.errstate(invalid='ignore', divide='ignore'):
        row = np.where(in_front, cy - _F * y_cam / z_cam, np.nan)
        col = np.where(in_front, cx + _F * x_cam / z_cam, np.nan)
    return row, col


def adjust_pixel_for_dataset_crop(row, col, crop, raw_size, final_size):
    """Applies AgentDemonstrations' own deterministic (rt, rb, cl, cr) crop + resize-to-final_size
    on top of a BASELINE_CANVAS_SIZE-space (row, col) from project_world_to_pixel_baseline - the
    same linear crop-then-resize compute_crop_adjustment implements in normalized-coordinate
    matrix form (utils/projection_utils.py), done directly in pixel space here since there's no
    matrix compositing to gain from going through the 4x4-homogeneous route for a single point set.
    No-op for augmentation randomness (rand_crop/rand_translate/rand_flip) - matches the
    diagnostic's own rand_flip=False, rand_crop=None, rand_translate=None setup."""
    rt, rb, cl, cr = crop
    raw_h, raw_w = raw_size
    final_h, final_w = final_size
    row = (row - rt) * final_h / (raw_h - rt - rb)
    col = (col - cl) * final_w / (raw_w - cl - cr)
    return row, col
