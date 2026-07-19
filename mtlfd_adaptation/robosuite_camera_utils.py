"""
Camera intrinsics/extrinsics/backprojection helpers for the live UR5e robosuite sim.

NOTE: robosuite's own `robosuite.utils.camera_utils` (which ships these exact utilities in
upstream/PyPI robosuite) does NOT exist in the vendored fork this project uses
(Multi-Task-LFD-Training-Framework/../robosuite, pinned to an older robosuite checkout without
that module) - confirmed by grepping the installed package. These are reimplemented here from
MuJoCo's core (version-independent) camera fields (`cam_fovy`, `cam_xpos`, `cam_xmat`,
`model.stat.extent`, `model.vis.map.{zfar,znear}`), matching the standard formulas robosuite uses
upstream and the depth-conversion formula already used in-repo by
`multi_task_robosuite_env/custom_osc_pose_wrapper.py::_get_real_depth`. See
docs/03_test_adaptation.md.
"""
import numpy as np
import robosuite.utils.transform_utils as T

# converts MuJoCo's native camera frame (x right, y up, z out of the screen) to the standard
# vision/pinhole convention (x right, y down, z into the scene) used below.
_AXIS_CORRECTION = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width):
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = sim.model.cam_fovy[cam_id]
    f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
    return np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])


def get_camera_extrinsic_matrix(sim, camera_name):
    """Returns the 4x4 camera->world transform, in the vision-convention camera frame."""
    cam_id = sim.model.camera_name2id(camera_name)
    camera_pos = np.array(sim.data.cam_xpos[cam_id])
    camera_rot = np.array(sim.data.cam_xmat[cam_id]).reshape(3, 3)
    pose = T.make_pose(camera_pos, camera_rot)
    return pose @ _AXIS_CORRECTION


def get_real_depth_map(sim, depth_map):
    """depth_map: normalized [0,1] depth buffer -> real depth in meters."""
    extent = sim.model.stat.extent
    far = sim.model.vis.map.zfar * extent
    near = sim.model.vis.map.znear * extent
    return near / (1.0 - depth_map * (1.0 - near / far))


def project_world_to_pixel(point_world, intrinsic, extrinsic_cam_to_world):
    """Inverse of pixel_to_world: project a 3D world point to (row, col) pixel coords, for debug
    visualization only."""
    point_cam = np.linalg.inv(extrinsic_cam_to_world) @ np.append(point_world, 1.0)
    x, y, z = point_cam[:3]
    if z <= 1e-6:
        return None
    col = intrinsic[0, 0] * x / z + intrinsic[0, 2]
    row = intrinsic[1, 1] * y / z + intrinsic[1, 2]
    return row, col


def pixel_to_world(row, col, depth, intrinsic, extrinsic_cam_to_world):
    """Backproject a single (row, col) pixel with known real depth (meters) to a 3D world point,
    in the same world frame as env.sim.data.site_xpos / obs['ee_aa'] / obs['*_pos']."""
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x_cam = (col - cx) * depth / fx
    y_cam = (row - cy) * depth / fy
    point_cam = np.array([x_cam, y_cam, depth, 1.0])
    point_world = extrinsic_cam_to_world @ point_cam
    return point_world[:3]
