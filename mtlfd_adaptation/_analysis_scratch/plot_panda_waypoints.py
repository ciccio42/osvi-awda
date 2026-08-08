import os
import pickle
import numpy as np
import cv2

REPO_ROOT = "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
OUT_DIR = os.path.join(REPO_ROOT, "mtlfd_adaptation", "_analysis_scratch", "panda_out")
os.makedirs(OUT_DIR, exist_ok=True)

from hem.robosuite import get_env  # noqa: E402
import robosuite.utils.transform_utils as T  # noqa: E402


def get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width):
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = sim.model.cam_fovy[cam_id]
    f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
    return np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])


_AXIS_CORRECTION = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def get_camera_extrinsic_matrix(sim, camera_name):
    cam_id = sim.model.camera_name2id(camera_name)
    camera_pos = np.array(sim.data.cam_xpos[cam_id])
    camera_rot = np.array(sim.data.cam_xmat[cam_id]).reshape(3, 3)
    pose = T.make_pose(camera_pos, camera_rot)
    return pose @ _AXIS_CORRECTION


def project_world_to_pixel(point_world, intrinsic, extrinsic_cam_to_world):
    point_cam = np.linalg.inv(extrinsic_cam_to_world) @ np.append(point_world, 1.0)
    x, y, z = point_cam[:3]
    if z <= 1e-6:
        return None
    col = intrinsic[0, 0] * x / z + intrinsic[0, 2]
    row = intrinsic[1, 1] * y / z + intrinsic[1, 2]
    return row, col


env = get_env('PandaPickPlaceDistractor', use_camera_obs=True, camera_height=240, camera_width=320)
env.reset()
K = get_camera_intrinsic_matrix(env.sim, 'frontview', 240, 320)
Rt = get_camera_extrinsic_matrix(env.sim, 'frontview')
env.close()

EXAMPLES = ["traj0.pkl", "traj500.pkl"]

for fname in EXAMPLES:
    path = os.path.join(REPO_ROOT, "dataset", "panda", fname)
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    traj = loaded["traj"]
    n = len(traj)
    out_inds = np.linspace(0, n - 1, num=50, endpoint=True, dtype=int)
    poses = np.stack([traj[i]['obs']['ee_aa'][:3] for i in out_inds])
    grasp_frames = [False] + [traj[i]['action'][-1] > 0.01 for i in range(1, n)]
    grasps = np.stack([grasp_frames[i] for i in out_inds])

    img0 = traj[0]['obs']['image'].copy()
    img = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
    prev_px = None
    n_projected = 0
    for pos, g in zip(poses, grasps):
        proj = project_world_to_pixel(pos, K, Rt)
        if proj is None:
            continue
        row, col = proj
        color = (0, 0, 255) if g else (0, 255, 0)
        if 0 <= row < img.shape[0] and 0 <= col < img.shape[1]:
            cv2.circle(img, (int(col), int(row)), 3, color, -1)
            n_projected += 1
            if prev_px is not None:
                cv2.line(img, prev_px, (int(col), int(row)), (255, 0, 0), 1)
            prev_px = (int(col), int(row))
    cv2.putText(img, f"panda traj_points (absolute, correct) {n_projected}/50 in-frame", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    out_path = os.path.join(OUT_DIR, f"{fname.replace('.pkl', '')}_gt_waypoints.png")
    cv2.imwrite(out_path, img)
    print("wrote", out_path, f"({n_projected}/50 in-frame)")

print("DONE")
