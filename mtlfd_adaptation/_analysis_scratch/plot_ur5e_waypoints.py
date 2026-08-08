import os, sys
import numpy as np
import cv2
import debugpy

REPO_ROOT = "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
TRAINING_ROOT = "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training"
DEBUG = True
sys.path.insert(0, TRAINING_ROOT)

# Must import robosuite/multi_task_robosuite_env BEFORE osvi-awda's own repo root goes on
# sys.path below - osvi-awda vendors its own old robosuite/ fork (no UR5e/OSC_POSE support) that
# would otherwise shadow the real pip-installed one (see test_mtlfd_rollout.py's own comment).
import robosuite.utils.transform_utils as T  # noqa: E402
from robosuite import load_controller_config  # noqa: E402
from multi_task_robosuite_env import get_env  # noqa: E402

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "mtlfd_adaptation"))
from trajectory_bridge import load_traj  # noqa: E402
from robosuite_camera_utils import (  # noqa: E402
    get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, project_world_to_pixel)

OUT_DIR = "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda/mtlfd_adaptation/_analysis_scratch/ur5e_out"
os.makedirs(OUT_DIR, exist_ok=True)

CONTROLLER_PATH = os.path.join(os.path.dirname(REPO_ROOT), 'Multi-Task-LFD-Training-Framework',
                                'tasks', 'multi_task_robosuite_env', 'controllers', 'config', 'osc_pose.json')
controller_config = load_controller_config(custom_fpath=CONTROLLER_PATH)
action_ranges = np.array([[-0.05, 0.25], [-0.45, 0.5], [0.82, 1.2], [-5, 5], [-5, 5], [-5, 5]])

env = get_env('UR5e_PickPlaceDistractor', controller_configs=controller_config, task_id=1,
               has_renderer=False, has_offscreen_renderer=True, reward_shaping=False,
               use_camera_obs=True, ranges=action_ranges, render_gpu_device_id=0,
               render_camera='camera_front', object_set=2)
obs = env.reset()
start_pos = np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id])
H, W = 200, 360  # matches the dataset's stored image resolution
K = get_camera_intrinsic_matrix(env.sim, 'camera_front', H, W)
Rt = get_camera_extrinsic_matrix(env.sim, 'camera_front')
env.close()
print("live env eef reset position (start_pos):", start_pos)

EXAMPLES = [
    ("task_01", "traj000.pkl"),
    ("task_07", "traj010.pkl"),
]

if DEBUG:
    print("DEBUG: waiting for debugger to attach on port 5678...")
    debugpy.listen(('0.0.0.0', 5678))
    debugpy.wait_for_client()

for task, fname in EXAMPLES:
    path = f"/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset/pick_place/ur5e_pick_place/{task}/{fname}"
    traj, cmd = load_traj(path)
    n = len(traj)
    out_inds = np.linspace(0, n - 1, num=50, endpoint=True, dtype=int)
    poses = np.stack([traj.get(i)['obs']['ee_aa'][:3] for i in out_inds])
    grasp_frames = [False] + [traj.get(i)['action'][-1] > 0.01 for i in range(1, n)]
    grasps = np.stack([grasp_frames[i] for i in out_inds])

    img0 = traj.get(0)['obs']['image'].copy()
    img = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
    img_buggy = img.copy()

    prev_px = None
    for pos, g in zip(poses, grasps):
        proj = project_world_to_pixel(pos, K, Rt)
        if proj is None:
            continue
        row, col = proj
        color = (0, 0, 255) if g else (0, 255, 0)  # red=grasp closed, green=open
        if 0 <= row < img.shape[0] and 0 <= col < img.shape[1]:
            cv2.circle(img, (int(col), int(row)), 3, color, -1)
            if prev_px is not None:
                cv2.line(img, prev_px, (int(col), int(row)), (255, 0, 0), 1)
            prev_px = (int(col), int(row))
    cv2.putText(img, "GROUND TRUTH traj_points (correct, absolute)", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    out_path = os.path.join(OUT_DIR, f"{task}_{fname.replace('.pkl', '')}_gt_correct.png")
    cv2.imwrite(out_path, img)
    print("wrote", out_path, "command:", cmd)

    # Reproduce test_mtlfd_rollout.py's bug: abs_positions = waypoints[:,:3] + start_pos.
    # Ground-truth traj_points are ALREADY absolute, so adding start_pos double-offsets them.
    prev_px = None
    n_projected = 0
    for pos, g in zip(poses, grasps):
        buggy_pos = pos + start_pos
        proj = project_world_to_pixel(buggy_pos, K, Rt)
        color = (0, 0, 255) if g else (0, 255, 0)
        if proj is not None:
            row, col = proj
            if 0 <= row < img_buggy.shape[0] and 0 <= col < img_buggy.shape[1]:
                cv2.circle(img_buggy, (int(col), int(row)), 3, color, -1)
                n_projected += 1
                if prev_px is not None:
                    cv2.line(img_buggy, prev_px, (int(col), int(row)), (255, 0, 0), 1)
                prev_px = (int(col), int(row))
    cv2.putText(img_buggy, f"BUGGY (+start_pos) -> {n_projected}/50 points land in-frame",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    out_path_buggy = os.path.join(OUT_DIR, f"{task}_{fname.replace('.pkl', '')}_bug_shifted.png")
    cv2.imwrite(out_path_buggy, img_buggy)
    print("wrote", out_path_buggy, f"({n_projected}/50 waypoints project inside the frame)")

print("DONE")
