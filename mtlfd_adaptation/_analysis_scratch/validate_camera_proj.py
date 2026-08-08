"""One-off validation: does a camera projection built purely from the static
PickPlaceDistractor.yaml pose+fovy (no live sim access) match the live sim's actual
camera_front intrinsic/extrinsic, and does projecting real trajectory ee_aa[:3] through it
reproduce the dataset's stored eef_point?
"""
import glob
import os
import pickle
import sys

import numpy as np

FRAMEWORK_ROOT = '/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework'
sys.path.insert(0, os.path.join(FRAMEWORK_ROOT, 'training'))
sys.path.insert(0, os.path.join(FRAMEWORK_ROOT, 'tasks'))

# must import the real pip-installed robosuite BEFORE osvi-awda's REPO_ROOT goes on sys.path -
# osvi-awda has its own vendored robosuite/ dir that would otherwise shadow it (see
# test_mtlfd_rollout.py's identical comment/ordering).
import robosuite.utils.transform_utils as T  # noqa: E402
from robosuite import load_controller_config  # noqa: E402
from robosuite.utils import RandomizationError  # noqa: E402
from multi_task_robosuite_env import get_env  # noqa: E402

REPO_ROOT = '/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda'
sys.path.insert(0, REPO_ROOT)

from mtlfd_adaptation.robosuite_camera_utils import (  # noqa: E402
    get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, project_world_to_pixel,
)

CONTROLLER_PATH = os.path.join(FRAMEWORK_ROOT, 'tasks', 'multi_task_robosuite_env',
                                'controllers', 'config', 'osc_pose.json')

# ---- 1. Spin up the live sim once, just to read camera_front's true intrinsic/extrinsic ----
controller_config = load_controller_config(custom_fpath=CONTROLLER_PATH)
action_ranges = np.array([[-0.05, 0.25], [-0.45, 0.5], [0.82, 1.2], [-5, 5], [-5, 5], [-5, 5]])
while True:
    try:
        env = get_env('UR5e_PickPlaceDistractor', controller_configs=controller_config,
                       task_id=0, has_renderer=False, has_offscreen_renderer=True,
                       reward_shaping=False, use_camera_obs=True, ranges=action_ranges,
                       render_gpu_device_id=0, render_camera='camera_front', object_set=2)
        break
    except RandomizationError:
        continue

H, W = 200, 360
K_live = get_camera_intrinsic_matrix(env.sim, 'camera_front', H, W)
E_live = get_camera_extrinsic_matrix(env.sim, 'camera_front')
cam_id = env.sim.model.camera_name2id('camera_front')
print("=== LIVE SIM camera_front ===")
print("fovy:", env.sim.model.cam_fovy[cam_id])
print("cam_xpos:", env.sim.data.cam_xpos[cam_id])
print("cam_xmat:\n", np.array(env.sim.data.cam_xmat[cam_id]).reshape(3, 3))
print("K_live:\n", K_live)
print("E_live (cam->world):\n", E_live)
env.close()

# ---- 2. Build the SAME matrices purely from the static yaml declaration, no live sim ----
yaml_pos = np.array([0.45, -0.002826249197217832, 1.27])
yaml_quat_wxyz = np.array([0.6620018964346217, 0.26169506249574287, 0.25790267731943883, 0.6532651777140575])
yaml_quat_xyzw = np.array([yaml_quat_wxyz[1], yaml_quat_wxyz[2], yaml_quat_wxyz[3], yaml_quat_wxyz[0]])
fovy_yaml = 60.0

R_yaml = T.quat2mat(yaml_quat_xyzw)
f_yaml = 0.5 * H / np.tan(fovy_yaml * np.pi / 360)
K_yaml = np.array([[f_yaml, 0, W / 2], [0, f_yaml, H / 2], [0, 0, 1]])
_AXIS_CORRECTION = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
pose_yaml = T.make_pose(yaml_pos, R_yaml)
E_yaml = pose_yaml @ _AXIS_CORRECTION

print("\n=== CLOSED-FORM FROM YAML (no live sim) ===")
print("K_yaml:\n", K_yaml)
print("E_yaml (cam->world):\n", E_yaml)
print("\n=== DIFF (should be ~0) ===")
print("K diff:", np.abs(K_live - K_yaml).max())
print("E diff:", np.abs(E_live - E_yaml).max())

# ---- 3. Project real trajectory ee_aa[:3] through K_yaml/E_yaml and compare to stored eef_point ----
f = sorted(glob.glob('/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset/pick_place/ur5e_pick_place/task_00/*.pkl'))[0]
traj = pickle.load(open(f, 'rb'))['traj']
print(f"\n=== projecting real frames from {f} ===")
print(f"{'idx':>4} {'stored eef_point':>20} {'proj row,col':>16} {'proj (flip variants)':>60}")
for i in [0, len(traj) // 4, len(traj) // 2, 3 * len(traj) // 4, len(traj) - 1]:
    obs = traj.get(i)['obs']
    ee_pos = obs['ee_aa'][:3]
    stored = obs['eef_point']
    rc = project_world_to_pixel(ee_pos, K_yaml, E_yaml)
    if rc is None:
        print(i, stored, "None (behind camera?)")
        continue
    row, col = rc
    variants = {
        'raw(row,col)': (row, col),
        'row,W-col': (row, W - col),
        'H-row,col': (H - row, col),
        'H-row,W-col': (H - row, W - col),
        'col,row': (col, row),
        'col,H-row': (col, H - row),
        'W-col,row': (W - col, row),
    }
    vstr = ' | '.join(f"{k}=({v[0]:.0f},{v[1]:.0f})" for k, v in variants.items())
    print(f"{i:>4} {str(stored):>20} raw=({row:.1f},{col:.1f})   {vstr}")
