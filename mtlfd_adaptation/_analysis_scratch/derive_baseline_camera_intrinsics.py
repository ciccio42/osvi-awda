"""
Derives a static camera intrinsic+extrinsic projection (world XYZ -> pixel) for the baseline
Panda pick-place dataset (dataset/panda/*.pkl), for use in diagnose_waypoint_error_baseline.py's
--plot-dir overlays.

Why this is needed: the projection_matrix baked into AgentTeacherDataset samples
(hem/datasets/agent_dataset.py::traj_to_base_matrix -> INVERSE_MATS['ost']) is a PURE extrinsic
(world<->camera) with NO intrinsics - correct for compute_loss_trajectory's loss math (which
treats predicted (u,v) as raw perspective ray ratios), but NOT sufficient to place a world point
at the right PIXEL on screen for visualization - see baseline_data/gen_projection_matrices.py's
docstring. The actual ground-truth obs['eef_point'] pixel labels stored in the dataset were
computed by a totally separate function - hem/robosuite/custom_ik_wrapper.py::project_point -
using the LIVE sim's cam_fovy for real intrinsics. This script spins up that same live env once,
extracts fovy/extrinsic for the 'frontview' camera (fixed/non-articulated, so a one-time snapshot
is valid for the whole dataset - same reasoning as mtlfd_adaptation/camera_projection.py's
BASE_PROJECTION), builds a closed-form K+extrinsic projection, and VALIDATES it against real
stored obs['eef_point'] labels from the actual dataset pkls before printing the constants.

Run inside the 'awda' env (needs mujoco_py/robosuite):
    PYTHONPATH=. python mtlfd_adaptation/_analysis_scratch/derive_baseline_camera_intrinsics.py
"""
import os
import sys

import numpy as np

REPO_ROOT = "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
sys.path.insert(0, REPO_ROOT)

_AXIS_CORRECTION = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def main():
    from hem.robosuite import get_env
    from hem.datasets import load_traj, get_files

    # hem/robosuite/custom_ik_wrapper.py::project_point hardcodes frame_width=frame_height=320 as
    # DEFAULTS regardless of the actual render resolution (camera_height/width the env was created
    # with) - post_proc_obs calls it with no override, so those defaults (not the dataset's real
    # 240-row image) are what generated the stored obs['eef_point'] labels. Must replicate exactly,
    # bug-for-bug, to match. The env itself still needs *some* camera_height/width to construct -
    # use 320x320 too so sim.model.cam_fovy is queried from the same config either way (fovy is
    # resolution-independent, but keep it consistent for clarity).
    PROJECT_POINT_H, PROJECT_POINT_W = 320, 320
    CROP = [80, 0]   # project_point's own default crop offset

    env = get_env('PandaPickPlaceDistractor', use_camera_obs=True,
                   camera_height=PROJECT_POINT_H, camera_width=PROJECT_POINT_W)
    env.reset()
    sim = env.sim
    cam_id = sim.model.camera_name2id('frontview')
    fovy = float(sim.model.cam_fovy[cam_id])
    cam_pos = np.array(sim.data.cam_xpos[cam_id])
    cam_rot = np.array(sim.data.cam_xmat[cam_id]).reshape(3, 3)
    print(f'fovy={fovy}')
    print(f'cam_pos={cam_pos.tolist()}')
    print(f'cam_rot=\n{cam_rot}')

    f = 0.5 * PROJECT_POINT_H / np.tan(fovy * np.pi / 360)
    K = np.array([[f, 0, PROJECT_POINT_W / 2], [0, f, PROJECT_POINT_H / 2], [0, 0, 1]])
    print(f'f={f}')
    print(f'K=\n{K}')

    def project(world_pt):
        """Bug-for-bug port of hem/robosuite/custom_ik_wrapper.py::project_point (including its
        320x320-regardless-of-actual-resolution intrinsics and its row/col-swapped local var
        names), returning (row, col) exactly as stored in obs['eef_point']."""
        model_matrix = np.zeros((3, 4))
        model_matrix[:3, :3] = cam_rot.T
        cam_coord = np.ones((4, 1))
        cam_coord[:3, 0] = np.asarray(world_pt) - cam_pos
        clip = K.dot(model_matrix.dot(cam_coord))
        row, col = clip[:2].reshape(-1) / clip[2]
        row, col = row, PROJECT_POINT_H - col
        col, row = int(max(col - CROP[0], 0)), int(max(row - CROP[1], 0))
        return col, row   # empirically: matches obs['eef_point']'s (row, col) storage order -
        # every validated sample below came out exactly row/col-transposed until this swap was
        # added (project_point's own "row, col = clip[:2]/clip[2]" naming is misleading: clip[0]
        # is actually the column-like axis and clip[1] the row-like axis for this camera, so its
        # local var names are swapped relative to what they end up meaning after the H-flip).

    # ---- validate against real stored obs['eef_point'] labels ----
    files = sorted(get_files(os.path.join(REPO_ROOT, 'dataset/panda/*.pkl')))
    traj = load_traj(files[0])
    print('\n=== validation against real dataset/panda/*.pkl obs[\'eef_point\'] ===')
    max_err = 0.0
    for i in [0, 5, 10, 15, 20, 30]:
        t = traj.get(i)
        ee = t['obs']['eef_pos']
        stored = t['obs']['eef_point']  # (row, col) as stored (post_proc_obs's own order)
        row, col = project(ee)
        err = np.hypot(row - stored[0], col - stored[1])
        max_err = max(max_err, err)
        print(f'frame {i:3d}: stored(row,col)=({stored[0]},{stored[1]})  '
              f'projected(row,col)=({row},{col})  err={err:.1f}px')
    print(f'\nmax pixel error: {max_err:.1f}px')


if __name__ == '__main__':
    main()
