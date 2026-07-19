"""
utils/transformation_mat_ost.pkl and utils/transformation_mats_square.pkl are missing (another
casualty of .gitignore's blanket *.png/*.pkl rules - see backfill_textures.py's docstring for the
PNG half of this). Unlike the textures, these are load-bearing for training correctness, not
cosmetic: hem/datasets/agent_dataset.py::traj_to_base_matrix falls back to
INVERSE_MATS['ost'] = inv(transformation_mat_ost.pkl's contents) for any trajectory whose obs
doesn't carry its own 'world_to_image_transform'/'cam_mat' (true for our Panda/Sawyer
PickPlaceDistractor data) and pick_place_simple.yaml sets image_waypoints: True, so this matrix
directly transforms the model's predicted image-plane waypoints back into world coordinates for
the SDTW loss (scripts/train_transformer.py::compute_loss_trajectory) - a wrong matrix would
distort the training signal, not just look wrong.

Traced compute_loss_trajectory's exact math: it treats all_waypoints[...,:2] as raw perspective
ray ratios (x_cam/z_cam, y_cam/z_cam) with NO intrinsics (fx/fy/cx/cy) applied - i.e. `projection`
must be a pure rigid extrinsic transform (camera<->world), not an intrinsics-aware
pixel-projection matrix. And traj_to_base_matrix takes np.linalg.inv() of whatever's in the pkl to
get the actually-used camera->world transform. So this file must contain the WORLD->CAMERA
extrinsic (its inverse is what's actually used).

Per user decision: rather than reconstruct the original (lost, unrecoverable) calibration, derive
a real, geometrically-correct one from our own robosuite Panda camera ('frontview', the default
camera_name in hem/robosuite/panda/panda_pick_place.py, confirmed as the source of obs['image']) -
reusing the exact camera-extrinsic math already validated in the earlier mtlfd_adaptation task
(mtlfd_adaptation/robosuite_camera_utils.py::get_camera_extrinsic_matrix), just against this
repo's own vendored robosuite/hem.robosuite instead.

Run inside the 'awda' env (needs mujoco_py/robosuite to construct a live env):
    PYTHONPATH=. python baseline_data/gen_projection_matrices.py
"""
import os
import pickle as pkl
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# converts MuJoCo's native camera frame (x right, y up, z out of the screen) to the standard
# vision/pinhole convention (x right, y down, z into the scene) - see mtlfd_adaptation's
# robosuite_camera_utils.py, same correction.
_AXIS_CORRECTION = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def get_camera_extrinsic_matrix(sim, camera_name, transform_utils):
    """Returns the 4x4 camera->world transform, in vision-convention camera frame."""
    cam_id = sim.model.camera_name2id(camera_name)
    camera_pos = np.array(sim.data.cam_xpos[cam_id])
    camera_rot = np.array(sim.data.cam_xmat[cam_id]).reshape(3, 3)
    pose = transform_utils.make_pose(camera_pos, camera_rot)
    return pose @ _AXIS_CORRECTION


def main():
    import robosuite.utils.transform_utils as T
    from hem.robosuite import get_env

    env = get_env('PandaPickPlaceDistractor', use_camera_obs=True,
                   camera_height=320, camera_width=320)
    env.reset()

    cam_to_world = get_camera_extrinsic_matrix(env.sim, 'frontview', T)
    world_to_cam = np.linalg.inv(cam_to_world)

    print("camera-to-world (what will actually be used, via INVERSE_MATS['ost']):")
    print(cam_to_world)
    print("camera position (world frame):", cam_to_world[:3, 3])

    os.makedirs(os.path.join(REPO_ROOT, 'utils'), exist_ok=True)
    with open(os.path.join(REPO_ROOT, 'utils', 'transformation_mat_ost.pkl'), 'wb') as f:
        pkl.dump(world_to_cam, f)
    print("wrote utils/transformation_mat_ost.pkl (world->camera; agent_dataset.py inverts it)")

    # transformation_mats_square.pkl is a dict keyed by traj.setting_name (metaworld/mosaic task
    # names) - never indexed by our Panda/Sawyer path (which always falls through to the 'ost'
    # fallback above, see traj_to_base_matrix), but the top-level `pkl.load(...).items()` at
    # agent_dataset.py module-import time still needs *some* valid dict to exist. Empty is
    # correct here - inventing fake per-task calibrations nobody will ever look up would be worse
    # than just not having them.
    square_path = os.path.join(REPO_ROOT, 'utils', 'transformation_mats_square.pkl')
    if not os.path.exists(square_path):
        with open(square_path, 'wb') as f:
            pkl.dump({}, f)
        print("wrote utils/transformation_mats_square.pkl (empty dict - never indexed by our path)")


if __name__ == '__main__':
    main()
