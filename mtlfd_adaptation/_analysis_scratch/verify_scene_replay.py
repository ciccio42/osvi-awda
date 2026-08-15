"""
One-off validation for set_objects_from_training_trajectory: reset a live env, teleport its
objects to match a real training trajectory's layout, then compare the LIVE env's own recomputed
obj_bb (from the actual new object positions) against the TRAINING trajectory's stored obj_bb for
the same objects. If the qpos reconstruction is correct, these should match closely (same object
positions -> same camera_front projection -> same bounding boxes) - this is a much stronger check
than eyeballing an image, since it's the env's own bbox-computation code validating the teleport.

Usage: python -u mtlfd_adaptation/_analysis_scratch/verify_scene_replay.py <task_id> <agent_pkl>
"""
import os
import sys

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from mtlfd_adaptation.test_mtlfd_rollout import (  # noqa: E402
    CONTROLLER_PATH, AGENT_SETTLE_STEPS, get_env, load_controller_config,
    set_objects_from_training_trajectory, stabilize,
)
from mtlfd_adaptation.trajectory_bridge import load_traj  # noqa: E402


def main():
    task_id = int(sys.argv[1])
    agent_pkl = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else '.'

    controller_config = load_controller_config(custom_fpath=CONTROLLER_PATH)
    action_ranges = np.array([[-0.05, 0.25], [-0.45, 0.5], [0.82, 1.2], [-5, 5], [-5, 5], [-5, 5]])
    env = get_env('UR5e_PickPlaceDistractor', controller_configs=controller_config,
                   task_id=task_id, has_renderer=False, has_offscreen_renderer=True,
                   reward_shaping=False, use_camera_obs=True, ranges=action_ranges,
                   render_gpu_device_id=0, render_camera='camera_front', object_set=2)
    obs = env.reset()
    print('[verify] pre-teleport obj_bb (camera_front):')
    for name, bb in obs['obj_bb']['camera_front'].items():
        print(f'  {name}: {bb["center"]}')

    set_objects_from_training_trajectory(env, agent_pkl)
    obs = stabilize(env, n_steps=5)   # a few settle steps, matching normal rollout usage

    live_bb = obs['obj_bb']['camera_front']
    print('[verify] post-teleport LIVE obj_bb (camera_front):')
    for name, bb in live_bb.items():
        print(f'  {name}: {bb["center"]}')

    agent_traj, _ = load_traj(agent_pkl)
    stored_bb = agent_traj.get(AGENT_SETTLE_STEPS)['obs']['obj_bb']['camera_front']
    print('[verify] TRAINING TRAJECTORY stored obj_bb (camera_front), same objects:')
    for name, bb in stored_bb.items():
        print(f'  {name}: {bb["center"]}')

    print('[verify] pixel distance (live teleported vs. training stored), per object:')
    max_dist = 0.0
    for name in stored_bb:
        if name == 'bin' or name not in live_bb:
            continue
        d = float(np.hypot(*(np.array(live_bb[name]['center']) - np.array(stored_bb[name]['center']))))
        max_dist = max(max_dist, d)
        print(f'  {name}: {d:.2f}px')
    print(f'[verify] max pixel distance: {max_dist:.2f}px '
          f'({"PASS - looks correct" if max_dist < 5 else "FAIL - investigate"})')

    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(obs['camera_front_image']).save(os.path.join(out_dir, 'live_teleported.png'))
    Image.fromarray(agent_traj.get(AGENT_SETTLE_STEPS)['obs']['image']).save(
        os.path.join(out_dir, 'training_stored.png'))
    print(f'[verify] wrote {out_dir}/live_teleported.png, {out_dir}/training_stored.png')
    env.close()


if __name__ == '__main__':
    main()
