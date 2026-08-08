"""One-off: predict a waypoint plan for a real training-task demo/agent pair and compare the
projected absolute world positions against the real trajectory's own known ee_aa[:3] range, to
check whether the image_waypoints=True projection fix produces plausible workspace coordinates."""
import glob
import os
import sys

import numpy as np
import torch

# must import the real pip-installed robosuite BEFORE osvi-awda's REPO_ROOT goes on sys.path -
# osvi-awda has its own vendored robosuite/ dir that would otherwise shadow it (matches
# test_mtlfd_rollout.py's own comment/ordering).
import robosuite.utils.transform_utils  # noqa: E402,F401
from robosuite import load_controller_config  # noqa: E402,F401

REPO_ROOT = '/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda'
sys.path.insert(0, REPO_ROOT)

from hem.datasets.util import crop as crop_fn  # noqa: E402
from mtlfd_adaptation.camera_projection import (  # noqa: E402
    NO_AUGMENTATION_STATS, build_sample_projection, project_normalized_depth_to_world,
)
from mtlfd_adaptation.test_mtlfd_rollout import load_model, make_demo_context, preprocess_frame, predict_waypoints  # noqa: E402
from mtlfd_adaptation.trajectory_bridge import load_traj  # noqa: E402

MODEL_DIR = sys.argv[1]
SAVED_STEP = int(sys.argv[2])
TASK_ID = int(sys.argv[3]) if len(sys.argv) > 3 else 1

device = torch.device('cpu')
model, config = load_model(MODEL_DIR, SAVED_STEP, device)
ds_cfg = config['dataset']
height, width = ds_cfg['height'], ds_cfg['width']
crop = tuple(ds_cfg.get('crop', (0, 0, 0, 0)))
demo_crop = tuple(ds_cfg.get('demo_crop', (0, 0, 0, 0)))

demo_root = os.path.join(ds_cfg['root_dir'].replace('${EXPERT_DATA}', '/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset'),
                          ds_cfg.get('task_name', 'pick_place'), f"{ds_cfg.get('demo_name', 'human_rgb')}_{ds_cfg.get('task_name', 'pick_place')}")
agent_root = os.path.join(ds_cfg['root_dir'].replace('${EXPERT_DATA}', '/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset'),
                           ds_cfg.get('task_name', 'pick_place'), f"{ds_cfg.get('agent_name', 'ur5e')}_{ds_cfg.get('task_name', 'pick_place')}")

demo_file = sorted(glob.glob(os.path.join(demo_root, f'task_{TASK_ID:02d}', '*.pkl')))[0]
agent_file = sorted(glob.glob(os.path.join(agent_root, f'task_{TASK_ID:02d}', '*.pkl')))[0]

demo_traj, _ = load_traj(demo_file)
agent_traj, _ = load_traj(agent_file)
elements = [x for x in agent_traj]

demo_video = make_demo_context(demo_traj, ds_cfg.get('T_context', 10), height, width, demo_crop)
o1 = preprocess_frame(elements[0]['obs']['image'], crop, height, width)

waypoints = predict_waypoints(model, demo_video, o1, device)
print('config image_waypoints:', config.get('image_waypoints', False))
print('raw waypoints (u,v,z,grasp) or (dx,dy,dz,grasp):\n', np.round(waypoints, 4))

if config.get('image_waypoints', False):
    projection = build_sample_projection(crop, NO_AUGMENTATION_STATS, (height, width))
    abs_positions = project_normalized_depth_to_world(waypoints[:, :3], projection)
else:
    abs_positions = waypoints[:, :3] + elements[0]['obs']['ee_aa'][:3][None]

print('\nprojected absolute world positions:\n', np.round(abs_positions, 4))

real_poses = np.stack([e['obs']['ee_aa'][:3] for e in elements])
print('\nreal trajectory ee_aa[:3] range (this exact demo):')
print(' min:', real_poses.min(axis=0), 'max:', real_poses.max(axis=0))
print(' start pos (elements[0]):', elements[0]['obs']['ee_aa'][:3])
