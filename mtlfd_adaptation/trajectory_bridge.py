"""
Bridges osvi-awda's dataset code to the trajectory pickle files produced by the
Multi-Task-LFD-Training-Framework (human_rgb_pick_place / ur5e_pick_place).

Those pkl files are saved with `multi_task_il.datasets.savers.Trajectory`, which JPEG-compresses
`camera_front_image` (and `eye_in_hand_image`) on write and decompresses them lazily in `.get(t)`.
osvi-awda's own dataset code (hem/datasets/agent_dataset.py) expects `obs['image']` instead of
`obs['camera_front_image']`. This module exposes a `load_traj` with osvi-awda's expected contract
(`traj.get(t)`, `traj.setting_name`, `traj.fname`, `len(traj)`), backed by the target framework's
own decompression logic, so we don't have to reimplement JPEG (de)compression here.
"""
import os
import sys
import pickle as pkl

_TRAINING_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'Multi-Task-LFD-Training-Framework', 'training'))
if _TRAINING_ROOT not in sys.path:
    sys.path.insert(0, _TRAINING_ROOT)


class AliasedTrajectory:
    """Wraps a multi_task_il Trajectory so `.get(t)['obs']['image']` is available (aliased from
    `camera_front_image`), matching what osvi-awda's dataset code expects."""

    def __init__(self, traj):
        self._traj = traj

    def get(self, t, decompress=True):
        step = self._traj.get(t, decompress=decompress)
        obs = step['obs']
        if 'image' not in obs and 'camera_front_image' in obs:
            obs['image'] = obs['camera_front_image']
        return step

    def __getitem__(self, t):
        return self.get(t)

    def __len__(self):
        return len(self._traj)

    def __iter__(self):
        for t in range(len(self)):
            yield self.get(t)

    def get_raw_state(self, t):
        """Passthrough to the underlying Trajectory's raw mujoco state snapshot (`sim.get_state()
        .flatten()` at collection time, when the collector passed one to `Trajectory.append`) -
        NOT part of `.get(t)`'s obs dict, stored separately. Used by
        test_mtlfd_rollout.py's set_objects_from_training_trajectory to replicate a training
        scenario's exact object layout in a live rollout."""
        return self._traj.get_raw_state(t)


def load_traj(fname):
    """Returns (traj, command) with traj.get(t)['obs']['image'] available, and traj.setting_name /
    traj.fname set, mirroring hem.datasets.load_traj's contract (used by osvi-awda's model code for
    e.g. logging / camera-projection lookups)."""
    with open(fname, 'rb') as f:
        sample = pkl.load(f)
    raw_traj = sample['traj']
    command = sample.get('command', None)
    traj = AliasedTrajectory(raw_traj)
    # .../pick_place/{agent_or_demo_name}_pick_place/task_NN/trajXXX.pkl -> setting = task_NN
    traj.setting_name = fname.split(os.sep)[-2]
    traj.fname = fname
    return traj, command
