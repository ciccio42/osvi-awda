"""
Thin training entrypoint that reuses osvi-awda's own InverseImitation model, Trainer, and
forward()/compute_loss_trajectory() (scripts/train_transformer.py) UNCHANGED, only swapping in
MTLFDAgentTeacherDataset (registered as dataset type "mtlfd agent teacher" in
hem/datasets/__init__.py) and adding periodic debug-image dumps. See docs/02_training_adaptation.md.

Usage (mirrors scripts/train_transformer.py):
    EXPERT_DATA=/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset \
    python -u mtlfd_adaptation/train_mtlfd.py mtlfd_adaptation/experiments/pick_place_mtlfd.yaml \
        --save-parent <checkpoint_dir>
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from hem.models.inverse_module import InverseImitation  # noqa: E402
from hem.models import Trainer  # noqa: E402
from pyutil import set_seed, experiment_log  # noqa: E402
from scripts.train_transformer import forward as osvi_forward  # noqa: E402
from mtlfd_adaptation import debug_utils  # noqa: E402
from mtlfd_adaptation.mtlfd_dataset import MTLFDAgentTeacherDataset  # noqa: E402,F401




def dump_preprocessing_debug_images(trainer, n=4):
    debug_dir = os.path.join(trainer.save_dir, 'debug_images', 'preprocessing')
    context, traj = next(iter(trainer._train_loader))
    for i in range(min(n, context['video'].shape[0])):
        debug_utils.save_batch_debug_images(debug_dir, context, traj, index=i, tag=f'train_{i}')
    context, traj = next(iter(trainer._val_loader))
    for i in range(min(n, context['video'].shape[0])):
        debug_utils.save_batch_debug_images(debug_dir, context, traj, index=i, tag=f'val_{i}')
    print(f'[mtlfd] wrote preprocessing debug images to {debug_dir}', flush=True)


def make_debug_forward(base_forward, save_dir, img_log_freq):
    """Wraps train_transformer.forward to periodically dump predicted-vs-GT waypoint overlays for
    a validation batch item, on top of the unmodified loss computation (loss/backprop are
    untouched - this only adds a read-only extra forward pass for visualization)."""
    step_counter = {'n': 0}

    def wrapped(config, m, device, context, traj, append=True, val=False):
        loss, stats = base_forward(config, m, device, context, traj, append=append, val=val)
        if val:
            step = step_counter['n']
            if step % img_log_freq == 0:
                try:
                    with torch.no_grad():
                        out = m(traj['states'], traj['images'], context['video'], ret_dist=False,
                                 ents=traj['head_label'])
                    pred = out['waypoints'][0].detach().cpu().numpy()
                    gt = traj['traj_points'][0].detach().cpu().numpy()
                    o1 = traj['images'][0, 0].detach().cpu().numpy()
                    proj = traj['projection_matrix'][0].detach().cpu().numpy()
                    out_path = os.path.join(save_dir, 'debug_images', 'training',
                                             f'step_{step:07d}', 'waypoints.png')
                    debug_utils.save_waypoint_overlay(out_path, o1, pred, gt, proj)
                except Exception as e:  # debug visualization must never break training
                    print(f'[mtlfd] debug overlay failed at step {step}: {e}', flush=True)
            step_counter['n'] += 1
        return loss, stats
    return wrapped


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='OSVI-AWDA training adapted to Multi-Task-LFD-Training-Framework pick_place data')
    parser.add_argument('experiment_file', type=str, help='path to YAML experiment config file')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--save-parent', type=str, default='')
    parser.add_argument('--device', type=int, default=None, nargs='+')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--reg-log', action='store_true')
    parser.add_argument('--nodeterm', action='store_true')
    parser.add_argument('--debug-preprocessing-only', action='store_true',
                         help='dump preprocessing debug images and exit, without training')
    parser.add_argument('--debug', action='store_true',
                         help='listen for a debugpy client and block until one attaches, '
                              'before doing anything else')
    parser.add_argument('--debug-port', type=int, default=5678)
    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', args.debug_port))
        print(f'[mtlfd] debugpy listening on port {args.debug_port}, waiting for client to '
              f'attach...', flush=True)
        debugpy.wait_for_client()

    trainer = Trainer(args, 'osvi_mtlfd',
                       "OSVI-AWDA trained on Multi-Task-LFD-Training-Framework pick_place data")
    config = trainer.config
    set_seed(trainer.seed, determ=config.get('determ', True) and trainer.determ)
    experiment_log(trainer.save_dir)

    dump_preprocessing_debug_images(trainer)
    if args.debug_preprocessing_only:
        sys.exit(0)

    action_model = InverseImitation(**config['policy'])
    if trainer.resume is not None:
        # weights_only=False: see the matching comment in test_mtlfd_rollout.py::load_model.
        action_model.load_state_dict(
            torch.load(trainer.resume, map_location=torch.device('cpu'),
                       weights_only=False).state_dict())

    debug_forward = make_debug_forward(osvi_forward, trainer.save_dir,
                                        trainer._config.get('img_log_freq', 500))
    trainer.train(action_model, debug_forward)
