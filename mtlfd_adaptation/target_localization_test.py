"""
Grasp-waypoint pointing-accuracy test, run under LIVE robosuite rendering (unlike
_analysis_scratch/test_training_pointing_accuracy.py, which scores the model on the literal stored
training pixels with no simulator at all) - this exists to isolate whether re-rendering a scene
live, even with an identical object layout, shifts the model's predictions (the "rendering
distribution shift" hypothesis discussed alongside that offline test's results).

Per split:
  - train: env objects teleported to match a real TRAIN-task trajectory's exact recorded layout
    (mtlfd_dataset.MTLFDAgentTeacherDataset(mode='train'), i.e. train_mtlfd.py's own `dataset`).
  - valid: same, but sampling from mode='test' pairs - train_mtlfd.py's Trainer literally builds
    its `val_dataset` with mode='test' (hem/models/trainer.py), so this IS what the codebase calls
    validation-during-training, even though (for this checkpoint) it's the same task set the
    standalone `test` split below also uses (test_tasks is this experiment's only held-out split).
  - test: NO spawn matching - the env's own novel/random reset, seeded from a fixed seeds.txt
    (mtlfd_adaptation/seeds/seeds.txt) for reproducibility, over the checkpoint's own test_tasks.

Does NOT execute any rollout primitives (move_to/grasp/drop) - only env.reset() (+ teleport for
train/valid) and one predict_waypoints() forward pass per sample, then checks whether waypoint
index 1 (the grasp waypoint - the "5-waypoint trajectory" plan's 2nd point) lands nearest the
correct object's live obj_bb, exactly as test_training_pointing_accuracy.py checks against stored
obj_bb (see docs/03's waypoint-index convention this assumes: image_waypoints=True).

Usage:
    python -u mtlfd_adaptation/target_localization_test.py <model_dir> --saved_step 430000 \
        --splits train valid test --episodes_per_task 10
"""
import argparse
import glob
import json
import os
import random
import sys
from collections import Counter

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from robosuite import load_controller_config  # noqa: E402
from robosuite.utils import RandomizationError  # noqa: E402
from multi_task_robosuite_env import get_env  # noqa: E402
from mtlfd_adaptation import debug_utils  # noqa: E402
from mtlfd_adaptation.camera_projection import (  # noqa: E402
    BASE_PROJECTION, CANVAS_SIZE, NO_AUGMENTATION_STATS, build_sample_projection,
    project_normalized_depth_to_world, project_world_to_pixel,
)
from mtlfd_adaptation.mtlfd_dataset import MTLFDAgentTeacherDataset  # noqa: E402
from mtlfd_adaptation.test_mtlfd_rollout import (  # noqa: E402
    CONTROLLER_PATH, _task_id_from_path, draw_waypoints_on_frame, load_model, make_demo_context,
    predict_waypoints, preprocess_frame, seed_everything, set_objects_from_training_trajectory,
    stabilize,
)
from mtlfd_adaptation.trajectory_bridge import load_traj  # noqa: E402

# object_set=2's fixed name<->index order (new_pp.py object_to_id) - what live obs's `target-object`
# (an int) refers into, and the keys obj_bb's per-camera dict uses (besides 'bin').
OBJECT_NAMES = ['greenbox', 'yellowbox', 'bluebox', 'redbox']
DEFAULT_SEEDS_FILE = os.path.join(REPO_ROOT, 'mtlfd_adaptation', 'seeds', 'seeds.txt')


def build_pairs_by_task(ds_cfg, mode):
    """(demo_file, agent_file) pairs genuinely belonging to MTLFDAgentTeacherDataset(mode=mode),
    grouped by task_id - same construction test_mtlfd_rollout.py's --training_pairs uses, just
    mode-parametrized (that helper is hardcoded to mode='train')."""
    dataset = MTLFDAgentTeacherDataset(mode=mode, **ds_cfg)
    pairs_by_task = {}
    for a_i, d_i in dataset.pairs:
        task_id = _task_id_from_path(dataset.agent_files[a_i])
        pairs_by_task.setdefault(task_id, []).append(
            (dataset.demo_files[d_i], dataset.agent_files[a_i]))
    return pairs_by_task, dataset


def run_sample(env, model, config, demo_file, height, width, crop, device, sample_dir,
               training_agent_file, seed):
    """One forward pass, no rollout execution. training_agent_file=None -> novel/random reset
    (test split); given -> teleport to match that trajectory's layout (train/valid splits)."""
    demo_traj, _ = load_traj(demo_file)
    ds_cfg = config['dataset']
    demo_video = make_demo_context(demo_traj, ds_cfg.get('T_context', 10), height, width,
                                    tuple(ds_cfg.get('demo_crop', (0, 0, 0, 0))))

    tries = 0
    while True:
        try:
            if seed is not None:
                np.random.seed(seed + tries)
            env.reset()
            break
        except RandomizationError:
            tries += 1
    if training_agent_file is not None:
        set_objects_from_training_trajectory(env, training_agent_file)
    obs = stabilize(env)

    native_rgb = obs['camera_front_image']   # native (200,360) RGB - same pixel space as obj_bb
    o1 = preprocess_frame(native_rgb, crop, height, width)

    waypoints = predict_waypoints(model, demo_video, o1, device)   # [5,4] (u,v,depth,grasp)
    projection = build_sample_projection(crop, NO_AUGMENTATION_STATS, (height, width))
    abs_positions = project_normalized_depth_to_world(waypoints[:, :3], projection)

    grasp_world = abs_positions[1]   # waypoint index 1 = the grasp waypoint, shape (3,)
    pred_row, pred_col = project_world_to_pixel(grasp_world, BASE_PROJECTION, CANVAS_SIZE)

    obj_bb = obs['obj_bb']['camera_front']
    target_idx = int(obs['target-object'])
    target_name = OBJECT_NAMES[target_idx] if target_idx < len(OBJECT_NAMES) else str(target_idx)

    result = {
        'demo_file': demo_file,
        'training_agent_file': training_agent_file,
        'seed': int(seed) if seed is not None else None,
    }
    if not np.isfinite(pred_row) or target_name not in obj_bb:
        result['skipped'] = ('predicted grasp point projects behind camera' if not np.isfinite(pred_row)
                              else 'target object missing from obj_bb')
    else:
        dists = {}
        for name, bb in obj_bb.items():
            if name == 'bin':
                continue
            col, row = bb['center']
            dists[name] = float(np.hypot(pred_row - row, pred_col - col))
        nearest_name = min(dists, key=dists.get)
        correct = nearest_name == target_name
        result.update({'target_object': target_name, 'nearest_object': nearest_name,
                        'correct': correct, 'pred_pixel_row_col': [float(pred_row), float(pred_col)],
                        'distances_px': dists})

    os.makedirs(sample_dir, exist_ok=True)
    debug_utils.save_frame_grid(demo_video, os.path.join(sample_dir, 'demo_context.png'))
    Image.fromarray(native_rgb).save(os.path.join(sample_dir, 'original_img.png'))
    plan_img_bgr = draw_waypoints_on_frame(debug_utils.unnormalize(o1, normalized=True),
                                            abs_positions, waypoints[:, 3], crop, height, width)
    Image.fromarray(cv2.cvtColor(plan_img_bgr, cv2.COLOR_BGR2RGB)).save(
        os.path.join(sample_dir, 'predicted_plan.png'))
    with open(os.path.join(sample_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    return result


def summarize(results):
    scored = [r for r in results if 'correct' in r]
    n_correct = sum(r['correct'] for r in scored)
    confusion = Counter((r['target_object'], r['nearest_object']) for r in scored)
    return {
        'n_total': len(results),
        'n_scored': len(scored),
        'n_skipped': len(results) - len(scored),
        'n_correct': n_correct,
        'pointing_accuracy': n_correct / len(scored) if scored else None,
        'confusion_target_to_nearest': {f'{t}->{p}': c for (t, p), c in confusion.items()},
    }


def run_split(split, env_builder, model, config, ds_cfg, height, width, crop, device, out_root,
              episodes_per_task, seed, seeds_file):
    split_dir = os.path.join(out_root, split)
    os.makedirs(split_dir, exist_ok=True)

    all_results = []
    if split in ('train', 'valid'):
        mode = 'train' if split == 'train' else 'test'
        pairs_by_task, dataset = build_pairs_by_task(ds_cfg, mode)
        print(f'[target-loc] {split}: mode={mode!r} task_ids={sorted(pairs_by_task)} '
              f'({len(dataset)} total pairs)', flush=True)
        random.seed(seed)
        ep_counter = 0
        for task_id in sorted(pairs_by_task):
            task_pairs = pairs_by_task[task_id]
            env = env_builder(task_id)
            for ep in range(episodes_per_task):
                demo_file, agent_file = task_pairs[random.randrange(len(task_pairs))]
                sample_dir = os.path.join(split_dir, f'task{task_id:02d}_ep{ep:02d}')
                r = run_sample(env, model, config, demo_file, height, width, crop, device,
                                sample_dir, training_agent_file=agent_file,
                                seed=seed + ep_counter)
                r.update(task_id=task_id, episode=ep)
                all_results.append(r)
                print(f'[target-loc] {split} task {task_id} ep {ep}: {r}', flush=True)
                ep_counter += 1
            env.close()
    else:   # test: novel/seeded init, no spawn matching
        ds_cfg_test = dict(ds_cfg)
        test_tasks = sorted(MTLFDAgentTeacherDataset(mode='test', **ds_cfg_test).test_tasks)
        demo_root = os.path.join(ds_cfg['root_dir'], ds_cfg.get('task_name', 'pick_place'),
                                  f"{ds_cfg.get('demo_name', 'human_rgb')}_"
                                  f"{ds_cfg.get('task_name', 'pick_place')}")
        with open(seeds_file) as f:
            episode_seeds = [int(line) for line in f if line.strip()]
        n_needed = len(test_tasks) * episodes_per_task
        assert len(episode_seeds) >= n_needed, (
            f'{seeds_file} has {len(episode_seeds)} seeds, need >= {n_needed}')
        print(f'[target-loc] test: task_ids={test_tasks}, using first {n_needed} seeds from '
              f'{seeds_file}', flush=True)
        ep_counter = 0
        for task_id in test_tasks:
            demo_files = sorted(glob.glob(os.path.join(demo_root, f'task_{task_id:02d}', '*.pkl')))
            assert demo_files, f'no demo files for task_id {task_id} in {demo_root}'
            env = env_builder(task_id)
            for ep in range(episodes_per_task):
                ep_seed = episode_seeds[ep_counter]
                np.random.seed(ep_seed)
                demo_file = demo_files[np.random.randint(len(demo_files))]
                sample_dir = os.path.join(split_dir, f'task{task_id:02d}_ep{ep:02d}')
                r = run_sample(env, model, config, demo_file, height, width, crop, device,
                                sample_dir, training_agent_file=None, seed=ep_seed)
                r.update(task_id=task_id, episode=ep)
                all_results.append(r)
                print(f'[target-loc] test task {task_id} ep {ep}: {r}', flush=True)
                ep_counter += 1
            env.close()

    summary = summarize(all_results)
    with open(os.path.join(split_dir, 'summary.json'), 'w') as f:
        json.dump({'summary': summary, 'episodes': all_results}, f, indent=2)
    print(f'[target-loc] {split} SUMMARY: {summary}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('--saved_step', type=int, required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'],
                         choices=['train', 'valid', 'test'])
    parser.add_argument('--episodes_per_task', type=int, default=10)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0,
                         help='same default as train_mtlfd.py --seed (this checkpoint was trained '
                              'with the default, i.e. 0 - see args.txt)')
    parser.add_argument('--seeds_file', type=str, default=DEFAULT_SEEDS_FILE)
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    model, config = load_model(args.model_dir, args.saved_step, device)
    ds_cfg = dict(config['dataset'])
    ds_cfg.pop('type', None)
    assert config.get('image_waypoints', False), (
        'this test assumes image_waypoints=True (needs the normalized-image<->world projection '
        'that branch trains under)')
    height, width = ds_cfg.get('height', 100), ds_cfg.get('width', 180)
    crop = tuple(ds_cfg.get('crop', (0, 0, 0, 0)))

    out_root = args.out_dir or os.path.join(args.model_dir, 'target_localization_test')
    os.makedirs(out_root, exist_ok=True)

    controller_config = load_controller_config(custom_fpath=CONTROLLER_PATH)
    action_ranges = np.array([[-0.05, 0.25], [-0.45, 0.5], [0.82, 1.2], [-5, 5], [-5, 5], [-5, 5]])

    def env_builder(task_id):
        while True:
            try:
                return get_env('UR5e_PickPlaceDistractor', controller_configs=controller_config,
                                task_id=task_id, has_renderer=False, has_offscreen_renderer=True,
                                reward_shaping=False, use_camera_obs=True, ranges=action_ranges,
                                render_gpu_device_id=args.gpu_id, render_camera='camera_front',
                                object_set=2)
            except RandomizationError:
                continue

    all_summaries = {}
    for split in args.splits:
        all_summaries[split] = run_split(
            split, env_builder, model, config, ds_cfg, height, width, crop, device, out_root,
            args.episodes_per_task, args.seed, args.seeds_file)

    print('[target-loc] ALL SPLITS SUMMARY:', json.dumps(all_summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
