"""
Real-data counterpart to target_localization_test.py: same pointing-accuracy metric (does the
predicted grasp waypoint - waypoint index 1 of the 5-waypoint plan - land nearest the correct
target object's obj_bb, out of all objects+bins) and the same output layout (per-split summary.json
+ per-episode task%02d_ep%02d/{demo_context,original_img,predicted_plan}.png + result.json), run
against real_eye_in_hand_ur5e_pick_place trajectories instead of a live robosuite sim.

Differs from target_localization_test.py in two structural ways forced by real data:
  - No live env at all (no get_env/env.reset/set_objects_from_training_trajectory/stabilize) - a
    recorded real trajectory's own obs at AGENT_SETTLE_STEPS (post-settle, matching
    mtlfd_dataset.py's own AGENT_SETTLE_STEPS convention) stands in for "teleport env objects to
    match this trajectory's layout, then re-observe": there's no live re-renderer for the real
    world, so the recorded frame/obj_bb simply ARE the observation.
  - No 'test' split: that split's whole point in the original script is a NOVEL/randomly-reset
    scene (env.reset() with no teleport, seeded from seeds.txt) - there's no real-data equivalent
    of "generate a new random real scene," so only 'train' (train_tasks) and 'valid' (test_tasks,
    i.e. the finetune's held-out tasks 0/5/10/15) are supported here.
  - target_object has no ground-truth label on real data (no obs['target-object'] field, unlike
    sim). Inferred instead as the obj_bb object closest to the trajectory's own real eef_pos at its
    gripper-close frame (action[-1] > 0.01 - same grasp-mining convention mtlfd_dataset.py's
    real-data branch and eval_zero_shot_real.py use) - result.json marks this via
    'target_object_source': 'inferred_from_grasp_frame_nearest_bbox' so it's never mistaken for a
    ground-truth label when comparing against the sim checkpoint's results.

Usage:
    python -u mtlfd_adaptation/target_localization_test_real.py <model_dir> --saved_step 70000 \
        --splits train valid --episodes_per_task 10
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hem.models.inverse_module import InverseImitation  # noqa: E402
from hem.datasets.util import crop as crop_fn  # noqa: E402
from hem.datasets.util import randomize_video, resize  # noqa: E402
from mtlfd_adaptation import debug_utils  # noqa: E402
from mtlfd_adaptation.camera_projection_real import (  # noqa: E402
    BASE_PROJECTION, CANVAS_SIZE, NO_AUGMENTATION_STATS, build_sample_projection,
    project_normalized_depth_to_world, project_world_to_final_pixel, project_world_to_pixel,
)
from mtlfd_adaptation.mtlfd_dataset import AGENT_SETTLE_STEPS, MTLFDAgentTeacherDataset  # noqa: E402
from mtlfd_adaptation.trajectory_bridge import load_traj  # noqa: E402

OBJECT_NAMES = ['greenbox', 'yellowbox', 'bluebox', 'redbox']
GRASP_THRESHOLD = 0.1


def _task_id_from_path(path):
    m = re.search(r'task_(\d+)', path)
    assert m, f'could not find task_NN in path {path!r}'
    return int(m.group(1))


def preprocess_frame(img, crop, height, width):
    """Deterministic (no augmentation) crop+resize - mirrors test_mtlfd_rollout.py's version."""
    img = crop_fn(img, crop)
    img = resize(img, (width, height))
    frames, _ = randomize_video(img[None], None, None, None, 0, np.array([0, 0]), True,
                                 rand_flip=False)
    return np.transpose(frames, (0, 3, 1, 2)).astype(np.float32)[0]


def make_demo_context(traj, T_context, height, width, crop, sample_sides=True):
    """Mirrors test_mtlfd_rollout.py's version."""
    clip = lambda x: int(max(0, min(x, len(traj) - 1)))
    per_bracket = max(len(traj) / T_context, 1)
    frames = []
    for i in range(T_context):
        n = clip(int((i + 0.5) * per_bracket))
        if sample_sides and i == T_context - 1:
            n = len(traj) - 1
        elif sample_sides and i == 0:
            n = 0
        frames.append(preprocess_frame(traj.get(n)['obs']['image'], crop, height, width))
    return np.stack(frames, axis=0)


def load_model(model_dir, saved_step, device):
    with open(os.path.join(model_dir, 'config.yaml')) as f:
        config = yaml.safe_load(f)
    model = InverseImitation(**config['policy'])
    ckpt_path = os.path.join(model_dir, f'model_save-{saved_step}.pt')
    loaded = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = loaded.state_dict() if hasattr(loaded, 'state_dict') else loaded
    model.load_state_dict(state_dict)
    return model.to(device).eval(), config


def predict_waypoints(model, demo_video, o1_img, device):
    context = torch.from_numpy(demo_video)[None].to(device)
    images = torch.from_numpy(np.stack([o1_img, o1_img]))[None].to(device)
    states = torch.zeros((1, 2, 1), device=device)
    ents = torch.zeros((1,), dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(states, images, context, ret_dist=False, ents=ents)
    return out['waypoints'][0].cpu().numpy()[-5:]   # [5,4]


def draw_waypoints_on_frame(img, abs_positions, grasp_attrs, crop, height, width, indices=None):
    """Real-projection counterpart of test_mtlfd_rollout.py::draw_waypoints_on_frame (that version
    is hardcoded to the sim camera_projection module's project_world_to_final_pixel)."""
    if indices is None:
        indices = range(len(abs_positions))
    prev_px = None
    for idx in indices:
        proj = project_world_to_final_pixel(abs_positions[idx], crop, (height, width))
        if proj is None:
            continue
        row, col = proj
        px = (int(col), int(row))
        color = (0, 0, 255) if grasp_attrs[idx] > GRASP_THRESHOLD else (0, 255, 0)
        if 0 <= px[0] < img.shape[1] and 0 <= px[1] < img.shape[0]:
            cv2.circle(img, px, 3, color, -1)
            if prev_px is not None:
                cv2.line(img, prev_px, px, (255, 0, 0), 1)
            prev_px = px
    return img


def infer_target_object(agent_traj):
    """No sim-only target-object label exists on real data - infer the picked object as the
    obj_bb object nearest the real eef position (projected to raw canvas pixels via the full,
    uncropped BASE_PROJECTION) at the trajectory's gripper-close frame."""
    elements = [agent_traj.get(t) for t in range(len(agent_traj))]
    n = len(elements)
    grasp_frames = [elements[i + 1]['action'][-1] > 0.01 for i in range(n - 1)] + [False]
    grasp_ind = next((i for i, g in enumerate(grasp_frames) if g), n - 1)
    obs = elements[grasp_ind]['obs']
    row, col = project_world_to_pixel(obs['eef_pos'][None], BASE_PROJECTION, CANVAS_SIZE)
    bb = obs['obj_bb']['camera_front']
    dists = {name: float(np.hypot(row[0] - bb[name]['center'][1], col[0] - bb[name]['center'][0]))
             for name in OBJECT_NAMES if name in bb}
    return min(dists, key=dists.get)


def build_pairs_by_task(ds_cfg, mode):
    # ds_cfg (from this checkpoint's own config.yaml) already carries is_real: True - see
    # experiments/pick_place_mtlfd_real_eye_in_hand_ur5e.yaml.
    assert ds_cfg.get('is_real'), 'expected a real-data checkpoint config (dataset.is_real: True)'
    dataset = MTLFDAgentTeacherDataset(mode=mode, **ds_cfg)
    pairs_by_task = {}
    for a_i, d_i in dataset.pairs:
        task_id = _task_id_from_path(dataset.agent_files[a_i])
        pairs_by_task.setdefault(task_id, []).append(
            (dataset.demo_files[d_i], dataset.agent_files[a_i]))
    return pairs_by_task, dataset


def run_sample(model, config, demo_file, agent_file, height, width, crop, device, sample_dir,
               seed):
    demo_traj, _ = load_traj(demo_file)
    agent_traj, _ = load_traj(agent_file, is_real=True)
    ds_cfg = config['dataset']
    demo_video = make_demo_context(demo_traj, ds_cfg.get('T_context', 10), height, width,
                                    tuple(ds_cfg.get('demo_crop', (0, 0, 0, 0))))

    obs = agent_traj.get(AGENT_SETTLE_STEPS)['obs']
    native_rgb = obs['image']
    o1 = preprocess_frame(native_rgb, crop, height, width)

    waypoints = predict_waypoints(model, demo_video, o1, device)   # [5,4] (u,v,depth,grasp)
    projection = build_sample_projection(crop, NO_AUGMENTATION_STATS, (height, width))
    abs_positions = project_normalized_depth_to_world(waypoints[:, :3], projection)

    grasp_world = abs_positions[1]   # waypoint index 1 = the grasp waypoint
    pred_row, pred_col = project_world_to_pixel(grasp_world, BASE_PROJECTION, CANVAS_SIZE)

    obj_bb = obs['obj_bb']['camera_front']
    target_name = infer_target_object(agent_traj)

    result = {
        'demo_file': demo_file, 'training_agent_file': agent_file,
        'seed': int(seed) if seed is not None else None,
        'target_object_source': 'inferred_from_grasp_frame_nearest_bbox',
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


def run_split(split, model, config, ds_cfg, height, width, crop, device, out_root,
              episodes_per_task, seed):
    split_dir = os.path.join(out_root, split)
    os.makedirs(split_dir, exist_ok=True)
    mode = 'train' if split == 'train' else 'test'
    pairs_by_task, dataset = build_pairs_by_task(ds_cfg, mode)
    print(f'[target-loc-real] {split}: mode={mode!r} task_ids={sorted(pairs_by_task)} '
          f'({len(dataset)} total pairs)', flush=True)

    random.seed(seed)
    all_results = []
    ep_counter = 0
    for task_id in sorted(pairs_by_task):
        task_pairs = pairs_by_task[task_id]
        for ep in range(episodes_per_task):
            demo_file, agent_file = task_pairs[random.randrange(len(task_pairs))]
            sample_dir = os.path.join(split_dir, f'task{task_id:02d}_ep{ep:02d}')
            r = run_sample(model, config, demo_file, agent_file, height, width, crop, device,
                            sample_dir, seed=seed + ep_counter)
            r.update(task_id=task_id, episode=ep)
            all_results.append(r)
            print(f'[target-loc-real] {split} task {task_id} ep {ep}: {r}', flush=True)
            ep_counter += 1

    summary = summarize(all_results)
    with open(os.path.join(split_dir, 'summary.json'), 'w') as f:
        json.dump({'summary': summary, 'episodes': all_results}, f, indent=2)
    print(f'[target-loc-real] {split} SUMMARY: {summary}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir')
    parser.add_argument('--saved_step', type=int, required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'valid'],
                         choices=['train', 'valid'])
    parser.add_argument('--episodes_per_task', type=int, default=10)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

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

    all_summaries = {}
    for split in args.splits:
        all_summaries[split] = run_split(
            split, model, config, ds_cfg, height, width, crop, device, out_root,
            args.episodes_per_task, args.seed)

    print('[target-loc-real] ALL SPLITS SUMMARY:', json.dumps(all_summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
