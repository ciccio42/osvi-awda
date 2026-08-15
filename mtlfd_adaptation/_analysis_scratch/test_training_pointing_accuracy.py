"""
Diagnostic (complements test_demo_sensitivity.py): run the model's own waypoint/path-generation
forward pass directly on real TRAINING pairs (human-demo, robot-trajectory) - the exact same
MTLFDAgentTeacherDataset(mode='train') sampling train_mtlfd.py itself uses, augmentation included -
and check how often the predicted grasp waypoint lands nearest to the WRONG object instead of the
trajectory's actual target, using each trajectory's own stored ground truth (obj_bb + target-object
fields, written at data-collection time - see new_pp.py's object_to_id for the object_set=2 name
order this assumes: greenbox=0, yellowbox=1, bluebox=2, redbox=3).

test_demo_sensitivity.py already ruled out the human-demo as the driver of a bad LIVE-rollout plan
(near-zero variance across demos). This test asks the complementary question: does the mistargeting
show up even on data the model was actually TRAINED on (not live sim, not held-out)? If yes, that
points at a fundamental model/label problem rather than train/test distribution shift; if the model
is accurate here but wrong live, the live sim's rendering/domain is the more likely culprit.

Does NOT need a live robosuite env at all (no get_env/mujoco rendering, no GPU offscreen context) -
ground truth comes directly from each trajectory pkl's own stored fields, and only the policy
network itself needs the GPU. The projection chain (predicted normalized-image coords -> world ->
native pixel) uses each SAMPLE's own real per-sample projection_matrix (which already accounts for
whatever random flip/crop/translate augmentation was actually applied to that sample's images), so
this is correct regardless of augmentation - no need to disable it to get a fair comparison.

Run via mtlfd_adaptation/_analysis_scratch/run_training_pointing_accuracy.sh.

Usage:
    python -u mtlfd_adaptation/_analysis_scratch/test_training_pointing_accuracy.py <model_dir> \
        --saved_step 430000 --n_samples 200 --out_dir <dir>
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)   # append, not insert(0, ...) - see test_demo_sensitivity.py's note

from hem.models.inverse_module import InverseImitation  # noqa: E402
from mtlfd_adaptation.camera_projection import (  # noqa: E402
    BASE_PROJECTION, CANVAS_SIZE, project_normalized_depth_to_world, project_world_to_pixel,
)
from mtlfd_adaptation.mtlfd_dataset import AGENT_SETTLE_STEPS, MTLFDAgentTeacherDataset  # noqa: E402
from mtlfd_adaptation.trajectory_bridge import load_traj  # noqa: E402

GRASP_THRESHOLD = 0.1
# object_set=2's fixed name<->index order (new_pp.py object_to_id) - what `target-object` (an int)
# refers into, and the keys obj_bb's per-camera dict uses (besides 'bin', which isn't a candidate).
OBJECT_NAMES = ['greenbox', 'yellowbox', 'bluebox', 'redbox']
N_DEBUG_IMAGES = 12


def load_model(model_dir, saved_step, device):
    with open(os.path.join(model_dir, 'config.yaml')) as f:
        config = yaml.safe_load(f)
    model = InverseImitation(**config['policy'])
    ckpt_path = os.path.join(model_dir, f'model_save-{saved_step}.pt')
    loaded = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = loaded.state_dict() if hasattr(loaded, 'state_dict') else loaded
    model.load_state_dict(state_dict)
    return model.to(device).eval(), config


def save_pointing_debug_image(out_path, native_rgb, obj_bb, pred_row, pred_col, target_name,
                               nearest_name, correct):
    img = Image.fromarray(native_rgb).convert('RGB')
    draw = ImageDraw.Draw(img)
    for name, bb in obj_bb.items():
        if name == 'bin':
            continue
        col, row = bb['center']
        color = (0, 200, 0) if name == target_name else (120, 120, 120)
        r = 6
        draw.ellipse([col - r, row - r, col + r, row + r], outline=color, width=2)
        draw.text((col + r + 2, row - 6), name, fill=color)
    if np.isfinite(pred_row) and np.isfinite(pred_col):
        mark_color = (0, 160, 255) if correct else (255, 0, 0)
        r = 5
        draw.line([pred_col - r, pred_row - r, pred_col + r, pred_row + r], fill=mark_color, width=2)
        draw.line([pred_col - r, pred_row + r, pred_col + r, pred_row - r], fill=mark_color, width=2)
    draw.text((2, 2), f'target={target_name} nearest={nearest_name} correct={correct}',
               fill=(255, 255, 0))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('--saved_step', type=int, required=True)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    model, config = load_model(args.model_dir, args.saved_step, device)
    ds_cfg = dict(config['dataset'])
    ds_cfg.pop('type', None)
    assert config.get('image_waypoints', False), (
        'this diagnostic assumes image_waypoints=True (needs the normalized-image<->world '
        'projection that branch trains under)')

    dataset = MTLFDAgentTeacherDataset(**ds_cfg, mode='train')
    print(f'[training-pointing] train dataset: {len(dataset)} (agent,demo) pairs across '
          f'train_tasks={dataset.train_tasks}', flush=True)

    n = min(args.n_samples, len(dataset))
    indices = random.sample(range(len(dataset)), n)

    out_dir = (args.out_dir or os.path.join(
        args.model_dir, f'training_pointing_accuracy/step-{args.saved_step}'))
    debug_dir = os.path.join(out_dir, 'debug_images')
    os.makedirs(debug_dir, exist_ok=True)

    results = []
    n_correct = 0
    n_scored = 0
    n_debug_saved = 0
    for idx in indices:
        a_i, d_i = dataset.pairs[idx % len(dataset.pairs)]
        agent_file = dataset.agent_files[a_i]
        demo_file = dataset.demo_files[d_i]

        context, traj = dataset[idx]

        with torch.no_grad():
            states = torch.from_numpy(traj['states'])[None].to(device)
            images = torch.from_numpy(traj['images'])[None].to(device)
            video = torch.from_numpy(context['video'])[None].to(device)
            ents = torch.zeros((1,), dtype=torch.long, device=device)
            out = model(states, images, video, ret_dist=False, ents=ents)
        waypoints = out['waypoints'][0].cpu().numpy()[-5:]   # last 5 = the "5-waypoint" trajectory

        # ground truth, read directly off the raw (unaugmented) agent trajectory - independent of
        # whatever random flip/crop/translate this particular sample got, since it's compared
        # against the prediction only after projecting the prediction back to WORLD coords (via
        # this sample's own real projection_matrix, which already accounts for that augmentation)
        # and then to NATIVE pixel space (via the fixed, augmentation-independent BASE_PROJECTION) -
        # i.e. both sides of the comparison end up expressed in the same raw-trajectory pixel frame.
        agent_traj, _ = load_traj(agent_file)
        obs_settle = agent_traj.get(AGENT_SETTLE_STEPS)['obs']
        obj_bb = obs_settle['obj_bb']['camera_front']
        target_idx = int(obs_settle['target-object'])
        target_name = OBJECT_NAMES[target_idx] if target_idx < len(OBJECT_NAMES) else str(target_idx)
        native_rgb = obs_settle['image']

        grasp_on = waypoints[:, 3] > GRASP_THRESHOLD
        grasp_idx = np.where(grasp_on)[0]
        if len(grasp_idx) == 0 or target_name not in obj_bb:
            results.append({'agent_file': agent_file, 'demo_file': demo_file,
                             'skipped': 'no grasp-on waypoint predicted' if len(grasp_idx) == 0
                                        else 'target object missing from obj_bb'})
            continue
        pred_uvz = waypoints[grasp_idx[0], :3]

        world_xyz = project_normalized_depth_to_world(pred_uvz[None], traj['projection_matrix'])[0]
        pred_row, pred_col = project_world_to_pixel(world_xyz, BASE_PROJECTION, CANVAS_SIZE)

        if not np.isfinite(pred_row):
            results.append({'agent_file': agent_file, 'demo_file': demo_file,
                             'skipped': 'predicted grasp point projects behind the camera'})
            continue

        dists = {}
        for name, bb in obj_bb.items():
            if name == 'bin':
                continue
            col, row = bb['center']
            dists[name] = float(np.hypot(pred_row - row, pred_col - col))
        nearest_name = min(dists, key=dists.get)
        correct = nearest_name == target_name
        n_scored += 1
        n_correct += int(correct)

        results.append({
            'agent_file': agent_file, 'demo_file': demo_file,
            'target_object': target_name, 'nearest_object': nearest_name, 'correct': correct,
            'pred_pixel_row_col': [float(pred_row), float(pred_col)],
            'distances_px': dists,
        })

        if n_debug_saved < N_DEBUG_IMAGES:
            save_pointing_debug_image(
                os.path.join(debug_dir, f'{n_debug_saved:02d}_{"ok" if correct else "WRONG"}.png'),
                native_rgb, obj_bb, pred_row, pred_col, target_name, nearest_name, correct)
            n_debug_saved += 1

    accuracy = n_correct / n_scored if n_scored else None
    confusion = Counter((r['target_object'], r['nearest_object']) for r in results if 'correct' in r)

    summary = {
        'model_dir': args.model_dir,
        'saved_step': args.saved_step,
        'n_requested': args.n_samples,
        'n_sampled': n,
        'n_scored': n_scored,
        'n_skipped': n - n_scored,
        'n_correct': n_correct,
        'pointing_accuracy': accuracy,
        'confusion_target_to_nearest': {f'{t}->{p}': c for (t, p), c in confusion.items()},
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump({'summary': summary, 'per_sample': results}, f, indent=2)

    print(f'[training-pointing] scored {n_scored}/{n} samples ({n - n_scored} skipped: '
          f'no grasp waypoint / missing bbox / behind camera)')
    if accuracy is not None:
        print(f'[training-pointing] pointing accuracy: {n_correct}/{n_scored} = {accuracy*100:.1f}%')
    else:
        print('[training-pointing] no scoreable samples')
    print('[training-pointing] confusion (target -> predicted nearest):')
    for (t, p), c in sorted(confusion.items()):
        print(f'    {t:>10} -> {p:<10}: {c}')
    print(f'[training-pointing] wrote {out_dir}/summary.json, {n_debug_saved} debug images')


if __name__ == '__main__':
    main()
