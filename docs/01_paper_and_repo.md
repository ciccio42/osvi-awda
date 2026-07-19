# OSVI-AWDA: paper and repository overview

## 1. The paper

**"One-shot Visual Imitation via Attributed Waypoints and Demonstration Augmentation"**
Matthew Chang, Saurabh Gupta — University of Illinois Urbana-Champaign. arXiv:2302.04856.

### 1.1 Problem setting

*One-shot visual imitation*: given a single RGB video demonstration **v** of a task, and a single
image **o₁** of a *novel instance* of that task (different object locations, possibly a different
embodiment than the one in the video), the agent must execute the depicted task with no other
supervision. Formally, prior methods cast this as a conditional policy `π(aₜ | v, o₁:ₜ, s₁:ₜ)`
learned by behavior cloning on an offline dataset of (video, robot trajectory) pairs for other
tasks.

### 1.2 Three failure modes of prior methods (diagnosed via T-OSVI, a representative transformer
baseline)

1. **The DAgger problem** — purely offline behavior cloning suffers from compounding errors at
   test time (distribution shift), especially because the baseline conditions on a sliding window
   of its own recent execution frames, so an early mistake compounds.
2. **Last-centimeter errors** — grasping is a fine-motor-control problem that behavior cloning
   struggles to learn from a few thousand demonstrations; the gripper often reaches near the
   correct object but fails to grasp it cleanly.
3. **Mis-fitting to task context** — because a small set of training tasks only ever exhibits a
   handful of behaviors per object/scene ("task context"), an action-prediction model can satisfy
   the training objective by keying off the *scene* rather than the *motion shown in the
   demonstration*, so it repeats a memorized behavior for a familiar-looking scene instead of the
   behavior actually depicted in a novel demonstration.

### 1.3 Method: AWDA (Attributed Waypoints + Demonstration Augmentation)

AWDA is a **hierarchical, modular** approach that separates *task inference* (what to do) from
*task execution* (how to do it):

- **Attributed waypoints.** Instead of predicting a dense per-timestep action, the model predicts a
  short sequence of waypoints `w = (p, a) ∈ ℝ³⁺ᵏ` — a 3D end-effector position plus `k` binary
  attributes describing robot/environment state at that point (the paper uses a single attribute,
  "is an object currently in the gripper"). With one such attribute there are 4 motor primitives:
  free-space motion (no object in hand), grasping, carrying (moving while holding an object), and
  dropping.
- **Task inference module `f(v, o₁) → waypoints`.** A ResNet-18 feature extractor (shared between
  the demonstration video and the instance image, spatial features + sinusoidal positional
  encoding), followed by a temporal self-attention transformer, followed by a two-headed MLP that
  projects the final timestep's features to waypoints. Ground-truth waypoint attributes are mined
  automatically from robot trajectories (`is grasped` = the commanded action closes the gripper
  while the jaws haven't yet closed) — no manual labeling needed.
- **Soft-DTW loss.** Because the number of "real" waypoints varies per demonstration and predicted
  waypoints needn't align 1:1 in time with recorded trajectory points, the predicted waypoint
  sequence is linearly interpolated into a dense trajectory and compared against the ground-truth
  trajectory with Soft Dynamic Time Warping (differentiable DTW). The model is trained to predict
  1, 2, 3, 4, and 5 waypoints simultaneously (15 waypoints total, `(w+1)·w/2` for `w=5`) and the
  SDTW losses for each sub-count are averaged.
- **Task execution via motor primitives.** At test time, `f(v, o₁)` is invoked *once* to get the
  full waypoint sequence, then hand-crafted, closed-loop primitives move the robot between
  consecutive waypoints: free-space motion via inverse kinematics; grasping by localizing the
  nearest object from an eye-in-hand depth camera (mask by depth → connected components → pick the
  connected component closest to image center → project its centroid to 3D via camera intrinsics)
  and executing a scripted approach/descend/close/lift sequence; dropping by opening the gripper.
  This sidesteps both the DAgger problem (primitives are closed-loop and don't need to imitate
  fine-grained recovery behavior) and last-centimeter errors (grasping uses depth-based
  localization instead of learned fine control).
- **Asymmetric Demonstration Mixup (ADM).** To break the model's ability to shortcut task inference
  by keying off scene context, training pairs are blended with other training pairs
  *asymmetrically*: the instance image `o₁` and every frame of the demonstration video `v` are
  blended toward another example's `o₁`/first-frame respectively, but the supervision target
  (ground-truth waypoints) is kept entirely from the original pair. This forces the model to track
  *motion in the demonstration* rather than the *static content of the scene*, since the scene
  content is now partially decorrelated from the task label. Implemented as the `mixup: True` flag
  in `scripts/train_transformer.py::forward`.
- **Trajectory Synthesis (TS).** A second augmentation: synthesize extra training pairs by driving
  the end-effector through 1–3 random free-space waypoints (via IK) and using that same synthetic
  trajectory as its own "demonstration" and "ground truth" (v = õ = the same synthetic motion).
  This further decorrelates task/context because these samples have no fixed relationship to any
  real object. The model has a second output head so it can absorb these synthesized samples
  without polluting the head used for real, task-driven predictions.

### 1.4 Results

Evaluated on 4 benchmarks (a Panda/Sawyer pick-and-place task-set, Meta-World, MOSAIC, and the
real-robot BC-Z dataset). Headline numbers: **100%** success on pick-and-place (vs. 10% for T-OSVI,
1% for DAML), **48%** on Meta-World [all] (vs. 28% for T-OSVI). Ablations relevant to this
adaptation:

| Pick-and-place | DAML | T-OSVI | no AD | no ADM | only waypoints | Full (ADM + TS) |
|---|---|---|---|---|---|---|
| success | .01 | .10 | .98 | .01 | .98 | 1.00 |

i.e. for this specific task-set, **ADM alone already gets to 0.98**, and Trajectory Synthesis (the
`AD` column) adds only ~0.02 on top — which is why the adaptation in this repo implements ADM but
skips TS (see `docs/02_training_adaptation.md`).

## 2. The `osvi-awda` repository

Code released alongside the paper (`README.md`, `requirements.txt`). Built on modified forks of
Robosuite, T-OSVI ("one_shot_transformers"), Meta-World, and MOSAIC (all vendored under
`robosuite/`, `robosuite_env/`, `metaworld/`, `mosaic/`).

### 2.1 Layout

| Path | Role |
|---|---|
| `hem/models/inverse_module.py` | `InverseImitation` — the full policy: `_TransformerFeatures` (ResNet-18 + positional encoding + self-attention), `waypoint_head` (two-headed MLP), plus dead-code paths for a non-waypoints action-prediction baseline (`T-OSVI`-style BC + inverse-dynamics model) that this repo also supports for ablations. |
| `hem/models/trainer.py` | `Trainer` — plain (non-Hydra) YAML-config training harness: builds train/val `DataLoader`s from a registered dataset class, runs the epoch/step loop, logs to TensorBoard, checkpoints every `save_freq` steps. |
| `hem/datasets/agent_teacher_dataset.py` | `AgentTeacherDataset` — pairs a "teacher" (demonstration) trajectory with an "agent" (robot) trajectory; supports several dataset layouts (pick-and-place `.pkl` files split by task index, MOSAIC/Meta-World/BC-Z directory layouts, etc). |
| `hem/datasets/agent_dataset.py` | `AgentDemonstrations`/`TeacherDemonstrations` — per-trajectory frame sampling, augmentation, and the attributed-waypoint ground-truth mining (`traj_points`, grasp attribute from `action[-1] > 0.01`, `grasp_point`). |
| `hem/datasets/savers/trajectory.py` | osvi-awda's own bespoke `Trajectory` pickle format (distinct from — but structurally similar to — the Multi-Task-LFD-Training-Framework's own `Trajectory` class; see docs/02). |
| `scripts/train_transformer.py` | Training entrypoint: `forward()` (the loss function, including `compute_loss_trajectory` — the SDTW loss over sub-waypoint-counts — and the ADM/mixup and Trajectory-Synthesis-mixup code paths) plus a thin `__main__` that builds `InverseImitation` and calls `Trainer.train`. |
| `soft_dtw_cuda.py` | Differentiable Soft-DTW (CUDA, via `numba.cuda`), MIT-licensed third-party code (Maghoumi). |
| `experiments/pick_place_simple.yaml`, `experiments/metaworld_with_ts.yaml` | The paper's own training configs for the two benchmarks discussed above. |
| `scripts/evaluate.py`, `scripts/collect_demonstrations.py`, `scripts/gen_metaworld_data.py` | osvi-awda's own data-generation and rollout-evaluation scripts, tied to its bundled sim envs — **not used** by this adaptation (see docs/03), since we instead use the Multi-Task-LFD-Training-Framework's own UR5e pick-place simulation. |

### 2.2 Config → training-loop data flow (paper's own `pick_place_simple.yaml`)

`dataset.type: "agent teacher"` → `AgentTeacherDataset(**dataset_cfg, mode=...)`. Each `__getitem__`
returns `(teacher_context, agent_pairs)` (a 2-tuple, since `agent_context: 0` is falsy) where:

- `teacher_context = {'video': [T_context,3,H,W], 'projection_matrix': [4,4], 'fname': str}` — the
  demonstration video, augmented and resized.
- `agent_pairs = {'images': [T_pair+1,3,H,W], 'traj_points': [50,4], 'head_label': int,
  'setting_name': str, 'grasp_point': [3], 'states': ..., 'actions': ..., 'projection_matrix':
  [4,4]}` — the robot trajectory window plus its mined attributed-waypoint ground truth.

`Trainer.train(model, forward)` iterates the DataLoader, calls
`forward(config, model, device, context, traj)` each step, which (for `waypoints: true`) computes
`out = model(...)` → predicted waypoints, then `compute_loss_trajectory(out['waypoints'],
traj['traj_points'], ...)` → the SDTW loss, and backpropagates.

This exact `(context, traj)` contract — and specifically which of its keys are actually
load-bearing for the `waypoints: true` training path — is what
`mtlfd_adaptation/mtlfd_dataset.py::MTLFDAgentTeacherDataset` had to reproduce against the
Multi-Task-LFD-Training-Framework's on-disk dataset; see `docs/02_training_adaptation.md` for the
full trace.
