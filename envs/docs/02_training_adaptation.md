# Adapting OSVI-AWDA's training to the Multi-Task-LFD-Training-Framework dataset

## Goal

Train OSVI-AWDA's actual method (attributed waypoints + Soft-DTW loss + Asymmetric Demonstration
Mixup) — not the Multi-Task-LFD-Training-Framework's own BC/`VideoImitation` model — using the
Multi-Task-LFD-Training-Framework's dataset conventions and files:
`/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset/pick_place/human_rgb_pick_place` (teacher
video demonstrations) paired with `.../ur5e_pick_place` (agent robot trajectories).

## 1. Why osvi-awda's own dataset code couldn't be pointed at the new directories directly

osvi-awda's `hem/datasets/agent_dataset.py::AgentDemonstrations` calls a module-level `load_traj`
(`hem/datasets/__init__.py`) that unpickles `{'traj': <hem.datasets.savers.trajectory.Trajectory>}`
— **osvi-awda's own bespoke pickle format**, distinct from the target framework's. Probing the
actual on-disk files (`pickle.Unpickler.find_class` override, no full import needed) showed both
`human_rgb_pick_place` and `ur5e_pick_place` pkls are saved with
**`multi_task_il.datasets.savers.Trajectory`** instead — the Multi-Task-LFD-Training-Framework's own
class, which JPEG-compresses `obs['camera_front_image']`/`obs['eye_in_hand_image']` and decompresses
lazily inside `.get(t)`. Naively pointing `AgentDemonstrations` at these files would either fail to
unpickle (wrong class registered under that module path) or, once patched to unpickle, would find
`obs['camera_front_image']` (compressed JPEG bytes) where osvi-awda's code expects
`obs['image']` (raw decompressed array).

Directly reusing the target framework's own `multi_task_il.datasets.multi_task_datasets.MultiTaskPairedDataset`
was considered and rejected: it unconditionally asserts `obs['eye_in_hand_image'] is not None` for
every agent frame (`ur5e_pick_place`'s "opt" pkls don't carry it — see §3 below) and its
`__getitem__` produces a batch contract (raw BC action distributions + bounding boxes) built for a
totally different model (`VideoImitation`), not OSVI-AWDA's waypoint/SDTW objective.

**Conclusion: write a small, purpose-built bridge + a new dataset class**, reusing only the pieces
that are actually shared (the trajectory pickle/decompression logic, and the on-disk directory
convention), and reusing osvi-awda's own `InverseImitation` model, `Trainer`, and loss function
completely unmodified.

## 2. New files (all under `osvi-awda/`, package `mtlfd_adaptation/`)

| File | Role |
|---|---|
| `mtlfd_adaptation/trajectory_bridge.py` | `load_traj(fname)` — inserts `Multi-Task-LFD-Training-Framework/training` onto `sys.path`, unpickles via the *real* `multi_task_il.datasets.savers.Trajectory` (so JPEG decompression "just works"), and wraps it (`AliasedTrajectory`) so `.get(t)['obs']['image']` is available (aliased from `camera_front_image`) — the one field osvi-awda's frame-sampling code actually needs by that name. |
| `mtlfd_adaptation/mtlfd_dataset.py` | `MTLFDAgentTeacherDataset` — see §4. |
| `mtlfd_adaptation/experiments/pick_place_mtlfd.yaml` | Training config; see §5. |
| `mtlfd_adaptation/train_mtlfd.py` | Entrypoint. Imports `forward`/`InverseImitation`/`Trainer` from osvi-awda **unchanged**; only swaps the dataset and adds debug-image dumping (§6). |
| `mtlfd_adaptation/debug_utils.py` | Debug-image dumping helpers (frame grids, waypoint overlays). |
| `mtlfd_adaptation/train_mtlfd.sh` | SBATCH launcher, modeled on `Multi-Task-LFD-Training-Framework/bashes/train_mosaic_target_obj_detector_double_policy.sh`. |
| `hem/datasets/__init__.py` | +1 line: registers dataset type `"mtlfd agent teacher"` → `MTLFDAgentTeacherDataset`. The only edit to osvi-awda's existing code. |

## 3. Cross-referencing the actual pkl schema against what osvi-awda's attribute-mining needs

Directly inspecting sample trajectories (`traj000.pkl` in `task_00` of both directories, via
`multi_task_il`'s real `Trajectory.get()`) confirmed:

- `human_rgb_pick_place`: obs has **only** `camera_front_image` (376×672×3 uint8) per frame. No
  state/action/reward. This is exactly what osvi-awda's demo/context loading needs (it never reads
  anything but the image for the teacher stream) — **no adapter logic needed here beyond the JPEG
  decompression + key alias**.
- `ur5e_pick_place`: obs has `ee_aa` (6,) float32 — first 3 components are the 3D end-effector
  position, exactly the field osvi-awda's own attributed-waypoint mining
  (`hem/datasets/agent_dataset.py::_get_pairs`) reads (`obs['ee_aa'][:3]`) — plus a 7-dim `action`
  (xyz + axis-angle + gripper at `action[-1]`, `-1`=open/`+1`=close). Since `'grasp'` is **not** a
  key in this obs (that key only exists for Meta-World/MOSAIC/BC-Z data), osvi-awda's own code
  already falls back to exactly the right branch for pick-place-style data:
  `grasp_frames = [False] + [action[-1] > 0.01 for t in range(1, len(traj))]`. **This grasp-attribute
  mining logic needed zero changes** — `MTLFDAgentTeacherDataset` reproduces it verbatim against
  `ur5e_pick_place`'s schema.
- `ur5e_pick_place` obs does **not** contain `eye_in_hand_image`/depth (confirmed on both the "opt"
  dataset used for training here, and even on one `real_eye_in_hand_ur5e_pick_place` sample
  checked) — this is why `MultiTaskPairedDataset`'s hard assert on that key is a dead end for this
  data (§1), and why the paper's depth-based grasp motor primitive (test time only, not needed for
  training) is instead driven by *live* eye-in-hand rendering from the simulator rather than
  anything read from these pkls — see `docs/03_test_adaptation.md`.

## 4. `MTLFDAgentTeacherDataset` — what's real vs. placeholder

Traced exactly which fields of the `(context, traj)` batch dict osvi-awda's
`scripts/train_transformer.py::forward()` actually *reads* on the `waypoints: true` path (as
opposed to fields that are only used by the non-waypoints BC-baseline branch, which this
adaptation never exercises):

| Key | Real or placeholder | Why |
|---|---|---|
| `context['video']`, `traj['images']` | **real** | Fed to the model; `images` gets `[:, :-1]`'d inside `InverseImitation.forward` — with `T_pair=1` (2 sampled frames) this leaves exactly 1 real frame (`o1`), matching the paper's "single image `o1`" input. |
| `traj['traj_points']` | **real** | The SDTW supervision target — attributed waypoints mined from `ee_aa` + the grasp threshold above. |
| `traj['head_label']`, `traj['setting_name']` | **real** | Used for per-head loss masking (always 0 — single-head, no Trajectory-Synthesis second head, see §5) and per-subtask val-loss breakdown. |
| `traj['states']`, `traj['actions']` | **placeholder (zeros)** | Extracted by `forward()` but **never used** on the waypoints path — `InverseImitation.forward` only touches `states`/`context`/`images` when `self.waypoints is not None`, and `actions` is only consumed by the BC-baseline's `l_bc`/`l_inv` losses, never reached here. Confirmed by reading `inverse_module.py::InverseImitation.forward` line by line. |
| `traj['grasp_point']` | **placeholder-ish (computed but unused)** | `forward()` only uses it via `out['pred_grasp_point']`, a key `InverseImitation` never actually produces — so this is dead code on this path; computed anyway since it's nearly free. |
| `traj['projection_matrix']`, `context['projection_matrix']` | **placeholder (`np.eye(4)`)** | Only numerically used if `image_waypoints: true` (predicting *image-plane-projected* waypoints, which needs a real camera calibration). We set `image_waypoints: False` (see §5) specifically so a real projection matrix is never required — the model instead predicts raw 3D waypoints in ee-space. The dict key must still be *present* (the line `projection = traj['projection_matrix']` runs unconditionally) even though its value is never used numerically in that mode. |

This trace is what let the dataset class skip re-deriving state/action/points machinery entirely —
only `images`, `traj_points`, `head_label`, and `setting_name` needed to be real.

## 5. Config (`mtlfd_adaptation/experiments/pick_place_mtlfd.yaml`)

Derived from the paper's own `experiments/pick_place_simple.yaml` (same `policy` block —
architecture/hyperparameters are untouched, since this is the same task family: Panda/Sawyer
pick-and-place vs. UR5e pick-and-place). Differences:

- `dataset.type: "mtlfd agent teacher"`, `dataset.root_dir: ${EXPERT_DATA}` (set to
  `/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset` at launch), `agent_name: ur5e`,
  `demo_name: human_rgb`, `test_tasks: [12, 13, 14, 15]` — 4 of the 16 pick-place object×bin
  variations held out for validation, matching the convention already used throughout
  Multi-Task-LFD-Training-Framework's own pick_place configs/checkpoint names
  (`...-SKIP-12-13-14-15-...`).
- `image_waypoints: False` (top level) — see §4; predicts raw 3D ee-space waypoints instead of
  image-plane-projected ones, since no real camera calibration for these cameras is threaded
  through the new dataset class.
- `mixup: True` retained — this *is* Asymmetric Demonstration Mixup (implemented entirely inside
  `scripts/train_transformer.py::forward`, operates purely on the already-batched tensors, so it
  needed zero dataset-side changes).
- **Trajectory Synthesis intentionally omitted** (per the approved plan): the paper's own ablation
  (Table V) shows pick-and-place already gets `.98` success with `no AD` vs `1.00` with the full
  method — a 2-point gap not worth the engineering cost of a synthetic-trajectory generator for
  this task. Consequently `ent_head`/`num_ent_head` (the two-headed waypoint predictor used to keep
  synthesized-trajectory samples from polluting real predictions) are left at their defaults
  (single head) — matching the paper's own "single head" ablation row, which still gets `.99` on
  pick-and-place.
- `batches: 20000` (bounded step budget) replaces the paper's `epochs: 2000` — with 12 training
  subtasks and up to 100×40 demo/agent pairs per subtask, one "epoch" here is far larger than in
  the paper's own runs, so a fixed step budget is a more meaningful/portable knob than an epoch
  count for this reduced-scope adaptation.

## 6. `train_mtlfd.py` — what's reused vs. added

`InverseImitation`, `Trainer`, and `scripts.train_transformer.forward`/`compute_loss_trajectory`
are **imported and called unmodified**. The only additions:

- Debug-image dumps (deliverable #5): before training starts, one batch each from the train/val
  loaders is dumped as demo-context and agent-frame PNG grids to
  `<save_dir>/debug_images/preprocessing/` (`--debug-preprocessing-only` lets this run standalone,
  without touching the GPU/model, for a fast sanity check).
- A thin wrapper around `forward()` that, every `img_log_freq` validation steps, runs one extra
  (no-grad) forward pass on a validation example and saves a predicted-vs-ground-truth waypoint
  overlay to `<save_dir>/debug_images/training/step_XXXXXXX/waypoints.png`. This never touches the
  loss/backward pass — if the overlay code throws, it's caught and logged, training continues.
- Training-loss curves (train/val SDTW) are otherwise logged exactly as osvi-awda's own
  `Trainer`/`SummaryWriter` already does — no changes there.

## 7. Environment

Training only needs `InverseImitation`'s actual dependencies (torch, torchvision, opencv, einops,
numpy, PyYAML, matplotlib, `numba`+CUDA for `soft_dtw_cuda.py`) — **not** osvi-awda's own bundled
Robosuite/Metaworld/dm_control stack (`requirements.txt`), since neither its data-generation
scripts nor its own sim envs are used. All of the former were already present in this cluster's
`multi_task_lfd_cuda_12_8` conda env (which also already carries the target framework's own
`multi_task_il`/`multi_task_robosuite_env`/`multi_task_test` packages, needed later for the test
harness).

**One real blocker found and fixed**: `soft_dtw_cuda.py` (the paper's differentiable Soft-DTW
implementation, third-party MIT code, unmodified) uses `numba.cuda` kernels. On this cluster's A100
nodes (CUDA driver reporting runtime version 12.8), `numba==0.57.1`'s legacy ctypes-based CUDA
driver bindings **segfault** inside `numba/cuda/cudadrv/driver.py::safe_cuda_api_call` the moment a
tensor is handed to a `@cuda.jit` kernel via `numba.cuda.as_cuda_array` — reproduced in isolation
(a 4-line SDTW-only repro script) on both the pre-existing `multi_task_lfd` (torch 1.13/cu117) and
`multi_task_lfd_cuda_12_8` (torch 2.8/cu128) envs, confirming it's a numba/driver ABI
incompatibility, not anything specific to this adaptation's code. Fix: cloned
`multi_task_lfd_cuda_12_8` into an isolated `osvi_mtlfd` env (so the shared, actively-used
production env is never touched), then:

1. `pip install "cuda-python==12.6.*"` inside `osvi_mtlfd`. The default latest (`cuda-python`
   13.0.3) was tried first and made things *worse* — 13.x restructured its imports into
   `cuda.bindings.*`, breaking with `ImportError: cannot import name 'cuda' from 'cuda'` on
   `numba`'s `from cuda import cuda as binding, nvrtc` import. Pinning to the `12.6.x` line matches
   what `numba==0.57.1` actually expects to import.
2. Set `NUMBA_CUDA_USE_NVIDIA_BINDING=1` (exported in `train_mtlfd.sh`/`test_mtlfd.sh`), which
   switches numba onto the modern NVIDIA-official CUDA binding instead of its legacy ctypes one —
   the ctypes path is what was segfaulting against this cluster's driver.

Verified fixed via an isolated forward+backward Soft-DTW test on an A100 node before touching the
real training loop, and confirmed again by the full training run itself (§8) running for
thousands of steps without a single CUDA-side crash.

## 8. Results

An initial interactive smoke test confirmed the loss is finite and decreasing (train 5.23 → 0.8,
val 2.87 → 0.9 over ~920 steps, ~10 minutes on one A100), before committing to the full run.

The full training job (`sbatch mtlfd_adaptation/train_mtlfd.sh`, SLURM job 521070, 20000-step
budget) was then launched and, per explicit instruction, this adaptation's evaluation deliverables
were produced against an in-progress checkpoint rather than waiting for that job to reach its full
step count:

| Step | Train SDTW loss | Val SDTW loss |
|---|---|---|
| 0 | 5.2306 | 2.8728 |
| 2000 | 1.0820 | 1.1759 |
| 4000 | 1.0846 | 1.2760 |
| 6000 | 0.9394 | 0.9492 |
| 8000 | 0.7845 | 0.8927 |
| 10000 | 0.8343 | 0.9713 |

Train loss shows a clear, monotone-ish downward trend; val loss drops sharply early then oscillates
in the 0.85–1.0 range on this small (4-subtask) held-out split — discussed in more depth, alongside
qualitative waypoint-prediction debug images and a full rollout evaluation on the held-out subtasks
at the `model_save-6000.pt` checkpoint, in `docs/04_training_verification.md`.
