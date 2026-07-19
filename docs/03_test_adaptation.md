# Adapting OSVI-AWDA's testing/rollout to the Multi-Task-LFD-Training-Framework's UR5e simulation

## Goal

Evaluate a trained OSVI-AWDA checkpoint by actually executing it: given a held-out
`human_rgb_pick_place` demonstration video and a live UR5e pick-place robosuite simulation (the
same simulation used to collect `ur5e_pick_place`), infer the attributed-waypoint plan once and
execute it via the paper's 4 hand-crafted motor primitives, in the spirit of
`Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py` but with a fundamentally
different control loop.

## 1. Why `test_any_task.py` couldn't be reused directly

`test_any_task.py` (and `test/multi_task_test/pick_place.py::pick_place_eval_demo_cond`,
`utils.py::get_action`/`task_run_action`) implement a **per-timestep behavior-cloning rollout**:
build a fixed conditioning context once, then call the policy *every simulation step*, sampling an
action from its predicted mixture-of-logistics distribution, denormalizing, and stepping the env —
closed-loop the whole episode.

OSVI-AWDA is architecturally different (paper section IV): the task-inference model
`f(v, o₁)` is called **exactly once** per episode to produce a full attributed-waypoint plan; task
*execution* is then entirely delegated to hand-crafted, closed-loop **motor primitives** that move
between consecutive waypoints. There is no action-distribution sampling loop to reuse from
`get_action`. What *is* reused from the existing test harness: the live UR5e `pick_place`
environment construction, its OSC-pose controller, and its privileged-state success-metric helpers.

## 2. Building the live UR5e `pick_place` env

Traced the construction path `test_any_task.py` → `build_env_context` (`test/multi_task_test/utils.py`)
→ `TASK_MAP['pick_place']['env_fn']` = `get_expert_trajectory`
(`tasks/multi_task_robosuite_env/controllers/controllers/expert_pick_place.py`) → `get_env`
(`tasks/multi_task_robosuite_env/__init__.py`). `mtlfd_adaptation/test_mtlfd_rollout.py` calls
`get_env(...)` **directly** (skipping `get_expert_trajectory`'s wrapper, which does a confusing
GPU-id remap and an unrelated `CUDA_VISIBLE_DEVICES`-parsing assert not needed here):

```python
env = get_env('UR5e_PickPlaceDistractor', controller_configs=controller_config, task_id=variation,
               has_renderer=False, has_offscreen_renderer=True, reward_shaping=False,
               use_camera_obs=True, ranges=action_ranges, render_gpu_device_id=gpu_id,
               render_camera='camera_front', object_set=2)
```

This returns a `CustomOSCPoseWrapper` (`tasks/multi_task_robosuite_env/custom_osc_pose_wrapper.py`)
around the raw `UR5ePickPlace` robosuite env. `task_id` (0-15) is a **constructor-time** parameter —
`object_id = task_id // 4` (which of 4 target objects), `bin_id = task_id % 4` (which of 4 target
bins); changing variation requires building a new env instance, `env.reset()` only re-randomizes
object placement *within* the fixed variation. We iterate the 4 held-out task_ids (12-15, same
subtasks excluded from training — see `docs/02_training_adaptation.md` §5) to evaluate on unseen
task instances, matching the paper's held-out-task evaluation protocol.

The controller is loaded from the exact same config file the existing test harness uses:
`tasks/multi_task_robosuite_env/controllers/config/osc_pose.json` (`OSC_POSE`, `control_delta:
true`, `kp=150`).

## 3. Action interface: absolute-pose waypoint following, not the training-time action space

`env.step(action)` on `CustomOSCPoseWrapper` takes a **7-dim absolute-pose action**
`[x, y, z, ax, ay, az, gripper]` — world-frame target position, world-frame target orientation
(axis-angle), gripper (`-1`=open, `+1`=close) — and internally converts it to the delta the inner
`control_delta=True` OSC controller expects, repeating the tracking step **5×** per outer
`env.step()` call. This means a simple closed-loop "move toward a 3D target" primitive can be built
by repeatedly calling `env.step()` with the current position nudged toward the goal (clipped to a
max step size), letting the OSC impedance controller handle the smooth tracking — exactly the
pattern already used by this framework's own scripted experts
(`expert_pick_place.py::PickPlaceController`) and its `primitive.py::reaching_primitive`. That
existing pattern is what `mtlfd_adaptation/test_mtlfd_rollout.py::move_to` reimplements (clip
`target - current` to `±0.02m` per step, loop until within `0.008m` or a max-iteration timeout).

## 4. The 4 motor primitives (paper section IV-A / Appendix VII)

OSVI-AWDA predicts 5 attributed waypoints `(x, y, z, grasp_attr)`, **relative to the robot's
end-effector position at the moment `o₁` was captured** (confirmed by tracing
`compute_loss_trajectory` in `scripts/train_transformer.py`: the ground-truth trajectory is shifted
by `- cureepos` — the trajectory's own start position — before the SDTW comparison, and the
predicted-waypoint side of that comparison is built starting from the origin). So execution first
reads the *live* starting end-effector position, then adds each predicted waypoint's `(x,y,z)` onto
it to get an absolute world-frame target.

For each consecutive pair of waypoints, the transition between their (thresholded) grasp attributes
picks one of 4 primitives, exactly matching the paper's `2^(k+1)` formula for `k=1` attribute:

| Transition | Primitive | Implementation |
|---|---|---|
| off → off | **free-space motion** | `move_to` with gripper open |
| off → on | **grasping** | see §5 |
| on → on | **carrying** | `move_to` with gripper closed |
| on → off | **dropping** | open the gripper in place |

## 5. Grasping: live eye-in-hand depth localization (paper Appendix VII-A)

Per the approved plan, this is implemented **paper-faithfully** using the simulator's live
eye-in-hand camera rather than substituting privileged object-position ground truth — confirmed
feasible by inspecting `/mnt/beegfs/frosa/robot_datasets/dataset/no_opt_dataset/pick_place/ur5e_pick_place`,
whose recorded trajectories *do* carry `eye_in_hand_image`/`eye_in_hand_depth` (+ `extent`/
`zfar`/`znear` camera intrinsics), proving the same simulation env that produced the "opt" training
dataset can render this camera; the config
(`tasks/multi_task_robosuite_env/config/PickPlaceDistractor.yaml`, `camera_names: [...,
'robot0_eye_in_hand']`, `camera_depths: true`) shows rendering happens automatically every step
once that camera name is requested — no special render call needed.

`localize_grasp_target` (`mtlfd_adaptation/test_mtlfd_rollout.py`) reimplements the paper's
described procedure exactly:
1. Mask out background (`depth > 1m`).
2. Estimate the floor-plane distance as the median of the remaining (valid) depth values.
3. Mask pixels `>1cm` above that floor estimate (potential-object mask).
4. Connected components (`cv2.connectedComponents`) on that mask.
5. Pick the component whose centroid is closest to the image center (closest object = target,
   matching the paper's stated heuristic).
6. Backproject that centroid pixel + its real depth to a 3D world point.

Step 6 needed a small self-contained module, `mtlfd_adaptation/robosuite_camera_utils.py`: initial
research suggested reusing `robosuite.utils.camera_utils` (which ships exactly these utilities
upstream), but that module **does not exist** in the specific vendored robosuite fork this project
uses (confirmed by grep — an unverified claim from that research pass that would have broken at
runtime). Reimplemented from MuJoCo's core, version-independent camera fields instead
(`sim.model.cam_fovy`, `sim.data.cam_xpos`/`cam_xmat`, `sim.model.stat.extent`,
`sim.model.vis.map.{zfar,znear}`), using the same real-depth conversion formula already present
in-repo (`custom_osc_pose_wrapper.py::_get_real_depth`) so the two are guaranteed consistent.
`get_camera_extrinsic_matrix` re-derives the camera pose from `sim.data.cam_xpos`/`cam_xmat` on
every call, which correctly tracks the eye-in-hand camera as it moves with the gripper (it's not a
fixed camera).

The grasp primitive then: hovers 15cm above the waypoint's coarse hint position, localizes the
actual target via the above procedure, approaches 15cm above the *refined* position, descends,
closes the gripper, and lifts 15cm — matching the paper's Appendix VII-A step list.

## 6. Success metrics — reused, not reimplemented

Rather than re-deriving privileged-state success checks, `run_episode` imports and reuses
`check_reach`/`check_pick`/`check_bin` directly from `multi_task_test.utils` (same thresholds the
existing harness already uses: `0.03m` reach, `0.05m` lift, `±0.08m`/bin-height-band for
misplacement), and reads ground-truth episode success straight from the environment's own
`_check_success()` — the same signal robosuite already computes as the `reward` returned by
`env.step()`.

## 7. Debug images / outputs

Per episode, `mtlfd_adaptation/test_mtlfd_rollout.py` writes to
`<results_dir>/debug_images/task{NN}_ep{NN}/`:
- `predicted_plan.png` — the 5 predicted waypoints (green=free-motion, red=grasp-on) projected
  onto the starting `camera_front` frame using the real camera intrinsics/extrinsics from §5.
- `frame_XXXX.png` — sampled frames through the primitive execution (every 3rd control step).
- `result.json` — per-episode reach/pick/success flags.

A `summary.json` aggregates success/reach/pick rates across all evaluated episodes — see
`docs/04_training_verification.md` for actual numbers once a trained checkpoint is evaluated.

## 8. Environment

Requires `multi_task_il`/`multi_task_robosuite_env`/`multi_task_test` importable together with
osvi-awda's own dependencies (torch, opencv, etc.) — i.e. the same `osvi_mtlfd` env built for
training (see `docs/02_training_adaptation.md` §7), which was cloned from
`multi_task_lfd_cuda_12_8` specifically because that's the env with all three of those packages
correctly `pip install -e`'d against this repo (confirmed by checking `<package>.__file__` across
every conda env on the cluster — a naively-plausible-looking `multi_task_robosuite_1_5` env turned
out to have same-named packages installed from an unrelated project (`VLA-Benchmark`), not this
one — worth flagging since it's an easy trap).
