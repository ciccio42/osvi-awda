"""
Auto-resubmitting wrapper around train_pick_place_simple.sh, for a training run that outlives the
cluster's per-job walltime (train_pick_place_simple.sh has #SBATCH --time=07:00:00). Directly
adapted from mtlfd_adaptation/run_train_mtlfd.py's submit -> poll -> resubmit pattern - see plan
section 4. osvi-awda's own scripts/train_transformer.py uses the exact same hem/models/trainer.py
Trainer resume mechanics (model weights + optimizer state + step counter, all keyed off a
checkpoint directory passed with --resume) as train_mtlfd.py did, verified by reading both files
directly - no new resume logic was needed here.

Unlike run_train_mtlfd.py, --max-restarts has NO default: experiments/pick_place_simple.yaml has
no `batches` key (only `epochs: 2000`), and the "agent teacher" dataset's same-task dense pairing
makes the real total step count a property of the actual collected data, not computable from the
yaml alone (~9.3M steps by hand-estimate - see the plan's Context section). Run
compute_target_steps.py + a short throughput-measurement run FIRST (per the plan) to size
--max-restarts from measured reality before launching this.

Usage (run from a login node / tmux session, not inside an sbatch job):
    python3 baseline_train/run_train_pick_place_simple.py --max-restarts <computed-from-ETA>

To take over monitoring/resuming an already-running job instead of submitting a fresh one:
    python3 baseline_train/run_train_pick_place_simple.py --max-restarts <N> --attach-job-id <jobid>
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

REPO_ROOT = "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
TRAIN_SH = os.path.join(REPO_ROOT, "baseline_train", "train_pick_place_simple.sh")
EXPERIMENT_YAML = os.path.join(REPO_ROOT, "experiments", "pick_place_simple.yaml")
SAVE_PARENT = "/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/osvi_awda_baseline"
POLL_INTERVAL_S = 60


def run_name():
    return os.path.splitext(os.path.basename(EXPERIMENT_YAML))[0]


def find_checkpoint_dir():
    """Mirrors how Trainer computes save_dir for a fresh (non-resume) run:
    <SAVE_PARENT>/<run_name>/bc_inv_ckpt-<timestamp>/ (save_name='bc_inv', set in
    scripts/train_transformer.py's own Trainer(args, 'bc_inv', ...) call). Picks the most
    recently created one, in case stale directories from earlier runs are still sitting around."""
    pattern = os.path.join(SAVE_PARENT, run_name(), "*_ckpt-*")
    candidates = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getctime)


def highest_checkpoint_step(ckpt_dir):
    steps = []
    for fname in os.listdir(ckpt_dir):
        m = re.match(r"model_save-(\d+)\.pt$", fname)  # excludes model_save-optim-*.pt
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else None


def submit(resume_from=None):
    env = os.environ.copy()
    args = ["sbatch"]
    if resume_from is not None:
        env["EXPERIMENT_FILE"] = resume_from
        args += [TRAIN_SH, "--resume"]
    else:
        args += [TRAIN_SH]

    result = subprocess.run(args, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch submission failed: {result.stderr}")
    for line in result.stdout.splitlines():
        if "Submitted batch job" in line:
            return line.split()[-1]
    raise RuntimeError(f"could not parse job id from sbatch output: {result.stdout!r}")


def wait_for_job(job_id, poll_interval=POLL_INTERVAL_S):
    print(f"[run_train_pick_place_simple] waiting for job {job_id} to finish...", flush=True)
    while True:
        result = subprocess.run(["squeue", "-j", job_id, "-h"], capture_output=True, text=True)
        if not result.stdout.strip():
            print(f"[run_train_pick_place_simple] job {job_id} is no longer in the queue.", flush=True)
            return
        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attach-job-id", type=str, default=None,
                         help="don't submit a fresh job; wait on this already-running one first")
    parser.add_argument("--max-restarts", type=int, required=True,
                         help="size this from a measured throughput + real target-step count "
                              "(compute_target_steps.py), not a guess - see module docstring")
    parser.add_argument("--max-steps", type=int, default=None,
                         help="optional: stop early once this step count is reached, if you "
                              "computed a real target via compute_target_steps.py")
    args = parser.parse_args()

    if args.attach_job_id is not None:
        wait_for_job(args.attach_job_id)
    else:
        job_id = submit(resume_from=None)
        print(f"[run_train_pick_place_simple] submitted fresh job {job_id}", flush=True)
        wait_for_job(job_id)

    for restart_i in range(1, args.max_restarts + 1):
        ckpt_dir = find_checkpoint_dir()
        if ckpt_dir is None:
            print("[run_train_pick_place_simple] no checkpoint directory found after the job "
                  "ended - the run likely crashed before Trainer even started. Not resuming "
                  "blind; check the slurm log in baseline_train/slurm_logs/. Exiting.", flush=True)
            sys.exit(1)

        step = highest_checkpoint_step(ckpt_dir)
        if step is None:
            print(f"[run_train_pick_place_simple] {ckpt_dir} exists but has no model_save-*.pt "
                  "yet - the job died before the first save_freq interval. Not resuming blind. "
                  "Exiting.", flush=True)
            sys.exit(1)

        print(f"[run_train_pick_place_simple] highest checkpoint step in {ckpt_dir}: {step}",
              flush=True)
        if args.max_steps is not None and step >= args.max_steps:
            print(f"[run_train_pick_place_simple] reached target ({step} >= {args.max_steps}). "
                  "Done.", flush=True)
            return

        print(f"[run_train_pick_place_simple] restart {restart_i}/{args.max_restarts}: resuming "
              f"from {ckpt_dir} (step {step})", flush=True)
        job_id = submit(resume_from=ckpt_dir)
        print(f"[run_train_pick_place_simple] submitted resume job {job_id}", flush=True)
        wait_for_job(job_id)

    print(f"[run_train_pick_place_simple] hit --max-restarts ({args.max_restarts}) - investigate "
          "before restarting manually (this may be expected if you sized --max-restarts to a "
          "deliberately partial run of the ~9.3M-step full target).", flush=True)


if __name__ == "__main__":
    main()
