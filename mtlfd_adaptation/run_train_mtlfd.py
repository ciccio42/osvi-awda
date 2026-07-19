"""
Auto-resubmitting wrapper around train_mtlfd.sh, for a training run that outlives the cluster's
per-job walltime (train_mtlfd.sh has #SBATCH --time=07:00:00). Modeled on
Multi-Task-LFD-Training-Framework/bashes/run_bash.py's submit -> poll -> resubmit loop, adapted to
this adaptation's step-based (not epoch-based) checkpoints and to osvi-awda's own Trainer, which
already supports resuming model weights + optimizer state + step counter from a checkpoint
directory (hem/models/trainer.py: passing a directory as `experiment_file` with `--resume` makes
it auto-discover the latest model_save-*.pt via pyutil.sorted_file_match - verified by reading that
code path directly; train_mtlfd.py's own model.load_state_dict(..., weights_only=False) call
handles the model weights half of that). No changes to Trainer/train_mtlfd.py's resume logic were
needed - it was already correct, just never driven by an auto-resubmit loop until now.

Usage (run from a login node / tmux session, not inside an sbatch job - this only calls
sbatch/squeue and sleeps, so it needs no GPU itself):

    python3 mtlfd_adaptation/run_train_mtlfd.py

To take over monitoring/resuming an already-running job instead of submitting a fresh one:

    python3 mtlfd_adaptation/run_train_mtlfd.py --attach-job-id <jobid>
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

import yaml

REPO_ROOT = "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
TRAIN_SH = os.path.join(REPO_ROOT, "mtlfd_adaptation", "train_mtlfd.sh")
EXPERIMENT_YAML = os.path.join(REPO_ROOT, "mtlfd_adaptation", "experiments", "pick_place_mtlfd.yaml")
SAVE_PARENT = "/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/osvi_mtlfd"
POLL_INTERVAL_S = 60
MAX_RESTARTS = 15  # 15 * 7h = ~4.4 days of wall-clock budget, generous headroom over one run


def target_steps():
    """Reads the real target off the experiment config instead of duplicating the number here -
    single source of truth. Note: Trainer's actual epoch loop can slightly overshoot this (it
    computes epochs = ceil(batches / len(loader)) and never breaks mid-epoch), so treat this as a
    lower bound - the checkpoint at exactly this step number is always saved before the job ends,
    since save_freq (2000) divides it evenly here."""
    with open(EXPERIMENT_YAML) as f:
        cfg = yaml.safe_load(f)
    return cfg["batches"]


def run_name():
    return os.path.splitext(os.path.basename(EXPERIMENT_YAML))[0]


def find_checkpoint_dir():
    """Mirrors how Trainer computes save_dir for a fresh (non-resume) run:
    <SAVE_PARENT>/<run_name>/train_ckpt-<timestamp>/. Picks the most recently created one, in case
    stale directories from earlier runs are still sitting around."""
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
    print(f"[run_train_mtlfd] waiting for job {job_id} to finish...", flush=True)
    while True:
        result = subprocess.run(["squeue", "-j", job_id, "-h"], capture_output=True, text=True)
        if not result.stdout.strip():
            print(f"[run_train_mtlfd] job {job_id} is no longer in the queue.", flush=True)
            return
        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attach-job-id", type=str, default=None,
                         help="don't submit a fresh job; wait on this already-running one first")
    parser.add_argument("--max-restarts", type=int, default=MAX_RESTARTS)
    args = parser.parse_args()

    max_steps = target_steps()
    print(f"[run_train_mtlfd] target steps: {max_steps}", flush=True)

    if args.attach_job_id is not None:
        wait_for_job(args.attach_job_id)
    else:
        job_id = submit(resume_from=None)
        print(f"[run_train_mtlfd] submitted fresh job {job_id}", flush=True)
        wait_for_job(job_id)

    for restart_i in range(1, args.max_restarts + 1):
        ckpt_dir = find_checkpoint_dir()
        if ckpt_dir is None:
            print("[run_train_mtlfd] no checkpoint directory found after the job ended - "
                  "the run likely crashed before Trainer even started. Not resuming blind; "
                  "check the slurm log in mtlfd_adaptation/slurm_logs/. Exiting.", flush=True)
            sys.exit(1)

        step = highest_checkpoint_step(ckpt_dir)
        if step is None:
            print(f"[run_train_mtlfd] {ckpt_dir} exists but has no model_save-*.pt yet - the job "
                  "died before the first save_freq interval. Not resuming blind. Exiting.",
                  flush=True)
            sys.exit(1)

        print(f"[run_train_mtlfd] highest checkpoint step in {ckpt_dir}: {step}", flush=True)
        if step >= max_steps:
            print(f"[run_train_mtlfd] reached target ({step} >= {max_steps}). Done.", flush=True)
            return

        print(f"[run_train_mtlfd] restart {restart_i}/{args.max_restarts}: resuming from "
              f"{ckpt_dir} (step {step})", flush=True)
        job_id = submit(resume_from=ckpt_dir)
        print(f"[run_train_mtlfd] submitted resume job {job_id}", flush=True)
        wait_for_job(job_id)

    print(f"[run_train_mtlfd] hit --max-restarts ({args.max_restarts}) without reaching "
          f"{max_steps} steps. Investigate before restarting manually.", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
