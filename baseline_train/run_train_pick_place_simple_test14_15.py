"""
Auto-resubmitting wrapper for the paper-faithful pick_place_simple_test14_15.yaml campaign (14
train / 2 test tasks, batches: 500000 - matches 2302.04856v1.pdf's own reported protocol).
Duplicate of run_train_pick_place_simple.py, pointed at the new experiment file/sbatch script;
see that file's own docstring for the resume-mechanics background (unchanged here).

Usage (from a login node / tmux session):
    python3 baseline_train/run_train_pick_place_simple_test14_15.py [--max-restarts N]
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
TRAIN_SH = os.path.join(REPO_ROOT, "baseline_train", "train_pick_place_simple_test14_15.sh")
EXPERIMENT_YAML = os.path.join(REPO_ROOT, "experiments", "pick_place_simple_test14_15.yaml")
SAVE_PARENT = "/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/osvi_awda_baseline"
POLL_INTERVAL_S = 60
# Measured steady-state throughput from the unmodified-yaml campaign (2.82-2.91 bps, same model/
# hardware) -> 500000/2.85 ~= 175,440s ~= 48.7h -> ceil(48.7/7) = 7 blocks; padded ~40% for
# restart/queue overhead, matching the margin used for the main campaign's own restart budget.
DEFAULT_MAX_RESTARTS = 10


def target_steps():
    with open(EXPERIMENT_YAML) as f:
        cfg = yaml.safe_load(f)
    return cfg["batches"]


def run_name():
    return os.path.splitext(os.path.basename(EXPERIMENT_YAML))[0]


def find_checkpoint_dir():
    pattern = os.path.join(SAVE_PARENT, run_name(), "*_ckpt-*")
    candidates = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getctime)


def highest_checkpoint_step(ckpt_dir):
    steps = []
    for fname in os.listdir(ckpt_dir):
        m = re.match(r"model_save-(\d+)\.pt$", fname)
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
        env["EXPERIMENT_FILE"] = EXPERIMENT_YAML
        args += [TRAIN_SH]

    result = subprocess.run(args, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch submission failed: {result.stderr}")
    for line in result.stdout.splitlines():
        if "Submitted batch job" in line:
            return line.split()[-1]
    raise RuntimeError(f"could not parse job id from sbatch output: {result.stdout!r}")


def wait_for_job(job_id, poll_interval=POLL_INTERVAL_S):
    print(f"[run_train_test14_15] waiting for job {job_id} to finish...", flush=True)
    while True:
        result = subprocess.run(["squeue", "-j", job_id, "-h"], capture_output=True, text=True)
        if not result.stdout.strip():
            print(f"[run_train_test14_15] job {job_id} is no longer in the queue.", flush=True)
            return
        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attach-job-id", type=str, default=None)
    parser.add_argument("--max-restarts", type=int, default=DEFAULT_MAX_RESTARTS)
    args = parser.parse_args()

    max_steps = target_steps()
    print(f"[run_train_test14_15] target steps: {max_steps}", flush=True)

    if args.attach_job_id is not None:
        wait_for_job(args.attach_job_id)
    else:
        job_id = submit(resume_from=None)
        print(f"[run_train_test14_15] submitted fresh job {job_id}", flush=True)
        wait_for_job(job_id)

    for restart_i in range(1, args.max_restarts + 1):
        ckpt_dir = find_checkpoint_dir()
        if ckpt_dir is None:
            print("[run_train_test14_15] no checkpoint directory found after the job ended - "
                  "the run likely crashed before Trainer even started. Not resuming blind; "
                  "check the slurm log in baseline_train/slurm_logs/. Exiting.", flush=True)
            sys.exit(1)

        step = highest_checkpoint_step(ckpt_dir)
        if step is None:
            print(f"[run_train_test14_15] {ckpt_dir} exists but has no model_save-*.pt yet - "
                  "the job died before the first save_freq interval. Not resuming blind. "
                  "Exiting.", flush=True)
            sys.exit(1)

        print(f"[run_train_test14_15] highest checkpoint step in {ckpt_dir}: {step}", flush=True)
        if step >= max_steps:
            print(f"[run_train_test14_15] reached target ({step} >= {max_steps}). Done.", flush=True)
            return

        print(f"[run_train_test14_15] restart {restart_i}/{args.max_restarts}: resuming from "
              f"{ckpt_dir} (step {step})", flush=True)
        job_id = submit(resume_from=ckpt_dir)
        print(f"[run_train_test14_15] submitted resume job {job_id}", flush=True)
        wait_for_job(job_id)

    print(f"[run_train_test14_15] hit --max-restarts ({args.max_restarts}) without reaching "
          f"{max_steps} steps. Investigate before restarting manually.", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
