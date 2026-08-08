#!/bin/bash
# Runs scripts/evaluate.py in --eval-only --write-images --rest mode (README's "Evaluation"
# section): evaluates every checkpoint saved so far in <logdir> that isn't already in log.db, on
# the held-out (test_tasks) sim rollouts, writing per-episode videos/trajectories and success
# rates into log.db + tensorboard. Safe to run while training is still appending new checkpoints
# to the same logdir - --rest computes the remaining set once at startup from what's on disk.
#
# Usage: sbatch baseline_train/eval_rest.sh <logdir>

#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=07:00:00
#SBATCH --job-name=osvi_eval_rest
#SBATCH --output=baseline_train/slurm_logs/eval_rest_%j.out

set -e

REPO_ROOT="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
LOGDIR="$1"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate awda
export MUJOCO_PY_MUJOCO_PATH=/home/rsofnc000/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=.
export EXPERT_DATA=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda/dataset

cd "$REPO_ROOT"
srun python -m scripts.evaluate "$LOGDIR" --eval-only --write-images --rest
