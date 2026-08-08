#!/bin/bash
# Runs scripts/evaluate.py at the paper's own eval protocol (Table IV: 20 episodes/snapshot/task)
# against a given checkpoint. Usage: sbatch baseline_train/eval_paper_scale.sh <logdir> <bn>

#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=osvi_eval_paper
#SBATCH --output=baseline_train/slurm_logs/eval_paper_%j.out

set -e

REPO_ROOT="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
LOGDIR="$1"
BN="$2"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate awda
export MUJOCO_PY_MUJOCO_PATH=/home/rsofnc000/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=.
export EXPERT_DATA=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda/dataset

cd "$REPO_ROOT"
srun python -m scripts.evaluate "$LOGDIR" --bn "$BN" --instances 20 --envs 20 --write-images
