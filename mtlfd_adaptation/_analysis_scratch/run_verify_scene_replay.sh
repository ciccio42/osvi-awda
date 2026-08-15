#!/bin/bash
# Launcher for verify_scene_replay.py - see that file's docstring.
# Usage: TASK_ID=12 AGENT_PKL=/path/to/traj.pkl sbatch mtlfd_adaptation/_analysis_scratch/run_verify_scene_replay.sh

#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --exclude=gnode09,gnode04,gnode03,gnode06,gnode12,gnode10
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --export=ALL
#SBATCH --job-name=verify_scene_replay
#SBATCH --output=mtlfd_adaptation/slurm_logs/verify_scene_replay_%j.out

set -e

CONDA_ENV=osvi_mtlfd
REPO_ROOT="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"

TASK_ID="${TASK_ID:?set TASK_ID}"
AGENT_PKL="${AGENT_PKL:?set AGENT_PKL}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/mtlfd_adaptation/_analysis_scratch/scene_replay_check}"

export MUJOCO_PY_MUJOCO_PATH=/home/rsofnc000/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin:/usr/lib/nvidia
export CUDA_VISIBLE_DEVICES=0
export NUMBA_CUDA_USE_NVIDIA_BINDING=1

mkdir -p "$REPO_ROOT/mtlfd_adaptation/slurm_logs"
cd "$REPO_ROOT"
rm -f ./*.png ./*.jpg 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

srun python -u mtlfd_adaptation/_analysis_scratch/verify_scene_replay.py "$TASK_ID" "$AGENT_PKL" "$OUT_DIR"
