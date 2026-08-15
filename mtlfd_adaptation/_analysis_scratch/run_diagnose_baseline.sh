#!/bin/bash
# One-off launcher for diagnose_waypoint_error_baseline.py. Needs a GPU node (login node has no
# CUDA device); pass MODEL_DIR/SAVED_STEP via env vars, e.g.:
#   MODEL_DIR=... SAVED_STEP=510000 sbatch mtlfd_adaptation/_analysis_scratch/run_diagnose_baseline.sh

#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --exclude=gnode09,gnode04,gnode03,gnode06,gnode12
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --export=ALL
#SBATCH --job-name=diagnose_baseline
#SBATCH --output=mtlfd_adaptation/slurm_logs/diagnose_baseline_%j.out

set -e

CONDA_ENV="${CONDA_ENV:-osvi_mtlfd}"
REPO_ROOT="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"

export MUJOCO_PY_MUJOCO_PATH=/home/rsofnc000/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin:/usr/lib/nvidia
export NUMBA_CUDA_USE_NVIDIA_BINDING=1

mkdir -p "$REPO_ROOT/mtlfd_adaptation/slurm_logs"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

PLOT_ARGS=()
if [ -n "$PLOT_DIR" ]; then
    PLOT_ARGS=(--plot-dir "$PLOT_DIR")
fi

srun python -u mtlfd_adaptation/_analysis_scratch/diagnose_waypoint_error_baseline.py \
    "$MODEL_DIR" "$SAVED_STEP" --split "${SPLIT:-test}" --n-samples "${N_SAMPLES:-40}" "${PLOT_ARGS[@]}"
