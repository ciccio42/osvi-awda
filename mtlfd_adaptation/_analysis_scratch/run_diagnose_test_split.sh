#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --exclude=gnode09,gnode04,gnode03,gnode06,gnode12
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --export=ALL
#SBATCH --job-name=diagnose_test_split
#SBATCH --output=mtlfd_adaptation/slurm_logs/diagnose_test_split_%j.out

set -e

CONDA_ENV=osvi_mtlfd
REPO_ROOT="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
# override any of these via env vars, e.g.:
#   MODEL_DIR=... SAVED_STEP=... PLOT_DIR=... SPLIT=test sbatch run_diagnose_test_split.sh
MODEL_DIR="${MODEL_DIR:-/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/osvi_awda_baseline/pick_place_simple/bc_inv_ckpt-1784458550}"
SAVED_STEP="${SAVED_STEP:-290000}"
PLOT_DIR="${PLOT_DIR:-}"
SPLIT="${SPLIT:-test}"

export MUJOCO_PY_MUJOCO_PATH=/home/rsofnc000/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin:/usr/lib/nvidia
export EXPERT_DATA=/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset
export NUMBA_CUDA_USE_NVIDIA_BINDING=1

mkdir -p "$REPO_ROOT/mtlfd_adaptation/slurm_logs"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

PLOT_ARGS=()
if [ -n "$PLOT_DIR" ]; then
    PLOT_ARGS=(--plot-dir "$PLOT_DIR")
fi

srun python -u mtlfd_adaptation/_analysis_scratch/diagnose_waypoint_error.py \
    "$MODEL_DIR" \
    "$SAVED_STEP" \
    --split "$SPLIT" \
    --n-samples "${N_SAMPLES:-40}" "${PLOT_ARGS[@]}"
