#!/bin/bash
# Launcher for test_training_pointing_accuracy.py - see that file's docstring. Only needs a GPU for
# the policy network itself (no live robosuite/mujoco rendering, unlike run_demo_sensitivity.sh).
#
# Usage: MODEL_DIR=... SAVED_STEP=... N_SAMPLES=200 sbatch \
#            mtlfd_adaptation/_analysis_scratch/run_training_pointing_accuracy.sh

#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --exclude=gnode09,gnode04,gnode03,gnode06,gnode12,gnode10
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --export=ALL
#SBATCH --job-name=training_pointing_accuracy
#SBATCH --output=mtlfd_adaptation/slurm_logs/training_pointing_accuracy_%j.out

set -e

CONDA_ENV=osvi_mtlfd
REPO_ROOT="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR}"
SAVED_STEP="${SAVED_STEP:?set SAVED_STEP}"
N_SAMPLES="${N_SAMPLES:-200}"
OUT_DIR="${OUT_DIR:-}"

export MUJOCO_PY_MUJOCO_PATH=/home/rsofnc000/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin:/usr/lib/nvidia
export CUDA_VISIBLE_DEVICES=0
export NUMBA_CUDA_USE_NVIDIA_BINDING=1

mkdir -p "$REPO_ROOT/mtlfd_adaptation/slurm_logs"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

OUT_ARGS=()
if [ -n "$OUT_DIR" ]; then
    OUT_ARGS=(--out_dir "$OUT_DIR")
fi

srun python -u mtlfd_adaptation/_analysis_scratch/test_training_pointing_accuracy.py "$MODEL_DIR" \
    --saved_step "$SAVED_STEP" \
    --n_samples "$N_SAMPLES" \
    --gpu_id 0 \
    "${OUT_ARGS[@]}"
