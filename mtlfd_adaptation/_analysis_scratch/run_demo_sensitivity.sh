#!/bin/bash
# Launcher for test_demo_sensitivity.py - see that file's docstring. Needs the same live
# robosuite/multi_task_robosuite_env sim as test_mtlfd.sh (mirrors its env setup exactly).
#
# Usage: MODEL_DIR=... SAVED_STEP=... TASK_ID=12 N_DEMOS=8 sbatch \
#            mtlfd_adaptation/_analysis_scratch/run_demo_sensitivity.sh

#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --exclude=gnode09,gnode04,gnode03,gnode06,gnode12,gnode10
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --export=ALL
#SBATCH --job-name=demo_sensitivity
#SBATCH --output=mtlfd_adaptation/slurm_logs/demo_sensitivity_%j.out

set -e

CONDA_ENV=osvi_mtlfd
REPO_ROOT="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR}"
SAVED_STEP="${SAVED_STEP:?set SAVED_STEP}"
TASK_ID="${TASK_ID:-12}"
N_DEMOS="${N_DEMOS:-8}"
OUT_DIR="${OUT_DIR:-}"

export MUJOCO_PY_MUJOCO_PATH=/home/rsofnc000/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin:/usr/lib/nvidia
export CUDA_VISIBLE_DEVICES=0
export NUMBA_CUDA_USE_NVIDIA_BINDING=1

mkdir -p "$REPO_ROOT/mtlfd_adaptation/slurm_logs"
cd "$REPO_ROOT"
rm -f ./*.png ./*.jpg 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

OUT_ARGS=()
if [ -n "$OUT_DIR" ]; then
    OUT_ARGS=(--out_dir "$OUT_DIR")
fi

srun python -u mtlfd_adaptation/_analysis_scratch/test_demo_sensitivity.py "$MODEL_DIR" \
    --saved_step "$SAVED_STEP" \
    --task_id "$TASK_ID" \
    --n_demos "$N_DEMOS" \
    --gpu_id 0 \
    "${OUT_ARGS[@]}"
