#!/bin/bash
# Ablation of train_mtlfd_heldout_0_5_10_15_imgwp_noflip.sh: samples_per_task: 2 (exact per-batch
# task balance, see experiments/pick_place_mtlfd_heldout_0_5_10_15_imgwp_noflip_balanced.yaml's
# note). Same auto-resume-on-timeout mechanism as the other launchers (_auto_resume_lib.sh).

#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --exclude=gnode09,gnode04,gnode03,gnode06,gnode12,gnode10
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=07:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --export=ALL
#SBATCH --job-name=osvi_mtlfd_heldout_0_5_10_15_balanced
#SBATCH --output=mtlfd_adaptation/slurm_logs/train_heldout_0_5_10_15_balanced_%j.out

set -e

CONDA_ENV=osvi_mtlfd
REPO_ROOT="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
SAVE_PATH="${SAVE_PATH:-/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/osvi_mtlfd}"
EXPERIMENT_FILE="${EXPERIMENT_FILE:-mtlfd_adaptation/experiments/pick_place_mtlfd_heldout_0_5_10_15_imgwp_noflip_balanced.yaml}"
WORKERS=8

# auto-resume-on-timeout setup (see _auto_resume_lib.sh) - must match this script's own path and
# --output= pattern above (with %j substituted for the real job id).
LAUNCHER_SCRIPT="mtlfd_adaptation/train_mtlfd_heldout_0_5_10_15_imgwp_noflip_balanced.sh"
OUTPUT_LOG="mtlfd_adaptation/slurm_logs/train_heldout_0_5_10_15_balanced_${SLURM_JOB_ID}.out"

export MUJOCO_PY_MUJOCO_PATH=/home/rsofnc000/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin:/usr/lib/nvidia
export EXPERT_DATA=/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset
export NUMBA_CUDA_USE_NVIDIA_BINDING=1

mkdir -p "$REPO_ROOT/mtlfd_adaptation/slurm_logs" "$SAVE_PATH"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
source mtlfd_adaptation/_auto_resume_lib.sh
trap _auto_resume_on_timeout USR1

srun python -u mtlfd_adaptation/train_mtlfd.py \
    "$EXPERIMENT_FILE" \
    --save-parent "$SAVE_PATH" \
    --workers $WORKERS \
    "$@" &
TRAIN_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    wait "$TRAIN_PID"
done
