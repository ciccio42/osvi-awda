#!/bin/bash
# Duplicate of train_pick_place_simple_test14_15.sh, pointed at experiments/
# pick_place_simple_test12_13_14_15.yaml - held-out tasks 12/13/14/15 (whole "Can" object) instead
# of just 14/15, matching the mtlfd-adaptation side's heldout_12_13_14_15 split for a direct
# baseline-vs-adaptation comparison. Also adds the SLURM-native auto-resume-on-timeout mechanism
# (mtlfd_adaptation/_auto_resume_lib.sh) - see train_pick_place_simple_test0_5_10_15.sh's comment
# for why this works unmodified against scripts/train_transformer.py too.

#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --exclude=gnode09,gnode04,gnode03,gnode06,gnode12,gnode10
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=07:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --export=ALL
#SBATCH --job-name=awda_pick_place_test12_13_14_15
#SBATCH --output=baseline_train/slurm_logs/train_test12_13_14_15_%j.out

set -e

CONDA_ENV=awda
REPO_ROOT="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
SAVE_PATH="${SAVE_PATH:-/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/osvi_awda_baseline}"
EXPERIMENT_FILE="${EXPERIMENT_FILE:-experiments/pick_place_simple_test12_13_14_15.yaml}"

# auto-resume-on-timeout setup (see mtlfd_adaptation/_auto_resume_lib.sh) - must match this
# script's own path and --output= pattern above (with %j substituted for the real job id).
LAUNCHER_SCRIPT="baseline_train/train_pick_place_simple_test12_13_14_15.sh"
OUTPUT_LOG="baseline_train/slurm_logs/train_test12_13_14_15_${SLURM_JOB_ID}.out"

export MUJOCO_PY_MUJOCO_PATH=/home/rsofnc000/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin:/usr/lib/nvidia
export EXPERT_DATA=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda/dataset
export PYTHONPATH=$PYTHONPATH:$REPO_ROOT
export NUMBA_CUDA_USE_NVIDIA_BINDING=1

mkdir -p "$REPO_ROOT/baseline_train/slurm_logs" "$SAVE_PATH"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
source mtlfd_adaptation/_auto_resume_lib.sh
trap _auto_resume_on_timeout USR1

srun python -u scripts/train_transformer.py \
    "$EXPERIMENT_FILE" \
    --save-parent "$SAVE_PATH" \
    "$@" &
TRAIN_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    wait "$TRAIN_PID"
done
