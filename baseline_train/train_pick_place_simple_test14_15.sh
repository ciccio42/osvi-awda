#!/bin/bash
# Duplicate of train_pick_place_simple.sh, pointed at the paper-faithful config
# (experiments/pick_place_simple_test14_15.yaml: 14 train / 2 test tasks, batches: 500000 -
# matches 2302.04856v1.pdf's own reported protocol, distinct from the unmodified-yaml campaign
# already running under bc_inv_ckpt-1784458550). Same SAVE_PATH parent as the original - the
# checkpoint subfolder is auto-derived from the experiment file's basename
# (hem/models/trainer.py), so it lands in .../pick_place_simple_test14_15/bc_inv_ckpt-<ts>/,
# naturally encoding the held-out task ids in the path.

#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --exclude=gnode09
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=07:00:00
#SBATCH --export=ALL
#SBATCH --job-name=awda_pick_place_test14_15
#SBATCH --output=baseline_train/slurm_logs/train_test14_15_%j.out

set -e

CONDA_ENV=awda
REPO_ROOT="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
SAVE_PATH="${SAVE_PATH:-/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/osvi_awda_baseline}"
EXPERIMENT_FILE="${EXPERIMENT_FILE:-experiments/pick_place_simple_test14_15.yaml}"

export MUJOCO_PY_MUJOCO_PATH=/home/rsofnc000/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin:/usr/lib/nvidia
export EXPERT_DATA=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda/dataset
export PYTHONPATH=$PYTHONPATH:$REPO_ROOT
export NUMBA_CUDA_USE_NVIDIA_BINDING=1

mkdir -p "$REPO_ROOT/baseline_train/slurm_logs" "$SAVE_PATH"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

srun python -u scripts/train_transformer.py \
    "$EXPERIMENT_FILE" \
    --save-parent "$SAVE_PATH" \
    "$@"
