#!/bin/bash
# Duplicate of train_mtlfd.sh, pointed at pick_place_mtlfd_heldout_0_5_10_15.yaml (comparison run
# B: tasks 0,5,10,15 held out of training). Same SAVE_PATH parent - checkpoint subfolder
# auto-derives from the experiment file's basename, landing in
# .../osvi_mtlfd/pick_place_mtlfd_heldout_0_5_10_15/bc_inv_ckpt-<ts>/.

#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --exclude=gnode09
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=07:00:00
#SBATCH --export=ALL
#SBATCH --job-name=osvi_mtlfd_heldout
#SBATCH --output=mtlfd_adaptation/slurm_logs/train_heldout_%j.out

set -e

CONDA_ENV=osvi_mtlfd
REPO_ROOT="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
SAVE_PATH="${SAVE_PATH:-/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/osvi_mtlfd}"
EXPERIMENT_FILE="${EXPERIMENT_FILE:-mtlfd_adaptation/experiments/pick_place_mtlfd_heldout_0_5_10_15.yaml}"
WORKERS=8

export MUJOCO_PY_MUJOCO_PATH=/home/rsofnc000/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin:/usr/lib/nvidia
export EXPERT_DATA=/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset
export NUMBA_CUDA_USE_NVIDIA_BINDING=1

mkdir -p "$REPO_ROOT/mtlfd_adaptation/slurm_logs" "$SAVE_PATH"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

srun python -u mtlfd_adaptation/train_mtlfd.py \
    "$EXPERIMENT_FILE" \
    --save-parent "$SAVE_PATH" \
    --workers $WORKERS \
    "$@"
