#!/bin/bash
# Trains OSVI-AWDA (attributed waypoints + SDTW + ADM) on the Multi-Task-LFD-Training-Framework's
# human_rgb_pick_place / ur5e_pick_place dataset. See docs/02_training_adaptation.md.
#
# Modeled on Multi-Task-LFD-Training-Framework/bashes/train_mosaic_target_obj_detector_double_policy.sh
# (SBATCH resources, MUJOCO/LD_LIBRARY_PATH env vars).

#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --exclude=gnode09
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=07:00:00
#SBATCH --export=ALL
#SBATCH --job-name=osvi_mtlfd_train
#SBATCH --output=mtlfd_adaptation/slurm_logs/train_%j.out

set -e

CONDA_ENV=osvi_mtlfd   # isolated clone with a numba/cuda-python fix for soft_dtw_cuda.py - see docs/02
REPO_ROOT="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
SAVE_PATH="${SAVE_PATH:-/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/osvi_mtlfd}"
# Positional arg passed to train_mtlfd.py. Normally the experiment YAML; for a resumed run
# (see run_train_mtlfd.py) this is overridden to the checkpoint directory instead - Trainer
# auto-discovers the latest model_save-*.pt inside it once --resume is also passed via "$@".
EXPERIMENT_FILE="${EXPERIMENT_FILE:-mtlfd_adaptation/experiments/pick_place_mtlfd.yaml}"
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
