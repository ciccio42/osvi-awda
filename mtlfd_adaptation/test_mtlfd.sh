#!/bin/bash
# Rolls out an OSVI-AWDA checkpoint (trained via train_mtlfd.sh) against the live UR5e pick_place
# robosuite sim from Multi-Task-LFD-Training-Framework, conditioned on held-out human_rgb demos.
# See docs/03_test_adaptation.md.
#
# Modeled on Multi-Task-LFD-Training-Framework/bashes/test_mosaic_cond_target_obj.sh.
#
# Usage: sbatch mtlfd_adaptation/test_mtlfd.sh <model_dir> <saved_step> [episodes]

#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --exclude=gnode09,gnode04
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --export=ALL
#SBATCH --job-name=osvi_mtlfd_test
#SBATCH --output=mtlfd_adaptation/slurm_logs/test_%j.out

set -e

MODEL_DIR="$1"
SAVED_STEP="$2"
EPISODES="${3:-4}"

CONDA_ENV=osvi_mtlfd
REPO_ROOT="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"

export MUJOCO_PY_MUJOCO_PATH=/home/rsofnc000/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin:/usr/lib/nvidia
export CUDA_VISIBLE_DEVICES=0
export NUMBA_CUDA_USE_NVIDIA_BINDING=1

mkdir -p "$REPO_ROOT/mtlfd_adaptation/slurm_logs"
cd "$REPO_ROOT"
rm -f ./*.png ./*.jpg 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

srun python -u mtlfd_adaptation/test_mtlfd_rollout.py "$MODEL_DIR" \
    --saved_step "$SAVED_STEP" \
    --episodes "$EPISODES" \
    --task_ids 0 5 10 15 \
    --gpu_id 0 \
    --debug
