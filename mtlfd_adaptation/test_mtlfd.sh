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
#SBATCH --exclude=gnode09,gnode04,gnode03,gnode06,gnode12,gnode10
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
EPISODES="${3:-10}"
# set REPLAY_TRAINING_SCENES=1 to teleport each episode's objects to match a random real training
# trajectory's layout instead of the env's own random reset (set_objects_from_training_trajectory,
# --replay_training_scenes) - isolates live-rendering effects from unseen-layout effects.
REPLAY_TRAINING_SCENES="${REPLAY_TRAINING_SCENES:-0}"
# set TRAINING_PAIRS=1 to sample demo_file/training_agent_file as a genuine (demo, agent) couple
# from the checkpoint's own mode='train' dataset pairs (--training_pairs) instead of picking each
# independently at random - implies REPLAY_TRAINING_SCENES.
TRAINING_PAIRS="${TRAINING_PAIRS:-0}"
# set SEEDS_FILE=/path/to/seeds.txt to replay a fixed, pre-generated list of per-episode seeds
# (--seeds_file) instead of deriving them fresh from --seed.
SEEDS_FILE="${SEEDS_FILE:-}"
RESULTS_DIR="${RESULTS_DIR:-}"

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

EXTRA_ARGS=()
if [ "$REPLAY_TRAINING_SCENES" = "1" ]; then
    EXTRA_ARGS+=(--replay_training_scenes)
fi
if [ "$TRAINING_PAIRS" = "1" ]; then
    EXTRA_ARGS+=(--training_pairs)
fi
if [ -n "$RESULTS_DIR" ]; then
    EXTRA_ARGS+=(--results_dir "$RESULTS_DIR")
fi
if [ -n "$SEEDS_FILE" ]; then
    EXTRA_ARGS+=(--seeds_file "$SEEDS_FILE")
fi

srun python -u mtlfd_adaptation/test_mtlfd_rollout.py "$MODEL_DIR" \
    --saved_step "$SAVED_STEP" \
    --episodes "$EPISODES" \
    --task_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
    --gpu_id 0 \
    "${EXTRA_ARGS[@]}"
