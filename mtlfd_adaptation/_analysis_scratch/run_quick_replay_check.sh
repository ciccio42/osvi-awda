#!/bin/bash
# Quick one-off check: run a single episode of test_mtlfd_rollout.py with --replay_training_scenes
# to confirm training_scene_frame.png (the training pkl's own stored frame, saved by
# set_objects_from_training_trajectory's debug_path) actually gets written correctly, before
# committing to the full multi-episode/multi-task rollout.
#
# Usage: MODEL_DIR=... SAVED_STEP=... TASK_ID=12 EPISODES=10 sbatch mtlfd_adaptation/_analysis_scratch/run_quick_replay_check.sh

#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --exclude=gnode09,gnode04,gnode03,gnode06,gnode12,gnode10
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --export=ALL
#SBATCH --job-name=quick_replay_check
#SBATCH --output=mtlfd_adaptation/slurm_logs/quick_replay_check_%j.out

set -e

CONDA_ENV=osvi_mtlfd
REPO_ROOT="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR}"
SAVED_STEP="${SAVED_STEP:?set SAVED_STEP}"
TASK_ID="${TASK_ID:-12}"
EPISODES="${EPISODES:-1}"
TRAINING_PAIRS="${TRAINING_PAIRS:-0}"
SEEDS_FILE="${SEEDS_FILE:-}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/mtlfd_adaptation/_analysis_scratch/quick_replay_check}"

EXTRA_ARGS=(--replay_training_scenes)
if [ "$TRAINING_PAIRS" = "1" ]; then
    EXTRA_ARGS=(--training_pairs)
fi
if [ -n "$SEEDS_FILE" ]; then
    EXTRA_ARGS+=(--seeds_file "$SEEDS_FILE")
fi

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
    --task_ids "$TASK_ID" \
    --results_dir "$RESULTS_DIR" \
    --gpu_id 0 \
    "${EXTRA_ARGS[@]}"
