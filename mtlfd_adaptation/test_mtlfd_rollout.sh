#!/bin/bash
# Rolls out an OSVI-AWDA/MTLFD checkpoint (trained via train_mtlfd*.sh) against the live UR5e
# pick_place robosuite sim, via test_mtlfd_rollout.py. Generalizes the older test_mtlfd.sh (which
# hardcoded --task_ids 12 13 14 15): lets you pick episode count and either an explicit task-id
# list or --split train/test to auto-pull the task list from the checkpoint's own config.yaml
# (dataset.train_tasks / dataset.test_tasks), so you don't have to remember which tasks a given
# checkpoint was trained/held-out on.
#
# Usage:
#   sbatch mtlfd_adaptation/test_mtlfd_rollout.sh <model_dir> <saved_step> \
#       [--episodes N] [--split train|test] [--task-ids "1 2 3 ..."] [--results-dir DIR]
#
# --split train (default) or --split test resolve task_ids from config.yaml automatically.
# --task-ids overrides both and is used verbatim (space-separated, unquoted-at-call-site is fine
#   since sbatch/bash will pass it through as one argument if you quote it here).
# --results-dir overrides the default results_pick_place/step-<N>-<split>tasks output location
#   (the split suffix keeps a --split train run and a --split test run of the SAME checkpoint/step
#   from overwriting each other's summary.json).
#
# Examples:
#   sbatch mtlfd_adaptation/test_mtlfd_rollout.sh "$CKPT_DIR" 200000 --split train --episodes 4
#   sbatch mtlfd_adaptation/test_mtlfd_rollout.sh "$CKPT_DIR" 200000 --task-ids "0 5 10 15" --episodes 10

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

MODEL_DIR="$1"; shift
SAVED_STEP="$1"; shift

EPISODES=4
SPLIT=train
TASK_IDS=""
RESULTS_DIR=""

while [ $# -gt 0 ]; do
    case "$1" in
        --episodes) EPISODES="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --task-ids) TASK_IDS="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ "$SPLIT" != "train" ] && [ "$SPLIT" != "test" ]; then
    echo "--split must be 'train' or 'test' (got '$SPLIT')" >&2
    exit 1
fi

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

if [ -z "$TASK_IDS" ]; then
    TASK_IDS=$(python -c "
import yaml
with open('$MODEL_DIR/config.yaml') as f:
    cfg = yaml.safe_load(f)
tasks = cfg['dataset']['${SPLIT}_tasks']
print(' '.join(str(t) for t in tasks))
")
    echo "[test_mtlfd_rollout] auto-resolved --split $SPLIT -> task_ids: $TASK_IDS"
fi

if [ -z "$RESULTS_DIR" ]; then
    RESULTS_DIR="$MODEL_DIR/results_pick_place/step-${SAVED_STEP}-${SPLIT}tasks"
fi

echo "[test_mtlfd_rollout] model_dir=$MODEL_DIR saved_step=$SAVED_STEP episodes=$EPISODES" \
     "task_ids=[$TASK_IDS] results_dir=$RESULTS_DIR"

srun python -u mtlfd_adaptation/test_mtlfd_rollout.py "$MODEL_DIR" \
    --saved_step "$SAVED_STEP" \
    --episodes "$EPISODES" \
    --task_ids $TASK_IDS \
    --results_dir "$RESULTS_DIR" \
    --gpu_id 0
