#!/bin/bash
# Generic sbatch launcher for scripts/collect_demonstrations.py (unmodified), used for both the
# pilot and full-scale runs of both agents. See plan section 2.
#
# Usage: sbatch [--time=HH:MM:SS] baseline_data/collect_demonstrations.sh <env> <N> <per_task_group> <out_subdir>
#   e.g. sbatch baseline_data/collect_demonstrations.sh PandaPickPlaceDistractor 1600 100 panda
#        sbatch baseline_data/collect_demonstrations.sh SawyerPickPlaceDistractor 64 4 _pilot/sawyer

#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=07:00:00
#SBATCH --job-name=osvi_collect
#SBATCH --output=baseline_data/slurm_logs/collect_%j.out
# Originally planned for defq (CPU-only, no GPU needed for collection) per the plan's section 2,
# but defq's QOS config is broken cluster-side right now (srun/sbatch reject even a bare
# `hostname` on defq with "Invalid qos specification" regardless of account used - not something
# fixable from here). Falling back to gpuq as the plan's own contingency anticipated.

set -e

ENV_NAME="$1"
N="$2"
PER_TASK_GROUP="$3"
OUT_SUBDIR="$4"

CONDA_ENV=awda
REPO_ROOT="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
OUT_DIR="$REPO_ROOT/dataset/$OUT_SUBDIR"

mkdir -p "$OUT_DIR" "$REPO_ROOT/baseline_data/slurm_logs"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export MUJOCO_PY_MUJOCO_PATH=/home/rsofnc000/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=$PYTHONPATH:$REPO_ROOT

echo "Collecting env=$ENV_NAME N=$N per_task_group=$PER_TASK_GROUP -> $OUT_DIR"
srun python scripts/collect_demonstrations.py --env "$ENV_NAME" \
    --num_workers 30 --N "$N" --collect_cam --per_task_group "$PER_TASK_GROUP" --n_env 800 \
    "$OUT_DIR"

echo "Done. File count: $(ls "$OUT_DIR"/*.pkl 2>/dev/null | wc -l)"
