#!/bin/bash
# Launcher for target_localization_test.py - see that file's docstring.
# Usage: MODEL_DIR=... SAVED_STEP=... [SPLITS="train valid test"] [EPISODES_PER_TASK=10] \
#        sbatch mtlfd_adaptation/run_target_localization_test.sh

#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --exclude=gnode09,gnode04,gnode03,gnode06,gnode12,gnode10
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --export=ALL
#SBATCH --job-name=target_loc_test
#SBATCH --output=mtlfd_adaptation/slurm_logs/target_loc_test_%j.out

set -e

CONDA_ENV=osvi_mtlfd
REPO_ROOT="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR}"
SAVED_STEP="${SAVED_STEP:?set SAVED_STEP}"
SPLITS="${SPLITS:-train valid test}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
# 0 = train_mtlfd.py's own --seed default, which is what this checkpoint was actually trained with
# (no --seed override in its args.txt) - keep this FIXED across every checkpoint-step invocation so
# train/valid draw the identical (task, episode) -> (demo_file, agent_file, env-reset-seed) sequence
# regardless of which step's weights are loaded, making step-to-step comparisons apples-to-apples.
SEED="${SEED:-0}"
OUT_DIR="${OUT_DIR:-}"

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
if [ -n "$OUT_DIR" ]; then
    EXTRA_ARGS+=(--out_dir "$OUT_DIR")
fi

srun python -u mtlfd_adaptation/target_localization_test.py "$MODEL_DIR" \
    --saved_step "$SAVED_STEP" \
    --splits $SPLITS \
    --episodes_per_task "$EPISODES_PER_TASK" \
    --seed "$SEED" \
    --gpu_id 0 \
    "${EXTRA_ARGS[@]}"
