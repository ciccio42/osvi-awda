#!/bin/bash
# Runs osvi-awda's own scripts/train_transformer.py against experiments/pick_place_simple.yaml
# UNMODIFIED (the paper's own setup) - see plan section 4. Modeled on
# mtlfd_adaptation/train_mtlfd.sh's conventions, adapted for the 'awda' env and this different,
# unwrapped training entrypoint (no --workers override, to keep the yaml's own loader_workers: 10
# untouched, since the task requires the paper's config unmodified).

#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --exclude=gnode09
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=07:00:00
#SBATCH --export=ALL
#SBATCH --job-name=awda_pick_place_simple
#SBATCH --output=baseline_train/slurm_logs/train_%j.out

set -e

CONDA_ENV=awda
REPO_ROOT="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
SAVE_PATH="${SAVE_PATH:-/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/osvi_awda_baseline}"
# Positional arg passed to train_transformer.py. Normally the experiment YAML; for a resumed run
# (see run_train_pick_place_simple.py) this is overridden to a checkpoint directory instead -
# Trainer auto-discovers the latest bc_inv_ckpt-*/model_save-*.pt inside it (hem/models/trainer.py).
EXPERIMENT_FILE="${EXPERIMENT_FILE:-experiments/pick_place_simple.yaml}"

export MUJOCO_PY_MUJOCO_PATH=/home/rsofnc000/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin:/usr/lib/nvidia
export EXPERT_DATA=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda/dataset
export PYTHONPATH=$PYTHONPATH:$REPO_ROOT
export NUMBA_CUDA_USE_NVIDIA_BINDING=1   # cheap insurance in case soft_dtw_cuda needs it here too

mkdir -p "$REPO_ROOT/baseline_train/slurm_logs" "$SAVE_PATH"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

srun python -u scripts/train_transformer.py \
    "$EXPERIMENT_FILE" \
    --save-parent "$SAVE_PATH" \
    "$@"
