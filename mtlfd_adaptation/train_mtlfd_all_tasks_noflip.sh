#!/bin/bash
# Duplicate of train_mtlfd_all_tasks.sh, pointed at pick_place_mtlfd_all_tasks_noflip.yaml -
# ablation run: rand_flip: False + mtlfd_dataset.py's AGENT_SETTLE_STEPS fix (module-level,
# applies regardless of yaml), validated first via a short smoke run (pick_place_mtlfd_smoke_noflip
# .yaml) before committing to this full production run. Same SAVE_PATH parent as the other
# production runs - checkpoint subfolder auto-derives from the experiment file's basename, landing
# in .../osvi_mtlfd/pick_place_mtlfd_all_tasks_noflip/osvi_mtlfd_ckpt-<ts>/.
#
# Deliberately does NOT pass --debug (unlike a since-fixed version of train_mtlfd_all_tasks.sh) -
# that flag makes train_mtlfd.py block on debugpy.wait_for_client() before anything else runs,
# which silently stalls an unattended sbatch job for its entire wall-clock limit with zero
# training progress. Confirmed this is what happened to the last pick_place_mtlfd_all_tasks
# production attempt (its checkpoint dir never got past initial setup files).

#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --exclude=gnode09,gnode04,gnode03,gnode06,gnode12,gnode10
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=07:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --export=ALL
#SBATCH --job-name=osvi_mtlfd_all_tasks_noflip
#SBATCH --output=mtlfd_adaptation/slurm_logs/train_all_tasks_noflip_%j.out

set -e

CONDA_ENV=osvi_mtlfd
REPO_ROOT="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
SAVE_PATH="${SAVE_PATH:-/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/osvi_mtlfd}"
EXPERIMENT_FILE="${EXPERIMENT_FILE:-mtlfd_adaptation/experiments/pick_place_mtlfd_all_tasks_noflip.yaml}"
WORKERS=8

# auto-resume-on-timeout setup (see _auto_resume_lib.sh) - must match this script's own path and
# --output= pattern above (with %j substituted for the real job id).
LAUNCHER_SCRIPT="mtlfd_adaptation/train_mtlfd_all_tasks_noflip.sh"
OUTPUT_LOG="mtlfd_adaptation/slurm_logs/train_all_tasks_noflip_${SLURM_JOB_ID}.out"

export MUJOCO_PY_MUJOCO_PATH=/home/rsofnc000/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin:/usr/lib/nvidia
export EXPERT_DATA=/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset
export NUMBA_CUDA_USE_NVIDIA_BINDING=1

mkdir -p "$REPO_ROOT/mtlfd_adaptation/slurm_logs" "$SAVE_PATH"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
source mtlfd_adaptation/_auto_resume_lib.sh
trap _auto_resume_on_timeout USR1

srun python -u mtlfd_adaptation/train_mtlfd.py \
    "$EXPERIMENT_FILE" \
    --save-parent "$SAVE_PATH" \
    --workers $WORKERS \
    "$@" &
TRAIN_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    wait "$TRAIN_PID"
done
