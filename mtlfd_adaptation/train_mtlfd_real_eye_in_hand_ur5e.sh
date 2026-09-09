#!/bin/bash
# Real-world finetune launcher: warm-starts pick_place_mtlfd_heldout_0_5_10_15_imgwp_noflip_balanced
# (step 290000) onto real_eye_in_hand_ur5e_pick_place via --init-weights (see
# experiments/pick_place_mtlfd_real_eye_in_hand_ur5e.yaml's header). Same auto-resume-on-timeout
# mechanism as the other launchers (_auto_resume_lib.sh) - a timeout-triggered resubmission uses
# --resume against THIS run's own fresh checkpoint dir, not the sim checkpoint, so --init-weights
# is only needed on the initial submission.

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
#SBATCH --job-name=osvi_mtlfd_real_eye_in_hand_ur5e
#SBATCH --output=mtlfd_adaptation/slurm_logs/train_real_eye_in_hand_ur5e_%j.out

set -e

CONDA_ENV=osvi_mtlfd
REPO_ROOT="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
SAVE_PATH="${SAVE_PATH:-/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/osvi_mtlfd}"
EXPERIMENT_FILE="${EXPERIMENT_FILE:-mtlfd_adaptation/experiments/pick_place_mtlfd_real_eye_in_hand_ur5e.yaml}"
INIT_WEIGHTS="${INIT_WEIGHTS:-/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/osvi_mtlfd/pick_place_mtlfd_heldout_0_5_10_15_imgwp_noflip_balanced/osvi_mtlfd_ckpt-1786711914/model_save-290000.pt}"
WORKERS=8

# auto-resume-on-timeout setup (see _auto_resume_lib.sh) - must match this script's own path and
# --output= pattern above (with %j substituted for the real job id).
LAUNCHER_SCRIPT="mtlfd_adaptation/train_mtlfd_real_eye_in_hand_ur5e.sh"
OUTPUT_LOG="mtlfd_adaptation/slurm_logs/train_real_eye_in_hand_ur5e_${SLURM_JOB_ID}.out"

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

# --init-weights is only meaningful on a fresh (non---resume) launch - _auto_resume_on_timeout's
# resubmission passes --resume instead, which would conflict with also passing --init-weights, so
# only add it here when "$@" doesn't already contain --resume.
INIT_ARGS=()
case " $* " in
    *" --resume "*) ;;
    *) INIT_ARGS=(--init-weights "$INIT_WEIGHTS") ;;
esac

srun python -u mtlfd_adaptation/train_mtlfd.py \
    "$EXPERIMENT_FILE" \
    --save-parent "$SAVE_PATH" \
    --workers $WORKERS \
    "${INIT_ARGS[@]}" \
    "$@" &
TRAIN_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    wait "$TRAIN_PID"
done
