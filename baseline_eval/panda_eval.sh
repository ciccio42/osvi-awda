#!/bin/bash
# Runs baseline_eval/panda_video_heatmap.py (video/heatmap/trajectory-plot generation for
# collected panda trajectories) - see plan section 3. Doesn't need mujoco_py/torch/GPU, just
# pickle + cv2/matplotlib/imageio, so it's CPU-only.
#
# Usage: sbatch baseline_eval/panda_eval.sh [--dataset_dir DIR] [--out_dir DIR] [--limit N] ...
# (any extra args are forwarded to panda_video_heatmap.py)

#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH --job-name=osvi_panda_eval
#SBATCH --output=baseline_eval/slurm_logs/eval_%j.out
# Planned for defq (this doesn't need a GPU at all) per the plan's section 3, but defq's QOS
# config is broken cluster-side right now - see collect_demonstrations.sh's header comment for
# the same issue. Uses gpuq WITHOUT --gres=gpu so it only consumes CPU cores, no GPU allocated.

set -e

REPO_ROOT="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
mkdir -p "$REPO_ROOT/baseline_eval/slurm_logs"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate awda
export PYTHONPATH=$PYTHONPATH:$REPO_ROOT

srun python -m baseline_eval.panda_video_heatmap --num_workers 32 "$@"
