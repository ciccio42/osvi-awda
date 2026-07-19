#!/bin/bash
# Validates the CPU-only / OSMesa offscreen-render path for mujoco-py, since data collection is
# planned to run on defq (no GPU) - see plan section 1, smoke test 4. Run only after
# setup_awda_env.sh has successfully built the 'awda' env.
#
# Run via: srun -A did_robot_learning_359 --partition=defq --cpus-per-task=4 --mem=8G
#   --time=00:20:00 bash baseline_data/smoke_test_defq.sh
set -uo pipefail

REPO_ROOT="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate awda
export MUJOCO_PY_MUJOCO_PATH=/home/rsofnc000/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export PYTHONPATH=$PYTHONPATH:$REPO_ROOT

mkdir -p dataset/_smoke
rm -rf dataset/_smoke/panda_cpu
PYTHONPATH=. python scripts/collect_demonstrations.py --env PandaPickPlaceDistractor \
  --num_workers 2 --N 2 --collect_cam --per_task_group 2 --n_env 2 \
  dataset/_smoke/panda_cpu
STATUS=$?
echo "CPU/OSMesa smoke collection exit status: $STATUS"
ls -la dataset/_smoke/panda_cpu/
if [ $STATUS -ne 0 ]; then
    echo "CPU_OSMESA_FAILED - fall back to running real data collection on gpuq --gres=gpu:1"
else
    echo "CPU_OSMESA_OK"
fi
