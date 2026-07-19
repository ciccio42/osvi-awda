#!/bin/bash
# Builds the 'awda' conda env per osvi-awda's own README.md, plus deviations documented in
# /home/rsofnc000/.claude/plans/sequential-wibbling-penguin.md section 1 (torch needs an explicit
# +cu117 wheel; mujoco-py isn't in requirements.txt and reuses the existing mujoco210 install;
# a couple of requirements.txt lines are unused-in-repo and dropped only if they fail to install).
# Then runs the smoke tests: soft_dtw_cuda.py (numba/CUDA - the known risk from the earlier
# mtlfd_adaptation work), bulk imports, and a 2-trajectory collect_demonstrations.py GPU/EGL run.
#
# Run via: srun -A did_robot_learning_359 --partition=gpuq --gres=gpu:1 --cpus-per-task=8 --mem=32G
#   --time=02:00:00 bash baseline_data/setup_awda_env.sh
# Idempotent/re-runnable: skips conda-env creation if 'awda' already exists, so a partial failure
# can be fixed and re-run without re-downloading torch's ~1.8GB wheel etc.
set -o pipefail   # no -e / no -u: fallback branches below intentionally handle failures themselves,
                  # and -u trips on legitimately-unset vars like $PYTHONPATH before first export

REPO_ROOT="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda"
cd "$REPO_ROOT"

echo "=== [1/6] create conda env 'awda' (python 3.7.4) ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | grep -qE '^awda\s'; then
    echo "awda env already exists, skipping creation"
else
    conda create -n awda python=3.7.4 -y
fi
conda activate awda
python --version
pip install --upgrade pip
# gym==0.19.0 (needed - pyutil.py/metaworld import it directly) fails 'setup.py egg_info' under
# modern setuptools' stricter extras_require validation (setuptools>=66). Pin an older
# setuptools/wheel first, matching the standard fix for this exact class of old-package failure.
pip install "setuptools<66" "wheel<0.38"

echo "=== [2/6] install requirements.txt ==="
if ! pip install -r requirements.txt; then
    echo "unfiltered install failed - retrying with sklearn/mpi4py/dm-control dropped (unused in this repo)"
    grep -vE '^(sklearn$|mpi4py==|dm-control==)' requirements.txt > /tmp/awda_requirements_filtered.txt
    pip install -r /tmp/awda_requirements_filtered.txt
fi

echo "=== [3/6] install torch+cu117 ==="
if python -c "import torch" 2>/dev/null; then
    echo "torch already installed, skipping"
else
    pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html \
      || pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
fi
python -c "import torch; print('TORCH_OK', torch.__version__, 'cuda_available', torch.cuda.is_available())"

echo "=== [4/6] install mujoco-py (reusing existing mujoco210) ==="
# mujoco-py's EGL C-extension needs X11/mesa headers that aren't present system-wide on compute
# nodes (only on login nodes, it turns out - checked via dpkg). The sibling 'osvi_mtlfd' env
# builds this exact mujoco-py version successfully because it has gcc_linux-64 + xorg-*/mesa-*
# conda-forge packages, whose activate.d hooks point CC/CFLAGS at $CONDA_PREFIX/include instead of
# system headers - reproducing that exact package set here rather than guessing at a subset.
# libxcrypt: the conda sysroot's own crypt.h is missing, which CPython 3.7's own Python.h pulls in
# unconditionally on this build - needed for ANY C-extension build in this env, not just mujoco-py.
# gcc_linux-64/sysroot_linux-64 are pinned to exactly osvi_mtlfd's proven-working versions: an
# unpinned install grabbed gcc 15.2.0, and GCC 14+ turned -Wincompatible-pointer-types into a hard
# error by default (previously just a warning) - mujoco_py 2.1.2.14's cymj.c trips this and fails
# to compile under 15.2.0, but built fine under 12.1.0 in the sibling env.
conda install -y -c conda-forge "gcc_linux-64=12.1.0" "gxx_linux-64=12.1.0" "sysroot_linux-64=2.12" \
    libxcrypt \
    xorg-libx11 xorg-libxext xorg-libxrender xorg-libxdamage xorg-libxfixes xorg-libxrandr \
    xorg-damageproto xorg-fixesproto xorg-glproto xorg-kbproto xorg-randrproto xorg-renderproto \
    xorg-util-macros xorg-xextproto xorg-xf86vidmodeproto xorg-xproto mesalib libglu \
    libx11-cos6-x86_64 libx11-common-cos6-x86_64 mesa-libegl-cos6-x86_64 mesa-libgl-cos6-x86_64
# conda activate again to pick up the new activate.d hooks (CC/CFLAGS) from the packages just installed
conda deactivate
conda activate awda

export MUJOCO_PY_MUJOCO_PATH=/home/rsofnc000/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin:/usr/lib/nvidia
pip install mujoco-py==2.1.2.14
# mujoco-py 2.1.2.14's cymj.pyx fails to compile under modern Cython (3.x tightened exception-spec
# checking on 'except *' callbacks). Pin an older Cython before the lazy first-import build.
pip install "Cython<3"
python -c "import mujoco_py; print('MUJOCO_PY_OK', mujoco_py.__file__)"

echo "=== [5/6] smoke test: soft_dtw_cuda.py (numba/CUDA) ==="
export PYTHONPATH=$PYTHONPATH:$REPO_ROOT
# soft_dtw_cuda.py needs TWO separate fixes on this cluster under python 3.7 (root-caused by
# reading numba/cuda/cuda_paths.py + cudadrv/driver.py directly, not guessed):
#
# 1. Driver-API ABI segfault: numba's legacy ctypes CUDA driver bindings segfault against this
#    cluster's driver. Fix: NUMBA_CUDA_USE_NVIDIA_BINDING=1 + a cp37-installable cuda-python
#    build. cuda-python==12.6.*/numba==0.57.1 (what fixed this under python 3.9 in osvi_mtlfd -
#    see docs/02_training_adaptation.md §7) have no cp37 wheels; 12.8.0's cp37 wheel exists but
#    its cuda-bindings~=12.8.0 dependency doesn't, so use 12.0.0 (last fully cp37-installable).
#
# 2. NVVM IR-version mismatch (separate from #1, and NOT fixed by #1 alone): numba's
#    get_cuda_paths() checks 'Conda environment' (nvvm found directly under $CONDA_PREFIX/lib)
#    BEFORE 'CUDA_HOME' - but with nothing there, it falls through to CUDA_HOME, which this
#    cluster sets to the CUDA 12.8 toolkit regardless of which cuda-python pip package is
#    installed (pip-installed cuda-python is bindings-only, it doesn't bundle its own nvvm).
#    That 12.8 nvvm emits/expects IR version 2.0, but numba's python-3.7 ceiling (<=0.56.4) pairs
#    with an older llvmlite that only emits IR 1.6 -> NvvmError: incompatible IR detected. Fix:
#    install nvidia-cuda-nvcc-cu11 (a standalone pip package bundling CUDA 11.8's nvvm/libdevice,
#    no system toolkit needed) and place it directly under $CONDA_PREFIX/lib so 'Conda
#    environment' finds it first, entirely bypassing the system's mismatched CUDA_HOME nvvm.
SDTW_TEST='
import torch
from soft_dtw_cuda import SoftDTW
a = torch.randn(4, 20, 8, requires_grad=True, device="cuda")
b = torch.randn(4, 25, 8, device="cuda")
loss = SoftDTW(use_cuda=True, gamma=0.1)(a, b).mean()
loss.backward()
print("SOFT_DTW_OK loss=", loss.item(), "grad_ok=", a.grad is not None)
'
python -c "$SDTW_TEST"
if [ $? -ne 0 ]; then
    echo "soft_dtw_cuda failed - applying both fixes."
    export NUMBA_CUDA_USE_NVIDIA_BINDING=1
    pip install "cuda-python==12.0.0"
    pip install "nvidia-cuda-nvcc-cu11==11.8.89"
    NVCC_PKG_DIR=$(python -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))")
    ln -sf "$NVCC_PKG_DIR/nvvm/lib64/libnvvm.so" "$CONDA_PREFIX/lib/libnvvm.so.11.8"
    ln -sf "$NVCC_PKG_DIR/nvvm/libdevice/libdevice.10.bc" "$CONDA_PREFIX/lib/libdevice.10.bc"
    python -c "$SDTW_TEST"
fi

echo "=== [6/6] bulk import check + collect_demonstrations.py GPU/EGL smoke run ==="
python -c "import torch, torchvision, cv2, numba, pybullet, mujoco_py, h5py, matplotlib, imageio; print('BULK_IMPORT_OK')"

# .gitignore's blanket *.png rule strips all MuJoCo texture assets from this repo - see
# backfill_textures.py's docstring. Needed before any env that loads the bin arena / distractor
# objects (i.e. before this smoke test, and before real data collection) can construct at all.
python baseline_data/backfill_textures.py

mkdir -p dataset/_smoke
rm -rf dataset/_smoke/panda_gpu
PYTHONPATH=. python scripts/collect_demonstrations.py --env PandaPickPlaceDistractor \
  --num_workers 1 --N 2 --collect_cam --per_task_group 2 --n_env 2 \
  dataset/_smoke/panda_gpu
echo "GPU smoke collection output:"
ls -la dataset/_smoke/panda_gpu/

echo "=== DONE ==="
