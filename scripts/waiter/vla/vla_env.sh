#!/bin/bash
# Shared environment for waiter phase 3 jobs. Source it (needs ROOT = repo/worktree root); do not execute.
#
# Interpreters:
#   VLA_PY  venv-lehome-train: Python 3.11, lerobot 0.4.3, torch 2.7.0+cu128 (no SM70/V100 kernels), transformers
#           4.57.6, numpy 2.4. No MuJoCo.
#   ENV_PY  main venv: MuJoCo 3.3.5, numpy 1.26, torch 2.7.0+cu128. Runs env workers (env_worker.py) for the
#           phase 2 WaiterEnv; rollout_vla.py talks to them over pipes without pickling numpy arrays.
# Weights: lerobot/smolvla_base + SmolVLM2 tokenizer come from the shared HF cache, strictly offline.
: "${ROOT:?ROOT must point at the repo/worktree root}"
export TMPDIR=/tmp
export HF_HOME=/nfs/hpc/share/sanchej7/Humanoid_Lite/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
VLA_PY=${VLA_PY:-/nfs/hpc/share/sanchej7/Humanoid_Lite/venv-lehome-train/bin/python}
ENV_PY=${ENV_PY:-/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python}
VLA_DIR="$ROOT/scripts/waiter/vla"
# Node-local memmap cache (uncompressed frames; ~150 KB per frame). Removed at job exit unless VLA_CACHE was given.
if [ -z "${VLA_CACHE:-}" ]; then
    VLA_CACHE=/tmp/waiter-vla-cache-${SLURM_JOB_ID:-$$}
    VLA_CACHE_OWNED=1
fi
mkdir -p "$VLA_CACHE"

vla_cleanup() {
    if [ "${VLA_CACHE_OWNED:-0}" = 1 ]; then rm -rf "$VLA_CACHE"; fi
}

vla_banner() {
    echo "[vla] host=$(hostname) job=${SLURM_JOB_ID:-none} root=$ROOT"
    echo "[vla] git=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null) dirty=$(git -C "$ROOT" status --porcelain 2>/dev/null | wc -l)"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/[vla] gpu: /'
    "$VLA_PY" -c "import torch,lerobot,transformers,numpy;print('[vla] torch',torch.__version__,'cuda',torch.version.cuda,'available',torch.cuda.is_available(),'lerobot',getattr(lerobot,'__version__','?'),'transformers',transformers.__version__,'numpy',numpy.__version__)"
    echo "[vla] cache=$VLA_CACHE free=$(df -h --output=avail /tmp | tail -1)"
}
