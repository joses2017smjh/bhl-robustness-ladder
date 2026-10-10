#!/bin/bash
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint=a40|rtx8000|l40s|h100|h200|h200-70g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=waiter-dsrl
#SBATCH --output=%x-%j.log
#
# Waiter phase 4 DSRL launcher: one GPU (frozen VLA + SAC) + N CPU env workers on the same node.
#   sbatch [--time=HH:MM:SS --cpus-per-task=K] dsrl_gpu.sbatch train <train_dsrl.py args...>
#   sbatch dsrl_gpu.sbatch eval  <eval_dsrl.py args...>
#   sbatch dsrl_gpu.sbatch check <smolvla_noise.py args...>
#   sbatch dsrl_gpu.sbatch trace <trace_base.py args...>      (dev diagnostic: per-decision object poses)
#   sbatch dsrl_gpu.sbatch probe <probe_features.py args...>  (dev diagnostic: what the SAC observation encodes)
# Give --n-envs <= cpus-per-task - 2. --env-python is added automatically for train/eval.
# For runs longer than the time limit pass --max-wall-min (a few minutes under --time) and resubmit with --resume.
#
# Interpreters (no installs into existing venvs):
#   learner: venv-lehome-train (lerobot 0.4.3, torch 2.7.0+cu128, numpy 2.4)  -> DSRL_LEARNER_PY
#   env workers: venv (mujoco 3.3.5, onnxruntime, numpy 1.26; the phase 1/2 MuJoCo stack) -> DSRL_ENV_PY
set -euo pipefail
WS=/nfs/hpc/share/sanchej7/Humanoid_Lite
REPO=${DSRL_REPO:-$WS/bhl-robustness-ladder}
LEARNER_PY=${DSRL_LEARNER_PY:-$WS/venv-lehome-train/bin/python}
ENV_PY=${DSRL_ENV_PY:-$WS/venv/bin/python}
MODE=${1:?usage: dsrl_gpu.sbatch train|eval|check|trace|probe args...}
shift

export TMPDIR=/tmp
export XDG_CACHE_HOME=$WS/.cache
export HF_HOME=$XDG_CACHE_HOME/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$REPO/src
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export PYTHONUNBUFFERED=1

echo "[dsrl-sbatch] job=${SLURM_JOB_ID:-} host=$(hostname) mode=$MODE repo=$REPO"
echo "[dsrl-sbatch] git=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) dirty_files=$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# The cu128 torch wheel has no SM70 kernels: fail in seconds, not after loading the VLA.
"$LEARNER_PY" -c "import torch; torch.ones(1, device='cuda').add_(1); torch.cuda.synchronize(); print('[dsrl-sbatch] cuda ok', torch.cuda.get_device_name())"

D=$REPO/scripts/waiter/dsrl
case "$MODE" in
  train) exec "$LEARNER_PY" "$D/train_dsrl.py" --env-python "$ENV_PY" "$@" ;;
  eval)  exec "$LEARNER_PY" "$D/eval_dsrl.py" --env-python "$ENV_PY" "$@" ;;
  check) exec "$LEARNER_PY" "$D/smolvla_noise.py" "$@" ;;
  trace) exec "$LEARNER_PY" "$D/trace_base.py" --env-python "$ENV_PY" "$@" ;;
  probe) exec "$LEARNER_PY" "$D/probe_features.py" --env-python "$ENV_PY" "$@" ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
