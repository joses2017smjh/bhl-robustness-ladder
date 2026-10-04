#!/bin/bash
# Run something from this repo on a compute node, the way that works on this cluster.
#   run.sh gpu  <driver.py args>     driver on a GPU node with EGL (needed for --png screenshots)
#   run.sh cpu  <driver.py args>     driver on a CPU node (no rendering available there)
#   run.sh test <pytest args>        pytest on a CPU node; log + pytest's own exit code -> $BHL_LOG
#   run.sh py   <python args>        any repo Python on a CPU node (PYTHONPATH=src)
# Never runs heavy Python on the interactive desktop node. Paths are resolved from this file, so any cwd works.
set -euo pipefail
SKILL=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SKILL/../../.." && pwd)
PY=/nfs/hpc/share/$USER/Humanoid_Lite/venv/bin/python
mode=${1:-}; shift || true
[ -n "$mode" ] || { sed -n '2,7p' "$0"; exit 2; }
# shellcheck disable=SC1091
source "$REPO/slurm/_env.sh" >/dev/null 2>&1
# -n 1: with --gres=gpu:1 and no -n, srun launched the command TWICE (two tasks) from this desktop's shell.
# TMPDIR=/tmp: the desktop's TMPDIR (/scratch/$USER/tmp) does not exist on compute nodes.
common=(-n 1 --mem=8G -t "${BHL_TIME:-00:20:00}")
envs=(env TMPDIR=/tmp PYTHONPATH="$REPO/src" OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1)
case "$mode" in
  gpu)
    # dgxh is left out: its MIG slices fall back to software rendering. dgx2 needs the EGL device pinned to 0.
    slurm_clean srun -J bhl-driver-smoke "${common[@]}" -p share,eecs,gpu,ampere,dgx2 --gres=gpu:1 -c 2 \
      bash -c 'case "$(hostname)" in dgx2-*) export MUJOCO_EGL_DEVICE_ID=0;; esac; exec "$@"' _ \
      "${envs[@]}" MUJOCO_GL=egl "$PY" "$SKILL/driver.py" "$@" ;;
  cpu)
    slurm_clean srun -J bhl-driver-smoke "${common[@]}" -p share --constraint=el9 -c 2 \
      "${envs[@]}" "$PY" "$SKILL/driver.py" "$@" ;;
  test)
    log=${BHL_LOG:-/nfs/hpc/share/$USER/Humanoid_Lite/logs/bhl-test-$(date +%Y%m%d-%H%M%S).txt}
    echo "log: $log"
    # pytest's output goes to a file: MuJoCo warnings bury the summary, and a pipe would hide pytest's exit status.
    slurm_clean srun -J bhl-test-smoke "${common[@]}" -p share --constraint=el9 -c 2 --mem=16G \
      bash -c 'cd "$1"; shift; log=$1; shift; "$@" > "$log" 2>&1; echo "PYTEST_EXIT=$?" >> "$log"' _ \
      "$REPO" "$log" "${envs[@]}" "$PY" -m pytest -q -p no:cacheprovider "$@" || true
    grep -E "passed|failed|error|no tests ran" "$log" | tail -1 || true
    tail -1 "$log"
    grep -q "^PYTEST_EXIT=0$" "$log" ;;
  py)
    slurm_clean srun -J bhl-py-smoke "${common[@]}" -p share --constraint=el9 -c 2 \
      bash -c 'cd "$1"; shift; exec "$@"' _ "$REPO" "${envs[@]}" "$PY" "$@" ;;
  *) echo "unknown mode $mode (gpu | cpu | test | py)"; exit 2 ;;
esac
