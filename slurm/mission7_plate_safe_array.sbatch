#!/bin/bash
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
set -euo pipefail
repo=${1:?}; snapshot=${2:?}; out=${3:?}; stage=${4:?}
cd "$repo"; PY="${PYTHON:-python3}"
export PYTHONPATH="$snapshot/src:$snapshot/scripts" MISSION7_REPO="$repo" MISSION7_OUT="$out" MISSION7_STAGE="$stage" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec "$PY" "$snapshot/scripts/mission7_plate_safe_single.py"
