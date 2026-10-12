#!/bin/bash
set -euo pipefail
export PATH=/apps/slurm/25.11.8/bin:/usr/bin:/bin
export TMPDIR=/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec /nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python /nfs/hpc/share/sanchej7/humanoid-research-20261011/euroc-actual-20261012/worker.py "$1"
