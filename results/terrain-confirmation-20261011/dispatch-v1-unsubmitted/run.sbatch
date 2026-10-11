#!/bin/bash
set -euo pipefail
cd /tmp
export TMPDIR=/tmp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python /nfs/hpc/share/sanchej7/humanoid-research-20261011/terrain-confirmation-dispatch-v1/terrain_confirmation_dispatch.py --plan /nfs/hpc/share/sanchej7/humanoid-research-20261011/terrain-confirmation-dispatch-v1/plan.json --plan-sha256 ed54e4be5950e5b5bc1f1c261b7ca368d65249a145c564ccc3b265e2011baa12
