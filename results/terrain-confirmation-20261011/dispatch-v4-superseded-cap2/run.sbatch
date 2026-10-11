#!/bin/bash
set -euo pipefail
cd /tmp
export TMPDIR=/tmp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python /nfs/hpc/share/sanchej7/humanoid-research-20261011/terrain-confirmation-dispatch-v4/terrain_confirmation_dispatch.py --plan /nfs/hpc/share/sanchej7/humanoid-research-20261011/terrain-confirmation-dispatch-v4/plan.json --plan-sha256 67771d8042087f5a5c5a18c944203f15bb3f5f115f77f6dc115458d286c4d1fd
