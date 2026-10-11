#!/bin/bash
set -euo pipefail
cd /tmp
export TMPDIR=/tmp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python /nfs/hpc/share/sanchej7/humanoid-research-20261011/terrain-confirmation-dispatch-v2/terrain_confirmation_dispatch.py --plan /nfs/hpc/share/sanchej7/humanoid-research-20261011/terrain-confirmation-dispatch-v2/plan.json --plan-sha256 9d331dae00ab2e02ec22eb81cb98f7f9c2c15e1315b2793d46570e0dd1b39eda
