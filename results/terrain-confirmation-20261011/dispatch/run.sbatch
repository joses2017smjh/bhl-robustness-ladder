#!/bin/bash
set -euo pipefail
cd /tmp
export TMPDIR=/tmp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python /nfs/hpc/share/sanchej7/humanoid-research-20261011/terrain-confirmation-dispatch-v5/terrain_confirmation_dispatch.py --plan /nfs/hpc/share/sanchej7/humanoid-research-20261011/terrain-confirmation-dispatch-v5/plan.json --plan-sha256 e2f92da10a0fc161e24e67dc59587365c40df40a8b7749fe3031eddd1fcc64f3
