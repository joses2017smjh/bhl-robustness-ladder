#!/bin/bash
set -euo pipefail
cd /tmp
export TMPDIR=/tmp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Exclude home PATH entries that hung the previous controller before sacct exec.
export PATH=/apps/slurm/25.11.8/bin:/usr/bin:/bin:/nfs/hpc/share/sanchej7/humanoid-research-20261011/terrain-confirmation-dispatch-v6/controller-tools
sha256sum -c <<'CHECKSUMS'
b9b87b7f50c4e56f42e373805ebddb4ebfd9d73f9d96b947a9dc4a338c66be59  /apps/slurm/25.11.8/bin/sacct
0b8c8dfa4350b20957f80f6dcf36729a7f5e5862b787c202403447a5dfd6a181  /apps/slurm/25.11.8/bin/sbatch
141507c337e8b202ad398550c3b73d72f5af92e86f71665214538a81efd4c409  /nfs/hpc/share/sanchej7/humanoid-research-20261011/terrain-confirmation-dispatch-v6/controller-tools/gh
CHECKSUMS
/apps/slurm/25.11.8/bin/sacct -X -j 21768612 --noheader --parsable2 --format=JobIDRaw,State,ExitCode
exec /nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python /nfs/hpc/share/sanchej7/humanoid-research-20261011/terrain-confirmation-dispatch-v6/terrain_confirmation_dispatch.py --plan /nfs/hpc/share/sanchej7/humanoid-research-20261011/terrain-confirmation-dispatch-v6/plan.json --plan-sha256 e2f92da10a0fc161e24e67dc59587365c40df40a8b7749fe3031eddd1fcc64f3
