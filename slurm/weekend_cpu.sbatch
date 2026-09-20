#!/bin/bash
#SBATCH --job-name=wknd-mujoco
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
REPO=/nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/$USER/Humanoid_Lite/venv/bin/python
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}" OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
script=${1:?repo-relative Python script}; shift
cd "$REPO"
exec "$PY" "$REPO/$script" "$@"
