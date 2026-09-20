#!/bin/bash
#SBATCH --job-name=wknd-mujoco
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# Supply cluster account/partition through sbatch flags.
set -euo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
PY=${BHL_PYTHON:-python}
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}" OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
script=${1:?repo-relative Python script}; shift
cd "$REPO"
exec "$PY" "$REPO/$script" "$@"
