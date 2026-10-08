#!/bin/bash
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --constraint=haswell&el8
#SBATCH --array=0-3%2
set -euo pipefail
campaign=${1:?frozen campaign path}
python_bin=${2:?Python 3.11 interpreter}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec "$python_bin" "$campaign/snapshot/scripts/bench/navigation_development.py" run --campaign "$campaign" --cell "$SLURM_ARRAY_TASK_ID"
