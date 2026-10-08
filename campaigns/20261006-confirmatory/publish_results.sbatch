#!/bin/bash
#SBATCH --job-name=humanoid-doc-results
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/nfs/stak/users/sanchej7/hpc-share/Computer_Vision/project_results_upgrade/humanoid_confirmatory/jobs/%x-%j.log
#SBATCH --error=/nfs/stak/users/sanchej7/hpc-share/Computer_Vision/project_results_upgrade/humanoid_confirmatory/jobs/%x-%j.log
set -euo pipefail
CAMPAIGN=/nfs/stak/users/sanchej7/hpc-share/Computer_Vision/project_results_upgrade/humanoid_confirmatory
PY=/nfs/stak/users/sanchej7/hpc-share/Humanoid_Lite/venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec "$PY" "$CAMPAIGN/publish_existing_results.py" --campaign "$CAMPAIGN"
