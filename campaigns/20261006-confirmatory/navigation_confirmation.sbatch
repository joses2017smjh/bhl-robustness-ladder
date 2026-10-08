#!/bin/bash
#SBATCH --job-name=nav-confirm
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --array=0-7%2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/stak/users/sanchej7/hpc-share/Computer_Vision/project_results_upgrade/humanoid_confirmatory/jobs/%x-%A_%a.log
#SBATCH --error=/nfs/stak/users/sanchej7/hpc-share/Computer_Vision/project_results_upgrade/humanoid_confirmatory/jobs/%x-%A_%a.log
set -euo pipefail
CAMPAIGN=/nfs/stak/users/sanchej7/hpc-share/Computer_Vision/project_results_upgrade/humanoid_confirmatory
PY=/nfs/stak/users/sanchej7/hpc-share/Humanoid_Lite/venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
exec "$PY" "$CAMPAIGN/navgym_confirmatory.py" run --campaign "$CAMPAIGN" --cell "$SLURM_ARRAY_TASK_ID"
