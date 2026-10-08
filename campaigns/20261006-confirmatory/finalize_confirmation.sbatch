#!/bin/bash
#SBATCH --job-name=humanoid-verdict
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/nfs/stak/users/sanchej7/hpc-share/Computer_Vision/project_results_upgrade/humanoid_confirmatory/jobs/%x-%j.log
#SBATCH --error=/nfs/stak/users/sanchej7/hpc-share/Computer_Vision/project_results_upgrade/humanoid_confirmatory/jobs/%x-%j.log
set -euo pipefail
CAMPAIGN=/nfs/stak/users/sanchej7/hpc-share/Computer_Vision/project_results_upgrade/humanoid_confirmatory
PY=/nfs/stak/users/sanchej7/hpc-share/Humanoid_Lite/venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$PY" "$CAMPAIGN/dr_confirmatory.py" summary --campaign "$CAMPAIGN" --final > "$CAMPAIGN/dr_final_summary.log"
exec "$PY" "$CAMPAIGN/navgym_confirmatory.py" summary --campaign "$CAMPAIGN" --final
