#!/bin/bash
#SBATCH --signal=B:TERM@30
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/bhl-schedule-report-mpl-${SLURM_JOB_ID}"
exec /nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python "$1/run_report.py" "$1" "$2" "$3"
