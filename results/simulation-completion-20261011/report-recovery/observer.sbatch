#!/bin/bash
#SBATCH --job-name=bhl-report-closeout
#SBATCH --account=eecs
#SBATCH --partition=share,eecs
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/nfs/hpc/share/sanchej7/humanoid-research-20261011/completion-report-observer-v2/observer-%j.log
#SBATCH --error=/nfs/hpc/share/sanchej7/humanoid-research-20261011/completion-report-observer-v2/observer-%j.log
set -euo pipefail
cd /tmp
export PATH=/apps/slurm/25.11.8/bin:/usr/bin:/bin
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python /nfs/hpc/share/sanchej7/humanoid-research-20261011/completion-report-observer-v2/research_completion_report.py --plan /nfs/hpc/share/sanchej7/humanoid-research-20261011/completion-report-observer-v2/plan.json --plan-sha256 6eacc15598c195d24e3d0d97226ee4fd94e7a4330cab379adb80c1c48279a3cc --once --publish
