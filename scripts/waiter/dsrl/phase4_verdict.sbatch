#!/bin/bash
#SBATCH --account=eecs
#SBATCH --partition=share,dgx2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --job-name=waiter-p4-verdict
#SBATCH --output=%x-%j.log
#
# Waiter phase 4 verdict as a job, so that a predeclared chain (training -> the one scored evaluation -> verdict) can
# finish with nobody watching: phase4_gate.py checks the scored file against the frozen rule and the training run,
# then writes the verdict once. No GPU, no simulator.
#   sbatch --dependency=afterok:<scored evaluation job> phase4_verdict.sbatch <rule.json> <scored.json> <verdict.json>
set -euo pipefail
WS=/nfs/hpc/share/sanchej7/Humanoid_Lite
REPO=${DSRL_REPO:-$WS/bhl-robustness-ladder}
echo "[p4-verdict] job=${SLURM_JOB_ID:-} host=$(hostname) git=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) $(date '+%Y-%m-%d %H:%M')"
exec "$WS/venv/bin/python" "$REPO/scripts/waiter/dsrl/phase4_gate.py" --rule "$1" --eval "$2" --out "$3"
