#!/bin/bash
#SBATCH --job-name=waiter-lockremoval-select
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# Option 3 of the stance-lock work (2026-10-14; SLURM_JOBS.md "Predeclared now ... more standing-only seeds"): pick the
# candidate pair from the dev lock-removal checks (scripts/waiter/lockremoval_select.py) and, iff it reached
# SEL_MIN_DEV/32, run the scored lock-removal test on it right here: waiter_expert_gate.sbatch's protocol
# (GATE_START_ONLY=1) on the fresh seeds 9000-9099 (expert_gate.py --range lock2), PASS iff >= 80/100.
#   sbatch --dependency=afterany:<dev checks> --export=ALL,SEL_CANDIDATES_FILE=<file>,SEL_OUT=<dir> \
#       slurm/repo20260923/waiter_lockremoval_select.sbatch
# (the file holds one RUN=DEPLOY=Q4_JSON=DEV_GATE_JSON per line; sbatch's argument handling splits a long list)
set -uo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
OUT=${SEL_OUT:?set SEL_OUT}
cd "$REPO"
echo "=== lock-removal select | $(hostname) | HEAD $(git rev-parse --short HEAD) | $(date)"
mapfile -t CANDS < "${SEL_CANDIDATES_FILE:?set SEL_CANDIDATES_FILE}"
ARGS=(); for c in "${CANDS[@]}"; do [ -n "$c" ] && ARGS+=(--candidate "$c"); done
line=$("$PY" scripts/waiter/lockremoval_select.py "${ARGS[@]}" --min-dev "${SEL_MIN_DEV:-26}" --out "$OUT/selection.json" | tee /dev/stderr | tail -1)
case "$line" in
  "SELECT: none"*) echo "LOCK-REMOVAL (${SEL_LABEL:-option 3}): no candidate reached the dev bar; no scored run"; exit 0 ;;
  "SELECT: "*) set -- $line; RUN=$2; DEPLOY=$3 ;;
  *) echo "LOCK-REMOVAL (${SEL_LABEL:-option 3}): selection failed"; exit 1 ;;
esac
echo "scored lock-removal test on $RUN ($DEPLOY), seeds 9000-9099"
GATE_OUT="$OUT/scored" GATE_DEPLOY="$DEPLOY" GATE_START_ONLY=1 GATE_SEED0=9000 GATE_SEED1=9099 GATE_RANGE=lock2 \
  GATE_MIN_SUCCESS=80 bash slurm/repo20260923/waiter_expert_gate.sbatch
