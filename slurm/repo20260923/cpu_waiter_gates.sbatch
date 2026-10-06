#!/bin/bash
#SBATCH --job-name=waiter-gates
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
# Waiter phase 1 gates (docs/WAITER_PROGRAM.md; frozen 2026-10-05) for seed = array index, on the MuJoCo Waiter
# model (--variant waiter; arms commanded to the default pose and grippers open unless the gate commands them):
#   Q1 = scripts/bench/turn_test.py --protocol v2 (PASS) and cpu_turn_qualify.sbatch with TURNQ_VARIANT=waiter
#        (QUALIFIED: turn 9/10, walk 2/3, push <= 9/60), both protocols unchanged;
#   Q2 = scripts/waiter/q2_gate.py (0 falls in 12 and drift <= 15 deg on >= 10/12, 0.5 kg per hand);
# then scripts/waiter/wbc_select.py writes results/waiter-20261005/wbc/selection.json once all three seeds are in.
# usage: sbatch --dependency=aftercorr:<training array> slurm/repo20260923/cpu_waiter_gates.sbatch
set -uo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
RES=${WAITER_GATES_RES:-$REPO/results/waiter-20261005/wbc}
PREFIX=${WAITER_GATES_PREFIX:-waiter-wbc}
SEED=$(( ${WAITER_GATES_SEED0:-0} + ${SLURM_ARRAY_TASK_ID:-0} ))
RUN=${WAITER_GATES_RUN:-$PREFIX-s$SEED}
JOBTAG=${SLURM_JOB_ID:-local$$}
export TMPDIR=/tmp PYTHONPATH="$REPO/src:$REPO/scripts" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
CACHE=/tmp/waiter-gates-$JOBTAG; mkdir -p "$CACHE" "$RES"/{turn-test-v2,qualify,q2}
cd "$REPO"
echo "=== waiter gates $RUN | $(hostname) | HEAD $(git rev-parse --short HEAD) | $(date)"
mapfile -t dirs < <(ls -d "$U"/logs/rsl_rl/humanoid/*_"$RUN" 2>/dev/null | sort)
[ "${#dirs[@]}" -eq 1 ] || { echo "WAITER-GATES RESULT: INCOMPLETE ($RUN: ${#dirs[@]} run dirs)"; exit 1; }
DEPLOY=${dirs[0]}/exported/deploy.yaml
grep -q "^waiter_wbc:" "$DEPLOY" 2>/dev/null || { echo "WAITER-GATES RESULT: INCOMPLETE (no stamped deploy.yaml: $DEPLOY)"; exit 1; }
echo "deploy: $DEPLOY"
V2=$RES/turn-test-v2/$RUN.json
if [ ! -f "$V2" ]; then
    "$PY" scripts/bench/turn_test.py --protocol v2 --deploy "$DEPLOY" --upstream "$U" --cache-dir "$CACHE" \
        --variant waiter --out "$CACHE/v2.json" 2>&1 | grep -v "^\[" | tail -8
    [ -f "$CACHE/v2.json" ] && cp -n "$CACHE/v2.json" "$V2"
fi
TURNQ_RUN="$RUN" TURNQ_OUT_DIR="$RES/qualify" TURNQ_VARIANT=waiter bash slurm/repo20260923/cpu_turn_qualify.sbatch 2>&1 | tail -6
Q2=$RES/q2/$RUN.json
[ -f "$Q2" ] || "$PY" scripts/waiter/q2_gate.py --deploy "$DEPLOY" --upstream "$U" --cache-dir "$CACHE" --out "$Q2" 2>&1 | tail -3
for f in "$V2" "$RES/qualify/${RUN}__qualify.json" "$Q2"; do
    [ -f "$f" ] && echo "$(basename "$f"): $("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['verdict'])" "$f")"
done
"$PY" scripts/waiter/wbc_select.py --res "$RES" --prefix "$PREFIX" --seeds $(( ${WAITER_GATES_SEED0:-0} )) $(( ${WAITER_GATES_SEED0:-0} + 1 )) $(( ${WAITER_GATES_SEED0:-0} + 2 )) || true
echo "=== done $(date)"
