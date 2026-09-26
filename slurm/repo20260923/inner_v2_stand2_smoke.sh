#!/bin/bash
# CubeToShelfStand2 construction/reset/step smoke (3 vision variants). Its own
# summary file, so the committed task_v2_smoke.txt and task_v2_stand_smoke.txt
# stay as published. The verdict is computed from the printed table and the
# manager tables in the log, never from the python exit code (Isaac's shutdown
# hard-exits the interpreter, so that code cannot be trusted).
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
OUTDIR="$REPO/results/repo-gpu-20260923"
OUT="$OUTDIR/task_v2_stand2_smoke.txt"
LOG="$OUTDIR/task_v2_stand2_smoke.log"
mkdir -p "$OUTDIR"
"$PY" scripts/bench/task_v2_smoke.py --num_envs "${NUM_ENVS:-4}" --enable_cameras \
    --only CubeToShelfStand2 > "$LOG" 2>&1 || true
# The summary table (everything after the header row) is the result file.
awk '/^task +obs +success term/{p=1} p' "$LOG" > "$OUT" || true
cat "$OUT"

ok_rows=$(grep -cE '^TaskV2-BHL-CubeToShelfStand2-(Blind|Depth|Rgb)-v0 +[0-9]+ +True +ok$' "$OUT" || true)
summary=$(grep -oE '^[0-9]+/[0-9]+ built and stepped' "$OUT" || true)
# Each env build prints its reward and curriculum manager tables: the new terms
# must be active and v1's success-charging termination_penalty must not be.
n_fall=$(grep -cE '\| +fall_penalty +\|' "$LOG" || true)
n_gate=$(grep -cE '\| +upright_gate +\|' "$LOG" || true)
n_old=$(grep -cE '\| +termination_penalty +\|' "$LOG" || true)
echo "smoke: ok_rows=$ok_rows summary='${summary}' fall_penalty_tables=$n_fall upright_gate_tables=$n_gate termination_penalty_tables=$n_old"
if [ "$ok_rows" = 3 ] && [ "$summary" = "3/3 built and stepped" ] \
   && [ "$n_fall" -ge 3 ] && [ "$n_gate" -ge 3 ] && [ "$n_old" = 0 ]; then
    echo "SMOKE-VERDICT CubeToShelfStand2: PASS (3/3 variants built, reset and stepped with the Stand2 terms active)"
    exit 0
fi
echo "SMOKE-VERDICT CubeToShelfStand2: FAIL (see $LOG)"
grep -nE "^(Traceback|[A-Za-z]+Error)" "$LOG" | head -5 || true
exit 1
