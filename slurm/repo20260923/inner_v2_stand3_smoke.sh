#!/bin/bash
# CubeToShelfStand3 construction/reset/step smoke (3 vision variants). Its own
# dated output directory, so every published smoke file stays as it is. The
# verdict is computed from the printed table and the manager tables in the log,
# never from the python exit code (Isaac's shutdown hard-exits the interpreter,
# so that code cannot be trusted).
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
OUTDIR="$REPO/results/repo-gpu-20260923/stand3_2026-09-27"
OUT="$OUTDIR/task_v2_stand3_smoke.txt"
LOG="$OUTDIR/task_v2_stand3_smoke.log"
mkdir -p "$OUTDIR"
"$PY" scripts/bench/task_v2_smoke.py --num_envs "${NUM_ENVS:-4}" --enable_cameras \
    --only CubeToShelfStand3 > "$LOG" 2>&1 || true
# The summary table (everything after the header row) is the result file.
awk '/^task +obs +success term/{p=1} p; /built and stepped/{exit}' "$LOG" > "$OUT" || true
cat "$OUT"

ok_rows=$(grep -cE '^TaskV2-BHL-CubeToShelfStand3-(Blind|Depth|Rgb)-v0 +[0-9]+ +True +ok$' "$OUT" || true)
summary=$(grep -oE '^[0-9]+/[0-9]+ built and stepped' "$OUT" || true)
# Same observation layout as CubeToShelfStand2 (194 / 322 / 578): Stand3 only
# clips the terms, it adds and removes none.
widths=$(awk '/^TaskV2-BHL-CubeToShelfStand3-/{printf "%s ", $2}' "$OUT")
# Each env build prints its manager tables. Count a term's rows across the log:
# each of the 3 builds must show every Stand3 term, and none of the removed ones.
n() { grep -cE "\| +$1 +\|" "$LOG" || true; }
declare -A want=( [deck_progress]=1 [action_rate_clipped]=1 [placed]=1 [fall_penalty]=1
                  [lifting_object]=1 [lift_progress]=1 [success]=1
                  [over_deck]=1 [cube_abs_x]=1 [cube_tilted]=1 [tipped_over_deck]=1
                  [upright_gate]=1 )
gone=(carry object_xy action_rate termination_penalty)
bad=""
for t in "${!want[@]}"; do c=$(n "$t"); [ "$c" -ge 3 ] || bad="$bad missing:$t($c)"; done
for t in "${gone[@]}"; do c=$(n "$t"); [ "$c" = 0 ] || bad="$bad present:$t($c)"; done
[ "$widths" = "194 322 578 " ] || bad="$bad obs_widths:[$widths]"
echo "smoke: ok_rows=$ok_rows summary='${summary}' widths='${widths}' term_check='${bad:- all ok}'"
if [ "$ok_rows" = 3 ] && [ "$summary" = "3/3 built and stepped" ] && [ -z "$bad" ]; then
    echo "SMOKE-VERDICT CubeToShelfStand3: PASS (3/3 variants built, reset and stepped with the Stand3 terms active and Stand2's removed terms absent)"
    exit 0
fi
echo "SMOKE-VERDICT CubeToShelfStand3: FAIL (see $LOG)"
grep -nE "^(Traceback|[A-Za-z]+Error)" "$LOG" | head -5 || true
exit 1
