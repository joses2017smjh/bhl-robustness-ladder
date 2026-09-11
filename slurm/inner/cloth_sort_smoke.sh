#!/bin/bash
# Construct / reset / step cloth-sort gym ids. Rigid by default.
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
ARGS=(--num_envs "${NUM_ENVS:-4}" --steps "${BENCH_STEPS:-6}" --headless)
[ "${INCLUDE_DEFORMABLE:-0}" = "1" ] && ARGS+=(--include-deformable --cloth-res "${CLOTH_RES:-8}")
OUT="${BENCH_OUT:-$REPO/results/cloth/isaac_smoke.json}"
ARGS+=(--out "$OUT")
LOG="${OUT%.json}.txt"
mkdir -p "$(dirname "$OUT")"
"$PY" "$REPO/scripts/bench/cloth_sort_smoke.py" "${ARGS[@]}" \
    2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
grep -q "built and stepped" "$LOG" || {
    echo "smoke: produced no summary line -- treating as failure" >&2; exit 1; }
exit $rc
