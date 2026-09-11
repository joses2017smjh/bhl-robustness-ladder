#!/bin/bash
# Three leg poses, zero action, measure tilt/height. Diagnostic, not training.
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
OUT="${BENCH_OUT:-$REPO/results/cloth/spawn_probe.json}"
LOG="${OUT%.json}.txt"
mkdir -p "$(dirname "$OUT")"
"$PY" "$REPO/scripts/cloth/spawn_probe.py" \
    --num_envs "${NUM_ENVS:-4}" --steps "${PROBE_STEPS:-60}" \
    --headless --out "$OUT" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
grep -q "final tilt" "$LOG" || {
    echo "spawn probe: no summary table -- treating as failure" >&2; exit 1; }
exit $rc
