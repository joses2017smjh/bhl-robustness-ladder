#!/bin/bash
# Hands vs table vs garment in the Isaac cloth scene. Diagnostic, no policy.
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
OUT="${BENCH_OUT:-$REPO/results/cloth/hand_probe.json}"
LOG="${OUT%.json}.txt"
mkdir -p "$(dirname "$OUT")"
"$PY" "$REPO/scripts/cloth/hand_probe.py" --num_envs "${NUM_ENVS:-4}" --headless \
    --out "$OUT" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
grep -q "hand probe done" "$LOG" || { echo "hand probe: no summary -- failure" >&2; exit 1; }
exit $rc
