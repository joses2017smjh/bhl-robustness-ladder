#!/bin/bash
set -uo pipefail
cd "$UPSTREAM"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
: > "$GATE_OUT"
rc=0
for c in friction stairs; do
    echo "=== G-B: $c ==="
    "$PY" "$REPO/scripts/bench/terrain_gate.py" --check "$c" 2>&1 | tail -22 || rc=1
done
grep -c "ALL PASS" "$GATE_OUT" | grep -q "^2$" || { echo "TERRAIN GATE FAILED"; exit 1; }
grep -q "FAIL  |" "$GATE_OUT" && { echo "TERRAIN GATE FAILED"; exit 1; }
echo "TERRAIN GATE PASSED"
