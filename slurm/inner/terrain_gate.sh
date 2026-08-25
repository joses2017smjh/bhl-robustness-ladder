#!/bin/bash
set -uo pipefail
cd "$UPSTREAM"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
: > "$GATE_OUT"
rc=0
for c in friction stairs; do
    echo "=== G-B: $c ==="
    # Full output to a file. A `| tail -N` here throws away the traceback and
    # turns a readable exception into a bare "GATE FAILED", which has now cost
    # a round trip on the crew gate and again here.
    VLOG="$REPO/results/terrain_gate_$c.log"
    if ! "$PY" "$REPO/scripts/bench/terrain_gate.py" --check "$c" > "$VLOG" 2>&1; then
        rc=1
        grep -A25 "Traceback" "$VLOG" | tail -25 >&2
        grep -hoE "^[A-Za-z_]*(Error|Exception): .{0,200}" "$VLOG" | tail -3 >&2
    fi
    grep -E "CHECK \||PASS  \||FAIL  \||VERDICT" "$VLOG" || true
done
grep -c "ALL PASS" "$GATE_OUT" | grep -q "^2$" || { echo "TERRAIN GATE FAILED"; exit 1; }
grep -q "FAIL  |" "$GATE_OUT" && { echo "TERRAIN GATE FAILED"; exit 1; }
echo "TERRAIN GATE PASSED"
