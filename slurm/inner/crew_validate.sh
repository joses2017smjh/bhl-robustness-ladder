#!/bin/bash
# One python process per variant. Isaac Sim hangs building a second env in a
# process that has already built one, so a loop inside python cannot work here.
set -uo pipefail
cd "$UPSTREAM"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
: > "$GATE_OUT"
rc=0
for spec in "CoopLift-BHL-Cube-Crew3-v0 3" "CoopLift-BHL-Cube-Crew4-v0 4" \
            "CoopLift-BHL-Cube-Crew3-Depth-v0 3" "CoopLift-BHL-Cube-Crew4-Depth-v0 4"; do
    set -- $spec
    echo "=== $1 ==="
    timeout 900 "$PY" "$REPO/scripts/bench/crew_validate.py" --task "$1" --crew "$2" || rc=1
done
cp "$GATE_OUT" "$REPO/results/crew_validate.log" 2>/dev/null || true
echo "=== verdicts ==="; cat "$GATE_OUT"
grep -q FAIL "$GATE_OUT" && { echo "GATE FAILED"; exit 1; }
[ "$(grep -c PASS "$GATE_OUT")" -eq 4 ] || { echo "GATE INCOMPLETE"; exit 1; }
echo "GATE PASSED"
