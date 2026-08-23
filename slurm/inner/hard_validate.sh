#!/bin/bash
# Two iterations of each NEW task through the real driver, before any of them
# gets a 30-hour slot. Same discipline the crew work needed and did not have.
set -uo pipefail
cd "$UPSTREAM"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
: > "$GATE_OUT"
for spec in "CoopLift-BHL-Cube-Random-v0 rnd" "CoopLift-BHL-Cube-Occluded-v0 occ" \
            "CoopLift-BHL-Cube-Occluded-Depth-v0 occd"; do
    set -- $spec
    echo "=== $1 ==="
    VLOG="$REPO/results/hard_gate_$2.log"
    export TASK="$1" RUN_NAME="_gate-$2" SEED=0 NUM_ENVS=8 MAX_ITER=2
    export OVERRIDE_FILE="$WORKSPACE/overrides/gate-hard.txt"
    mkdir -p "$WORKSPACE/overrides"; echo "env.stage_lift_on_pinch=true" > "$OVERRIDE_FILE"
    if timeout 1500 bash "$REPO/slurm/inner/train.sh" > "$VLOG" 2>&1; then
        echo "$1 PASS" >> "$GATE_OUT"
    else
        echo "$1 FAIL" >> "$GATE_OUT"
        grep -A25 "Traceback" "$VLOG" | tail -25 >&2
        grep -hoE "^[A-Za-z_]*(Error|Exception): .{0,200}" "$VLOG" | tail -3 >&2
    fi
done
echo "=== verdicts ==="; cat "$GATE_OUT"
grep -q FAIL "$GATE_OUT" && { echo "GATE FAILED"; exit 1; }
[ "$(grep -c PASS "$GATE_OUT")" -eq 3 ] || { echo "GATE INCOMPLETE"; exit 1; }
echo "GATE PASSED"
