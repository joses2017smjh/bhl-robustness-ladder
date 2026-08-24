#!/bin/bash
# Two iterations of each algorithm variant through the real driver. The hooks
# construct policy and RND config objects that Hydra never sees, so a typo there
# surfaces at manager build -- cheap now, four wasted 40-hour slots later.
set -uo pipefail
cd "$UPSTREAM"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
: > "$GATE_OUT"
mkdir -p "$WORKSPACE/overrides"
export OVERRIDE_FILE="$WORKSPACE/overrides/gate-algo.txt"
echo "env.stage_lift_on_pinch=true" > "$OVERRIDE_FILE"
run () {  # name task envs
    local n=$1; shift; export TASK=$1 NUM_ENVS=$2
    export RUN_NAME="_gate-$n" SEED=0 MAX_ITER=2
    local VLOG="$REPO/results/algo_gate_$n.log"
    if timeout 1500 bash "$REPO/slurm/inner/train.sh" > "$VLOG" 2>&1; then
        echo "$n PASS" >> "$GATE_OUT"
    else
        echo "$n FAIL" >> "$GATE_OUT"
        grep -hoE "^[A-Za-z_]*(Error|Exception): .{0,200}" "$VLOG" | tail -2 >&2
    fi
}
BHL_RND=0.003 run rnd CoopLift-BHL-Cube-v0 8
unset BHL_RND
BHL_POLICY=recurrent run lstm CoopLift-BHL-Cube-Occluded-v0 8
BHL_POLICY=recurrent run lstmdepth CoopLift-BHL-Cube-Occluded-Depth-v0 8
unset BHL_POLICY
echo "=== verdicts ==="; cat "$GATE_OUT"
grep -q FAIL "$GATE_OUT" && { echo "GATE FAILED"; exit 1; }
[ "$(grep -c PASS "$GATE_OUT")" -eq 3 ] || { echo "GATE INCOMPLETE"; exit 1; }
echo "GATE PASSED"
