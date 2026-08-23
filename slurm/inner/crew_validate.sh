#!/bin/bash
# Gate the crew configs through the SAME driver the training arms use.
#
# Not a bespoke validator, and not train.py called directly: inner/train.sh is
# where the "did it actually reach an iteration" check lives, and the whole
# reason this gate exists is that exit codes from this stack are not evidence.
# Anything the gate does differently from a training arm is a gap the gate
# cannot see through -- which is exactly how the previous two versions passed.
set -uo pipefail
cd "$UPSTREAM"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
: > "$GATE_OUT"

# Cheapest possible first: importing the task package runs the registration
# loop, which constructs every crew configclass. A malformed generated class
# raises there, in seconds, without building a scene.
echo "=== import check ==="
if "$PY" -c "
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import berkeley_humanoid_lite.tasks, bhl_robust.tasks  # noqa: F401
import gymnasium as gym
ids = [i for i in gym.registry if 'Crew' in i]
print('REGISTERED', sorted(ids))
assert len(ids) == 4, f'expected 4 crew tasks, got {len(ids)}'
print('IMPORT OK')
app.close()
" 2>&1 | tail -6 | grep -q "IMPORT OK"; then
    echo "import ok"
else
    echo "IMPORT FAILED -- crew configclasses do not construct" >&2
    echo "import FAIL" >> "$GATE_OUT"
fi

for spec in "CoopLift-BHL-Cube-Crew3-v0 3" "CoopLift-BHL-Cube-Crew4-v0 4" \
            "CoopLift-BHL-Cube-Crew3-Depth-v0 3" "CoopLift-BHL-Cube-Crew4-Depth-v0 4"; do
    set -- $spec
    echo "=== $1 ==="
    export TASK="$1" RUN_NAME="_gate-crew$2" SEED=0 NUM_ENVS=8 MAX_ITER=2
    export OVERRIDE_FILE="$WORKSPACE/overrides/gate-crew.txt"
    mkdir -p "$WORKSPACE/overrides"; echo "env.stage_lift_on_pinch=true" > "$OVERRIDE_FILE"
    if timeout 1500 bash "$REPO/slurm/inner/train.sh" 2>&1 | tail -25; then
        echo "$1 PASS" >> "$GATE_OUT"
    else
        echo "$1 FAIL" >> "$GATE_OUT"
    fi
done

cp "$GATE_OUT" "$REPO/results/crew_validate.log" 2>/dev/null || true
echo "=== verdicts ==="; cat "$GATE_OUT"
grep -q FAIL "$GATE_OUT" && { echo "GATE FAILED"; exit 1; }
[ "$(grep -c PASS "$GATE_OUT")" -eq 4 ] || { echo "GATE INCOMPLETE"; exit 1; }
echo "GATE PASSED"
