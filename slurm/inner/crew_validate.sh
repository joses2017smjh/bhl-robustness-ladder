#!/bin/bash
# Gate the crew configs by running the REAL training entrypoint for two
# iterations, one process per variant.
#
# The previous version called `parse_env_cfg` and passed all four variants
# while three of four training arms died thirty seconds in. `parse_env_cfg`
# instantiates the config object; training reaches it through Hydra, which
# round-trips it via to_dict/from_dict and drops anything that is not a
# declared dataclass field. A validator that does not take the path training
# takes is not a validator, it is a second opinion from a different code path.
#
# Two iterations is enough: everything that broke, broke during manager
# construction, before the first gradient step.
set -uo pipefail
cd "$UPSTREAM"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
: > "$GATE_OUT"
for spec in "CoopLift-BHL-Cube-Crew3-v0 3" "CoopLift-BHL-Cube-Crew4-v0 4" \
            "CoopLift-BHL-Cube-Crew3-Depth-v0 3" "CoopLift-BHL-Cube-Crew4-Depth-v0 4"; do
    set -- $spec
    echo "=== $1 ==="
    if timeout 1200 "$PY" "$REPO/scripts/train.py" --task "$1" --headless \
         --seed 0 --num_envs 8 --max_iterations 2 --run_name "_gate-crew$2" \
         env.stage_lift_on_pinch=true 2>&1 | tail -40; then
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
