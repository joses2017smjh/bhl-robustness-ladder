#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" scripts/bench/marl_gate.py --task "${TASK:-Velocity-BHL-Arms-Bumpy-v0}" \
    --num_envs "${NUM_ENVS:-8}" 2>&1 | tee "$REPO/results/marl_gate.txt"
rc=${PIPESTATUS[0]}
grep -qE "^G-B4 (PASS|FAIL)" "$REPO/results/marl_gate.txt" || {
    echo "marl_gate: no verdict line -- treating as failure" >&2; exit 1; }
exit $rc
