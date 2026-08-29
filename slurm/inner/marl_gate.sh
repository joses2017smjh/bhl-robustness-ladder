#!/bin/bash
# G-B4. One partition per process: Isaac Sim does not survive building a second
# scene after the first is torn down, and the previous version hung for an hour
# on limb2 after passing limb4.
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
OUT="$REPO/results/marl_gate.txt"
: > "$OUT"
rc=0
for part in limb4 limb2; do
    echo "########## $part ##########" | tee -a "$OUT"
    "$PY" scripts/bench/marl_gate.py --task "${TASK:-Velocity-BHL-Arms-Bumpy-v0}" \
        --num_envs "${NUM_ENVS:-8}" --partition "$part" 2>&1 | tee -a "$OUT" || rc=1
done
grep -cE "^G-B4 PASS" "$OUT" | grep -q '^2$' || {
    echo "G-B4 OVERALL FAIL -- expected a PASS from both partitions" | tee -a "$OUT"; exit 1; }
echo "G-B4 OVERALL PASS -- both partitions" | tee -a "$OUT"
exit $rc
