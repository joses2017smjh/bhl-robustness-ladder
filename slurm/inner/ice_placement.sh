#!/bin/bash
set -uo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
OUT="$REPO/results/ice_placement_probe.txt"
for task in Velocity-BHL-Biped-Ice-Depth-v0; do
  timeout 1200 "$PY" scripts/bench/ice_placement_probe.py --task "$task" --num_envs "${NUM_ENVS:-4096}" 2>&1 | tee "$OUT.log"
done
grep -E "^task |^robot|^patch 0|^terrain origin|^reach bound|^ICE-PLACEMENT" "$OUT.log" > "$OUT"
cat "$OUT"
grep -q "^ICE-PLACEMENT" "$OUT"
