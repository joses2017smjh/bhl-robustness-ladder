#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
OUT="${BENCH_OUT:-$REPO/results/spawn_quat_probe.json}"
mkdir -p "$(dirname "$OUT")"
timeout 1200 "$PY" scripts/bench/spawn_quat_probe.py --headless \
    --task "${TASK:-TaskV2-BHL-CubeToShelf-Blind-v0}" \
    --num_envs "${NUM_ENVS:-4}" 2>&1 | tee "${OUT%.json}.log"
grep -E "^task |^configured |^SPAWN-QUAT |^wrote " "${OUT%.json}.log"
grep -q "^SPAWN-QUAT " "${OUT%.json}.log"
# 21342562: three candidate 4-tuples stand when applied after reset; the
# configured spawn does not. After wrapping robot init_state.rot with
# native_quat of the documented yaw, as_configured has to stand on its own.
if grep -q "^SPAWN-QUAT as_configured STANDING" "${OUT%.json}.log"; then
  echo "SPAWN-QUAT GATE PASS: as_configured STANDING"
else
  echo "SPAWN-QUAT GATE FAIL: as_configured is not STANDING" >&2
  exit 1
fi
