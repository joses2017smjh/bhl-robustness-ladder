#!/bin/bash
# CubeToShelfStand construction/reset/step smoke (3 vision variants); its own
# summary file so the committed results/task_v2_smoke.txt stays as published.
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
OUT="$REPO/results/repo-gpu-20260923/task_v2_stand_smoke.txt"
"$PY" scripts/bench/task_v2_smoke.py --num_envs "${NUM_ENVS:-4}" --enable_cameras --only CubeToShelfStand 2>&1 | tee "$OUT"
rc=${PIPESTATUS[0]}
grep -q "built and stepped" "$OUT" || { echo "smoke: produced no summary line -- treating as failure" >&2; exit 1; }
exit $rc
