#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
ARGS=(--num_envs "${NUM_ENVS:-4}" --enable_cameras)
[ -n "${SMOKE_ONLY:-}" ] && ARGS+=(--only "$SMOKE_ONLY")
"$PY" scripts/bench/task_v2_smoke.py "${ARGS[@]}" \
    2>&1 | tee "$REPO/results/task_v2_smoke.txt"
rc=${PIPESTATUS[0]}
grep -q "built and stepped" "$REPO/results/task_v2_smoke.txt" || {
    echo "smoke: produced no summary line -- treating as failure" >&2; exit 1; }
exit $rc
