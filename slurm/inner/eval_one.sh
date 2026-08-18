#!/bin/bash
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
exec "$PY" -m bhl_robust.eval.run_eval \
    --deploy-cfg "$DEPLOY_CFG" --upstream "$UPSTREAM" \
    --cache-dir "$CACHE_DIR" --out "$OUT_CSV" --label "$LABEL" \
    --episode-s "${EPISODE_S:-10}" --seeds "${N_SEEDS:-3}" --push-speed "${PUSH_SPEED:-0}"
