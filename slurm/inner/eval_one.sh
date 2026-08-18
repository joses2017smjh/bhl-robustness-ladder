#!/bin/bash
# Evaluate one exported policy; optionally render clips.
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
ARGS=(--deploy-cfg "$DEPLOY_CFG" --upstream "$UPSTREAM" --cache-dir "$CACHE_DIR"
      --out "$OUT_CSV" --label "$LABEL"
      --episode-s "${EPISODE_S:-10}" --seeds "${N_SEEDS:-3}" --push-speed "${PUSH_SPEED:-0}")
[ -n "${VIDEO_DIR:-}" ] && ARGS+=(--video-dir "$VIDEO_DIR")
exec "$PY" -m bhl_robust.eval.run_eval "${ARGS[@]}"
