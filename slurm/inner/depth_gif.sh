#!/bin/bash
# Render the paired RGB|depth clip for one policy on rough ground.
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
exec "$PY" scripts/render_depth.py \
    --deploy-cfg "$DEPLOY_CFG" --upstream "$UPSTREAM" --cache-dir "$CACHE_DIR" \
    --out "$OUT_MP4" --label "$LABEL" \
    --terrain-difficulty "${TERRAIN_D:-0.60}" --episode-s "${EPISODE_S:-10}" \
    --depth-res "${DEPTH_RES:-64}" --panel "${PANEL:-448}"
