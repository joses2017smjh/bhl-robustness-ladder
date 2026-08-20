#!/bin/bash
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
mkdir -p "$REPO/docs/gifs"
exec "$PY" scripts/render_pick.py \
    --upstream "$UPSTREAM" --cache-dir "$CACHE_DIR" \
    --out "$REPO/docs/gifs/squat_pick.mp4" \
    --gif "$REPO/docs/gifs/squat_pick.gif"
