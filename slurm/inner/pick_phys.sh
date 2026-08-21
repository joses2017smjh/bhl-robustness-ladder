#!/bin/bash
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
exec "$PY" scripts/render_pick.py \
    --upstream "$UPSTREAM" --cache-dir "$CACHE_DIR"
