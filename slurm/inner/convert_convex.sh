#!/bin/bash
set -euo pipefail
cd "$UPSTREAM"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
exec "$PY" "$REPO/scripts/convert_convex_usd.py" --headless
