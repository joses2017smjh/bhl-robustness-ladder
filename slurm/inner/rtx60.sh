#!/bin/bash
set -uo pipefail
cd "$UPSTREAM"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
exec "$PY" "$REPO/scripts/bench/rtx60_probe.py"
