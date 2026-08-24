#!/bin/bash
set -uo pipefail
cd "$UPSTREAM"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
exec "$PY" "$REPO/scripts/bench/port_audit.py"
