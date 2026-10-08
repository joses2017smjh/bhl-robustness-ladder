#!/bin/bash
set -uo pipefail
cd "$UPSTREAM"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
limit=$1; shift
timeout -k 30 "$limit" "$PY" "$@"
