#!/bin/bash
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
script=${1:?repo-relative Python script}; shift
exec "$PY" "$REPO/$script" "$@"
