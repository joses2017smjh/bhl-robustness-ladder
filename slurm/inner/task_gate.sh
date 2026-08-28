#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" scripts/bench/task_gate.py --upstream "$UPSTREAM" --cache-dir "$CACHE_DIR" \
    --grasp-z "${GRASP_Z:-0.30}" | tee "$REPO/results/task_gate.txt"
