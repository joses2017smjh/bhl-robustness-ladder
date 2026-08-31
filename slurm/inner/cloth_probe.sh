#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" scripts/bench/cloth_probe.py --enable_cameras 2>&1 | tee "$REPO/results/cloth_probe.txt"
