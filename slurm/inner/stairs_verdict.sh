#!/bin/bash
# G-B2's verdict, read from the event file rather than scraped from stdout.
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
ARGS=(--run "$GATE_RUN" --at "${MAX_ITER:-300}")
[ -n "${GATE_CTRL:-}" ] && ARGS+=(--control "$GATE_CTRL")
"$PY" scripts/bench/terrain_level.py "${ARGS[@]}"
