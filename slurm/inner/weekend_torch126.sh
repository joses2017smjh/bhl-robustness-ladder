#!/bin/bash
# Existing weekend_gpu/cpu launcher must select stack=train (LeRobot py3.11).
# install: download a fresh isolated overlay; smoke: run real CUDA kernels;
# run <inner-script> <args...>: use the overlay for the authorized fold job.
set -euo pipefail
mode=${1:?install, smoke, or run}
shift
overlay="$REPO/results/weekend-20260919/deps/torch-cu126"
if [ "$mode" = install ]; then
    exec "$PY" "$REPO/scripts/bench/torch_cu126.py" install --target "$overlay"
fi
[ -f "$overlay/bhl-overlay.json" ] || { echo "Torch cu126 overlay is not installed" >&2; exit 1; }
export PYTHONPATH="$overlay:$REPO/src:${PYTHONPATH:-}"
if [ "$mode" = smoke ]; then
    exec "$PY" "$REPO/scripts/bench/torch_cu126.py" smoke --target "$overlay" --cuda
elif [ "$mode" = run ]; then
    inner=${1:?repo-relative inner script}
    shift
    "$PY" "$REPO/scripts/bench/torch_cu126.py" smoke --target "$overlay" --cuda
    exec bash "$REPO/$inner" "$@"
fi
echo "Unknown cu126 mode: $mode" >&2
exit 2
