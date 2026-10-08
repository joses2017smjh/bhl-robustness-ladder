#!/bin/bash
set -uo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
run=$1; out=$2; shift 2
timeout 780 "$PY" scripts/bench/ice_exposure_probe.py --run-dir "$run" --checkpoint model_5999.pt \
    --num-envs 64 --out "$out" "$@"
echo "inner exit $?"
