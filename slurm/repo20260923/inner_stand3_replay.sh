#!/bin/bash
# Stand3 replay, one checkpoint, inside the v60 container (called by
# gpu_stand3_replay.sbatch through bhl_exec). Forwarded: PYTHONPATH = the pinned
# as-trained src/ snapshot (alone -- NOT $REPO/src, which may carry the
# quaternion fix), BENCH_OUT = the output prefix; everything else is in
# $BENCH_OUT.args.json. The verdict is read from the JSON the script writes,
# never from this exit code (Isaac's shutdown hard-exits with 0 after a crash).
set -uo pipefail
: "${BENCH_OUT:?BENCH_OUT not forwarded}" "${PYTHONPATH:?PYTHONPATH not forwarded}"
SRC_ROOT=${PYTHONPATH%%:*}
if [ ! -f "$SRC_ROOT/bhl_robust/__init__.py" ]; then
    echo "inner_stand3_replay: $SRC_ROOT is not a bhl_robust source root"; exit 1
fi
export PYTHONPATH="$SRC_ROOT"
cd "$REPO"
timeout --kill-after=60 2400 "$PY" "$REPO/scripts/bench/stand3_replay.py" isaac \
    --args-file "$BENCH_OUT.args.json" 2>&1 | grep -vE "pthread_setaffinity|EGLError"
exit 0
