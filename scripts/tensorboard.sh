#!/bin/bash
# Serve the training curves for all runs.
#
# Runs on the LOGIN node, outside the container: uv's managed CPython is a
# portable standalone build, so it works on the login node's glibc 2.28 even
# though Isaac Sim itself cannot.
#
# In VS Code Remote, the forwarded port is picked up automatically -- just open
# the Ports panel. Over plain ssh, tunnel it yourself:
#     ssh -N -L 6006:localhost:6006 sanchej7@submit-b.hpc.engr.oregonstate.edu
#
# Usage: ./scripts/tensorboard.sh [port]

set -euo pipefail

PORT=${1:-6006}
WORKSPACE=/nfs/hpc/share/$USER/Humanoid_Lite
LOGDIR=$WORKSPACE/bhl-robustness-ladder/external/Berkeley-Humanoid-Lite/logs/rsl_rl/biped

if [ ! -d "$LOGDIR" ]; then
    echo "No runs yet at $LOGDIR" >&2
    exit 1
fi

echo "runs found:"
ls -1 "$LOGDIR" | sed 's/^/  /'
echo
echo "serving on http://localhost:$PORT"
echo "  Curriculum/push_levels  -> the push ramp (experiment 1)"
echo "  Train/mean_reward       -> per-rung learning curves (experiment 2)"
echo

exec "$WORKSPACE/venv/bin/tensorboard" \
    --logdir "$LOGDIR" \
    --port "$PORT" \
    --bind_all \
    --reload_multifile true
