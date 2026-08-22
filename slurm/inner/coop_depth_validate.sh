#!/bin/bash
# Validate the payload-tracking depth camera before anything trains on it.
set -euo pipefail
cd "$UPSTREAM"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

# Both arms are validated, not just the one that trains first. They differ in
# observation width, and the width is what the MuJoCo replay keys off, so a
# wrong layout in either arm would only surface at replay time -- after the GPU
# hours were already spent.
"$PY" "$REPO/scripts/bench/coop_depth_validate.py" \
    --num_envs "${NUM_ENVS:-4}" --headless
"$PY" "$REPO/scripts/bench/coop_depth_validate.py" \
    --num_envs "${NUM_ENVS:-4}" --keep-object-pose --headless
