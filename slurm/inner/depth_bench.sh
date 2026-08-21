#!/bin/bash
# Validate the ray-cast depth against closed-form geometry, then benchmark it.
set -euo pipefail
cd "$UPSTREAM"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" "$REPO/scripts/bench/depth_validate.py" --res 64 --num_envs 16 || true
for cfg in "--no_camera --num_envs 2048" "--num_envs 2048 --res 64" \
           "--no_camera --num_envs 4096" "--num_envs 4096 --res 48"; do
    # shellcheck disable=SC2086
    "$PY" "$REPO/scripts/bench/depth_raycast.py" $cfg --steps 60 || true
done
