#!/bin/bash
set -uo pipefail
cd "$UPSTREAM"
run="$RUN_DIR"
timeout 900 "$PY" "$REPO/scripts/train_play.py" \
    --task "${TASK:-Velocity-Berkeley-Humanoid-Lite-v0}" \
    --headless --num_envs 8 --load_run "$run" 2>&1 | tail -3
d="logs/rsl_rl/humanoid/$run"
if [ -f "$d/exported/policy.onnx" ]; then
    cp configs/policy_latest.yaml "$d/exported/deploy.yaml"
    sed -i "s|^policy_checkpoint_path:.*|policy_checkpoint_path: \"$UPSTREAM/$d/exported/policy.onnx\"|" \
        "$d/exported/deploy.yaml"
    echo "deploy.yaml written"
fi
