#!/bin/bash
set -uo pipefail
cd "$UPSTREAM"
run="$RUN_DIR"
echo "exporting: $run"
timeout 600 "$PY" "$REPO/scripts/train_play.py" \
    --task Velocity-Berkeley-Humanoid-Lite-Biped-v0 \
    --headless --num_envs 8 --load_run "$run" 2>&1 | tail -5
echo "--- artifacts ---"
ls -la "logs/rsl_rl/biped/$run/exported/" 2>/dev/null || echo "NO exported/ dir"
if [ -f "logs/rsl_rl/biped/$run/exported/policy.onnx" ]; then
    cp configs/policy_latest.yaml "logs/rsl_rl/biped/$run/exported/deploy.yaml"
    sed -i "s|^policy_checkpoint_path:.*|policy_checkpoint_path: \"$UPSTREAM/logs/rsl_rl/biped/$run/exported/policy.onnx\"|" \
        "logs/rsl_rl/biped/$run/exported/deploy.yaml"
    echo "deploy.yaml written"
fi
