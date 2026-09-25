#!/bin/bash
set -uo pipefail
cd "$UPSTREAM"
run="$RUN_DIR"
# --play-steps 5: train_play.py exports, writes configs/policy_latest.yaml and
# then enters `while simulation_app.is_running()`; without a step cap it plays
# forever. `timeout` alone only sends SIGTERM, which Kit ignores, so the
# 2026-09-24 turning-gait exports spun for 12 h after writing policy.onnx
# (21412131/21412132, and arms-push-s2 before them). --kill-after escalates.
timeout --kill-after=60 900 "$PY" "$REPO/scripts/train_play.py" \
    --task "${TASK:-Velocity-Berkeley-Humanoid-Lite-v0}" \
    --headless --num_envs 8 --play-steps 5 --load_run "$run" > "/tmp/export-$run.log" 2>&1
tail -3 "/tmp/export-$run.log"
d="logs/rsl_rl/humanoid/$run"
if [ -f "$d/exported/policy.onnx" ]; then
    cp configs/policy_latest.yaml "$d/exported/deploy.yaml"
    sed -i "s|^policy_checkpoint_path:.*|policy_checkpoint_path: \"$UPSTREAM/$d/exported/policy.onnx\"|" \
        "$d/exported/deploy.yaml"
    echo "deploy.yaml written"
fi
