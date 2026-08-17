#!/bin/bash
# Export every trained run to ONNX + a deploy YAML.
#
# play.py writes the ONNX and the YAML BEFORE entering its infinite render loop,
# so a timeout is a safe way to stop it. It also writes the YAML to a single
# fixed path (configs/policy_latest.yaml) that each run would overwrite, so the
# file is copied into the run directory immediately after each export.
set -uo pipefail
cd "$UPSTREAM"

LOGROOT=logs/rsl_rl/biped
exported=0; skipped=0; failed=0

for run in "$LOGROOT"/*/; do
    name=$(basename "$run")
    [ -f "$run/exported/policy.onnx" ] && { echo "skip (already exported): $name"; skipped=$((skipped+1)); continue; }
    ls "$run"/model_*.pt >/dev/null 2>&1 || { echo "skip (no checkpoint): $name"; skipped=$((skipped+1)); continue; }

    echo "=== exporting $name ==="
    timeout 600 "$PY" "$REPO/scripts/train_play.py" \
        --task "${TASK:-Velocity-Berkeley-Humanoid-Lite-Biped-v0}" \
        --headless --num_envs 8 --load_run "$name" >/dev/null 2>&1

    if [ -f "$run/exported/policy.onnx" ]; then
        cp configs/policy_latest.yaml "$run/exported/deploy.yaml" 2>/dev/null
        # The YAML records an absolute-ish path; repoint it at this run.
        sed -i "s|^policy_checkpoint_path:.*|policy_checkpoint_path: \"$UPSTREAM/$run/exported/policy.onnx\"|" \
            "$run/exported/deploy.yaml"
        echo "  ok -> $run/exported/"
        exported=$((exported+1))
    else
        echo "  FAILED: $name"
        failed=$((failed+1))
    fi
done

echo "exported=$exported skipped=$skipped failed=$failed"
