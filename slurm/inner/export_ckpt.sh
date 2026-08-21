#!/bin/bash
# Export ONE named checkpoint of a run, then move the artifacts somewhere they
# will survive the next export.
#
# train_play.py writes to `<run>/exported/`, derived from the checkpoint's own
# directory, so every checkpoint of a run targets the same folder and the last
# export wins. The sweep needs six of them side by side, so each is relocated to
# `<run>/exported_iter<N>/` immediately after it is written.
#
# Required: RUN_DIR, CKPT (e.g. model_3000.pt)
set -uo pipefail
cd "$UPSTREAM"
run="$RUN_DIR"; ckpt="$CKPT"
iter=$(echo "$ckpt" | sed 's/[^0-9]*\([0-9]*\).*/\1/')
dst="logs/rsl_rl/${EXP:-biped}/$run/exported_iter${iter}"

if [ -f "$dst/deploy.yaml" ]; then echo "have $run @ $iter"; exit 0; fi

echo "exporting: $run @ $ckpt"
timeout 900 "$PY" "$REPO/scripts/train_play.py" \
    --task "${TASK:-Velocity-Berkeley-Humanoid-Lite-Biped-v0}" \
    --headless --num_envs 8 --load_run "$run" --checkpoint "$ckpt" 2>&1 | tail -3

src="logs/rsl_rl/${EXP:-biped}/$run/exported"
if [ -f "$src/policy.onnx" ]; then
    mkdir -p "$dst"
    cp "$src/policy.onnx" "$dst/policy.onnx"
    cp configs/policy_latest.yaml "$dst/deploy.yaml"
    sed -i "s|^policy_checkpoint_path:.*|policy_checkpoint_path: \"$UPSTREAM/$dst/policy.onnx\"|" \
        "$dst/deploy.yaml"
    echo "wrote $dst/deploy.yaml"
else
    echo "EXPORT FAILED: $run @ $ckpt"
fi
