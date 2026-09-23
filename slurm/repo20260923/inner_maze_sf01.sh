#!/bin/bash
# SF-01 / SF-02 (Isaac side): one Full-stage checkpoint per boot, all settings in one process.
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
: "${VARIANT:?}" "${CKPT:?}" "${SEED:?}"; TASK="Velocity-BHL-MazeRecovery-Full-${VARIANT}-v0"
OUT="${MAZE_SF01_OUT:-$REPO/results/repo-gpu-20260923/maze-sf01}"; mkdir -p "$OUT"
arm=$(echo "$TASK" | sed -E 's/.*Full-([A-Za-z]+)-v0/\1/')
json="$OUT/${arm}-s${SEED}.json"; log="$OUT/${arm}-s${SEED}.log"
echo "=== inner: $TASK | seed $((100 + SEED)) | ckpt $CKPT | $(date) ==="
set +e
timeout 2400 "$PY" "$REPO/scripts/bench/maze_recovery_probe.py" --headless --enable_cameras \
    --task "$TASK" --num-envs 32 --steps 600 --seed $((100 + SEED)) \
    --checkpoint "$CKPT" --keep-corruption --output "$json" \
    --settings '[{"name":"baseline_noise_on"},{"name":"gyro0.10","gyro_std":0.10},{"name":"gyro0.20","gyro_std":0.20},{"name":"grav0.05","gravity_std":0.05},{"name":"grav0.10","gravity_std":0.10},{"name":"delay1","imu_delay_steps":1},{"name":"delay2","imu_delay_steps":2},{"name":"delay4","imu_delay_steps":4},{"name":"pos0.05","pose_bias_m":0.05},{"name":"pos0.10","pose_bias_m":0.10},{"name":"pos0.20","pose_bias_m":0.20},{"name":"posnoise0.05","pose_noise_m":0.05},{"name":"yaw3","pose_yaw_deg":3},{"name":"yaw10","pose_yaw_deg":10}]' > "$log" 2>&1
rc=$?; set -e
grep -E '^\{"name"' "$log" | cut -c1-200 || true
echo "=== inner exit $rc for $TASK s$SEED ==="
