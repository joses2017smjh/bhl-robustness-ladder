#!/bin/bash
# Hand-height measurement for one TaskV2 task, one Kit boot, PD-hold only:
#   standing_pinch_arms : upstream standing legs, arms at the coop_lift_env_cfg
#                         _PINCH_JOINT_POS values, root z = 0 (upstream standing height)
# Reads TASK, NUM_ENVS, PROBE_STEPS, SPAWN_DIAG_OUT from the environment.
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
OUT="${SPAWN_DIAG_OUT:-$REPO/results/repo-gpu-20260923/spawn_hands}"; mkdir -p "$OUT"
tag=standing_pinch_arms; LOG="$OUT/${TASK}__${tag}.log"
echo "=== inner: $TASK | $tag | envs ${NUM_ENVS:-4} | steps ${PROBE_STEPS:-60} | $(date) ==="
set +e
timeout 900 "$PY" scripts/bench/task_v2_spawn_diag.py --headless --task "$TASK" --num_envs "${NUM_ENVS:-4}" \
    --steps "${PROBE_STEPS:-60}" --conditions pd_hold --init_joint_pose standing_pinch_arms --init_z 0.0 \
    --tag "$tag" --out_dir "$OUT" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}; set -e
grep -E "init_state z|init joint_pos|configured object spawn|hand bodies|^SPAWN-DIAG |^HAND-HEIGHT |^    " "$LOG" || true
echo "=== inner exit $rc for $tag ==="
