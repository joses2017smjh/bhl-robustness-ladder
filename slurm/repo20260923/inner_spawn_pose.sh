#!/bin/bash
# Posture comparison for one TaskV2 task, PD-hold only, one Kit boot each:
#   pinch    : configured crouch, feet placed on the floor (z offset -0.095)
#   standing : upstream standing joint pose at the upstream root z = 0
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
OUT="${SPAWN_DIAG_OUT:-$REPO/results/repo-gpu-20260923/spawn_pose}"; mkdir -p "$OUT"
run() { tag=$1; shift; LOG="$OUT/${TASK}__${tag}.log"
  echo "=== inner: $TASK | $tag | $(date) ==="; set +e
  timeout 900 "$PY" scripts/bench/task_v2_spawn_diag.py --headless --task "$TASK" --num_envs "${NUM_ENVS:-4}" \
      --steps "${PROBE_STEPS:-200}" --conditions pd_hold --tag "$tag" --out_dir "$OUT" "$@" 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}; set -e; grep -E "init_state z|init joint_pos|^SPAWN-DIAG |^    " "$LOG" || true; echo "=== inner exit $rc for $tag ==="; }
run pinch --init_z_offset -0.095
run standing --init_joint_pose standing --init_z 0.0
run standing_z-0.03 --init_joint_pose standing --init_z -0.03
