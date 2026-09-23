#!/bin/bash
# Spawn-height sweep for one TaskV2 task: one Kit boot per offset, pd_hold only.
# Reads TASK, NUM_ENVS, PROBE_STEPS, SPAWN_DIAG_OUT, Z_OFFSETS from the environment.
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
OUT="${SPAWN_DIAG_OUT:-$REPO/results/repo-gpu-20260923/spawn_zsweep}"; mkdir -p "$OUT"
for off in ${Z_OFFSETS:-0 -0.05 -0.095 -0.12}; do
  tag="z${off}"; LOG="$OUT/${TASK}__${tag}.log"
  echo "=== inner: $TASK | offset $off | envs ${NUM_ENVS:-4} | steps ${PROBE_STEPS:-200} | $(date) ==="
  set +e
  timeout 900 "$PY" scripts/bench/task_v2_spawn_diag.py --headless --task "$TASK" --num_envs "${NUM_ENVS:-4}" \
      --steps "${PROBE_STEPS:-200}" --conditions pd_hold --init_z_offset "$off" --tag "$tag" --out_dir "$OUT" 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}; set -e
  grep -E "init_state z|^SPAWN-DIAG |^    " "$LOG" || true
  echo "=== inner exit $rc for offset $off ==="
done
