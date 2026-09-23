#!/bin/bash
# Inside the container (via bhl_exec). One task per Python process: Isaac Lab
# allows one SimulationContext per process, so the sbatch calls this once per
# gym id. Reads TASK, NUM_ENVS, PROBE_STEPS, SPAWN_DIAG_OUT from the environment.
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
: "${TASK:?TASK must be forwarded}"
OUT="${SPAWN_DIAG_OUT:-$REPO/results/repo-gpu-20260923/spawn_diag}"
mkdir -p "$OUT"
LOG="$OUT/${TASK}.log"
echo "=== inner: $TASK | envs ${NUM_ENVS:-4} | steps ${PROBE_STEPS:-200} | stack ${BHL_STACK:-?} | $(date) ==="
set +e
timeout 1200 "$PY" scripts/bench/task_v2_spawn_diag.py --headless \
    --task "$TASK" \
    --num_envs "${NUM_ENVS:-4}" \
    --steps "${PROBE_STEPS:-200}" \
    --conditions zero pd_hold gauss \
    --out_dir "$OUT" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
grep -E "^task |^  robot_|^SPAWN-DIAG |^    |^wrote " "$LOG" || true
echo "=== inner exit $rc for $TASK ==="
exit "$rc"
