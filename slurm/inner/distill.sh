#!/bin/bash
# Distillation driver. Same shape as inner/train.sh, plus the resume flags that
# point the student at its teacher's checkpoint.
set -euo pipefail
cd "$UPSTREAM"
: "${TASK:?}"; : "${RUN_NAME:?}"; : "${SEED:?}"; : "${LOAD_RUN:?}"

echo "task=$TASK run=$RUN_NAME seed=$SEED teacher=$LOAD_RUN iters=${MAX_ITER:-cfg}"
ARGS=(
    --task "$TASK" --headless --seed "$SEED" --run_name "$RUN_NAME"
    --resume True --load_run "$LOAD_RUN"
)
[ -n "${NUM_ENVS:-}" ] && ARGS+=(--num_envs "$NUM_ENVS")
[ -n "${MAX_ITER:-}" ] && ARGS+=(--max_iterations "$MAX_ITER")

$PY "${TRAIN_SCRIPT:-$REPO/scripts/train_distill.py}" "${ARGS[@]}"
