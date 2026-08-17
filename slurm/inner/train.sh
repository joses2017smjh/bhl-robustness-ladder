#!/bin/bash
# Generic RSL-RL training driver, parameterised by environment variables so the
# same script serves the baseline, the push curriculum, the DR ladder, and the
# terrain runs.
#
# Required: TASK, RUN_NAME, SEED
# Optional: NUM_ENVS, MAX_ITER, OVERRIDE_FILE (hydra overrides), TRAIN_SCRIPT
#
# Every variable here must appear in BHL_FORWARD_VARS in slurm/_env.sh --
# --cleanenv drops anything not forwarded explicitly.
set -euo pipefail
cd "$UPSTREAM"

: "${TASK:?TASK not set (is it listed in BHL_FORWARD_VARS in _env.sh?)}"
: "${RUN_NAME:?RUN_NAME not set}"
: "${SEED:?SEED not set}"

# Overlay tasks need the vendored entrypoint, which registers bhl_robust ids
# after SimulationApp starts. Upstream tasks can use either.
TRAIN_SCRIPT=${TRAIN_SCRIPT:-$REPO/scripts/train.py}

echo "task=$TASK run=$RUN_NAME seed=$SEED envs=${NUM_ENVS:-cfg} iters=${MAX_ITER:-cfg}"
echo "entrypoint=$TRAIN_SCRIPT"

ARGS=(
    --task "$TASK"
    --headless
    --seed "$SEED"
    --run_name "$RUN_NAME"
)
[ -n "${NUM_ENVS:-}" ] && ARGS+=(--num_envs "$NUM_ENVS")
[ -n "${MAX_ITER:-}" ] && ARGS+=(--max_iterations "$MAX_ITER")

# Hydra overrides arrive as a file, one whitespace-separated blob, because
# Apptainer's --env cannot carry a value containing commas. Word-splitting is
# intended here: each override must become its own argv entry.
OVERRIDES=""
if [ -n "${OVERRIDE_FILE:-}" ] && [ -s "${OVERRIDE_FILE}" ]; then
    OVERRIDES=$(tr -s '[:space:]' ' ' < "$OVERRIDE_FILE")
    echo "overrides: $OVERRIDES"
fi

# shellcheck disable=SC2086
$PY "$TRAIN_SCRIPT" "${ARGS[@]}" $OVERRIDES
