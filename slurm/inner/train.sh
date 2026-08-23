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

# The exit code of this python cannot be trusted, and that is not a guess.
# scripts/train.py ends with `main(); simulation_app.close()`, main() is
# Hydra-decorated, and Isaac Sim's shutdown hard-exits the interpreter. A config
# error therefore prints a full traceback and returns 0 -- which is how four
# dead crew-lift arms came back from Slurm marked COMPLETED, and how a gate that
# ran this same script reported four PASSes in sixty-eight seconds.
#
# So the run is judged on evidence instead: a training job that never reached
# its first iteration did not train, whatever it told the shell.
LOG=$(mktemp "${TMPDIR:-/tmp}/bhl-train-XXXXXX.log")
# shellcheck disable=SC2086
$PY "$TRAIN_SCRIPT" "${ARGS[@]}" $OVERRIDES 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

if grep -qE "Learning iteration" "$LOG"; then
    iters=$(grep -cE "Learning iteration" "$LOG")
    echo "train.sh: reached $iters logged iterations"
else
    echo "train.sh: FAILED -- no training iteration was ever logged" >&2
    grep -hoE "^(ValueError|RuntimeError|TypeError|KeyError|AttributeError|ModuleNotFoundError): .{0,160}" \
        "$LOG" | tail -3 >&2
    rm -f "$LOG"
    exit 1
fi
rm -f "$LOG"
exit "$rc"
