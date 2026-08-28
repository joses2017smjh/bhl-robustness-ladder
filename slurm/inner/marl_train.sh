#!/bin/bash
# Train one Tier-1 row. PPO goes through the existing path; MARL goes through
# scripts/train_marl.py, which is the only file that knows about agents.
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

if [ "${BHL_ALGO:-ppo}" = "ppo" ]; then
    exec "$REPO/slurm/inner/train.sh"
fi

LOG=$(mktemp "${TMPDIR:-/tmp}/bhl-marl-XXXXXX.log")
ARGS=(--task "$TASK" --num_envs "$NUM_ENVS" --seed "$SEED"
      --max_iterations "$MAX_ITER" --run_name "$RUN_NAME"
      --partition "$BHL_PARTITION" --algo "$BHL_ALGO" --headless)
[ "${BHL_ABLATE_ARM_DEV:-0}" = "1" ] && ARGS+=(--ablate-arm-deviation)

"$PY" scripts/train_marl.py "${ARGS[@]}" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

# Same guard as train.sh, and for the same reason: a Hydra-caught exception
# followed by simulation_app.close() hard-exits 0, so four dead crew arms once
# reported COMPLETED. An arm that logged no iteration did not train.
if grep -qE "Learning iteration|iteration [0-9]+/" "$LOG"; then
    echo "marl_train.sh: reached $(grep -cE 'Learning iteration|iteration [0-9]+/' "$LOG") logged iterations"
else
    echo "marl_train.sh: FAILED -- no training iteration was ever logged" >&2
    exit 1
fi
exit $rc
