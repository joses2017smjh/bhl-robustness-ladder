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
# The first block (run names marl-*) trained on skrl's default PPO settings; keep
# those reproducible. Everything else reads the rsl-rl control's settings.
case "$RUN_NAME" in marl-*) ARGS+=(--hparams skrl-default --critic policy) ;; *) ARGS+=(--hparams rsl --critic privileged) ;; esac

"$PY" scripts/train_marl.py "${ARGS[@]}" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

# Same guard as train.sh, and for the same reason: a Hydra-caught exception
# followed by simulation_app.close() hard-exits 0, so four dead crew arms once
# reported COMPLETED. An arm that logged no iteration did not train.
# skrl's SequentialTrainer prints a tqdm bar, not rsl-rl's "Learning iteration".
# Grepping only for the rsl-rl wording marked three completed six-hour runs as
# FAILED -- 144,000/144,000 timesteps each, with checkpoints on disk -- and came
# within one requeue of throwing away eighteen GPU-hours of finished work. A
# guard that reports a false negative is as expensive as one that reports a
# false positive; it just fails in the other direction.
# Judge the run by its last progress count, which must equal its total. The
# first version passed on any tqdm line or on the run directory -- and skrl
# prints "0/N [" and makes that directory before its first step, so a run that
# died at step 0 (21330372_0, KeyError) was reported COMPLETED. The one before it
# grepped only rsl-rl's wording and failed three finished six-hour runs. Both
# directions cost a day; the count is the evidence.
last=$(grep -aoE "[0-9]+/[0-9]+ \[" "$LOG" | tail -1 || true)
n=${last%%/*}; t=${last#*/}; t=${t%% *}
if [ -n "$last" ] && [ "$n" = "$t" ]; then
    echo "marl_train.sh: training ran to ${last% [}"
else
    echo "marl_train.sh: FAILED -- last progress '${last:-none}' is not the full run" >&2
    exit 1
fi
exit $rc
