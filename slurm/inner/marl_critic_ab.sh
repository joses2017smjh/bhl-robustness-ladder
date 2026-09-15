#!/bin/bash
# A/B on the MARL critic, stairs + depth, seed 0. Tracking reward and terrain
# level at matched iterations against rsl-rl PPO on the same task decide whether
# the plateau in 21328607/8 was the critic. The arm arrives in forwarded vars
# (BHL_PARTITION, BHL_ALGO, RUN_NAME); SLURM_ARRAY_TASK_ID does not cross --cleanenv.
set -uo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
C=privileged; case "$RUN_NAME" in *-policy-*) C=policy ;; esac
LOG=$(mktemp "${TMPDIR:-/tmp}/bhl-ab-XXXXXX.log")
"$PY" scripts/train_marl.py --task Velocity-BHL-Biped-Stairs-Depth-v0 --num_envs 4096 --seed 0 \
    --max_iterations "${MAX_ITER:-1500}" --run_name "$RUN_NAME" \
    --partition "$BHL_PARTITION" --algo "$BHL_ALGO" --critic "$C" --hparams rsl --headless 2>&1 | tee "$LOG"
last=$(grep -aoE "[0-9]+/[0-9]+ \[" "$LOG" | tail -1 || true)
n=${last%%/*}; t=${last#*/}; t=${t%% *}
[ -n "$last" ] && [ "$n" = "$t" ] || { echo "FAILED -- last progress '${last:-none}'" >&2; exit 1; }
