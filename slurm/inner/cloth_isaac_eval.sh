#!/bin/bash
# Cheap C0 Isaac scripted eval. Not C2 cloth, not 8k training.
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
mkdir -p "$REPO/results/cloth"
"$PY" "$REPO/scripts/cloth/eval_isaac.py" --headless \
    --rung "${RUNG:-C0}" \
    --num_envs "${NUM_ENVS:-4}" \
    --episodes "${EVAL_EPISODES:-4}" \
    --max_steps "${EVAL_STEPS:-80}" \
    --out "${BENCH_OUT:-$REPO/results/cloth/isaac_c0_scripted.json}"
echo "=== done: $(date) ==="
