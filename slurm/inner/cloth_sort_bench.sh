#!/bin/bash
# Isaac throughput. Cost-gated inside the Python. Not a training job.
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
mkdir -p "$REPO/results/cloth"
"$PY" "$REPO/scripts/cloth/estimate_cost.py" \
    --physics deformable --num-envs "${NUM_ENVS:-8}" \
    --iterations 1 --steps-per-iter "${BENCH_STEPS:-40}" --max-hours 2
"$PY" "$REPO/scripts/bench/cloth_sort_isaac_bench.py" \
    --headless --steps "${BENCH_STEPS:-40}" \
    --out-csv "$REPO/results/cloth_sort_bench.csv" \
    --out-md "$REPO/results/cloth_sort_bench.md"
echo "=== done: $(date) ==="
