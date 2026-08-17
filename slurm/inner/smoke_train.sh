#!/bin/bash
# Gate check: Isaac Sim launches headless on Turing behind Apptainer, the BHL
# biped task registers, and PPO steps. Not a training run.
set -euo pipefail
cd "$UPSTREAM"

echo "--- registered BHL tasks ---"
$PY ./scripts/list_envs.py 2>&1 | grep -i "berkeley\|Task" | head -20 \
    || echo "(list_envs produced no match; continuing to train)"

echo
echo "--- smoke training: 10 iterations, 64 envs ---"
$PY ./scripts/rsl_rl/train.py \
    --task Velocity-Berkeley-Humanoid-Lite-Biped-v0 \
    --headless \
    --num_envs 64 \
    --max_iterations 10 \
    --seed 0
