#!/bin/bash
set -euo pipefail
cd "$UPSTREAM"
source "$REPO/slurm/dr_levels.sh"
echo "### overrides being applied (aggressive) ###"
dr_overrides aggressive
$PY ./scripts/rsl_rl/train.py --task Velocity-Berkeley-Humanoid-Lite-Biped-v0 \
    --headless --num_envs 32 --max_iterations 2 --seed 0 \
    --experiment_name _hydracheck --run_name syntax \
    $(dr_overrides aggressive) 2>&1 | grep -iE "error|exception|override|friction|Iteration time|Total timesteps" | head -25
