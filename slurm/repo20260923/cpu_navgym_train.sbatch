#!/bin/bash
#SBATCH --job-name=navgym
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=haswell&el8
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=36:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# 2026-09-24: NavGym -- a Gymnasium maze-navigation environment on the biped
# gait's command interface, trained with Stable-Baselines3 PPO on CPU with a
# maze-size curriculum, evaluated on held-out mazes (seeds >= 10 000), then
# transferred onto the physics robot by scripts/bench/maze_explore.py --policy.
# Predeclared rules are in SLURM_JOBS.md.
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
OUT=${NAVGYM_OUT:-$REPO/results/navgym-20260924/run-${NAVGYM_SEED:-0}}
mkdir -p "$OUT"
echo "=== node: $(hostname) | HEAD $(git rev-parse --short HEAD) | seed ${NAVGYM_SEED:-0} | steps ${NAVGYM_STEPS:-60000000} | $(date) ==="
"$PY" scripts/bench/navgym_train.py --out "$OUT" --seed "${NAVGYM_SEED:-0}" --total-steps "${NAVGYM_STEPS:-60000000}" \
    --n-envs "${NAVGYM_ENVS:-16}" --eval-every "${NAVGYM_EVAL_EVERY:-1000000}" 2>&1 | grep -vE "^\[|Warning"
echo "=== done: $(date) ==="
