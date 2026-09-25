#!/bin/bash
#SBATCH --job-name=maze-explore
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=haswell&el8
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# 2026-09-24, user request: randomized mazes, sensor-driven, turn-then-walk.
# The multi-seed table for two maze configurations (CPU MuJoCo, no render):
#   base    5x5 cells, 2 extra openings   (MAZE_BASE)
#   harder  6x6 cells, 1 extra opening    (MAZE_HARD)
# Predeclared before submission (SLURM_JOBS.md): 12 seeds each, success =
# goal within 150 s with no fall; clean = also no wall contact. The GIF is
# rendered from the harder configuration if it scores >= 10/12 clean, else
# from the base configuration; the seed is the median clean completion time,
# never the fastest.
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
CACHE=${TMPDIR:-/tmp}/bhl-maze-explore-$SLURM_JOB_ID; mkdir -p "$CACHE"
OUT=$REPO/results/maze-explore-20260924; mkdir -p "$OUT"
SEEDS=${MAZE_SEEDS:-12}
echo "=== node: $(hostname) | HEAD $(git rev-parse --short HEAD) | $(date) ==="
echo "--- base 5x5 / 2 openings"
"$PY" scripts/bench/maze_explore.py --upstream "$U" --cache-dir "$CACHE/base" --seeds "$SEEDS" --n 5 --m 5 --extra-openings 2 \
    --out-dir "$OUT/base-5x5" --time-limit 150 2>&1 | grep -v "^\["
echo "--- harder 6x6 / 1 opening"
"$PY" scripts/bench/maze_explore.py --upstream "$U" --cache-dir "$CACHE/hard" --seeds "$SEEDS" --n 6 --m 6 --extra-openings 1 \
    --out-dir "$OUT/hard-6x6" --time-limit 180 2>&1 | grep -v "^\["
rm -rf "$CACHE"
echo "=== done: $(date) ==="
