#!/bin/bash
#SBATCH --job-name=terr-high
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Extends the terrain retention curve past d=0.4 with the full seed count.
# The first sweep stopped at 0.4, where the flat+DR policy still looks perfect;
# the interesting separation is above it, and single-seed spot checks are not
# enough to report.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=4
export CACHE_DIR=$WORKSPACE/mjcf_cache VIDEO_DIR="" EPISODE_S=10 N_SEEDS=5 PUSH_SPEED=0
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS TERRAIN_D"
R=$UPSTREAM/logs/rsl_rl/biped
mkdir -p "$REPO/results/terrain"

for pol in dr-default-s0 dr-default-s1 dr-default-s2 \
           terrain-bumpy-s0 terrain-bumpy-s1 terrain-bumpy-s2 \
           terrain-smooth-s0 terrain-smooth-s1; do
    d0=$(ls -d $R/*_${pol} 2>/dev/null | head -1)
    [ -n "$d0" ] && [ -f "$d0/exported/deploy.yaml" ] || continue
    for d in 0.60 0.80 1.00; do
        out=$REPO/results/terrain/${pol}__d${d}.csv
        [ -f "$out" ] && continue
        export DEPLOY_CFG=$d0/exported/deploy.yaml LABEL=$pol TERRAIN_D=$d OUT_CSV=$out
        echo "=== $pol @ d=$d ==="
        bhl_exec "$REPO/slurm/inner/eval_one.sh" 2>&1 | grep -E "falls" || true
    done
done
echo "=== done $(date) ==="
