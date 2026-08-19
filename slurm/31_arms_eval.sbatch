#!/bin/bash
#SBATCH --job-name=arms-eval
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Export and score the 22-DoF policies, mirroring the biped protocol exactly so
# the two robots are comparable. Raw training reward is NOT comparable between
# them -- the humanoid reward set carries extra arm-deviation penalties -- so
# sim2sim fall rate is the only cross-robot number that means anything.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=4
export CACHE_DIR=$WORKSPACE/mjcf_cache
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS TERRAIN_D RUN_DIR VARIANT"
LOGROOT=$UPSTREAM/logs/rsl_rl/humanoid
mkdir -p "$REPO/results/arms" "$REPO/results/demo_arms"

for run in "$LOGROOT"/*/; do
    name=$(basename "$run"); [ "$name" = isaaclab ] && continue
    label=$(echo "$name" | sed 's/^[0-9-]*_[0-9-]*_//')
    ls "$run"/model_*.pt >/dev/null 2>&1 || continue

    if [ ! -f "$run/exported/policy.onnx" ]; then
        echo "=== export $label ==="
        export RUN_DIR=$name TASK=Velocity-Berkeley-Humanoid-Lite-v0
        bhl_exec "$REPO/slurm/inner/export_one_humanoid.sh" >/dev/null 2>&1 || true
    fi
    [ -f "$run/exported/deploy.yaml" ] || { echo "  export FAILED: $label"; continue; }

    export DEPLOY_CFG=$run/exported/deploy.yaml LABEL=$label VARIANT=humanoid
    export EPISODE_S=10 N_SEEDS=5

    # Flat, plus the two conditions the comparison clips need.
    for spec in "0:0" "0:0.20" "0.5:0"; do
        PUSH=${spec%%:*}; D=${spec##*:}
        export PUSH_SPEED=$PUSH TERRAIN_D=$D
        export OUT_CSV=$REPO/results/arms/${label}__p${PUSH}_d${D}.csv
        [ -f "$OUT_CSV" ] && continue
        echo "=== $label  push=$PUSH terrain=$D ==="
        bhl_exec "$REPO/slurm/inner/eval_one_variant.sh" 2>&1 | grep -E "falls" || true
    done
done
echo "=== done $(date) ==="
