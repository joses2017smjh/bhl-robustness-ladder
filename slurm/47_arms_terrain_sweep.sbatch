#!/bin/bash
#SBATCH --job-name=arms-terr
#SBATCH --account=eecs
#SBATCH --partition=gpu,eecs,dgxh,ampere,preempt
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g|rtx6000|rtx2080|t4"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# sm_75 or newer, any VRAM -- eval starts SimulationApp briefly then runs on CPU.
# Excludes GTX 1080 (sm_61) and V100 (sm_70), which el9 alone would let through.

# Finish the 22-DoF retention curve: d = 0.40 through 1.00.
#
# The arms row in section 3 stops at d = 0.20, which is exactly where the biped
# curve starts separating -- randomization-only first cracks at 0.40 and reaches
# 37.8% by 1.00. Everything below that is a region where every policy is at zero
# and the two morphologies are trivially indistinguishable.
#
# The specific question the arms make interesting: a humanoid rejects a lateral
# disturbance partly by swinging its arms to shed angular momentum, which a
# 12-DoF biped cannot do. Rough ground is a disturbance the arms could in
# principle help with -- and upstream penalises arm deviation, so this may end
# up measuring that penalty suppressing the strategy the arms would supply.
#
# Eval only. A GPU is requested for the ONNX export path; the scoring is CPU.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=8
export CACHE_DIR=$WORKSPACE/mjcf_cache
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS TERRAIN_D RUN_DIR VARIANT"
LOGROOT=$UPSTREAM/logs/rsl_rl/humanoid
mkdir -p "$REPO/results/arms"

DIFFS="0.40 0.60 0.80 1.00"

for run in "$LOGROOT"/*/; do
    name=$(basename "$run"); [ "$name" = isaaclab ] && continue
    label=$(echo "$name" | sed 's/^[0-9-]*_[0-9-]*_//')
    ls "$run"/model_*.pt >/dev/null 2>&1 || continue

    if [ ! -f "$run/exported/policy.onnx" ]; then
        export RUN_DIR=$name TASK=Velocity-Berkeley-Humanoid-Lite-v0
        bhl_exec "$REPO/slurm/inner/export_one_humanoid.sh" >/dev/null 2>&1 || true
    fi
    [ -f "$run/exported/deploy.yaml" ] || { echo "  no export: $label"; continue; }

    export DEPLOY_CFG=$run/exported/deploy.yaml LABEL=$label VARIANT=humanoid
    export EPISODE_S=10 N_SEEDS=5 PUSH_SPEED=0 VIDEO_DIR=""

    for d in $DIFFS; do
        export TERRAIN_D=$d
        export OUT_CSV=$REPO/results/arms/${label}__p0_d${d}.csv
        [ -f "$OUT_CSV" ] && { echo "have $label d=$d"; continue; }
        echo "=== $label @ terrain d=$d ==="
        bhl_exec "$REPO/slurm/inner/eval_one_variant.sh" 2>&1 | grep -E "falls" || true
    done
done

echo "=== arms terrain sweep done: $(date) ==="
ls -1 "$REPO/results/arms"/*.csv 2>/dev/null | wc -l | sed 's/^/csv files: /'
