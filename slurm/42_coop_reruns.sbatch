#!/bin/bash
#SBATCH --job-name=coop-rerun
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# sm_75 or newer AND >=32 GB. The el9 tag is not enough: preempt mixes in
# GTX 1080 (sm_61) and V100 (sm_70), both below the pinned torch's cubin floor,
# and the 11 GB cards cannot hold this run's env count.

# Rerun the two observation-ablation arms that never actually ran.
#
# `nopriv` and `notrack` are consumed in `CoopLiftEnvCfg.__post_init__`, and
# Isaac Lab's hydra overlay writes onto the cfg AFTER that, so both trained as
# bit-identical copies of the control (policy 194 / critic 206 either way). The
# README says so, which is honest, but three rows of non-evidence in a published
# table is a correctness debt, not a caveat. `apply_strategy_flags` is now
# called again from scripts/train.py after hydra returns, so these are real.
#
# The third dead arm, `staged`, is 43_pickfirst_then_height -- same flag, and it
# is the experiment worth naming separately rather than a duplicate GPU here.
#
# This is also the first actual measurement of whether the privileged critic is
# load-bearing. Right now that claim rests on DexPBT's citation, not on this
# robot's data.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH"

IDX=$SLURM_ARRAY_TASK_ID
export TASK=CoopLift-BHL-Cube-v0 SEED=0 MAX_ITER=4000 NUM_ENVS=1024
mkdir -p "$WORKSPACE/overrides"
export OVERRIDE_FILE="$WORKSPACE/overrides/coop-rerun-${IDX}.txt"

case $IDX in
  0)
    export RUN_NAME=coop-cube-nopriv-r2-s0
    cat > "$OVERRIDE_FILE" <<'OVR'
env.privileged_critic=false
OVR
    ;;
  1)
    export RUN_NAME=coop-cube-notrack-r2-s0
    cat > "$OVERRIDE_FILE" <<'OVR'
env.actor_track_error=false
OVR
    ;;
  *) echo "bad array index $IDX"; exit 1 ;;
esac

echo "=== node: $(hostname) | $RUN_NAME | $(date) ==="
cat "$OVERRIDE_FILE"
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
