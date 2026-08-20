#!/bin/bash
#SBATCH --job-name=coop-strat
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Cube-only training-strategy ablations. One seed. The control is already
# running as coop-cube-pinch-s0 (squat spawn, two-scale reach, pinch-gated
# height, clamp, residual around the squat, still_alive). Each arm here
# turns off one of those knobs. Seeds still wait until pinch is nonzero.
#
# Do not route to `eecs`. Submitted even if QOSGrpGRES is full; it waits
# behind 36 and 33.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH"

IDX=$SLURM_ARRAY_TASK_ID
export TASK=CoopLift-BHL-Cube-v0 SEED=0 MAX_ITER=4000
gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk 'NR==1{print int($1)}')
if [ "${gpu_mib:-48000}" -lt 20000 ]; then
  export NUM_ENVS=256
else
  export NUM_ENVS=1024
fi

mkdir -p "$WORKSPACE/overrides"
export OVERRIDE_FILE="$WORKSPACE/overrides/coop-strat-${IDX}.txt"

case $IDX in
  0)
    # Toss channel: height pays without a pinch (first ladder run, squat spawn).
    export RUN_NAME=coop-cube-ungated-s0
    cat > "$OVERRIDE_FILE" <<'OVR'
env.gate_lift_on_pinch=false
OVR
    ;;
  1)
    # Single kernel at Isaac Lab Franka's σ=0.15. Hands are already in the
    # squat envelope, so two-scale may have been compensating for standing.
    export RUN_NAME=coop-cube-onesigma-s0
    cat > "$OVERRIDE_FILE" <<'OVR'
env.rewards.reaching_coarse.weight=0.0
env.rewards.reaching_fine.params.std=0.15
env.rewards.reaching_fine.weight=2.0
OVR
    ;;
  2)
    # No opposing-clamp. Reach + gated height only.
    export RUN_NAME=coop-cube-noclamp-s0
    cat > "$OVERRIDE_FILE" <<'OVR'
env.rewards.opposing_clamp.weight=0.0
OVR
    ;;
  3)
    # DexPBT stage 1: r_pick only. Lift weights stay zero for the whole run.
    export RUN_NAME=coop-cube-pickfirst-s0
    cat > "$OVERRIDE_FILE" <<'OVR'
env.rewards.lift_progress.weight=0.0
env.rewards.lifting_object.weight=0.0
OVR
    ;;
  4)
    # Absolute joint targets, not residual around the squat-hold (ULC / AdaptManip).
    export RUN_NAME=coop-cube-absolute-s0
    cat > "$OVERRIDE_FILE" <<'OVR'
env.actions.joint_pos_a.use_default_offset=false
env.actions.joint_pos_b.use_default_offset=false
OVR
    ;;
  5)
    # No living bonus. Standing-spawn cube/ball died in five steps because
    # that channel beat a saturated reach; squat spawn may not need it.
    export RUN_NAME=coop-cube-noalive-s0
    cat > "$OVERRIDE_FILE" <<'OVR'
env.rewards.still_alive.weight=0.0
OVR
    ;;
  *) echo "bad array index $IDX"; exit 1 ;;
esac

echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | $(date) ==="
echo "=== overrides ==="
cat "$OVERRIDE_FILE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
