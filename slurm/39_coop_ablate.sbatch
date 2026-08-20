#!/bin/bash
#SBATCH --job-name=coop-ablate
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Rest of the cube recipe ablation. slurm/38 turned off six knobs; these
# are the other ten that are still a hydra/flag switch. One seed. Not MAPPO,
# not standing spawn, not a factorial. Control: coop-cube-pinch-s0.

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
export OVERRIDE_FILE="$WORKSPACE/overrides/coop-ablate-${IDX}.txt"

case $IDX in
  0)
    export RUN_NAME=coop-cube-notilt-s0
    cat > "$OVERRIDE_FILE" <<'OVR'
env.rewards.object_tilt.weight=0.0
OVR
    ;;
  1)
    export RUN_NAME=coop-cube-nodrift-s0
    cat > "$OVERRIDE_FILE" <<'OVR'
env.rewards.object_xy.weight=0.0
OVR
    ;;
  2)
    # Sparse bonus only. README claim: bonus-only is too sparse for 6 Nm.
    export RUN_NAME=coop-cube-bonusonly-s0
    cat > "$OVERRIDE_FILE" <<'OVR'
env.rewards.lift_progress.weight=0.0
OVR
    ;;
  3)
    # Progress only. README claim: progress-only is a toss.
    export RUN_NAME=coop-cube-progressonly-s0
    cat > "$OVERRIDE_FILE" <<'OVR'
env.rewards.lifting_object.weight=0.0
OVR
    ;;
  4)
    export RUN_NAME=coop-cube-coarseonly-s0
    cat > "$OVERRIDE_FILE" <<'OVR'
env.rewards.reaching_fine.weight=0.0
OVR
    ;;
  5)
    # Competence gate frozen at 4 cm. h never rises.
    export RUN_NAME=coop-cube-fixedh-s0
    cat > "$OVERRIDE_FILE" <<'OVR'
env.curriculum.lift_height.params.min_height=0.04
env.curriculum.lift_height.params.max_height=0.04
OVR
    ;;
  6)
    # Wall-clock h ramp, the schedule that does not look at success.
    export RUN_NAME=coop-cube-clockh-s0
    cat > "$OVERRIDE_FILE" <<'OVR'
env.clock_lift_height=true
OVR
    ;;
  7)
    export RUN_NAME=coop-cube-nopriv-s0
    cat > "$OVERRIDE_FILE" <<'OVR'
env.privileged_critic=false
OVR
    ;;
  8)
    export RUN_NAME=coop-cube-notrack-s0
    cat > "$OVERRIDE_FILE" <<'OVR'
env.actor_track_error=false
OVR
    ;;
  9)
    # DexPBT r_pick then r_targ, not pick-only for the whole run.
    export RUN_NAME=coop-cube-staged-s0
    cat > "$OVERRIDE_FILE" <<'OVR'
env.stage_lift_on_pinch=true
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
