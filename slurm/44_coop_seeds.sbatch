#!/bin/bash
#SBATCH --job-name=coop-seeds
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# sm_75 or newer AND >=32 GB. The el9 tag is not enough: preempt mixes in
# GTX 1080 (sm_61) and V100 (sm_70), both below the pinned torch's cubin floor,
# and the 11 GB cards cannot hold this run's env count.

# Seeds 1 and 2 for the four arms whose gaps the section actually leans on.
#
# Fourteen ablation arms at one seed each is the weakest part of this repo and
# is out of character with sections 1-3, which ran three seeds and reported them
# individually. The spread being interpreted here is control 9.16, notilt 17.0,
# nodrift 20.9 -- and section 1 established that seed noise on this setup is not
# negligible. pickfirst's pinch 0.397 against the control's 0.081 is a big
# enough gap that I expect it to survive; "I expect it to survive" is not what
# the rest of this repo does.
#
# Four arms x two extra seeds, one array so the group GRES limit schedules it
# gracefully. Worth more than a fifteenth knob.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH"

IDX=$SLURM_ARRAY_TASK_ID
ARMS=(control pickfirst notilt nodrift)
ARM=${ARMS[$((IDX % 4))]}
export SEED=$(( IDX / 4 + 1 ))          # 0-3 -> seed 1, 4-7 -> seed 2
export TASK=CoopLift-BHL-Cube-v0 MAX_ITER=4000 NUM_ENVS=1024
export RUN_NAME="coop-cube-${ARM}-s${SEED}"

mkdir -p "$WORKSPACE/overrides"
export OVERRIDE_FILE="$WORKSPACE/overrides/coop-seed-${IDX}.txt"
case $ARM in
  control)   : > "$OVERRIDE_FILE" ;;
  # pickfirst is the lift weights held at zero for the whole run, via the reward
  # weights rather than the staging flag -- that is the arm that actually ran.
  pickfirst) printf 'env.rewards.lift_progress.weight=0.0\nenv.rewards.lifting_object.weight=0.0\n' > "$OVERRIDE_FILE" ;;
  notilt)    printf 'env.rewards.object_tilt.weight=0.0\n' > "$OVERRIDE_FILE" ;;
  nodrift)   printf 'env.rewards.object_xy.weight=0.0\n' > "$OVERRIDE_FILE" ;;
  *) echo "bad arm $ARM"; exit 1 ;;
esac

echo "=== node: $(hostname) | $RUN_NAME | $(date) ==="
cat "$OVERRIDE_FILE"
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
