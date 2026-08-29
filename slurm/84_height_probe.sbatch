#!/bin/bash
#SBATCH --job-name=height-probe
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Does the 41 cm collapse the MuJoCo replay shows also happen in PhysX?
#
# The replay drops both cube arms ~41 cm in the first 0.2 s, before contact,
# and holds that pose. base_contact cannot test it -- a robot folded onto its
# shins never puts its torso down -- and base height has never been logged,
# because no term in this task has ever referred to it.
#
# `base_height` is now a weight-0.0 reward term, so it is recorded as
# Episode_Reward/base_height and changes no gradient. 1,500 iterations is
# enough to see whether the curve sinks and stays down; the spawn is ~0.55 m.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
mkdir -p "$WORKSPACE/overrides"
export OVERRIDE_FILE="$WORKSPACE/overrides/height-probe.txt"
echo "env.stage_lift_on_pinch=true" > "$OVERRIDE_FILE"
export TASK=CoopLift-BHL-Cube-v0 RUN_NAME=_probe-height SEED=0
export MAX_ITER=${MAX_ITER:-6000} NUM_ENVS=1024

echo "=== node: $(hostname) | base-height probe | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
