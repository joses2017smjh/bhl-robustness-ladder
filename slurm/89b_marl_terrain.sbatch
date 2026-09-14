#!/bin/bash
#SBATCH --job-name=marl-terrain
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-11
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Tier 1's MARL rows: limb agents x terrain family x ray-cast depth.
#
# The work order's factorisation -- the 12-DoF biped at N=2, left leg | right
# leg -- under MAPPO and IPPO, on the four terrains the PPO rows already ran
# (ice, stairs, slippery, rough), depth on. The PPO rows are not here: they
# exist (21066022, 21076264, 21105232, and section 6 for rough).
#
# Plus limb1 on every terrain: one skrl agent owning all 12 joints, same
# trainer, models and hyperparameters. The first block (89_marl_train) showed
# that an rsl-rl PPO row prices the RL library as well as the split, so without
# limb1 a MAPPO-over-PPO gap is not a statement about factorisation.
#
# There are no arms on the biped, so there is no arm-deviation term to ablate;
# the work order chose this split partly for that reason.
#
#   index = 3 * terrain + row      terrain: 0 ice, 1 stairs, 2 slippery, 3 rough
#                                  row:     0 mappo, 1 ippo, 2 limb1
#
# Perturbation-rich terrains first, where the pre-registered predictions bite.
# Run names follow the analysis convention {algo}-{terrain}-{vision}-s{seed}.
# NUM_ENVS 4096 like every PPO row; if this OOMs, the work order's rule is to
# drop every arm to 2048 and re-run the PPO controls too, never a mix.
#
# DO NOT SUBMIT without G-B4t (88b_marl_terrain_gate.sbatch) passing first.
#
# Ice (indices 0-2) is NOT to be queued yet: B3's patches spawn at each env's
# grid origin while the robots reset onto terrain origins a median 72 m away
# (scripts/bench/ice_placement_probe.py, 21328532), so an "ice" row trains on
# bumpy ground. Submit --array=3-11 until the patches are moved.
#
# PPO settings and network come from the task's rsl-rl runner config
# (train_marl.py --hparams rsl, chosen in marl_train.sh by run name).
set -euo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
mkdir -p "$WORKSPACE/overrides"

IDX=$SLURM_ARRAY_TASK_ID
TERRAINS=(ice stairs slippery rough)
TASKS=(Velocity-BHL-Biped-Ice-Depth-v0 Velocity-BHL-Biped-Stairs-Depth-v0
       Velocity-BHL-Biped-Slippery-Depth-v0 Velocity-BHL-Biped-Depth-v0)
T=$(( IDX / 3 ))
case $(( IDX % 3 )) in
  0) ROW=mappo; export BHL_PARTITION=legs2 BHL_ALGO=mappo ;;
  1) ROW=ippo;  export BHL_PARTITION=legs2 BHL_ALGO=ippo  ;;
  2) ROW=limb1; export BHL_PARTITION=limb1 BHL_ALGO=ippo  ;;
esac
export TASK=${TASKS[$T]}
export SEED=${SEED:-0} MAX_ITER=${MAX_ITER:-6000} NUM_ENVS=${NUM_ENVS:-4096}
export RUN_NAME="${ROW}-${TERRAINS[$T]}-depth-s${SEED}"
export BHL_ABLATE_ARM_DEV=0
export OVERRIDE_FILE="$WORKSPACE/overrides/marl-terrain-${IDX}.txt"
: > "$OVERRIDE_FILE"

# Array tasks that start together must not boot Isaac together; the v51 stack
# has its own Kit data directory, so its own lock.
source "$REPO/slurm/inner/v60_boot_gate.sh"
V60_BOOT_LOCK="$WORKSPACE/.v51-boot.lock" v60_boot_gate
echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | envs=$NUM_ENVS | $(date) ==="
bhl_exec "$REPO/slurm/inner/marl_train.sh"
echo "=== done: $(date) ==="
