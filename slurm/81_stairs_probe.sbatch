#!/bin/bash
#SBATCH --job-name=stair-probe
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# G-B2: 300 iterations of single-agent PPO with depth on the stairs.
#
# The question is not "does it learn" but "can it enter the terrain at all". A
# curriculum pinned at level 0 means the first riser is a wall, and every stairs
# row in the grid would then measure the wall rather than the algorithm. Section
# 8 already found this robot falling on a 2.5 cm cable, so a 5 cm riser is not
# obviously enterable and the assumption is worth 300 iterations rather than
# 6,000 x 8 arms.
#
# Pass: terrain level moves off 0 within 2,000 iterations. The budget is 2,000
# and not 300 because the control does not clear the threshold until past 1,500.
# Fail: the level is pinned AND the control has promoted by the same iteration.
# Inconclusive: both are pinned, which means 300 iterations is too few for this
#   curriculum to promote on anything and the probe has measured its own budget.
#
# Two corrections are baked in here, both of which produced a wrong verdict
# before. The training stream is no longer piped through `tail -40` -- that kept
# one `Curriculum/terrain_levels` line out of three hundred and the checker read
# it as the final value. And the verdict now comes from the run's TensorBoard
# event file with a control column beside it, because "level 0 after 300
# iterations" is a fact about the stairs only if a robot on walkable terrain has
# left level 0 by then.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export TASK=Velocity-BHL-Biped-Stairs-Depth-v0
export RUN_NAME=_probe-stairs SEED=0 MAX_ITER=2000 NUM_ENVS=1024
export OVERRIDE_FILE=""

echo "=== node: $(hostname) | 2,000-iteration stairs entry probe | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh"

# The run the probe just wrote, and the depth-bumpy arm as its control: same
# observation, same curriculum, same robot, terrain known to be walkable.
B=$UPSTREAM/logs/rsl_rl/biped
RUN=$(ls -dt "$B"/*_probe-stairs 2>/dev/null | head -1)
CTRL=$(ls -dt "$B"/*_depth-bumpy-s0 2>/dev/null | head -1)
[ -n "$RUN" ] || { echo "no probe run under $B" >&2; exit 1; }

export GATE_RUN=$RUN GATE_CTRL=${CTRL:-}
bhl_exec "$REPO/slurm/inner/stairs_verdict.sh"
