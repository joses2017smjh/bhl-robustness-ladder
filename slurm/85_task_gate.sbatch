#!/bin/bash
#SBATCH --job-name=task-gate
#SBATCH --account=eecs
#SBATCH --partition=eecs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# G-T1 and G-T3 for the redesigned tasks. Forward kinematics only -- no policy,
# no training. The claim under test is that a payload at 0.30 m is reachable by
# a standing squat and not by the 41 cm collapse the current policies perform.
# If either half fails, the height is wrong and nine arms would have measured
# the height rather than the task.
set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CACHE_DIR=$WORKSPACE/mjcf_cache
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS GRASP_Z"
mkdir -p "$REPO/results"
bhl_exec "$REPO/slurm/inner/task_gate.sh"
