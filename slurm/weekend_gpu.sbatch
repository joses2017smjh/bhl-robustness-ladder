#!/bin/bash
#SBATCH --job-name=wknd-gate
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Submit from this repository with an explicit --chdir and log path.
# Usage: sbatch ... weekend_gpu.sbatch v60 slurm/inner/weekend_maze.sh ...
# Supply cluster account/partition through sbatch flags.
set -euo pipefail
stack=${1:?stack}; inner=${2:?inner script}; shift 2
export BHL_STACK=$stack
if [ "$stack" = train ]; then export BHL_STACK=v51; fi
repo=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
source "$repo/slurm/_env.sh"
setup_node_cache
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}" OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
BHL_FORWARD_VARS="$BHL_FORWARD_VARS SLURM_ARRAY_TASK_ID"
if [ "$stack" = train ]; then
    export UV_PROJECT_ENVIRONMENT=$WORKSPACE/venv-lehome-train
    PY=$UV_PROJECT_ENVIRONMENT/bin/python
    export HF_HOME=$WORKSPACE/.cache/huggingface
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    BHL_FORWARD_VARS="$BHL_FORWARD_VARS HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE"
elif [ "$stack" = v60 ]; then
    source "$REPO/slurm/inner/v60_boot_gate.sh"
    v60_boot_gate
fi
printf 'job=%s node=%s stack=%s inner=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$stack" "$inner"
bhl_exec "$REPO/$inner" "$@"
