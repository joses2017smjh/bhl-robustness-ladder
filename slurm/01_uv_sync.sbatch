#!/bin/bash
#SBATCH --job-name=bhl-sync
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Resolves the Isaac Sim 5.1 / Isaac Lab 2.3.2 stack (~30GB) into $WORKSPACE/venv,
# exactly as pinned by the upstream uv.lock.
#
# --locked is deliberate: the DR fidelity ladder is a measurement experiment,
# so the dependency set must be bit-identical across all three training runs.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

echo "=== node: $(hostname)  started: $(date) ==="
echo "venv target: $UV_PROJECT_ENVIRONMENT"
df -h /nfs/hpc/share | tail -1

bhl_exec "
    set -euo pipefail
    cd $UPSTREAM
    uv python install 3.11
    uv sync --locked --no-progress
    echo '=== resolved stack ==='
    uv run python -c \"
import sys, torch
print('python  ', sys.version.split()[0])
print('torch   ', torch.__version__, '| cuda build', torch.version.cuda)
import isaaclab; print('isaaclab', isaaclab.__version__)
\"
"

echo "=== venv size ==="
du -sh "$UV_PROJECT_ENVIRONMENT" 2>/dev/null || true
echo "=== done: $(date) ==="
