#!/bin/bash
#SBATCH --job-name=port60
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Scope the 2.x -> 3.x port by importing every overlay under the 6.0 stack and
# recording what actually fails. RTX renders on 6.0 (rgb std 24.4, 1902 unique
# colours, depth correct to the unit), so the port is now worth costing.

set -euo pipefail
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
echo "=== node: $(hostname) | $(date) ==="
bhl_exec "$REPO/slurm/inner/port_audit.sh"
echo "=== done: $(date) ==="
