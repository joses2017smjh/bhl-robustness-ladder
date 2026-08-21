#!/bin/bash
#SBATCH --job-name=part-probe
#SBATCH --account=eecs
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Can this project actually run outside the `gpu` partition?
#
# The `gpu` partition caps this user at 8 GPUs and the group at 20, which is
# what has every queued job sitting on QOSGrpGRES. Four other partitions accept
# the eecs account -- dgx2 (V100 32G, 62 GPUs, 16/user), dgxh (H100 40G, 40
# GPUs, 8/user), ampere (A40, 2/user), eecs (RTX 8000, 8/user). Their per-user
# caps are separate, so spreading across them is 42 GPUs rather than 8.
#
# Three things have to hold on a node before a real run is sent there, and none
# of them are safe to assume:
#   - Rocky 9. The `share` and `eecs3` GPU nodes are el8 (glibc 2.28) with M60s,
#     which is the wall this project built a container to get over in the first
#     place. They are excluded by node feature, not by hope.
#   - `apptainer --nv` injects a working driver.
#   - torch has kernels for the architecture. V100 is sm_70 and uncontroversial;
#     H100 is sm_90 and is the one worth checking rather than assuming, along
#     with warp, which is what the depth ray-caster runs on.
#
# Pass PROBE_PART to aim it: sbatch --partition=dgx2 slurm/54_partition_probe.sbatch

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OMP_NUM_THREADS=4

echo "=== node $(hostname) | partition ${SLURM_JOB_PARTITION:-?} | $(date) ==="
cat /etc/redhat-release 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "scratch: ${OV_CACHE}"
ls -d /nfs/hpc/share/$USER >/dev/null && echo "lustre: ok"

bhl_exec "$REPO/slurm/inner/probe_gpu.sh"
echo "=== done: $(date) ==="
