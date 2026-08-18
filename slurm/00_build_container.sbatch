#!/bin/bash
#SBATCH --job-name=bhl-build
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Builds the Isaac Sim runtime base image.
#
# Must run on a compute node, for two independent reasons:
#   1. The login node's thread limit makes mksquashfs fail ("Failed to create thread").
#   2. --fakeroot only resolves via root-mapped userns off the submit host
#      (this user has no /etc/subuid entry).

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh

# Node-local scratch: the build writes tens of thousands of small files and
# would crawl on Lustre. Note /scratch is node-local, NOT shared.
export APPTAINER_CACHEDIR=/scratch/$USER/apptainer-cache
export APPTAINER_TMPDIR=/scratch/$USER/apptainer-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

echo "=== node: $(hostname)  started: $(date) ==="

apptainer build --fakeroot --force \
    /scratch/$USER/bhl.sif \
    "$REPO/container/bhl.def"

mkdir -p "$(dirname "$SIF")"
cp /scratch/$USER/bhl.sif "$SIF"

echo "=== verifying ==="
apptainer exec --nv --cleanenv "$SIF" bash -c '
    echo "glibc:  $(ldd --version | head -1)"
    echo "uv:     $(uv --version)"
    echo "ffmpeg: $(ffmpeg -version 2>/dev/null | head -1)"
    echo "--- GPU visible inside container ---"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo "--- vulkan ICD injected by --nv ---"
    ls /usr/share/vulkan/icd.d/ 2>/dev/null || echo "(no icd.d)"
'

ls -lh "$SIF"
echo "=== done: $(date) ==="
