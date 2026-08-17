#!/bin/bash
# Shared paths + container invocation for every job in this project.
# Source this; do not execute it.

WORKSPACE=/nfs/hpc/share/$USER/Humanoid_Lite
REPO=$WORKSPACE/bhl-robustness-ladder
UPSTREAM=$REPO/external/Berkeley-Humanoid-Lite
SIF=$WORKSPACE/container/bhl.sif

# The venv is deliberately OUTSIDE the git tree. It is ~30GB, and uv bakes
# absolute paths into it, so keeping it out means the repo can be moved,
# cloned, or re-pinned without a 30-minute rebuild.
export UV_PROJECT_ENVIRONMENT=$WORKSPACE/venv

# /nfs/stak home has ~15GB free. Every cache goes to Lustre instead.
export UV_CACHE_DIR=$WORKSPACE/.uv-cache
export UV_PYTHON_INSTALL_DIR=$WORKSPACE/.uv-python
export XDG_CACHE_HOME=$WORKSPACE/.cache
export HOME_OVERRIDE=$WORKSPACE/.home

mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$XDG_CACHE_HOME" "$HOME_OVERRIDE"

# Isaac Sim writes large shader/ext caches at runtime. On Lustre these are
# painfully slow, so point them at node-local scratch inside each job.
setup_node_cache() {
    export OV_CACHE=/scratch/$USER/ov-cache
    export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache
    mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH"
}

# Run a command inside the container with the GPU and all paths wired up.
# --cleanenv is required: the host exports an lmod bash function that errors
# inside the image and a PATH that shadows the image's own tools.
bhl_exec() {
    apptainer exec --nv --cleanenv \
        --bind /nfs/hpc/share/$USER \
        --bind /scratch/$USER \
        --env UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT" \
        --env UV_CACHE_DIR="$UV_CACHE_DIR" \
        --env UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR" \
        --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
        --env HOME="$HOME_OVERRIDE" \
        --env OV_CACHE="${OV_CACHE:-$XDG_CACHE_HOME/ov}" \
        --env CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$XDG_CACHE_HOME/nv}" \
        "$SIF" bash -c "$1"
}
