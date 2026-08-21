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

# The interpreter to invoke. Calling the venv's python directly is more robust
# than `uv run`, which re-resolves the project and can silently fall back to an
# ephemeral environment when run from an unexpected cwd.
PY=$UV_PROJECT_ENVIRONMENT/bin/python

# Run a script inside the container with the GPU and all paths wired up.
#   bhl_exec <script.sh> [args...]
#
# Takes a script path rather than a command string: Isaac Lab invocations nest
# three levels of quoting otherwise, which is how the first attempt broke.
#
# --cleanenv is required (the host exports an lmod bash function that errors
# inside the image, plus a PATH that shadows the image's tools). HOME must be
# set with --home, not --env: Apptainer explicitly refuses APPTAINERENV_HOME.
#
# Because --cleanenv wipes the host environment, ANY variable an inner script
# needs must be forwarded explicitly. Forgetting this killed a 9-job array in
# one second with "TASK: unbound variable", so job parameters are forwarded
# from a declared list rather than ad hoc.
#
# Hydra overrides are passed as a FILE PATH (OVERRIDE_FILE), never inline:
# Apptainer's --env splits values on commas, so a range like [0.8,0.8] is
# parsed as two malformed key=value pairs and the exec is rejected outright.
BHL_FORWARD_VARS="TASK EXPERIMENT RUN_NAME SEED NUM_ENVS MAX_ITER OVERRIDE_FILE TRAIN_SCRIPT DEPLOY_CFG CACHE_DIR OUT_CSV LABEL EPISODE_S N_SEEDS PUSH_SPEED VIDEO_DIR MUJOCO_GL PYOPENGL_PLATFORM OMP_NUM_THREADS TERRAIN_D RUN_DIR VARIANT BHL_CONVEX_USD BHL_CONVEX_USD_DIR BENCH_OUT PYTHONPATH LD_LIBRARY_PATH DEPTH_ARGS BHL_SYMMETRY BHL_MIRROR_COEFF CKPT EXP LOAD_RUN"

# Isaac Sim bundles OpenUSD as `pxr` inside an extscache wheel; it is not on
# sys.path of a plain interpreter. libpython also has to be visible because
# that extension is linked against it and uv's CPython is not in the default
# linker path. Call this before any script that `import pxr`.
setup_pxr() {
    local libs py
    libs=$(ls -d "$UV_PROJECT_ENVIRONMENT"/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-* 2>/dev/null | head -1)
    py=$(ls -d "$UV_PYTHON_INSTALL_DIR"/cpython-3.11*-linux-x86_64-gnu/lib 2>/dev/null | head -1)
    [ -n "$libs" ] || { echo "setup_pxr: omni.usd.libs not found" >&2; return 1; }
    export PYTHONPATH="$REPO/src:$libs:${PYTHONPATH:-}"
    export LD_LIBRARY_PATH="${py:+$py:}$libs/bin:$libs/lib:${LD_LIBRARY_PATH:-}"
}

bhl_exec() {
    local envargs=()
    local v
    for v in $BHL_FORWARD_VARS; do
        envargs+=(--env "$v=${!v:-}")
    done

    # Node-local scratch only exists on some partitions (the GPU nodes have it,
    # generic `share` nodes may not). Binding it unconditionally aborts the
    # container on those nodes, so make it optional -- the MuJoCo evaluator is
    # CPU-only and deliberately runs wherever there is a free core.
    local scratchbind=()
    [ -d "/scratch/$USER" ] && scratchbind=(--bind "/scratch/$USER")

    apptainer exec --nv --cleanenv \
        --home "$HOME_OVERRIDE" \
        --bind /nfs/hpc/share/$USER \
        "${scratchbind[@]}" \
        --env UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT" \
        --env UV_CACHE_DIR="$UV_CACHE_DIR" \
        --env UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR" \
        --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
        --env OV_CACHE="${OV_CACHE:-$XDG_CACHE_HOME/ov}" \
        --env CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$XDG_CACHE_HOME/nv}" \
        --env UPSTREAM="$UPSTREAM" \
        --env REPO="$REPO" \
        --env PY="$PY" \
        "${envargs[@]}" \
        "$SIF" bash "$@"
}

# When this shell is itself inside a Slurm allocation (e.g. an Open OnDemand
# interactive session), srun/sbatch inherit SLURM_JOB_ID and try to run as a
# STEP of that job rather than submitting new work -- failing with
# "Job step's --cpus-per-task value exceeds that of job". Strip the inherited
# job context before submitting.
slurm_clean() {
    env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_NODELIST -u SLURM_NODEID \
        -u SLURM_TASKS_PER_NODE -u SLURM_CPUS_ON_NODE -u SLURM_JOB_CPUS_PER_NODE \
        -u SLURM_TRES_PER_TASK -u SLURM_JOB_GPUS -u SLURM_GPUS_ON_NODE \
        -u SLURM_JOB_NUM_NODES -u SLURM_MEM_PER_NODE -u SLURM_JOB_PARTITION \
        -u SLURM_EXPORT_ENV -u SLURM_TASK_PID -u SLURM_LOCALID -u SLURM_PROCID \
        -u SLURM_STEP_ID -u SLURM_STEPID -u SLURM_SUBMIT_DIR -u SLURMD_NODENAME \
        "$@"
}
