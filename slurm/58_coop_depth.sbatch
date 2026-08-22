#!/bin/bash
#SBATCH --job-name=coop-depth
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere,dgxh
#SBATCH --constraint="rtx8000|a40|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Vision inside the cooperative-lift loop, which §6 could not do and §5 did not
# have. The sensor is `MultiMeshRayCasterCamera`: warp ray-casting that tracks a
# rigid body's transform, so the camera can watch the cube it is lifting. §6's
# `RayCasterCamera` bakes its meshes at startup and would return a picture of
# the empty floor.
#
# Validation runs first and the training is gated on it, because this sensor
# family fails silently -- a wrong target list is an all-NaN or all-floor image,
# not an error, and 4,000 iterations of that looks like a converged run.
#
# Two arms, one control:
#   swap  drop `object_pos_in_root`, keep depth   -- deployable, the real question
#   both  keep both                               -- separates "vision suffices"
#                                                     from "vision is present"
# The §5 control is the already-trained `coop-cube-staged-r2-s0`.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS BENCH_OUT"
export BENCH_OUT=$REPO/results/coop_depth_validate.txt
mkdir -p "$REPO/results"

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "=== 1. does the camera see the payload, and does it track it? ==="
# `tee` into the log and grep the copy, rather than piping the probe straight
# into grep. The first version of this line did the latter and the first failure
# it caught was an import error whose traceback the grep discarded -- 32 seconds
# of job and no way to tell what broke. Isaac's startup chatter is worth hiding
# only while things work.
VLOG=$REPO/results/coop_depth_validate.log
# The diagnostics go to stdout, not stderr. `--output` and `--error` name the
# same file here, and Slurm opens them as two descriptors with independent
# offsets, so a burst on stderr can land under what stdout already wrote and
# vanish. That is why attempt two looked identical to attempt one.
NUM_ENVS=4 bhl_exec "$REPO/slurm/inner/coop_depth_validate.sh" >"$VLOG" 2>&1 || true
grep -E "CHECK|PASS|FAIL|RESULT" "$VLOG" || true

# The gate reads the probe's verdict out of the log rather than trusting its exit
# status, because `SimulationApp.close()` terminates the interpreter itself and
# the `raise SystemExit(1)` after it never reaches the shell. Attempt three exited
# 0 with four failed assertions and started training on a control arm that was a
# duplicate of the treatment. So: every arm must print a RESULT, and every RESULT
# must say ALL PASS.
arms=$(grep -c "^RESULT" "$VLOG" || true)
good=$(grep -c "^RESULT | ALL PASS" "$VLOG" || true)
echo "validation: $good of $arms arms passed"
if [ "$arms" -lt 2 ] || [ "$good" -ne "$arms" ]; then
    echo "FATAL: sensor validation did not pass for both arms; not training."
    grep -E "^FAIL|^RESULT" "$VLOG" || true
    exit 1
fi

# The override file lives on Lustre, not in /tmp: `bhl_exec` runs --cleanenv in
# a container whose /tmp is its own, and the inner script reads this by path.
mkdir -p "$WORKSPACE/overrides"
export TASK=CoopLift-BHL-Cube-Depth-v0 SEED=0
export NUM_ENVS=${NUM_ENVS_TRAIN:-1024} MAX_ITER=${MAX_ITER:-4000}

echo
echo "=== 2. swap: depth replaces the privileged object pose ==="
export OVERRIDE_FILE="$WORKSPACE/overrides/coop-depthswap.txt"
cat > "$OVERRIDE_FILE" <<'OVR'
env.drop_object_pose=true
env.stage_lift_on_pinch=true
OVR
export RUN_NAME=coop-cube-depthswap-s0
cat "$OVERRIDE_FILE"
bhl_exec "$REPO/slurm/inner/train.sh"

echo
echo "=== 3. both: depth alongside the object pose ==="
export OVERRIDE_FILE="$WORKSPACE/overrides/coop-depthboth.txt"
cat > "$OVERRIDE_FILE" <<'OVR'
env.drop_object_pose=false
env.stage_lift_on_pinch=true
OVR
export RUN_NAME=coop-cube-depthboth-s0
cat "$OVERRIDE_FILE"
bhl_exec "$REPO/slurm/inner/train.sh"

echo "=== done: $(date) ==="
