#!/bin/bash
#SBATCH --job-name=turngait
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --array=0-8%2
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
# 2026-09-24: a 22-DoF gait that turns. Three arms x three seeds, v51, the
# arms-dr1.0 recipe (6000 iterations, DR scale 1.0) with one reward change
# per arm (arms_env_cfg.py: TurnHip / TurnTrack / TurnBoth). After training:
# ONNX + deploy.yaml export (the s0/s1 path), then the MuJoCo turn test.
#
# Predeclared (SLURM_JOBS.md): PASS = turn_test.py PASS (>= 150 deg in 6 s
# at a 0.6 rad/s command, no fall; <= 15 deg drift over 6 s at 0.35 m/s) --
# the shipped arms-dr1.0-s0 measures 0.2 deg and -18.4 deg. Arms that pass
# then face the matched-push regression (fall rate <= 0.15 at n = 60,
# arms-dr1.0 measured 0.10) in a separate CPU job before any promotion.
set -euo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
source "$REPO/slurm/dr_levels.sh"
setup_node_cache
source "$REPO/slurm/inner/v60_boot_gate.sh"
V60_BOOT_LOCK="$WORKSPACE/.v51-boot.lock" v60_boot_gate
IDX=$SLURM_ARRAY_TASK_ID
ARMS=(TurnHip TurnTrack TurnBoth)
ARM=${ARMS[$((IDX / 3))]}
SEED=$((IDX % 3))
export SEED MAX_ITER=${MAX_ITER:-6000}
export TASK="Velocity-BHL-Arms-${ARM}-v0" RUN_NAME="arms-turn-${ARM,,}-s${SEED}"
mkdir -p "$WORKSPACE/overrides"
export OVERRIDE_FILE="$WORKSPACE/overrides/${RUN_NAME}.txt"
dr_overrides_scale 1.0 > "$OVERRIDE_FILE"
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS RUN_DIR"
LOGROOT=$UPSTREAM/logs/rsl_rl/humanoid
find_run() { ls -d "$LOGROOT"/*_"$RUN_NAME" 2>/dev/null | sort | tail -1; }
echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | HEAD $(git -C "$REPO" rev-parse --short HEAD) | $(date) ==="
cat "$OVERRIDE_FILE"
existing=$(find_run || true)
if [ -n "$existing" ] && [ -f "$existing/model_$((MAX_ITER - 1)).pt" ]; then
    echo "already trained: $existing"
else
    bhl_exec "$REPO/slurm/inner/train.sh"
    echo "=== training done: $(date) ==="
fi
run=$(find_run || true)
[ -n "$run" ] || { echo "EXPORT: FAIL -- no run dir for $RUN_NAME"; echo "TURNGAIT RESULT: FAIL (no run)"; exit 1; }
export RUN_DIR=$(basename "$run")
if [ ! -f "$run/exported/deploy.yaml" ]; then
    echo "=== export $RUN_DIR $(date) ==="
    export TASK=Velocity-Berkeley-Humanoid-Lite-v0
    bhl_exec "$REPO/slurm/inner/export_one_humanoid.sh" 2>&1 | grep -vE "pthread_setaffinity|EGLError" || true
fi
norm() { awk '/^policy_checkpoint_path:/{next} /^command_velocity:/{skip=1;next} skip&&/^- /{next} {skip=0; print}' "$1"; }
cfg="$run/exported/deploy.yaml"; ref=$(ls -d "$LOGROOT"/*_arms-dr1.0-s0 | sort | tail -1)/exported/deploy.yaml
ok=1; { [ -f "$cfg" ] && [ -f "$run/exported/policy.onnx" ]; } || ok=0
if [ "$ok" = 1 ]; then
    grep -q "^policy_checkpoint_path:.*$RUN_DIR/exported/policy.onnx" "$cfg" || ok=0
    grep -qE "^num_joints: 22$" "$cfg" || ok=0
    grep -qE "^num_observations: 75$" "$cfg" || ok=0
    diff <(norm "$ref") <(norm "$cfg") || { echo "deploy.yaml differs from arms-dr1.0-s0 (above)"; ok=0; }
fi
[ "$ok" = 1 ] && echo "EXPORT: OK $cfg" || { echo "EXPORT: FAIL $cfg"; echo "TURNGAIT RESULT: FAIL (export)"; exit 1; }
# --- turn test, CPU MuJoCo, outside the container ---
PYH=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
cd "$REPO"; export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=2
mkdir -p results/repo-gpu-20260923/turn-test
"$PYH" scripts/bench/turn_test.py --deploy "$cfg" --upstream "$UPSTREAM" --cache-dir "${TMPDIR:-/tmp}/turn-$SLURM_JOB_ID" --variant humanoid \
    --out "results/repo-gpu-20260923/turn-test/${RUN_NAME}.json" 2>&1 | grep -v "^\[" | tail -2
rc=${PIPESTATUS[0]}
case $rc in 0) echo "TURNGAIT RESULT: PASS ($RUN_NAME)";; 1) echo "TURNGAIT RESULT: NEGATIVE ($RUN_NAME turns/walks outside the rule)";; *) echo "TURNGAIT RESULT: ERROR ($RUN_NAME turn test rc $rc)";; esac
echo "=== done: $(date) ==="
