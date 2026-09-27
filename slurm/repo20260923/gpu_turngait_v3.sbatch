#!/bin/bash
#SBATCH --job-name=turngait-v3
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --array=0-2%2
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
# 2026-09-26: turning-gait arm TurnCmd, 3 seeds (array 0-2 = seed 0-2), v51,
# the arms-dr1.0 recipe: 6000 iterations (same as the TurnHip/TurnTrack/TurnBoth
# arms), DR scale 1.0 via dr_levels.sh. Task Velocity-BHL-Arms-TurnCmd-v0 =
# arms_env_cfg.HumanoidTurnCmdCfg = TurnBoth's reward changes + a command mix
# (turn_command.py): 25 % of envs per resample get a pure yaw-rate command
# (vx = vy = 0, |wz| ~ U(0.3, 1.0), random sign, no heading control), 25 % get
# low-speed walking with directly sampled wz (vx U(-0.5,0.5), vy U(-0.25,0.25),
# wz U(-1,1), no heading control), 50 % upstream (heading-derived wz); the 2 %
# standing draw applies to all. Obs/actions unchanged (75 / 22).
# After training: ONNX + deploy.yaml export (export_one_humanoid.sh, now with
# --play-steps 5), deploy.yaml checked against arms-dr1.0-s0, then the MuJoCo
# turn test with --protocol v2.
#
# PREDECLARED (2026-09-26, before any TurnCmd training):
#   seed PASS  = turn_test.py --protocol v2 verdict PASS: for reset seeds 0, 1, 2,
#                3.0 s warm-up then (0,0,+0.6) and (0,0,-0.6) rad/s for 6 s each;
#                ALL SIX runs >= 150 deg in the commanded direction, no fall; AND
#                the walk run (0.35,0,0), 1.0 s warm-up, |yaw drift| <= 15 deg,
#                no fall. (A missing/unreadable JSON counts as FAIL.)
#   arm PASS   = >= 2 of the 3 seeds PASS.
#   A passing arm then faces the matched-push regression (0.5 m/s shoves every
#   3 s, 12 s episodes, n = 60, fall rate <= 0.15; arms-dr1.0 measured 0.10) in
#   a separate CPU job before any use as the Mission 7 / carry gait.
#   Reference under v2 (measured 2026-09-26 on the login node): TurnBoth-s0,
#   the only v1 turner from standstill, gets 5/6 (turn-_s1 = -11.4 deg) -> FAIL.
#
# Verdict lines are computed from the JSON the turn test writes, never from
# exit codes. Outputs: results/repo-gpu-20260923/turn-test-v2/<run>.json (a new
# directory, so the committed v1 results in turn-test/ are never overwritten).
#
# Modes (env vars; everything else as the array run):
#   TURNGAIT_SMOKE=1      MAX_ITER=3, NUM_ENVS=64, run name *-smoke-s<seed>; no
#                         export / test. Prints "SMOKE: PASS|FAIL (...)" from the
#                         training log (TurnMixVelocityCommand in the command
#                         manager, iterations logged) + the checkpoint file.
#                           sbatch --array=0 --time=02:00:00 --export=ALL,TURNGAIT_SMOKE=1 <this>
#                         Exit 0 iff SMOKE: PASS, so the array can chain afterok.
#   TURNGAIT_EVAL_ONLY=1  no training, no GPU: v2 on existing exported runs
#                         (TURNGAIT_RUNS, space-separated run names; default the
#                         nine TurnHip/TurnTrack/TurnBoth runs + arms-dr1.0-s0 as
#                         reference). Run it on a CPU node with --wrap, e.g.
#                           sbatch --account=eecs --partition=share --constraint='haswell&el8' \
#                             --cpus-per-task=4 --mem=16G --time=03:00:00 --job-name=turngait-v2eval \
#                             --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out \
#                             --export=ALL,TURNGAIT_EVAL_ONLY=1 --wrap="bash <this>"
#   TURNGAIT_OUT_DIR      override the JSON directory (testing only).
set -euo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
PYH=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
LOGROOT=$UPSTREAM/logs/rsl_rl/humanoid
OUT_DIR=${TURNGAIT_OUT_DIR:-$REPO/results/repo-gpu-20260923/turn-test-v2}
JOBTAG=${SLURM_JOB_ID:-local$$}

# v2_test <deploy.yaml> <label>: run turn_test v2 into a fresh temp JSON, print
# one verdict line read from THAT JSON (so a stale <label>.json from an earlier
# run can never be read), then move it to $OUT_DIR/<label>.json.
v2_test() {
    local cfg=$1 label=$2 tmp
    tmp="$OUT_DIR/.$label.$JOBTAG.json"
    mkdir -p "$OUT_DIR"; rm -f "$tmp"
    ( cd "$REPO" && PYTHONPATH="$REPO/src" OMP_NUM_THREADS=2 "$PYH" scripts/bench/turn_test.py --protocol v2 \
        --deploy "$cfg" --upstream "$UPSTREAM" --cache-dir "${TMPDIR:-/tmp}/turnv2-$JOBTAG" --variant humanoid \
        --out "$tmp" 2>&1 | grep -v "^\[" | tail -8 ) || true
    "$PYH" - "$tmp" "$label" <<'EOF'
import json, sys
path, label = sys.argv[1:3]
try:
    d = json.load(open(path))
    assert d.get("protocol") == "v2", "not a v2 JSON"
    turns = ", ".join(f"{r['name']} {r['yaw_deg']:+.1f}" for r in d["turns"])
    print(f"TURN-V2 {label}: {d['verdict']} (turns ok {d['n_turn_ok']}/{d['n_turn']}: {turns}; "
          f"walk drift {d['walk']['yaw_deg']:+.1f} deg, fell {d['walk']['fell_at_s']})")
except Exception as exc:  # noqa: BLE001
    print(f"TURN-V2 {label}: FAIL (no readable v2 JSON: {exc!r})")
EOF
    if [ -f "$tmp" ]; then mv -f "$tmp" "$OUT_DIR/$label.json"; fi
}

# arm_summary <arm-run-prefix> -> "k/3 seeds PASS" line from the JSONs (predeclared >= 2/3).
arm_summary() {
    "$PYH" - "$OUT_DIR" "$1" <<'EOF'
import json, sys
from pathlib import Path
out, prefix = Path(sys.argv[1]), sys.argv[2]
res = {}
for s in range(3):
    p = out / f"{prefix}-s{s}.json"
    try:
        d = json.loads(p.read_text())
        res[s] = d["verdict"] if d.get("protocol") == "v2" else "FAIL"
    except Exception:  # noqa: BLE001
        res[s] = None
k = sum(v == "PASS" for v in res.values())
if any(v is None for v in res.values()):
    print(f"ARM {prefix} v2: INCOMPLETE ({k} PASS so far; per seed {res})")
else:
    print(f"ARM {prefix} v2: {'PASS' if k >= 2 else 'FAIL'} ({k}/3 seeds PASS, rule >= 2/3; per seed {res})")
EOF
}

# ---------------------------------------------------------------- eval only (CPU)
if [ "${TURNGAIT_EVAL_ONLY:-0}" = 1 ]; then
    RUNS=${TURNGAIT_RUNS:-"arms-turn-turnhip-s0 arms-turn-turnhip-s1 arms-turn-turnhip-s2 arms-turn-turntrack-s0 arms-turn-turntrack-s1 arms-turn-turntrack-s2 arms-turn-turnboth-s0 arms-turn-turnboth-s1 arms-turn-turnboth-s2 arms-dr1.0-s0"}
    echo "=== turn test v2, eval only | $(hostname) | HEAD $(git -C "$REPO" rev-parse --short HEAD) | $(date) ==="
    "$PYH" -c "import mujoco, onnxruntime; print('preflight OK', mujoco.__version__)"
    for r in $RUNS; do
        d=$(ls -d "$LOGROOT"/*_"$r" 2>/dev/null | sort | tail -1 || true)
        if [ -z "$d" ] || [ ! -f "$d/exported/deploy.yaml" ]; then echo "TURN-V2 $r: FAIL (no exported deploy.yaml)"; continue; fi
        v2_test "$d/exported/deploy.yaml" "$r"
    done
    for a in turnhip turntrack turnboth; do
        case " $RUNS " in *" arms-turn-$a-s"*) arm_summary "arms-turn-$a";; esac
    done
    echo "=== done: $(date) ==="
    exit 0
fi

# ---------------------------------------------------------------- train (GPU)
source "$REPO/slurm/dr_levels.sh"
setup_node_cache
source "$REPO/slurm/inner/v60_boot_gate.sh"
V60_BOOT_LOCK="$WORKSPACE/.v51-boot.lock" v60_boot_gate
SEED=${SLURM_ARRAY_TASK_ID:-0}
SMOKE=${TURNGAIT_SMOKE:-0}
export TASK="Velocity-BHL-Arms-TurnCmd-v0"
if [ "$SMOKE" = 1 ]; then
    export MAX_ITER=3 NUM_ENVS=64 RUN_NAME="arms-turn-turncmd-smoke-s${SEED}"
else
    export MAX_ITER=${MAX_ITER:-6000} RUN_NAME="arms-turn-turncmd-s${SEED}"
fi
export SEED
mkdir -p "$WORKSPACE/overrides"
export OVERRIDE_FILE="$WORKSPACE/overrides/${RUN_NAME}.txt"
dr_overrides_scale 1.0 > "$OVERRIDE_FILE"
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS RUN_DIR"
find_run() { ls -d "$LOGROOT"/*_"$RUN_NAME" 2>/dev/null | sort | tail -1; }
echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | smoke=$SMOKE | iters=$MAX_ITER | HEAD $(git -C "$REPO" rev-parse --short HEAD) | $(date) ==="
cat "$OVERRIDE_FILE"
TLOG="${TMPDIR:-/tmp}/turngait-v3-$JOBTAG.log"; : > "$TLOG"
existing=$(find_run || true)
if [ "$SMOKE" != 1 ] && [ -n "$existing" ] && [ -f "$existing/model_$((MAX_ITER - 1)).pt" ]; then
    echo "already trained: $existing"
else
    bhl_exec "$REPO/slurm/inner/train.sh" 2>&1 | tee "$TLOG" || true
    echo "=== training done: $(date) ==="
fi
run=$(find_run || true)
if [ "$SMOKE" = 1 ]; then
    why=""
    grep -q "TurnMixVelocityCommand" "$TLOG" || why="$why no-TurnMixVelocityCommand-in-command-manager"
    grep -qE "train.sh: reached [0-9]+ logged iterations" "$TLOG" || why="$why no-logged-iterations"
    { [ -n "$run" ] && [ -f "$run/model_$((MAX_ITER - 1)).pt" ]; } || why="$why no-model_$((MAX_ITER - 1)).pt"
    grep -hE "train.sh: (reached|mean episode)|Metrics/base_velocity/(pure_turn_env|direct_wz_env)" "$TLOG" | tail -4 || true
    echo "=== done: $(date) ==="
    # The exit status carries the evidence-based verdict so the array can chain afterok.
    if [ -z "$why" ]; then echo "SMOKE: PASS ($RUN_NAME, $run)"; exit 0; else echo "SMOKE: FAIL ($RUN_NAME:$why)"; exit 1; fi
fi
{ [ -n "$run" ] && [ -f "$run/model_$((MAX_ITER - 1)).pt" ]; } || {
    echo "TURNGAIT RESULT: FAIL (training incomplete: no model_$((MAX_ITER - 1)).pt for $RUN_NAME)"; exit 1; }
# Wrong-arm guard: a fresh training log must show the TurnCmd command term.
if [ -s "$TLOG" ] && ! grep -q "TurnMixVelocityCommand" "$TLOG"; then
    echo "TURNGAIT RESULT: FAIL (training log has no TurnMixVelocityCommand: $TASK is not the TurnCmd arm)"; exit 1; fi
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
# --- turn test v2, CPU MuJoCo, outside the container ---
VLOG="${TMPDIR:-/tmp}/turngait-v3-v2-$JOBTAG.log"
v2_test "$cfg" "$RUN_NAME" | tee "$VLOG"
line=$(grep "^TURN-V2 $RUN_NAME:" "$VLOG" | tail -1 || true)
case "$line" in
    *": PASS ("*) echo "TURNGAIT RESULT: PASS ($RUN_NAME, v2)";;
    *) echo "TURNGAIT RESULT: NEGATIVE ($RUN_NAME fails v2: ${line#*: })";;
esac
arm_summary "arms-turn-turncmd"
echo "=== done: $(date) ==="
