#!/bin/bash
#SBATCH --job-name=turngait-v5
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --array=0-5%4
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
# 2026-09-27: follow-up to TurnRest (gpu_turngait_v4.sbatch). TurnRest turned both
# ways 6/6 on 2/2 seeds but FAILED its predeclared push regression: 29/60 and 31/60
# falls (gate <= 9/60) against 7/60 for its parent arms-turn-turnboth-s0, on the
# identical protocol. Neither the parent nor TurnRest trained with pushes. Two arms,
# each a 3000-iteration FINE-TUNE of arms-turn-turnboth-s0 (model_5999.pt), same DR
# (scale 1.0 via dr_levels.sh), 3 seeds each; each differs from TurnRest in ONE factor:
#   array 0-2  CONT  (seeds 0-2): Velocity-BHL-Arms-TurnBoth-v0 = the parent's own
#              task (TurnBoth rewards, upstream command), continued. This is the
#              continued-training-only control v4 declared missing.
#   array 3-5  PUSH  (seeds 0-2): Velocity-BHL-Arms-TurnRestPush-v0 =
#              arms_env_cfg.HumanoidTurnRestPushCfg = TurnRest (rewards + command mix
#              unchanged) + upstream's interval push with the repo's competence-gated
#              push curriculum (ArmsPushEventsCfg + ArmsPushCurriculumCfg, exactly as
#              Velocity-BHL-Arms-PushAdaptive-v0: interval 5-9 s, magnitude 0 -> 1.0
#              in 0.02 steps at a 0.20 fall-rate target).
# Per seed, in the job: export, turn_test v2 (gating for PUSH), the non-gating MuJoCo
# diagnosis, then the single-checkpoint qualification cpu_turn_qualify.sbatch runs
# (its rule, unchanged: v2x turns on reset seeds 10-14 >= 9/10, walk |drift| <= 15 deg
# on >= 2/3, matched push 0.5 m/s every 3 s, 12 s, n = 60, fall rate <= 0.15).
#
# PREDECLARED (2026-09-27, before any v5 run exists):
#   PUSH arm PASSES iff >= 2 of its 3 seeds both PASS turn_test v2 AND are QUALIFIED by
#            cpu_turn_qualify's rule. Otherwise the arm FAILS; INCOMPLETE if any seed
#            lacks either JSON. A PASS is labelled "fine-tunes of one parent, one extra
#            factor"; a qualified seed is a candidate Mission 7 / carry gait.
#   CONT arm is a control, not gated. Reading rule on its push fall rates (qualify CSVs):
#            >= 2/3 seeds with push > 0.30 -> TurnRest's push regression is explained
#            by continued training alone; >= 2/3 seeds with push <= 0.15 -> it is not
#            (the rest-then-turn mix is implicated); otherwise inconclusive. Its v2
#            turn results are reported too.
#   Known limits: all six are fine-tunes of ONE parent; the push curriculum is the
#   repo's, whose 22-DoF from-scratch runs showed no detectable push benefit at four
#   seeds (docs/STATUS.md), so a PUSH failure is not evidence that pushes cannot help.
#
# Verdicts are computed from JSON, never exit codes (Isaac exits 0 on a crash):
# training counts only if the run dir holds model_8998.pt and a fresh training log
# shows the parent checkpoint being loaded and the arm's command class (and, for
# PUSH, the push_robot event and push_levels curriculum) in the managers.
# Outputs (new run names; nothing published is overwritten):
#   results/repo-gpu-20260923/turn-20260927/{turn-test-v2,diagnose,qualify}/<run>*
#
# Modes (env vars):
#   TURNGAIT_SMOKE=1      MAX_ITER=3, NUM_ENVS=64, run *-smoke-s<seed>; no export/test.
#                         "SMOKE: PASS|FAIL (...)"; exit 0 iff PASS so the array can
#                         chain afterok. Smoke both arms: --array=0,3.
#   TURNGAIT_EVAL_ONLY=1  CPU, no training: v2 + diagnosis on exported runs of
#                         TURNGAIT_ARM (cont|push); submit via --wrap as in v4.
set -euo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
PYH=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
LOGROOT=$UPSTREAM/logs/rsl_rl/humanoid
RES=${TURNGAIT_OUT_DIR:-$REPO/results/repo-gpu-20260923/turn-20260927}
OUT_DIR=$RES/turn-test-v2
DIAG_DIR=$RES/diagnose
JOBTAG=${SLURM_JOB_ID:-local$$}
PARENT_RUN=arms-turn-turnboth-s0
PARENT_ITER=5999
FT_ITERS=${FT_ITERS:-3000}
IDX=${SLURM_ARRAY_TASK_ID:-0}
ARM=${TURNGAIT_ARM:-$([ "$IDX" -lt 3 ] && echo cont || echo push)}
case $ARM in
    cont) PREFIX=arms-turn-turnboth-cont;   ARM_TASK=Velocity-BHL-Arms-TurnBoth-v0;     ARM_CMD=UniformVelocityCommand;;
    push) PREFIX=arms-turn-turnrestpush-ft; ARM_TASK=Velocity-BHL-Arms-TurnRestPush-v0; ARM_CMD=TurnRestMixVelocityCommand;;
    *) echo "TURNGAIT RESULT: FAIL (unknown TURNGAIT_ARM=$ARM)"; exit 2;;
esac

# arm_guard <training log>: prints the reasons a log is NOT this arm (empty = ok).
arm_guard() {
    grep -qE "\| +base_velocity +\| +$ARM_CMD +\|" "$1" || printf ' command-class-is-not-%s' "$ARM_CMD"
    if [ "$ARM" = push ]; then
        grep -q "push_robot" "$1" || printf ' no-push_robot-event'
        grep -q "push_levels" "$1" || printf ' no-push_levels-curriculum'
    else
        grep -q "push_robot" "$1" && printf ' unexpected-push_robot-event'
    fi
    return 0
}

# v2_test <deploy.yaml> <label>: turn_test v2 into a fresh temp JSON, one verdict
# line read from THAT JSON, then moved to $OUT_DIR/<label>.json, or to
# <label>.$JOBTAG.json if that already exists (never overwrite).
v2_test() {
    local cfg=$1 label=$2 tmp
    tmp="$OUT_DIR/.$label.$JOBTAG.json"
    mkdir -p "$OUT_DIR"; rm -f "$tmp"
    ( cd "$REPO" && PYTHONPATH="$REPO/src" OMP_NUM_THREADS=2 "$PYH" scripts/bench/turn_test.py --protocol v2 \
        --deploy "$cfg" --upstream "$UPSTREAM" --cache-dir "${TMPDIR:-/tmp}/turnv4-$JOBTAG" --variant humanoid \
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
    if [ -f "$tmp" ]; then                       # never overwrite an earlier v2 JSON (as diag());
        local dst="$OUT_DIR/$label.json"          # arm_summary reads the first one (v2 is deterministic)
        [ -e "$dst" ] && dst="$OUT_DIR/$label.$JOBTAG.json"
        mv "$tmp" "$dst"
    fi
}

# diag <deploy.yaml> <label>: non-gating MuJoCo diagnosis (reset seeds 0-2, +/-0.6).
diag() {
    local out="$DIAG_DIR/$2.json"
    mkdir -p "$DIAG_DIR"; [ -e "$out" ] && out="$DIAG_DIR/$2.$JOBTAG.json"     # never overwrite an earlier diagnosis
    ( cd "$REPO" && PYTHONPATH="$REPO/src:$REPO/scripts/bench" OMP_NUM_THREADS=2 "$PYH" scripts/bench/turn_diagnose.py mujoco \
        --deploy "$1" --labels "$2" --upstream "$UPSTREAM" --cache-dir "${TMPDIR:-/tmp}/turnv4d-$JOBTAG" --seeds 0 1 2 \
        --out "$out" 2>&1 | grep -E "^MUJOCO|lift-offs L|fell before" | sed 's/^/  diag /' ) || true
    [ -s "$out" ] || echo "  diag $2: no JSON written (non-gating)"
}

# arm_summary -> "k/3 seeds PASS" from the JSONs (predeclared >= 2/3).
arm_summary() {
    "$PYH" - "$OUT_DIR" "$PREFIX" <<'EOF'
import json, sys
from pathlib import Path
out, prefix = Path(sys.argv[1]), sys.argv[2]
res = {}
for s in range(3):
    try:
        d = json.loads((out / f"{prefix}-s{s}.json").read_text())
        res[s] = d["verdict"] if d.get("protocol") == "v2" else "FAIL"
    except Exception:  # noqa: BLE001
        res[s] = None
k = sum(v == "PASS" for v in res.values())
if any(v is None for v in res.values()):
    print(f"ARM {prefix} v2: INCOMPLETE ({k} PASS so far; per seed {res})")
else:
    print(f"ARM {prefix} v2: {'PASS' if k >= 2 else 'FAIL'} ({k}/3 seeds PASS, rule >= 2/3; per seed {res}) "
          f"[three fine-tunes of arms-turn-turnboth-s0]")
EOF
}

# ---------------------------------------------------------------- eval only (CPU)
if [ "${TURNGAIT_EVAL_ONLY:-0}" = 1 ]; then
    RUNS=${TURNGAIT_RUNS:-"$PREFIX-s0 $PREFIX-s1 $PREFIX-s2"}
    echo "=== turngait v5 eval only | $(hostname) | HEAD $(git -C "$REPO" rev-parse --short HEAD) | $(date) ==="
    "$PYH" -c "import mujoco, onnxruntime; print('preflight OK', mujoco.__version__)"
    for r in $RUNS; do
        d=$(ls -d "$LOGROOT"/*_"$r" 2>/dev/null | sort | tail -1 || true)
        if [ -z "$d" ] || [ ! -f "$d/exported/deploy.yaml" ]; then echo "TURN-V2 $r: FAIL (no exported deploy.yaml)"; continue; fi
        v2_test "$d/exported/deploy.yaml" "$r"
        diag "$d/exported/deploy.yaml" "$r"
    done
    case " $RUNS " in *" $PREFIX-s"*) arm_summary;; esac
    echo "=== done: $(date) ==="
    exit 0
fi

# ---------------------------------------------------------------- GPU from here
# The parent and the probed actors are MLPs [256,128,128]; a recurrent / RND /
# symmetry overlay forwarded from the submitting shell (BHL_FORWARD_VARS) would
# make runner.load fail on the state_dict and Isaac still exit 0. As in
# inner_sf04_finetune.sh, the only resume precedent.
unset BHL_POLICY BHL_RND BHL_SYMMETRY
source "$REPO/slurm/dr_levels.sh"
setup_node_cache
source "$REPO/slurm/inner/v60_boot_gate.sh"
V60_BOOT_LOCK="$WORKSPACE/.v51-boot.lock" v60_boot_gate

# ---------------------------------------------------------------- fine-tune (GPU)
SEED=$((IDX % 3))
SMOKE=${TURNGAIT_SMOKE:-0}
export TASK="$ARM_TASK"
if [ "$SMOKE" = 1 ]; then
    export MAX_ITER=3 NUM_ENVS=64 RUN_NAME="$PREFIX-smoke-s${SEED}"
else
    export MAX_ITER=$FT_ITERS RUN_NAME="$PREFIX-s${SEED}"
fi
FINAL_CKPT="model_$((PARENT_ITER + MAX_ITER - 1)).pt"
export SEED
mapfile -t parents < <(ls -d "$LOGROOT"/*_"$PARENT_RUN" 2>/dev/null | sort)
if [ "${#parents[@]}" -ne 1 ] || [ ! -f "${parents[0]}/model_$PARENT_ITER.pt" ]; then
    echo "TURNGAIT RESULT: FAIL (parent: expected exactly one $PARENT_RUN dir with model_$PARENT_ITER.pt, found ${#parents[@]})"; exit 1
fi
PARENT_DIR=$(basename "${parents[0]}")
mkdir -p "$WORKSPACE/overrides"
export OVERRIDE_FILE="$WORKSPACE/overrides/${RUN_NAME}.txt"
{ dr_overrides_scale 1.0
  printf 'agent.resume=true\nagent.load_run=%s\nagent.load_checkpoint=model_%s.pt\n' "$PARENT_DIR" "$PARENT_ITER"; } > "$OVERRIDE_FILE"
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS RUN_DIR"
find_run() { ls -d "$LOGROOT"/*_"$RUN_NAME" 2>/dev/null | sort | tail -1; }
echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | smoke=$SMOKE | +$MAX_ITER iters from $PARENT_DIR/model_$PARENT_ITER.pt -> $FINAL_CKPT | HEAD $(git -C "$REPO" rev-parse --short HEAD) | $(date) ==="
cat "$OVERRIDE_FILE"
TLOG="${TMPDIR:-/tmp}/turngait-v5-$JOBTAG.log"; : > "$TLOG"
existing=$(find_run || true)
if [ "$SMOKE" != 1 ] && [ -n "$existing" ] && [ -f "$existing/$FINAL_CKPT" ]; then
    echo "already trained: $existing"
else
    bhl_exec "$REPO/slurm/inner/train.sh" 2>&1 | tee "$TLOG" || true
    echo "=== training done: $(date) ==="
fi
run=$(find_run || true)
PARENT_RE="Loading model checkpoint from: .*${PARENT_DIR}/model_${PARENT_ITER}\.pt"
if [ "$SMOKE" = 1 ]; then
    why=""
    why="$why$(arm_guard "$TLOG")"
    grep -qE "$PARENT_RE" "$TLOG" || why="$why parent-checkpoint-not-loaded"
    grep -qE "train.sh: reached [0-9]+ logged iterations" "$TLOG" || why="$why no-logged-iterations"
    { [ -n "$run" ] && [ -f "$run/$FINAL_CKPT" ]; } || why="$why no-$FINAL_CKPT"
    grep -hE "Loading model checkpoint|train.sh: (reached|mean episode)|Metrics/base_velocity/(pure_turn_env|rest_turn_env|direct_wz_env)|push_levels|push_robot" "$TLOG" | tail -5 || true
    echo "=== done: $(date) ==="
    if [ -z "$why" ]; then echo "SMOKE: PASS ($RUN_NAME, $run)"; exit 0; else echo "SMOKE: FAIL ($RUN_NAME:$why)"; exit 1; fi
fi
{ [ -n "$run" ] && [ -f "$run/$FINAL_CKPT" ]; } || {
    echo "TURNGAIT RESULT: FAIL (training incomplete: no $FINAL_CKPT for $RUN_NAME)"; exit 1; }
# Wrong-arm / wrong-parent guards on a fresh training log.
if [ -s "$TLOG" ]; then
    g=$(arm_guard "$TLOG")
    [ -z "$g" ] || { echo "TURNGAIT RESULT: FAIL (training log is not the $ARM arm:$g)"; exit 1; }
    grep -qE "$PARENT_RE" "$TLOG" || {
        echo "TURNGAIT RESULT: FAIL (training log does not show $PARENT_DIR/model_$PARENT_ITER.pt being loaded)"; exit 1; }
fi
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
# --- turn test v2 (gating) + MuJoCo diagnosis (reported), CPU MuJoCo, outside the container ---
VLOG="${TMPDIR:-/tmp}/turngait-v5-v2-$JOBTAG.log"
v2_test "$cfg" "$RUN_NAME" | tee "$VLOG"
diag "$cfg" "$RUN_NAME"
line=$(grep "^TURN-V2 $RUN_NAME:" "$VLOG" | tail -1 || true)
case "$line" in
    *": PASS ("*) echo "TURNGAIT RESULT: PASS ($RUN_NAME, v2; fine-tune of $PARENT_RUN)";;
    *) echo "TURNGAIT RESULT: NEGATIVE ($RUN_NAME fails v2: ${line#*: })";;
esac
echo "=== qualification (cpu_turn_qualify.sbatch rule, unchanged) $(date) ==="
TURNQ_RUN="$RUN_NAME" bash "$REPO/slurm/repo20260923/cpu_turn_qualify.sbatch" 2>&1 | grep -E "TURN-QUALIFY|verdict|FAIL|INCOMPLETE" | tail -4 || true
arm_summary
"$PYH" - "$OUT_DIR" "$RES/qualify" "$PREFIX" "$ARM" <<'EOF'
import json, sys
from pathlib import Path
v2d, qd, prefix, arm = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
rows = {}
for s in range(3):
    run = f"{prefix}-s{s}"
    try:
        v2 = json.loads((v2d / f"{run}.json").read_text())["verdict"]
    except Exception:  # noqa: BLE001
        v2 = None
    try:
        q = json.loads((qd / f"{run}__qualify.json").read_text())
        rows[s] = (v2, q["verdict"], q["clauses"]["push"]["rate"])
    except Exception:  # noqa: BLE001
        rows[s] = (v2, None, None)
done = all(v[0] is not None and v[1] is not None for v in rows.values())
if arm == "push":
    k = sum(v[0] == "PASS" and v[1] == "QUALIFIED" for v in rows.values())
    verdict = ("PASS" if k >= 2 else "FAIL") if done else f"INCOMPLETE ({k} so far)"
    print(f"ARM-V5 {prefix}: {verdict} (rule: >= 2/3 seeds v2 PASS and QUALIFIED; per seed (v2, qualify, push) {rows})")
else:
    rates = [v[2] for v in rows.values() if v[2] is not None]
    hi, lo = sum(r > 0.30 for r in rates), sum(r <= 0.15 for r in rates)
    reading = ("continued training alone explains TurnRest's push regression" if hi >= 2 else
               "continued training alone does not (the rest-then-turn mix is implicated)" if lo >= 2 else
               "inconclusive")
    print(f"CONTROL-V5 {prefix}: {reading if done else 'INCOMPLETE'} (push rates {rates}; per seed (v2, qualify, push) {rows})")
EOF
echo "=== done: $(date) ==="
