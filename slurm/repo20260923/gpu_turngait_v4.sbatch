#!/bin/bash
#SBATCH --job-name=turngait-v4
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --array=0-2%2
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
# 2026-09-27: turning-gait arm TurnRest, FINE-TUNED from arms-turn-turnboth-s0
# (model_5999.pt), 3 fine-tune seeds (array 0-2 = seed 0-2), v51, DR scale 1.0
# via dr_levels.sh (the parent's recipe), 3000 further iterations.
# Task Velocity-BHL-Arms-TurnRest-v0 = arms_env_cfg.HumanoidTurnRestCfg =
# TurnBoth's reward set UNCHANGED (so the resumed value function matches) + a
# command mix (turn_command.TurnRestMixVelocityCommand), per resample:
#   15 % PURE_TURN  (0, 0, +/-U(0.3, 1.0)) sustained, no heading control
#   25 % REST_TURN  zero command for U(1.5, 4.0) s, then (0, 0, +/-U(0.3, 1.0)) sustained
#   20 % DIRECT     vx U(-0.5,0.5), vy U(-0.25,0.25), wz U(-1,1), no heading control
#   40 % upstream   heading-derived wz; the 2 % standing draw applies to all.
# Obs/actions unchanged (75 / 22), same export and deploy path.
#
# WHY (diagnosis, 2026-09-27, scripts/bench/turn_diagnose.py mujoco, reset seeds
# 0-2, 3 s zero-command settle then +/-0.6 rad/s for 6 s, all 12 turning-arm
# checkpoints; diagnose/mujoco_{existing,hip_track}_s012.json): 10 of the 12
# make ZERO lift-offs under a pure turn although their actions do respond to the
# command (||a(cmd) - a(0)|| 1.1-2.4, against ||a|| ~ 47-109 raw; TurnBoth-s0's
# 2.0): an 8-20 deg double-stance twist, never a step. TurnTrack-s0 steps and
# turns in the -wz direction only (2/6 runs >= 150 deg, one of them from a true
# standstill). TurnBoth-s0, the only checkpoint that turns both ways (5/6), is
# still stepping in place after the settle (1-9 lift-offs per foot in the last
# 2 s); its one failure is a run in which it had nearly stopped, though the
# opposite direction from the same settle state stepped and turned 212 deg, so
# near-rest alone does not explain it. Pooled over the 72 runs, a step was
# started from rest in 1 of 50 at-rest runs. The missing behaviour is step
# initiation from rest under a pure turn; REST_TURN asks for
# exactly that, and fine-tuning starts from the only checkpoint that already
# has a stepping turn in both directions (with TurnBoth's reward set, which the
# arm keeps unchanged). Isaac training curves (turn_diagnose.py tb): every
# turning arm, and the shipped arms-dr1.0, still ends ~27-35 % of episodes in a
# fall at iteration 6000, so continued training is part of the treatment.
# KNOWN CONFOUND: this arm changes the command mix AND trains 3000 more
# iterations; there is no
# continued-training-only control, so a PASS cannot be attributed to REST_TURN.
# It is also three fine-tunes of ONE parent, not three independent recipes.
#
# PREDECLARED (2026-09-27, before any TurnRest training):
#   seed PASS  = turn_test.py --protocol v2 verdict PASS (unchanged from v3): for
#                reset seeds 0, 1, 2, 3.0 s warm-up then (0,0,+0.6) and (0,0,-0.6)
#                rad/s for 6 s each; ALL SIX runs >= 150 deg in the commanded
#                direction, no fall; AND the walk run (0.35,0,0), 1.0 s warm-up,
#                |yaw drift| <= 15 deg, no fall. Missing/unreadable JSON = FAIL.
#   arm PASS   = >= 2 of the 3 seeds PASS.
#   A passing arm then faces the matched-push regression as a separate CPU job
#   (cpu_turn_qualify.sbatch with TURNQ_RUN=<seed run>: its push clause, 0.5 m/s
#   shoves every 3 s, 12 s, n = 60, fall rate <= 0.15; arms-dr1.0 0.10) before any
#   use as the Mission 7 / carry gait.
#   Disclosure: reset seeds 0-2 were used to inspect the PARENT (v2 5/6 and the
#   MuJoCo diagnosis above), and that inspection shaped this arm. The children
#   are new policies, never run before this job; the rule is v3's, unchanged.
#   Non-gating, reported: turn_diagnose.py mujoco on each child (lift-offs in the
#   last 2 s of the settle tell a gait that starts a step from rest apart from
#   one that never stops stepping).
#
# Verdicts are computed from JSON, never exit codes (Isaac exits 0 on a crash):
# training counts only if the run dir holds model_<5999 + FT_ITERS - 1>.pt (rsl-rl
# resumes at the loaded iteration: from model_5999, 3000 iterations end at
# model_8998.pt) and a fresh training log shows both the parent checkpoint being
# loaded and TurnRestMixVelocityCommand in the command manager.
# Outputs (new directories, nothing published is overwritten):
#   results/repo-gpu-20260923/turn-20260927/turn-test-v2/<run>.json
#   results/repo-gpu-20260923/turn-20260927/diagnose/<run>.json
#   results/repo-gpu-20260923/turn-20260927/isaac-probe/isaac_probe.json
#
# Modes (env vars; everything else as the array run):
#   TURNGAIT_SMOKE=1      MAX_ITER=3, NUM_ENVS=64, run *-smoke-s<seed>; no export /
#                         test. "SMOKE: PASS|FAIL (...)" requires: parent
#                         checkpoint loaded (log line), TurnRestMixVelocityCommand
#                         in the command manager, iterations logged, model_6001.pt.
#                           sbatch --array=0 --time=01:00:00 --export=ALL,TURNGAIT_SMOKE=1 <this>
#                         Exit 0 iff SMOKE: PASS, so the array can chain afterok.
#   TURNGAIT_EVAL_ONLY=1  no training, no GPU: v2 + MuJoCo diagnosis on existing
#                         exported runs (TURNGAIT_RUNS, default the three
#                         arms-turn-turnrest-ft-s* runs). CPU node, via --wrap so
#                         this header's GPU / array directives do not apply:
#                           sbatch --account=eecs --partition=share --constraint=el9 --cpus-per-task=4 --mem=16G \
#                             --time=02:00:00 --job-name=turngait-v4eval \
#                             --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out \
#                             --export=ALL,TURNGAIT_EVAL_ONLY=1 --wrap="bash <this>"
#   TURNGAIT_ISAAC_PROBE=1  GPU, no training: turn_diagnose.py isaac on
#                         TURNGAIT_PROBE_RUNS (default turncmd-s0/s1/s2,
#                         turnboth-s0/s1, arms-dr1.0-s0): the exported actors in
#                         Isaac, 32 envs (16 at +0.6, 16 at -0.6 rad/s) after a 3 s
#                         zero-command settle. UNTESTED before its first run (no
#                         Isaac on the login node). Reading rule, declared now:
#                         a run that turns >= 150 deg in >= 24/32 envs in Isaac
#                         but 0/6 in MuJoCo = sim2sim gap; < 8/32 in Isaac = pure-
#                         turn tracking not learned; in between = mixed, reported.
#                           sbatch --array=0 --time=01:00:00 --export=ALL,TURNGAIT_ISAAC_PROBE=1 <this>
#   TURNGAIT_OUT_DIR      override the result root (testing only).
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
PREFIX=arms-turn-turnrest-ft

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
          f"[three fine-tunes of arms-turn-turnboth-s0; confounded with +3000 iterations]")
EOF
}

# ---------------------------------------------------------------- eval only (CPU)
if [ "${TURNGAIT_EVAL_ONLY:-0}" = 1 ]; then
    RUNS=${TURNGAIT_RUNS:-"$PREFIX-s0 $PREFIX-s1 $PREFIX-s2"}
    echo "=== turngait v4 eval only | $(hostname) | HEAD $(git -C "$REPO" rev-parse --short HEAD) | $(date) ==="
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

# ---------------------------------------------------------------- Isaac probe (GPU)
if [ "${TURNGAIT_ISAAC_PROBE:-0}" = 1 ]; then
    RUNS=${TURNGAIT_PROBE_RUNS:-"arms-turn-turncmd-s0 arms-turn-turncmd-s1 arms-turn-turncmd-s2 arms-turn-turnboth-s0 arms-turn-turnboth-s1 arms-dr1.0-s0"}
    PDIR=$RES/isaac-probe; mkdir -p "$PDIR" "$WORKSPACE/overrides"
    dirs=(); labels=()
    for r in $RUNS; do
        d=$(ls -d "$LOGROOT"/*_"$r" 2>/dev/null | sort | tail -1 || true)
        if [ -z "$d" ] || [ ! -f "$d/exported/policy.pt" ]; then echo "ISAAC $r: skipped (no exported/policy.pt)"; continue; fi
        dirs+=("$d"); labels+=("$r")
    done
    [ "${#dirs[@]}" -gt 0 ] || { echo "ISAAC-PROBE RESULT: INCOMPLETE (no exported runs)"; exit 1; }
    export BENCH_OUT="$PDIR/.isaac_probe.$JOBTAG.json"; rm -f "$BENCH_OUT"
    inner="$WORKSPACE/overrides/turn-isaac-probe-$JOBTAG.sh"
    {
        echo '#!/bin/bash'
        echo 'set -uo pipefail'
        echo 'cd "$REPO"; export PYTHONPATH="$REPO/src:$REPO/scripts/bench:${PYTHONPATH:-}"'
        printf 'timeout --kill-after=60 2400 "$PY" "$REPO/scripts/bench/turn_diagnose.py" isaac --num-envs 32 --wz 0.6 --warm 3.0 --seconds 6.0 --out "$BENCH_OUT" --run-dirs'
        printf ' %q' "${dirs[@]}"; printf ' --labels'; printf ' %q' "${labels[@]}"; echo ' 2>&1 | grep -vE "pthread_setaffinity|EGLError"'
    } > "$inner"
    echo "=== Isaac probe | $(hostname) | HEAD $(git -C "$REPO" rev-parse --short HEAD) | runs: ${labels[*]} | $(date) ==="
    cat "$inner"
    bhl_exec "$inner" || true
    "$PYH" - "$BENCH_OUT" "$PDIR/isaac_probe.json" "${labels[@]}" <<'EOF'
import json, os, sys
tmp, final, labels = sys.argv[1], sys.argv[2], sys.argv[3:]
try:
    d = json.load(open(tmp))
except Exception as exc:  # noqa: BLE001
    print(f"ISAAC-PROBE RESULT: INCOMPLETE (no readable probe JSON: {exc!r}; Isaac exits 0 on a crash)")
    sys.exit(1)
missing = [l for l in labels if l not in d.get("runs", {})]
for l, r in d.get("runs", {}).items():
    k, n = r["n_ge_150deg_no_fall"], r["n"]
    reading = ("turns in Isaac (>= 24/32)" if k >= 24 else "does not turn in Isaac (< 8/32)" if k < 8 else "mixed")
    print(f"ISAAC {l}: {k}/{n} envs >= 150 deg in the commanded direction, no fall; median +dir {r['median_yaw_deg_plus']:.1f}, "
          f"-dir {r['median_yaw_deg_minus']:.1f} deg; median lift-offs {r['median_liftoffs']}; falls {r['n_fell']} -> {reading}")
final_path = final
if os.path.exists(final_path):
    final_path = final.replace(".json", f".{os.environ.get('SLURM_JOB_ID', 'local')}.json")
os.replace(tmp, final_path)
print(f"wrote {final_path}")
print(f"ISAAC-PROBE RESULT: {'INCOMPLETE (missing ' + ', '.join(missing) + ')' if missing else 'COMPLETE'} "
      "(LEARNED policies replayed in Isaac; reading rule: >= 24/32 in Isaac with 0/6 in MuJoCo = sim2sim gap, "
      "< 8/32 = not learned)")
EOF
    echo "=== done: $(date) ==="
    exit 0
fi

# ---------------------------------------------------------------- fine-tune (GPU)
SEED=${SLURM_ARRAY_TASK_ID:-0}
SMOKE=${TURNGAIT_SMOKE:-0}
export TASK="Velocity-BHL-Arms-TurnRest-v0"
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
TLOG="${TMPDIR:-/tmp}/turngait-v4-$JOBTAG.log"; : > "$TLOG"
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
    grep -q "TurnRestMixVelocityCommand" "$TLOG" || why="$why no-TurnRestMixVelocityCommand-in-command-manager"
    grep -qE "$PARENT_RE" "$TLOG" || why="$why parent-checkpoint-not-loaded"
    grep -qE "train.sh: reached [0-9]+ logged iterations" "$TLOG" || why="$why no-logged-iterations"
    { [ -n "$run" ] && [ -f "$run/$FINAL_CKPT" ]; } || why="$why no-$FINAL_CKPT"
    grep -hE "Loading model checkpoint|train.sh: (reached|mean episode)|Metrics/base_velocity/(pure_turn_env|rest_turn_env|direct_wz_env)" "$TLOG" | tail -5 || true
    echo "=== done: $(date) ==="
    if [ -z "$why" ]; then echo "SMOKE: PASS ($RUN_NAME, $run)"; exit 0; else echo "SMOKE: FAIL ($RUN_NAME:$why)"; exit 1; fi
fi
{ [ -n "$run" ] && [ -f "$run/$FINAL_CKPT" ]; } || {
    echo "TURNGAIT RESULT: FAIL (training incomplete: no $FINAL_CKPT for $RUN_NAME)"; exit 1; }
# Wrong-arm / wrong-parent guards on a fresh training log.
if [ -s "$TLOG" ]; then
    grep -q "TurnRestMixVelocityCommand" "$TLOG" || {
        echo "TURNGAIT RESULT: FAIL (training log has no TurnRestMixVelocityCommand: $TASK is not the TurnRest arm)"; exit 1; }
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
VLOG="${TMPDIR:-/tmp}/turngait-v4-v2-$JOBTAG.log"
v2_test "$cfg" "$RUN_NAME" | tee "$VLOG"
diag "$cfg" "$RUN_NAME"
line=$(grep "^TURN-V2 $RUN_NAME:" "$VLOG" | tail -1 || true)
case "$line" in
    *": PASS ("*) echo "TURNGAIT RESULT: PASS ($RUN_NAME, v2; fine-tune of $PARENT_RUN)";;
    *) echo "TURNGAIT RESULT: NEGATIVE ($RUN_NAME fails v2: ${line#*: })";;
esac
arm_summary
echo "=== done: $(date) ==="
