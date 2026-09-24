#!/bin/bash
#SBATCH --job-name=sf04-ft
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# SF-04 follow-up (docs/SENSOR_FUSION.md section 6): one robustness ingredient
# at a time, fine-tuned from the working Both Full s0 MLP checkpoint.
#
#   sbatch slurm/repo20260923/gpu_sf04_finetune.sbatch <Arm> <iterations>
#     Arm        BothDelay | BothDrop | BothBias | BothRobust
#                (task Velocity-BHL-MazeRecovery-Full-<Arm>-v0)
#     iterations new PPO iterations on top of the teacher's 5997 (1024 envs,
#                ~3.4 s each on the weekend nodes: 1000 ~ 1 h, 2000 ~ 2 h;
#                more than ~2300 does not fit the 3 h wall and is truncated)
#
# Inside the v60 container (one Kit boot per bhl_exec, each behind v60_boot_gate):
#   1. CONTROL first: the untouched teacher on the plain Both task, seven
#      settings -> sf04-finetune/control-both-s0.json (kept if it exists). It
#      runs before training so a failed or truncated fine-tune still leaves an
#      informative job.
#   2. TRAIN: slurm/inner/train.sh, OVERRIDE_FILE = agent.resume=true
#      agent.load_run=2026-09-19_16-25-50_wknd-full-both-s0
#      agent.load_checkpoint=model_5997.pt, RUN_NAME sf04-ft-<arm>-s0, SEED 0,
#      NUM_ENVS 1024, MAX_ITER <iterations>, no BHL_POLICY overlay (MLP).
#      Bounded by a timeout that leaves room for the evaluation.
#   3. EVAL: newest checkpoint of that run (maze_checkpoint.py resolve
#      --newer-than <job start>) with maze_recovery_probe.py --keep-corruption
#      --num-envs 32 --steps 600 --seed 100 and the same seven settings
#      -> sf04-finetune/<Arm>-s0.json.
#   4. Verdict from the JSONs (host python, no torch):
#        PASS      arm baseline >= 28/32 AND the trained-for setting beats the
#                  control by >= 8 envs (BothDelay: delay1; BothDrop:
#                  min(lidar_off, stereo_off); BothBias: gyro0.10; BothRobust:
#                  delay1, lidar_off and stereo_off each)
#        NEGATIVE  both JSONs complete but the rule fails
#        INCOMPLETE anything missing or a setting recorded an error
#      Final line: 'SF04-FT <Arm> RESULT: PASS|NEGATIVE|INCOMPLETE'. Exit 0 for
#      PASS and NEGATIVE (both are results), 1 for INCOMPLETE.
set -uo pipefail
ARM=${1:-}; ITERS=${2:-}
case "$ARM" in BothDelay|BothDrop|BothBias|BothRobust) ;;
    *) echo "SF04-FT ${ARM:-?} RESULT: INCOMPLETE (usage: sbatch gpu_sf04_finetune.sbatch <BothDelay|BothDrop|BothBias|BothRobust> <iterations>; got arm '$ARM')"; exit 2;; esac
# Positive integer without a leading zero: '00' would reach --max_iterations as 0 and re-save the teacher untouched.
case "$ITERS" in ''|*[!0-9]*|0*) echo "SF04-FT $ARM RESULT: INCOMPLETE (iterations must be a positive integer, got '$ITERS')"; exit 2;; esac
# Every step below boots Isaac Sim inside the container; that only ever happens on an allocated GPU node.
[ -n "${SLURM_JOB_ID:-}" ] || { echo "SF04-FT $ARM RESULT: INCOMPLETE (not inside a Slurm allocation; submit with sbatch)"; exit 2; }

export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}" OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
unset BHL_POLICY BHL_RND BHL_SYMMETRY # MLP teacher: never forward the train.py overlays
source "$REPO/slurm/inner/v60_boot_gate.sh"
BOOT_LOCK="${V60_BOOT_LOCK:-/nfs/hpc/share/$USER/Humanoid_Lite/.v60-boot.lock}"
PYH=$WORKSPACE/venv/bin/python; [ -x "$PYH" ] || PYH=$(command -v python3)
OUT=$REPO/results/repo-gpu-20260923/sf04-finetune; mkdir -p "$OUT"
INNER=$REPO/slurm/repo20260923/inner_sf04_finetune.sh
TEACHER=$UPSTREAM/logs/rsl_rl/biped/2026-09-19_16-25-50_wknd-full-both-s0/model_5997.pt
WALL_S=${WALL_S:-10800}               # must match #SBATCH --time
EVAL_RESERVE_S=${EVAL_RESERVE_S:-1500} # arm eval boot + 7 rollouts (~5-9 min) + verdict
JOB_START=$(date +%s)

echo "=== node: $(hostname) | SF-04 fine-tune $ARM x $ITERS it | stack $BHL_STACK | job ${SLURM_JOB_ID:-?} | $(date) ==="
cd "$REPO" && echo "repo HEAD $(git rev-parse --short HEAD 2>/dev/null) branch $(git branch --show-current 2>/dev/null)"

# Preflight (no Kit): the probe interface, the teacher, and whether P1's task id has landed.
grep -q -- "--keep-corruption" "$REPO/scripts/bench/maze_recovery_probe.py" && grep -q -- "--settings" "$REPO/scripts/bench/maze_recovery_probe.py" \
    || { echo "SF04-FT $ARM RESULT: INCOMPLETE (probe lacks --keep-corruption/--settings)"; exit 1; }
[ -f "$TEACHER" ] || { echo "SF04-FT $ARM RESULT: INCOMPLETE (teacher checkpoint missing: $TEACHER)"; exit 1; }
grep -q -- "\"$ARM\"" "$REPO/src/bhl_robust/tasks/maze_recovery_env_cfg.py" "$REPO/src/bhl_robust/tasks/maze_robust.py" 2>/dev/null \
    || echo "WARNING: '$ARM' is not named in maze_recovery_env_cfg.py/maze_robust.py -- Velocity-BHL-MazeRecovery-Full-$ARM-v0 may not be registered yet; training will fail and only the control will be produced"
EST_S=$(( ITERS * 34 / 10 + 300 )); MAX_TRAIN_S=$(( WALL_S - EVAL_RESERVE_S - 900 ))
[ "$EST_S" -le "$MAX_TRAIN_S" ] || echo "WARNING: $ITERS iterations (~${EST_S}s at 3.4 s/it) exceed the ~${MAX_TRAIN_S}s training budget; the run will be truncated and its newest 100-iteration checkpoint evaluated"

# One Kit boot per bhl_exec; release this job's boot lock as soon as its Kit is gone.
boot() { v60_boot_gate; GATE_PID=${!:-}; }
release() { kill "${GATE_PID:-}" 2>/dev/null || true; _v60_release "$BOOT_LOCK"; }

# 1. Control (teacher on the plain Both task) -- first, so the job is informative whatever training does.
if [ -s "$OUT/control-both-s0.json" ]; then
    echo "=== control-both-s0.json exists; not re-running the control ==="
else
    boot; echo "=== control | t=${SECONDS}s | $(date) ==="
    bhl_exec "$INNER" control "$(( WALL_S - SECONDS - 300 ))"; echo "=== control exit $? | t=${SECONDS}s ==="; release
fi

# 2. Fine-tune, bounded so the evaluation still fits. The timeout is taken
#    AFTER the boot gate (which can wait up to an hour), from the wall time
#    actually left; taken before it, a long wait would push the kill past the
#    wall and Slurm would end the job before the RESULT line.
if [ $(( WALL_S - SECONDS - EVAL_RESERVE_S )) -lt 600 ]; then
    echo "=== only $(( WALL_S - SECONDS - EVAL_RESERVE_S ))s left for training after the control; skipping train and eval ==="
else
    boot
    TRAIN_TMO=$(( WALL_S - SECONDS - EVAL_RESERVE_S )); [ "$TRAIN_TMO" -lt 60 ] && TRAIN_TMO=60   # 'timeout 0' = no limit
    echo "=== train $ARM $ITERS it (timeout ${TRAIN_TMO}s) | t=${SECONDS}s | $(date) ==="
    if [ "${EVAL_ONLY:-0}" = 1 ]; then echo "=== EVAL_ONLY=1: skipping training; evaluating the newest existing checkpoint of $ARM ==="; train_rc=0; JOB_START=0; else
    bhl_exec "$INNER" train "$ARM" "$ITERS" "$TRAIN_TMO"; train_rc=$?; fi
    echo "=== train exit $train_rc | t=${SECONDS}s ==="; release
    # 3. Evaluate the newest checkpoint written by this job's run.
    if [ $(( WALL_S - SECONDS )) -lt 420 ]; then
        echo "=== $(( WALL_S - SECONDS ))s left: not enough for the evaluation boot; skipping ==="
    else
        boot; echo "=== eval $ARM | t=${SECONDS}s | $(date) ==="
        # The probe is bounded by the wall time left (minus the verdict), so a hung Kit cannot eat the RESULT line.
        bhl_exec "$INNER" eval "$ARM" "$JOB_START" "$(( WALL_S - SECONDS - 300 ))"; echo "=== eval exit $? | t=${SECONDS}s ==="; release
    fi
fi

# 4. Table and verdict from the JSONs, never from exit codes.
"$PYH" - "$OUT" "$ARM" <<'PY'
import json, os, sys
out, arm = sys.argv[1], sys.argv[2]
names = ["baseline", "delay1", "delay2", "lidar_off", "stereo_off", "both_off", "gyro0.10"]
trained_for = {"BothDelay": ["delay1"], "BothDrop": ["lidar_off", "stereo_off"],
               "BothBias": ["gyro0.10"], "BothRobust": ["delay1", "lidar_off", "stereo_off"]}[arm]
problems = []

def load(label, path):
    if not os.path.exists(path):
        problems.append(f"{label}: missing {os.path.basename(path)}"); return None
    try:
        d = json.load(open(path))
    except Exception as exc:
        problems.append(f"{label}: unreadable {os.path.basename(path)}: {exc}"); return None
    n = d.get("num_envs")
    if n != 32:
        problems.append(f"{label}: num_envs {n} != 32")
    rows = {}
    for s in d.get("settings") or []:
        if "error" in s:
            problems.append(f"{label}: setting {s.get('name')} recorded an error"); rows[s.get("name")] = None
        elif "first_episode_success_rate" in s:
            rows[s["name"]] = round(float(s["first_episode_success_rate"]) * (n or 32))
    for nm in names:
        if nm not in rows:
            problems.append(f"{label}: setting {nm} absent"); rows[nm] = None
    return {"counts": rows, "checkpoint": d.get("checkpoint", ""), "task": d.get("task", ""),
            "corruption": d.get("observation_corruption")}

ctrl = load("control", os.path.join(out, "control-both-s0.json"))
ft = load(arm, os.path.join(out, f"{arm}-s0.json"))
# Identity: the control persists across jobs and the arm JSON must be the arm's own task.
if ctrl:
    if ctrl["task"] != "Velocity-BHL-MazeRecovery-Full-Both-v0":
        problems.append(f"control: task {ctrl['task']!r} is not the plain Both Full task")
    if not ctrl["checkpoint"].endswith("2026-09-19_16-25-50_wknd-full-both-s0/model_5997.pt"):
        problems.append(f"control: checkpoint {ctrl['checkpoint']!r} is not the Both Full s0 teacher")
    if ctrl["corruption"] is not True:
        problems.append("control: observation_corruption is not True (--keep-corruption missing)")
if ft:
    if ft["task"] != f"Velocity-BHL-MazeRecovery-Full-{arm}-v0":
        problems.append(f"{arm}: task {ft['task']!r} is not Velocity-BHL-MazeRecovery-Full-{arm}-v0")
    if f"sf04-ft-{arm.lower()}-s0/" not in ft["checkpoint"]:
        problems.append(f"{arm}: checkpoint {ft['checkpoint']!r} is not from run sf04-ft-{arm.lower()}-s0")
    if ft["corruption"] is not True:
        problems.append(f"{arm}: observation_corruption is not True (--keep-corruption missing)")
train = {}
try:
    train = json.load(open(os.path.join(out, f"{arm}-s0.train.json")))
except Exception:
    pass

def cell(v):
    return "--" if v is None else f"{v}/32"
print()
print(f"{'policy':22}" + "".join(f"{nm:>11}" for nm in names) + "  checkpoint")
for label, d in (("control (Both teacher)", ctrl), (f"{arm} fine-tuned", ft)):
    if d is None:
        print(f"{label:22}" + "".join(f"{'--':>11}" for _ in names) + "  (no JSON)"); continue
    ck = d["checkpoint"]; ck = os.path.join(os.path.basename(os.path.dirname(ck)), os.path.basename(ck)) if ck else "?"
    print(f"{label:22}" + "".join(f"{cell(d['counts'][nm]):>11}" for nm in names) + f"  {ck}")
if train:
    print(f"\ntrain: {train.get('task')} run {train.get('run_name')} requested {train.get('iterations_requested')} it, "
          f"rc {train.get('train_rc')}{' (TIMED OUT)' if train.get('timed_out') else ''}, "
          f"wall {int(train.get('finished_epoch', 0)) - int(train.get('started_epoch', 0))}s, checkpoint {train.get('checkpoint')}")
for p in problems:
    print("PROBLEM:", p)

verdict = "INCOMPLETE"
if ctrl and ft and not problems:
    c, f = ctrl["counts"], ft["counts"]
    checks = [(f"baseline {f['baseline']} >= 28/32", f["baseline"] >= 28)]
    if arm == "BothDrop":
        m_ft, m_c = min(f[nm] for nm in trained_for), min(c[nm] for nm in trained_for)
        checks.append((f"min(lidar_off, stereo_off) {m_ft} vs control {m_c} (>= +8)", m_ft - m_c >= 8))
    else:
        for nm in trained_for:
            checks.append((f"{nm} {f[nm]} vs control {c[nm]} (>= +8)", f[nm] - c[nm] >= 8))
    for text, ok in checks:
        print(f"check: {text}: {'ok' if ok else 'FAILED'}")
    verdict = "PASS" if all(ok for _, ok in checks) else "NEGATIVE"
print(f"\nSF04-FT {arm} RESULT: {verdict}")
sys.exit(0 if verdict in ("PASS", "NEGATIVE") else 1)
PY
rc=$?
echo "=== done: t=${SECONDS}s | $(date) ==="
exit $rc
