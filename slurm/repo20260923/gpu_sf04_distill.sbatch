#!/bin/bash
#SBATCH --job-name=sf04-distill
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# SF-04 follow-up, "something different" (docs/SENSOR_FUSION.md section 6):
# teacher-student distillation instead of PPO under degraded sensing.
#
#   sbatch slurm/repo20260923/gpu_sf04_distill.sbatch
#
# Teacher: the published Both Full s0 MLP
#   external/.../logs/rsl_rl/biped/2026-09-19_16-25-50_wknd-full-both-s0/model_5997.pt
# frozen, reading the CLEAN `teacher` observation group (Both terms, 113 dims).
# Student: LSTM (256, 1 layer) + MLP [256,128,128], reading the DEGRADED
# `policy` group (BothRobust: per-episode LiDAR/stereo dropout p 0.2, IMU bias
# std 0.02, IMU delay 0-1 step; identical 113-dim layout). MSE behaviour
# cloning on the student's own rollouts, 24-step truncated BPTT, lr 1e-3,
# 2000 iterations x 24 steps x 1024 envs. Task
#   Velocity-BHL-MazeRecovery-Full-BothRobustDistill-v0
# (src/bhl_robust/tasks/maze_distill.py: Full-stage BothRobust recovery env,
# same rewards/terminations/spawn, plus the teacher group).
#
# Inside the v60 container (one Kit boot per bhl_exec, each behind v60_boot_gate):
#   1. TRAIN  slurm/repo20260923/inner_sf04_distill.sh train <timeout>:
#      scripts/train_distill.py (DistillationRunner), teacher through
#      --resume True --load_run <teacher run> --checkpoint model_5997.pt,
#      RUN_NAME sf04-distill-s0, SEED 0, NUM_ENVS 1024, MAX_ITER 2000. Bounded
#      by a timeout that leaves room for the evaluation; a truncated run is
#      evaluated at its newest 100-iteration checkpoint.
#   2. EVAL   inner_sf04_distill.sh eval <job start>: the STUDENT, on its own
#      degraded observations with the per-episode randomization forced off and
#      the conditions explicit, maze_recovery_probe.py --runner distillation
#      --keep-corruption --num-envs 32 --steps 600 --seed 100, settings
#      baseline / delay1 / delay2 / lidar_off / stereo_off / both_off
#      -> results/repo-gpu-20260923/sf04-distill/student-s0.json
#   3. VERDICT from the JSON (host python, no torch), predeclared:
#        PASS       baseline >= 28/32 AND delay1 >= 24/32 AND lidar_off >= 24/32
#                   AND stereo_off >= 24/32
#        NEGATIVE   all six settings present without error, rule fails
#        INCOMPLETE JSON missing, a setting absent or recorded an error
#      Final line 'SF04-DISTILL RESULT: PASS|NEGATIVE|INCOMPLETE'. Exit 0 for
#      PASS and NEGATIVE (both are results), 1 for INCOMPLETE. If the SF-04
#      fine-tune control (sf04-finetune/control-both-s0.json, the untouched
#      teacher on the plain Both task) exists, its row is printed for context.
#
# Cost: ~2 h training (rollout-bound, same sim cost per iteration as PPO on
# 1024 envs, cheaper update) + ~10 min evaluation, one GPU. Nothing else is
# spent; the job does not retrain on failure.
set -uo pipefail
# Every step below boots Isaac Sim inside the container; that only ever happens on an allocated GPU node.
[ -n "${SLURM_JOB_ID:-}" ] || { echo "SF04-DISTILL RESULT: INCOMPLETE (not inside a Slurm allocation; submit with sbatch)"; exit 2; }
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}" OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
unset BHL_POLICY                      # the student architecture is the distillation cfg's, never the PPO overlay
source "$REPO/slurm/inner/v60_boot_gate.sh"
BOOT_LOCK="${V60_BOOT_LOCK:-/nfs/hpc/share/$USER/Humanoid_Lite/.v60-boot.lock}"
PYH=$WORKSPACE/venv/bin/python; [ -x "$PYH" ] || PYH=$(command -v python3)
OUT=$REPO/results/repo-gpu-20260923/sf04-distill; mkdir -p "$OUT"
INNER=$REPO/slurm/repo20260923/inner_sf04_distill.sh
CONTROL=$REPO/results/repo-gpu-20260923/sf04-finetune/control-both-s0.json
WALL_S=${WALL_S:-14400}               # must match #SBATCH --time
EVAL_RESERVE_S=${EVAL_RESERVE_S:-2400} # boot-gate wait + eval boot + the inner 1800 s probe timeout + verdict
JOB_START=$(date +%s)

# Forwarded into the container (all listed in BHL_FORWARD_VARS; the inner
# script uses the same defaults, so these are the single place to change them).
export TASK=Velocity-BHL-MazeRecovery-Full-BothRobustDistill-v0
export LOAD_RUN=2026-09-19_16-25-50_wknd-full-both-s0
export RUN_NAME=sf04-distill-s0
export SEED=0
export NUM_ENVS=1024
export MAX_ITER=2000
export TRAIN_SCRIPT=$REPO/scripts/train_distill.py
export ENABLE_CAMERAS=1
TEACHER=$UPSTREAM/logs/rsl_rl/biped/$LOAD_RUN/model_5997.pt

echo "=== node: $(hostname) | SF-04 distillation $RUN_NAME | stack $BHL_STACK | job ${SLURM_JOB_ID:-?} | $(date) ==="
cd "$REPO" && echo "repo HEAD $(git rev-parse --short HEAD 2>/dev/null) branch $(git branch --show-current 2>/dev/null)"

# Preflight (no Kit): teacher, task registration, probe option, entrypoint.
[ -f "$TEACHER" ] || { echo "SF04-DISTILL RESULT: INCOMPLETE (teacher checkpoint missing: $TEACHER)"; exit 1; }
[ -f "$REPO/src/bhl_robust/tasks/maze_distill.py" ] || { echo "SF04-DISTILL RESULT: INCOMPLETE (src/bhl_robust/tasks/maze_distill.py missing)"; exit 1; }
grep -q "BothRobustDistill-v0" "$REPO/src/bhl_robust/tasks/__init__.py" \
    || { echo "SF04-DISTILL RESULT: INCOMPLETE ($TASK not registered in src/bhl_robust/tasks/__init__.py)"; exit 1; }
grep -q -- "--runner" "$REPO/scripts/bench/maze_recovery_probe.py" && grep -q -- "--settings" "$REPO/scripts/bench/maze_recovery_probe.py" \
    || { echo "SF04-DISTILL RESULT: INCOMPLETE (probe lacks --runner/--settings)"; exit 1; }
[ -f "$TRAIN_SCRIPT" ] || { echo "SF04-DISTILL RESULT: INCOMPLETE (missing $TRAIN_SCRIPT)"; exit 1; }
rm -f "$OUT/student-s0.json"          # the verdict may only come from this job's evaluation

# One Kit boot per bhl_exec; release this job's boot lock as soon as its Kit is gone.
boot() { v60_boot_gate; GATE_PID=${!:-}; }
release() { kill "${GATE_PID:-}" 2>/dev/null || true; _v60_release "$BOOT_LOCK"; }

# 1. Distil, bounded so the evaluation still fits. The timeout is taken AFTER
#    the boot gate (which can wait up to an hour), from the wall time actually
#    left; taken before it, a long wait would push the kill past the wall and
#    Slurm would end the job before the RESULT line.
boot
TRAIN_TMO=$(( WALL_S - SECONDS - EVAL_RESERVE_S - 300 )); [ "$TRAIN_TMO" -lt 60 ] && TRAIN_TMO=60   # 'timeout 0' = no limit
echo "=== train (timeout ${TRAIN_TMO}s) | t=${SECONDS}s | $(date) ==="
bhl_exec "$INNER" train "$TRAIN_TMO"; train_rc=$?
echo "=== train exit $train_rc | t=${SECONDS}s ==="; release

# 2. Evaluate the newest checkpoint written by this job's run.
if [ $(( WALL_S - SECONDS )) -lt 420 ]; then
    echo "=== $(( WALL_S - SECONDS ))s left: not enough for the evaluation boot; skipping ==="
else
    boot; echo "=== eval student | t=${SECONDS}s | $(date) ==="
    bhl_exec "$INNER" eval "$JOB_START"; echo "=== eval exit $? | t=${SECONDS}s ==="; release
fi

# 3. Table and verdict from the JSON, never from exit codes.
"$PYH" - "$OUT" "$CONTROL" "$TASK" <<'PY'
import json, os, sys
out, control, task = sys.argv[1], sys.argv[2], sys.argv[3]
names = ["baseline", "delay1", "delay2", "lidar_off", "stereo_off", "both_off"]
rule = {"baseline": 28, "delay1": 24, "lidar_off": 24, "stereo_off": 24}
problems = []

def load(label, path, required):
    if not os.path.exists(path):
        if required:
            problems.append(f"{label}: missing {os.path.basename(path)}")
        return None
    try:
        d = json.load(open(path))
    except Exception as exc:
        if required:
            problems.append(f"{label}: unreadable {os.path.basename(path)}: {exc}")
        return None
    n = d.get("num_envs")
    if required and n != 32:
        problems.append(f"{label}: num_envs {n} != 32")
    rows = {}
    for s in d.get("settings") or []:
        if "error" in s:
            if required:
                problems.append(f"{label}: setting {s.get('name')} recorded an error")
            rows[s.get("name")] = None
        elif "first_episode_success_rate" in s:
            rows[s["name"]] = round(float(s["first_episode_success_rate"]) * (n or 32))
    for nm in names:
        if nm not in rows:
            if required:
                problems.append(f"{label}: setting {nm} absent")
            rows[nm] = None
    return {"counts": rows, "checkpoint": d.get("checkpoint", ""), "task": d.get("task", ""),
            "corruption": d.get("observation_corruption")}

student = load("student", os.path.join(out, "student-s0.json"), required=True)
ctrl = load("control", control, required=False)
train = {}
try:
    train = json.load(open(os.path.join(out, "train-s0.json")))
except Exception:
    pass
if student and student["task"] != task:
    problems.append(f"student: task {student['task']} != {task}")

def cell(v):
    return "--" if v is None else f"{v}/32"
print()
print(f"{'policy':26}" + "".join(f"{nm:>11}" for nm in names) + "  checkpoint")
for label, d in (("control (Both teacher, MLP)", ctrl), ("student (LSTM, degraded)", student)):
    if d is None:
        print(f"{label:26}" + "".join(f"{'--':>11}" for _ in names) + "  (no JSON)"); continue
    ck = d["checkpoint"]; ck = os.path.join(os.path.basename(os.path.dirname(ck)), os.path.basename(ck)) if ck else "?"
    print(f"{label:26}" + "".join(f"{cell(d['counts'][nm]):>11}" for nm in names) + f"  {ck}")
if train:
    print(f"\ntrain: {train.get('task')} run {train.get('run_name')} requested {train.get('iterations_requested')} it, "
          f"logged {train.get('iterations_logged')} it, rc {train.get('train_rc')}{' (TIMED OUT)' if train.get('timed_out') else ''}, "
          f"wall {int(train.get('finished_epoch', 0)) - int(train.get('started_epoch', 0))}s, "
          f"teacher load logged {train.get('teacher_checkpoint_load_logged')}, last behaviour loss {train.get('last_behavior_loss')}, "
          f"checkpoint {train.get('checkpoint')}")
for p in problems:
    print("PROBLEM:", p)

verdict = "INCOMPLETE"
if student and not problems:
    c = student["counts"]
    checks = [(f"{nm} {c[nm]} >= {thr}/32", c[nm] >= thr) for nm, thr in rule.items()]
    for text, ok in checks:
        print(f"check: {text}: {'ok' if ok else 'FAILED'}")
    verdict = "PASS" if all(ok for _, ok in checks) else "NEGATIVE"
print(f"\nSF04-DISTILL RESULT: {verdict}")
sys.exit(0 if verdict in ("PASS", "NEGATIVE") else 1)
PY
rc=$?
echo "=== done: t=${SECONDS}s | $(date) ==="
exit $rc
