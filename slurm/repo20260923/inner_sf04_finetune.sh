#!/bin/bash
# SF-04 follow-up (docs/SENSOR_FUSION.md section 6): isolate the BothRobust
# ingredients one at a time by FINE-TUNING the working MLP `Both` Full s0
# checkpoint on one robustified Full-stage task, instead of training a
# recurrent policy from scratch (which fell in 100 % of Corridor episodes).
#
# Runs inside bhl_exec (v60). One Kit boot per invocation, so the sbatch can
# guard every boot with v60_boot_gate. Modes:
#   inner_sf04_finetune.sh control [<timeout_s>]
#       Evaluate the untouched teacher (Both Full s0 model_5997.pt) on the plain
#       Both task with the settings matrix -> control-both-s0.json. Skipped when
#       that file already exists: it is the control every arm is compared to.
#   inner_sf04_finetune.sh train <Arm> <iterations> <timeout_s>
#       slurm/inner/train.sh on Velocity-BHL-MazeRecovery-Full-<Arm>-v0, resumed
#       from the teacher via OVERRIDE_FILE (agent.resume / load_run /
#       load_checkpoint hydra keys -- the same keys weekend_maze.sh writes and
#       the 21403141 log shows resolving on this stack). NO BHL_POLICY overlay:
#       the checkpoint is an MLP [256,128,128].
#   inner_sf04_finetune.sh eval <Arm> <newer_than_epoch> [<timeout_s>]
#       Resolve the newest checkpoint of run sf04-ft-<arm>-s0 written after
#       <newer_than_epoch> and evaluate it -> <Arm>-s0.json.
# The probe timeout is min(1800, <timeout_s>): the sbatch passes the wall time
# it has left so a hung Kit is killed before Slurm kills the job and the verdict
# line still prints.
#
# Exit codes are informational only; the sbatch computes the verdict from the
# JSONs. rsl-rl resumes at the loaded iteration (5997), so the fine-tune run's
# checkpoints are numbered model_6xxx.pt and count only the new iterations.
set -euo pipefail
mode=${1:?control | train <Arm> <iterations> <timeout_s> | eval <Arm> <newer_than_epoch>}
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
OUT="$REPO/results/repo-gpu-20260923/sf04-finetune"; mkdir -p "$OUT"
LOG_ROOT="$UPSTREAM/logs/rsl_rl/biped"
TEACHER_RUN=2026-09-19_16-25-50_wknd-full-both-s0
TEACHER_CKPT="$LOG_ROOT/$TEACHER_RUN/model_5997.pt"
SETTINGS='[{"name":"baseline"},{"name":"delay1","imu_delay_steps":1},{"name":"delay2","imu_delay_steps":2},{"name":"lidar_off","zero_terms":["lidar"]},{"name":"stereo_off","zero_terms":["stereo_l","stereo_r"]},{"name":"both_off","zero_terms":["lidar","stereo_l","stereo_r"]},{"name":"gyro0.10","gyro_std":0.10}]'
# The teacher and every fine-tuned arm are MLP actors; the recurrent overlay
# (forwarded through BHL_FORWARD_VARS if the submit shell had it) would make the
# runner an LSTM and the state_dict load fail before a single iteration. The
# RND / symmetry overlays are the same class of forwarded hazard.
unset BHL_POLICY BHL_RND BHL_SYMMETRY

check_arm() { case "$1" in BothDelay|BothDrop|BothBias|BothRobust) ;; *) echo "Invalid arm: $1 (BothDelay|BothDrop|BothBias|BothRobust)" >&2; exit 2;; esac; }
# min(1800, arg), floored at 60: `timeout 0` would mean no timeout at all.
probe_timeout() { local t=${1:-1800}; case "$t" in ''|*[!0-9]*) t=1800;; esac; [ "$t" -gt 1800 ] && t=1800; [ "$t" -lt 60 ] && t=60; echo "$t"; }

# probe <task> <checkpoint> <output.json> <log> <timeout_s>: one boot, seven settings.
# set +e around it: with --minimum-success 0 the probe still raises after
# writing the JSON when a first episode did not finish, and that JSON is wanted.
probe() {
    local task=$1 ckpt=$2 json=$3 log=$4 tmo=$5 rc
    echo "=== probe $task <- $ckpt | timeout ${tmo}s | $(date) ==="
    set +e
    timeout -k 60 "$tmo" "$PY" "$REPO/scripts/bench/maze_recovery_probe.py" --headless --enable_cameras \
        --task "$task" --num-envs 32 --steps 600 --seed 100 \
        --checkpoint "$ckpt" --keep-corruption --output "$json" --settings "$SETTINGS" > "$log" 2>&1
    rc=$?
    set -e
    grep -E '^\{"name"|SETTING-ERROR|ROLLOUT-ERROR|^(ValueError|RuntimeError|TypeError|KeyError|AttributeError)' "$log" | cut -c1-220 || true
    echo "=== probe exit $rc | json $([ -s "$json" ] && echo written || echo MISSING) ==="
    [ -s "$json" ]
}

case "$mode" in
control)
    final="$OUT/control-both-s0.json"
    if [ -s "$final" ]; then
        echo "control: $final exists, keeping it (delete it to re-run the control)"; exit 0
    fi
    [ -f "$TEACHER_CKPT" ] || { echo "control: teacher checkpoint missing: $TEACHER_CKPT" >&2; exit 3; }
    pending="$OUT/control-both-s0.pending.json"; rm -f "$pending"
    probe Velocity-BHL-MazeRecovery-Full-Both-v0 "$TEACHER_CKPT" "$pending" "$OUT/control-both-s0.log" "$(probe_timeout "${2:-}")"
    "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["checkpoint"]==sys.argv[2] and d["num_envs"]==32, d.get("checkpoint")' "$pending" "$TEACHER_CKPT"
    mv -f "$pending" "$final"
    echo "control: wrote $final"
    ;;
train)
    arm=${2:?Arm}; iters=${3:?iterations}; tmo=${4:?timeout_s}; check_arm "$arm"
    [ -f "$TEACHER_CKPT" ] || { echo "train: teacher checkpoint missing: $TEACHER_CKPT" >&2; exit 3; }
    export TASK="Velocity-BHL-MazeRecovery-Full-${arm}-v0" RUN_NAME="sf04-ft-${arm,,}-s0"
    export SEED=0 NUM_ENVS=1024 MAX_ITER=$iters ENABLE_CAMERAS=1
    export OVERRIDE_FILE="$OUT/${arm}-s0.overrides.txt"
    printf 'agent.resume=true\nagent.load_run=%s\nagent.load_checkpoint=model_5997.pt\n' "$TEACHER_RUN" > "$OVERRIDE_FILE"
    rm -f "$OUT/${arm}-s0.train.json"
    echo "=== train $TASK run=$RUN_NAME iters=$iters timeout=${tmo}s from $TEACHER_CKPT | $(date) ==="
    started=$(date +%s)
    set +e
    timeout -k 60 "$tmo" bash "$REPO/slurm/inner/train.sh"
    rc=$?
    set -e
    [ "$rc" = 124 ] && echo "train: TIMED OUT after ${tmo}s; the newest saved checkpoint (save_interval 100) will be evaluated"
    ckpt=$("$PY" "$REPO/scripts/bench/maze_checkpoint.py" resolve --log-root "$LOG_ROOT" --run-name "$RUN_NAME" --newer-than "$started" 2>/dev/null) || ckpt=""
    "$PY" - "$OUT/${arm}-s0.train.json" "$arm" "$TASK" "$RUN_NAME" "$iters" "$started" "$rc" "$ckpt" "$TEACHER_CKPT" <<'PY'
import json, sys, time
p, arm, task, run, iters, started, rc, ckpt, teacher = sys.argv[1:]
json.dump({"arm": arm, "task": task, "run_name": run, "iterations_requested": int(iters), "seed": 0, "num_envs": 1024,
           "started_epoch": int(started), "finished_epoch": int(time.time()), "train_rc": int(rc), "timed_out": rc == "124",
           "checkpoint": ckpt or None, "resumed_from": teacher, "policy": "MLP (no BHL_POLICY overlay)"}, open(p, "w"), indent=2)
PY
    echo "=== train exit $rc | newest checkpoint: ${ckpt:-NONE} | $(date) ==="
    [ -n "$ckpt" ]
    ;;
eval)
    arm=${2:?Arm}; newer=${3:?newer_than_epoch}; check_arm "$arm"; tmo=$(probe_timeout "${4:-}")
    RUN_NAME="sf04-ft-${arm,,}-s0"
    ckpt=$("$PY" "$REPO/scripts/bench/maze_checkpoint.py" resolve --log-root "$LOG_ROOT" --run-name "$RUN_NAME" --newer-than "$newer") \
        || { echo "eval: no checkpoint of $RUN_NAME newer than $newer (training failed or produced nothing)" >&2; exit 3; }
    json="$OUT/${arm}-s0.json"; rm -f "$json"   # the verdict may only come from this job's evaluation
    probe "Velocity-BHL-MazeRecovery-Full-${arm}-v0" "$ckpt" "$json" "$OUT/${arm}-s0.log" "$tmo"
    "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["checkpoint"]==sys.argv[2] and d["num_envs"]==32, d.get("checkpoint")' "$json" "$ckpt"
    echo "eval: wrote $json"
    ;;
*)
    echo "Unknown mode: $mode" >&2; exit 2;;
esac
