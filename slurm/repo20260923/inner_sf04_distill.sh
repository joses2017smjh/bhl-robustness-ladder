#!/bin/bash
# SF-04 teacher-student distillation (docs/SENSOR_FUSION.md section 6): distil
# the working Both Full s0 MLP teacher into a recurrent student that sees the
# degraded BothRobust sensors, then evaluate the STUDENT on its own degraded
# observations. Task and cfgs: src/bhl_robust/tasks/maze_distill.py.
#
# Runs inside bhl_exec (v60). One Kit boot per invocation, so the sbatch can
# guard every boot with v60_boot_gate. Modes:
#   inner_sf04_distill.sh train <timeout_s>
#       scripts/train_distill.py (DistillationRunner) on
#       Velocity-BHL-MazeRecovery-Full-BothRobustDistill-v0, teacher weights
#       through the rsl-rl resume path: --resume True --load_run <teacher run>
#       --checkpoint model_5997.pt. On rsl-rl 5.0.1 `Distillation.load()`
#       recognises the PPO checkpoint by its actor_state_dict and strict-loads
#       it into the MLP teacher only (iteration not restored, so this run's
#       checkpoints are model_0 ... model_1999). Called directly, as
#       slurm/inner/distill.sh does, with the evidence checks of
#       slurm/inner/train.sh: Kit exits 0 whatever happened, so a run is judged
#       by its logged iterations, not its exit code.
#   inner_sf04_distill.sh eval <newer_than_epoch>
#       Resolve the newest checkpoint of run sf04-distill-s0 written after
#       <newer_than_epoch> and evaluate the student with
#       maze_recovery_probe.py --runner distillation -> student-s0.json.
#
# Exit codes are informational; the sbatch computes the verdict from the JSON.
set -euo pipefail
mode=${1:?train <timeout_s> | eval <newer_than_epoch>}
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
OUT="$REPO/results/repo-gpu-20260923/sf04-distill"; mkdir -p "$OUT"
LOG_ROOT="$UPSTREAM/logs/rsl_rl/biped"
TASK=${TASK:-Velocity-BHL-MazeRecovery-Full-BothRobustDistill-v0}
LOAD_RUN=${LOAD_RUN:-2026-09-19_16-25-50_wknd-full-both-s0}
TEACHER_CKPT_NAME=model_5997.pt
TEACHER_CKPT="$LOG_ROOT/$LOAD_RUN/$TEACHER_CKPT_NAME"
RUN_NAME=${RUN_NAME:-sf04-distill-s0}
SEED=${SEED:-0}; NUM_ENVS=${NUM_ENVS:-1024}; MAX_ITER=${MAX_ITER:-2000}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-$REPO/scripts/train_distill.py}
SETTINGS='[{"name":"baseline"},{"name":"delay1","imu_delay_steps":1},{"name":"delay2","imu_delay_steps":2},{"name":"lidar_off","zero_terms":["lidar"]},{"name":"stereo_off","zero_terms":["stereo_l","stereo_r"]},{"name":"both_off","zero_terms":["lidar","stereo_l","stereo_r"]}]'
# The student architecture comes from the distillation cfg; the recurrent PPO
# overlay must never reach train_distill.py or the probe here.
unset BHL_POLICY

case "$mode" in
train)
    tmo=${2:?timeout_s}
    [ -f "$TEACHER_CKPT" ] || { echo "train: teacher checkpoint missing: $TEACHER_CKPT" >&2; exit 3; }
    [ -f "$TRAIN_SCRIPT" ] || { echo "train: TRAIN_SCRIPT missing: $TRAIN_SCRIPT" >&2; exit 3; }
    # rsl-rl's log root is logs/rsl_rl/<experiment> relative to the cwd; the
    # teacher run is resolved under it, and this run is written beside it.
    cd "$UPSTREAM"
    ARGS=(
        --task "$TASK" --headless --enable_cameras
        --seed "$SEED" --run_name "$RUN_NAME" --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITER"
        --resume True --load_run "$LOAD_RUN" --checkpoint "$TEACHER_CKPT_NAME"
    )
    log="$OUT/train-s0.log"; rm -f "$OUT/train-s0.json"
    echo "=== train $TASK run=$RUN_NAME seed=$SEED envs=$NUM_ENVS iters=$MAX_ITER timeout=${tmo}s teacher=$LOAD_RUN/$TEACHER_CKPT_NAME | $(date) ==="
    echo "entrypoint=$TRAIN_SCRIPT args: ${ARGS[*]}"
    started=$(date +%s)
    set +e
    timeout -k 60 "$tmo" "$PY" "$TRAIN_SCRIPT" "${ARGS[@]}" > "$log" 2>&1
    rc=$?
    set -e
    [ "$rc" = 124 ] && echo "train: TIMED OUT after ${tmo}s; the newest saved checkpoint (save_interval 100) will be evaluated"
    # Evidence, as slurm/inner/train.sh collects it.
    iters=$(grep -c "Learning iteration" "$log" || true)
    # `|| true`: under set -euo pipefail an absent line (a crash before the first
    # iteration) would otherwise abort here, before train-s0.json and the
    # traceback greps below -- exactly the case they exist for.
    eplen=$(grep -oE "Mean episode length: [0-9.]+" "$log" | tail -1 | awk '{print $NF}' || true)
    teacher_loaded=$(grep -c "Loading model checkpoint from: .*${LOAD_RUN}/${TEACHER_CKPT_NAME}" "$log" || true)
    behavior=$(grep -oE "behavior[^0-9-]*[-0-9.e+]+" "$log" | tail -1 | grep -oE "[-0-9.e+]+$" || true)
    grep -hoE "^(ValueError|RuntimeError|TypeError|KeyError|AttributeError|ModuleNotFoundError|AssertionError): .{0,200}" "$log" | tail -3 || true
    grep -E "Student Model|Teacher Model|Resolved observation sets|Loading model checkpoint|\[sf04\]|\[distill\]" "$log" | head -8 || true
    ckpt=$("$PY" "$REPO/scripts/bench/maze_checkpoint.py" resolve --log-root "$LOG_ROOT" --run-name "$RUN_NAME" --newer-than "$started" 2>/dev/null) || ckpt=""
    "$PY" - "$OUT/train-s0.json" "$TASK" "$RUN_NAME" "$SEED" "$NUM_ENVS" "$MAX_ITER" "$started" "$rc" "${iters:-0}" "${eplen:-}" "${teacher_loaded:-0}" "${behavior:-}" "$ckpt" "$TEACHER_CKPT" <<'PY'
import json, sys, time
p, task, run, seed, envs, iters_req, started, rc, iters, eplen, tl, beh, ckpt, teacher = sys.argv[1:]
json.dump({"task": task, "run_name": run, "seed": int(seed), "num_envs": int(envs), "iterations_requested": int(iters_req),
           "started_epoch": int(started), "finished_epoch": int(time.time()), "train_rc": int(rc), "timed_out": rc == "124",
           "iterations_logged": int(iters), "last_mean_episode_length": float(eplen) if eplen else None,
           "teacher_checkpoint_load_logged": int(tl) > 0, "last_behavior_loss": float(beh) if beh else None,
           "checkpoint": ckpt or None, "teacher": teacher, "runner": "DistillationRunner",
           "student": "RNNModel lstm 256x1 + MLP [256,128,128] on the degraded policy group",
           "teacher_model": "MLPModel [256,128,128] on the clean teacher group (PPO actor_state_dict)"},
          open(p, "w"), indent=2)
PY
    echo "train: logged iterations=${iters:-0} last mean episode length=${eplen:-?} teacher-load line=${teacher_loaded:-0} last behavior loss=${behavior:-?}"
    echo "=== train exit $rc | newest checkpoint: ${ckpt:-NONE} | $(date) ==="
    if [ "${iters:-0}" = 0 ]; then echo "train: FAILED -- no training iteration was ever logged" >&2; exit 1; fi
    if [ -n "${eplen:-}" ] && awk -v e="$eplen" 'BEGIN{exit !(e < 2.0)}'; then
        echo "train: FAILED -- mean episode length ${eplen}: every episode terminates immediately" >&2; exit 1
    fi
    [ -n "$ckpt" ]
    ;;
eval)
    newer=${2:?newer_than_epoch}
    ckpt=$("$PY" "$REPO/scripts/bench/maze_checkpoint.py" resolve --log-root "$LOG_ROOT" --run-name "$RUN_NAME" --newer-than "$newer") \
        || { echo "eval: no checkpoint of $RUN_NAME newer than $newer (training failed or produced nothing)" >&2; exit 3; }
    json="$OUT/student-s0.json"; log="$OUT/student-s0.log"; rm -f "$json"   # the verdict may only come from this job's evaluation
    echo "=== probe STUDENT $TASK <- $ckpt | $(date) ==="
    # set +e: with --minimum-success 0 the probe still raises after writing the
    # JSON when a first episode did not finish, and that JSON is wanted.
    set +e
    timeout -k 60 1800 "$PY" "$REPO/scripts/bench/maze_recovery_probe.py" --headless --enable_cameras \
        --task "$TASK" --runner distillation --num-envs 32 --steps 600 --seed 100 \
        --checkpoint "$ckpt" --keep-corruption --output "$json" --settings "$SETTINGS" > "$log" 2>&1
    rc=$?
    set -e
    grep -E '^\{"name"|\[distill\]|\[sf04\]|SETTING-ERROR|ROLLOUT-ERROR|PROBE-ERROR|^(ValueError|RuntimeError|TypeError|KeyError|AttributeError)' "$log" | cut -c1-220 || true
    echo "=== probe exit $rc | json $([ -s "$json" ] && echo written || echo MISSING) ==="
    [ -s "$json" ]
    "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["checkpoint"]==sys.argv[2] and d["num_envs"]==32 and d["task"]==sys.argv[3], (d.get("checkpoint"), d.get("num_envs"), d.get("task"))' "$json" "$ckpt" "$TASK"
    echo "eval: wrote $json"
    ;;
*)
    echo "Unknown mode: $mode" >&2; exit 2;;
esac
