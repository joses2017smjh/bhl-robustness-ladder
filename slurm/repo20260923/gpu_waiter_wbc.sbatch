#!/bin/bash
#SBATCH --job-name=waiter-wbc
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
# Waiter program phase 1 (docs/WAITER_PROGRAM.md; frozen 2026-10-05, SLURM_JOBS.md "Predeclared now ... the Waiter
# program"): train Velocity-BHL-Waiter-WBC-v0 FROM SCRATCH (seeds 0-2 = array index, 6000 iterations, the task's env
# count, R1's DR overrides at scale 1.0), export (train_play.py via export_one_humanoid.sh, under the shared export
# lock), check the export (24 joints, 12 leg actions, 83 actor observations in the ONNX) and stamp the gait_clock +
# waiter_wbc blocks (bhl_robust.eval.waiter_wbc stamp). The MuJoCo gates Q1 / Q2 run in a separate CPU job.
# Smoke: WAITER_SMOKE=1 (3 iterations, 64 envs, outputs under a directory ending in -smoke): the same steps.
# usage: S=$(slurm_clean sbatch --parsable --array=0 --export=ALL,WAITER_SMOKE=1 slurm/repo20260923/gpu_waiter_wbc.sbatch) && \
#        slurm_clean sbatch --dependency=afterok:${S%%;*} --export=ALL,WAITER_SMOKE=0 slurm/repo20260923/gpu_waiter_wbc.sbatch
# WAITER_VARIANT=1c (2026-10-07; SLURM_JOBS.md 'Predeclared now ... Waiter phase 1c'): task
# Velocity-BHL-Waiter-WBC-Curriculum-v0 (= 1b + the disturbance curriculum: arm goals, hand forces and hand payload
# scaled 0 before iteration 4000, linear to 1 at 7000, then 1), seeds 6-8, 10000 iterations, std <= 1.0; submit with
# --time=16:00:00. Its smoke compresses the ramp to iterations 0-2 (Hydra override) so the log shows the term move;
# its full run must log Curriculum/disturbance 0 first and 1.0 last, with the frozen ramp values in params/env.yaml.
set -euo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
PYH=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
LOGROOT=$UPSTREAM/logs/rsl_rl/humanoid
JOBTAG=${SLURM_JOB_ID:-local$$}
SMOKE=${WAITER_SMOKE:-0}
TASK_ID=Velocity-BHL-Waiter-WBC-v0
VARIANT=${WAITER_VARIANT:-1}         # 1 = phase 1 as frozen; 1b = + action-noise std bounded to <= 1.0, seeds 3-5
FULL_ITERS=6000
case "$VARIANT" in
    1)  PREFIX=waiter-wbc;   SEED0=0; RES_NAME=wbc ;;
    1b) PREFIX=waiter-wbc1b; SEED0=3; RES_NAME=wbc1b; export BHL_STD_MAX=1.0 ;;
    1c) PREFIX=waiter-wbc1c; SEED0=6; RES_NAME=wbc1c; export BHL_STD_MAX=1.0
        TASK_ID=Velocity-BHL-Waiter-WBC-Curriculum-v0; FULL_ITERS=10000 ;;
    *)  echo "WAITER-WBC RESULT: FAIL (unknown WAITER_VARIANT $VARIANT)"; exit 2 ;;
esac
OBS_POLICY=83
if [ "$SMOKE" = 1 ]; then
    RES=${WAITER_OUT_DIR:-/nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/campaign-waiter/$RES_NAME-smoke}
    case "$RES" in *-smoke) ;; *) echo "SMOKE: FAIL (smoke outputs must go to a directory ending in -smoke)"; exit 1;; esac
else
    RES=$REPO/results/waiter-20261005/$RES_NAME
fi
IDX=${SLURM_ARRAY_TASK_ID:-0}
case $IDX in 0|1|2) SEED=$((SEED0 + IDX));; *) echo "WAITER-WBC RESULT: FAIL (array index $IDX outside 0-2)"; exit 2;; esac
if [ "$SMOKE" = 1 ]; then
    export MAX_ITER=3 NUM_ENVS=64 RUN_NAME="$PREFIX-s${SEED}-smoke-$JOBTAG"
else
    export MAX_ITER=$FULL_ITERS RUN_NAME="$PREFIX-s${SEED}"
    unset NUM_ENVS
fi
FINAL_CKPT="model_$((MAX_ITER - 1)).pt"
export TASK="$TASK_ID" SEED
mkdir -p "$RES"/training
unset BHL_POLICY BHL_RND BHL_SYMMETRY
[ "$VARIANT" = 1 ] && unset BHL_STD_MAX
source "$REPO/slurm/dr_levels.sh"
setup_node_cache
source "$REPO/slurm/inner/v60_boot_gate.sh"
V60_BOOT_LOCK="$WORKSPACE/.v51-boot.lock" v60_boot_gate
EXPORT_LOCK="$WORKSPACE/.turngait-r12-export.lock"      # the shared lock: export_one_humanoid.sh rewrites configs/policy_latest.yaml
_release() { [ "$(cut -d' ' -f1 "$EXPORT_LOCK/holder" 2>/dev/null)" = "$JOBTAG" ] && rm -rf "$EXPORT_LOCK"; return 0; }
trap "_v60_release '$WORKSPACE/.v51-boot.lock'; _release" EXIT
export_lock() {
    local waited=0
    while ! mkdir "$EXPORT_LOCK" 2>/dev/null; do
        if [ -n "$(find "$EXPORT_LOCK" -maxdepth 0 -mmin +20 2>/dev/null)" ]; then rm -rf "$EXPORT_LOCK"; continue; fi
        sleep $(( 10 + RANDOM % 10 )); waited=$(( waited + 15 ))
        [ $waited -gt 3600 ] && return 1
    done
    echo "$JOBTAG $(hostname) $(date +%s)" > "$EXPORT_LOCK/holder"
}
if [ "$SMOKE" = 1 ]; then OVR_DIR="$RES/overrides"; else OVR_DIR="$WORKSPACE/overrides"; fi
mkdir -p "$OVR_DIR"
export OVERRIDE_FILE="$OVR_DIR/${RUN_NAME}.txt"
dr_overrides_scale 1.0 > "$OVERRIDE_FILE"
if [ "$VARIANT" = 1c ] && [ "$SMOKE" = 1 ]; then
    # smoke only: compress the ramp into the smoke's 3 iterations, so its log shows the term move
    printf ' env.curriculum.disturbance.params.start_iter=0 env.curriculum.disturbance.params.end_iter=2' >> "$OVERRIDE_FILE"
fi
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS RUN_DIR BHL_STD_MAX"
find_run() { ls -d "$LOGROOT"/*_"$RUN_NAME" 2>/dev/null | sort | tail -1; }
echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | smoke=$SMOKE | from scratch, $MAX_ITER iters -> $FINAL_CKPT | RES $RES | HEAD $(git -C "$REPO" rev-parse --short HEAD) | $(date) ==="
git -C "$REPO" status --porcelain -- src/bhl_robust/waiter_asset.py src/bhl_robust/tasks/waiter_env_cfg.py \
    src/bhl_robust/tasks/waiter_wbc_mdp.py src/bhl_robust/eval/waiter_wbc.py src/bhl_robust/tasks/__init__.py "$0" || true
TLOG="${TMPDIR:-/tmp}/waiter-wbc-$JOBTAG.log"; : > "$TLOG"
existing=$(find_run || true)
if [ "$SMOKE" != 1 ] && [ -n "$existing" ] && [ -f "$existing/$FINAL_CKPT" ]; then
    echo "already trained: $existing (reused)"
else
    bhl_exec "$REPO/slurm/inner/train.sh" 2>&1 | tee "$TLOG" || true
    echo "=== training done: $(date) ==="
fi
run=$(find_run || true)
why=""
[ -n "$run" ] && [ -f "$run/$FINAL_CKPT" ] || why="$why no-$FINAL_CKPT"
grep -q "waiter-wbc id NOT registered" "$TLOG" && why="$why task-not-registered"
if { [ "$VARIANT" = 1b ] || [ "$VARIANT" = 1c ]; } && [ "$(ls -d "$LOGROOT"/*_"$RUN_NAME" 2>/dev/null | wc -l)" -ge 1 ] && [ -s "$TLOG" ]; then
    grep -q "\[overlay\] action-noise std bounded to \[0.001, 1.0\]" "$TLOG" || why="$why std-bound-overlay-missing"
    "$PYH" - "$TLOG" <<'PY' || why="$why noise-std-above-1.0"
import re, sys
v = [float(x) for x in re.findall(r"Mean action noise std:\s*([0-9.]+)", open(sys.argv[1], errors="ignore").read())]
sys.exit(0 if v and max(v) <= 1.0 + 1e-6 else 1)
PY
fi
if [ "$VARIANT" = 1c ] && [ -n "$run" ] && [ -s "$TLOG" ]; then
    # phase 1c: the curriculum ran (logged 0 first, and 1.0 last in a full run), the PPO rollout is the 24 steps the
    # iteration count assumes, and a full run carries the frozen ramp values
    (cd "$REPO" && PYTHONPATH="$REPO/src" "$PYH" - "$TLOG" "$run" "$SMOKE" <<'PY') || why="$why curriculum-check"
import re, sys
from pathlib import Path
from bhl_robust.eval.gait_clock import load_env_yaml
tlog, run, smoke = sys.argv[1], Path(sys.argv[2]), sys.argv[3] == "1"
v = [float(x) for x in re.findall(r"Curriculum/disturbance:\s*([0-9.]+)", open(tlog, errors="ignore").read())]
agent = (run / "params" / "agent.yaml").read_text()
ok = bool(v) and v[0] == 0.0 and re.search(r"^num_steps_per_env: 24$", agent, re.M) is not None
if smoke:
    ok = ok and max(v) > 0.0
else:
    p = load_env_yaml(run / "params" / "env.yaml")["curriculum"]["disturbance"]["params"]
    ok = ok and v[-1] >= 0.999 and (p["start_iter"], p["end_iter"], p["steps_per_iter"]) == (4000, 7000, 24) \
        and list(p["force_range"]) == [-6.0, 6.0] and list(p["payload_range"]) == [0.0, 0.8]
print(f"curriculum check: {'OK' if ok else 'FAIL'} (logged {len(v)} values, first {v[0] if v else None}, "
      f"last {v[-1] if v else None}, max {max(v) if v else None})")
sys.exit(0 if ok else 1)
PY
fi
if [ -z "$why" ]; then
    "$PYH" - "$RES/training/$RUN_NAME.json" "$RUN_NAME" "$TASK_ID" "$SEED" "$FINAL_CKPT" "$run" "$JOBTAG" "$TLOG" <<'PY'
import json, re, sys
out, run_name, task, seed, ckpt, run, job, tlog = sys.argv[1:]
log = open(tlog, errors="ignore").read()
its = [int(m) for m in re.findall(r"Learning iteration (\d+)/", log)]
json.dump({"run": run_name, "task": task, "seed": int(seed), "final_ckpt": ckpt, "run_dir": run, "job": job,
           "logged_iterations": (max(its) + 1) if its else None}, open(out, "w"), indent=1)
PY
    cp "$TLOG" "$RES/training/$RUN_NAME.$JOBTAG.log" 2>/dev/null || true
    export RUN_DIR=$(basename "$run")
    cfg="$run/exported/deploy.yaml"
    if [ ! -f "$cfg" ]; then
        if export_lock; then
            echo "=== export $RUN_DIR $(date) ==="
            TASK="$TASK_ID" bhl_exec "$REPO/slurm/inner/export_one_humanoid.sh" 2>&1 | grep -vE "pthread_setaffinity|EGLError" || true
            _release
        else why="$why export-lock"; fi
    fi
    if [ -f "$cfg" ] && [ -f "$run/exported/policy.onnx" ]; then
        w=$("$PYH" -c "import onnxruntime as o, sys; s=o.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider']); print(s.get_inputs()[0].shape[-1], s.get_outputs()[0].shape[-1])" "$run/exported/policy.onnx" 2>/dev/null || echo "?")
        [ "$w" = "$OBS_POLICY 12" ] || why="$why onnx-io-$w"
        s=$( (cd "$REPO" && PYTHONPATH="$REPO/src" "$PYH" -m bhl_robust.eval.waiter_wbc stamp --deploy "$cfg" --env-yaml "$run/params/env.yaml" --task "$TASK_ID" 2>&1 | tail -1) || true)
        echo "$s"
        case "$s" in "WAITER-WBC: stamped"*|"WAITER-WBC: already stamped"*) ;; *) why="$why stamp";; esac
        cp "$cfg" "$RES/training/$RUN_NAME.deploy.yaml"
    else
        why="$why no-deploy.yaml-or-policy.onnx"
    fi
fi
if [ -n "$why" ]; then
    echo "WAITER-WBC RESULT: FAIL ($RUN_NAME:$why)"
    [ "$SMOKE" = 1 ] && echo "WAITER-WBC SMOKE: FAIL"
    exit 1
fi
echo "WAITER-WBC RESULT: TRAINED + EXPORTED ($RUN_NAME, $run)"
[ "$SMOKE" = 1 ] && echo "WAITER-WBC SMOKE: PASS ($RUN_NAME: $MAX_ITER iterations, export 83 -> 12, stamped)"
echo "=== done $(date) ==="
