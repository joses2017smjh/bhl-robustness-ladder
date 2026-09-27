#!/bin/bash
#SBATCH --job-name=v2-stand2-train
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --exclude=dgxh-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --array=0-1
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
# CubeToShelfStand2 (2026-09-26): blind, seeds 0 and 1 (array index = seed).
# CubeToShelfStand2 is a DIFFERENT, EASIER task than CubeToShelf (cube at
# standing hand height); its numbers are never compared with CubeToShelf's,
# and it is reported as CubeToShelfStand2, not as a CubeToShelfStand v1 rerun.
# Recipe as gpu_v2_stand_train.sbatch: v60, cameras on for stack parity,
# 1024 envs, 8000 iterations. Submit afterok on gpu_v2_stand2_smoke.sbatch.
#
# What changed from v1 (TaskV2-BHL-CubeToShelfStand, job 21408514) and why:
# v1 stood (334-378-step episodes, 45% time-outs at iteration 300) and then
# traded standing for the lift bonus -- fall rate 0.49 -> 0.93 between
# iterations 325 and 1000 at no cost in return, because one lifted step paid
# 0.60 and a fall cost 0.40 once. Stand2 multiplies every shaping term by both
# robots' upright gate and prices a fall at 20 units (fall-only, so success is
# no longer charged). Diagnosis and numbers: src/bhl_robust/tasks/stand_mdp.py.
#
# Predeclared before any Stand2 run (2026-09-26; constants in stand_mdp.py):
#   kill    at model_1000: over iterations 901-1000, mean episode length
#           < 100 steps (4 s), OR mean (time_out + success) < 0.10
#           -> "cannot stand"; the run is stopped. Anything else continues to
#           8000, including a run that stands with success 0 ("stands but has
#           not learned the task yet"). v1 at its kill had length 175 but
#           time_out 0.067 -- length alone would not have stopped it.
#           This job enforces the rule itself (watcher below) and writes the
#           verdict JSON before stopping training.
#   result  Episode_Termination/success averaged over the last 200 iterations
#           >= 0.10 on at least one of the two seeds = "learned placement
#           (standing variant v2)"; else NEGATIVE. A killed seed counts as
#           not placed.
#   label   secondary, so a NEGATIVE is attributable: last-200 mean
#           (time_out + success) >= 0.50 = "stands" (reported per seed).
#   scope   CubeToShelfStand2 only; never compared with CubeToShelf numbers.
#
# Verdict lines (KILL-RULE, RESULT) are computed from the run's TensorBoard
# scalars via JSON under results/repo-gpu-20260923/stand2/, never from exit
# codes.
set -euo pipefail
export BHL_STACK=v60 ENABLE_CAMERAS=1
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
JOBTAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID:-0}"
export OV_CACHE=/scratch/$USER/ov-cache-${JOBTAG}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${JOBTAG}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH" "$WORKSPACE/overrides"
export TASK="TaskV2-BHL-CubeToShelfStand2-Blind-v0"
export SEED=${SLURM_ARRAY_TASK_ID:-${SEED:-0}} MAX_ITER=${MAX_ITER:-8000} NUM_ENVS=${NUM_ENVS:-1024}
export RUN_NAME="v2-cubetoshelfstand2-blind-s${SEED}"
export OVERRIDE_FILE="$WORKSPACE/overrides/v2-stand2-${JOBTAG}.txt"
: > "$OVERRIDE_FILE"

KILL_AT=1000
LOGROOT=$UPSTREAM/logs/rsl_rl/task_v2
OUTDIR=$REPO/results/repo-gpu-20260923/stand2
HOSTPY=$WORKSPACE/venv/bin/python          # host interpreter with tensorboard
mkdir -p "$OUTDIR"
# Same filesystem as the run dirs (NFS), so -newer is not a cross-host clock
# comparison.
STAMP=$(mktemp "$OUTDIR/.start-${JOBTAG}-XXXXXX")
trap 'rm -f "$STAMP"' EXIT
KILLED_FLAG="$OUTDIR/killed_s${SEED}_${JOBTAG}.flag"

find_run() {  # this job's run dir: newest *_RUN_NAME created after job start
    find "$LOGROOT" -maxdepth 1 -type d -name "*_${RUN_NAME}" -newer "$STAMP" 2>/dev/null | sort | tail -1 || true
}

# stand2_eval MODE RUN_DIR OUT_JSON [KILLED]: MODE kill|seed. Prints nothing;
# the JSON is the verdict. Rules come from stand_mdp.py (loaded by path: the
# bhl_robust.tasks package needs a simulator to import).
stand2_eval() {
    REPO="$REPO" OMP_NUM_THREADS=2 "$HOSTPY" - "$@" <<'PY'
import importlib.util, json, sys
mode, run, out = sys.argv[1], sys.argv[2], sys.argv[3]
killed = len(sys.argv) > 4 and sys.argv[4] == "1"
spec = importlib.util.spec_from_file_location(
    "stand_mdp", __import__("os").environ["REPO"] + "/src/bhl_robust/tasks/stand_mdp.py")
sm = importlib.util.module_from_spec(spec); spec.loader.exec_module(sm)
sc = sm.read_scalars(run)
last = max([s for s, _ in sc.get(sm.TAG_LEN, [])] or [-1])
if mode == "kill":
    res = sm.evaluate_kill(sc)
    res["ready"] = last >= sm.KILL_AT_ITER
else:
    res = sm.evaluate_seed(sc, killed=killed)
res.update({"mode": mode, "run_dir": run, "latest_iter_logged": last})
json.dump(res, open(out, "w"), indent=1, default=str)
PY
}
jget() { "$HOSTPY" -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2]))" "$1" "$2"; }

kill_tree() {  # TERM then KILL the training subtree (Isaac can ignore TERM)
    local pids p kids
    pids="$1"; kids="$1"
    while [ -n "$kids" ]; do
        kids=$(for p in $kids; do pgrep -P "$p" || true; done | tr '\n' ' ' | xargs || true)
        pids="$pids $kids"
    done
    kill -TERM $pids 2>/dev/null || true
    sleep 60
    kill -KILL $pids 2>/dev/null || true
}

watch_kill() {  # enforce the predeclared kill rule at model_1000
    local run js v i
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        run=$(find_run)
        if [ -n "$run" ] && [ -f "$run/model_${KILL_AT}.pt" ]; then
            js="$OUTDIR/kill_s${SEED}_${JOBTAG}.json"
            # the event writer flushes on its own clock: wait (<= 30 min) until
            # iteration 1000 is in the events file
            for i in $(seq 1 60); do
                stand2_eval kill "$run" "$js" || true
                [ -f "$js" ] && [ "$(jget "$js" ready)" = "True" ] && break
                sleep 30
            done
            v=$( [ -f "$js" ] && jget "$js" verdict || echo UNKNOWN)
            if [ "$(jget "$js" ready 2>/dev/null)" != "True" ]; then
                echo "KILL-RULE $RUN_NAME: UNEVALUATED -- iteration $KILL_AT never reached the events file; training continues"
                return 0
            fi
            echo "KILL-RULE $RUN_NAME @model_${KILL_AT}: $v ep_len=$(jget "$js" ep_len) time_out=$(jget "$js" time_out) success=$(jget "$js" success) fallen=$(jget "$js" fallen) reasons=$(jget "$js" reasons) ($js)"
            if [ "$v" = "KILL" ]; then
                echo "$js" > "$KILLED_FLAG"
                echo "KILL-RULE $RUN_NAME: stopping training ($(date))"
                kill_tree "$TRAIN_PID"
            fi
            return 0
        fi
        sleep 120
    done
}

echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | envs=$NUM_ENVS | HEAD $(git -C "$REPO" rev-parse --short HEAD) | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh" &
TRAIN_PID=$!
watch_kill &
WATCH_PID=$!
train_rc=0
wait "$TRAIN_PID" || train_rc=$?
# After a rule kill, let the watcher finish kill_tree's SIGKILL pass.
[ -f "$KILLED_FLAG" ] || kill "$WATCH_PID" 2>/dev/null || true
wait "$WATCH_PID" 2>/dev/null || true
echo "=== training ended rc=$train_rc: $(date) ==="

killed=0; [ -f "$KILLED_FLAG" ] && killed=1
run=$(find_run)
if [ -z "$run" ]; then
    echo "RESULT $RUN_NAME: UNEVALUATED -- no run directory *_${RUN_NAME} under $LOGROOT"
    exit 1
fi
ARRAY_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
js="$OUTDIR/result_s${SEED}_${ARRAY_ID}.json"
if stand2_eval seed "$run" "$js" "$killed" && [ -f "$js" ]; then
    echo "RESULT $RUN_NAME: placed=$(jget "$js" placed) stands=$(jget "$js" stands) killed=$(jget "$js" killed)" \
         "success_last200=$(jget "$js" success) time_out_last200=$(jget "$js" time_out)" \
         "last_iter=$(jget "$js" last_iter) ($js)"
else
    echo "RESULT $RUN_NAME: UNEVALUATED -- could not read $run"
    exit 1
fi

# Both seeds of THIS array in: the predeclared two-seed verdict (whichever task
# finishes last prints it). Results are tagged by array id, so a resubmitted
# seed is never pooled with an earlier attempt of the other.
if [ -f "$OUTDIR/result_s0_${ARRAY_ID}.json" ] && [ -f "$OUTDIR/result_s1_${ARRAY_ID}.json" ]; then
    REPO="$REPO" OMP_NUM_THREADS=2 "$HOSTPY" - "$OUTDIR" "$ARRAY_ID" <<'PY'
import importlib.util, json, os, sys
d, aid = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location(
    "stand_mdp", os.environ["REPO"] + "/src/bhl_robust/tasks/stand_mdp.py")
sm = importlib.util.module_from_spec(spec); spec.loader.exec_module(sm)
seeds = [json.load(open(os.path.join(d, f"result_s{s}_{aid}.json"))) for s in (0, 1)]
verdict = sm.pair_result(seeds)
json.dump({"verdict": verdict, "seeds": seeds, "task": "CubeToShelfStand2 (blind)",
           "note": "different, easier task than CubeToShelf; never compared with it"},
          open(os.path.join(d, f"result_pair_{aid}.json"), "w"), indent=1)
print(f"RESULT CubeToShelfStand2 blind (2 seeds): {verdict}")
PY
fi
# A run stopped by its predeclared kill rule is an outcome, not a job failure.
[ "$killed" = 1 ] && exit 0
exit "$train_rc"
