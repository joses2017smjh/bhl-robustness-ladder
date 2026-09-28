#!/bin/bash
#SBATCH --job-name=v2-stand3-train
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
# CubeToShelfStand3 "side deck" (2026-09-27): blind, seeds 0 and 1 (array
# index = seed). CubeToShelfStand3 is a DIFFERENT, EASIER task than CubeToShelf
# AND than CubeToShelfStand2: the shelf 1.2 m away is replaced by a deck on
# each side of the plinth and placement is lift, shift 0.19 m, lower. Its
# numbers are never compared with CubeToShelf's or Stand2's. Recipe as
# gpu_v2_stand2_train.sbatch: v60, cameras on for stack parity, 1024 envs,
# 8000 iterations. Submit afterok on gpu_v2_stand3_smoke.sbatch.
#
# Why (diagnosis of Stand2, job 21434946, from its TensorBoard; full numbers in
# src/bhl_robust/tasks/stand_mdp.py, "Stand3"): s0 stood (time_out 0.64) and
# lifted (lift curriculum to +0.19 m) but left the cube over the plinth (mean x
# ~+0.03 m) with the shelf 1.2 m away; s1 diverged at iteration 4468. Loss/value
# led every runaway (s0 4345-4358: 90 -> inf while episode action_rate stayed
# normal; s1 4466-4468: 1.6e16 -> 3.5e34 -> NaN); episode action_rate spikes
# came 0-14 iterations later, when the runaway episodes ended (s0 4359-4360:
# -2.9e18 / -2.8e12), and s1 crashed before any did. The learning rate was at
# its 1e-5 floor throughout. Consistent with (not proof of) an actor-output
# runaway fed back through the unclipped last-action observation; policy std
# rose 1.0 -> 2.0/2.1 under entropy_coef 0.005.
# Changes (Stand3 only; Stand2 and _V2_RUNNER untouched): decks at |x| 0.17-0.47
# with a 2 cm lip over the plinth (it stops a slide; a push can still tip the
# cube over it, so a success means "seated on the deck", mechanism not asserted;
# Curriculum/cube_tilted and Curriculum/tipped_over_deck record which); success
# = cube seated on a deck, still, 12 steps (release not checked: the hands may
# still be on it), with `placed` reading the success termination at weight 5000
# = 200 units, 2.04x the 98-unit discounted bound on the richest hover (24.5
# weight/step, lifted over a deck centre); Stand2's lift terms kept, NOT gated
# by position, so the shift costs nothing; deck progress replaces carry;
# object_xy removed; lift curriculum capped at +0.06; every observation clipped
# (last action +/-10, rest +/-100), action_rate on clipped actions, joint
# targets clipped to +/-3 rad (no joint limit exceeds 2.44); runner
# TaskV2Stand3PPORunnerCfg (log std, entropy_coef 0.001).
# Stated up front: measured feet-planted reach (MuJoCo FK) shifts the hand
# midpoint 0.044 m (arms) / 0.119 m (arms + 0.5 rad pelvis twist); the required
# shift is 0.19 m, so ~0.07 m must come from lateral lean or a shuffle. The
# deck is NOT within arm reach; it is 6x closer than Stand2's shelf.
#
# Predeclared before any Stand3 run (2026-09-27; constants in stand_mdp.py):
#   kill      Stand2's rule: at model_1000, over iterations 901-1000,
#             mean episode length < 100 steps (4 s), OR mean
#             (time_out + success) < 0.10 -> "cannot stand"; training is
#             stopped by the watcher below. A killed seed is decided, not placed.
#             Amended 2026-09-27 on review, before any Stand3 run
#             (stand_mdp.kill_verdict3): the length clause is skipped when the
#             mean success over 901-1000 is >= 0.10, because success ends the
#             episode and a quick placement is short -- a placing seed is never
#             killed for short episodes.
#   complete  a seed is COMPLETE only if its events reach iteration 7999
#             (8000 iterations) AND every Loss/value, success and time_out value
#             in its last 200 iterations is finite. Isaac exits 0 after a crash,
#             so exit codes are never read. A seed that is neither complete nor
#             killed is INCOMPLETE and is never a result.
#   result    Episode_Termination/success (cube seated on a deck, held 12 steps)
#             averaged over the last 200 iterations >= 0.10 on at least one of
#             the two seeds, counted on COMPLETE seeds only
#             = "learned placement (side-deck variant v3)".
#             NEGATIVE only if both seeds are decided (complete or killed) and
#             none placed. Otherwise INCOMPLETE (no verdict); no seed is added
#             or rerun after seeing a result.
#   labels    secondary, so a NEGATIVE is attributable, per complete seed:
#             "stands" = last-200 mean (time_out + success) >= 0.50;
#             "shifts" = last-200 mean Curriculum/over_deck >= 0.10 (cube over a
#             deck, off the floor, 10% of the time);
#             "stable" = no iteration with Loss/value > 1000 or non-finite
#             (Stand2 had 15 and 6; the 200-unit bonus cannot reach 1000 on
#             its own: one surprise is ~4e4 squared error over ~25k samples).
#   scope     CubeToShelfStand3 only; never compared with CubeToShelf or
#             CubeToShelfStand2 numbers. Data looked at for this design:
#             Stand2 21434946 s0/s1 and v1 21408514 only; both Stand3 seeds are
#             fresh runs.
#
# Also checked at start (RUNNER-CHECK): the run's params/agent.yaml must show
# entropy_coef 0.001 and std_type log, and params/env.yaml action_rate_clipped
# and a `placed` reward of stand_mdp:success_bonus at weight 5000.0;
# otherwise the task was registered with the wrong runner or cfg, training is
# stopped and the seed is INCOMPLETE.
#
# Verdict lines (RUNNER-CHECK, KILL-RULE, RESULT) come from JSON written under
# results/repo-gpu-20260923/stand3_2026-09-27/, never from exit codes.
set -euo pipefail
export BHL_STACK=v60 ENABLE_CAMERAS=1
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
JOBTAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID:-0}"
export OV_CACHE=/scratch/$USER/ov-cache-${JOBTAG}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${JOBTAG}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH" "$WORKSPACE/overrides"
export TASK="TaskV2-BHL-CubeToShelfStand3-Blind-v0"
# 8000 is the predeclared budget. MAX_ITER can be lowered for a dry run, but the
# completion check below always demands iteration 7999, so a shorter run can
# never be read as a result.
export SEED=${SLURM_ARRAY_TASK_ID:-${SEED:-0}} MAX_ITER=${MAX_ITER:-8000} NUM_ENVS=${NUM_ENVS:-1024}
export RUN_NAME="v2-cubetoshelfstand3-blind-s${SEED}"
export OVERRIDE_FILE="$WORKSPACE/overrides/v2-stand3-${JOBTAG}.txt"
: > "$OVERRIDE_FILE"

KILL_AT=1000
LOGROOT=$UPSTREAM/logs/rsl_rl/task_v2
OUTDIR=$REPO/results/repo-gpu-20260923/stand3_2026-09-27
HOSTPY=$WORKSPACE/venv/bin/python          # host interpreter with tensorboard
mkdir -p "$OUTDIR"
# Same filesystem as the run dirs (NFS), so -newer is not a cross-host clock
# comparison.
STAMP=$(mktemp "$OUTDIR/.start-${JOBTAG}-XXXXXX")
trap 'rm -f "$STAMP"' EXIT
KILLED_FLAG="$OUTDIR/killed_s${SEED}_${JOBTAG}.flag"
BADCFG_FLAG="$OUTDIR/badcfg_s${SEED}_${JOBTAG}.flag"

find_run() {  # this job's run dir: newest *_RUN_NAME created after job start
    find "$LOGROOT" -maxdepth 1 -type d -name "*_${RUN_NAME}" -newer "$STAMP" 2>/dev/null | sort | tail -1 || true
}

# stand3_eval MODE RUN_DIR OUT_JSON [KILLED]: MODE kill|seed. Prints nothing;
# the JSON is the verdict. Rules come from stand_mdp.py (loaded by path: the
# bhl_robust.tasks package needs a simulator to import).
stand3_eval() {
    REPO="$REPO" OMP_NUM_THREADS=2 "$HOSTPY" - "$@" <<'PY'
import importlib.util, json, os, sys
mode, run, out = sys.argv[1], sys.argv[2], sys.argv[3]
killed = len(sys.argv) > 4 and sys.argv[4] == "1"
spec = importlib.util.spec_from_file_location(
    "stand_mdp", os.environ["REPO"] + "/src/bhl_robust/tasks/stand_mdp.py")
sm = importlib.util.module_from_spec(spec); spec.loader.exec_module(sm)
sc = sm.read_scalars(run)
last = max([s for s, _ in sc.get(sm.TAG_LEN, [])] or [-1])
if mode == "kill":
    res = sm.evaluate_kill3(sc)
    res["ready"] = last >= sm.KILL_AT_ITER
else:
    res = sm.evaluate_seed3(sc, killed=killed)
res.update({"mode": mode, "run_dir": run, "latest_iter_logged": last,
            "task": "CubeToShelfStand3 (blind)", "note": sm.TASK3_NOTE})
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

placed_ok() {  # rewards.placed block: stand_mdp:success_bonus, weight 5000.0
    awk '/^rewards:/{r=1; next} r && /^[a-z]/{r=0} r && /^  [a-z_]+:$/{b=($1=="placed:")}
         r && b && /^    func: bhl_robust\.tasks\.stand_mdp:success_bonus$/{f=1}
         r && b && /^    weight: 5000\.0$/{w=1} END{exit !(f && w)}' "$1" 2>/dev/null
}

check_runner() {  # the run trains with Stand3's runner and cfg, or it stops
    local run="$1" i
    for i in $(seq 1 60); do
        [ -f "$run/params/agent.yaml" ] && [ -f "$run/params/env.yaml" ] && break
        sleep 30
    done
    if grep -qE '^ *entropy_coef: 0\.001$' "$run/params/agent.yaml" 2>/dev/null \
       && grep -qE '^ *std_type: log$' "$run/params/agent.yaml" 2>/dev/null \
       && grep -qE '^ *action_rate_clipped:' "$run/params/env.yaml" 2>/dev/null \
       && placed_ok "$run/params/env.yaml"; then
        echo "RUNNER-CHECK $RUN_NAME: PASS (entropy_coef 0.001, std_type log, action_rate_clipped present, placed = success_bonus x 5000.0)"
        return 0
    fi
    echo "RUNNER-CHECK $RUN_NAME: FAIL -- params/agent.yaml or env.yaml is not Stand3's; stopping ($(date))"
    echo "$run" > "$BADCFG_FLAG"
    return 1
}

watch_kill() {  # runner check, then the predeclared kill rule at model_1000
    local run js v i checked=0
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        run=$(find_run)
        if [ -n "$run" ] && [ "$checked" = 0 ]; then
            checked=1
            if ! check_runner "$run"; then kill_tree "$TRAIN_PID"; return 0; fi
        fi
        if [ -n "$run" ] && [ -f "$run/model_${KILL_AT}.pt" ]; then
            js="$OUTDIR/kill_s${SEED}_${JOBTAG}.json"
            # the event writer flushes on its own clock: wait (<= 30 min) until
            # iteration 1000 is in the events file
            for i in $(seq 1 60); do
                stand3_eval kill "$run" "$js" || true
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

echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | envs=$NUM_ENVS | iters=$MAX_ITER | HEAD $(git -C "$REPO" rev-parse --short HEAD) | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh" &
TRAIN_PID=$!
watch_kill &
WATCH_PID=$!
train_rc=0
wait "$TRAIN_PID" || train_rc=$?
# After a rule kill or a failed runner check, let the watcher finish kill_tree.
if [ -f "$KILLED_FLAG" ] || [ -f "$BADCFG_FLAG" ]; then :; else kill "$WATCH_PID" 2>/dev/null || true; fi
wait "$WATCH_PID" 2>/dev/null || true
echo "=== training ended rc=$train_rc: $(date) ==="

killed=0; [ -f "$KILLED_FLAG" ] && killed=1
run=$(find_run)
if [ -z "$run" ]; then
    echo "RESULT $RUN_NAME: INCOMPLETE -- no run directory *_${RUN_NAME} under $LOGROOT"
    exit 1
fi
ARRAY_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
js="$OUTDIR/result_s${SEED}_${ARRAY_ID}.json"
if stand3_eval seed "$run" "$js" "$killed" && [ -f "$js" ]; then
    if [ -f "$BADCFG_FLAG" ]; then
        # wrong runner/cfg: never a result, whatever the scalars say
        "$HOSTPY" -c "import json,sys; d=json.load(open(sys.argv[1])); d.update(complete=False, decided=False, placed=False, stands=False, shifts=False, bad_config=True); json.dump(d, open(sys.argv[1],'w'), indent=1)" "$js"
    fi
    status=$( [ "$(jget "$js" decided)" = "True" ] && echo DECIDED || echo "INCOMPLETE (not a result)")
    echo "RESULT $RUN_NAME: $status complete=$(jget "$js" complete) placed=$(jget "$js" placed) stands=$(jget "$js" stands)" \
         "shifts=$(jget "$js" shifts) stable=$(jget "$js" stable) killed=$(jget "$js" killed)" \
         "success_last200=$(jget "$js" success) time_out_last200=$(jget "$js" time_out)" \
         "over_deck_last200=$(jget "$js" over_deck) cube_abs_x_last200=$(jget "$js" cube_abs_x)" \
         "cube_tilted_last200=$(jget "$js" cube_tilted) tipped_over_deck_last200=$(jget "$js" tipped_over_deck)" \
         "unstable_iters=$(jget "$js" unstable_iters) last_iter=$(jget "$js" last_iter) ($js)"
else
    echo "RESULT $RUN_NAME: INCOMPLETE -- could not read $run"
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
verdict = sm.pair_result3(seeds)
json.dump({"verdict": verdict, "seeds": seeds, "task": "CubeToShelfStand3 (blind)",
           "note": sm.TASK3_NOTE}, open(os.path.join(d, f"result_pair_{aid}.json"), "w"), indent=1)
print(f"RESULT CubeToShelfStand3 blind (2 seeds): {verdict}")
PY
fi
# A run stopped by its predeclared kill rule is an outcome, not a job failure.
[ "$killed" = 1 ] && exit 0
[ -f "$BADCFG_FLAG" ] && exit 1
exit "$train_rc"
