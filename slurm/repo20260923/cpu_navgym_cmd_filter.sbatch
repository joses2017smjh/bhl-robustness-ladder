#!/bin/bash
#SBATCH --job-name=navgym-cmdfilter
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
# 2026-10-07: a SCRIPTED deployment low-pass on the NavGym v5 navigator's command (bhl_robust/eval/cmd_filter.py),
# screened on development mazes, then confirmed on never-used mazes. Verdicts: scripts/bench/navgym_cmd_filter.py.
# Why: the 2026-10-06 confirmation was NEGATIVE only on zero falls (6 falls in 288 learned episodes, A* 0 in 96); in
# every fall the commanded wz flipped between -1 and +1 rad/s at walking speed. tau 0.2 s = the gym's nominal 0.25 s
# lag minus the gait's measured 0.06 s (solutions-20260930/sysid); chosen from those numbers, not tuned on mazes.
#
# PREDECLARED (2026-10-07, FROZEN before any filtered episode exists):
#   DEV (maze seeds 72000-72047, scored on 2026-10-06, so development data with no claim): arms wz-lpf and
#   lpf-post-brake (tau 0.2 s) x actors armV5-s8/s9/s10 x {nominal, drop35}, 48 episodes per cell. An arm
#   QUALIFIES iff its 6 cells are complete (48 valid episodes each, layouts paired with 2026-10-06), it has 0
#   falls over all 288 episodes, and every actor reaches >= 40/48 nominal and >= 36/48 drop35 goals. Selected:
#   the qualifying arm with the most goals over its 288 episodes; tie -> wz-lpf. No arm qualifies -> DEV
#   NEGATIVE and no confirmation runs. The 6 fall episodes of 2026-10-06 are rerun unfiltered (reported, not gated).
#   CONFIRM (never-used maze seeds 78000-78047): the dev-selected arm only, actors armV5-s8/s9/s10 x
#   {nominal, drop35}, plus the SCRIPTED A* reference (unfiltered) x {nominal, drop35}, not gated. PASS iff
#   EACH actor reaches >= 40/48 nominal and >= 36/48 drop35 goals with 0 falls (the 2026-10-06 rule);
#   otherwise NEGATIVE; INCOMPLETE if any cell is missing, short or mislabelled.
#   Maze seeds 78000-78047: checked 2026-10-07 (no seed file under results/, solutions-20260930/ or the 2026-10-06
#   campaign; no --seed-start or "seed": reference in slurm/, scripts/, docs/, SLURM_JOBS.md) -- never used.
#   Labels: LEARNED gait (dr-default-s0) + LEARNED NavGym v5 final actors; SCRIPTED command low-pass + speed brake;
#   ORACLE pose and goal; A* reference SCRIPTED. Simulation only.
#
# Modes (NAVGYM_CF_MODE):
#   dev              array 0-11: arm = (wz-lpf, lpf-post-brake)[i/6], actor = (s8, s9, s10)[(i%6)/2], cond = (nominal, drop35)[i%2]
#   repro            one job: the 6 fall episodes of 2026-10-06, --policy-cmd-filter none
#   dev-verdict      one job (afterany on dev + repro): navgym_cmd_filter.py dev-verdict --final
#   confirm          array 0-7: 0-5 the dev-selected arm per actor/condition (as dev), 6-7 A* nominal / drop35;
#                    runs nothing unless the dev verdict is a final SELECTED (task 0 records not_run.json)
#   confirm-verdict  one job (afterany on confirm): navgym_cmd_filter.py confirm-verdict --final
# A cell directory that already holds episodes is never run again (maze_explore --no-overwrite refuses).
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
GAIT=$U/logs/rsl_rl/biped/2026-08-17_09-54-10_dr-default-s0/exported/deploy.yaml
V5=$REPO/results/navgym-v5-20261002
REF=/nfs/hpc/share/sanchej7/Computer_Vision/project_results_upgrade/humanoid_confirmatory/runs
ROOT=${NAVGYM_CF_ROOT:-$REPO/results/navgym-v5-cmdfilter-20261007}
MODE=${NAVGYM_CF_MODE:?set NAVGYM_CF_MODE to dev, repro, dev-verdict, confirm or confirm-verdict}
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
ARMS=(wz-lpf lpf-post-brake)
ACTORS=(armV5-s8 armV5-s9 armV5-s10)
CONDS=(nominal drop35)
sensor_mode() { [ "$1" = nominal ] && echo reactive || echo reactive_dropout; }
COMMON=(--upstream "$U" --gait "$GAIT" --n 6 --m 6 --extra-openings 1 --time-limit 180 --dropout-probability 0.35 --no-overwrite)
CACHE=${TMPDIR:-/tmp}/bhl-navgym-cmdfilter-${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}; mkdir -p "$CACHE"
trap 'rm -rf "$CACHE"' EXIT
echo "=== node: $(hostname) | HEAD $(git rev-parse --short HEAD) | mode $MODE | task ${SLURM_ARRAY_TASK_ID:-none} | $(date) ==="
run() {   # run <out dir> <maze_explore args...>; verdicts come from the per-seed JSONs, never from this exit code
    local out=$1; shift
    mkdir -p "$out"
    set +e
    "$PY" scripts/bench/maze_explore.py "${COMMON[@]}" --cache-dir "$CACHE" --out-dir "$out" "$@" > "$out/run.log" 2>&1
    local rc=$?
    set -e
    echo "maze_explore exit code $rc (informational): $out/run.log"
    grep "MAZE-EXPLORE SUMMARY" "$out/run.log" || true
}
learned_cell() {   # learned_cell <arm> <actor> <cond> <seed_start> <root>
    sha256sum "$V5/$2/actor.onnx"
    run "$5/$1/$2-$3" --seeds 48 --seed-start "$4" --sensor-mode "$(sensor_mode "$3")" \
        --policy "$V5/$2/actor.onnx" --policy-capture-pose --policy-cmd-filter "$1" --policy-cmd-tau 0.2
}
case "$MODE" in
dev)
    i=${SLURM_ARRAY_TASK_ID:?dev needs an array index 0-11}
    [ "$i" -ge 0 ] && [ "$i" -le 11 ] || { echo "dev array index $i outside 0-11"; exit 2; }
    learned_cell "${ARMS[$((i / 6))]}" "${ACTORS[$(((i % 6) / 2))]}" "${CONDS[$((i % 2))]}" 72000 "$ROOT/dev" ;;
repro)
    for spec in armV5-s8:nominal:72014 armV5-s9:nominal:72001 armV5-s9:drop35:72000 armV5-s9:drop35:72005 \
                armV5-s10:nominal:72044 armV5-s10:nominal:72045; do
        IFS=: read -r a c s <<< "$spec"
        run "$ROOT/dev/repro/$a-$c" --seeds 1 --seed-start "$s" --sensor-mode "$(sensor_mode "$c")" \
            --policy "$V5/$a/actor.onnx" --policy-capture-pose --policy-cmd-filter none
        mv "$ROOT/dev/repro/$a-$c/run.log" "$ROOT/dev/repro/$a-$c/run-$s.log"
        mv "$ROOT/dev/repro/$a-$c/summary.json" "$ROOT/dev/repro/$a-$c/summary-$s.json"
    done ;;
dev-verdict)
    "$PY" scripts/bench/navgym_cmd_filter.py dev-verdict "$ROOT/dev" --reference "$REF" --final ;;
confirm)
    i=${SLURM_ARRAY_TASK_ID:?confirm needs an array index 0-7}
    DEV_V=$ROOT/dev/verdict.json
    [ -f "$DEV_V" ] || { echo "NO DEV VERDICT: $DEV_V"; exit 4; }
    if ! ARM=$("$PY" scripts/bench/navgym_cmd_filter.py selected-arm "$DEV_V"); then
        if [ "$i" = 0 ] && [ ! -e "$ROOT/confirm/not_run.json" ]; then
            mkdir -p "$ROOT/confirm"
            "$PY" -c "import json,sys; v=json.load(open(sys.argv[1])); json.dump({'status': 'NOT RUN', 'reason': 'the dev screen did not select an arm (predeclared)', 'dev_verdict': v.get('verdict')}, open(sys.argv[2], 'w'), indent=1)" "$DEV_V" "$ROOT/confirm/not_run.json"
        fi
        echo "NAVGYM-CMDFILTER CONFIRM: not run (dev verdict did not select an arm)"; exit 0
    fi
    if [ "$i" -le 5 ]; then
        learned_cell "$ARM" "${ACTORS[$((i / 2))]}" "${CONDS[$((i % 2))]}" 78000 "$ROOT/confirm"
    elif [ "$i" -le 7 ]; then
        c=${CONDS[$((i - 6))]}
        run "$ROOT/confirm/astar/astar-$c" --seeds 48 --seed-start 78000 --sensor-mode "$(sensor_mode "$c")"
    else
        echo "confirm array index $i outside 0-7"; exit 2
    fi ;;
confirm-verdict)
    [ -e "$ROOT/confirm/not_run.json" ] && { echo "confirmation not run by the predeclared rule"; cat "$ROOT/confirm/not_run.json"; exit 0; }
    "$PY" scripts/bench/navgym_cmd_filter.py confirm-verdict "$ROOT/confirm" --dev-verdict "$ROOT/dev/verdict.json" --final ;;
*)
    echo "unknown NAVGYM_CF_MODE $MODE"; exit 2 ;;
esac
echo "=== done: $(date) ==="
