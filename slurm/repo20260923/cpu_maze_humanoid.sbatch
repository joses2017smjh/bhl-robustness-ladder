#!/bin/bash
#SBATCH --job-name=maze-humanoid
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#
# 2026-09-28: the randomized lidar maze (maze_explore.py) on the 22-DoF HUMANOID, CPU MuJoCo, no render.
#   LEARNED gait arms-turn-turnboth-s0 (one checkpoint) | SCRIPTED A* planner + turn-then-walk |
#   ORACLE pose and goal
# The gait is the only QUALIFIED turning checkpoint (cpu_turn_qualify.sbatch, 2026-09-27: turn 10/10,
# walk 3/3, push 7/60) -- ONE checkpoint of 12 turning-arm seeds, not a recipe. It marches in place at
# zero command and turns while stepping. Map, planner and controller are SCRIPTED; the maze is seen only
# through the lidar; localization and the goal coordinate are ORACLE. The biped A* (LEARNED biped gait
# dr-default-s0, the published configuration, default flags) runs on the same seeds as the reference.
#
# Humanoid path (maze_explore.py --variant humanoid; the default biped path is byte-identical, shown on the
# published hard-6x6 seed-11 trace): model berkeley_humanoid_lite.xml (22 DoF); team_sensors deck mounts
# (lidar 0.72 m / stereo 0.70 m above the base frame, which sits at the feet: world z 0.72 / 0.70 at spawn,
# ~0.66 / 0.64 standing, below the 1.10 m walls; 108/108 lidar rays hit walls in a 6x6 maze); fall = the
# run_eval rule behind the 7/60 push number (tilt > 0.78 rad OR root sink > 0.25 m below spawn).
# Footprint vs planner: humanoid planar radius 0.309 m (arms at +-0.307 m) vs biped 0.204 m; corridor clear
# width 1.32 m (1.40 m cell - 0.08 m wall); the biped's 0.30 m inflation is below the humanoid's radius, so
# the humanoid runs 0.50 m (free channel 1.32 - 2 x 0.50 = 0.32 m nominal, ~0.2 m after the 0.10 m grid).
#
# Scored block: hard 6x6, 1 extra opening, maze seeds 12-23, 180 s per episode. Never run by anything
# (2026-09-28, grep over results/): per-seed random-maze JSONs cover (6,6,1) seeds 0-11 and 30000-30011
# and (5,5,2) seeds 0-11 only; no file seed12..seed23 exists under results/; NavGym held-out sets are
# 10000+ / 20000+. Disclosed: the 2-D NavGym trains its learned actor on maze seeds < 10000 -- irrelevant
# here (neither the scripted planner nor either gait ever learned from a maze).
#
# PILOT, every episode (maze seeds >= 100 only; humanoid; cruise 0.30 m/s and turn rate 0.6 rad/s held at
# the biped's published values -- 0.6 rad/s is also the qualification's turn command -- and NOT varied;
# settle 1.0 s; controller thresholds at their defaults; hard = 6x6/1 opening/180 s, base = 5x5/2 openings/
# 150 s; only the planner inflation was varied):
#   smoke (code checks, time-outs by construction): hard 100 with a 20 s limit, inflate 0.40; after the
#   freeze, base 100-101 with a 25 s limit, base 102 with a 4 s limit (blank-render frame/footer check),
#   and a dry run of this launcher's flow on hard 100 with a 5 s limit, both arms, scratch output.
#   Round 1 -- map integrated from the BASE xy (the biped's code path), superseded:
#     inflate 0.40, hard 100-105: reached 6/6, 0 falls, clean 1/6 (wall steps 0/318/55/14/32/40, mostly
#                                 while walking), times 40.2/113.7/53.4/53.0/64.4/49.6 s
#     inflate 0.50, hard 100-105: reached 6/6, 0 falls, clean 6/6, times 45.6/128.6/55.0/55.8/67.7/54.4 s
#     inflate 0.50, base 100-105: reached 6/6, 0 falls, clean 6/6, times 36.2/34.0/37.3/48.1/37.9/69.6 s
#     inflate 0.50, hard 106-111: reached 5/6, 0 falls, clean 5/6 -- seed 106 STUCK for the last 120 s,
#                                 times 107-111: 73.8/65.7/56.2/122.4/61.0 s
#     two diagnostic re-runs of hard 106 (inflate 0.50), stopped at t = 70 s: the deck lidar sits 0.12 m
#     forward and 0.72 m up, and the pitching gait puts it ~0.2 m ahead of the base, so scans integrated
#     from the base drew the wall the robot faced 0.1 m too close (occupied cells at x = 0.55 against a
#     wall face at 0.66); the inflated map closed the corridor, A* found no path, the old plan through
#     the wall was kept, and the speed brake held the robot (range scale 0.04) with the heading error
#     under the turn threshold. Fix (humanoid path only): each scan is integrated from the lidar's own
#     pose at capture, rays projected on the floor plane, floor returns clearing cells but marking nothing.
#   Round 2 -- map from the sensor pose (the code that runs here); the basis of the freeze:
#     inflate 0.50, hard 100-111: reached 12/12, 0 falls, clean 12/12, median 67.2 s, times
#                                 48.0/129.0/59.4/58.6/69.3/54.7/88.6/77.6/66.8/57.9/126.6/67.2 s
#     inflate 0.50, base 100-105: reached 6/6, 0 falls, clean 6/6, times 39.6/39.1/39.0/50.8/38.6/71.5 s
#     inflate 0.40, hard 100-111: reached 12/12, 0 falls, clean 11/12 (seed 107: 6 wall steps while
#                                 turning), median 63.9 s, times 40.5/107.5/60.2/53.7/63.9/60.2/83.2/69.6/
#                                 66.0/62.7/116.2/57.6 s
#     measured over round 2: lidar origin up to 0.21-0.23 m ahead of the base per episode; 0-105 floor
#     returns per episode; max tilt 0.13-0.16 rad, max sink 0.05-0.09 m (fall bars 0.78 rad / 0.25 m).
#   Byte-identity of the default biped path: the published hard-6x6 seed 11 (a published seed, not a
#   scored one) re-run before and after the edit: seed11.json and summary.json identical except
#   wall_seconds, trace equal to results/maze-explore-20260924/hard-6x6/seed11.json.
# Freeze rule for the round-2 inflation choice, written 2026-09-28 before any round-2 inflate-0.40 episode
# existed (round-2 inflate-0.50 hard seeds 100, 101, 106 had been seen): on round-2 hard seeds 100-111,
# take the inflation with more episodes reached with 0 falls; tie -> more clean episodes; tie -> the
# larger inflation (more margin over the 0.309 m radius). Cruise and turn rate were not varied.
# Result: 12 = 12 reached with 0 falls, clean 12 (0.50) > 11 (0.40) -> inflate 0.50.
# Frozen from the pilot (maze_explore.HUMANOID_FROZEN): --cruise 0.30 --turn-rate 0.6 --inflate 0.50
# --settle-s 1.0, turn/walk/waypoint thresholds at the controller defaults (0.40 / 0.15 rad, 0.4 rad/s,
# 0.30 m); the goal radius is the judge's 0.30 m for both robots. Nothing is re-tuned after this run.
#
# PREDECLARED RULE (2026-09-28, before any scored seed has run; the verdict is maze_explore.py
# humanoid-verdict, the function humanoid_maze_verdict, read from the per-seed JSONs, never exit codes):
# PASS iff >= 10 of the 12 humanoid episodes (hard 6x6, 1 extra opening, maze seeds 12-23, 180 s each,
# frozen settings) reach the goal AND there are 0 falls in the 12; else FAIL; INCOMPLETE if any humanoid
# seed JSON or the humanoid summary is missing or does not match the declared gait/maze/settings. Clean
# (no wall contact) counts and median completion times are reported beside the biped A* reference on
# the same seeds, not gated.
#
# Stated risks (not scored): (1) marching in place -- at zero command the gait keeps stepping, so the
# robot drifts while "stopped"; (2) stopping at the goal -- the judge ends the episode on the first entry
# into the 0.30 m goal radius (as for the biped), so holding position at the goal is NOT tested; (3)
# turning in place inside a corridor -- the gait turns while stepping and translates during turns, with
# 0.35 m of lateral margin per side when centred; wall contacts per controller state are in every JSON;
# (4) the pilot-106 deadlock mechanism is still possible with the corrected map: if A* finds no path the
# old plan is kept, the brake can zero the forward speed while the heading error stays under the turn
# threshold, and the gait does not answer small yaw-rate commands -- counted per episode as
# planner_failures, and a scored "stuck" is reported as such, never as reached; (5) time: the slowest
# round-2 hard episodes took 126.6 and 129.0 s of the 180 s limit (the 0.50 m inflation lengthens paths).
# Sensor-model asymmetry (also in verdict.json "comparability"): the humanoid maps from the lidar pose
# with floor returns dropped, the biped reference from its base xy exactly as published.
#
# Outputs (new directory; nothing is overwritten):
#   results/maze-humanoid-20260928/humanoid-hard-6x6/seed{12..23}.json + summary.json
#   results/maze-humanoid-20260928/biped-hard-6x6/seed{12..23}.json + summary.json
#   results/maze-humanoid-20260928/verdict.json   (written only when PASS or FAIL; re-printed if present)
# An arm with a summary.json is reused, not recomputed; an arm with partial per-seed JSONs is refused
# (move it aside to rerun); maze_explore.py --no-overwrite refuses any existing per-seed file.
#
# Usage (from the repo; MAZE_HUM_COMMIT = the commit that holds this launcher and --variant humanoid):
#   MAZE_HUM_COMMIT=<sha> sbatch --export=ALL,MAZE_HUM_COMMIT=<sha> slurm/repo20260923/cpu_maze_humanoid.sbatch
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
OUT=$REPO/results/maze-humanoid-20260928
# Run from a snapshot of a pinned commit, not the live working tree; no HEAD fallback.
PIN=${MAZE_HUM_COMMIT:?set MAZE_HUM_COMMIT to the commit holding maze_explore.py --variant humanoid}
if ! git -C "$REPO" show "$PIN:scripts/bench/maze_explore.py" 2>/dev/null | grep -c -- '"--variant"' >/dev/null; then  # -c reads all input (no SIGPIPE under pipefail)
    echo "pinned commit $PIN has no maze_explore.py --variant; refusing to run" >&2; exit 2
fi
COMMON=(--seed-start 12 --seeds 12 --n 6 --m 6 --extra-openings 1 --time-limit 180 --no-overwrite)
HUMANOID_FLAGS=(--variant humanoid --cruise 0.30 --turn-rate 0.6 --inflate 0.50 --settle-s 1.0)
BIPED_FLAGS=()
JOB=${SLURM_JOB_ID:-local$$}
CACHE=${TMPDIR:-/tmp}/bhl-maze-humanoid-$JOB; mkdir -p "$CACHE"
WT=$CACHE/wt
git -C "$REPO" worktree add --detach "$WT" "$PIN" >/dev/null
trap 'git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1 || true; rm -rf "$CACHE"' EXIT
cd "$WT"
export PYTHONPATH="$WT/src" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
echo "=== maze humanoid | $(hostname) | pinned commit $PIN | $(date) ==="
mkdir -p "$OUT"

if [ -f "$OUT/verdict.json" ]; then
    "$PY" scripts/bench/maze_explore.py humanoid-verdict "$OUT"     # re-prints; never overwrites
    exit 0
fi

# the pinned code's frozen settings must be exactly the flags below, or nothing runs
"$PY" - "${HUMANOID_FLAGS[@]}" <<'EOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("me", "scripts/bench/maze_explore.py")
me = importlib.util.module_from_spec(spec); spec.loader.exec_module(me)
flags = sys.argv[1:]
got = {flags[i][2:].replace("-", "_"): flags[i + 1] for i in range(0, len(flags), 2) if flags[i] != "--variant"}
want = {k: v for k, v in me.HUMANOID_FROZEN.items() if v is not None}
if set(got) != set(want) or any(float(got[k]) != want[k] for k in want):
    sys.exit(f"launcher flags {got} != pinned HUMANOID_FROZEN {me.HUMANOID_FROZEN}; refusing to run")
print("frozen settings match the pinned code:", me.HUMANOID_FROZEN)
EOF

run_arm() {   # name dir flags...
    local name=$1 dir=$2; shift 2
    if [ -f "$dir/summary.json" ]; then echo "have $dir (complete; reused, not recomputed)"; return 0; fi
    if compgen -G "$dir/seed*.json" >/dev/null; then
        echo "$name: partial per-seed outputs in $dir -- NOT overwritten; move them aside to rerun"; return 0
    fi
    echo "--- $name $(date)"
    "$PY" scripts/bench/maze_explore.py --upstream "$U" --cache-dir "$CACHE/$name" "${COMMON[@]}" "$@" --out-dir "$dir" 2>&1 \
        | grep -v "^\[" || echo "$name: maze_explore exited non-zero (the verdict reads the JSONs, not this)"
}
run_arm humanoid "$OUT/humanoid-hard-6x6" "${HUMANOID_FLAGS[@]}"
run_arm biped "$OUT/biped-hard-6x6" "${BIPED_FLAGS[@]}"

echo "=== verdict $(date) ==="
"$PY" scripts/bench/maze_explore.py humanoid-verdict "$OUT"
echo "=== done: $(date) ==="
