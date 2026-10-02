#!/bin/bash
#SBATCH --job-name=m7-plate-bench-v2
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=haswell&el8
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#
# Mission 7 plate bench v2 with the M3 stage (M3 of docs/SOLUTIONS_2026-10-01.md section 3), declared 2026-10-02
# before any bench-v2, replay or route episode ran with --stage-gait m3.  CPU MuJoCo, no GPU, no render.
#   LEARNED gait: the shipped arms-dr1.0-s0 gait throughout (M3 never swaps the policy) | SCRIPTED stage:
#   scripts/mission7_plate_stage.py PlateStage with --stage-gait m3 | ORACLE layout and plate pose (as the existing
#   stage uses them).
#
# Disclosure: bench v2 was introduced AFTER the M2 bench FAIL (21507962: 8/64 clears, 8 falls) revealed the run-up
# confound: 26/64 crossings ended in v1's own run-up (the shipped gait walking from route[k-1] to the pre-point holding
# the entry yaw) before the stage acted, so v1's >= 62/64 was unreachable for any stage gait.  M2's FAIL stands; v1
# (scripts/mission7_plate_bench.py, cpu_m7_plate_bench.sbatch) is unchanged and reproducible.  v2 keeps v1's grid,
# seeds, physics, 5 Hz stage polling, action transform and clear definition, and removes the run-up.
#
# Budget: M3 is predeclared (SLURM_JOBS.md, "User authorizations recorded 2026-10-01") as conditional on an M2 bench
# FAIL, a line of the same size as M2's 106 episodes under "don't worry about the budget"; it is flagged in the report.
# Here: 41 bench-v2 crossings (this job) + 10 exact-replay episodes + 32 route-gate episodes = 83 <= 106.  A smoke
# (mode `smoke`) is NOT part of the line: it runs the m3 stage on exploration train layouts >= 250 only; the replay's
# own --smoke layout (validation 0) runs in it only on the shipped and turnboth paths, for the identity checks.
#
# PREDECLARED RULE (frozen before any episode; a chain, each step only after the previous PASSES):
# - Bench v2: with N crossings run (64 minus the dropped set) and N_h per heading, bench v2 PASSES iff clears >=
#   N - 2, 0 falls, and clears >= N_h - 1 for every heading. Otherwise FAIL; INCOMPLETE if any declared crossing is
#   missing. The clear definition is unchanged: mission7_gates' real clear (base along >= +0.35 m past the plate
#   centre along the door direction), within 10 s of takeover.
# - Then the unchanged exact ten-fall replay once with --stage-gait m3, through the existing replay-gate machinery
#   (complete, 10 episodes, 10 upright, 0 falls; release arm m3-shipped-step).
# - Then the route gate as coded by M1: Doors and Transport each >= 16/16 successes on validation layouts 0-15
#   (32 episodes).
# Here N = 41 (0 deg: 12, +90: 11, -90: 10, 180: 8), so PASS needs >= 39/41 clears, 0 falls, and >= 11, 10, 9, 7 clears
# at 0, +90, -90, 180 deg (the same strictness as M2's 62/64 and 15/16).  The verdict is
# scripts/mission7_plate_bench_v2.py bench_v2_verdict(), written to <out>/verdict.json and read back from that JSON
# (never from an exit code).  Falls (tilt >= 0.78) count in EVERY phase after the spawn: settle, walk, stage, post.
# Nothing below is tuned on a bench, replay or route result.
#
# Grid (identical to v1, mission7_plate_bench.grid()): one crossing per TRAINING-split layout L = 0..31 (seeds 0..31)
# and door d = 0/1.  Entry heading relative to the door direction 0, +90, -90, 180 deg for L mod 4 = 0, 1, 2, 3; plate
# round (correct side) for floor(L/4) mod 2 = 0, square (wrong side) for 1; entry standstill for d XOR
# (floor(L/8) mod 2) = 0, walking for 1.  Disjoint from the replay (validation 0, 1, 4, 5, 7, 9, 12-15) and the route
# gate (validation 0-15).
#
# Entry protocol (scripts/mission7_plate_bench_v2.py; NO run-up):
#   P = plate centre - 0.30 m along the door direction (PlateStage's pre-point); entry yaw = door yaw + heading.
#   Spawn = the env's own standing pose (DebugEnv(stage doors, split train, seed 7000 + 2L + d) reset to L: default
#   joints + N(0, 0.02) rad, base at the model's qpos0 height, which puts the lowest foot geom 0.039 m above the floor
#   and 0.009 m above a plate top), moved in x, y and yaw only, velocity zeroed (as v1); then 3.0 s at zero command on
#   the shipped gait (the settle).
#   standstill: spawn at P, settle 3.0 s, take over.
#   walking: spawn at S = P - 0.30 m along the entry-yaw unit vector, settle 3.0 s, walk FORWARD at 0.30 m/s with the
#   route's heading hold clip(1.2 x yaw error, +-0.35) on the entry yaw for the nominal 1.0 s, take over at arrival
#   (within 0.13 m of P, as v1); not arrived after 3.0 s of walking = a miss.
#   Takeover = PlateStage._start(d, side) on the bench's plate, PlateStage.command until hand-back, then 2.0 s at zero
#   command; polled every 5 gait updates (0.2 s) through the route's action transform; runner.step physics, doors
#   closed, no MissionState termination; a stage not handed back 60 s after takeover is a miss.  Wall/door contacts,
#   spawn contacts and plate presses are recorded per crossing and reported, not gated.
# Drop rule (walking entries only; layout geometry alone, before any episode): a walking crossing runs only if no wall
#   or closed door lies within the robot's planar radius 0.316 m (measured envelope 0.632 m elbow to elbow; FK of the
#   default pose 0.295 m) of the straight segment S -> P, and no foot box (FK of the default pose: x -0.080..+0.143 m,
#   |y| 0.021..0.093 m in the base frame) lies on any plate at S.  DROPPED (23 of the 32 walking crossings; all 8
#   walking 180-deg crossings drop by construction, S being the plate centre):
#     L3 d1, L4 d1, L5 d1, L6 d1, L7 d1, L9 d0, L10 d0, L11 d0, L12 d0, L13 d0, L14 d0, L15 d0, L16 d1, L19 d1, L21 d1,
#     L22 d1, L23 d1, L24 d0, L26 d0, L27 d0, L29 d0, L30 d0, L31 d0
#   Run: 32 standstill + 9 walking (L0 d1, L1 d1, L2 d1, L8 d0, L17 d1, L18 d1, L20 d1, L25 d0, L28 d0) = 41.
# Spec tension, disclosed (not dropped, not shifted: the spec places every standstill base directly at P): in 6
#   standstill crossings, all in 1.5 m cells, the spawn pose starts with an elbow inside a wall (mj_forward at the spawn
#   instant, dist < 0): L4 d0 6.6 mm, L5 d0 6.9, L11 d1 1.3, L22 d0 4.2, L24 d1 7.0, L30 d1 9.3 mm (P lies 0.29 m from
#   the wall face against the 0.294 m elbow reach).  Every standstill spawn at P also has a foot over the plate edge
#   (P is 0.06 m from the plate edge; the feet reach 0.08-0.14 m from the base), the replay's own settle point.
#
# M3 stage (scripts/mission7_plate_stage.py M3_*; the shipped gait cannot turn in place: yaw-only 0.35 rad/s for 2 s
# turned 0.036 rad in the 2026-09-21 open-floor gait study): approach and settle unchanged (the settle is declared
# 0.40 s; as polled it lasts 0.40 or 0.60 s, see "Timing as polled"), then
#   turn while stepping: [0.30 m/s, 0, +-0.40 rad/s] toward the door direction until |error| < 0.25 rad (bound
#     2 pi/0.40 = 15.7 s); the short way, and within 0.25 rad of +-pi the way whose arc ends toward the door centreline.
#     0.30 m/s: ONE pre-registered sweep on open floor built from exploration train layout 255 (srun 21508399; rule in
#     the scratch PROGRESS.md before the result): .15/.20 never stepped, .25 stepped at +0.40 only, .30 stepped both
#     ways (+0.70 rad / 0.55 m and -0.89 rad / 0.52 m in 2 s) = the existing measured 0.30 m/s onset threshold.
#     0.40 rad/s: the interface bound (tanh x 0.4).  0.25 rad: BearingController.heading_tolerance.
#   cross: [0.30 m/s, 0, clip(1.2 x error, +-0.35)] (the route's heading hold), no lateral command, until the base is
#     0.35 m past the plate centre (= mission7_gates.CLEAR_ALONG_M) or the stage's 4.0 s cross bound.
#   turn back to the takeover heading with the turn law (within 0.25 rad of +-pi: toward the centreline), then hand back.
#   stall watchdog (turn and cross): base moved < 0.05 m over the last 1.2 s (RecoveryRouteController) -> prev_actions
#     := 0 (the stage's cross-kick evidence) + 0.40 s at -0.30 m/s (the route's pulse length and speed, backward off the
#     plate edge), then the phase resumes with a fresh window; at most 2 recoveries per crossing, then hand back.
#   No policy swap: env.controller.policy stays the shipped policy (checked per crossing).
# Timing as polled, disclosed (not changed): the settle comparison (PlateStage's `now < phase_until`, inherited
#   unchanged; v1 and M2 had it too) and the m3 cross bound (`now < cross_start_s + cross_max_s`) compare
#   float-accumulated d.time without an epsilon, so at the bench's 0.2 s polling the declared 0.40 s settle lasts 0.40
#   or 0.60 s and the 4.0 s cross bound 4.0 or 4.2 s (smoke 21508726: L251 d1 and L253 d1 settled 0.40 s, L250 d0
#   0.60 s; M2 record L0 d1 0.60 s).  The watchdog window, the recovery pulse and the bench's own settle, walk and
#   post-hand-back timers carry a 1e-9 s epsilon and are exact.
# Spec deviation, disclosed (M2's one-factor pattern; the spec says "at takeover"): M3 acts only after the unchanged
#   approach and settle.  At takeover the base is within 0.13 m of P, so the approach ends at once and the settle
#   commands zero for 0.40-0.60 s: a walking entry is brought to a standstill before the first M3 command, and in every
#   crossing 0.40-0.60 s of the 10 s clear window passes before M3 acts.
# Kinematic consequences, disclosed before any episode (not tuning): at 0.30 m/s and 0.40 rad/s the arc radius is
#   0.75 m nominal (0.68 m at the sweep's ~0.27 m/s).  A 90-deg turn moves the base about one radius toward the door
#   (often past +0.35 during the turn) and one radius toward the side it faced, which meets the plate-side wall in
#   1.5 m cells when it faced it.  A 180-deg U-turn moves it 2 radii (1.4-1.5 m) sideways in a 1.5-1.7 m cell: wall
#   contact is expected, and the timeline 0.40-0.60 s settle + (pi - 0.25)/0.40 = 7.2 s + 0.65 m / 0.27 m/s = 2.4 s =
#   10.0-10.2 s sits at or past the 10 s window at nominal tracking, so the 180-deg heading (8 standstill crossings,
#   >= 7 needed) is predicted to FAIL; with it the bench.  The turn back is a forward arc from along >= +0.35 toward
#   the closed bench door (face 0.825-0.925 m past the plate centre): contact there, and any fall, counts.
# Smoke observations, disclosed before any scored episode (smoke 21508726, cn-c21, 6 min, PASS, on the uncommitted
# working tree; exploration train layouts 250-253 and the replay's own --smoke layout; not scored; no constant,
# parameter or protocol choice changed in response).  Edited after it, on review: header disclosures only (this
# paragraph, "Timing as polled", the takeover deviation, the budget's smoke sentence, one step-4 comment line), a
# comment in the stage's M3 block, and the smoke's step 6 removed (the m3 replay on validation layout 0; below):
#   - machinery: default path byte-identical to the pre-M2 stage, turnboth path byte-identical to the M2 stage, the v1
#     bench's L250 d0 record identical with the M2 scripts, and 38/38 M2 crossings re-scored to their recorded clears.
#   - bench v2 L251 d1 (180 deg, round, standstill): the U-turn left the base 0.81 m BEFORE the plate centre (along
#     -0.81 m at the start of the crossing); two watchdog recoveries in the crossing; the 4.0 s crossing bound ended it
#     at along -0.49 m: no clear, no fall; wall contact in 401 stage samples.
#   - bench v2 L253 d1 (+90 deg, square, standstill): two watchdog recoveries in the turn, then a clear 9.64 s after
#     takeover (inside the 10 s window); no fall; wall contact in 438 stage samples.
#   - bench v2 L250 d0 (+90 deg, round, walking, explicit smoke labels): walk 1.2 s; the turn stalled with the robot in
#     wall contact (149 stage samples), two recoveries, watchdog hand-back: no crossing, no clear, no fall.
#   - exact replay, validation layout 0 (the replay's own --smoke layout), --stage-gait m3: the turn stalled, two
#     recoveries, watchdog hand-back at 30.8 s; the replayed commands then FELL at 34.0 s (the replay gate needs 10/10).
#     This was a pre-read of 1 of the 10 replay-gate layouts (validation 0, also inside the route gate's 0-15), on
#     cn-c21 rather than the pinned cn-c22, with the M3 treatment.  The smoke step that ran it was removed afterwards,
#     so no later smoke runs M3 on a scored layout.
#   - route integration (exploration train layout 252, early handoff): takeover at 18.8 s; 0.76 s into the M3 turn the
#     episode ended with Mission 7's excessive_collision (22.4 s).
#   So 1 of 3 smoke bench crossings cleared inside 10 s, every stage spent time in wall contact, the watchdog fired in
#   every smoke stage, and the replay's smoke layout fell after a watchdog hand-back: a bench-v2 FAIL is predicted
#   (and, were the bench to pass, a replay FAIL); the rule and the protocol stand as declared.
#
# Chain wiring: after a scored run this job reads <out>/verdict.json.  PASS -> it runs
#   M7_BENCH_V2_VERDICT=<out>/verdict.json bash slurm/repo20260923/submit_m7_replay_gates.sh --only m3-shipped-step --submit
# (the exact ten-fall replay with --stage-gait m3, pinned to cn-c22, source-frozen, then its follow-up, which releases
# the route gate with --stage-gait m3 only on a 10/10).  The release script refuses (RELEASE_BLOCKED, exit 3, nothing
# queued) unless that verdict is a scored bench-v2 m3 PASS and every replay-snapshot source still has the sha256 in
# <out>/provenance.json.  FAIL or INCOMPLETE -> nothing is submitted; the chain stops.  DRY_RUN=1 prints the release.
# The bench refuses an existing output directory and any physics source that differs from HEAD.
#
#   usage (from anywhere):
#     sbatch --job-name=m7-plate-bench-v2-smoke --time=01:00:00 slurm/repo20260923/cpu_m7_plate_bench_v2.sbatch smoke
#     sbatch --dependency=afterok:<smoke> slurm/repo20260923/cpu_m7_plate_bench_v2.sbatch bench
#   outputs: bench -> results/mission7-campaign-20260923/m3-plate-bench-v2/; smoke ->
#   results/mission7-campaign-20260923/m3-bench-v2-smoke/job-<id>/.  The smoke exits non-zero on any machinery failure,
#   is not scored and never releases anything; it refuses to run (exit 2) unless the job name contains "smoke" and the
#   job's own time limit is <= 1:00:00.
set -euo pipefail
mode=${1:?usage: cpu_m7_plate_bench_v2.sbatch bench|smoke}
REPO=${M7_REPO:-/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder}
PY=${PYTHON:-/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python}
CAMPAIGN=results/mission7-campaign-20260923
REPLAY_SOURCE=results/mission7-replay-smoke-20260921
REPLAY_BASELINE=results/mission7-approach-followup-20260922/replay-diagnose-cn-c22
RELEASE=${M7_BENCH_RELEASE:-slurm/repo20260923/submit_m7_replay_gates.sh}
DRY_RUN=${DRY_RUN:-0}
STAGE_GAIT=m3   # the only gait predeclared on bench v2
# The pre-M2 stage (every replay gate up to 2026-09-24) and the M2 stage (a6c64e0, the turnboth path) for identity.
REFERENCE_COMMIT=094797e
REFERENCE_STAGE_SHA256=fdde7a559bc92d047cc10f6c66775eb24e444db7b211404378bd0e65d3555576
M2_COMMIT=a6c64e0
M2_STAGE_SHA256=adb200c509f00fa9cce503353ce5b2149d07715137051f50ba05ad7696c5a1a7
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/scripts" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "m7 plate bench v2: mode=$mode gait=$STAGE_GAIT host=$(hostname) job=${SLURM_JOB_ID:-local} head=$(git rev-parse --short HEAD) dry_run=$DRY_RUN"

# Release the exact replay (--stage-gait m3) only on a bench-v2 PASS read from verdict.json.
release_if_pass() {
  local out=$1 verdict abs_out release_tmp rc=0
  verdict=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$out/verdict.json")
  case "$out" in /*) abs_out=$out ;; *) abs_out=$REPO/$out ;; esac
  case "$verdict" in
    PASS)
      echo "M7 BENCH V2 PASS -> releasing the exact ten-fall replay with --stage-gait m3 (tag m3-shipped-step)"
      if [ "$DRY_RUN" != 0 ]; then
        echo "DRY_RUN: would run: M7_BENCH_V2_VERDICT=$abs_out/verdict.json bash $RELEASE --only m3-shipped-step --submit"
        return 0
      fi
      release_tmp="$abs_out/.release-tmp"
      mkdir -p "$release_tmp"
      M7_BENCH_V2_VERDICT="$abs_out/verdict.json" TMPDIR="$release_tmp" bash "$RELEASE" --only m3-shipped-step --submit || rc=$?
      if [ "$rc" = 3 ]; then
        echo "RELEASE_BLOCKED: the release script refused m3-shipped-step (not a scored bench-v2 m3 PASS, or a source the"
        echo "  replay would snapshot differs from $out/provenance.json); nothing was queued; the verdict stands in $out/verdict.json."
        return 1
      elif [ "$rc" != 0 ]; then
        echo "M7 BENCH V2 RELEASE FAILED (exit $rc): the PASS stands in $out/verdict.json; run by hand once the cause is fixed:"
        echo "  cd $REPO && M7_BENCH_V2_VERDICT=$abs_out/verdict.json bash $RELEASE --only m3-shipped-step --submit"
        return 1
      fi ;;
    FAIL)
      echo "M7 BENCH V2 FAIL -> the chain stops: the replay and the route gate are NOT submitted." ;;
    *)
      echo "M7 BENCH V2 $verdict -> nothing released (a bench that is not complete is not a verdict)." ;;
  esac
}

if [ "$mode" = bench ]; then
  OUT=${M7_BENCH_V2_OUT:-$CAMPAIGN/m3-plate-bench-v2}
  [ ! -e "$OUT" ] || { echo "refusing: $OUT exists (a bench result is never overwritten)" >&2; exit 2; }
  # the physics sources: the replay snapshot's set (scripts/submit_mission7_plate_stage.py) minus its sbatch, plus both
  # bench launchers; scripts/mission7_plate_bench_v2.py source_files() lists the same files for provenance.json
  sources=(scripts/mission7*.py src/bhl_robust/mission src/bhl_robust/__init__.py src/bhl_robust/sensor_io.py
           src/bhl_robust/eval/__init__.py src/bhl_robust/eval/multi_robot.py src/bhl_robust/eval/mjcf_assets.py
           src/bhl_robust/eval/livery.py src/bhl_robust/eval/team_sensors.py slurm/repo20260923/cpu_m7_plate_bench.sbatch
           slurm/repo20260923/cpu_m7_plate_bench_v2.sbatch)
  for f in scripts/mission7_plate_bench_v2.py slurm/repo20260923/cpu_m7_plate_bench_v2.sbatch; do
    git ls-files --error-unmatch "$f" >/dev/null 2>&1 || { echo "refusing: $f is not committed" >&2; exit 2; }
  done
  if ! git diff --quiet HEAD -- "${sources[@]}"; then
    echo "refusing: bench sources differ from HEAD; commit them first:" >&2; git status --short -- "${sources[@]}" >&2; exit 2
  fi
  "$PY" scripts/mission7_plate_bench_v2.py --repo "$REPO" --out "$OUT" --mode scored --stage-gait "$STAGE_GAIT" --preflight
  "$PY" scripts/mission7_plate_bench_v2.py --repo "$REPO" --out "$OUT" --mode scored --stage-gait "$STAGE_GAIT"
  release_if_pass "$OUT"
  echo "M7_PLATE_BENCH_V2_COMPLETE"
  exit 0
fi

[ "$mode" = smoke ] || { echo "unknown mode $mode (bench|smoke)" >&2; exit 2; }

# Smoke limits (fail closed): a Slurm job whose name contains "smoke" and whose own time limit is <= 1:00:00.
smoke_limits_ok() {
  local name=${SLURM_JOB_NAME:-} id=${SLURM_JOB_ID:-} limit a b c seconds
  case "$name" in *smoke*) ;; *) echo "refusing: smoke mode needs a Slurm job name containing 'smoke' (got '$name')" >&2; return 1 ;; esac
  [ -n "$id" ] || { echo "refusing: smoke mode runs only as a Slurm job (no SLURM_JOB_ID)" >&2; return 1; }
  for _ in 1 2 3; do
    limit=$(squeue -h -j "$id" -o %l 2>/dev/null | head -n 1 | tr -d '[:space:]') || limit=""
    [ -z "$limit" ] || break
    sleep 10
  done
  # squeue prints M:SS below an hour, H:MM:SS below a day, D-HH:MM:SS above; anything else is refused
  case "$limit" in
    *[!0-9:]*|""|:*|*:|*::*|*:*:*:*) echo "refusing: smoke time limit '$limit' is not <= 1:00:00" >&2; return 1 ;;
  esac
  IFS=: read -r a b c <<<"$limit"
  if [ -n "$c" ]; then seconds=$(( 10#$a * 3600 + 10#$b * 60 + 10#$c ))
  elif [ -n "$b" ]; then seconds=$(( 10#$a * 60 + 10#$b ))
  else echo "refusing: smoke time limit '$limit' is not <= 1:00:00" >&2; return 1; fi
  [ "$seconds" -le 3600 ] || { echo "refusing: smoke time limit $limit is over 1:00:00" >&2; return 1; }
  echo "smoke limits: job $id '$name', time limit $limit"
}
smoke_limits_ok || exit 2
SMOKE=${M7_SMOKE_DIR:-$CAMPAIGN/m3-bench-v2-smoke/job-${SLURM_JOB_ID:-local}}
case "$SMOKE" in *-smoke/*|*-smoke) ;; *) echo "refusing: smoke output must sit under a *-smoke directory" >&2; exit 2 ;; esac
[ ! -e "$SMOKE" ] || { echo "refusing: $SMOKE exists" >&2; exit 2; }
mkdir -p "$SMOKE/reference" "$SMOKE/reference-m2" "$SMOKE/v1-reference-tree"
# exploration train layouts only: 251 d1 = grid 180 deg round standstill; 253 d1 = grid +90 deg square standstill;
# 250 d0 with explicit labels +90 deg round walking (no grid-labelled walking crossing on 250-255 passes the drop rule)
SMOKE_CROSSINGS=251:1,250:0:90:round:walking,253:1
V1_SMOKE_CROSSING=250:0   # v1 identity: the M2 smokes' own walking crossing (-90 deg round), --stage-gait turnboth
start=$(date +%s)

# 1. preflights (imports, argument parsing, the m3 constants)
"$PY" scripts/mission7_plate_bench_v2.py --repo "$REPO" --out "$SMOKE/bench" --mode smoke --stage-gait "$STAGE_GAIT" \
      --crossings "$SMOKE_CROSSINGS" --preflight
"$PY" scripts/mission7_plate_stage.py --repo "$REPO" --campaign "$REPLAY_SOURCE" --baseline "$REPLAY_BASELINE" \
      --out "$SMOKE/replay-m3" --smoke --stage-gait m3 --preflight | tee "$SMOKE/replay-m3-preflight.txt"

# 2. default-path identity in MuJoCo: the pre-M2 stage file and this one, shipped gait, replay layout 0, same node
git show "$REFERENCE_COMMIT:scripts/mission7_plate_stage.py" > "$SMOKE/reference/mission7_plate_stage.py"
echo "$REFERENCE_STAGE_SHA256  $SMOKE/reference/mission7_plate_stage.py" | sha256sum -c -
"$PY" "$SMOKE/reference/mission7_plate_stage.py" --repo "$REPO" --campaign "$REPLAY_SOURCE" --baseline "$REPLAY_BASELINE" \
      --out "$SMOKE/default-reference" --smoke
"$PY" scripts/mission7_plate_stage.py --repo "$REPO" --campaign "$REPLAY_SOURCE" --baseline "$REPLAY_BASELINE" \
      --out "$SMOKE/default-new" --smoke
cmp "$SMOKE/default-reference/episodes.json" "$SMOKE/default-new/episodes.json"
cmp "$SMOKE/default-reference/result.json" "$SMOKE/default-new/result.json"
echo "SMOKE default path: episodes.json and result.json byte-identical to the pre-M2 stage"

# 3. turnboth-path identity: the M2 stage (a6c64e0) and this one, --stage-gait turnboth, replay layout 0, same node
git show "$M2_COMMIT:scripts/mission7_plate_stage.py" > "$SMOKE/reference-m2/mission7_plate_stage.py"
echo "$M2_STAGE_SHA256  $SMOKE/reference-m2/mission7_plate_stage.py" | sha256sum -c -
"$PY" "$SMOKE/reference-m2/mission7_plate_stage.py" --repo "$REPO" --campaign "$REPLAY_SOURCE" --baseline "$REPLAY_BASELINE" \
      --out "$SMOKE/turnboth-reference" --smoke --stage-gait turnboth
"$PY" scripts/mission7_plate_stage.py --repo "$REPO" --campaign "$REPLAY_SOURCE" --baseline "$REPLAY_BASELINE" \
      --out "$SMOKE/turnboth-new" --smoke --stage-gait turnboth
cmp "$SMOKE/turnboth-reference/episodes.json" "$SMOKE/turnboth-new/episodes.json"
cmp "$SMOKE/turnboth-reference/result.json" "$SMOKE/turnboth-new/result.json"
echo "SMOKE turnboth path: episodes.json and result.json byte-identical to the M2 stage"

# 4. v1 bench identity: v1 with the M2 commit's mission7 scripts (= HEAD's) and with this tree's, one crossing,
#    decompressed records
git archive "$M2_COMMIT" scripts/mission7_plate_bench.py scripts/mission7_plate_stage.py scripts/mission7_gates.py \
    scripts/mission7_overnight.py | tar -x -C "$SMOKE/v1-reference-tree"
PYTHONPATH="$REPO/src:$SMOKE/v1-reference-tree/scripts:$REPO/scripts" "$PY" "$SMOKE/v1-reference-tree/scripts/mission7_plate_bench.py" \
    --repo "$REPO" --out "$SMOKE/v1-reference" --mode smoke --crossings "$V1_SMOKE_CROSSING"
"$PY" scripts/mission7_plate_bench.py --repo "$REPO" --out "$SMOKE/v1-new" --mode smoke --crossings "$V1_SMOKE_CROSSING"
for side in reference new; do
  [ -s "$SMOKE/v1-$side/crossings/L250-d0.json.gz" ] || { echo "v1 identity: $side crossing record missing" >&2; exit 1; }
  zcat "$SMOKE/v1-$side/crossings/L250-d0.json.gz" > "$SMOKE/v1-$side.L250-d0.json"
done
cmp "$SMOKE/v1-reference.L250-d0.json" "$SMOKE/v1-new.L250-d0.json"
rm -f "$SMOKE/v1-reference.L250-d0.json" "$SMOKE/v1-new.L250-d0.json"   # decompressed copies of the two records kept gzipped
echo "SMOKE v1 bench: the L250 d0 crossing record is identical with the M2 scripts and with this tree's"

# 5. the 38 M2 crossings that reached the stage, re-scored with bench v2's per-crossing clear computation (exit 1 on
#    any mismatch)
"$PY" scripts/mission7_plate_bench_v2.py --repo "$REPO" --out "$SMOKE/rescore" --mode rescore-m2

# 6. removed on review (2026-10-02, after smoke 21508726): the exact replay with --stage-gait m3 ran on the replay's own
#    --smoke layout, validation 0, one of the ten scored replay-gate layouts.  No smoke runs M3 on a scored layout; the
#    m3 replay CLI is the step-1 preflight (checked in step 9), the m3 trace through the replay gate's real-clear
#    reader is tests/test_mission7_plate_bench_v2.py.

# 7. bench-v2 crossings end to end with --stage-gait m3 (standstill 180, standstill +90, walking +90), exploration only
"$PY" scripts/mission7_plate_bench_v2.py --repo "$REPO" --out "$SMOKE/bench" --mode smoke --stage-gait "$STAGE_GAIT" \
      --crossings "$SMOKE_CROSSINGS"

# 8. the route-gate controller with --stage-gait m3 on an EXPLORATION train layout (never a route-gate layout):
#    RouteHandoffController with the follow-up's route composition, Doors stage, until the first hand-back + 3 s or
#    120 s.  Integration only: nothing here is scored.
"$PY" - "$SMOKE/route-integration" "$REPO" <<'PYEOF'
import json, pathlib, sys
from bhl_robust.mission.approach_debug import DebugEnv
from mission7_route_handoff_probe import RouteHandoffController
out, repo = pathlib.Path(sys.argv[1]), sys.argv[2]
out.mkdir(parents=True)
env = DebugEnv(repo, out / "cache", stage="doors", split="train", seed=4252)
env.reset(252)
shipped = env.controller.policy
controller = RouteHandoffController(env, handoff_mode="early", stage_activate=True, exit_ramp_s=1.2,
                                    stage_wait_open_s=0.0, rejoin_advance=True, exit_ramp_center=True,
                                    stage_gait="m3")
done, swapped = False, False
while not done and env.runner.d.time < 120.:
    env.runner.contact_trace = []
    action = controller.action()
    _, _, done, _ = env.step(action)
    swapped |= env.controller.policy is not shipped
    controller.observe_contacts(env.runner.contact_trace)
    controller.observe_step(action)
    handbacks = [h for h in controller.stage.history if h["phase"] == "recorded"]
    if handbacks and env.runner.d.time > handbacks[-1]["time_s"] + 3.:
        break
summary = {"layout": ["train", 252], "time_s": float(env.runner.d.time), "done": bool(done),
           "failure": env.state.failure, "open": list(env.state.open), "crossed": list(env.state.crossed),
           "stage_activations": controller.stage_activation_times, "phases": [h["phase"] for h in controller.stage.history],
           "history": controller.stage.history, "gait_events": controller.stage.gait_events,
           "m3_events": controller.stage.m3_events, "exit_ramp_events": controller.exit_ramp_events,
           "rejoin_advance_events": controller.rejoin_advance_events, "policy_ever_swapped": swapped}
(out / "route_integration.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
print("ROUTE_INTEGRATION", json.dumps({k: summary[k] for k in ("time_s", "done", "failure", "phases", "policy_ever_swapped")}))
PYEOF

# 9. machinery checks; any failure exits non-zero
"$PY" - "$SMOKE" "$start" <<'PYEOF'
import gzip, json, pathlib, sys, time
smoke, start = pathlib.Path(sys.argv[1]), float(sys.argv[2])
checks, problems = {}, []

def check(name, ok, detail=None):
    checks[name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        problems.append(name)

M3_PHASES = {"approach", "settle", "turn", "cross", "turn_back", "recover", "recorded", "wait_open", "cross_kick"}
rescore = json.loads((smoke / "rescore/rescore_m2.json").read_text())
check("rescore_m2_38_identical", rescore["all_match"] and rescore["reached_stage"] == 38 and rescore["matches"] == 38,
      {k: rescore[k] for k in ("reached_stage", "matches", "recorded_clears", "rescored_clears")})

# step 6 removed: the replay CLI is checked through its m3 preflight (step 1); no m3 replay episode runs on a scored layout
lines = [line for line in (smoke / "replay-m3-preflight.txt").read_text().splitlines() if line.strip()]
try:
    pre = json.loads(lines[-1]) if lines else {}
except ValueError:
    pre = {}
m3_check = pre.get("stage_gait_check") if isinstance(pre.get("stage_gait_check"), dict) else {}
check("replay_m3_preflight", pre.get("status") == "PREFLIGHT_OK" and pre.get("stage_gait") == "m3"
      and m3_check.get("turn_vx_mps") == 0.3 and m3_check.get("max_recoveries") == 2,
      {"status": pre.get("status"), "stage_gait": pre.get("stage_gait"), "stage_gait_check": m3_check})
check("replay_m3_no_scored_episode", not (smoke / "replay-m3").exists(), "no m3 replay output (validation layouts are scored)")

verdict = json.loads((smoke / "bench/verdict.json").read_text())
check("bench_v2_smoke_verdict_never_pass", verdict["verdict"] in ("SMOKE_RUN", "SMOKE_INCOMPLETE")
      and verdict.get("bench") == "v2" and verdict.get("stage_gait") == "m3", verdict["verdict"])
provenance = json.loads((smoke / "bench/provenance.json").read_text())
check("bench_v2_provenance", provenance.get("bench") == "v2" and provenance.get("stage_gait") == "m3"
      and "scripts/mission7_plate_bench_v2.py" in provenance.get("sources_sha256", {})
      and "slurm/repo20260923/cpu_m7_plate_bench_v2.sbatch" in provenance.get("sources_sha256", {}),
      sorted(provenance.get("sources_sha256", {}))[:3])
crossings = sorted((smoke / "bench/crossings").glob("*.json.gz"))
check("bench_v2_three_crossings_written", len(crossings) == 3, [p.name for p in crossings])
entries, headings = set(), set()
for path in crossings:
    with gzip.open(path, "rt") as stream:
        c = json.load(stream)
    entries.add(c["entry"]); headings.add(c["heading_deg"])
    name = path.name.split(".")[0]
    check(f"{name}_complete", c["complete"] and c["end_reason"] in ("complete", "fall", "walk_timeout", "stage_timeout"),
          c["end_reason"])
    events = c["m3_events"] if isinstance(c["m3_events"], list) else None
    check(f"{name}_shipped_gait_no_swap", c["gait_events"] == [] and c["policy_is_shipped_at_end"] is True
          and events is not None and (events[:1] and events[0]["event"] == "takeover" if c["takeover_s"] is not None
                                      else events == []), (events or [])[:3])
    check(f"{name}_phases_valid", set(h["phase"] for h in c["stage_history"]) <= M3_PHASES,
          [h["phase"] for h in c["stage_history"]])
    check(f"{name}_outcome_reported", True, {k: c[k] for k in ("heading_deg", "plate", "entry", "end_reason", "takeover_s",
          "handback_s", "clear", "real_clear", "clear_after_takeover_s", "fell", "fall_phase", "maximum_tilt")}
          | {"spawn": c["spawn"], "events": c["events"], "phases": [h["phase"] for h in c["stage_history"]],
             "m3_events": c["m3_events"], "wall_contacts": c["wall_contacts"],
             "plate_activation": c["plate_activation"], "samples": len(c["samples"])})
check("bench_v2_standstill_and_walking", entries == {"standstill", "walking"}, sorted(entries))
check("bench_v2_has_90_or_180", bool(headings & {90, -90, 180}), sorted(headings))
route = json.loads((smoke / "route-integration/route_integration.json").read_text())
check("route_integration_m3_no_swap", route["gait_events"] == [] and route["policy_ever_swapped"] is False, route["phases"])
# informational, never a failure: whether the route reached a plate in its 120 s is the route's own outcome
check("route_integration_stage_exercised_reported", True,
      "the stage took over" if route["phases"] else "not exercised (no takeover before the route ended or 120 s)")
summary = {"ok": not problems, "problems": problems, "wall_s": time.time() - start, "checks": checks}
(smoke / "smoke_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
print(json.dumps(summary, indent=2, default=str))
sys.exit(0 if not problems else 1)
PYEOF
# lossless: keep the smoke directory small (episodes.json traces gzipped in place, never deleted)
for f in "$SMOKE"/*/episodes.json; do if [ -f "$f" ]; then gzip "$f"; fi; done
echo "M7_PLATE_BENCH_V2_SMOKE_PASS (machinery only; not scored; $(( $(date +%s) - start )) s)"
