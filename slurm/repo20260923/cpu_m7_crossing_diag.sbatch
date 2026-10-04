#!/bin/bash
#SBATCH --job-name=m7-crossing-diag
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#
# Mission 7 crossing diagnosis (workstream m7-crossdiag; frozen design: SLURM_JOBS.md, "User approval recorded
# 2026-10-03 09:55", item (D) "Mission 7 crossing diagnosis (investigation, no gate)", with the coordinator's "Correction
# to design (D)" (layouts 170-249), "Coordinator re-freeze of (D)'s measurement and classifier, 2026-10-03 11:40" and
# "Coordinator correction to (D)'s re-frozen classifier, 2026-10-04 10:45": both obstruction evidences of a blocked swing
# also require a floor-side touchdown, after the implementer's smoke-record audit (smoke 21539565, layouts 32/33/38) found
# 4 of 11 "blocked" swings had landed on the plate top).
# CPU MuJoCo, no GPU, no render.  INVESTIGATION, NO GATE: nothing here is a bench verdict or changes a bench rule.
#   LEARNED gaits: the shipped arms-dr1.0-s0 gait outside the stage + the LEARNED clock-s2 stage gait
#   (arms-turngait-clock-s2) | SCRIPTED stage: PlateStage, stage_gait=clocks2 (M2's turnboth law) | ORACLE layout and
#   plate pose (as the existing stage uses them).
#
# What runs (scripts/bench/m7_crossing_diag.py): bench-v2 crossings exactly as the clocks2 bench v2 (21517668) runs them:
# scripts/mission7_plate_bench_v2.py run_crossing(repo, cache, spec, "clocks2"), imported and called unchanged (spawn,
# 3.0 s settle, walking/standstill entries, stage law, polling, falls, clear definition, scoring); no frozen source is
# edited.  A READ-ONLY recorder wraps bhl_robust.mission.approach_debug.physical_sample for the crossing (run_crossing
# imports it function-locally, once per policy step) and logs per policy step: each foot's ankle-roll link and sole box
# (world position, toe/heel along, toe_to_edge = the most advanced point of the box's outline (12 edges, 41 points each)
# relative to the plate's near edge at that point's lateral offset, toe and sole height, inside_plate), per-foot
# contacts (normal force, contact normal,
# penetration) bucketed as floor, target plate top, target plate edge face, other plates, walls, own body, other; base
# along/lateral, commanded (vx, vy, wz) vs measured body velocity and yaw rate, the stage phase, wall contact.  The smoke
# checks that a wrapped crossing's bench record is byte-identical to an unwrapped one.
#
# Layouts: exploration train layouts 170-249 only (never bench crossings, train layouts 0-31 or validation layouts; at
# most 80 crossings): the crossings bench v2's construction yields there (grid labels; walking crossings dropped exactly
# as mission7_plate_bench_v2.dropped_crossings() drops them).
# CORRECTION (coordinator decision 2026-10-03, recorded in the ledger as a correction): 250-289 as first frozen does not
# exist beyond 255; corrected to 170-249 before any diagnosis episode.  (bhl_robust.mission.layout.SPLITS["train"] =
# (0, 256), so 250-289 would have yielded only 6 standstill crossings, L250-L255 d1.)  No Mission 7 crossing record exists
# for any layout in 170-249 (checked before any diagnosis episode; other workstreams' smokes are kept to layouts 32-169),
# so no run crossing has a public outcome.
# CAP RULE (deterministic, frozen before any diagnosis episode): the kept crossings are taken in grid order (layout
# ascending, door 0 before door 1) and the first 80 are run; the rest are listed as capped out, never run.  On 170-249:
# 160 grid crossings, 61 walking ones dropped, 99 kept; the run is the first 80, L170 d1 through L234 d0 (65 layouts;
# 0 deg 24, +90 21, -90 19, 180 16; 64 standstill, 16 walking); capped out: L234 d1, L235 d1, L236 d0, L236 d1, L237 d1,
# L238 d1, L239 d1, L240 d0, L241 d0, L242 d0, L242 d1, L243 d0, L244 d0, L245 d0, L246 d0, L247 d0, L248 d1, L249 d0,
# L249 d1.  The instructed smoke layouts 290-299 do not exist: the smoke runs exploration train layouts 32, 33 and 38
# (outside 170-249), which run mode refuses.
#
# Stall (design D = edge_dwell.py of solutions-20260930/campaign-m7-180-gemini/m7-180-analysis, verbatim): along = (base
# xy - plate centre) . door direction; edge dwell = time from the stage's first "cross" entry to the first sample (any
# later phase) with along > -0.20 m; STALL iff edge dwell >= 2.0 s (1e-6 s tolerance) or never past -0.20 m, for any
# crossing with a "cross" entry; no "cross" entry = no_cross, never a stall.  Window W = the "cross" entry to the earliest
# of: the first pass of -0.20 m, the crossing's end (next "turn_back"/"recorded"), the trace's end.
#
# Plate contacts are classified by the contact NORMAL (MuJoCo frame[0:3], sign-corrected to point from the plate into the
# foot), never by the contact height: box-on-cylinder contacts sink (smoke 21532133, L33 d1 step 283: a foot standing
# mid-plate had one contact at z 0.0175 and its lowest corner at z 0.005).
#
# FROZEN THRESHOLDS (re-frozen 2026-10-03 per the coordinator re-freeze; corrected 2026-10-04 10:45; before any episode on 170-249)
#   STALL_DWELL_S = 2.0                 design D: edge dwell >= 2.0 s is a stall (1e-6 s tolerance)
#   ONPLATE_ALONG_M = -0.2              design D / edge_dwell.py: past the edge = base along > -0.20 m
#   TOP_NORMAL_MIN_Z = 0.7071           a plate contact is TOP (support) iff its normal, from the plate into the foot, has
#                                       world z >= 0.7071 (within 45 deg of vertical); otherwise it is the EDGE FACE
#   SUPPORT_MIN_N = 8.0                 a foot is loaded iff floor + plate-top + other-plate-top normal force >= 8.0 N (5% of
#                                       the robot's 160.2 N weight: 16.33 kg in the compiled model)
#   SWING_MIN_STEPS = 2                 a swing = >= 2 consecutive unloaded policy steps (>= 0.08 s); its touchdown = the next
#                                       loaded step
#   LEADING_TOL_M = 0.01                a swing is the LEADING foot's iff at touchdown its toe along >= the other foot's toe
#                                       along - 0.01 m (at or ahead of it)
#   EDGE_FORCE_MIN_N = 1.0              obstruction (a): target-plate edge-face normal force >= 1.0 N at any step from lift-off
#                                       to touchdown (1 N = the bench's own plate-press force threshold)
#   EDGE_STOP_BEFORE_M = 0.02           obstruction (b): touchdown toe_to_edge >= -0.02 m ...
#   EDGE_STOP_AFTER_M = 0.01            ... and <= +0.01 m, on the floor side: floor force >= 8.0 N and target-plate-top force
#                                       < 8.0 N at touchdown
#   BLOCKED_MIN_SWINGS = 3              blocked step-up iff >= 3 blocked swings (leading AND obstructed) touch down inside W
#   GAIT_PERIOD_STEPS = 20              one clock-s2 gait period (0.8 s) in policy steps (0.04 s)
#   SLOW_MIN_PERIODS = 2                slow progress needs >= 2 full gait periods in W
#   ADVANCE_MIN_PER_PERIOD_M = 0.004    a period advances iff base along gains >= 0.004 m over it (differences one gait
#                                       period apart cancel the gait's fore-aft sway)
#   ADVANCING_PERIOD_FRACTION = 0.75    keeps advancing iff >= 75% of W's full periods advance
#   SLOW_MAX_FRACTION = 0.5             too slowly iff mean along speed over W < 0.5 x mean commanded vx over W
#   WALL_DOMINANT_FRACTION = 0.5        wall-blocked iff >= 50% of W's policy steps have a wall/door/goal-post contact (the
#                                       bench sample's own "contacts" field)
#   PRECEDENCE = wall_blocked > blocked_step_up > slow_progress > other   (coordinator re-freeze; conservative for F1)
#   RULES (coordinator re-freeze 2026-10-03 11:40; the pure functions upward_normal, contact_key, is_loaded, is_leading,
#   is_blocked_swing, stall_criteria, classify, decide and decision_lines encode them; the tests check each):
#   - Plate contacts are classified by the contact NORMAL only: MuJoCo's frame[0:3] points from geom1 to geom2, so it is
#     negated when the plate is geom2; n_up_z = the world z of the normal from the plate into the foot; TOP iff n_up_z
#     >= 0.7071, else EDGE FACE; all TOP force (target and other plates) is support; contact heights are never used.
#   - A BLOCKED swing is a swing of the LEADING foot (toe along = its sole box's most advanced corner along the door
#     direction; at touchdown >= the other foot's toe along - 0.01 m) with obstruction evidence (a) or (b) above AND a
#     FLOOR-SIDE touchdown: floor force >= 8.0 N and target-plate-top force < 8.0 N at touchdown, i.e. the foot did not
#     get over the edge (coordinator correction 2026-10-04 10:45: a swing that brushes the edge face and lands on the
#     plate top got over; the smoke audit found 4 of 11 such swings); no swing-height clause (clock-s2's flat-ground
#     swings are themselves low; heights are reported only).
#   - Each stall gets one class on its window W by the precedence wall_blocked > blocked_step_up > slow_progress > other;
#     every criterion and the overlaps are reported.
#   - Falls: a crossing that fell (bench v2: tilt >= 0.78 in any phase) is excluded from the stall count and reported
#     separately; the blocked-step-up fraction is printed both ways (non-fall; including fall stalls); the decision uses
#     the non-fall stalls.
#   - SPECIFICITY: among the crossings that CLEAR (bench v2's clear), the share whose own window W (their own edge dwell)
#     meets the blocked-step-up criterion (>= 3 blocked swings, regardless of precedence); printed on the decision line.
#   - F1 is TRAINED iff non-fall stalls > 0 AND 3 x blocked >= non-fall stalls AND clears > 0 AND meeting clears x non-fall
#     stalls < blocked x clears (exact integers: the non-fall fraction >= 1/3 and the clear share strictly lower than
#     it).  0 non-fall stalls -> F1 NOT trained; 0 clears -> the specificity check cannot hold -> F1 NOT trained.  The
#     decision line always prints the non-fall fraction and the clear share, never "undefined" or a True/False flag.
#   - A partial or aborted run (a declared crossing missing or failed, an imported source changed during the run, the
#     module failing, the job terminated) prints INCOMPLETE and decides nothing; so does a log without a DECISION line.
#   - Smoke check (machinery only): a foot standing inside a ROUND target plate (all 8 sole corners inside the footprint
#     shrunk by 0.01 m, >= 8.0 N of target-plate force) reads as loaded in >= 95% of >= 10 such foot-steps.
# END FROZEN THRESHOLDS
#
# Classes (on the stall's window W, by the frozen precedence; every criterion and the overlaps are reported):
#   wall_blocked: wall/door/goal-post contact in >= 50% of W's steps | blocked_step_up: >= 3 blocked swings (the LEADING
#   foot's, with obstruction (a) or (b)) touch down in W | slow_progress: >= 2 full gait periods in W, >= 75% advance
#   >= 0.004 m, mean along speed < 0.5 x mean commanded vx (> 0) | other: else.
#
# DECISION: the frozen block's RULES above (falls, specificity, the F1 rule and its zero cases, INCOMPLETE); this job
# prints the decision, the coordinator acts on it.  Printed lines (run mode; the smoke prints the same lines with the
# label M7_CROSSING_DIAG_SMOKE, each tagged "(SMOKE: machinery only, never a decision)"):
#   M7_CROSSING_DIAG STALLS non-fall <n> (fall stalls <k>, excluded; crossings that fell <f>) class_counts_non_fall {...}
#   M7_CROSSING_DIAG BLOCKED_STEP_UP_FRACTION non-fall <b>/<n> = <v>; including fall stalls <b2>/<n2> = <v2>
#   M7_CROSSING_DIAG SPECIFICITY clears meeting the blocked-step-up criterion on their own edge dwell <m>/<c> = <v>
#   M7_CROSSING_DIAG DECISION F1 TRAINED (<reason>); non-fall blocked-step-up fraction <b>/<n> = <v>; clear share
#     meeting the criterion <m>/<c> = <v>      (or: DECISION F1 NOT TRAINED (<reason>); the same two numbers)
#   M7_CROSSING_DIAG DECISION INCOMPLETE (<reason>; decides nothing)
#   (a 0 denominator prints "<x>/0 (none)"; no line prints "undefined" or a True/False flag)
# Reading: only a "M7_CROSSING_DIAG DECISION F1 TRAINED" or "M7_CROSSING_DIAG DECISION F1 NOT TRAINED" line followed by
# "M7_CROSSING_DIAG_RUN_COMPLETE" (job exit 0) decides.  "DECISION INCOMPLETE", a non-zero exit, or a log without a
# DECISION line (e.g. the job was killed) decides nothing.  summary.json "decision" holds the same numbers.
# Output: summary.json (per-crossing rows: edge dwell, clear time, criteria and features on W and on the first 2.0 s,
# classes, swings; counts, overlaps, the decision; sources' sha256), report.md, provenance.json, crossings/L###-d#.json.gz
# (the bench record + the per-step diagnosis records).
#
# Sources: the module records the sha256 of the clocks2 bench v2's 41 sources, this workstream's three files and every
# imported repo module (sys.modules) plus the imported upstream modules, and compares them with
# results/mission7-campaign-20260923/clocks2-plate-bench-v2/provenance.json; it REFUSES (before any episode) when an
# imported source differs from the bench's, unless M7_DIAG_ALLOW_SOURCE_DRIFT=1 (then the drift is recorded).  Both
# policies' sha256 must equal the bench's (clock-s2 c1862f1e..., checked by the stage's own pinned export check).
# Run from a worktree of the bench's code: REPO = $M7_DIAG_REPO, else the checkout holding this launcher, else the
# script Slurm recorded (scontrol Command=, as sbatch runs a spooled copy); it fails closed unless REPO holds the module.
#
# Disclosed: the clocks2 bench v2 ran on haswell&el8 (cn-c21); this launcher runs on el9 nodes as instructed, so
# trajectories are not bit-comparable across CPU types (the layouts differ anyway).  The stage gait is LEARNED clock-s2,
# the stage SCRIPTED, layout and plate pose ORACLE.
#
# Smoke (machinery only, never a diagnosis result, never a reserved or scored layout; exits non-zero on any failure):
# preflight; then L32 d0 (0 deg, round, standstill) run unwrapped and wrapped (identity: byte-identical bench records),
# L33 d1 (+90 deg, round, walking) and L38 d0 (-90 deg, square, standstill) wrapped; machinery checks (records 1:1 with
# the bench samples, the feet carry the robot in the settle, floor-loaded soles at z ~ 0, floor contact normals point up,
# every foot's force buckets recount from its contacts by normal, a foot standing inside a round plate reads as loaded
# (>= 10 foot-steps, >= 95% loaded), finite foot metrics, swings found on both feet, clock-s2 swapped in at takeover,
# sources identical to the bench's, the model's weight); leading-foot swing statistics are reported; summary, report and
# the decision lines (tagged SMOKE) are written as in run mode.
#
#   usage (from the repository or worktree root; M7_DIAG_REPO names the checkout explicitly):
#     M7_DIAG_REPO=<checkout> sbatch --export=ALL --job-name=m7-crossing-diag-smoke --time=00:30:00 <checkout>/slurm/repo20260923/cpu_m7_crossing_diag.sbatch smoke
#     M7_DIAG_REPO=<checkout> sbatch --export=ALL <checkout>/slurm/repo20260923/cpu_m7_crossing_diag.sbatch run
#   outputs: run -> ${M7_DIAG_OUT:-<checkout>/results/mission7-campaign-20260923/crossing-diag-20261003}/; smoke ->
#   ${M7_DIAG_SMOKE_DIR:-<checkout>/results/mission7-campaign-20260923/crossing-diag-20261003-smoke/job-<id>}/.  Both refuse
#   an existing output directory.  Run mode refuses unless this workstream's three files are committed and unmodified in
#   REPO; M7_DIAG_DRY_RUN=1 stops it after the preflight.  The smoke refuses (exit 2) unless the job name contains
#   "smoke" and the job's own time limit is <= 1:00:00.
set -euo pipefail
mode=${1:?usage: cpu_m7_crossing_diag.sbatch run|smoke}
PY=${PYTHON:-/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python}
MODULE=scripts/bench/m7_crossing_diag.py
CAMPAIGN=results/mission7-campaign-20260923
OWN_FILES=(scripts/bench/m7_crossing_diag.py slurm/repo20260923/cpu_m7_crossing_diag.sbatch tests/test_m7_crossing_diag.py)
DRY_RUN=${M7_DIAG_DRY_RUN:-0}

# REPO: $M7_DIAG_REPO, else the checkout this launcher sits in (bash, srun), else the script Slurm recorded.
resolve_repo() {
  local candidate="" cmd=""
  if [ -n "${M7_DIAG_REPO:-}" ]; then
    candidate=$M7_DIAG_REPO
  else
    candidate=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd -P) || candidate=""
    if [ -z "$candidate" ] || [ ! -f "$candidate/$MODULE" ]; then
      candidate=""
      if [ -n "${SLURM_JOB_ID:-}" ]; then
        cmd=$(scontrol show job -o "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/.* Command=\([^ ]*\).*/\1/p' | head -n 1) || cmd=""
        if [ -n "$cmd" ]; then
          candidate=$(cd "$(dirname "$cmd")/../.." 2>/dev/null && pwd -P) || candidate=""
        fi
      fi
    fi
  fi
  if [ -z "$candidate" ] || [ ! -f "$candidate/$MODULE" ]; then
    return 1
  fi
  printf '%s\n' "$candidate"
}
REPO=$(resolve_repo) || { echo "refusing: no checkout holding $MODULE found (set M7_DIAG_REPO)" >&2; exit 2; }
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/scripts" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
[ -d "${TMPDIR:-/nonexistent}" ] || export TMPDIR=/tmp
head=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) || head=unknown
echo "m7 crossing diagnosis: mode=$mode repo=$REPO head=$head host=$(hostname) job=${SLURM_JOB_ID:-local} dry_run=$DRY_RUN"

if [ "$mode" = run ]; then
  # An aborted run decides nothing: a terminated job (time limit, scancel; from here on, the preflight included) or a
  # failing module prints INCOMPLETE.
  trap 'echo "M7_CROSSING_DIAG DECISION INCOMPLETE (the job was terminated; decides nothing)"; exit 143' TERM
  OUT=${M7_DIAG_OUT:-$REPO/$CAMPAIGN/crossing-diag-20261003}
  [ ! -e "$OUT" ] || { echo "refusing: $OUT exists (a diagnosis result is never overwritten)" >&2; exit 2; }
  for f in "${OWN_FILES[@]}"; do
    git -C "$REPO" ls-files --error-unmatch "$f" >/dev/null 2>&1 || { echo "refusing: $f is not committed in $REPO" >&2; exit 2; }
  done
  if ! git -C "$REPO" diff --quiet HEAD -- "${OWN_FILES[@]}"; then
    echo "refusing: this workstream's files differ from HEAD in $REPO; commit them first" >&2; exit 2
  fi
  "$PY" "$MODULE" --repo "$REPO" --out "$OUT" --mode run --preflight
  if [ "$DRY_RUN" != 0 ]; then
    echo "DRY_RUN: stopping before any episode; would run: $PY $MODULE --repo $REPO --out $OUT --mode run"
    exit 0
  fi
  rc=0
  "$PY" "$MODULE" --repo "$REPO" --out "$OUT" --mode run || rc=$?
  if [ "$rc" != 0 ]; then
    echo "M7_CROSSING_DIAG DECISION INCOMPLETE (the diagnosis exited $rc; decides nothing)"
    exit "$rc"
  fi
  echo "M7_CROSSING_DIAG_RUN_COMPLETE -> $OUT"
  exit 0
fi

[ "$mode" = smoke ] || { echo "unknown mode $mode (run|smoke)" >&2; exit 2; }

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
SMOKE=${M7_DIAG_SMOKE_DIR:-$REPO/$CAMPAIGN/crossing-diag-20261003-smoke/job-${SLURM_JOB_ID:-local}}
case "$SMOKE" in *-smoke/*|*-smoke) ;; *) echo "refusing: smoke output must sit under a *-smoke directory" >&2; exit 2 ;; esac
[ ! -e "$SMOKE" ] || { echo "refusing: $SMOKE exists" >&2; exit 2; }
start=$(date +%s)
"$PY" "$MODULE" --repo "$REPO" --out "$SMOKE" --mode smoke --preflight
"$PY" "$MODULE" --repo "$REPO" --out "$SMOKE" --mode smoke
echo "M7_CROSSING_DIAG_SMOKE_PASS (machinery only; exploration layouts 32, 33, 38; not a diagnosis result; $(( $(date +%s) - start )) s)"
