#!/bin/bash
# Mission 7 crossing: queue the un-submitted PlateStage variants through the exact
# ten-fall replay gate (2026-09-24), one CPU job each, sequential on cn-c22.
#
#   usage (from the repo root):
#     bash slurm/repo20260923/submit_m7_replay_gates.sh            # dry run: preflight + plan, submits nothing
#     bash slurm/repo20260923/submit_m7_replay_gates.sh --submit   # queue the chain
#   options: --only tag[,tag]   restrict to some arms      --max-route-gates N   route-gate slots (default 1)
#            --after JOBID      chain the first queued gate after an already-queued job (resume a partial
#                               chain without oversubscribing cn-c22)
#
# Every gate goes through the EXISTING snapshot submitter
# (scripts/submit_mission7_plate_stage.py: source snapshot + sha256 manifest +
# in-job --preflight + verbatim pass-through to scripts/mission7_plate_stage.py),
# pinned to --nodelist=cn-c22 because the replay is bitwise.  The submitter has
# no --dependency flag and sbatch (25.05.9) has NO dependency input variable
# (its SBATCH_* table stops at WCKEY; SLURM_JOB_DEPENDENCY is output only), so
# the chain is expressed through a per-gate `sbatch` shim placed first on PATH
# for the submitter's call only: it execs the real sbatch with
# --dependency=afterany:<previous gate> prepended (the submitter's clean_env
# keeps PATH; it strips only SLURM_*, TMPDIR and CUDA_VISIBLE_DEVICES).  The
# applied dependency is then verified with squeue; if it is missing the gate is
# cancelled and the chain stops.  afterany, not afterok: a failed or incomplete
# gate still releases the next one; only the node is serialized.  Each gate is
# followed by slurm/repo20260923/m7_replay_gate_followup.sbatch
# (--dependency=afterany:<gate>, share/haswell&el8, 1 CPU, 15 min), which writes
# the compact summary and applies the predeclared route-gate rule below.
#
# Recovered parameter sets (SLURM_JOBS.md 2685-2702, git 2ec87f5 / 58417a7,
# results/mission7-campaign-20260923/local-stage-v2/replay-v{3,4}):
#   V2  = --pre-point=0.45 --cross-clear=0.35 --cross-kick, settle 0.40
#         -> 7/10 on cn-c22 (21401689), layouts 4, 5, 7 fell crossing sideways
#   V3  = V2 + --align-yaw as implemented in 2ec87f5: an IN-PLACE yaw turn during
#         the settle (tol 0.20 rad, bounded 1.5 s) -> local 8/10 (7 and 14 fell;
#         the frozen gait produced no rotation, so on the misaligned layouts V3
#         behaved as V2 with up to 1.5 s more standstill before the kick).
#         That code path was replaced in 58417a7 and is NOT reproducible from
#         the HEAD snapshot (the stage code is not edited here).
#   V4  = V2 + --align-yaw as implemented at HEAD: the yaw term applied while
#         moving, during the approach and the crossing -> local 9/10 (7 fell).
# So at HEAD "V2 + align_yaw" IS the historical V4, and the requested matrix
# {V2+align, V3, V4, V3+align, V4+align} is defined explicitly from the documented
# failure modes (sideways crossing -> --align-yaw; standstill fixed point before
# the kick -> --settle-s), five distinct arms, tags named by their parameters:
ARMS=(
  # tag                  flags forwarded to scripts/submit_mission7_plate_stage.py
  "v2-align              --pre-point 0.45 --cross-clear 0.35 --cross-kick --align-yaw"                  # V2 + align_yaw (= historical V4, local 9/10): queued first
  "v3-settle190          --pre-point 0.45 --cross-clear 0.35 --cross-kick --settle-s 1.90"              # V3 stand-in: V2 with the settle lengthened to 0.40 + 1.5 s, the standstill historical V3 actually produced
  "v4-settle080          --pre-point 0.45 --cross-clear 0.35 --cross-kick --settle-s 0.80"              # V4 (task fallback): V2 with a longer settle, modest step
  "v3-settle190-align    --pre-point 0.45 --cross-clear 0.35 --cross-kick --settle-s 1.90 --align-yaw"  # V3 + align_yaw
  "v4-settle080-align    --pre-point 0.45 --cross-clear 0.35 --cross-kick --settle-s 0.80 --align-yaw"  # V4 + align_yaw
  # M2 (docs/SOLUTIONS_2026-10-01.md section 3; 2026-10-02): every stage flag at its replay default, the stage
  # running on TurnBoth-s0 (turn in place, straight crossing, turn back).  Released ONLY by a bench PASS:
  # slurm/repo20260923/cpu_m7_plate_bench.sbatch runs `$0 --only m2-turnboth --submit` after reading its
  # verdict.json.  Enforced here too (m2_bench_gate, --submit only): the arm is refused unless
  # $M2_BENCH_VERDICT is a scored bench PASS and every source the replay snapshot copies is byte-identical to
  # the bench's provenance.json record.  The five arms above ran on 2026-09-24 and their destinations exist,
  # so a run without --only stops at the destination check.
  "m2-turnboth           --stage-gait turnboth"
  # M3 (2026-10-02; conditional on M2's bench FAIL): every stage flag at its replay default, the stage on the SHIPPED
  # gait (turn while stepping, straight crossing, turn back, stall watchdog).  Released ONLY by a bench v2 PASS:
  # slurm/repo20260923/cpu_m7_plate_bench_v2.sbatch runs `$0 --only m3-shipped-step --submit` after reading its
  # verdict.json.  Enforced here too (m3_bench_gate, --submit only): refused unless $M3_BENCH_VERDICT is a scored
  # bench-v2 PASS with --stage-gait m3 and every source the replay snapshot copies matches its provenance.json.
  "m3-shipped-step       --stage-gait m3"
  # --- m7-clocks2 --- (2026-10-02; SLURM_JOBS.md "User approval recorded 2026-10-02 14:15", item A): every stage flag
  # at its replay default, the stage on arms-turngait-clock-s2's CONTROLLER swapped in at takeover (M2's turnboth law).
  # Released ONLY by a bench v2 PASS with --stage-gait clocks2: slurm/repo20260923/cpu_m7_plate_bench_v2_clocks2.sbatch
  # runs `$0 --only m7-clocks2 --submit` after reading its verdict.json.  Enforced here too (clocks2_bench_gate,
  # --submit only): refused unless $CLOCKS2_BENCH_VERDICT is a scored bench-v2 clocks2 PASS whose provenance.json
  # records the pinned clock-s2 weights, the export on disk still has them, and every source the replay snapshot copies
  # (src/bhl_robust/eval/gait_clock.py included) matches its record.
  "m7-clocks2            --stage-gait clocks2"
  # --- m7-fix --- (2026-10-03; SLURM_JOBS.md "User approval recorded 2026-10-03 09:55", item F, parts F2 + F3): the stage
  # on an export stage gait's CONTROLLER (clock-s2 by default; the export the fix bench ran when it ran one, passed as
  # M7_FIX_STAGE_GAIT_EXPORT + M7_FIX_STAGE_GAIT_EXPORT_SHA256, which rewrite this entry below) with --cross-budget window
  # (F2) and --yaw-cap 0.6 (F3), every other stage flag at its replay default.  Released ONLY by a bench v2 PASS with
  # the same gait and options: slurm/repo20260923/cpu_m7_plate_bench_v2_fix.sbatch runs `$0 --only m7-fix --submit`
  # after reading its verdict.json.  Enforced here too (fix_bench_gate, --submit only): refused unless
  # $FIX_BENCH_VERDICT is a scored bench-v2 PASS with that gait whose verdict.json and provenance.json record both
  # options, whose provenance records the pinned stage-gait weights (and, for an export, that export directory), the
  # policy on disk still has them, and every source the replay snapshot copies (gait_clock.py and env.py included)
  # matches its record.
  "m7-fix                --stage-gait clocks2 --cross-budget window --yaw-cap 0.6"
)
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
CAMPAIGN=results/mission7-campaign-20260923
SUBMITTER=scripts/submit_mission7_plate_stage.py
FOLLOWUP=slurm/repo20260923/m7_replay_gate_followup.sbatch
DEFAULT_SOURCE_CAMPAIGN=results/mission7-replay-smoke-20260921
DEFAULT_BASELINE=results/mission7-approach-followup-20260922/replay-diagnose-cn-c22
# The scored M2 bench verdict that alone releases m2-turnboth (overridable for tests; the bench job passes its own).
M2_BENCH_VERDICT=${M7_BENCH_VERDICT:-$CAMPAIGN/m2-plate-bench/verdict.json}
# The scored bench-v2 verdict (--stage-gait m3) that alone releases m3-shipped-step (the bench-v2 job passes its own).
M3_BENCH_VERDICT=${M7_BENCH_V2_VERDICT:-$CAMPAIGN/m3-plate-bench-v2/verdict.json}
# --- m7-clocks2 --- the scored bench-v2 verdict (--stage-gait clocks2) that alone releases m7-clocks2 (the clocks2
# bench job passes its own), and the clock-s2 weights that bench must have run (= mission7_plate_stage.CLOCKS2_*).
CLOCKS2_BENCH_VERDICT=${M7_CLOCKS2_BENCH_VERDICT:-$CAMPAIGN/clocks2-plate-bench-v2/verdict.json}
CLOCKS2_POLICY=external/Berkeley-Humanoid-Lite/logs/rsl_rl/humanoid/2026-10-02_03-58-04_arms-turngait-clock-s2/exported/policy.onnx
CLOCKS2_POLICY_SHA256=c1862f1e4429c1e6ac2e4fae1391f3f40ccc353382c8bbf4b88efee4c6c8d8de
# --- m7-fix --- the scored fix bench-v2 verdict that alone releases m7-fix (the fix bench job passes its own), and the
# stage gait that bench ran: the clock-s2 preset, or the export M7_FIX_STAGE_GAIT_EXPORT pinned to
# M7_FIX_STAGE_GAIT_EXPORT_SHA256 (both or neither).
FIX_BENCH_VERDICT=${M7_FIX_BENCH_VERDICT:-$CAMPAIGN/fix-plate-bench-v2/verdict.json}
FIX_EXPORT=${M7_FIX_STAGE_GAIT_EXPORT:-}
FIX_EXPORT_SHA256=${M7_FIX_STAGE_GAIT_EXPORT_SHA256:-}
cd "$REPO"
if [ -n "$FIX_EXPORT" ] || [ -n "$FIX_EXPORT_SHA256" ]; then
  [ -n "$FIX_EXPORT" ] && [ -n "$FIX_EXPORT_SHA256" ] || { echo "M7_FIX_STAGE_GAIT_EXPORT and M7_FIX_STAGE_GAIT_EXPORT_SHA256 go together" >&2; exit 2; }
  [[ "$FIX_EXPORT_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "M7_FIX_STAGE_GAIT_EXPORT_SHA256 is not a sha256: $FIX_EXPORT_SHA256" >&2; exit 2; }
  case "$FIX_EXPORT" in *[[:space:]]*) echo "M7_FIX_STAGE_GAIT_EXPORT may not contain whitespace: '$FIX_EXPORT'" >&2; exit 2 ;; esac
  [ -f "$FIX_EXPORT/policy.onnx" ] || { echo "no policy.onnx in M7_FIX_STAGE_GAIT_EXPORT=$FIX_EXPORT" >&2; exit 2; }
  for i in "${!ARMS[@]}"; do
    if [ "${ARMS[$i]%% *}" = m7-fix ]; then
      ARMS[$i]="m7-fix                --stage-gait export --stage-gait-export $FIX_EXPORT --stage-gait-export-sha256 $FIX_EXPORT_SHA256 --cross-budget window --yaw-cap 0.6"
    fi
  done
  FIX_GAIT=export; FIX_POLICY=$FIX_EXPORT/policy.onnx; FIX_POLICY_SHA256=$FIX_EXPORT_SHA256
else
  FIX_GAIT=clocks2; FIX_POLICY=$CLOCKS2_POLICY; FIX_POLICY_SHA256=$CLOCKS2_POLICY_SHA256
fi

submit=0; only=""; max_route=1; after=""
while [ $# -gt 0 ]; do
  case "$1" in
    --submit) submit=1 ;;
    --only) only=$2; shift ;;
    --max-route-gates) max_route=$2; shift ;;
    --after) after=$2; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done

rule() {
cat <<RULE
================================================================================
PREDECLARED RULE (Mission 7 crossing, exact replay gates -> route gate)
  1. Each arm replays the ten retained falls on cn-c22 with the recorded
     activation schedule, unchanged geometry and fall predicate (the 10/10 gate
     of 21397732).  Verdict per arm, computed from result.json by the follow-up:
       PASS        complete, 10 episodes, 10 upright, 0 falls, exact_replay_gate_passed
       FAIL        complete with any fall
       INCOMPLETE  anything else (crash, timeout, missing result.json)
     written to $CAMPAIGN/replay-gate-<tag>/summary.json and rolled up in
     $CAMPAIGN/replay-gate-matrix-summary.json.
  2. ONLY a 10/10 PASS may proceed to the route gate.  8/10, 9/10 and any
     INCOMPLETE do not, and nothing here lowers that bar.
  3. The route gate is Doors 16 + Transport 16 (validation layouts 0-15, two
     jobs, scripts/submit_mission7_route_handoff.py, early handoff,
     --constraint=haswell&el8, 2 CPUs / 12 GB / 2 h each).  Its stage flags are
     read from the passing gate's own submission.json probe_args, plus
     --stage-wait-open=0.0 (the replay gates run at wait-open 0; the route probe
     defaults to 2.0), and the fixed route-side composition of the only
     end-to-end Doors success (local-stage-v2/doors-1-v3-center):
     --stage-activate --rejoin-advance --exit-ramp-center --exit-ramp 1.2
     --rejoin-fix none.  Route-side flags are outside the replay by
     construction and are not tuned to gate results.
     Route-gate criterion (docs/MISSION7_TASKS.md): Doors >= 16/16 AND
     Transport >= 16/16.  Anything below stays FAIL; every episode counts.
  4. Episode envelope.  2026-09-24 arms (docs/MISSION7_TASKS.md: 101 of 512
     unspent; unreconciled with the ledger, set aside by the user 2026-10-01):
     five gates = 50, one route gate = 32.  m2-turnboth: the 106-episode line
     authorized 2026-10-01 (64 bench + 10 this replay + 32 route gate), released
     only by a bench PASS (cpu_m7_plate_bench.sbatch).  m3-shipped-step: M3's
     line of the same size (conditional on M2's bench FAIL; bench v2 runs the
     64 grid crossings minus its geometry-dropped walking entries + 10 this
     replay + 32 route gate), released only by a bench-v2 PASS with --stage-gait
     m3 (cpu_m7_plate_bench_v2.sbatch).  m7-clocks2: the 83-episode line
     approved 2026-10-02 (41 bench v2 + 10 this replay + 32 route gate),
     released only by a bench-v2 PASS with --stage-gait clocks2
     (cpu_m7_plate_bench_v2_clocks2.sbatch).  m7-fix: the 83-episode line
     approved 2026-10-03 (41 bench v2 + 10 this replay + 32 route gate), the
     stage gait the fix bench ran with --cross-budget window --yaw-cap 0.6,
     released only by that bench's PASS (cpu_m7_plate_bench_v2_fix.sbatch).
     The route gate is
     released for the FIRST 10/10 in chain order only (route-gate slots:
     $max_route); later passes are recorded PASS and their route-gate commands
     printed as ROUTE_GATE_DEFERRED for explicit re-authorization.
     A layout whose recorded trace ends before the stage reaches \`cross\` is
     upright trivially; summary.json therefore also reports crossings_started
     and crossings_completed, and a 10/10 is reported together with them.
  5. Chain order: ${selected[*]%% *} -- v2-align first because it is the only arm
     with a local 9/10 behind it; the reference V2 (7/10, 21401689) is not rerun.
================================================================================
RULE
}

# M2 bench gate, applied to m2-turnboth with --submit only: exit 3 (RELEASE_BLOCKED) unless $1 is a scored bench
# PASS and every file in $2.. (the sources the replay snapshot will copy) has the sha256 the bench recorded in the
# provenance.json beside that verdict, so the replay runs the code the bench validated.
m2_bench_gate() {
  "$PY" - "$@" <<'PYEOF' || return 3
import hashlib, json, sys
from pathlib import Path


def blocked(why):
    print(f"RELEASE_BLOCKED m2-turnboth: {why}", file=sys.stderr)
    sys.exit(3)


verdict_path, files = Path(sys.argv[1]), sys.argv[2:]
if not verdict_path.is_file():
    blocked(f"no bench verdict at {verdict_path}")
try:
    verdict = json.loads(verdict_path.read_text())
    recorded = json.loads((verdict_path.parent / "provenance.json").read_text())["sources_sha256"]
    if not isinstance(verdict, dict) or not isinstance(recorded, dict):
        raise TypeError("verdict or sources_sha256 is not a JSON object")
except (OSError, ValueError, KeyError, TypeError) as exc:
    blocked(f"unreadable bench verdict or provenance beside {verdict_path}: {type(exc).__name__}: {exc}")
if verdict.get("mode") != "scored" or verdict.get("verdict") != "PASS":
    blocked(f"{verdict_path} is mode {verdict.get('mode')!r}, verdict {verdict.get('verdict')!r}; only a scored PASS releases")
if not files:
    blocked("no snapshot sources to compare")
changed = [name for name in files if name not in recorded or not Path(name).is_file()
           or hashlib.sha256(Path(name).read_bytes()).hexdigest() != recorded[name]]
if changed:
    blocked(f"sources changed since the bench (or absent from its provenance): {changed}")
print(f"m2 bench gate: scored PASS at {verdict_path}; {len(files)} snapshot sources identical to the bench's record")
PYEOF
}

# M3 bench-v2 gate, applied to m3-shipped-step with --submit only: exit 3 (RELEASE_BLOCKED) unless $1 is a scored
# bench-v2 PASS run with --stage-gait m3 and every file in $2.. has the sha256 recorded in the provenance.json beside it.
m3_bench_gate() {
  "$PY" - "$@" <<'PYEOF' || return 3
import hashlib, json, sys
from pathlib import Path


def blocked(why):
    print(f"RELEASE_BLOCKED m3-shipped-step: {why}", file=sys.stderr)
    sys.exit(3)


verdict_path, files = Path(sys.argv[1]), sys.argv[2:]
if not verdict_path.is_file():
    blocked(f"no bench-v2 verdict at {verdict_path}")
try:
    verdict = json.loads(verdict_path.read_text())
    provenance = json.loads((verdict_path.parent / "provenance.json").read_text())
    recorded = provenance["sources_sha256"]
    if not isinstance(verdict, dict) or not isinstance(recorded, dict):
        raise TypeError("verdict or sources_sha256 is not a JSON object")
except (OSError, ValueError, KeyError, TypeError) as exc:
    blocked(f"unreadable bench-v2 verdict or provenance beside {verdict_path}: {type(exc).__name__}: {exc}")
if (verdict.get("bench"), verdict.get("mode"), verdict.get("stage_gait"), verdict.get("verdict")) != ("v2", "scored", "m3", "PASS") \
        or (provenance.get("bench"), provenance.get("mode"), provenance.get("stage_gait")) != ("v2", "scored", "m3"):
    blocked(f"{verdict_path} is bench {verdict.get('bench')!r}, mode {verdict.get('mode')!r}, stage gait "
            f"{verdict.get('stage_gait')!r}, verdict {verdict.get('verdict')!r}; only a scored bench-v2 m3 PASS releases")
if not files:
    blocked("no snapshot sources to compare")
changed = [name for name in files if name not in recorded or not Path(name).is_file()
           or hashlib.sha256(Path(name).read_bytes()).hexdigest() != recorded[name]]
if changed:
    blocked(f"sources changed since the bench (or absent from its provenance): {changed}")
print(f"m3 bench gate: scored bench-v2 m3 PASS at {verdict_path}; {len(files)} snapshot sources identical to the bench's record")
PYEOF
}

# --- m7-clocks2 --- bench-v2 gate, applied to m7-clocks2 with --submit only: exit 3 (RELEASE_BLOCKED) unless $1 is a
# scored bench-v2 PASS run with --stage-gait clocks2, its provenance.json records the clock-s2 policy sha256 $2, the
# policy file $3 still has it, and every file in $4.. (the snapshot sources, gait_clock.py among them) has the sha256
# recorded there.
clocks2_bench_gate() {
  "$PY" - "$@" <<'PYEOF' || return 3
import hashlib, json, sys
from pathlib import Path


def blocked(why):
    print(f"RELEASE_BLOCKED m7-clocks2: {why}", file=sys.stderr)
    sys.exit(3)


if len(sys.argv) < 4:
    blocked("usage: clocks2_bench_gate VERDICT PINNED_SHA256 POLICY SOURCES...")
verdict_path, pinned, policy, files = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4:]
if not verdict_path.is_file():
    blocked(f"no bench-v2 clocks2 verdict at {verdict_path}")
try:
    verdict = json.loads(verdict_path.read_text())
    provenance = json.loads((verdict_path.parent / "provenance.json").read_text())
    recorded, weights = provenance["sources_sha256"], provenance["weights_sha256"]
    if not all(isinstance(x, dict) for x in (verdict, recorded, weights)):
        raise TypeError("verdict, sources_sha256 or weights_sha256 is not a JSON object")
except (OSError, ValueError, KeyError, TypeError) as exc:
    blocked(f"unreadable bench-v2 verdict or provenance beside {verdict_path}: {type(exc).__name__}: {exc}")
if (verdict.get("bench"), verdict.get("mode"), verdict.get("stage_gait"), verdict.get("verdict")) != ("v2", "scored", "clocks2", "PASS") \
        or (provenance.get("bench"), provenance.get("mode"), provenance.get("stage_gait")) != ("v2", "scored", "clocks2"):
    blocked(f"{verdict_path} is bench {verdict.get('bench')!r}, mode {verdict.get('mode')!r}, stage gait "
            f"{verdict.get('stage_gait')!r}, verdict {verdict.get('verdict')!r}; only a scored bench-v2 clocks2 PASS releases")
if weights.get("clocks2") != pinned:
    blocked(f"the bench ran clock-s2 weights {weights.get('clocks2')!r}, not the pinned {pinned}")
if not policy.is_file() or hashlib.sha256(policy.read_bytes()).hexdigest() != pinned:
    blocked(f"{policy} is missing or no longer has the pinned sha256 {pinned}")
if not files:
    blocked("no snapshot sources to compare")
if "src/bhl_robust/eval/gait_clock.py" not in files:
    blocked("src/bhl_robust/eval/gait_clock.py (in the clocks2 snapshot) is not among the compared sources")
changed = [name for name in files if name not in recorded or not Path(name).is_file()
           or hashlib.sha256(Path(name).read_bytes()).hexdigest() != recorded[name]]
if changed:
    blocked(f"sources changed since the bench (or absent from its provenance): {changed}")
print(f"clocks2 bench gate: scored bench-v2 clocks2 PASS at {verdict_path}; pinned clock-s2 weights; {len(files)} "
      "snapshot sources identical to the bench's record")
PYEOF
}

# --- m7-fix --- bench-v2 gate, applied to m7-fix with --submit only: exit 3 (RELEASE_BLOCKED) unless $1 is a scored
# bench-v2 PASS run with stage gait $2 (clocks2 | export) whose verdict.json and provenance.json both record
# --cross-budget window --yaw-cap 0.6 (stage_fix), its provenance records the pinned policy sha256 $3 for that gait, the
# policy file $4 still has it, for an export the bench's export directory is $5 (empty for clocks2), and every file in
# $6.. (the snapshot sources, gait_clock.py and env.py among them) has the sha256 recorded there.
fix_bench_gate() {
  "$PY" - "$@" <<'PYEOF' || return 3
import hashlib, json, os, sys
from pathlib import Path

OPTIONS = {"cross_budget": "window", "yaw_cap_rps": 0.6}


def blocked(why):
    print(f"RELEASE_BLOCKED m7-fix: {why}", file=sys.stderr)
    sys.exit(3)


if len(sys.argv) < 6:
    blocked("usage: fix_bench_gate VERDICT GAIT PINNED_SHA256 POLICY EXPORT_DIR SOURCES...")
verdict_path, gait, pinned, policy, export_dir = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]), sys.argv[5]
files = sys.argv[6:]
if gait not in ("clocks2", "export"):
    blocked(f"stage gait {gait!r} is neither clocks2 nor export")
if not verdict_path.is_file():
    blocked(f"no bench-v2 fix verdict at {verdict_path}")
try:
    verdict = json.loads(verdict_path.read_text())
    provenance = json.loads((verdict_path.parent / "provenance.json").read_text())
    recorded, weights = provenance["sources_sha256"], provenance["weights_sha256"]
    if not all(isinstance(x, dict) for x in (verdict, recorded, weights)):
        raise TypeError("verdict, sources_sha256 or weights_sha256 is not a JSON object")
except (OSError, ValueError, KeyError, TypeError) as exc:
    blocked(f"unreadable bench-v2 verdict or provenance beside {verdict_path}: {type(exc).__name__}: {exc}")
if (verdict.get("bench"), verdict.get("mode"), verdict.get("stage_gait"), verdict.get("verdict")) != ("v2", "scored", gait, "PASS") \
        or (provenance.get("bench"), provenance.get("mode"), provenance.get("stage_gait")) != ("v2", "scored", gait):
    blocked(f"{verdict_path} is bench {verdict.get('bench')!r}, mode {verdict.get('mode')!r}, stage gait "
            f"{verdict.get('stage_gait')!r}, verdict {verdict.get('verdict')!r}; only a scored bench-v2 {gait} PASS releases")
for name, record in (("verdict.json", verdict), ("provenance.json", provenance)):
    fix = record.get("stage_fix")
    if not isinstance(fix, dict) or {key: fix.get(key) for key in OPTIONS} != OPTIONS:
        blocked(f"{name} does not record --cross-budget window --yaw-cap 0.6 (stage_fix {fix!r})")
if weights.get(gait) != pinned:
    blocked(f"the bench ran {gait} weights {weights.get(gait)!r}, not the pinned {pinned}")
if not policy.is_file() or hashlib.sha256(policy.read_bytes()).hexdigest() != pinned:
    blocked(f"{policy} is missing or no longer has the pinned sha256 {pinned}")
if gait == "export":
    ran = provenance.get("stage_gait_export")
    if not export_dir or not ran or os.path.realpath(ran) != os.path.realpath(export_dir):
        blocked(f"the bench ran the export {ran!r}, not {export_dir!r}")
    if os.path.realpath(policy) != os.path.realpath(os.path.join(export_dir, "policy.onnx")):
        blocked(f"{policy} is not the policy of the export {export_dir}")
elif export_dir:
    blocked(f"an export directory ({export_dir}) was given for the clocks2 preset")
if not files:
    blocked("no snapshot sources to compare")
for need in ("src/bhl_robust/eval/gait_clock.py", "src/bhl_robust/mission/env.py"):
    if need not in files:
        blocked(f"{need} (in the m7-fix snapshot) is not among the compared sources")
changed = [name for name in files if name not in recorded or not Path(name).is_file()
           or hashlib.sha256(Path(name).read_bytes()).hexdigest() != recorded[name]]
if changed:
    blocked(f"sources changed since the bench (or absent from its provenance): {changed}")
print(f"fix bench gate: scored bench-v2 {gait} PASS with --cross-budget window --yaw-cap 0.6 at {verdict_path}; pinned "
      f"{gait} weights; {len(files)} snapshot sources identical to the bench's record")
PYEOF
}

# ---- pre-checks (both modes) ------------------------------------------------------------------
[ -x "$PY" ] || { echo "venv python missing: $PY" >&2; exit 1; }
[ -f "$DEFAULT_SOURCE_CAMPAIGN/fullroute/legacy-doors.json" ] || { echo "replay source missing" >&2; exit 1; }
[ -f "$DEFAULT_BASELINE/result.json" ] || { echo "authoritative baseline missing" >&2; exit 1; }
bash -n "$FOLLOWUP"
"$PY" -m py_compile "$SUBMITTER" scripts/mission7_plate_stage.py scripts/submit_mission7_route_handoff.py scripts/mission7_gates.py
# The snapshot submitter records `git rev-parse HEAD` in every receipt; a dirty snapshot file would misrepresent it.
snapshot_files=(src/bhl_robust/mission/*.py src/bhl_robust/__init__.py src/bhl_robust/sensor_io.py src/bhl_robust/eval/__init__.py
                src/bhl_robust/eval/multi_robot.py src/bhl_robust/eval/mjcf_assets.py src/bhl_robust/eval/livery.py
                src/bhl_robust/eval/team_sensors.py slurm/mission7_plate_stage.sbatch scripts/mission7*.py)
if ! git diff --quiet -- "${snapshot_files[@]}"; then
  echo "snapshot sources have uncommitted changes; commit or stash before submitting a hash-frozen gate:" >&2
  git status --short -- "${snapshot_files[@]}" >&2; exit 1
fi
# m2_bench_gate compares these: every snapshot source except the replay's own sbatch, which the bench never ran.
m2_gate_files=(); for f in "${snapshot_files[@]}"; do [ "$f" = slurm/mission7_plate_stage.sbatch ] || m2_gate_files+=("$f"); done
selected=()
for arm in "${ARMS[@]}"; do
  tag=${arm%% *}
  if [ -n "$only" ] && ! grep -qx "$tag" <<<"${only//,/$'\n'}"; then continue; fi
  selected+=("$arm")
done
[ ${#selected[@]} -gt 0 ] || { echo "no arm selected" >&2; exit 2; }
for arm in "${selected[@]}"; do
  tag=${arm%% *}
  for d in "replay-gate-$tag" "route-gate-$tag-doors" "route-gate-$tag-transport"; do
    [ ! -e "$CAMPAIGN/$d" ] || { echo "destination exists, choose a new tag: $CAMPAIGN/$d" >&2; exit 1; }
  done
done
# --- m7-clocks2 --- its snapshot also copies the clock controller module (both submitters add it for the export stage
# gaits), so the receipt's `git rev-parse HEAD` must describe that file too; clocks2_bench_gate compares it as well.
c2_selected=0; for arm in "${selected[@]}"; do [ "${arm%% *}" != m7-clocks2 ] || c2_selected=1; done
clocks2_gate_files=("${m2_gate_files[@]}" src/bhl_robust/eval/gait_clock.py)
if [ "$c2_selected" = 1 ] && ! git diff --quiet -- src/bhl_robust/eval/gait_clock.py; then
  echo "src/bhl_robust/eval/gait_clock.py has uncommitted changes; commit it before submitting a hash-frozen m7-clocks2 gate" >&2
  exit 1
fi
# --- m7-fix --- the same for the m7-fix arm (an export stage gait: its snapshot copies gait_clock.py as well).
fix_selected=0; for arm in "${selected[@]}"; do [ "${arm%% *}" != m7-fix ] || fix_selected=1; done
if [ "$fix_selected" = 1 ] && ! git diff --quiet -- src/bhl_robust/eval/gait_clock.py; then
  echo "src/bhl_robust/eval/gait_clock.py has uncommitted changes; commit it before submitting a hash-frozen m7-fix gate" >&2
  exit 1
fi
if [ -e "$CAMPAIGN/replay-gate-route-lock-1" ]; then
  echo "note: route-gate slot 1 already held ($(cat "$CAMPAIGN/replay-gate-route-lock-1/holder" 2>/dev/null)); a new PASS will be DEFERRED unless --max-route-gates is raised" >&2
fi

# ---- local preflight (MuJoCo import + argument parse, no physics) --------------------------------
export PYTHONPATH="$REPO/src:$REPO/scripts" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for arm in "${selected[@]}"; do
  read -r tag flags <<<"$arm"   # read trims the column padding between tag and flags
  # shellcheck disable=SC2086
  out=$(timeout 300 "$PY" scripts/mission7_plate_stage.py --repo "$REPO" --campaign "$DEFAULT_SOURCE_CAMPAIGN" --baseline "$DEFAULT_BASELINE" \
        --out "$CAMPAIGN/replay-gate-$tag" $flags --preflight)
  grep -q '"status": "PREFLIGHT_OK"' <<<"$out" || { echo "preflight failed for $tag: $out" >&2; exit 1; }
  echo "preflight $tag: $out"
  # shellcheck disable=SC2086
  "$PY" "$SUBMITTER" --out-campaign "$CAMPAIGN" --output-name "replay-gate-$tag" $flags | sed "s/^/plan $tag: /"
done
rule
m2_selected=0; for arm in "${selected[@]}"; do [ "${arm%% *}" != m2-turnboth ] || m2_selected=1; done
m3_selected=0; for arm in "${selected[@]}"; do [ "${arm%% *}" != m3-shipped-step ] || m3_selected=1; done
if [ "$submit" = 0 ]; then
  [ "$m2_selected" = 0 ] || echo "note: --submit queues m2-turnboth only if m2_bench_gate passes: $M2_BENCH_VERDICT must be a scored bench PASS and ${#m2_gate_files[@]} snapshot sources must match its provenance.json"
  [ "$m3_selected" = 0 ] || echo "note: --submit queues m3-shipped-step only if m3_bench_gate passes: $M3_BENCH_VERDICT must be a scored bench-v2 m3 PASS and ${#m2_gate_files[@]} snapshot sources must match its provenance.json"
  [ "$c2_selected" = 0 ] || echo "note: --submit queues m7-clocks2 only if clocks2_bench_gate passes: $CLOCKS2_BENCH_VERDICT must be a scored bench-v2 clocks2 PASS with the pinned clock-s2 weights and ${#clocks2_gate_files[@]} snapshot sources must match its provenance.json"
  [ "$fix_selected" = 0 ] || echo "note: --submit queues m7-fix only if fix_bench_gate passes: $FIX_BENCH_VERDICT must be a scored bench-v2 $FIX_GAIT PASS recording --cross-budget window --yaw-cap 0.6 with the pinned $FIX_GAIT weights ($FIX_POLICY_SHA256) and ${#clocks2_gate_files[@]} snapshot sources must match its provenance.json"
  echo "DRY RUN: nothing submitted.  Re-run with --submit to queue ${#selected[@]} gate(s) + ${#selected[@]} follow-up(s)."
  exit 0
fi
# M2: checked last, right before anything is queued; a refusal (RELEASE_BLOCKED, exit 3) submits nothing.
if [ "$m2_selected" = 1 ]; then
  m2_bench_gate "$M2_BENCH_VERDICT" "${m2_gate_files[@]}" || exit 3
fi
# M3: the same placement and the same snapshot set, against the bench-v2 verdict.
if [ "$m3_selected" = 1 ]; then
  m3_bench_gate "$M3_BENCH_VERDICT" "${m2_gate_files[@]}" || exit 3
fi
# --- m7-clocks2 --- the same placement, against the clocks2 bench-v2 verdict, gait_clock.py among the compared sources.
if [ "$c2_selected" = 1 ]; then
  clocks2_bench_gate "$CLOCKS2_BENCH_VERDICT" "$CLOCKS2_POLICY_SHA256" "$CLOCKS2_POLICY" "${clocks2_gate_files[@]}" || exit 3
fi
# --- m7-fix --- the same placement, against the fix bench-v2 verdict (gait_clock.py and env.py among the compared sources).
if [ "$fix_selected" = 1 ]; then
  fix_export_arg=""; [ "$FIX_GAIT" != export ] || fix_export_arg=$FIX_EXPORT
  fix_bench_gate "$FIX_BENCH_VERDICT" "$FIX_GAIT" "$FIX_POLICY_SHA256" "$FIX_POLICY" "$fix_export_arg" "${clocks2_gate_files[@]}" || exit 3
fi

# ---- submit the chain ----------------------------------------------------------------------------
# Strip session Slurm variables for our own sbatch calls (the submitter does the same for its call).
clean=(); for v in $(compgen -e | grep -E '^(SLURM_|SBATCH_|TMPDIR$|CUDA_VISIBLE_DEVICES$)'); do clean+=(-u "$v"); done
real_sbatch=$(command -v sbatch) || { echo "sbatch is not on PATH" >&2; exit 1; }
shim_dir=$(mktemp -d); trap 'rm -rf "$shim_dir"' EXIT
# A noexec TMPDIR would break the shim at gate 2, after gate 1 is queued: prove the shim directory executes first.
printf '#!/bin/bash\necho shim-ok\n' > "$shim_dir/sbatch"; chmod +x "$shim_dir/sbatch"
[ "$("$shim_dir/sbatch")" = shim-ok ] || { echo "shim directory $shim_dir is not executable (noexec TMPDIR?); nothing submitted" >&2; exit 1; }
stamp=$(date -u +%Y%m%dT%H%M%SZ)
receipt="$CAMPAIGN/replay-gate-chain-$stamp.json"
prev="$after"; rows=(); ledger_arms=""
for arm in "${selected[@]}"; do
  read -r tag flags <<<"$arm"   # read trims the column padding between tag and flags
  # The dependency shim is rebuilt per gate; without a predecessor the submitter sees the plain PATH.
  dep_path="$PATH"
  if [ -n "$prev" ]; then
    printf '#!/bin/bash\nexec %q --dependency=afterany:%s "$@"\n' "$real_sbatch" "$prev" > "$shim_dir/sbatch"
    chmod +x "$shim_dir/sbatch"; dep_path="$shim_dir:$PATH"
  fi
  # shellcheck disable=SC2086
  gate_json=$(env "${clean[@]}" PATH="$dep_path" "$PY" "$SUBMITTER" --out-campaign "$CAMPAIGN" --output-name "replay-gate-$tag" $flags --submit)
  gate_id=$(printf '%s' "$gate_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
  gate_dir="$CAMPAIGN/replay-gate-$tag"
  # Slurm records the dependency it actually applied; verify rather than trust the shim route.
  if [ -n "$prev" ]; then
    applied_dep=$(env "${clean[@]}" squeue -h -j "$gate_id" -o '%E' 2>/dev/null || true)
    prev_alive=$(env "${clean[@]}" squeue -h -j "$prev" -o '%T' 2>/dev/null || true)
    if ! grep -q "$prev" <<<"$applied_dep" && [ -n "$prev_alive" ]; then
      echo "gate $gate_id did not pick up dependency afterany:$prev (squeue shows '$applied_dep' while $prev is $prev_alive); cancelling $gate_id" >&2
      env "${clean[@]}" scancel "$gate_id"; exit 1
    fi
  fi
  follow_id=$(env "${clean[@]}" sbatch --parsable --dependency="afterany:$gate_id" --job-name="m7-gate-followup-$tag" \
              --account=eecs --partition=share --constraint='haswell&el8' --cpus-per-task=1 --mem=2G --time=00:15:00 \
              --chdir="$REPO" --output="$REPO/$gate_dir/m7-gate-followup-%j.out" --error="$REPO/$gate_dir/m7-gate-followup-%j.out" \
              "$FOLLOWUP" "$gate_dir" "$tag" "$max_route" "$CAMPAIGN" | cut -d';' -f1)
  echo "queued $tag: gate $gate_id (afterany:${prev:-none}, cn-c22) -> follow-up $follow_id (afterany:$gate_id)"
  rows+=("{\"tag\": \"$tag\", \"gate_job\": \"$gate_id\", \"gate_dependency\": \"${prev:+afterany:$prev}\", \"followup_job\": \"$follow_id\", \"flags\": \"$flags\", \"dir\": \"$gate_dir\"}")
  ledger_arms+=" \`$tag\` gate \`$gate_id\` / follow-up \`$follow_id\` (\`$flags\`);"
  prev=$gate_id
done
{
  printf '{\n  "submitted_utc": "%s",\n  "git_commit": "%s",\n  "max_route_gates": %s,\n' "$stamp" "$(git rev-parse HEAD)" "$max_route"
  printf '  "followup_sha256": "%s",\n  "wrapper_sha256": "%s",\n' "$(sha256sum "$FOLLOWUP" | cut -d' ' -f1)" "$(sha256sum "$0" | cut -d' ' -f1)"
  printf '  "rule": "only a 10/10 exact replay proceeds to the route gate (Doors 16 + Transport 16, validation 0-15); first PASS in chain order only",\n'
  printf '  "chain": [\n'; printf '    %s,\n' "${rows[@]}" | sed '$ s/,$//'; printf '  ]\n}\n'
} > "$receipt"
{
  printf '\nMission7 replay-gate chain (%s): **SUBMITTED** %d exact ten-fall replay gates sequentially on `cn-c22` (afterany), each with a follow-up that writes `summary.json` and releases the route gate (Doors 16 + Transport 16, validation 0-15, `submit_mission7_route_handoff.py`) only for a 10/10, first PASS in chain order only (%s route-gate slot). Arms:' "$(date -u +%F)" "${#rows[@]}" "$max_route"
  printf '%s' "$ledger_arms"
  printf ' chain receipt: `%s`.\n' "$receipt"
} >> SLURM_JOBS.md
echo "chain receipt: $receipt"
echo "SUBMITTED ${#rows[@]} gates + ${#rows[@]} follow-ups"
