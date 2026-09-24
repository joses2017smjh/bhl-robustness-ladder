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
)
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
CAMPAIGN=results/mission7-campaign-20260923
SUBMITTER=scripts/submit_mission7_plate_stage.py
FOLLOWUP=slurm/repo20260923/m7_replay_gate_followup.sbatch
DEFAULT_SOURCE_CAMPAIGN=results/mission7-replay-smoke-20260921
DEFAULT_BASELINE=results/mission7-approach-followup-20260922/replay-diagnose-cn-c22
cd "$REPO"

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
  4. Episode envelope (docs/MISSION7_TASKS.md: 101 of 512 unspent): five gates
     = 50, one route gate = 32.  The route gate is released for the FIRST 10/10
     in chain order only (route-gate slots: $max_route); later passes are
     recorded PASS and their route-gate commands printed as ROUTE_GATE_DEFERRED
     for explicit re-authorization.
     A layout whose recorded trace ends before the stage reaches \`cross\` is
     upright trivially; summary.json therefore also reports crossings_started
     and crossings_completed, and a 10/10 is reported together with them.
  5. Chain order: ${selected[*]%% *} -- v2-align first because it is the only arm
     with a local 9/10 behind it; the reference V2 (7/10, 21401689) is not rerun.
================================================================================
RULE
}

# ---- pre-checks (both modes) ------------------------------------------------------------------
[ -x "$PY" ] || { echo "venv python missing: $PY" >&2; exit 1; }
[ -f "$DEFAULT_SOURCE_CAMPAIGN/fullroute/legacy-doors.json" ] || { echo "replay source missing" >&2; exit 1; }
[ -f "$DEFAULT_BASELINE/result.json" ] || { echo "authoritative baseline missing" >&2; exit 1; }
bash -n "$FOLLOWUP"
"$PY" -m py_compile "$SUBMITTER" scripts/mission7_plate_stage.py scripts/submit_mission7_route_handoff.py
# The snapshot submitter records `git rev-parse HEAD` in every receipt; a dirty snapshot file would misrepresent it.
snapshot_files=(src/bhl_robust/mission/*.py src/bhl_robust/__init__.py src/bhl_robust/sensor_io.py src/bhl_robust/eval/__init__.py
                src/bhl_robust/eval/multi_robot.py src/bhl_robust/eval/mjcf_assets.py src/bhl_robust/eval/livery.py
                src/bhl_robust/eval/team_sensors.py slurm/mission7_plate_stage.sbatch scripts/mission7*.py)
if ! git diff --quiet -- "${snapshot_files[@]}"; then
  echo "snapshot sources have uncommitted changes; commit or stash before submitting a hash-frozen gate:" >&2
  git status --short -- "${snapshot_files[@]}" >&2; exit 1
fi
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
if [ "$submit" = 0 ]; then
  echo "DRY RUN: nothing submitted.  Re-run with --submit to queue ${#selected[@]} gate(s) + ${#selected[@]} follow-up(s)."
  exit 0
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
  printf '\nMission7 replay-gate chain (2026-09-24): **SUBMITTED** %d exact ten-fall replay gates sequentially on `cn-c22` (afterany), each with a follow-up that writes `summary.json` and releases the route gate (Doors 16 + Transport 16, validation 0-15, `submit_mission7_route_handoff.py`) only for a 10/10, first PASS in chain order only (%s route-gate slot). Arms:' "${#rows[@]}" "$max_route"
  printf '%s' "$ledger_arms"
  printf ' chain receipt: `%s`.\n' "$receipt"
} >> SLURM_JOBS.md
echo "chain receipt: $receipt"
echo "SUBMITTED ${#rows[@]} gates + ${#rows[@]} follow-ups"
