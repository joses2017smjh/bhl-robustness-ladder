#!/bin/bash
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=haswell&el8
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
# Mission 7 replay-gate follow-up (2026-09-24).  Queued by
# slurm/repo20260923/submit_m7_replay_gates.sh with --dependency=afterany:<gate>.
#
#   usage: sbatch m7_replay_gate_followup.sbatch <gate_dir> <tag> [max_route_gates] [campaign_dir]
#
# 1. Reads <gate_dir>/result.json (written by scripts/mission7_plate_stage.py on
#    cn-c22) and <gate_dir>/submission.json (the snapshot submitter's receipt),
#    writes <gate_dir>/summary.json (compact: verdict, upright, per-layout falls,
#    probe args, job id) and rebuilds <campaign>/replay-gate-matrix-summary.json
#    from every replay-gate-*/result.json (idempotent, so the last follow-up in
#    the chain leaves a complete matrix).
# 2. PREDECLARED RULE: only an exact 10/10 (complete, 10 episodes, 10 upright,
#    0 falls, exact_replay_gate_passed == true) may proceed to the route gate.
#    Anything else prints FAIL or INCOMPLETE and submits nothing.  The route
#    gate is Doors 16 + Transport 16 on validation layouts 0-15 through the
#    EXISTING scripts/submit_mission7_route_handoff.py (early handoff,
#    haswell&el8, 2 CPUs / 12 GB / 2 h each), with the stage flags derived from
#    the gate's own probe_args (never hard-coded) plus --stage-wait-open=0.0,
#    because the replay gates run at wait-open 0 while the route probe defaults
#    to 2.0; the route-side composition is the one that produced the only
#    end-to-end Doors success (local-stage-v2/doors-1-v3-center):
#    --stage-activate --rejoin-advance --exit-ramp-center --exit-ramp=1.2
#    --rejoin-fix none.  Route-side flags are outside the replay by
#    construction (like F2/F3 were) and are fixed here, not tuned to results.
# 3. Episode envelope (docs/MISSION7_TASKS.md: 101 of 512 unspent): five gates
#    spend 50, one route gate 32.  A second passing variant would exceed the
#    envelope, so the route gate fires for the FIRST 10/10 in chain order only
#    (atomic mkdir lock, slots = max_route_gates, default 1).  Later passes are
#    recorded PASS and their route-gate commands printed under
#    ROUTE_GATE_DEFERRED for explicit re-authorization.  Nothing is weakened.
#
# DRY_RUN=1 runs the same logic without sbatch (plans the route submission via
# the submitter's non---submit mode and touches no lock).
set -euo pipefail
unset SBATCH_DEPENDENCY || true   # defensive only: the chain dependency is passed on the command line, never by env
gate_dir=${1:?gate_dir}; tag=${2:?tag}; max_route=${3:-1}; campaign=${4:-results/mission7-campaign-20260923}
REPO=${M7_REPO:-/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder}
PY=${PYTHON:-/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python}
cd "$REPO"
export PYTHONUNBUFFERED=1
DRY_RUN=${DRY_RUN:-0}
echo "m7 replay-gate follow-up: tag=$tag gate_dir=$gate_dir max_route_gates=$max_route campaign=$campaign dry_run=$DRY_RUN host=$(hostname) job=${SLURM_JOB_ID:-local}"

# ---- 1. verdict + compact summaries -------------------------------------------------------------
verdict_file=$(mktemp)
"$PY" - "$gate_dir" "$tag" "$campaign" "$verdict_file" <<'PYEOF'
import datetime as dt, json, pathlib, sys
gate = pathlib.Path(sys.argv[1]).resolve(); tag = sys.argv[2]
campaign = pathlib.Path(sys.argv[3]).resolve(); verdict_file = pathlib.Path(sys.argv[4])

def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None

def summarize(gate_dir):
    result = load(gate_dir / "result.json"); submission = load(gate_dir / "submission.json")
    rows = (result or {}).get("episode_comparison") or []
    upright = int((result or {}).get("upright") or 0); falls = int((result or {}).get("falls") or 0)
    episodes = int((result or {}).get("episodes") or 0)
    complete = bool(result and result.get("complete"))
    passed = bool(complete and episodes == 10 and upright == 10 and falls == 0
                  and result.get("exact_replay_gate_passed") is True
                  and result.get("status") == "COMPLETED_INTERVENTION")
    verdict = "PASS" if passed else ("FAIL" if complete and episodes == 10 else "INCOMPLETE")
    return {
        "tag": gate_dir.name.replace("replay-gate-", "", 1),
        "verdict": verdict, "exact_replay_gate_passed": passed,
        "upright": upright, "falls": falls, "episodes": episodes, "complete": complete,
        "status": (result or {}).get("status"),
        "job_id": (submission or {}).get("job_id"), "requested_node": (submission or {}).get("requested_node"),
        "probe_args": (submission or {}).get("probe_args"), "git_commit": (submission or {}).get("git_commit"),
        "source_sha256_plate_stage": ((submission or {}).get("source_sha256") or {}).get("scripts/mission7_plate_stage.py"),
        "layouts": [{"layout_index": r.get("layout_index"), "fall": r.get("fall"), "first_fall_s": r.get("first_fall_s"),
                     "maximum_tilt": r.get("maximum_tilt"), "phases": [h.get("phase") for h in r.get("stage_history", [])]}
                    for r in rows],
        "fallen_layouts": [r.get("layout_index") for r in rows if r.get("fall")],
        # A layout whose recorded trace ends before the stage reaches `cross` counts as upright
        # trivially (9/12/13 did so in the local V3/V4 replays); the gate is 10/10 upright as
        # documented, but the reader must see how many crossings the pass actually exercised.
        "crossings_started": sum(1 for r in rows if "cross" in [h.get("phase") for h in r.get("stage_history", [])]),
        "crossings_completed": sum(1 for r in rows if (lambda ph: "cross" in ph and "recorded" in ph[ph.index("cross"):])(
            [h.get("phase") for h in r.get("stage_history", [])])),
        "gate_dir": str(gate_dir),
        "rule": "only a 10/10 exact replay (complete, 10 episodes, 10 upright, 0 falls) proceeds to the route gate",
        "summarized_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

mine = summarize(gate)
(gate / "summary.json").write_text(json.dumps(mine, indent=2) + "\n")
matrix = {"rule": mine["rule"], "reference_v2_job": "21401689 (7/10 FAIL: --pre-point=0.45 --cross-clear=0.35 --cross-kick, settle 0.40)",
          "rebuilt_utc": mine["summarized_utc"], "gates": []}
for d in sorted(campaign.glob("replay-gate-*")):
    if d.is_dir() and (d / "submission.json").exists():
        s = summarize(d); s.pop("layouts", None); matrix["gates"].append(s)
(campaign / "replay-gate-matrix-summary.json").write_text(json.dumps(matrix, indent=2) + "\n")
verdict_file.write_text(mine["verdict"] + "\n")
print(f"M7 REPLAY GATE {tag}: {mine['verdict']} {mine['upright']}/{mine['episodes']} upright, falls {mine['falls']} "
      f"(fallen layouts {mine['fallen_layouts']}); crossings started {mine['crossings_started']}, completed "
      f"{mine['crossings_completed']} of {mine['episodes']}; job {mine['job_id']}; summary -> {gate / 'summary.json'}")
PYEOF
verdict=$(cat "$verdict_file"); rm -f "$verdict_file"

# ---- 2. lossless compaction of the raw trace (ledger convention: gzip in place, never delete) ----
if [ "$DRY_RUN" = 0 ] && [ -f "$gate_dir/episodes.json" ] && [ ! -f "$gate_dir/episodes.json.gz" ]; then
  gzip "$gate_dir/episodes.json" && echo "episodes.json gzipped in place ($(du -h "$gate_dir/episodes.json.gz" | cut -f1))"
fi

if [ "$verdict" != PASS ]; then
  echo "ROUTE_GATE_NOT_RELEASED tag=$tag verdict=$verdict (predeclared rule: only 10/10 proceeds)"
  echo "M7_GATE_FOLLOWUP_COMPLETE"
  exit 0
fi

# ---- 3. translate the gate's probe_args into route-submitter flags ------------------------------
flags_file=$(mktemp)
"$PY" - "$gate_dir" > "$flags_file" <<'PYEOF'
import json, pathlib, sys
args = json.loads((pathlib.Path(sys.argv[1]) / "submission.json").read_text())["probe_args"]
out, wait_open = [], "0.0"
for a in args:
    key, _, val = a.partition("=")
    if key == "--pre-point": out += ["--pre-point", val]
    elif key == "--cross-clear": out += ["--cross-clear", val]
    elif key == "--settle-s": out += ["--settle-s", val]
    elif key == "--cross-kick": out.append("--cross-kick")
    elif key == "--align-yaw": out.append("--align-yaw")
    elif key == "--stage-lateral": out += ["--stage-lateral", val]
    elif key == "--wait-open": wait_open = val
    elif key == "--press-hold": out.append("--stage-press-hold")
    elif key == "--smoke": raise SystemExit("a --smoke gate is not a gate; refusing to release the route gate")
    else: raise SystemExit(f"unknown replay-gate flag {a!r}; refusing to guess a route translation")
out += ["--stage-wait-open", wait_open]
print("\n".join(out))
PYEOF
mapfile -t stage_flags < "$flags_file"; rm -f "$flags_file"
[ ${#stage_flags[@]} -ge 3 ] || { echo "ROUTE_GATE_BLOCKED: stage-flag translation produced nothing"; exit 3; }
route_campaign=${M7_ROUTE_CAMPAIGN:-$campaign}   # override only for local dry-run tests against scratch copies
route_common=(--campaign "$route_campaign" --indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --handoff early
              --constraint 'haswell&el8' --rejoin-fix none --stage-activate --rejoin-advance --exit-ramp-center --exit-ramp 1.2
              "${stage_flags[@]}")
cmd_doors=("$PY" scripts/submit_mission7_route_handoff.py "${route_common[@]}" --stage doors --output-name "route-gate-$tag-doors")
cmd_transport=("$PY" scripts/submit_mission7_route_handoff.py "${route_common[@]}" --stage transport --output-name "route-gate-$tag-transport")
echo "route gate composition for $tag: ${route_common[*]}"

# The route destinations are checked BEFORE the lock is taken so a blocked release never holds a slot.
for d in "route-gate-$tag-doors" "route-gate-$tag-transport"; do
  if [ -e "$route_campaign/$d" ]; then echo "ROUTE_GATE_BLOCKED: destination exists: $route_campaign/$d"; exit 3; fi
done

# ---- 4. envelope lock: first 10/10 in chain order only (slots = max_route_gates) ----------------
slot=""
if [ "$DRY_RUN" = 0 ]; then
  for i in $(seq 1 "$max_route"); do
    if mkdir "$campaign/replay-gate-route-lock-$i" 2>/dev/null; then slot=$i; echo "$tag $(date -u +%FT%TZ) ${SLURM_JOB_ID:-local}" > "$campaign/replay-gate-route-lock-$i/holder"; break; fi
  done
  if [ -z "$slot" ]; then
    echo "ROUTE_GATE_DEFERRED tag=$tag: PASS recorded, but all $max_route route-gate slot(s) are taken (episode envelope). Re-authorize explicitly and run:"
    printf '  %q ' "${cmd_doors[@]}" --submit; echo; printf '  %q ' "${cmd_transport[@]}" --submit; echo
    echo "M7_GATE_FOLLOWUP_COMPLETE"; exit 0
  fi
fi

# ---- 5. release the route gate through the existing snapshot submitter ---------------------------
# If sbatch fails from inside the job, free the envelope slot and any half-built route directory so
# a later PASS (or a manual re-run of the printed commands) can still release the route gate.
doors_id=""
release_on_failure() {
  rc=$?; [ "$rc" -eq 0 ] && return 0
  for d in "route-gate-$tag-doors" "route-gate-$tag-transport"; do
    [ -f "$route_campaign/$d/submission.json" ] || rm -rf "${route_campaign:?}/$d"
  done
  if [ -n "$doors_id" ]; then
    # Doors is already queued (half the slot is spent): keep the lock, only Transport needs the manual run.
    echo "ROUTE_GATE_SUBMIT_FAILED tag=$tag rc=$rc (doors job $doors_id already queued, slot $slot kept; submit Transport manually):"
    printf '  %q ' "${cmd_transport[@]}" --submit; echo
    return 0
  fi
  [ -z "$slot" ] || { rm -f "$campaign/replay-gate-route-lock-$slot/holder"; rmdir "$campaign/replay-gate-route-lock-$slot" 2>/dev/null || true; }
  echo "ROUTE_GATE_SUBMIT_FAILED tag=$tag rc=$rc (slot released; run manually):"
  printf '  %q ' "${cmd_doors[@]}" --submit; echo; printf '  %q ' "${cmd_transport[@]}" --submit; echo
}
trap release_on_failure EXIT
if [ "$DRY_RUN" != 0 ]; then
  echo "ROUTE_GATE_READY (planning only: DRY_RUN=$DRY_RUN, sbatch $(command -v sbatch >/dev/null 2>&1 && echo 'on PATH' || echo 'not on PATH'))"
  "${cmd_doors[@]}"; "${cmd_transport[@]}"
  printf 'run: '; printf '%q ' "${cmd_doors[@]}" --submit; echo
  printf 'run: '; printf '%q ' "${cmd_transport[@]}" --submit; echo
  echo "M7_GATE_FOLLOWUP_COMPLETE"; exit 0
fi
command -v sbatch >/dev/null 2>&1 || { echo "sbatch is not on PATH inside this job"; exit 4; }   # -> trap: slot released, commands printed
receipt="$route_campaign/route-gate-$tag.json"
doors_json=$("${cmd_doors[@]}" --submit); doors_id=$(printf '%s' "$doors_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
transport_json=$("${cmd_transport[@]}" --submit); transport_id=$(printf '%s' "$transport_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
"$PY" - "$receipt" "$tag" "$doors_id" "$transport_id" "$slot" "${SLURM_JOB_ID:-local}" "${route_common[@]}" <<'PYEOF'
import datetime as dt, json, sys
path, tag, doors, transport, slot, job, *composition = sys.argv[1:]
json.dump({"tag": tag, "released_by_followup_job": job, "route_gate_slot": int(slot), "doors_job": doors, "transport_job": transport,
           "layouts": "validation 0-15", "route_gate_criterion": "Doors >= 16/16 AND Transport >= 16/16 (docs/MISSION7_TASKS.md gate table)",
           "composition": composition, "submitted_utc": dt.datetime.now(dt.timezone.utc).isoformat()},
          open(path, "w"), indent=2)
PYEOF
echo "ROUTE_GATE_SUBMITTED tag=$tag doors=$doors_id transport=$transport_id receipt=$receipt"
echo "verdict later: $PY -c \"import json;[print(d, sum(e['success'] for e in json.load(open(f'$route_campaign/route-gate-$tag-{d}/result.json'))['episodes']), '/16') for d in ('doors','transport')]\""
echo "M7_GATE_FOLLOWUP_COMPLETE"
