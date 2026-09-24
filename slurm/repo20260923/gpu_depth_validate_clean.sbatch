#!/bin/bash
#SBATCH --job-name=depth-validate-clean
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --constraint=rtx8000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Repo-wide GPU campaign (2026-09-23): ray-cast depth validation on a CLEAN
# worktree of the committed HEAD.
#
# Why: the depth bench rerun 21402550 (results/repo-gpu-20260923/depth-bench/)
# measured a flat-plane mean relative error of 0.050 against the documented
# 0.029 (docs/REPORT.md "Validation first", FINDINGS section 5, commit
# b34b99a), but it ran on the main checkout with the user's uncommitted edits to
# src/bhl_robust/tasks/depth_env_cfg.py, terrain_env_cfg.py and
# src/bhl_robust/terrains/*.py (captured in that directory's worktree.patch), so
# the mismatch cannot be attributed to the committed code. Its rough-terrain
# departure check also timed out at 601 s, leaving the measurement incomplete.
#
# This job re-runs depth_validate.py ALONE (no throughput rows) with the
# committed tree and a 1500 s timeout:
#   1. git worktree add --detach /scratch/$USER/depth-clean-$SLURM_JOB_ID HEAD
#      (the user's dirty files stay in the main checkout; the worktree has only
#      what is committed). The Berkeley-Humanoid-Lite submodule content is not
#      in a fresh worktree, so its gitlink directory is replaced by a symlink to
#      the main checkout's external/ dir. Note the venv's editable install of
#      berkeley_humanoid_lite already resolves to that same main-checkout path;
#      only bhl_robust (PYTHONPATH, not pip-installed) comes from the worktree,
#      and that is the code under test.
#   2. Inside the container, before booting Isaac, assert that `bhl_robust`
#      resolves from the worktree's src (importlib.util.find_spec; no import).
#   3. scripts/bench/depth_validate.py --res 64 --num_envs 16 (the depth_bench.sh
#      arguments) from the worktree, timeout 1500 s (the bench's 600 s was not
#      enough for the second, rough-terrain env in the same process).
#   4. git worktree remove --force, from an EXIT/TERM trap, so the worktree and
#      its registration under $REPO/.git/worktrees go away on every path.
#      `git worktree prune` is deliberately NOT called: the user's other
#      worktrees live under /tmp on other nodes and a prune from this node
#      would drop their registrations.
#
# Stack: v51 (Isaac Sim 5.1 / Lab 2.3.2), the stack every published depth
# number was measured on. GPU: rtx8000 on partition gpu, as gpu_depth_bench.sbatch
# (b34b99a dates the measuring GPUs as Turing with RT cores). The relative-error
# claim is not hardware-bound, so to trade comparability for queue time:
#   sbatch --partition=gpu,dgxh,ampere --constraint="a40|rtx8000|l40s|h100|h200|h200-70g" ...
#
# Outputs (results/repo-gpu-20260923/depth-validate-clean/):
#   depth_validate.txt      BENCH_OUT: NODE line, then depth_validate.py's
#                           CHECK/PASS/FAIL lines, then a STEP line
#   depth_validate.json     scripts/bench/depth_bench_summary.py's parse() over it
#                           plus the validate-only claims and verdict below
#   step_validate_res64_n16.log   full stdout/stderr of the Isaac step
#   provenance.txt          worktree HEAD, worktree cleanliness, the main
#                           checkout's dirty depth files this run excludes,
#                           the import origin proven inside the container
#
# Final line, always printed:
#   DEPTH-VALIDATE-CLEAN RESULT: finite=<x> mre=<y> vs documented 0.029 -> PASS|MISMATCH|INCOMPLETE (...)
# with the rule (10 % relative tolerance, the bench parser's default):
#   INCOMPLETE  flat-plane finite fraction or mean relative error absent
#               (step crashed/timed out before the closed-form check), or the
#               error is within tolerance but the rough-ground departure was
#               not measured (a constant sensor cannot be excluded without it)
#   PASS        finite prints 1.000, |mre - 0.029| <= 10 % of 0.029, and the
#               validator's own "terrain moves the depth image" verdict is PASS
#   MISMATCH    everything measured, and any of those three fails
# Exit 0 for PASS or MISMATCH (a valid measurement either way), 1 for INCOMPLETE.
set -uo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
source "$REPO/slurm/inner/v60_boot_gate.sh"

JOB=${SLURM_JOB_ID:-local}
OUT=$REPO/results/repo-gpu-20260923/depth-validate-clean
# Absolute on purpose: depth_validate.py opens BENCH_OUT relative to its cwd.
export BENCH_OUT=$OUT/depth_validate.txt
SUMMARY=$OUT/depth_validate.json
STEP_LOG=$OUT/step_validate_res64_n16.log
mkdir -p "$OUT"
# depth_validate.py APPENDS. Remove this directory's previous outputs so the
# verdict can only come from this job's lines.
rm -f "$BENCH_OUT" "$SUMMARY" "$STEP_LOG" "$OUT"/inner_*.sh "$OUT/provenance.txt"

HOST_PY=$WORKSPACE/venv/bin/python; [ -x "$HOST_PY" ] || HOST_PY=$(command -v python3)
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 | tr ',' ' ' | tr -s ' ')
DOC_MRE=0.029
BOOT_LOCK=$WORKSPACE/.v51-boot.lock

# The final line is printed from exactly one place so it cannot be missed.
finish_incomplete() {  # finish_incomplete <reason>
    echo "DEPTH-VALIDATE-CLEAN RESULT: finite=NA mre=NA vs documented $DOC_MRE -> INCOMPLETE ($1)"
    echo "=== done: $(date) ==="
    exit 1
}

# ---------------------------------------------------------------- worktree
if mkdir -p "/scratch/$USER" 2>/dev/null && [ -w "/scratch/$USER" ]; then
    WT=/scratch/$USER/depth-clean-$JOB
else
    # Fallback for a node without node-local scratch; still bound into the
    # container (bhl_exec binds /nfs/hpc/share/$USER).
    WT=$XDG_CACHE_HOME/depth-clean-$JOB
    echo "note: /scratch/$USER unavailable on $(hostname); worktree under $WT" >&2
fi
SUBLINK=$WT/external/Berkeley-Humanoid-Lite

_cleaned=0
cleanup() {
    [ "$_cleaned" -eq 1 ] && return 0
    _cleaned=1
    _v60_release "$BOOT_LOCK"
    if [ -e "$WT" ] || git -C "$REPO" worktree list --porcelain 2>/dev/null | grep -qx "worktree $WT"; then
        # The symlink is unlinked explicitly first: rm -f never follows a link,
        # and the main checkout's submodule content must survive this.
        [ -L "$SUBLINK" ] && rm -f "$SUBLINK"
        if git -C "$REPO" worktree remove --force "$WT" 2>/dev/null; then
            echo "cleanup: worktree $WT removed"
        else
            rm -rf "$WT" 2>/dev/null
            echo "cleanup: git worktree remove failed; directory removed, registration may linger (git worktree prune when convenient)" >&2
        fi
    fi
    return 0
}
trap cleanup EXIT
trap 'cleanup; exit 143' TERM

echo "=== node: $(hostname) | depth validate (clean worktree) | stack $BHL_STACK | gpu ${GPU:-unknown} | $(date) ==="
if ! git -C "$REPO" worktree add --detach "$WT" HEAD; then
    echo "NODE | host=$(hostname) gpu=${GPU:-unknown} job=$JOB stack=$BHL_STACK head=unknown dirty=unknown worktree=failed" > "$BENCH_OUT"
    finish_incomplete "git worktree add failed"
fi
HEAD=$(git -C "$WT" rev-parse --short HEAD 2>/dev/null || echo unknown)
# Cleanliness is captured BEFORE the submodule symlink goes in: once the
# gitlink path is a symlink, `git status` in the worktree refuses to run.
WT_DIRTY=no; git -C "$WT" diff --quiet HEAD -- src scripts 2>/dev/null || WT_DIRTY=yes
WT_STATUS=$(git -C "$WT" status --short 2>&1)
DEPTH_CODE=(src/bhl_robust/tasks/depth_env_cfg.py src/bhl_robust/tasks/terrain_env_cfg.py
            src/bhl_robust/terrains src/bhl_robust/tasks/__init__.py src/bhl_robust/quat_order.py
            scripts/bench/depth_validate.py scripts/bench/depth_raycast.py slurm/inner/depth_bench.sh)
MAIN_DIRTY=no; git -C "$REPO" diff --quiet HEAD -- "${DEPTH_CODE[@]}" 2>/dev/null || MAIN_DIRTY=yes

# A gitlink checks out as an empty directory; the venv's editable install
# points at the main checkout's submodule, and the symlink makes the worktree's
# own path resolve to the same content.
if [ -d "$SUBLINK" ] && [ ! -L "$SUBLINK" ]; then
    rmdir "$SUBLINK" || finish_incomplete "submodule dir in worktree not empty"
fi
ln -s "$UPSTREAM" "$SUBLINK" || finish_incomplete "symlink to $UPSTREAM failed"
[ -d "$SUBLINK/source" ] || finish_incomplete "submodule content not reachable through $SUBLINK"

{
    echo "job $JOB host $(hostname) gpu ${GPU:-unknown} stack $BHL_STACK date $(date -Is)"
    echo "worktree $WT HEAD $HEAD (detached) src+scripts-dirty $WT_DIRTY"
    echo "main checkout $REPO HEAD $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) branch $(git -C "$REPO" branch --show-current 2>/dev/null) depth-code-dirty $MAIN_DIRTY"
    echo "upstream $UPSTREAM HEAD $(git -C "$UPSTREAM" rev-parse --short HEAD 2>/dev/null) (symlinked into the worktree; used by both runs)"
    echo "--- main checkout: uncommitted depth/terrain changes EXCLUDED from this run (git status --short)"
    git -C "$REPO" status --short -- "${DEPTH_CODE[@]}" 2>/dev/null
    echo "--- worktree: git status --short before the submodule symlink (expect empty)"
    echo "${WT_STATUS:-(clean)}"
} | tee "$OUT/provenance.txt"
echo "NODE | host=$(hostname) gpu=${GPU:-unknown} job=$JOB stack=$BHL_STACK head=$HEAD dirty=$WT_DIRTY worktree=clean main_depth_dirty=$MAIN_DIRTY" > "$BENCH_OUT"
[ "$WT_DIRTY" = no ] || finish_incomplete "fresh worktree is not clean"

# Inner script (inside the container). REPO, PY, UPSTREAM are set by bhl_exec;
# BENCH_OUT and TMPDIR are in BHL_FORWARD_VARS. PYTHONPATH is the worktree's
# src ONLY (not appended to $REPO/src), so bhl_robust cannot come from the
# main checkout. Two modes: `check` proves the import origin without booting
# Isaac; `run <timeout_s>` runs the validator from the worktree.
INNER=$OUT/inner_$JOB.sh
cat > "$INNER" <<'EOF'
#!/bin/bash
set -uo pipefail
mode=$1; wt=$2; shift 2
export PYTHONPATH="$wt/src"
cd "$wt/external/Berkeley-Humanoid-Lite" || exit 3
case "$mode" in
    check)
        "$PY" - "$wt" <<'PY'
import importlib.util, sys
wt = sys.argv[1]
for name in ("bhl_robust", "berkeley_humanoid_lite"):
    try:
        spec = importlib.util.find_spec(name)
        origin = getattr(spec, "origin", None) or ""
    except Exception as exc:  # noqa: BLE001 -- the editable finder may not give an origin
        spec, origin = None, f"<find_spec raised {exc!r}>"
    # berkeley_humanoid_lite is informational only: its editable finder maps it
    # to the main checkout's submodule (same content the symlink points at).
    print(f"IMPORT_ORIGIN | {name} = {origin or '(editable finder, no origin)'}", flush=True)
    if name == "bhl_robust" and not origin.startswith(wt + "/src/"):
        print(f"IMPORT_ORIGIN | FAIL: bhl_robust does not resolve from {wt}/src", flush=True)
        sys.exit(2)
print("IMPORT_ORIGIN | ok", flush=True)
PY
        ;;
    run)
        limit=$1; shift
        timeout -k 30 "$limit" "$PY" "$wt/scripts/bench/depth_validate.py" "$@"
        ;;
    *) echo "inner: unknown mode $mode" >&2; exit 3 ;;
esac
EOF

# ------------------------------------------------- import-origin proof
echo "=== import origin check (container, no Isaac) | $(date) ==="
CHECK_LOG=$OUT/import_check.log
bhl_exec "$INNER" check "$WT" > "$CHECK_LOG" 2>&1
rc=$?
grep "^IMPORT_ORIGIN" "$CHECK_LOG" | tee -a "$OUT/provenance.txt"
if [ "$rc" -ne 0 ]; then
    tail -n 20 "$CHECK_LOG"
    echo "STEP | name=import_check rc=$rc" >> "$BENCH_OUT"
    finish_incomplete "bhl_robust did not resolve from the clean worktree (rc=$rc)"
fi
echo "STEP | name=import_check rc=0" >> "$BENCH_OUT"

# ------------------------------------------------------- validate step
# The boot gate may wait up to an hour for another v51 job's Kit startup, so
# the 1500 s limit is clamped to what is left of the 1.5 h before the summary.
DEADLINE=$(( 85 * 60 ))
V60_BOOT_LOCK=$BOOT_LOCK v60_boot_gate
# v60_boot_gate installs its own EXIT trap; put ours back (it releases the lock too).
trap cleanup EXIT
remaining=$(( DEADLINE - SECONDS ))
LIMIT=1500
if (( remaining < 300 )); then
    echo "STEP | name=validate_res64_n16 rc=skipped_deadline secs=0" >> "$BENCH_OUT"
    finish_incomplete "skipped: ${remaining}s left before the deadline after the boot gate"
fi
(( LIMIT > remaining - 120 )) && LIMIT=$(( remaining - 120 ))

echo "=== step validate_res64_n16 (timeout ${LIMIT}s) | $(date) ==="
t0=$SECONDS
bhl_exec "$INNER" run "$WT" "$LIMIT" --res 64 --num_envs 16 > "$STEP_LOG" 2>&1
rc=$?
secs=$(( SECONDS - t0 ))
# depth_validate.py writes its lines to BENCH_OUT only at the very end, so a
# crash or timeout in the rough-terrain half would lose the flat-plane numbers
# it had already printed. Recover them from its stdout when the file lacks them.
salvaged=0
if ! grep -q "flat-ground depth matches" "$BENCH_OUT"; then
    salvaged=$(grep -cE "^(CHECK|PASS|FAIL) +\|" "$STEP_LOG")
    grep -E "^(CHECK|PASS|FAIL) +\|" "$STEP_LOG" >> "$BENCH_OUT"
fi
echo "STEP | name=validate_res64_n16 rc=$rc secs=$secs timeout_s=$LIMIT salvaged_stdout_lines=$salvaged" >> "$BENCH_OUT"
echo "--- validate_res64_n16: rc=$rc in ${secs}s (log $STEP_LOG)"
[ "$rc" -eq 0 ] || tail -n 15 "$STEP_LOG"

echo "=== bench lines ==="
grep -E "^(NODE|CHECK|PASS|FAIL|STEP)" "$BENCH_OUT" || echo "(no bench lines)"

# ------------------------------------------------------------ summary
# The parser is the committed depth_bench_summary.py from the worktree (its
# parse() reads CHECK/PASS/FAIL lines; its evaluate() would also demand the
# throughput rows this job does not run, so the validate-only claims and the
# verdict are computed here from the parsed fields, with the same tolerance).
echo "=== summary vs documented claims (10 % tolerance) ==="
"$HOST_PY" - "$BENCH_OUT" "$SUMMARY" "$WT/scripts/bench" "$REPO/scripts/bench" <<'PY'
import json, sys
from pathlib import Path
bench, out, *paths = sys.argv[1:]
try:
    for p in paths:
        if Path(p, "depth_bench_summary.py").is_file():
            sys.path.insert(0, p); break
    import depth_bench_summary as dbs  # noqa: E402

    TOL = 0.10
    text = Path(bench).read_text(errors="replace")
    d = dbs.parse(text)
    v, o = d["validate"], d["orientations"]
    finite = v.get("flat_finite_fraction")
    mre = o.get(dbs.DOC_BEST_ORIENTATION, {}).get("mean_relative_error")
    if mre is None and v.get("best_orientation") == dbs.DOC_BEST_ORIENTATION:
        mre = v.get("best_error")
    flip = o.get("row-flipped", {}).get("mean_relative_error")
    depart = v.get("rough_departure")
    moves = v.get("terrain_moves_verdict")
    step = next((s for s in d["steps"] if s.get("name") == "validate_res64_n16"), {})

    def within(m, doc):
        return m is not None and abs(m - doc) <= TOL * abs(doc)

    claims = [
        {"claim": "finite_fraction", "measured": finite, "documented": dbs.DOC_FINITE_FRACTION,
         "verdict": "NOT_MEASURED" if finite is None else ("PASS" if finite >= 0.9995 else "MISMATCH"),
         "rule": "validator prints 3 decimals; the '100 % finite' claim needs 1.000"},
        {"claim": "mean_relative_error", "measured": mre, "documented": dbs.DOC_MEAN_REL_ERR,
         "rel_diff": None if mre is None else (mre - dbs.DOC_MEAN_REL_ERR) / dbs.DOC_MEAN_REL_ERR,
         "verdict": "NOT_MEASURED" if mre is None else ("PASS" if within(mre, dbs.DOC_MEAN_REL_ERR) else "MISMATCH"),
         "pixels": o.get(dbs.DOC_BEST_ORIENTATION, {}).get("pixels"), "documented_pixels": dbs.DOC_MRE_PIXELS,
         "validate_verdict": v.get("validate_verdict"), "validate_tol": v.get("validate_tol")},
        {"claim": "row_flipped_relative_error", "measured": flip, "documented": dbs.DOC_ROW_FLIPPED_ERR,
         "verdict": "NOT_MEASURED" if flip is None else ("PASS" if within(flip, dbs.DOC_ROW_FLIPPED_ERR) else "MISMATCH")},
        {"claim": "best_orientation", "measured": v.get("best_orientation"), "documented": dbs.DOC_BEST_ORIENTATION,
         "verdict": "NOT_MEASURED" if v.get("best_orientation") is None else
                    ("PASS" if v.get("best_orientation") == dbs.DOC_BEST_ORIENTATION else "MISMATCH")},
        {"claim": "rough_departure", "measured": depart, "documented": None,
         "terrain_moves_verdict": moves, "terrain_env_spread_m": v.get("terrain_env_spread_m"),
         "verdict": "NOT_MEASURED" if depart is None else ("PASS" if moves == "PASS" else "MISMATCH"),
         "rule": "validator: departure > 3x the flat-plane error (a constant sensor would not move)"},
    ]
    reasons = []
    if finite is None or mre is None:
        verdict = "INCOMPLETE"
        reasons.append("no flat-plane " + ("finite fraction" if finite is None else "relative error"))
        reasons.append(f"step rc={step.get('rc', '?')}")
    else:
        mre_ok = within(mre, dbs.DOC_MEAN_REL_ERR)
        finite_ok = finite >= 0.9995
        if not mre_ok:
            reasons.append(f"mre {mre:.4f} is {(mre - dbs.DOC_MEAN_REL_ERR) / dbs.DOC_MEAN_REL_ERR:+.0%} vs 0.029"
                           f" (validator {v.get('validate_verdict')} at tol {v.get('validate_tol')})")
        if not finite_ok:
            reasons.append(f"finite {finite:.3f} < 1.000")
        if depart is None:
            reasons.append(f"rough departure NOT_MEASURED (step rc={step.get('rc', '?')})")
        elif moves != "PASS":
            reasons.append(f"terrain_moves={moves} (departure {depart:.4f})")
        if not mre_ok or not finite_ok or (depart is not None and moves != "PASS"):
            verdict = "MISMATCH"
        elif depart is None:
            verdict = "INCOMPLETE"
        else:
            verdict = "PASS"
            reasons.append(f"departure {depart:.4f}, terrain_moves PASS")
    counts = {k: sum(c["verdict"] == k for c in claims) for k in ("PASS", "MISMATCH", "NOT_MEASURED")}
    summary = {"bench_txt": bench, "mode": "validate-only (clean worktree)", "tolerance": TOL,
               "verdict": verdict, "reasons": reasons, "claims": claims, "claim_counts": counts,
               "node": d["node"], "validate": v, "orientations": o, "steps": d["steps"],
               "warnings": d["warnings"], "parser": dbs.__file__}
    Path(out).write_text(json.dumps(summary, indent=2))

    n = d["node"]
    print(f"NODE   | host={n.get('host', '?')} gpu={n.get('gpu', '?')} stack={n.get('stack', '?')} "
          f"head={n.get('head', '?')} dirty={n.get('dirty', '?')}")
    for c in claims:
        m, doc = c["measured"], c["documented"]
        ms = "NA" if m is None else (f"{m:.4f}" if isinstance(m, float) else str(m))
        ds = "n/a" if doc is None else (f"{doc:.4f}" if isinstance(doc, float) else str(doc))
        print(f"{c['verdict']:12s} | {c['claim']:28s} measured {ms:>12s} vs documented {ds:>8s}")
    for w in d["warnings"]:
        print(f"WARN   | {w}")
    for s in d["steps"]:
        print(f"STEP   | {' '.join(f'{k}={val}' for k, val in s.items())}")
    fs = "NA" if finite is None else f"{finite:.3f}"
    mres = "NA" if mre is None else f"{mre:.4f}"
    ds = "NA" if depart is None else f"{depart:.4f}"
    print(f"DEPTH-VALIDATE-CLEAN RESULT: finite={fs} mre={mres} vs documented {dbs.DOC_MEAN_REL_ERR} "
          f"-> {verdict} ({'; '.join(reasons)}; departure={ds} terrain_moves={moves or 'NOT_MEASURED'}"
          f"; head={n.get('head', '?')} gpu={n.get('gpu', '?')})")
    sys.exit(0 if verdict in ("PASS", "MISMATCH") else 1)
except Exception as exc:  # noqa: BLE001 -- exactly one RESULT line on every path
    print(f"DEPTH-VALIDATE-CLEAN RESULT: finite=NA mre=NA vs documented 0.029 -> INCOMPLETE "
          f"(summary crashed: {exc!r}; raw lines in {bench})", flush=True)
    sys.exit(1)
PY
rc=$?
if [ "$rc" -gt 1 ]; then
    # Only when the interpreter itself could not run (the heredoc handles its own exceptions).
    finish_incomplete "summary interpreter failed rc=$rc; raw lines in $BENCH_OUT"
fi
echo "=== done: $(date) ==="
exit $rc
