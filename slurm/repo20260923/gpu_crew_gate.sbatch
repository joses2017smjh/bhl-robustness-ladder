#!/bin/bash
#SBATCH --job-name=crew-gate
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Repo-wide campaign (2026-09-23), COOP-21: rerun the crew-of-3/4 gate.
#
# Same job as slurm/62_crew_gate.sbatch (v51 stack, same partition, constraint,
# 8 CPU / 64 GB / 1 GPU; 2 h wall instead of its 3 h) with outputs under
#   results/repo-gpu-20260923/crew-gate/
#
# Why it is being rerun. The four recorded gate failures (last: 21006649,
# 2026-08-23) died on `policy/object_pos_a: scene entity 'robot_a' does not
# exist`. That was not the crew configs: scripts/train.py runs
# apply_depth_flags after Hydra on every task, and its restore branch added
# the pair's object-pose terms to any policy group that lacked them. Commit
# 73bc035 (2026-08-24, "That was the crew bug") guarded it; no gate has run
# since. tests/test_coop_crew_generated.py pushes all four crew classes through
# the real configclass/Hydra round trip plus both post-Hydra hooks, with the
# pair as the positive control, and passes on the current tree. The generated
# file is unchanged. This job is the Isaac-side confirmation.
#
# Procedure: the four train-path checks of slurm/inner/crew_validate.sh,
# verbatim in substance -- each of the four registered crew tasks through
# slurm/inner/train.sh (the driver the training arms use) with
# NUM_ENVS=8 MAX_ITER=2 SEED=0 and the gate's override
# env.stage_lift_on_pinch=true, PASS only if an iteration was logged --
# preceded by the import check. Two observability changes, no gate weakened:
#   1. The old import check printed IMPORT OK into a redirected file without
#      flushing and then called SimulationApp.close(), which tears the process
#      down from C++, so the verdict was lost and every run reported
#      "import FAIL" whether or not the import succeeded. Here the verdict is
#      written to a file before close(), and the check also constructs each
#      registered crew cfg through parse_env_cfg (crew_diagnose.py's path B).
#   2. Per-task logs are kept whole under the results dir, one per task id
#      (the old names collided: crew3 and crew3-depth both wrote
#      crew_gate_3.log), and the error grep accepts dotted exception names
#      (omegaconf.errors.*, hydra.errors.*), which the old pattern dropped --
#      21006649 has no error line on record for that reason.
# The inner script is emitted below into the results dir so this launcher is
# self-contained; slurm/inner/crew_validate.sh is not modified.
#
# Verdict: results/repo-gpu-20260923/crew-gate/crew_gate.json, built by host
# python from the verdict files and logs (never from exit codes), and the last
# line 'CREW-GATE RESULT: PASS|FAIL'. PASS means import OK and all four crew
# tasks logged a training iteration under the gate's settings -- NOT that a
# crew lifts anything. Exit 0 on PASS, 1 on FAIL (so a training array can
# still depend on it with afterok, as 61_crew_lift did).
#
# Time: five Kit boots ran in 66-88 s on 21006636/49 (they died before
# training); a passing run is ~5-10 min. The wall is 2 h, which bounds the
# worst case: import check 900 s + four tasks x the original 1500 s timeout,
# each with a 15 s SIGKILL grace (timeout -k) = 6975 s, leaving ~225 s for
# container start and the host-side verdict.
set -uo pipefail
[ -n "${SLURM_JOB_ID:-}" ] || { echo "CREW-GATE RESULT: FAIL (not inside a Slurm allocation; submit with sbatch)"; exit 2; }
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export CREW_GATE_OUT=${CREW_GATE_OUT:-$REPO/results/repo-gpu-20260923/crew-gate}
export GATE_OUT=$CREW_GATE_OUT/crew_gate.txt
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS GATE_OUT WORKSPACE CREW_GATE_OUT"
PYH=$WORKSPACE/venv/bin/python; [ -x "$PYH" ] || PYH=$(command -v python3)
TASKS=(CoopLift-BHL-Cube-Crew3-v0 CoopLift-BHL-Cube-Crew4-v0 CoopLift-BHL-Cube-Crew3-Depth-v0 CoopLift-BHL-Cube-Crew4-Depth-v0)

mkdir -p "$CREW_GATE_OUT/overrides"
# Remove stale outputs so PASS can only come from this job's files.
rm -f "$GATE_OUT" "$CREW_GATE_OUT/import_verdict.txt" "$CREW_GATE_OUT/crew_gate_import.log" "$CREW_GATE_OUT/crew_gate.json"
for t in "${TASKS[@]}"; do rm -f "$CREW_GATE_OUT/$t.log"; done

echo "=== node: $(hostname) | crew gate rerun | stack $BHL_STACK | job $SLURM_JOB_ID | $(date) ==="
cd "$REPO" && echo "repo HEAD $(git rev-parse --short HEAD 2>/dev/null) branch $(git branch --show-current 2>/dev/null)"
# Preflight: the guard that fixed the crew bug must be in the tree, or this
# job would only re-measure the known failure (the maze launchers use the
# same fail-fast pattern for their probe flag).
grep -q 'hasattr(cfg, "drop_object_pose")' "$REPO/src/bhl_robust/tasks/coop_depth_env_cfg.py" \
    || { echo "CREW-GATE RESULT: FAIL (apply_depth_flags guard missing from coop_depth_env_cfg.py; see commit 73bc035)"; exit 1; }
# depth_env_cfg.py is the Depth arms' transitive import (camera constants via
# coop_depth_env_cfg) and carries the user's uncommitted edits; recorded, not gated.
for f in src/bhl_robust/tasks/coop_crew_generated.py src/bhl_robust/tasks/coop_depth_env_cfg.py \
         src/bhl_robust/tasks/coop_lift_env_cfg.py src/bhl_robust/tasks/depth_env_cfg.py \
         scripts/train.py slurm/inner/train.sh; do
    if git -C "$REPO" diff --quiet -- "$f" 2>/dev/null; then echo "worktree clean: $f"; else echo "worktree MODIFIED: $f"; fi
done

# ---------------------------------------------------------------------------
# Inner script (runs inside the container). Emitted here, not in slurm/inner/.
# ---------------------------------------------------------------------------
INNER=$CREW_GATE_OUT/inner_crew_gate.sh
cat > "$INNER" <<'INNER_EOF'
#!/bin/bash
# Crew gate, inner half. Generated by slurm/repo20260923/gpu_crew_gate.sbatch.
set -uo pipefail
cd "$UPSTREAM"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1   # otherwise Hydra prints one line and hides the stack
OUT=$CREW_GATE_OUT
: > "$GATE_OUT"

echo "=== import check ==="
IMPLOG=$OUT/crew_gate_import.log
VERDICT=$OUT/import_verdict.txt
rm -f "$VERDICT"
# The verdict goes to a file BEFORE app.close(): SimulationApp.close() ends the
# process from C++ and a buffered print never reaches a redirected stdout.
# Run from a file, like every other Isaac script in this repo, not from stdin.
cat > "$OUT/import_check.py" <<'PY'
import os, sys, traceback
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
out = os.environ["CREW_GATE_OUT"]
verdict = "import FAIL"
try:
    import berkeley_humanoid_lite.tasks, bhl_robust.tasks  # noqa: F401
    import gymnasium as gym
    from isaaclab.managers import ObservationTermCfg
    from isaaclab_tasks.utils import parse_env_cfg
    ids = sorted(i for i in gym.registry if "Crew" in i)
    print("REGISTERED", ids, flush=True)
    clean = True
    for i in ids:
        cfg = parse_env_cfg(i, device="cuda:0", num_envs=8)
        pol = cfg.observations.policy
        n_terms = sum(1 for v in vars(pol).values() if isinstance(v, ObservationTermCfg))
        pair = sorted(k for k, v in vars(pol).items() if v is not None and k.endswith(("_a", "_b")))
        robots = sorted(k for k in vars(cfg.scene) if k.startswith("robot_"))
        clean = clean and not pair and "robot_a" not in robots
        print(f"CONSTRUCTED {i} crew_size={getattr(cfg, 'crew_size', None)} policy_terms={n_terms} "
              f"robots={robots} pair_terms={pair}", flush=True)
    verdict = "import OK" if (len(ids) == 4 and clean) else "import FAIL"
    if len(ids) != 4:
        print(f"expected 4 crew tasks, got {len(ids)}", flush=True)
except Exception:
    traceback.print_exc()
    sys.stderr.flush()
with open(os.path.join(out, "import_verdict.txt"), "w") as f:
    f.write(verdict + "\n")
print("IMPORT", verdict, flush=True)
sys.stdout.flush()
app.close()
PY
# Bounded so a Kit hang here (including inside app.close()) cannot eat the wall:
# a timed-out check leaves no verdict file and is judged "import FAIL" below.
timeout -k 15 900 "$PY" "$OUT/import_check.py" > "$IMPLOG" 2>&1
if [ -f "$VERDICT" ] && grep -q "import OK" "$VERDICT"; then
    echo "import ok"; echo "import OK" >> "$GATE_OUT"
else
    echo "IMPORT FAILED -- crew configclasses do not register/construct (see $IMPLOG)" >&2
    grep -A30 "Traceback" "$IMPLOG" | tail -30 >&2
    echo "import FAIL" >> "$GATE_OUT"
fi

# The four train-path checks, as slurm/inner/crew_validate.sh runs them.
for spec in "CoopLift-BHL-Cube-Crew3-v0 3" "CoopLift-BHL-Cube-Crew4-v0 4" \
            "CoopLift-BHL-Cube-Crew3-Depth-v0 3" "CoopLift-BHL-Cube-Crew4-Depth-v0 4"; do
    set -- $spec
    echo "=== $1 | $(date) ==="
    export TASK="$1" RUN_NAME="_gate-crew$2" SEED=0 NUM_ENVS=8 MAX_ITER=2
    export OVERRIDE_FILE="$OUT/overrides/gate-crew.txt"
    echo "env.stage_lift_on_pinch=true" > "$OVERRIDE_FILE"
    VLOG="$OUT/$1.log"
    # -k: Kit traps SIGTERM into its shutdown path, which can itself hang; a
    # plain timeout would then wait forever and the wall, not this, ends the job.
    if timeout -k 15 1500 bash "$REPO/slurm/inner/train.sh" > "$VLOG" 2>&1; then
        echo "$1 PASS" >> "$GATE_OUT"
    else
        echo "$1 FAIL" >> "$GATE_OUT"
        grep -A25 "Traceback" "$VLOG" | tail -25 >&2
        grep -hoE "^[A-Za-z_.]*(Error|Exception): .{0,200}" "$VLOG" | tail -3 >&2
    fi
    grep -hE "^train.sh: " "$VLOG" | tail -2
done

echo "=== verdicts ==="; cat "$GATE_OUT"
INNER_EOF
chmod +x "$INNER"

bhl_exec "$INNER"
echo "=== container exit $? | $(date) ==="

# ---------------------------------------------------------------------------
# Verdict from the files, not from exit codes.
# ---------------------------------------------------------------------------
"$PYH" - "$CREW_GATE_OUT" "$GATE_OUT" "${SLURM_JOB_ID:-}" "$(hostname)" "$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)" \
    "$(git -C "$REPO" branch --show-current 2>/dev/null)" "$REPO" "${TASKS[@]}" <<'PY'
import datetime, json, os, re, subprocess, sys
out, gate_out, job_id, node, head, branch, repo = sys.argv[1:8]
tasks = sys.argv[8:]
lines = open(gate_out).read().splitlines() if os.path.exists(gate_out) else []
verdicts = {}
for ln in lines:
    parts = ln.rsplit(" ", 1)
    if len(parts) == 2:
        verdicts[parts[0]] = parts[1]
reasons = []
imp_log = os.path.join(out, "crew_gate_import.log")
imp_text = open(imp_log, errors="replace").read() if os.path.exists(imp_log) else ""
registered = re.findall(r"^REGISTERED (.*)$", imp_text, re.M)
constructed = re.findall(r"^CONSTRUCTED (.*)$", imp_text, re.M)
import_ok = verdicts.get("import") == "OK"
if not import_ok:
    reasons.append(f"import verdict {verdicts.get('import', 'MISSING')}")
result = {
    "job_id": job_id, "node": node, "date_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "repo_head": head, "branch": branch, "stack": "v51",
    "gate_override": "env.stage_lift_on_pinch=true", "num_envs": 8, "max_iterations": 2, "seed": 0,
    "import": {"verdict": verdicts.get("import", "MISSING"), "registered": registered[-1] if registered else None,
               "constructed": constructed, "log": imp_log},
    "tasks": {},
}
for t in tasks:
    log = os.path.join(out, f"{t}.log")
    text = open(log, errors="replace").read() if os.path.exists(log) else ""
    iters = len(re.findall(r"Learning iteration", text))
    eplen = re.findall(r"Mean episode length: ([0-9.]+)", text)
    errs = re.findall(r"^[A-Za-z_.]*(?:Error|Exception): .{0,300}", text, re.M)
    v = verdicts.get(t, "MISSING")
    result["tasks"][t] = {"verdict": v, "iterations_logged": iters,
                          "mean_episode_length": float(eplen[-1]) if eplen else None,
                          "last_error": errs[-1] if errs else None, "log": log if text else None}
    if v != "PASS":
        reasons.append(f"{t}: {v}" + (f" ({errs[-1][:120]})" if errs else ""))
    elif iters < 1:
        reasons.append(f"{t}: PASS recorded but no logged iteration in {log}")
passed = import_ok and all(r["verdict"] == "PASS" and r["iterations_logged"] >= 1 for r in result["tasks"].values()) \
    and len(result["tasks"]) == 4
result["verdict"] = "PASS" if passed else "FAIL"
result["reasons"] = reasons
for f in ("src/bhl_robust/tasks/coop_crew_generated.py", "src/bhl_robust/tasks/coop_depth_env_cfg.py",
          "src/bhl_robust/tasks/coop_lift_env_cfg.py", "src/bhl_robust/tasks/depth_env_cfg.py",
          "scripts/train.py", "slurm/inner/train.sh"):
    rc = subprocess.run(["git", "-C", repo, "diff", "--quiet", "--", f]).returncode
    result.setdefault("worktree", {})[f] = "clean" if rc == 0 else "modified"
with open(os.path.join(out, "crew_gate.json"), "w") as fh:
    json.dump(result, fh, indent=2)
print(f"{'task':40} {'verdict':8} {'iters':>5} {'ep_len':>7}  last error")
print(f"{'import':40} {result['import']['verdict']:8}")
for t, r in result["tasks"].items():
    print(f"{t:40} {r['verdict']:8} {r['iterations_logged']:>5} {str(r['mean_episode_length']):>7}  {r['last_error'] or ''}")
for r in reasons:
    print("PROBLEM:", r)
print("CREW-GATE RESULT: " + ("PASS (import OK, 4/4 crew tasks logged a training iteration under the gate settings)"
                             if passed else "FAIL (see PROBLEM lines and crew_gate.json)"))
sys.exit(0 if passed else 1)
PY
rc=$?
echo "=== done: t=${SECONDS}s | $(date) ==="
exit $rc
