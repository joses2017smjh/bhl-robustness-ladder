#!/bin/bash
#SBATCH --job-name=depth-bench
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --constraint=rtx8000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Repo-wide GPU campaign (2026-09-23): re-run the ray-cast depth validation and
# throughput bench, because the published numbers have no evidence file
# (results/depth_bench.txt is absent; REPO_TASKS LOC-09):
#
#   100 % finite pixels, 2.9 % mean relative error over 3,136 px (row-flipped
#   102 %) against the closed-form flat-plane depth image; 21,844 vs 21,490
#   env-steps/s at 4,096 envs (physics / + 48x48 depth), a 1.6 % depth cost;
#   13,971 / 13,989 / 13,956 at 2,048 envs (physics / 32x32 / 64x64).
#   Sources: docs/REPORT.md "Validation first", docs/FINDINGS.md section 5,
#   commit b34b99a.
#
# Stack: v51 (Isaac Sim 5.1 / Lab 2.3.2). The numbers were measured on 2026-08-21
# on v51 (b34b99a, 48_depth_train / 49_depth_clip), _env.sh calls v51 "every
# published number", and native_quat() leaves CAM_ROT untouched on v51.
#
# GPU: rtx8000 (partition gpu). b34b99a says the measuring GPUs were Turing with
# RT cores and the docs never name the card, so the Turing card that also fits
# 4,096 envs is the closest match; absolute env-steps/s are hardware-bound. The
# card actually used is written into the bench file's NODE line. To trade
# comparability for queue time, override on the command line:
#   sbatch --partition=gpu,dgxh,ampere --constraint="a40|rtx8000|l40s|h100|h200|h200-70g" ...
#
# Steps are depth_bench.sh's, run directly (same scripts, same arguments, same
# order) so each Isaac boot gets its own timeout and exit code instead of
# `|| true` with no limit -- a hang must not eat the hour before the verdict
# prints. One extra step after the four: 2,048 envs with a 32x32 camera, the
# REPORT row depth_bench.sh never ran. It is optional for completeness.
#
# Outputs (results/repo-gpu-20260923/depth-bench/):
#   depth_bench.txt            BENCH_OUT: NODE, CHECK/PASS/FAIL, RESULT/DEPTH, STEP lines
#   depth_bench_summary.json   scripts/bench/depth_bench_summary.py over it
#   step_<name>.log            full stdout/stderr of each Isaac step (unfiltered)
#   provenance.txt, worktree.patch   HEAD, dirty files, diff of the depth/terrain code
#
# Job PASS = valid measurement: validation printed its flat-plane finite
# fraction, its closed-form verdict line and the rough-ground departure, and
# all four depth_bench.sh RESULT rows (plus DEPTH lines for both depth rows) are
# present. Per-claim PASS / MISMATCH / NOT_MEASURED (10 % relative tolerance)
# is the scientific result and is printed on the same final line; it does not
# decide PASS.
set -uo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
source "$REPO/slurm/inner/v60_boot_gate.sh"

OUT=$REPO/results/repo-gpu-20260923/depth-bench
# Absolute on purpose: the Isaac steps cd into $UPSTREAM, and both bench scripts
# open BENCH_OUT relative to their cwd.
export BENCH_OUT=$OUT/depth_bench.txt
SUMMARY=$OUT/depth_bench_summary.json
mkdir -p "$OUT"
# Both bench scripts APPEND. Remove this directory's previous outputs so PASS
# can only come from this job's lines.
rm -f "$BENCH_OUT" "$SUMMARY" "$OUT"/step_*.log "$OUT"/inner_*.sh

HOST_PY=$WORKSPACE/venv/bin/python; [ -x "$HOST_PY" ] || HOST_PY=$(command -v python3)
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 | tr ',' ' ' | tr -s ' ')
HEAD=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
DEPTH_CODE=(src/bhl_robust/tasks/depth_env_cfg.py src/bhl_robust/tasks/terrain_env_cfg.py
            src/bhl_robust/terrains src/bhl_robust/tasks/__init__.py src/bhl_robust/quat_order.py
            scripts/bench/depth_validate.py scripts/bench/depth_raycast.py slurm/inner/depth_bench.sh)
DIRTY=no; git -C "$REPO" diff --quiet HEAD -- "${DEPTH_CODE[@]}" 2>/dev/null || DIRTY=yes

echo "=== node: $(hostname) | depth bench | stack $BHL_STACK | gpu ${GPU:-unknown} | $(date) ==="
{
    echo "job ${SLURM_JOB_ID:-local} host $(hostname) gpu ${GPU:-unknown} stack $BHL_STACK date $(date -Is)"
    echo "repo HEAD $HEAD branch $(git -C "$REPO" branch --show-current 2>/dev/null) depth-code-dirty $DIRTY"
    echo "upstream HEAD $(git -C "$UPSTREAM" rev-parse --short HEAD 2>/dev/null)"
    echo "--- git status (depth/terrain code)"
    git -C "$REPO" status --short -- "${DEPTH_CODE[@]}" 2>/dev/null
    echo "--- git diff --stat HEAD (depth/terrain code)"
    git -C "$REPO" diff --stat HEAD -- "${DEPTH_CODE[@]}" 2>/dev/null
} | tee "$OUT/provenance.txt"
git -C "$REPO" diff HEAD -- "${DEPTH_CODE[@]}" > "$OUT/worktree.patch" 2>/dev/null || true

echo "NODE | host=$(hostname) gpu=${GPU:-unknown} job=${SLURM_JOB_ID:-local} stack=$BHL_STACK head=$HEAD dirty=$DIRTY" > "$BENCH_OUT"

# Inner script (inside the container): written here so no new file or forwarded
# variable is needed. REPO, PY, UPSTREAM are set by bhl_exec; BENCH_OUT and
# TMPDIR are in BHL_FORWARD_VARS. Mirrors depth_bench.sh's cd + PYTHONPATH.
INNER=$OUT/inner_${SLURM_JOB_ID:-local}.sh
cat > "$INNER" <<'EOF'
#!/bin/bash
set -uo pipefail
cd "$UPSTREAM"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
limit=$1; shift
timeout -k 30 "$limit" "$PY" "$@"
EOF

DEADLINE=$(( 55 * 60 ))   # leave >= 5 min of the 1 h for the summary
run_step() {  # run_step <name> <timeout_s> <script> [args...]
    local name=$1 limit=$2 rc t0 remaining log
    shift 2
    remaining=$(( DEADLINE - SECONDS ))
    if (( remaining < 240 )); then
        echo "STEP | name=$name rc=skipped_deadline secs=0" >> "$BENCH_OUT"
        echo "--- $name: skipped, ${remaining}s left before the deadline"
        return
    fi
    (( limit > remaining - 60 )) && limit=$(( remaining - 60 ))
    V60_BOOT_LOCK="$WORKSPACE/.v51-boot.lock" v60_boot_gate
    log=$OUT/step_${name}.log
    echo "=== step $name (timeout ${limit}s) | $(date) ==="
    t0=$SECONDS
    bhl_exec "$INNER" "$limit" "$@" > "$log" 2>&1
    rc=$?
    local salvaged=0
    # depth_validate.py appends its lines to BENCH_OUT only at the very end, so
    # a crash or timeout in its rough-terrain half would lose the flat-plane
    # numbers it had already printed. Recover them from its own stdout.
    if [[ $name == validate* ]] && ! grep -q "flat-ground depth matches" "$BENCH_OUT"; then
        salvaged=$(grep -cE "^(CHECK|PASS|FAIL) +\|" "$log")
        grep -E "^(CHECK|PASS|FAIL) +\|" "$log" >> "$BENCH_OUT"
    fi
    echo "STEP | name=$name rc=$rc secs=$(( SECONDS - t0 )) salvaged_stdout_lines=$salvaged" >> "$BENCH_OUT"
    echo "--- $name: rc=$rc in $(( SECONDS - t0 ))s (log $log)"
    [ "$rc" -eq 0 ] || tail -n 15 "$log"
}

# depth_bench.sh, verbatim arguments and order.
run_step validate_res64_n16    600 "$REPO/scripts/bench/depth_validate.py" --res 64 --num_envs 16
run_step physics_2048          480 "$REPO/scripts/bench/depth_raycast.py" --no_camera --num_envs 2048 --steps 60
run_step depth64_2048          480 "$REPO/scripts/bench/depth_raycast.py" --num_envs 2048 --res 64 --steps 60
run_step physics_4096          540 "$REPO/scripts/bench/depth_raycast.py" --no_camera --num_envs 4096 --steps 60
run_step depth48_4096          540 "$REPO/scripts/bench/depth_raycast.py" --num_envs 4096 --res 48 --steps 60
# Extra: the REPORT's 2,048-env 32x32 row (not in depth_bench.sh; optional).
run_step depth32_2048          480 "$REPO/scripts/bench/depth_raycast.py" --num_envs 2048 --res 32 --steps 60

echo "=== bench lines ==="
grep -E "^(NODE|CHECK|PASS|FAIL|RESULT|DEPTH|STEP)" "$BENCH_OUT" || echo "(no bench lines)"

# ------------------------------------------------------------- PASS / FAIL
# From the bench file via the parser's JSON, not from any step's exit code.
echo "=== summary vs documented claims (10 % tolerance) ==="
"$HOST_PY" "$REPO/scripts/bench/depth_bench_summary.py" "$BENCH_OUT" --json "$SUMMARY" --tol 0.10 \
    | grep -v "^DEPTH-BENCH "
"$HOST_PY" - "$SUMMARY" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as exc:  # noqa: BLE001
    print(f"DEPTH-BENCH FAIL | no readable summary ({exc!r})")
    sys.exit(1)
c = d.get("claim_counts", {})
claims = " ".join(f"{x['claim']}={x['verdict']}" for x in d.get("claims", []))
gpu = d.get("node", {}).get("gpu", "unknown")
counts = f"PASS={c.get('PASS', 0)} MISMATCH={c.get('MISMATCH', 0)} NOT_MEASURED={c.get('NOT_MEASURED', 0)}"
if d.get("measurement_complete") is True and not d.get("missing"):
    print(f"DEPTH-BENCH PASS | measurement complete | claims {counts} | gpu={gpu} | {claims}")
    sys.exit(0)
print(f"DEPTH-BENCH FAIL | incomplete: {'; '.join(d.get('missing', ['?']))} | claims {counts} | gpu={gpu}")
sys.exit(1)
PY
rc=$?
echo "=== done: $(date) ==="
exit $rc
