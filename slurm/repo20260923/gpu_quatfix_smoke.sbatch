#!/bin/bash
#SBATCH --job-name=quatfix-smoke
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --exclude=dgxh-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# QUATFIX smoke (predeclared 2026-09-28, before any run of the fix). Does the
# payload-quaternion fix hold in a real Isaac Lab 3.0 (v60) build?
#
# The bug (SLURM_JOBS.md, 2026-09-28 re-verification): the coop-lift / TaskV2 /
# crew payload spawned with the raw literal rot=(1, 0, 0, 0), which v60 stores
# (x, y, z, w) = 180 deg about x, and coop_lift_mdp.object_tilt_l2 read q[:, 1],
# q[:, 2] as x, y (on v60: y, z). The two cancelled at rest. Fix (this commit):
# the payload rot goes through native_quat (coop_lift_env_cfg._object,
# coop_crew_env_cfg._crew_payload, coop_crew_generated.py via
# scripts/gen_crew_cfg.py) and object_tilt_l2 reads through unpack_wxyz. v51 is
# unchanged (native_quat is the identity there; the reward reads the same two
# columns -- proven bit-for-bit on the login node).
#
# Task: TaskV2-BHL-CubeToShelfStand3-Blind-v0 (the task whose diagnostics exposed
# the bug), v60, ENABLE_CAMERAS=1 as in its training launcher. Two parts, one job:
#   (A) probe  slurm/repo20260923/inner_quatfix_smoke.sh: 16 envs, reset, then 20
#              steps of N(0,1) actions, seed 100. SCRIPTED (no policy): checks the
#              config and the two readers, not behaviour. -> probe_<job>.json
#   (B) run    slurm/inner/train.sh, 1024 envs (the training regime, so the
#              iteration-0 logging is the one Stand3 logged), 20 iterations
#              (TensorBoard flushes every 10 s and the writer is never closed
#              before Isaac's hard exit; 20 iterations leave iteration 0 flushed),
#              seed 100, run name quatfix-smoke-stand3-blind-s100-j<job>.
#              Iteration 0 is read from TensorBoard, falling back to the
#              "Learning iteration 0/" console table (same value, 4 decimals).
# Seeds: 100 for both parts. Not a Stand3 training seed (0, 1; refused below).
#
# PREDECLARED RULE. PASS iff ALL of:
#   A1  probe quat_order == "xyzw" and the built config's object
#       init_state.rot == (0, 0, 0, 1) (the stack reads xyzw; the config writes
#       the xyzw identity).
#   A2  after reset, over all envs: max object_tilt_l2 <= 1e-4 and max
#       stand_mdp._cube_tilt_deg <= 1.0 deg.
#   A3  on quaternions built by Isaac Lab's own quat_from_euler_xyz (identity,
#       90 deg about x, 90 deg about y, 90 deg yaw, 180 deg about x):
#       object_tilt_l2 = 0 / 1 / 1 / 0 / 4 within 1e-3 and _cube_tilt_deg =
#       0 / 90 / 90 / 0 / 180 within 0.5 deg.
#   A4  on the live object quaternion (reset + 20 steps): object_tilt_l2 equals
#       (1 - R22)^2 from Isaac Lab's own matrix_from_quat within 1e-4, and
#       _cube_tilt_deg equals acos(R22) within 0.5 deg, at every check.
#   B1  iteration 0: Curriculum/cube_tilted <= 0.01 (pre-fix: 1.0000 on both
#       Stand3 seeds, 21443327).
#   B2  iteration 0: |Episode_Reward/object_tilt| <= 0.01. Logged value =
#       sum(weight x value x dt) / 20 s over the episodes that ended; at rest ~0
#       (pre-fix both-bugs: -0.000005). With the reward fixed and the spawn NOT
#       fixed the cube reads 4 at rest: -1.5 x 4 x 0.04 = -0.24 per step, about
#       -0.15 at iteration 0 (~12-step episodes; Stand3's still_alive 0.024) and
#       -0.012 even for a 1-step episode -- so B2 separates that case.
# FAIL iff every input is present and any clause fails. INCOMPLETE (no verdict)
# iff the probe JSON is missing or its status is not ok, quat_order is not
# "xyzw" (not a v60 build: nothing tested), or no run dir / either tag at
# iteration 0 in both TensorBoard and the log. Exit codes are never read; the
# verdict is verdict_<job>.json under results/repo-gpu-20260923/quatfix_2026-09-28/.
# Stated up front: A1 + B1 + B2 alone would also pass with the spawn fixed and
# the reward NOT fixed (both read 0 at rest); A3 and A4 are the clauses that
# show the reward reads the right axes on v60.
#
# SCOPE. Every v60 coop-lift / TaskV2 / crew run after this fix is a NEW
# configuration: it is never compared with, pooled with, or used to re-score the
# Stand / Stand2 / Stand3 / CoopLift results recorded before it. This 20-iteration
# run is a smoke, never a training result.
set -euo pipefail
export BHL_STACK=v60 ENABLE_CAMERAS=1
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
JOB=${SLURM_JOB_ID:?run under sbatch}
export OV_CACHE=/scratch/$USER/ov-cache-${JOB}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${JOB}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH" 2>/dev/null || true
source "$REPO/slurm/inner/v60_boot_gate.sh"

export QUATFIX_OUT=$REPO/results/repo-gpu-20260923/quatfix_2026-09-28 QUATFIX_JOB=$JOB
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS QUATFIX_OUT QUATFIX_JOB"
mkdir -p "$QUATFIX_OUT"
VERDICT="$QUATFIX_OUT/verdict_${JOB}.json"
TRAINLOG="$QUATFIX_OUT/train_${JOB}.log"
for f in "$VERDICT" "$QUATFIX_OUT/probe_${JOB}.json" "$TRAINLOG"; do
    [ -e "$f" ] && { echo "QUATFIX-SMOKE: refusing to overwrite $f"; exit 1; }
done
# pinned (review 2026-09-28): nothing inherited from the submitting shell changes this smoke
unset OVERRIDE_FILE TRAIN_SCRIPT BHL_LEGACY_YAW
export TASK=TaskV2-BHL-CubeToShelfStand3-Blind-v0 SEED=100
case "$SEED" in 0|1) echo "QUATFIX-SMOKE: seed $SEED is a Stand3 training seed; refused"; exit 1;; esac
export RUN_NAME="quatfix-smoke-stand3-blind-s${SEED}-j${JOB}"
LOGROOT=$UPSTREAM/logs/rsl_rl/task_v2
HOSTPY=$WORKSPACE/venv/bin/python          # host interpreter with tensorboard
STAMP="$QUATFIX_OUT/.start-${JOB}"
: > "$STAMP"
HEAD_SHA=$(git -C "$REPO" rev-parse --short HEAD)
echo "=== node: $(hostname) | QUATFIX smoke | $TASK | seed $SEED | HEAD $HEAD_SHA | $(date) ==="

# (A) probe
v60_boot_gate
NUM_ENVS=16 PROBE_STEPS=20 bhl_exec "$REPO/slurm/repo20260923/inner_quatfix_smoke.sh" || true
echo "=== probe done: $(date) ==="

# (B) 20-iteration run
v60_boot_gate
NUM_ENVS=1024 MAX_ITER=20 bhl_exec "$REPO/slurm/inner/train.sh" 2>&1 | tee "$TRAINLOG" || true
RUN_DIR=$(find "$LOGROOT" -maxdepth 1 -type d -name "*_${RUN_NAME}" -newer "$STAMP" 2>/dev/null | sort | tail -1 || true)
echo "=== run done: ${RUN_DIR:-<no run dir>} | $(date) ==="

# (C) verdict, from the probe JSON and the run's TensorBoard / log table
set +e
REPO="$REPO" OMP_NUM_THREADS=2 "$HOSTPY" - "$QUATFIX_OUT/probe_${JOB}.json" "${RUN_DIR:-}" "$TRAINLOG" \
    "$VERDICT" "$HEAD_SHA" "$JOB" "$SEED" <<'PY'
import importlib.util, json, math, os, re, sys
probe_p, run_dir, train_log, out, head, job, seed = sys.argv[1:8]
TAG_TILTED, TAG_OT = "Curriculum/cube_tilted", "Episode_Reward/object_tilt"
spec = importlib.util.spec_from_file_location(
    "stand_mdp", os.environ["REPO"] + "/src/bhl_robust/tasks/stand_mdp.py")
sm = importlib.util.module_from_spec(spec); spec.loader.exec_module(sm)

res = {"job": job, "head": head, "seed": int(seed), "run_dir": run_dir or None,
       "task": "TaskV2-BHL-CubeToShelfStand3-Blind-v0",
       "label": "SCRIPTED probe + 20-iteration PPO smoke (not a result)",
       "scope": ("every v60 coop-lift / TaskV2 / crew run after this fix is a NEW configuration, "
                 "never compared with or used to re-score Stand / Stand2 / Stand3 / CoopLift"),
       "clauses": {}, "missing": []}
C = res["clauses"]

# ---- (A) probe
probe = None
if os.path.isfile(probe_p):
    try:
        probe = json.load(open(probe_p))
    except Exception as exc:                                        # noqa: BLE001
        res["missing"].append(f"probe JSON unreadable: {exc}")
else:
    res["missing"].append("probe JSON missing")
if probe is not None and probe.get("status") != "ok":
    res["missing"].append("probe status " + str(probe.get("status")))
    probe = None
res["probe"] = probe
on_v60 = probe is not None and probe.get("quat_order") == "xyzw"
if probe is not None and not on_v60:
    res["missing"].append(f"quat_order {probe.get('quat_order')!r}: not a v60 build")
if on_v60:
    close = lambda a, b, tol: len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))
    C["A1_order_and_cfg_rot"] = probe["cfg_object_rot"] == [0.0, 0.0, 0.0, 1.0]
    r = probe["reset"]
    C["A2_reset_upright"] = r["object_tilt_max"] <= 1e-4 and r["cube_tilt_deg_max"] <= 1.0
    k = probe["known"]
    C["A3_known_rotations"] = (close(k["object_tilt"], k["want_object_tilt"], 1e-3)
                               and close(k["cube_tilt_deg"], k["want_cube_tilt_deg"], 0.5))
    la = probe["live_agreement"]
    C["A4_live_agrees_with_isaac"] = (la["max_abs_err_object_tilt"] <= 1e-4
                                      and la["max_abs_err_tilt_deg"] <= 0.5)

# ---- (B) iteration 0: TensorBoard first, the console table as fallback
it0 = {}
if run_dir and os.path.isdir(run_dir):
    try:
        sc = sm.read_scalars(run_dir)
        for tag in (TAG_TILTED, TAG_OT):
            v = [x for s, x in sc.get(tag, []) if s == 0]
            if v:
                it0[tag] = {"value": float(v[0]), "source": "tensorboard"}
        res["tb_last_iter"] = max([s for s, _ in sc.get(sm.TAG_LEN, [])] or [-1])
    except Exception as exc:                                        # noqa: BLE001
        res["tb_error"] = str(exc)
table = {}
if os.path.isfile(train_log):
    txt = re.sub(r"\x1b\[[0-9;]*m", "", open(train_log, errors="replace").read())
    m = re.search(r"Learning iteration 0/\d+(.*?)(?:Learning iteration 1/|\Z)", txt, re.S)
    if m:
        for tag in (TAG_TILTED, TAG_OT):
            mm = re.search(re.escape(tag) + r":\s+(-?[0-9.]+(?:e-?\d+)?)", m.group(1))
            if mm:
                table[tag] = float(mm.group(1))
res["log_table_it0"] = table
for tag in (TAG_TILTED, TAG_OT):
    if tag not in it0 and tag in table:
        it0[tag] = {"value": table[tag], "source": "log table (4 decimals)"}
    if tag in it0 and tag in table and it0[tag]["source"] == "tensorboard":
        it0[tag]["log_table_agrees"] = abs(it0[tag]["value"] - table[tag]) <= 6e-5
res["iteration0"] = it0
if not run_dir:
    res["missing"].append("no run dir")
for tag in (TAG_TILTED, TAG_OT):
    if tag not in it0:
        res["missing"].append(f"{tag} at iteration 0")
if TAG_TILTED in it0:
    C["B1_cube_tilted_it0"] = it0[TAG_TILTED]["value"] <= 0.01
if TAG_OT in it0:
    C["B2_object_tilt_it0"] = abs(it0[TAG_OT]["value"]) <= 0.01

want = ["A1_order_and_cfg_rot", "A2_reset_upright", "A3_known_rotations",
        "A4_live_agrees_with_isaac", "B1_cube_tilted_it0", "B2_object_tilt_it0"]
if res["missing"] or any(w not in C for w in want):
    verdict = "INCOMPLETE (no verdict: " + "; ".join(res["missing"] or ["clause not evaluated"]) + ")"
elif all(C[w] for w in want):
    verdict = "PASS"
else:
    verdict = "FAIL (" + ", ".join(w for w in want if not C[w]) + ")"
res["verdict"] = verdict
if os.path.exists(out):
    sys.exit(f"refusing to overwrite {out}")
json.dump(res, open(out, "w"), indent=1)
v = lambda t: it0.get(t, {}).get("value")
print(f"QUATFIX-SMOKE-VERDICT: {verdict} | clauses {C} | it0 cube_tilted={v(TAG_TILTED)} "
      f"object_tilt={v(TAG_OT)} | probe order={probe and probe.get('quat_order')} ({out})")
sys.exit(0 if verdict == "PASS" else 1)
PY
rc=$?
set -e
rm -f "$STAMP"
echo "=== done: $(date) ==="
exit $rc
