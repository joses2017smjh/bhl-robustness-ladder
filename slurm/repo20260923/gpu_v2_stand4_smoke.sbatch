#!/bin/bash
#SBATCH --job-name=v2-stand4-smoke
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --exclude=dgxh-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# CubeToShelfStand4 SMOKE (2026-10-01, docs/SOLUTIONS_2026-10-01.md section 4,
# row C2). Gate for gpu_v2_stand4_train.sbatch (submit that with
# --dependency=afterok:<this job>). CubeToShelfStand4 ("roll-proof lift") is a
# DIFFERENT task from CubeToShelfStand3 and CubeToShelf and is never compared
# with their numbers. Every term, constant and reason: src/bhl_robust/tasks/stand4_mdp.py.
#
# Task: TaskV2-BHL-CubeToShelfStand4-Blind-v0, v60, ENABLE_CAMERAS=1 as in its
# training launcher. Three parts, one job:
#   (A) probe  slurm/repo20260923/inner_v2_stand4_smoke.sh: 16 envs, reset, 30
#              zero-action steps, then the cube teleported (i) 2.5 m up into free
#              space for 1 step, (ii) onto robot_a's left hand-link origin for
#              4 steps and (iii) after a fresh reset, onto a deck for 16 steps
#              (stand4_mdp.seat_pose_local: 10 envs flat on both decks, 4 rolled
#              90 deg about x as the control, 2 tilted 10 deg, consistency only).
#              Seed 100. SCRIPTED (no policy): wiring, not behaviour.
#              -> probe_<job>.json
#   (B) run    slurm/inner/train.sh, the real training entrypoint (Hydra, pickled
#              env cfg, Stand3's runner): 1024 envs, 20 iterations, seed 100, run
#              name v2-cubetoshelfstand4-blind-s100-j<job>-smoke under
#              logs/rsl_rl/task_v2/ (a directory ending in -smoke). Never a result.
#   (C) verdict on the host, stand4_mdp.smoke_verdict (pure, unit-tested in
#              tests/test_stand4.py), from the probe JSON, the run's TensorBoard /
#              console tables, its params/*.yaml and this job's training log.
# Seed 100 for both parts; the scored seeds 0 and 1 are refused below.
#
# PREDECLARED SMOKE RULE. PASS iff ALL of:
#   A1  the built env's reward manager holds lifting_object =
#       stand4_mdp:stand4_object_is_lifted x 15.0 and placed =
#       stand_mdp:success_bonus x 5000.0; its termination manager success =
#       stand4_mdp:stand4_cube_placed; its curriculum manager lift_height,
#       upright_gate and the seven Stand4 diagnostics.
#   A2  Stand3 vs Stand4 cfg.to_dict() differ in exactly the declared keys
#       (stand4_mdp.CFG_DIFF_ALLOWED: both robots' usd_path, the cube's contact
#       reporting, the cube_contact sensor, the lift and success funcs and their
#       new params, the seven diagnostics) -- every one of them and nothing else.
#   A3  env_0 of the built scene has a hand_collider prim with
#       PhysicsCollisionAPI on both hands of both robots; Stand4's robots spawn
#       from the overlay, Stand3's from berkeley_humanoid_lite.usd.
#   A4  the cube sensor's force matrix is (16, 1, 26, 3), filter count 26.
#   A5  after every zero-action step, env._bhl_s4 holds every key of
#       stand4_mdp.CACHE_KEYS, per env, and is the same dict as env.extras["stand4"].
#   A6  cube 2.5 m up: on every env that did not reset, every robot-link force
#       is < 1.0 N and the cube reads released.
#   A7  cube on robot_a's left hand-link origin: within 4 steps the
#       arm_left_hand_link column reaches >= 1.0 N on >= 1 env that did not
#       reset; on every such env and step the release predicate is the negation
#       of "some column >= 1.0 N", and equals the termination's cached reading.
#   A8  (added on review 2026-10-02, before this smoke's first run with it) the
#       success chain fires with the real sensor: cube at rest on a deck after a
#       fresh reset, 16 zero-action steps, each env followed until its first
#       reset (a success resets its env on the step it fires; that step
#       counts), recomputed on the host by stand4_mdp.seat_stage_eval from the
#       raw per-step records: >= 1 flat env fires `success`; on every firing the
#       cache reads seated, released and placed_now, hold >= 12, and the
#       `placed` reward of that step is > 0; on every followed step success ==
#       (hold >= 12), (placed > 0) == success and placed_now == seated &
#       released & tilt <= 8 deg; no rolled-90-deg env ever fires, and >= 1 of
#       its followed steps is seated and released with tilt > 8 deg (blocked by
#       the tilt clause alone). Tilt is the cube's own z axis vs world up
#       (stand4_mdp.TILT_NOTE), so the rolled cube reads 90 deg; under a face
#       tilt (most vertical axis) this control would fire. The 10-deg group
#       falls flat within a few steps and is checked for consistency only.
#   B1  the run logs every key in stand4_mdp.SMOKE_TAGS (TensorBoard or the
#       console table): Curriculum/lift_height (minimal_height),
#       Curriculum/upright_gate, the seven Curriculum/<diagnostic> (pinch
#       distance, cube tilt, lowest-corner clearance, roll-proof share, lift-paid
#       share, released share, max robot-link force), Episode_Termination/success,
#       Episode_Reward/lifting_object, Episode_Reward/placed.
#   B2  >= 20 distinct "Learning iteration" lines in the training log.
#   B3  no "train.sh: FAILED", no Python traceback, no PhysX buffer-overflow line
#       in the training log.
#   B4  stand4_mdp.config_check(params/agent.yaml, params/env.yaml) passes: the
#       exact check the training launcher's RUNNER-CHECK runs at start.
# INCOMPLETE (no verdict) iff the probe JSON is missing or its status is not ok,
# or the training log or the run's params are missing. The job exits 0 iff PASS;
# the verdict is verdict_<job>.json, never an exit code. The sha256 of the files
# this job read (the working tree when it STARTED) is in sha256_<job>.txt and in
# the verdict; the training must run on the same bytes, and
# gpu_v2_stand4_train.sbatch enforces it (BYTES-CHECK: submit it with
# --dependency=afterok:<this job> --export=ALL,STAND4_SMOKE_JOB=<this job>; it
# refuses to start unless this verdict reads PASS and the Stand4 files listed in
# sha256_<job>.txt, all but __init__.py and these two launcher scripts, match).
# Outputs: results/repo-gpu-20260923/stand4_2026-10-01-smoke/ and the -smoke run dir.
set -euo pipefail
export BHL_STACK=v60 ENABLE_CAMERAS=1
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
JOB=${SLURM_JOB_ID:?run under sbatch}
export OV_CACHE=/scratch/$USER/ov-cache-${JOB}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${JOB}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH" 2>/dev/null || true
source "$REPO/slurm/inner/v60_boot_gate.sh"

export STAND4_OUT=$REPO/results/repo-gpu-20260923/stand4_2026-10-01-smoke STAND4_JOB=$JOB
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS STAND4_OUT STAND4_JOB"
mkdir -p "$STAND4_OUT"
VERDICT="$STAND4_OUT/verdict_${JOB}.json"
TRAINLOG="$STAND4_OUT/train_${JOB}.log"
HASHES="$STAND4_OUT/sha256_${JOB}.txt"
for f in "$VERDICT" "$STAND4_OUT/probe_${JOB}.json" "$TRAINLOG" "$HASHES"; do
    [ -e "$f" ] && { echo "STAND4-SMOKE: refusing to overwrite $f"; exit 1; }
done
# pinned: nothing inherited from the submitting shell changes this smoke
unset OVERRIDE_FILE TRAIN_SCRIPT BHL_LEGACY_YAW
export TASK=TaskV2-BHL-CubeToShelfStand4-Blind-v0 SEED=100
case "$SEED" in 0|1) echo "STAND4-SMOKE: seed $SEED is a scored Stand4 seed; refused"; exit 1;; esac
export RUN_NAME="v2-cubetoshelfstand4-blind-s${SEED}-j${JOB}-smoke"
LOGROOT=$UPSTREAM/logs/rsl_rl/task_v2
HOSTPY=$WORKSPACE/venv/bin/python          # host interpreter with tensorboard
if [ -n "$(find "$LOGROOT" -maxdepth 1 -type d -name "*_${RUN_NAME}" 2>/dev/null)" ]; then
    echo "STAND4-SMOKE: refusing to reuse run name $RUN_NAME"; exit 1
fi
STAMP="$STAND4_OUT/.start-${JOB}"
: > "$STAMP"
HEAD_SHA=$(git -C "$REPO" rev-parse --short HEAD)
( cd "$REPO" && sha256sum src/bhl_robust/tasks/stand4_mdp.py src/bhl_robust/tasks/task_v2_env_cfg.py \
      src/bhl_robust/tasks/__init__.py src/bhl_robust/tasks/stand_mdp.py \
      src/bhl_robust/tasks/coop_lift_mdp.py src/bhl_robust/tasks/coop_lift_env_cfg.py \
      slurm/repo20260923/gpu_v2_stand4_smoke.sbatch slurm/repo20260923/inner_v2_stand4_smoke.sh \
      assets/cloth/berkeley_humanoid_lite_hand_colliders.usda ) > "$HASHES"
echo "=== node: $(hostname) | STAND4 smoke | $TASK | seed $SEED | HEAD $HEAD_SHA (+ working tree, $HASHES) | $(date) ==="

# (A) probe
v60_boot_gate
NUM_ENVS=16 PROBE_STEPS=30 bhl_exec "$REPO/slurm/repo20260923/inner_v2_stand4_smoke.sh" || true
echo "=== probe done: $(date) ==="

# (B) 20-iteration run of the real training path
v60_boot_gate
NUM_ENVS=1024 MAX_ITER=20 bhl_exec "$REPO/slurm/inner/train.sh" 2>&1 | tee "$TRAINLOG" || true
RUN_DIR=$(find "$LOGROOT" -maxdepth 1 -type d -name "*_${RUN_NAME}" -newer "$STAMP" 2>/dev/null | sort | tail -1 || true)
echo "=== run done: ${RUN_DIR:-<no run dir>} | $(date) ==="

# (C) verdict: stand4_mdp.smoke_verdict on the probe JSON, the run's tags, its
# params and the training log
set +e
REPO="$REPO" OMP_NUM_THREADS=2 "$HOSTPY" - "$STAND4_OUT/probe_${JOB}.json" "${RUN_DIR:-}" "$TRAINLOG" \
    "$VERDICT" "$HEAD_SHA" "$JOB" "$SEED" "$HASHES" <<'PY'
import importlib.util, json, os, sys
probe_p, run_dir, train_log_p, out, head, job, seed, hashes_p = sys.argv[1:9]
spec = importlib.util.spec_from_file_location(
    "stand4_mdp", os.environ["REPO"] + "/src/bhl_robust/tasks/stand4_mdp.py")
s4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s4)

res = {"job": job, "head": head, "seed": int(seed), "run_dir": run_dir or None,
       "task": "TaskV2-BHL-CubeToShelfStand4-Blind-v0",
       "label": "SCRIPTED probe + 20-iteration PPO smoke (never a result)",
       "scope": s4.TASK4_NOTE, "sha256": open(hashes_p).read().splitlines()}
probe = None
if os.path.isfile(probe_p):
    try:
        probe = json.load(open(probe_p))
    except Exception as exc:                                        # noqa: BLE001
        res["probe_error"] = repr(exc)
res["probe_summary"] = None if probe is None else {
    k: probe.get(k) for k in ("status", "stage", "quat_order", "cfg_diff_check", "managers",
                              "hand_colliders", "usd_paths", "sensor", "cache", "diag_values",
                              "free_space", "overlap", "seat", "traceback")}
if probe is not None and isinstance(probe.get("seat"), dict):
    try:
        res["seat_recomputed"] = s4.seat_stage_eval(probe["seat"].get("groups", []),
                                                    probe["seat"].get("records", []))
    except Exception as exc:                                        # noqa: BLE001
        res["seat_recomputed"] = {"error": repr(exc)}
tags = set()
if run_dir and os.path.isdir(run_dir):
    try:
        sc = s4.sm.read_scalars(run_dir)
        tags |= set(sc)
        res["tb_last_iter"] = max([s for s, _ in sc.get(s4.TAG_LEN, [])] or [-1])
    except Exception as exc:                                        # noqa: BLE001
        res["tb_error"] = repr(exc)
train_log = open(train_log_p, errors="replace").read() if os.path.isfile(train_log_p) else None
if train_log is not None:
    tags |= s4.console_tags(train_log)
    res["logged_iterations"] = len(s4.logged_iterations(train_log))
cfg = None
if run_dir:
    a, e = (os.path.join(run_dir, "params", f) for f in ("agent.yaml", "env.yaml"))
    if os.path.isfile(a) and os.path.isfile(e):
        cfg = s4.config_check(open(a).read(), open(e).read())
res["config_check"] = cfg
v = s4.smoke_verdict(probe, tags, train_log, cfg)
res.update(v)
res["tags_seen_stand4"] = sorted(t for t in tags if t.startswith(("Curriculum/", "Episode_")))
if os.path.exists(out):
    sys.exit(f"refusing to overwrite {out}")
json.dump(res, open(out, "w"), indent=1, default=str)
print(f"STAND4-SMOKE-VERDICT: {v['verdict']} | clauses {v['clauses']} | missing {v['missing']} "
      f"| tags_missing {v['tags_missing']} | config_check failed "
      f"{None if cfg is None else cfg['failed']} ({out})")
sys.exit(0 if v["verdict"] == "PASS" else 1)
PY
rc=$?
set -e
rm -f "$STAMP"
echo "=== done rc=$rc: $(date) ==="
exit $rc
