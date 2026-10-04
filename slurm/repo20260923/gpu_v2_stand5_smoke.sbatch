#!/bin/bash
#SBATCH --job-name=v2-stand5-smoke
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
# CubeToShelfStand5 SMOKE (2026-10-03; SLURM_JOBS.md "User approval recorded
# 2026-10-03 09:55", item (S)). Gate for gpu_v2_stand5_train.sbatch (submit that
# with --dependency=afterok:<this job> --export=ALL,STAND5_SMOKE_JOB=<this job>).
# Stand5 = CubeToShelfStand4 + exactly two changes (src/bhl_robust/tasks/stand5_mdp.py):
# (i) the actor and the critic observe the cube's own z axis in each robot's root
# frame (object_zaxis_a/b, ORACLE like object_pos_a/b); (ii) the lift curriculum
# promotes on Stand4's roll-proof lift condition instead of centre height + pinch.
#
# Task: TaskV2-BHL-CubeToShelfStand5-Blind-v0, v60, ENABLE_CAMERAS=1 as in its
# training launcher. Stand4's smoke settings exactly (seed 100, 16-env probe,
# 1024 envs x 20 iterations), so the dumped params compare with Stand4's smoke
# 21506757 key for key. Three parts, one job:
#   (A) probe  slurm/inner/inner_v2_stand5_smoke.sh: 16 envs, seed 100. Stand4's
#              and Stand5's configs instantiated (Stand4's never built), Stand5
#              built; then (obs) the cube at rest in free space in five known
#              orientations, one zero-action step: every env's observation tail,
#              both terms, the raw quaternions and Isaac Lab's own matrix_from_quat
#              reference; (curriculum) after a fresh reset the cube in four pose
#              groups, one step, then the curriculum term and Stand4's
#              (coop_lift_mdp.lift_height_curriculum) called directly on chosen
#              envs at known levels and pinch distances, raw inputs recorded;
#              (steps) 30 zero-action steps. SCRIPTED (no policy): wiring, not
#              behaviour. -> probe_<job>.json
#   (B) run    slurm/inner/train.sh, the real training entrypoint (Hydra, pickled
#              env cfg, Stand3's runner): 1024 envs, 20 iterations, seed 100, run
#              name v2-cubetoshelfstand5-blind-s100-j<job>-smoke under
#              logs/rsl_rl/task_v2/ (a directory ending in -smoke). Never a result.
#   (C) host   stand5_mdp.smoke_verdict5 (pure, unit-tested in tests/test_stand5.py)
#              on the probe JSON, the run's TensorBoard / console tables, its
#              params/*.yaml, this job's training log, and Stand4's reference
#              smoke 21506757 (its verdict, sha256 list, params/*.yaml and log).
# Seed 100 for both parts; the scored seeds 0 and 1 are refused below.
#
# PREDECLARED SMOKE RULE. PASS iff ALL of:
#   A1  the built env's managers: reward, termination and curriculum terms are
#       Stand4's (names and order, from Stand4's config; lifting_object =
#       stand4_mdp:stand4_object_is_lifted x 15.0, placed = stand_mdp:success_bonus
#       x 5000.0, success = stand4_mdp:stand4_cube_placed); lift_height =
#       stand5_mdp:stand5_lift_height_curriculum with Stand4's parameters
#       (lifting_object, step 0.02, 0.04 -> 0.06, target 0.35) plus clearance 0.02
#       and tilt_max_deg 15.0, where Stand4's is coop_lift_mdp:lift_height_curriculum;
#       each observation group is Stand4's terms followed by object_zaxis_a,
#       object_zaxis_b (stand5_mdp:object_zaxis_in_root, robot_a / robot_b, no
#       noise, no scale, clip +/-100).
#   A2  Stand4 vs Stand5 cfg.to_dict() differ in exactly the declared keys
#       (stand5_mdp.CFG_DIFF_ALLOWED5: the four new observation terms, the
#       curriculum func and its two new params) -- every one and nothing else.
#   A3  observation widths: the built groups' (term, shape) tables equal the
#       tables in Stand4's smoke 21506757 training log with object_zaxis_a (3,)
#       and object_zaxis_b (3,) appended: policy 194 -> 200, critic 206 -> 212.
#   A4  after the obs step, each group's last 6 observation columns equal the two
#       terms evaluated on the same state (|diff| <= 1e-5 on every env that did
#       not reset): the terms land last, unscaled, without noise.
#   A5  axis order, recomputed on the host from the raw quaternions: the stack
#       stores (x, y, z, w) (quat_order); the cube's world z axis reads identity
#       (0,0,1), +90 deg about x (0,-1,0), +90 deg about y (1,0,0), 180 deg about
#       x (0,0,-1), yaw 30 deg (0,0,1) (|diff| <= 1e-3), every case on >= 1 env
#       that did not reset; both terms equal R_robot^T z from Isaac Lab's
#       matrix_from_quat and from this repo's kernels on the stored numbers
#       (<= 1e-4), unit length; control: the same stored numbers read in the
#       other order miss the x+90 and x180 vectors by > 0.5.
#   A6  curriculum, recomputed on the host from the raw records: the roll-proof
#       flags recomputed from the raw state equal the probe's, and read true /
#       false / false on the free-space flat / free-space 20-deg / edge-on-plinth
#       groups; every direct call's level, the minimal_height it wrote into
#       lifting_object and env._bhl_lift_h equal coop's update on the ROLL-PROOF
#       share of its envs; Stand4's function on the same state and level equals
#       the update on centre > spawn + level AND pinch < 0.20; one call promotes
#       under Stand5 only (roll-proof envs, pinch forced far) and one under
#       Stand4 only (centre high, not roll-proof, pinch forced 0), each on >= 1
#       env; one call ran on int32 env ids (the dtype the env's reset passes).
#   B1  the run logs every key of stand4_mdp.SMOKE_TAGS (TensorBoard or console):
#       Curriculum/lift_height, Curriculum/upright_gate, Stand4's seven
#       Curriculum/<diagnostic>, Episode_Termination/success,
#       Episode_Reward/lifting_object, Episode_Reward/placed.
#   B2  >= 20 distinct "Learning iteration" lines in the training log.
#   B3  no "train.sh: FAILED", no Python traceback, no PhysX buffer-overflow line
#       in the training log.
#   B4  stand5_mdp.config_check5(params/agent.yaml, params/env.yaml) passes: the
#       exact check the training launcher's RUNNER-CHECK runs at start (Stand4's
#       config_check clauses plus both observation groups and the curriculum).
#   B5  the run's params/env.yaml differs from Stand4's smoke 21506757
#       params/env.yaml in exactly CFG_DIFF_ALLOWED5 (every key, nothing else),
#       and params/agent.yaml in exactly run_name (YAML SafeLoader parse, python
#       tags kept as data).
#   B6  the training log's observation tables equal A3's expectation, and the
#       actor's / critic's first layers read 200 / 212 inputs.
#   C1  Stand4's sources are the bytes Stand4's smoke 21506757 hashed:
#       stand4_mdp.py, stand_mdp.py, coop_lift_mdp.py, coop_lift_env_cfg.py and
#       the hand-collider overlay; task_v2_env_cfg.py up to its "# --- stand5 ---"
#       marker hashes to that smoke's whole-file hash (Stand4 and every other task
#       byte-identical); and every file hashed at this job's start but
#       tasks/__init__.py still has those bytes at the verdict.
#   C2  the exported/recorded config: the run's params/agent.yaml and env.yaml are
#       copied to agent_<job>.yaml / env_<job>.yaml in the outputs directory,
#       byte-identical (sha256).
# INCOMPLETE (no verdict) iff the probe JSON is missing or its status is not ok,
# the training log or the run's params are missing, or Stand4's reference smoke
# 21506757 (verdict PASS, sha256 list, params, training log) is missing; the
# verdict records the sha256 of each reference file it read. The job
# exits 0 iff PASS; the verdict is verdict_<job>.json, never an exit code. The
# sha256 of the files this job read (the working tree when it STARTED) is in
# sha256_<job>.txt and in the verdict; the training must run on the same bytes,
# and gpu_v2_stand5_train.sbatch enforces it (BYTES-CHECK: it refuses to start
# unless this verdict reads PASS and the Stand5 files listed in sha256_<job>.txt,
# all but tasks/__init__.py and these two launcher scripts, match).
# Outputs: results/repo-gpu-20260923/stand5_2026-10-03-smoke/ and the -smoke run dir.
set -euo pipefail
export BHL_STACK=v60 ENABLE_CAMERAS=1
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
JOB=${SLURM_JOB_ID:?run under sbatch}
export OV_CACHE=/scratch/$USER/ov-cache-${JOB}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${JOB}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH" 2>/dev/null || true
source "$REPO/slurm/inner/v60_boot_gate.sh"

export STAND5_OUT=$REPO/results/repo-gpu-20260923/stand5_2026-10-03-smoke STAND5_JOB=$JOB
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS STAND5_OUT STAND5_JOB"
mkdir -p "$STAND5_OUT"
VERDICT="$STAND5_OUT/verdict_${JOB}.json"
TRAINLOG="$STAND5_OUT/train_${JOB}.log"
HASHES="$STAND5_OUT/sha256_${JOB}.txt"
for f in "$VERDICT" "$STAND5_OUT/probe_${JOB}.json" "$TRAINLOG" "$HASHES" \
         "$STAND5_OUT/agent_${JOB}.yaml" "$STAND5_OUT/env_${JOB}.yaml"; do
    [ -e "$f" ] && { echo "STAND5-SMOKE: refusing to overwrite $f"; exit 1; }
done
# pinned: nothing inherited from the submitting shell changes this smoke
unset OVERRIDE_FILE TRAIN_SCRIPT BHL_LEGACY_YAW
export TASK=TaskV2-BHL-CubeToShelfStand5-Blind-v0 SEED=100
case "$SEED" in 0|1) echo "STAND5-SMOKE: seed $SEED is a scored Stand5 seed; refused"; exit 1;; esac
export RUN_NAME="v2-cubetoshelfstand5-blind-s${SEED}-j${JOB}-smoke"
LOGROOT=$UPSTREAM/logs/rsl_rl/task_v2
HOSTPY=$WORKSPACE/venv/bin/python          # host interpreter with tensorboard
if [ -n "$(find "$LOGROOT" -maxdepth 1 -type d -name "*_${RUN_NAME}" 2>/dev/null || true)" ]; then
    echo "STAND5-SMOKE: refusing to reuse run name $RUN_NAME"; exit 1
fi
STAMP="$STAND5_OUT/.start-${JOB}"
: > "$STAMP"
HEAD_SHA=$(git -C "$REPO" rev-parse --short HEAD)
( cd "$REPO" && sha256sum src/bhl_robust/tasks/stand5_mdp.py src/bhl_robust/tasks/stand4_mdp.py \
      src/bhl_robust/tasks/task_v2_env_cfg.py src/bhl_robust/tasks/__init__.py \
      src/bhl_robust/tasks/stand_mdp.py src/bhl_robust/tasks/coop_lift_mdp.py \
      src/bhl_robust/tasks/coop_lift_env_cfg.py src/bhl_robust/quat_order.py \
      assets/cloth/berkeley_humanoid_lite_hand_colliders.usda \
      slurm/repo20260923/gpu_v2_stand5_smoke.sbatch slurm/inner/inner_v2_stand5_smoke.sh ) > "$HASHES"
echo "=== node: $(hostname) | STAND5 smoke | $TASK | seed $SEED | HEAD $HEAD_SHA (+ working tree, $HASHES) | $(date) ==="

# (A) probe
v60_boot_gate
NUM_ENVS=16 PROBE_STEPS=30 bhl_exec "$REPO/slurm/inner/inner_v2_stand5_smoke.sh" || true
echo "=== probe done: $(date) ==="

# (B) 20-iteration run of the real training path
v60_boot_gate
NUM_ENVS=1024 MAX_ITER=20 bhl_exec "$REPO/slurm/inner/train.sh" 2>&1 | tee "$TRAINLOG" || true
RUN_DIR=$(find "$LOGROOT" -maxdepth 1 -type d -name "*_${RUN_NAME}" -newer "$STAMP" 2>/dev/null | sort | tail -1 || true)
echo "=== run done: ${RUN_DIR:-<no run dir>} | $(date) ==="

# (C) host checks and the verdict (stand5_mdp.smoke_verdict5)
set +e
REPO="$REPO" OMP_NUM_THREADS=2 "$HOSTPY" - "$STAND5_OUT/probe_${JOB}.json" "${RUN_DIR:-}" "$TRAINLOG" \
    "$VERDICT" "$HEAD_SHA" "$JOB" "$SEED" "$HASHES" "$STAND5_OUT" <<'PY'
import hashlib, importlib.util, json, os, shutil, sys
probe_p, run_dir, train_log_p, out, head, job, seed, hashes_p, out_dir = sys.argv[1:10]
repo = os.environ["REPO"]
spec = importlib.util.spec_from_file_location(
    "stand5_mdp", repo + "/src/bhl_robust/tasks/stand5_mdp.py")
s5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s5)
s4 = s5.s4


def read(p):
    return open(p, errors="replace").read() if p and os.path.isfile(p) else None


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


res = {"job": job, "head": head, "seed": int(seed), "run_dir": run_dir or None, "task": s5.TASK5_ID,
       "label": "SCRIPTED probe + 20-iteration PPO smoke (never a result)",
       "scope": s5.TASK5_NOTE, "sha256": open(hashes_p).read().splitlines(),
       "stand4_reference": {"job": s5.STAND4_REF_SMOKE_JOB, "run": s5.STAND4_REF_RUN}}
probe = None
if os.path.isfile(probe_p):
    try:
        probe = json.load(open(probe_p))
    except Exception as exc:                                        # noqa: BLE001
        res["probe_error"] = repr(exc)
res["probe_summary"] = None if probe is None else {
    k: probe.get(k) for k in ("status", "stage", "quat_order", "cfg_diff_check", "managers",
                              "stand4_expected", "steps", "traceback")}
order = (probe or {}).get("quat_order")
if probe is not None and order in ("xyzw", "wxyz"):
    for key, fn in (("obs", s5.obs_stage_eval), ("curriculum", s5.curriculum_stage_eval)):
        try:
            res[f"{key}_recomputed"] = fn(probe.get(key) or {}, order)
        except Exception as exc:                                    # noqa: BLE001
            res[f"{key}_recomputed"] = {"error": repr(exc)}
tags = set()
if run_dir and os.path.isdir(run_dir):
    try:
        sc = s4.sm.read_scalars(run_dir)
        tags |= set(sc)
        res["tb_last_iter"] = max([s for s, _ in sc.get(s4.TAG_LEN, [])] or [-1])
        res["tb_lift_level"] = sc.get(s4.TAG_LIFT_LEVEL, [])[-5:]
    except Exception as exc:                                        # noqa: BLE001
        res["tb_error"] = repr(exc)
train_log = read(train_log_p)
if train_log is not None:
    tags |= s4.console_tags(train_log)
    res["logged_iterations"] = len(s4.logged_iterations(train_log))
    res["train_obs_tables"] = s5.obs_tables(train_log)
    res["train_mlp_in_features"] = s5.mlp_in_features(train_log)
# Stand4's reference smoke (21506757): only a PASSing reference is used
ref_dir = os.path.join(repo, s5.STAND4_REF_DIR)
ref_run = os.path.join(repo, s5.STAND4_REF_RUN)
j4 = s5.STAND4_REF_SMOKE_JOB
try:
    ref_ok = json.load(open(os.path.join(ref_dir, f"verdict_{j4}.json"))).get("verdict") == "PASS"
except Exception as exc:                                            # noqa: BLE001
    ref_ok = False
    res["stand4_reference_error"] = repr(exc)
ref_sha = read(os.path.join(ref_dir, f"sha256_{j4}.txt")) if ref_ok else None
ref_log = read(os.path.join(ref_dir, f"train_{j4}.log")) if ref_ok else None
ref_env = read(os.path.join(ref_run, "params", "env.yaml")) if ref_ok else None
ref_agent = read(os.path.join(ref_run, "params", "agent.yaml")) if ref_ok else None
res["stand4_reference"].update(verdict_pass=ref_ok, have_sha=ref_sha is not None,
                               have_log=ref_log is not None,
                               have_params=ref_env is not None and ref_agent is not None)
# what was compared against, by content (these reference files are not all in git)
res["stand4_reference"]["files_sha256"] = {
    p: (sha(p) if os.path.isfile(p) else None)
    for p in (os.path.join(ref_dir, f"verdict_{j4}.json"), os.path.join(ref_dir, f"sha256_{j4}.txt"),
              os.path.join(ref_dir, f"train_{j4}.log"), os.path.join(ref_run, "params", "env.yaml"),
              os.path.join(ref_run, "params", "agent.yaml"))}
cfg = diff = recorded = None
a = e = None
if run_dir:
    a, e = (read(os.path.join(run_dir, "params", f)) for f in ("agent.yaml", "env.yaml"))
if a is not None and e is not None:
    cfg = s5.config_check5(a, e)
    if ref_env is not None and ref_agent is not None:
        try:
            diff = s5.dump_diff(e, ref_env, a, ref_agent)
        except Exception as exc:                                    # noqa: BLE001
            res["dump_diff_error"] = repr(exc)
    rec = {}
    for name in ("agent", "env"):                  # the exported/recorded config
        src = os.path.join(run_dir, "params", f"{name}.yaml")
        dst = os.path.join(out_dir, f"{name}_{job}.yaml")
        if os.path.exists(dst):
            sys.exit(f"refusing to overwrite {dst}")
        shutil.copyfile(src, dst)
        rec[name] = {"path": dst, "sha256": sha(dst), "source": src, "source_sha256": sha(src)}
    recorded = {"ok": all(v["sha256"] == v["source_sha256"] for v in rec.values()), "files": rec}
stand4_bytes = (s5.stand4_sources_check(repo, ref_sha, start_sha_text=open(hashes_p).read())
                if ref_sha is not None else None)
res.update(config_check=cfg, dump_diff=diff, recorded=recorded, stand4_sources=stand4_bytes)
v = s5.smoke_verdict5(probe, tags, train_log, cfg, diff, ref_log, stand4_bytes, recorded)
res.update(v)
res["tags_seen"] = sorted(t for t in tags if t.startswith(("Curriculum/", "Episode_")))
if os.path.exists(out):
    sys.exit(f"refusing to overwrite {out}")
json.dump(res, open(out, "w"), indent=1, default=str)
print(f"STAND5-SMOKE-VERDICT: {v['verdict']} | clauses {v['clauses']} | missing {v['missing']} "
      f"| tags_missing {v['tags_missing']} | config_check failed "
      f"{None if cfg is None else cfg['failed']} | dump diff "
      f"{None if diff is None else (diff['env']['unexpected'], diff['env']['missing'], diff['agent']['keys'])} "
      f"| stand4 sources failed {None if stand4_bytes is None else stand4_bytes['failed']} ({out})")
sys.exit(0 if v["verdict"] == "PASS" else 1)
PY
rc=$?
set -e
rm -f "$STAMP"
echo "=== done rc=$rc: $(date) ==="
exit $rc
