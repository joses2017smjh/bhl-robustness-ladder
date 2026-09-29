#!/bin/bash
#SBATCH --job-name=stand3-replay
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --exclude=dgxh-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# CubeToShelfStand3 REPLAY DIAGNOSTIC (2026-09-28). REPORT-ONLY: no gate, and it
# changes no recorded verdict -- Stand3 (job 21443327) is NEGATIVE by its
# predeclared training rule (last-200 success 0.0020 / 0.0012; stands 2/2,
# shifts 0/2, stable 2/2) and stays so whatever this prints. It measures WHY the
# cube is never seated, so that a Stand4 is funded (or not) on a measured
# failure mode. CubeToShelfStand3 is a different, easier task than CubeToShelf
# and Stand2 and is never compared with them.
#
# PREDECLARED 2026-09-28, before any replay ran (literals in
# scripts/bench/stand3_replay.py, equal to stand_mdp.py's, asserted by
# tests/test_stand3_replay.py and again in the job):
#
# WHAT: the two FINAL checkpoints, model_7999.pt of
#   task_v2/2026-09-27_17-56-20_v2-cubetoshelfstand3-blind-s0
#   task_v2/2026-09-27_20-03-28_v2-cubetoshelfstand3-blind-s1
# under their OWN training config, TaskV2-BHL-CubeToShelfStand3-Blind-v0, built
# from the source each run recorded (<run>/git/bhl-robustness-ladder.diff:
# commit 493c123 / 9a89001 -- identical under src/ -- plus the uncommitted src/
# diff captured at training), rebuilt below into a node-local snapshot and put
# first (alone) on PYTHONPATH. That is the config BEFORE the 2026-09-28
# quaternion fix: the cube spawns with the raw literal rot=(1, 0, 0, 0), which
# Isaac Lab 3.0 reads as (x, y, z, w) = 180 deg about x, and object_tilt_l2
# indexes wxyz. The policy observation has no cube orientation
# (object_pos_in_root uses the ROBOT quaternion), so the fix would not change
# the policy's inputs; the pin is kept because this is the policy as trained.
#   SOURCE-CHECK  the six Stand3 task files of the snapshot byte-identical to
#                 `git show <commit>:src/...`, the raw literal present, the fix
#                 absent; sha256 manifest written next to the results.
#   CONFIG-CHECK  the live env cfg (dumped after construction, built the way
#                 scripts/train.py built it: resolve_task_config, no overrides)
#                 against the run's params/env.yaml. Allowlisted: seed,
#                 scene.num_envs, sim.device, sim.log_dir. Diffs in rewards,
#                 curriculum and UI/recording fields are reported, not gated;
#                 any other diff stops the replay (STAND3_REPLAY_CONFIG_CHECK=
#                 report records the diffs and replays anyway, but a replay that
#                 is not the config as trained never gets a reading).
# PROTOCOL: v60, --enable_cameras (stack parity with training; the blind task
#   has no camera), 32 envs, reset seed 1000 (training seeded 0 / 1), 3 episodes
#   per env = 96 per checkpoint, deterministic actions (rsl-rl 5 MLPModel with
#   stochastic_output=False = the mean; checked by acting twice on the first
#   observation), observation noise ON and reset randomisation as trained. State
#   is logged after physics and before the auto-reset (termination-manager
#   hook), so each terminal step is recorded. Label: learned (PPO checkpoint,
#   deterministic mean action); cube and robot state read from the simulator
#   (oracle) only to classify; nothing scripted.
# LOGGED per step per env: cube x, y, z, velocity and speed; cube face tilt and
#   which body axis is up (a 90-deg roll in the hands shows as an axis change);
#   hand-cube distance for all 4 hands (hand-link origin to the cube surface, in
#   the cube frame) and each hand link's net contact force; robot base x/y/z;
#   pelvis tilt (the fall check's), lean along world x (toward a deck = pelvis
#   roll for a robot facing +/-y) and y, pelvis yaw; over_deck / seated /
#   hold counter / success / fallen / time_out as the env computed them.
# THRESHOLDS (float32, as the env's terms):
#   over_deck  |x| > DECK_EDGE 0.17 and z > OFF_FLOOR_Z 0.34
#   seated     0.19 < |x| < 0.45, |y| < 0.10, |z - DECK_SEATED_Z 0.57| < 0.02,
#              speed < SEAT_SPEED 0.05 m/s; success = SEAT_HOLD_STEPS 12 in a row
#   reach      REQUIRED_SHIFT 0.19; MEASURED_SHIFT_ARMS_TWIST 0.119 (feet planted,
#              arms + <= 0.5 rad pelvis twist); MEASURED_SHIFT_ARMS 0.044
#   contact    PROXY (not a filtered hand-cube sensor): hand-link origin within
#              HAND_NEAR_M 0.05 m of the cube surface AND net force >= 1.0 N
# CLASSES of every over_deck step, tested in this order (exhaustive, exclusive):
#   MOVING              speed >= 0.05
#   HELD_ABOVE          still and z - 0.57 >= 0.02 (held above seated height)
#   SEATED              the env's seated test
#   RESTING_NOT_SEATED  still, not above, not seated: LOW (0.57 - z >= 0.02),
#                       X_BAND (|x| outside (0.19, 0.45)), Y_BAND (|y| >= 0.10)
# PER EPISODE: max |cube x| while z > 0.34 ("carried", primary) and over all
#   steps, against 0.19 and 0.119 (the reach-limit hypothesis); how it ended;
#   longest SEATED run; base shuffle, lean and twist at the carried maximum.
# FUNDING READING (report-only, per checkpoint; not a gate):
#   S = share of the 96 episodes ending in the success termination;
#   R = share whose carried max |x| >= 0.19; M = median carried max |x|;
#   D = among the R episodes, share with any SEATED or RESTING_NOT_SEATED step.
#   R1 PLACES       S >= 0.10 -> the deterministic policy places; no Stand4 on a
#                   placement-failure argument.
#   R2 REACH-SHORT  S < 0.10, R < 0.10, M >= 0.044 (MEASURED_SHIFT_ARMS): shifted
#                   but never to the seat band -> a reach problem; a Stand4 that
#                   closes the reach gap (seat band within the demonstrated shift,
#                   or a lean/shuffle incentive). Reach-limit sub-reading, reported:
#                   M < 0.119 = inside the feet-planted arm+twist envelope
#                   (consistent with the reach-limit hypothesis); M >= 0.119 =
#                   partly beyond it (leans or shuffles, not far enough).
#   R3 NO-SHIFT     S < 0.10, R < 0.10, M < 0.044: the shift itself is not learned;
#                   deck distance is not shown to bind -> no Stand4 on this evidence.
#   R4 LOWERING/HOLD  S < 0.10, R >= 0.10, D < 0.50: reaches |x| >= 0.19 but is
#                   not set down -> a Stand4 aimed at lowering / release.
#   R5 SETTLE       S < 0.10, R >= 0.10, D >= 0.50: set down but never seated
#                   still for 12 steps -> a Stand4 aimed at the settle.
#   A Stand4 is justified by this diagnostic only if BOTH checkpoints give the
#   same R2, R4 or R5. R1, R3, a disagreement, or an INCOMPLETE replay (episodes
#   unfinished, no trace, CONFIG-CHECK not PASS, source not pre-fix) -> no Stand4
#   on this evidence. Also reported (no reading): share of MOVING over-deck steps
#   at or above the seat window (a swaying held cube counts as MOVING).
#
# Verdict lines (SOURCE-CHECK, DRY-RUN, CONFIG-CHECK, REPLAY, READING, FUNDING)
# are printed from files and JSON by the host interpreter, never from exit codes
# (Isaac exits 0 after a crash). Outputs go to a NEW per-job directory
#   results/repo-gpu-20260923/stand3_replay_2026-09-28/<jobid>/
# (refused if it exists): <label>.args.json, src_manifest_<label>.txt,
# <label>.meta.json, <label>.trace.npz (float32, compressed), <label>.live_env.yaml,
# <label>.live_agent.yaml, replay_<label>.json, reading.json.
#
# Modes:
#   STAND3_REPLAY_DRYRUN=1 bash <this>   login node, no Isaac, no GPU: snapshot +
#       SOURCE-CHECK, the script's dry-run preflight on both real runs, then a
#       SYNTHETIC trace per label through classify/report (every line says
#       SYNTHETIC). Output under ${TMPDIR:-/tmp}/stand3-replay-dryrun/ unless
#       STAND3_REPLAY_OUT_DIR is set; results/ is never touched.
#   STAND3_REPLAY_CONFIG_CHECK=report    see CONFIG-CHECK above (default gate).
# Submit after the maintenance window (GPU nodes reserved until 2026-10-01 16:00):
#   sbatch --begin=2026-10-01T16:00 slurm/repo20260923/gpu_stand3_replay.sbatch
set -euo pipefail
export BHL_STACK=v60 ENABLE_CAMERAS=1
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
DRY=${STAND3_REPLAY_DRYRUN:-0}
CC_MODE=${STAND3_REPLAY_CONFIG_CHECK:-gate}
JOBTAG=${SLURM_JOB_ID:-local$$}
HOSTPY=$WORKSPACE/venv/bin/python          # host interpreter (numpy, torch, yaml)
SCRIPT=$REPO/scripts/bench/stand3_replay.py
LOGROOT=$UPSTREAM/logs/rsl_rl/task_v2
LABELS=(s0 s1)
declare -A RUNDIR=([s0]=2026-09-27_17-56-20_v2-cubetoshelfstand3-blind-s0
                   [s1]=2026-09-27_20-03-28_v2-cubetoshelfstand3-blind-s1)
declare -A SYNTH=([s0]=reach_hold [s1]=setdown_jitter)   # dry run only (SYNTHETIC)
PINNED=(bhl_robust/tasks/coop_lift_env_cfg.py bhl_robust/tasks/coop_lift_mdp.py
        bhl_robust/tasks/stand_mdp.py bhl_robust/tasks/task_v2_env_cfg.py
        bhl_robust/tasks/task_v2_mdp.py bhl_robust/quat_order.py)

if [ "$DRY" = 1 ]; then
    RES_ROOT=${STAND3_REPLAY_OUT_DIR:-${TMPDIR:-/tmp}/stand3-replay-dryrun}
    SNAPBASE=${STAND3_REPLAY_SNAP_ROOT:-${TMPDIR:-/tmp}}
else
    RES_ROOT=${STAND3_REPLAY_OUT_DIR:-$REPO/results/repo-gpu-20260923/stand3_replay_2026-09-28}
    setup_node_cache
    export OV_CACHE=/scratch/$USER/ov-cache-${JOBTAG}
    export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${JOBTAG}
    mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH"
    SNAPBASE=$TMPDIR                       # node-local scratch, bound into the container
fi
OUTDIR=$RES_ROOT/$JOBTAG
if [ -e "$OUTDIR" ]; then echo "REPLAY: refusing to overwrite $OUTDIR"; exit 1; fi
mkdir -p "$OUTDIR"
SNAPROOT=$SNAPBASE/stand3-replay-src-$JOBTAG
trap 'rm -rf "$SNAPROOT"' EXIT
echo "=== node: $(hostname) | Stand3 replay | dry=$DRY config_check=$CC_MODE | HEAD $(git -C "$REPO" rev-parse --short HEAD) | out $OUTDIR | $(date) ==="

# build_snapshot LABEL: the run's own src/ tree (commit + recorded uncommitted
# src/ diff) under $SNAPROOT/LABEL/src, then SOURCE-CHECK. Uses git only (no
# patch/filterdiff on the compute nodes). The recorded diff has no final newline
# (rsl-rl strips it) and its last hunk is malformed without the `echo`.
# GIT_CEILING_DIRECTORIES keeps `git apply` from adopting an enclosing repo
# (Humanoid_Lite itself is one), so paths resolve against the snapshot.
build_snapshot() {
    local label=$1 run=$LOGROOT/${RUNDIR[$1]} dest=$SNAPROOT/$1 rec commit f bad=""
    rec=$run/git/bhl-robustness-ladder.diff
    commit=$(sed -n 2p "$rec" 2>/dev/null || true)
    if ! [[ "$commit" =~ ^[0-9a-f]{40}$ ]]; then
        echo "SOURCE-CHECK $label: FAIL (no commit recorded in $rec)"; return 1
    fi
    mkdir -p "$dest"
    git -C "$REPO" archive "$commit" src | tar -x -C "$dest"
    { sed '1,/^--- git diff ---$/d' "$rec"; echo; } > "$dest.diff"
    ( cd "$dest" && GIT_CEILING_DIRECTORIES="$SNAPROOT" git apply --include='src/*' --whitespace=nowarn "$dest.diff" ) \
        || bad="$bad recorded-src-diff-did-not-apply"
    for f in "${PINNED[@]}"; do
        cmp -s <(git -C "$REPO" show "$commit:src/$f") "$dest/src/$f" || bad="$bad differs-from-$commit:$f"
    done
    grep -qF 'init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z), rot=(1.0, 0.0, 0.0, 0.0))' \
        "$dest/src/bhl_robust/tasks/coop_lift_env_cfg.py" || bad="$bad raw-cube-literal-absent"
    if grep -q 'native_quat(_OBJECT_ROT_WXYZ)' "$dest/src/bhl_robust/tasks/coop_lift_env_cfg.py"; then
        bad="$bad quaternion-fix-present"
    fi
    ( cd "$dest" && find src -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum ) > "$OUTDIR/src_manifest_$label.txt"
    if [ -n "$bad" ]; then
        echo "SOURCE-CHECK $label: FAIL ($commit:$bad)"; return 1
    fi
    echo "SOURCE-CHECK $label: PASS (commit ${commit:0:7} + the run's recorded src/ diff; ${#PINNED[@]} Stand3 task files byte-identical to git; raw cube literal rot=(1,0,0,0) present = BEFORE the quaternion fix; manifest src_manifest_$label.txt)"
}

# jline FILE EXPR: print one line computed from a JSON file by the host python.
jline() {
    "$HOSTPY" - "$@" <<'PY' || true
import json, sys
path, what, label = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    m = json.load(open(path))
except Exception as exc:  # noqa: BLE001
    print(f"{what} {label}: INCOMPLETE (no readable {path}: {exc!r})"); sys.exit(0)
if what == "CONFIG-CHECK":
    cc = m.get("config_check") or {}
    src = m.get("source") or {}
    print(f"CONFIG-CHECK {label}: {cc.get('verdict')} (mode {cc.get('mode')}; gated diffs {cc.get('n_gated')}, "
          f"reported {cc.get('n_reported')}, allowlisted {[r['path'] for r in cc.get('allowlisted', [])]}; "
          f"live cube rot {src.get('object_init_rot_live')} order {src.get('quat_order')} pre_quat_fix "
          f"{src.get('pre_quat_fix')}; bhl_robust from {src.get('bhl_robust_file')})")
    for r in (cc.get("gated_diffs") or [])[:10]:
        print(f"  gated diff {r['path']}: trained {r['trained']} live {r['live']}")
    if m.get("stopped"):
        print(f"REPLAY {label}: INCOMPLETE ({m['stopped']})")
    det = m.get("determinism") or {}
    if det:
        print(f"DETERMINISM {label}: repeat_equal={det.get('repeat_equal')} ({det.get('action_mode')}); "
              f"contact proxy available={(m.get('contact_proxy') or {}).get('available')}; "
              f"convention {m.get('convention_check')}")
PY
}

fail=0
for label in "${LABELS[@]}"; do
    build_snapshot "$label" || { fail=1; continue; }
    "$HOSTPY" "$SCRIPT" make-args --label "$label" --run-dir "$LOGROOT/${RUNDIR[$label]}" \
        --src-root "$SNAPROOT/$label/src" --out-prefix "$OUTDIR/$label" --config-check "$CC_MODE" \
        --out "$OUTDIR/$label.args.json"
    OMP_NUM_THREADS=2 "$HOSTPY" "$SCRIPT" dry-run --args-file "$OUTDIR/$label.args.json" | grep -E "^DRY-RUN" \
        || { echo "DRY-RUN $label: FAIL (see above)"; fail=1; continue; }
    if [ "$DRY" = 1 ]; then
        "$HOSTPY" "$SCRIPT" synth --scenario "${SYNTH[$label]}" --out-prefix "$OUTDIR/$label" --num-envs 4
    else
        echo "=== Isaac replay $label | $(date) ==="
        export PYTHONPATH="$SNAPROOT/$label/src" BENCH_OUT="$OUTDIR/$label"
        bhl_exec "$REPO/slurm/repo20260923/inner_stand3_replay.sh" || true
        unset PYTHONPATH BENCH_OUT
        jline "$OUTDIR/$label.meta.json" CONFIG-CHECK "$label"
    fi
    if [ -f "$OUTDIR/$label.trace.npz" ] && [ -f "$OUTDIR/$label.meta.json" ]; then
        "$HOSTPY" "$SCRIPT" classify --trace "$OUTDIR/$label.trace.npz" --meta "$OUTDIR/$label.meta.json" \
            --out "$OUTDIR/replay_$label.json" --tag "$label" \
            || echo "REPLAY $label: INCOMPLETE (classify failed on $OUTDIR/$label.trace.npz)"
    else
        echo "REPLAY $label: INCOMPLETE (no trace written; Isaac exits 0 after a crash -- see $OUTDIR/$label.error.txt and the log above)"
    fi
done

# Pooled funding reading (report-only): both checkpoints must give the same R2/R4/R5.
"$HOSTPY" "$SCRIPT" report --replays "$OUTDIR/replay_s0.json" "$OUTDIR/replay_s1.json" \
    --tags s0 s1 --out "$OUTDIR/reading.json"
echo "=== done: $(date) | $OUTDIR ==="
[ "$fail" = 0 ] && [ -f "$OUTDIR/reading.json" ]
