#!/bin/bash
#SBATCH --job-name=v2-hands
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Repo-wide campaign (2026-09-23): cube-to-shelf hand-height diagnostic.
#
# Measurement for the task author, not a training run. The pinch crouch needs
# 28-30 Nm against the 6 Nm effort limit (21402699, results/repo-gpu-20260923/
# spawn_pose/), so the object has to be re-placed for a STANDING spawn. This
# job puts both robots in the upstream standing legs with the arms at the
# _PINCH_JOINT_POS values (`--init_joint_pose standing_pinch_arms`), root z 0,
# and PD-holds for 60 policy steps (4 envs) on CubeToShelf-Blind and its
# gripper twin, recording per step and per robot the z (and xy offset from the
# base) of the hand end bodies and the object's xyz. The number the author
# needs is the hand z at step 1 (first policy step, before the hold topples,
# which 21402699 put at step ~22) next to the current object spawn z
# (GRASP_Z = 0.30, `bhl_robust.reach_band`, imported by task_v2_env_cfg.py)
# and delta = hand_z - object_z, the amount the object must be raised by.
# task_v2_env_cfg.py is NOT changed.
#
# Stack: v60 (Isaac Sim 6.0 / Lab 3.0b2), the TaskV2 stack (see
# gpu_spawn_pose.sbatch). Blind variants need no cameras. Prior spawn probes
# took 2-4 min for 2-3 Kit boots; 40 min covers two 900 s inner timeouts plus
# boot-gate waits and still leaves the reducer time to print the RESULT lines.
#
# PASS means both JSONs exist, are complete, and carry a measured hand height;
# the final lines 'V2-HANDS RESULT: ...' are the measurement.
set -uo pipefail
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH" 2>/dev/null || true
export NUM_ENVS=${NUM_ENVS:-4}
export PROBE_STEPS=${PROBE_STEPS:-60}
export SPAWN_DIAG_OUT=${SPAWN_DIAG_OUT:-$REPO/results/repo-gpu-20260923/spawn_hands}
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS SPAWN_DIAG_OUT"
mkdir -p "$SPAWN_DIAG_OUT"
source "$REPO/slurm/inner/v60_boot_gate.sh"

# Exact ids from src/bhl_robust/tasks/__init__.py:
#   ("CubeToShelf", CUBE_VARIANTS) and ("CubeToShelfGrip", CUBE_GRIPPER_VARIANTS),
#   id = f"TaskV2-BHL-{_task}-{_vis.capitalize()}-v0"
TASKS=(TaskV2-BHL-CubeToShelf-Blind-v0 TaskV2-BHL-CubeToShelfGrip-Blind-v0)
COND=pd_hold__standing_pinch_arms

echo "=== node: $(hostname) | v2 hand height | stack $BHL_STACK | envs $NUM_ENVS | steps $PROBE_STEPS | $(date) ==="
cd "$REPO" && echo "repo HEAD $(git rev-parse --short HEAD 2>/dev/null) branch $(git branch --show-current 2>/dev/null)"
# Remove stale outputs so PASS can only come from this job's JSONs.
for t in "${TASKS[@]}"; do rm -f "$SPAWN_DIAG_OUT/${t}__"*.json; done

for t in "${TASKS[@]}"; do
    # Serialise each Isaac Sim 6.0 boot, not just the first (see v60_boot_gate.sh).
    v60_boot_gate
    echo "=== task $t | $(date) ==="
    TASK="$t" bhl_exec "$REPO/slurm/repo20260923/inner_v2_hands.sh"
    echo "=== task $t exit $? | $(date) ==="
done
echo "=== all boots done: $(date) ==="

# PASS/FAIL and the measurement from the JSONs, not from exit codes. The
# 'V2-HANDS RESULT:' lines are printed last, one per task.
REQ=""
for t in "${TASKS[@]}"; do REQ="$REQ $SPAWN_DIAG_OUT/${t}__${COND}.json"; done
PYH=$WORKSPACE/venv/bin/python; [ -x "$PYH" ] || PYH=$(command -v python3)
PYTHONPATH="$REPO/src:${PYTHONPATH:-}" "$PYH" - "$PROBE_STEPS" $REQ <<'PY'
import json, os, re, sys
steps = int(sys.argv[1]); req = sys.argv[2:]
# GRASP_Z as task_v2_env_cfg.py reads it (from bhl_robust.reach_band; import-light).
try:
    from bhl_robust.reach_band import GRASP_Z
except Exception:
    src = open(os.path.join(os.environ.get("REPO", "."), "src/bhl_robust/reach_band.py")).read()
    GRASP_Z = float(re.search(r"^GRASP_Z\s*=\s*([0-9.]+)", src, re.M).group(1))
ok = True
results = []
for p in req:
    task = os.path.basename(p).split("__")[0]
    if not os.path.exists(p):
        print(f"MISSING {os.path.basename(p)}"); ok = False
        results.append((task, None, GRASP_Z, None)); continue
    try:
        d = json.load(open(p))
    except Exception as exc:
        print(f"UNREADABLE {os.path.basename(p)}: {exc}"); ok = False
        results.append((task, None, GRASP_Z, None)); continue
    hh = d.get("hand_height") or {}
    s0 = (hh.get("step0") or {}).get("mean"); s1 = (hh.get("step1") or {}).get("mean")
    oz = d.get("configured_object_spawn_z")
    if oz is None:
        oz = GRASP_Z
    elif abs(oz - GRASP_Z) > 1e-6:
        print(f"NOTE {task}: configured object spawn z {oz} != reach_band.GRASP_Z {GRASP_Z}; using the configured value")
    good = (isinstance(d.get("verdict"), str) and not d["verdict"].startswith("ERROR")
            and len(d.get("timeseries", [])) == steps + 1 and s1 is not None)
    if not good:
        ok = False
    ff = {k: v.get("step") for k, v in d.get("first_fire", {}).items()}
    print(f"{task} {d.get('condition')} init_joint_pose={d.get('init_joint_pose')} {d.get('verdict')}  "
          f"first_fire={ff}  implied_ep_len={d.get('implied_episode_length_per_env')}  "
          f"{'complete' if good else 'INCOMPLETE'}")
    print(f"  hand bodies: {d.get('hand_bodies')}")
    for lbl in ("step0", "step1"):
        blk = hh.get(lbl) or {}
        print(f"  {lbl}: hand z mean={blk.get('mean')} min={blk.get('min')} max={blk.get('max')} "
              f"per_robot={blk.get('per_robot')}")
    ts = d.get("timeseries") or []
    if len(ts) > 1:
        r1 = ts[1]
        print(f"  step1 object_pos={r1.get('object_pos')}  object_xy_off_from_base_w={r1.get('object_xy_off_from_base_w')}")
        for r in ("robot_a", "robot_b"):
            if r in r1:
                print(f"  step1 {r}: hand_z={r1[r].get('hand_z')} hand_xy_off_w={r1[r].get('hand_xy_off_w')} "
                      f"gripper_z={r1[r].get('gripper_z')} base_z={r1[r].get('base_z')}")
    for line in d.get("diagnosis", [])[-2:]:
        print(f"      {line}")
    results.append((task, s1, oz, None if s1 is None else round(s1 - oz, 5)))
print("V2-HANDS SUMMARY: " + ("PASS (hand height measured for every task)" if ok
                             else "FAIL (missing/incomplete JSON or no hand body matched)"))
print("delta = hand_z(step 1, mean over robots x hands x envs) - object spawn z; positive = raise the object")
for task, s1, oz, delta in results:
    f = lambda v: "NA" if v is None else f"{v:.4f}"
    print(f"V2-HANDS RESULT: {task} hand_z={f(s1)} object_z={f(oz)} delta={f(delta)}")
sys.exit(0 if ok else 1)
PY
rc=$?
exit $rc
