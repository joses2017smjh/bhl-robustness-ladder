#!/bin/bash
#SBATCH --job-name=spawn-pose
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:55:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Repo-wide campaign (2026-09-23): CubeToShelf spawn-stability diagnostic.
#
# Why training episodes last 5 steps (21352980_0: mean episode length 5.00,
# fallen 1.0) when spawn_quat_probe.json said the configured spawn STANDS.
# For CubeToShelf blind and its gripper twin, 4 envs on the default spawn,
# 200 policy steps each of zero action / default-joint PD hold / N(0,1) action,
# no auto-reset, logging tilt (native and as `fallen` computes it), heights,
# bodies below ground, ankle/shoulder z, foot contact force, termination terms,
# object z. One JSON per task x condition under
#   results/repo-gpu-20260923/spawn_diag/
#
# Stack: v60 (Isaac Sim 6.0 / Lab 3.0b2). The TaskV2 tasks are v60-only
# (task_v2_env_cfg docstring; 86/87/97/99v launchers all export BHL_STACK=v60),
# and 21352980 trained with TaskV2PPORunnerCfg, the rsl-rl 5.x schema that only
# exists on v60. Blind variants need no cameras, so ENABLE_CAMERAS is not set
# (same as 99v_spawn_quat).
#
# PASS means the diagnostic ran and produced a verdict for every required
# task x condition -- NOT that the robot stands. The verdicts are the result.
set -uo pipefail
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH" 2>/dev/null || true
export NUM_ENVS=${NUM_ENVS:-4}
export PROBE_STEPS=${PROBE_STEPS:-200}
export SPAWN_DIAG_OUT=${SPAWN_DIAG_OUT:-$REPO/results/repo-gpu-20260923/spawn_pose}
export Z_OFFSETS="${Z_OFFSETS:-0 -0.05 -0.095 -0.12}"
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS SPAWN_DIAG_OUT Z_OFFSETS"
mkdir -p "$SPAWN_DIAG_OUT"
source "$REPO/slurm/inner/v60_boot_gate.sh"

# Exact ids from src/bhl_robust/tasks/__init__.py:
#   ("CubeToShelf", CUBE_VARIANTS) and ("CubeToShelfGrip", CUBE_GRIPPER_VARIANTS),
#   id = f"TaskV2-BHL-{_task}-{_vis.capitalize()}-v0"
TASKS=(TaskV2-BHL-CubeToShelf-Blind-v0)
REQUIRED_CONDS=(pd_hold__pinch pd_hold__standing)

echo "=== node: $(hostname) | spawn diag | stack $BHL_STACK | envs $NUM_ENVS | steps $PROBE_STEPS | $(date) ==="
cd "$REPO" && echo "repo HEAD $(git rev-parse --short HEAD 2>/dev/null) branch $(git branch --show-current 2>/dev/null)"
# Remove stale outputs so PASS can only come from this job's JSONs.
for t in "${TASKS[@]}"; do rm -f "$SPAWN_DIAG_OUT/${t}__"*.json; done

for t in "${TASKS[@]}"; do
    # Serialise each Isaac Sim 6.0 boot, not just the first (see v60_boot_gate.sh).
    v60_boot_gate
    echo "=== task $t | $(date) ==="
    TASK="$t" bhl_exec "$REPO/slurm/repo20260923/inner_spawn_pose.sh"
    echo "=== task $t exit $? | $(date) ==="
done

# PASS/FAIL from the JSONs, not from exit codes.
REQ=""
for t in "${TASKS[@]}"; do for c in "${REQUIRED_CONDS[@]}"; do REQ="$REQ $SPAWN_DIAG_OUT/${t}__${c}.json"; done; done
PYH=$WORKSPACE/venv/bin/python; [ -x "$PYH" ] || PYH=$(command -v python3)
"$PYH" - "$PROBE_STEPS" $REQ <<'PY'
import json, os, sys
steps = int(sys.argv[1]); req = sys.argv[2:]
ok = True
out_dir = os.path.dirname(req[0])
rows = []
for p in sorted(set(req) | {os.path.join(out_dir, f) for f in os.listdir(out_dir)
                              if f.endswith("__gauss.json")}):
    required = p in req
    if not os.path.exists(p):
        print(f"MISSING {os.path.basename(p)}"); ok = ok and not required; continue
    try:
        d = json.load(open(p))
    except Exception as exc:
        print(f"UNREADABLE {os.path.basename(p)}: {exc}"); ok = ok and not required; continue
    good = (isinstance(d.get("verdict"), str) and not d["verdict"].startswith("ERROR")
            and isinstance(d.get("first_fire"), dict)
            and len(d.get("timeseries", [])) == steps + 1)
    if required and not good:
        ok = False
    ff = {k: v.get("step") for k, v in d.get("first_fire", {}).items()}
    rows.append(f"{'req ' if required else 'extra'} {d.get('task')} {d.get('condition'):8} "
                f"{d.get('verdict')}  first_fire={ff}  "
                f"implied_ep_len={d.get('implied_episode_length_per_env')}  "
                f"max_tilt_native={d.get('max_tilt_native')}  "
                f"{'complete' if good else 'INCOMPLETE'}")
    for line in d.get("diagnosis", [])[-3:]:
        rows.append(f"      {line}")
print("\n".join(rows))
print("SPAWN-DIAG RESULT: " + ("PASS (verdicts produced for every required task x condition)"
                               if ok else "FAIL (missing/incomplete required JSON)"))
sys.exit(0 if ok else 1)
PY
rc=$?
echo "=== done: $(date) ==="
exit $rc
