#!/bin/bash
#SBATCH --job-name=maze-sensors-gif
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# 2026-09-29: the README hero episode (hard 6x6 maze seed 1, biped dr-default-s0, A* turn-then-walk,
# published 73.12 s) re-rendered with the display-only sensor panels:
#   --imu-panel   simulated IM10A-like 10-axis IMU (Hiwonder datasheet noise upper bounds, 100 Hz):
#                 gyro + accelerometer traces under the main view; 6-axis Mahony attitude vs truth,
#                 magnetometer and barometer readouts (simulated, not used)
#   --rig-panels  RGB and rendered depth (160x120 per eye) from the stereo mount + left-eye optical flow
# Nothing new enters the control loop: the IMU draws from its own random stream on a read-only substep
# hook, the eye renders only read the state. The sidecar records whether the rendered episode reproduces
# the published table episode (outcome and completion time); a mismatch is reported, not hidden.
# EGL: not on dgx2 (EGL context creation fails on dgx2-1, 2026-09-29).
set -euo pipefail
export MUJOCO_GL=egl
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
CACHE=${TMPDIR:-/tmp}/bhl-maze-sensors-$SLURM_JOB_ID; mkdir -p "$CACHE"
SEED=${MAZE_SEED:-1}
OUT=$REPO/results/maze-explore-20260924/hard-6x6
GIF=${MAZE_SENSORS_GIF:-$REPO/docs/gifs/random-maze-explore-sensors.gif}
for f in "$GIF" "${GIF%.gif}.json" "$OUT/seed${SEED}_sensors.json" "$OUT/seed${SEED}_sensors.mp4"; do
    [ -e "$f" ] && { echo "REFUSING: $f exists (never overwrite a published clip or episode)"; exit 3; }
done
"$PY" -c "import mujoco; r = mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"1\"/></worldbody></mujoco>'), 64, 64); r.render(); r.close(); print('EGL render preflight OK', mujoco.__version__)"
echo "=== node: $(hostname) | HEAD $(git rev-parse --short HEAD) | hard 6x6 seed $SEED + IMU and rig panels | $(date) ==="
"$PY" scripts/bench/maze_explore.py --upstream "$U" --cache-dir "$CACHE" --seeds 1 --seed-start "$SEED" --n 6 --m 6 --extra-openings 1 \
    --time-limit 180 --out-dir "$OUT" --tag "_sensors" --render --frames-root "$CACHE/frames" --stride 2 \
    --imu-panel --rig-panels \
    --gif "$GIF" --gif-speed "${MAZE_GIF_SPEED:-5.0}" --gif-fps 8 --gif-width "${MAZE_GIF_WIDTH:-1100}" 2>&1 | grep -v "^\[\|libEGL"
rm -rf "$CACHE"
[ -f "${GIF%.gif}.json" ] || { echo "NO GIF written (see above)"; exit 1; }
"$PY" - "$GIF" "$OUT" "$SEED" <<'PYEOF'
import json, sys
from pathlib import Path
gif, out, seed = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
side_p = gif.with_suffix(".json")
side = json.loads(side_p.read_text())
pub = json.loads((out / f"seed{seed}.json").read_text())
new = json.loads((out / f"seed{seed}_sensors.json").read_text())
side["episode_matches_table"] = {"table": {"outcome": pub["outcome"], "completion_s": pub["completion_s"]},
                                 "rendered": {"outcome": new["outcome"], "completion_s": new["completion_s"]},
                                 "match": pub["outcome"] == new["outcome"] and pub["completion_s"] == new["completion_s"]}
side_p.write_text(json.dumps(side, indent=2) + "\n")
print("MAZE-SENSORS GIF: " + json.dumps({k: side[k] for k in ("output", "output_mb", "episode_matches_table")}))
PYEOF
ls -la "$GIF" "${GIF%.gif}.json"
echo "=== done: $(date) ==="
