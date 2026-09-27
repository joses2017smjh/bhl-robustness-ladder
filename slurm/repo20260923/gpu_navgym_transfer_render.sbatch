#!/bin/bash
#SBATCH --job-name=navgym-transfer-gif
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# 2026-09-27: clip of ONE physics episode of a LEARNED NavGym v2 actor (EXPLORATORY transfer), EGL on a
# GPU node, modelled on gpu_maze_explore_render.sbatch. Submit after cpu_navgym_transfer.sbatch
# (--dependency=afterany:<transfer job>; a missing or ineligible table renders nothing). The actor and seed come ONLY from the rule declared in that
# launcher, applied in results/navgym-transfer-20260927/transfer_table.json "render":
#   the actor with the most hard-6x6 successes, rendered ONLY if it reached >= 8/12, on its
#   median-time successful seed. Otherwise this job renders nothing and says so.
# Frame footer (maze_explore.py --policy): "commands: LEARNED NavGym policy (PPO in the gym, deployed on
# the physics robot) | goal: oracle", with "pose: oracle" and the frozen learned gait. The sidecar
# (docs/gifs/navgym-learned-maze.json) adds: EXPLORATORY, the maze seed is inside the gym's training-seed
# range, and whether the rendered episode reproduces the table episode (outcome and completion time).
set -euo pipefail
export MUJOCO_GL=egl
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
OUT=${NAVGYM_TRANSFER_OUT:-$REPO/results/navgym-transfer-20260927}
GIF=$REPO/docs/gifs/navgym-learned-maze.gif
TABLE=$OUT/transfer_table.json
[ -f "$TABLE" ] || { echo "NO RENDER: $TABLE missing (run cpu_navgym_transfer.sbatch first)"; exit 0; }
if [ -e "$GIF" ]; then echo "REFUSING: $GIF exists (never overwrite a published clip)"; exit 3; fi
read -r ELIG ACTOR ONNX SEED <<<"$("$PY" -c "import json,sys; r=json.load(open(sys.argv[1]))['render']; print(r.get('eligible'), r.get('actor'), r.get('onnx'), r.get('seed'))" "$TABLE")"
if [ "$ELIG" != "True" ]; then
  echo "NO RENDER by the declared rule: $("$PY" -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))['render']))" "$TABLE")"
  exit 0
fi
CACHE=${TMPDIR:-/tmp}/bhl-navgym-gif-$SLURM_JOB_ID; mkdir -p "$CACHE"
"$PY" -c "import mujoco; r = mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"1\"/></worldbody></mujoco>'), 64, 64); r.render(); r.close(); print('EGL render preflight OK', mujoco.__version__)"
echo "=== node: $(hostname) | HEAD $(git rev-parse --short HEAD) | EXPLORATORY learned-actor clip | $ACTOR seed $SEED | $(date) ==="
"$PY" scripts/bench/maze_explore.py --upstream "$U" --cache-dir "$CACHE" --seeds 1 --seed-start "$SEED" --n 6 --m 6 --extra-openings 1 \
    --time-limit 180 --policy "$REPO/$ONNX" --out-dir "$OUT/$ACTOR" --tag "_render" --render --frames-root "$CACHE/frames" --stride 2 \
    --gif "$GIF" --gif-speed "${NAVGYM_GIF_SPEED:-4.0}" --gif-fps 8 --gif-width 860 2>&1 | grep -v "^\["
rm -rf "$CACHE"
[ -f "${GIF%.gif}.json" ] || { echo "NO GIF written (see above)"; exit 0; }
"$PY" - "$GIF" "$OUT" "$ACTOR" "$SEED" "$REPO" <<'PYEOF'
import json, os, sys
from pathlib import Path
gif, out, actor, seed, repo = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), Path(sys.argv[5])
side_p = gif.with_suffix(".json")
side = json.loads(side_p.read_text())
table = json.loads((out / actor / f"seed{seed}.json").read_text())
rend = json.loads((out / actor / f"seed{seed}_render.json").read_text())
side["status"] = ("EXPLORATORY: the NavGym v2 arm did not pass its predeclared gym bar, so this transfer was not due; "
                  "the multi-seed table is results/navgym-transfer-20260927/transfer_table.json")
side["labels"] = {"learned": "biped gait (frozen PPO) and the (vx, wz) navigation commands (NavGym v2 PPO actor, trained in the 2-D gym)",
                  "scripted": "nothing in the loop", "oracle": "pose and goal coordinate"}
side["maze_seed_in_gym_training_range"] = seed < 10_000
side["transfer_table"] = os.path.relpath(out / "transfer_table.json", repo)
side["render_rule"] = json.loads((out / "transfer_table.json").read_text())["render"]
side["episode_matches_table"] = {"table": {"outcome": table["outcome"], "completion_s": table["completion_s"]},
                                 "rendered": {"outcome": rend["outcome"], "completion_s": rend["completion_s"]},
                                 "match": table["outcome"] == rend["outcome"] and table["completion_s"] == rend["completion_s"]}
side_p.write_text(json.dumps(side, indent=2) + "\n")
print("NAVGYM TRANSFER GIF sidecar: " + json.dumps({k: side[k] for k in ("output", "output_mb", "maze_seed_in_gym_training_range", "episode_matches_table")}))
PYEOF
ls -la "$GIF" "${GIF%.gif}.json" "$OUT/$ACTOR"/seed${SEED}_render.* 2>/dev/null
echo "=== done: $(date) ==="
