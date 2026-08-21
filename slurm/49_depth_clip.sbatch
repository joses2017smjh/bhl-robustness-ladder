#!/bin/bash
#SBATCH --job-name=depth-clip
#SBATCH --account=eecs
#SBATCH --partition=gpu,eecs,dgxh,ampere,preempt
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# The depth evidence for the README, from both paths.
#
#   1. Validate the Isaac Lab ray-cast camera against the closed-form depth
#      image of a flat plane, and confirm terrain moves it. Warp raycasting
#      returns all-NaN rather than an error when `mesh_prim_paths` is wrong, so
#      this runs before anything trains on the sensor.
#   2. Benchmark it at 2,048 and 4,096 envs, with and without, so the depth tax
#      is isolated from the physics cost.
#   3. Render the paired third-person / egocentric-depth clip in MuJoCo, whose
#      offscreen renderer has never had the RTX problem.
#
# The clip and the sensor come from different simulators on purpose. A depth map
# that agrees across both is a depth map; one that only exists in Isaac Lab is an
# artefact of one implementation.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=8
export CACHE_DIR=$WORKSPACE/mjcf_cache
export BENCH_OUT=$REPO/results/depth_bench.txt
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS TERRAIN_D OUT_MP4 DEPTH_RES PANEL"
mkdir -p "$REPO/results" "$REPO/docs/gifs"

echo "=== 1+2: validate and benchmark the ray-cast depth ==="
bhl_exec "$REPO/slurm/inner/depth_bench.sh" 2>&1 | grep -E "CHECK|PASS|FAIL|RESULT|DEPTH" || true

echo "=== 3: paired RGB|depth clip in MuJoCo ==="
LOGROOT=$UPSTREAM/logs/rsl_rl/biped
run=$(ls -d "$LOGROOT"/*_terrain-bumpy-s0 2>/dev/null | head -1)
[ -n "$run" ] || { echo "no terrain-bumpy-s0 run"; exit 1; }
[ -f "$run/exported/deploy.yaml" ] || {
    export RUN_DIR=$(basename "$run"); bhl_exec "$REPO/slurm/inner/export_one.sh" | tail -2; }

export DEPLOY_CFG=$run/exported/deploy.yaml
export LABEL=terrain-bumpy-s0 TERRAIN_D=1.00 EPISODE_S=10 DEPTH_RES=64 PANEL=448
export OUT_MP4=$REPO/docs/gifs/depth_pair.mp4
bhl_exec "$REPO/slurm/inner/depth_gif.sh"

echo "=== 4: MP4 -> GIF ==="
ffmpeg -y -loglevel error -i "$OUT_MP4" \
  -t 9 -vf "fps=8,scale=820:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5" \
  "$REPO/docs/gifs/depth_pair.gif"
ls -la "$REPO/docs/gifs/depth_pair."* 2>/dev/null
echo "=== done: $(date) ==="
