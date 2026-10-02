#!/bin/bash
#SBATCH --job-name=turngait-clip
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere,dgx2,share,eecs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# 2026-10-02: README clip of the turning ablation (display only, not a gate). The turn_test v2 turn command on
# reset seed 0, given to two LEARNED gaits from job 21506543:
#   left   arms-turngait-clock-s2   R1, gait clock in the actor: the 1 of 3 R1 seeds that qualifies (all 3 turn)
#   right  arms-turngait-critic-s0  R2, gait clock in the critic only: seed 0, the first of 3 (no R2 seed turns)
# Reset seed 0 is the first scored v2 seed, chosen by that rule, not by eye. scripts/bench/turn_clip.py repeats
# turn_test.run_command step for step (tests/test_turn_clip.py checks it) and writes, per run, the yaw it reached
# next to the scored JSON's; a mismatch is reported in the sidecar, not hidden. Refuses to overwrite the GIF, its
# sidecar or the MP4.
# EGL on dgx2: device 0 (the job's own GPU) is pinned there, as in gpu_maze_sensors_render.sbatch.
set -euo pipefail
export MUJOCO_GL=egl
case "$(hostname)" in dgx2-*) export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0} ;; esac
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
H=$U/logs/rsl_rl/humanoid
RES=$REPO/results/repo-gpu-20260923/turngait-r12-20261001
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
CACHE=${TMPDIR:-/tmp}/bhl-turnclip-$SLURM_JOB_ID; mkdir -p "$CACHE"
GIF=${TURN_CLIP_GIF:-$REPO/docs/gifs/turngait-clock-actor-vs-critic.gif}
MP4=${TURN_CLIP_MP4:-$REPO/results/turngait-clip-20261002/turngait-clock-actor-vs-critic.mp4}
L=$(ls -d "$H"/*_arms-turngait-clock-s2 | tail -1)/exported/deploy.yaml
R=$(ls -d "$H"/*_arms-turngait-critic-s0 | tail -1)/exported/deploy.yaml
for f in "$L" "$R" "$RES/turn-test-v2/arms-turngait-clock-s2.json" "$RES/turn-test-v2/arms-turngait-critic-s0.json"; do
    [ -f "$f" ] || { echo "MISSING $f"; exit 2; }
done
for f in "$GIF" "${GIF%.gif}.json" "$MP4"; do
    [ -e "$f" ] && { echo "REFUSING: $f exists (never overwrite a published clip)"; exit 3; }
done
"$PY" -c "import mujoco; r = mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"1\"/></worldbody></mujoco>'), 64, 64); r.render(); r.close(); print('EGL render preflight OK', mujoco.__version__)"
echo "=== node: $(hostname) | HEAD $(git rev-parse --short HEAD) | $(date) ==="
"$PY" scripts/bench/turn_clip.py --upstream "$U" --cache-dir "$CACHE" --variant humanoid --seed 0 \
    --left-deploy "$L" --left-scored "$RES/turn-test-v2/arms-turngait-clock-s2.json" \
    --left-title "Gait clock in the actor (R1)" \
    --left-subtitle "arms-turngait-clock-s2: the 1 of 3 R1 seeds that qualifies" \
    --right-deploy "$R" --right-scored "$RES/turn-test-v2/arms-turngait-critic-s0.json" \
    --right-title "Gait clock in the critic only (R2)" \
    --right-subtitle "arms-turngait-critic-s0: seed 0 of 3; no R2 seed turns" \
    --footer-line "LEARNED gaits (PPO, frozen), MuJoCo physics, real time. Scored: every R1 seed turns (18/18 runs)," \
    --footer-line "1 of 3 qualifies (the others drift on straight walks), so the R1 recipe fails. R2: 0 of 3 seeds turn." \
    --mp4 "$MP4" --gif "$GIF" --gif-fps 10 --gif-width 880 2>&1 | grep -v "^\[\|libEGL"
rm -rf "$CACHE"
[ -f "${GIF%.gif}.json" ] || { echo "NO sidecar written (see above)"; exit 1; }
echo "TURN-CLIP COMPLETE"
