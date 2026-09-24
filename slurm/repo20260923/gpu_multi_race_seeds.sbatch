#!/bin/bash
#SBATCH --job-name=multi-race-seeds
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=haswell&el8
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Repo-wide campaign (2026-09-23), LOC-07: the flat race with a denominator.
#
# docs/gifs/multi_race.gif (slurm/34_multi_gif.sbatch -> slurm/inner/render_multi.sh,
# Slurm 20951392) is ONE 12 s rollout of four policies in one MuJoCo world with
# matched 0.45 m/s shoves every 3.5 s, seed 0: one fall (no-randomization at
# 4.8 s). No results file records the outcome and n=1 per policy. This job runs
# the same four exported checkpoints, same labels and flags, for seeds 0-9 and
# writes results/multi-race-20260923/{seed_<k>.json, multi_race_10seed.json,
# multi_race_seeds.csv} (the CSV is the per-robot fall/peak-x table LOC-07
# asked for; 10 seeds rather than its 5).
#
# Why this is a CPU job despite the file name: render_multi.py now takes
# --no-video, which builds no mujoco.Renderer, MP4 or GIF, and all four
# policies are 45-observation (blind) so MultiRunner never builds a depth
# renderer either. Nothing in the rollout touches OpenGL, so a GPU/EGL node
# would be reserved and not used. It is a plain-venv MuJoCo + onnxruntime
# rollout, exactly like cpu_eval.sbatch (SF-02, SF-03b). Login-node smoke:
# a 5 s race takes ~13 s with OMP_NUM_THREADS=2, so 10 x 12 s is ~5 min; 30 min
# is a wide margin. If a GPU node is wanted anyway (e.g. to add --gif later),
# switch the header to gpu_render_eval.sbatch's and export MUJOCO_GL=egl.
#
# Same policy set and flags as the GIF run (render_multi.sh): dr-default,
# dr-off, push-adaptive-s0, terrain-bumpy-s0; --seconds 12 --vx 0.3
# --push 0.45 --world flat (push-every default 3.5 s). Seed changes the
# initial joint noise and the shove directions; within a seed the shoves are
# matched across the four robots.
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
B=$U/logs/rsl_rl/biped
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/scripts" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
unset MUJOCO_GL PYOPENGL_PLATFORM      # no rendering: if anything asks for GL this must fail loudly
CACHE=${TMPDIR:-/tmp}/bhl-multi-race-${SLURM_JOB_ID:-local}; mkdir -p "$CACHE"
OUT=$REPO/results/multi-race-20260923; mkdir -p "$OUT"
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}

echo "=== multi-race seeds job ${SLURM_JOB_ID:-local} on $(hostname) $(date) ==="
echo "git: $(git rev-parse --short HEAD) $(git diff --quiet -- scripts/render_multi.py && echo clean || echo 'render_multi.py dirty')"
"$PY" -c "import mujoco, onnxruntime; print('preflight OK mujoco', mujoco.__version__)"

# Same resolution as slurm/inner/render_multi.sh (first match per name).
d1=$(ls -d "$B"/*_dr-off-s0 | head -1)
d2=$(ls -d "$B"/*_dr-default-s0 | head -1)
d3=$(ls -d "$B"/*_push-adaptive-s0 | head -1)
d4=$(ls -d "$B"/*_terrain-bumpy-s0 | head -1)
for d in "$d1" "$d2" "$d3" "$d4"; do
  [ -f "$d/exported/deploy.yaml" ] || { echo "missing export: $d/exported/deploy.yaml"; exit 2; }
  echo "policy: $(basename "$d")  $(sha256sum "$d/exported/deploy.yaml" | cut -c1-12)"
done

fail=0
for s in $SEEDS; do
  echo "=== seed $s ==="
  rc=0
  "$PY" scripts/render_multi.py \
      --deploy "$d2/exported/deploy.yaml" "$d1/exported/deploy.yaml" \
               "$d3/exported/deploy.yaml" "$d4/exported/deploy.yaml" \
      --labels "randomized" "no randomization" "push-trained" "terrain-trained" \
      --upstream "$U" --cache-dir "$CACHE" \
      --seconds 12 --vx 0.3 --push 0.45 --world flat \
      --seed "$s" --no-video --json "$OUT/seed_$s.json" || rc=$?
  if [ "$rc" -ne 0 ]; then echo "seed $s FAILED (rc $rc)"; fail=1; fi
done

echo "=== summary ==="
# Exit status of the parser is 0 only when every expected seed is present, so
# a partial run reads INCOMPLETE in both the RACE RESULT line and the JSON.
"$PY" scripts/bench/multi_race_summary.py --dir "$OUT" --out "$OUT/multi_race_10seed.json" \
    --csv "$OUT/multi_race_seeds.csv" --expect-seeds 10 --job "${SLURM_JOB_ID:-local}" || fail=1
echo "=== done $(date) ==="
if [ "$fail" -ne 0 ]; then echo "MULTI_RACE_INCOMPLETE"; exit 1; fi
echo "MULTI_RACE_COMPLETE"
