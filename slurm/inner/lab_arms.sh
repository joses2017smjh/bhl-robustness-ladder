#!/bin/bash
# The lab-floor clip, on the 22-DoF machine, with the depth band underneath.
#
# The biped version of this shot was the wrong robot for the section it sits in:
# §6 is about what a depth sensor is pointed at and §7 is about what the arms
# do, and the clip had no arms. Same four training recipes, same composed floor,
# 22 DoF.
#
# Order matters. The traverse bench runs first and prints the course profile and
# which policy finishes; the clip is then rendered with that policy as the hero.
# If the bench and the `--hero` flag disagree, the log says so rather than the
# GIF quietly promoting a robot that stalled.
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
U=$UPSTREAM/logs/rsl_rl/humanoid

d1=$(ls -d "$U"/*_arms-dr1.0-s0 | head -1)
d2=$(ls -d "$U"/*_arms-dr0.0-s0 | head -1)
d3=$(ls -d "$U"/*_arms-push-s0  | head -1)
d4=$(ls -d "$U"/*_arms-terrain-s0 | head -1)
DEPLOYS=("$d1/exported/deploy.yaml" "$d2/exported/deploy.yaml"
         "$d3/exported/deploy.yaml" "$d4/exported/deploy.yaml")
LABELS=("randomized" "no randomization" "push-trained" "terrain-trained")

echo "=== 1. does the 22-DoF machine clear the course? ==="
"$PY" scripts/bench/lab_traverse.py \
    --deploy "${DEPLOYS[@]}" --labels "${LABELS[@]}" \
    --upstream "$UPSTREAM" --cache-dir "$CACHE_DIR" \
    --variant humanoid --seconds "${TRAVERSE_S:-24}" --vx "${VX:-0.40}"

echo
echo "=== 2. the clip: 22 DoF, lab floor, depth band, terrain-trained in livery ==="
"$PY" scripts/render_multi.py \
    --deploy "${DEPLOYS[@]}" --labels "${LABELS[@]}" \
    --upstream "$UPSTREAM" --cache-dir "$CACHE_DIR" \
    --variant humanoid --world lab \
    --depth-of "terrain-trained" --hero "terrain-trained" --depth-res 64 \
    --seconds "${EPISODE_S:-16}" --vx "${VX:-0.40}" --push 0.0 \
    --out "$REPO/docs/gifs/multi_lab.mp4" \
    --gif "$REPO/docs/gifs/multi_lab.gif" \
    --gif-seconds "${EPISODE_S:-16}" --gif-fps 8 --gif-width 860

echo
echo "=== 3. the same course on 12 DoF, for the comparison the section needs ==="
B=$UPSTREAM/logs/rsl_rl/biped
b1=$(ls -d "$B"/*_dr-default-s0 | head -1)
b2=$(ls -d "$B"/*_dr-off-s0 | head -1)
b3=$(ls -d "$B"/*_push-adaptive-s0 | head -1)
b4=$(ls -d "$B"/*_terrain-bumpy-s0 | head -1)
"$PY" scripts/bench/lab_traverse.py \
    --deploy "$b1/exported/deploy.yaml" "$b2/exported/deploy.yaml" \
             "$b3/exported/deploy.yaml" "$b4/exported/deploy.yaml" \
    --labels "${LABELS[@]}" \
    --upstream "$UPSTREAM" --cache-dir "$CACHE_DIR" \
    --variant biped --seconds "${TRAVERSE_S:-24}" --vx "${VX:-0.40}" \
    | sed -n '/=== traversal/,$p'
