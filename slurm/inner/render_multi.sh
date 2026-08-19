#!/bin/bash
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
U=$UPSTREAM/logs/rsl_rl/biped

d1=$(ls -d "$U"/*_dr-off-s0 | head -1)
d2=$(ls -d "$U"/*_dr-default-s0 | head -1)
d3=$(ls -d "$U"/*_push-adaptive-s0 | head -1)
d4=$(ls -d "$U"/*_terrain-bumpy-s0 | head -1)

echo "=== flat race with matched shoves ==="
"$PY" scripts/render_multi.py \
    --deploy "$d2/exported/deploy.yaml" "$d1/exported/deploy.yaml" \
             "$d3/exported/deploy.yaml" "$d4/exported/deploy.yaml" \
    --labels "randomized" "no randomization" "push-trained" "terrain-trained" \
    --upstream "$UPSTREAM" --cache-dir "$CACHE_DIR" \
    --out "$REPO/docs/gifs/multi_race.mp4" \
    --gif "$REPO/docs/gifs/multi_race.gif" \
    --seconds 12 --vx 0.3 --push 0.45 --world flat

echo "=== lab floor, no push ==="
"$PY" scripts/render_multi.py \
    --deploy "$d2/exported/deploy.yaml" "$d1/exported/deploy.yaml" \
             "$d3/exported/deploy.yaml" "$d4/exported/deploy.yaml" \
    --labels "randomized" "no randomization" "push-trained" "terrain-trained" \
    --upstream "$UPSTREAM" --cache-dir "$CACHE_DIR" \
    --out "$REPO/docs/gifs/multi_lab.mp4" \
    --gif "$REPO/docs/gifs/multi_lab.gif" \
    --seconds 14 --vx 0.40 --push 0.0 --world lab
