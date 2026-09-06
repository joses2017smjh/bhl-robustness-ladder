#!/bin/bash
# The B3 ice clip: depth against blind on friction patches flush with the floor.
#
# MuJoCo, on a GPU node. Rendering it on the login node produced a 0-byte mp4 --
# EGL came up but the context could not be made current. render_carry survives
# that; render_multi does not, and the difference is not worth chasing when a
# GPU node has a real device.
set -uo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export MUJOCO_GL=egl
mkdir -p "$REPO/docs/gifs" "$REPO/results/clips"
"$PY" "$REPO/scripts/render_multi.py" \
    --deploy "$REPO/results/deploy_ice_blind.yaml" "$REPO/results/deploy_ice_depth.yaml" \
    --labels blind depth \
    --upstream "$UPSTREAM" --cache-dir "${CACHE_DIR:-/scratch/sanchej7/mjcf-ice}" \
    --out "$REPO/results/clips/ice_pair.mp4" \
    --gif "$REPO/docs/gifs/ice_pair.gif" \
    --seconds 12 --vx 0.4 --world flat --gif-width 860
rc=$?
echo "render_multi exit: $rc"
# Count the artefact, not the exit code.
for f in "$REPO/docs/gifs/ice_pair.gif" "$REPO/results/clips/ice_pair.mp4"; do
    if [ -s "$f" ]; then echo "  ok  $(du -h "$f" | cut -f1)  $f"
    else echo "  FAILED (missing or empty): $f"; fi
done
