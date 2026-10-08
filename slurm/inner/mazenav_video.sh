#!/bin/bash
# Isaac clips of the B5 *navigation* policies (mazenav-*-s0), through a camera
# sensor that follows each robot. Not the viewport (stale USD pose) and not
# the old maze-* terrain-perception checkpoints.
#
# Frames land under results/clips/frames/mazenav_<label>/ (gitignored). The
# gif is assembled on the login node from those PNGs. Success is frames
# written, never train_play's exit code. Do not overwrite maze_* frame dirs.
set -uo pipefail
cd "$UPSTREAM"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export BHL_CAMERA_CLIP=1 BHL_CLIP_EVERY=${BHL_CLIP_EVERY:-2} BHL_CLIP_SIZE=${BHL_CLIP_SIZE:-640:360}
L="$UPSTREAM/logs/rsl_rl/biped"
fails=0

record() {   # task run_glob label
    local task=$1 glob=$2 label=$3 run dir n
    if [ -n "${CLIP_ONLY:-}" ] && [ "$CLIP_ONLY" != "$label" ]; then return; fi
    run=$(ls -dt "$L"/*"$glob" 2>/dev/null | head -1)
    if [ -z "$run" ]; then echo "SKIP $label: no run matching *$glob"; fails=$((fails+1)); return; fi
    dir="$REPO/results/clips/frames/mazenav_${label}"
    rm -rf "$dir"; mkdir -p "$dir"
    echo "=== $label  task=$task  run=$(basename "$run") ==="
    BHL_CLIP_DIR="$dir" "$PY" "$REPO/scripts/train_play.py" \
        --task "$task" --num_envs 1 --headless --enable_cameras \
        --video_length "${VIDEO_LEN:-400}" --load_run "$(basename "$run")" || true
    n=$(find "$dir" -name 'frame_*.png' | wc -l)
    if [ "$n" -ge 50 ]; then echo "  ok -- $n frames in $dir"
    else echo "  FAILED $label -- $n frames"; fails=$((fails+1)); fi
}

record Velocity-BHL-Maze-Blind-v0   mazenav-blind-s0   blind
record Velocity-BHL-Maze-Lidar-v0   mazenav-lidar-s0   lidar
record Velocity-BHL-Maze-Stereo-v0  mazenav-stereo-s0  stereo
record Velocity-BHL-Maze-Both-v0    mazenav-both-s0    both

[ "$fails" -eq 0 ] && echo "mazenav video done" || { echo "mazenav video: $fails failed"; exit 1; }
