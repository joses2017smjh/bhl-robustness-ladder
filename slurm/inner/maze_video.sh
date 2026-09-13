#!/bin/bash
# Isaac clips of B5 arms through a camera *sensor* that follows each robot.
#
# Not the viewport: with fabric on, the viewport draws the robot at its stale USD
# pose -- the env's grid origin -- while physics has it on a terrain patch tens
# of metres away, which is why 21247917 filmed corridors and no body. The
# render probe (21299608) showed a camera sensor draws the body where it is.
#
# Frames are PNGs under results/clips/frames/<label>/ (gitignored); the gif is
# assembled from them on the login node. Success is counted in frames written,
# never in train_play's exit code.
set -uo pipefail
cd "$UPSTREAM"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export BHL_CAMERA_CLIP=1 BHL_CLIP_EVERY=${BHL_CLIP_EVERY:-2} BHL_CLIP_SIZE=${BHL_CLIP_SIZE:-640:360}
L="$UPSTREAM/logs/rsl_rl/biped"
fails=0

record() {   # task run_glob label
    local task=$1 glob=$2 label=$3 run dir n
    # CLIP_ONLY=label re-renders one arm without redoing the others.
    if [ -n "${CLIP_ONLY:-}" ] && [ "$CLIP_ONLY" != "$label" ]; then return; fi
    run=$(ls -dt "$L"/*"$glob" 2>/dev/null | head -1)
    if [ -z "$run" ]; then echo "SKIP $label: no run matching *$glob"; fails=$((fails+1)); return; fi
    dir="$REPO/results/clips/frames/maze_${label}"
    rm -rf "$dir"; mkdir -p "$dir"
    echo "=== $label  task=$task  run=$(basename "$run") ==="
    BHL_CLIP_DIR="$dir" "$PY" "$REPO/scripts/train_play.py" \
        --task "$task" --num_envs 1 --headless --enable_cameras \
        --video_length "${VIDEO_LEN:-400}" --load_run "$(basename "$run")" || true
    n=$(find "$dir" -name 'frame_*.png' | wc -l)
    if [ "$n" -ge 50 ]; then echo "  ok -- $n frames in $dir"
    else echo "  FAILED $label -- $n frames"; fails=$((fails+1)); fi
}

record Velocity-BHL-Maze-Blind-v0      maze-blind-s0      blind
record Velocity-BHL-Maze-Lidar-v0      maze-lidar-s0      lidar
record Velocity-BHL-Maze-Stereo-v0     maze-stereo-s0     stereo_p4
record Velocity-BHL-Maze-StereoP16-v0  maze-stereop16-s0  stereo_p16

[ "$fails" -eq 0 ] && echo "maze video done" || { echo "maze video: $fails failed"; exit 1; }
