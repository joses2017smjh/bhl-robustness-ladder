#!/bin/bash
# Isaac clips of the maze arms: blind (the control), lidar, the wide stereo that
# failed at pool=4, and the pooled both-sensors arm that beat everything.
#
# The MuJoCo replay cannot drive these. It renders one forward depth camera and
# refuses lidar rings and stereo pairs by name rather than reshaping them into a
# depth image, so the clips come from Isaac 6.0 -- where the policies trained --
# and carry no cross-simulator caveat.
#
# Same guard as gripper_video.sh: count mp4s newer than a marker, never trust
# train_play's exit code (Hydra + simulation_app.close() return 0 over a
# traceback). Each clip is copied out of its run directory as soon as it lands,
# because RecordVideo overwrites rl-video-step-0.mp4 on every render.
set -uo pipefail
cd "$UPSTREAM"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
L="$UPSTREAM/logs/rsl_rl/biped"
OUT="$REPO/results/clips"
mkdir -p "$OUT"
fails=0

record() {   # task run_glob label
    local task=$1 glob=$2 label=$3 run marker clip
    run=$(ls -dt "$L"/*"$glob" 2>/dev/null | head -1)
    if [ -z "$run" ]; then echo "SKIP $label: no run matching *$glob"; fails=$((fails+1)); return; fi
    echo "=== $label  task=$task  run=$(basename "$run") ==="
    marker="$L/.render-marker-$$"
    : > "$marker"
    "$PY" "$REPO/scripts/train_play.py" \
        --task "$task" --num_envs 4 --headless --enable_cameras \
        --video --video_length "${VIDEO_LEN:-400}" \
        --load_run "$(basename "$run")" || true
    clip=$(find "$run/videos" -name "*.mp4" -newer "$marker" 2>/dev/null | head -1)
    rm -f "$marker"
    if [ -n "$clip" ]; then
        cp -f "$clip" "$OUT/maze_${label}.mp4"
        echo "  ok -- $OUT/maze_${label}.mp4 ($(du -h "$clip" | cut -f1))"
    else
        echo "  FAILED $label -- no video was written"
        fails=$((fails+1))
    fi
}

record Velocity-BHL-Maze-Blind-v0    maze-blind-s0    blind
record Velocity-BHL-Maze-Lidar-v0    maze-lidar-s0    lidar
record Velocity-BHL-Maze-Stereo-v0   maze-stereo-s0   stereo_p4
record Velocity-BHL-Maze-BothP16-v0  maze-bothp16-s0  both_p16

echo "=== clips ==="; ls -la "$OUT"/maze_*.mp4 2>/dev/null
[ "$fails" -eq 0 ] && echo "maze video done" || { echo "maze video: $fails failed"; exit 1; }
