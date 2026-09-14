#!/bin/bash
# The corrected-stereo maze clip: mazefix-stereo-s0 through the camera-sensor
# recorder, with the policy's own left eye (corrected pose) and a second eye
# carrying the raw tuple every B5 run had before 3f7b679 dumped beside each frame.
# Success is frames on disk, never the exit code.
set -uo pipefail
cd "$UPSTREAM"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export BHL_CAMERA_CLIP=1 BHL_CLIP_EVERY=${BHL_CLIP_EVERY:-2} BHL_CLIP_SIZE=${BHL_CLIP_SIZE:-640:360}
L="$UPSTREAM/logs/rsl_rl/biped"
run=$(ls -dt "$L"/*_mazefix-stereo-s0 2>/dev/null | head -1)
[ -n "$run" ] || { echo "no mazefix-stereo-s0 run"; exit 1; }
dir="$REPO/results/clips/frames/maze_stereo_fixed"
rm -rf "$dir"; mkdir -p "$dir"
echo "=== run $(basename "$run") ==="
BHL_CLIP_DIR="$dir" "$PY" "$REPO/scripts/train_play.py" \
    --task Velocity-BHL-Maze-Stereo-v0 --num_envs 1 --headless --enable_cameras \
    --video_length "${VIDEO_LEN:-400}" --load_run "$(basename "$run")" \
    --clip-sensors stereo_l --clip-raw-stereo || true
n=$(find "$dir" -name 'frame_*.png' | wc -l)
s=$(find "$dir" -name 'stereo_raw_*.png' | wc -l)
echo "frames $n, raw-eye panels $s"
[ "$n" -ge 50 ] && [ "$s" -ge 50 ]
