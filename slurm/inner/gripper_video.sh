#!/bin/bash
# Record the gripper arms in Isaac, where RTX works.
#
# The MuJoCo replay harness cannot render these: it builds its crew from the
# 22-DoF MJCF, and the gripper asset is a 24-DoF URDF. So the clips come from
# Isaac Sim 6.0, which is also the stack these policies trained on -- no
# cross-simulator gap to caveat, and the renderer that segfaults on 5.1 works
# here.
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
L=$UPSTREAM/logs/rsl_rl/task_v2
pick() { ls -dt "$L"/*_"$1" 2>/dev/null | head -1; }

for pair in "grip-cubetoshelfgrip-blind-s0:gripper_cube" \
            "v2-cubetoshelf-blind-s0:welded_cube"; do
    run=$(pick "${pair%%:*}"); name=${pair##*:}
    [ -n "$run" ] || { echo "no run for ${pair%%:*}"; continue; }
    task=$(grep -oE 'TaskV2-BHL-[A-Za-z]+-[A-Za-z]+-v0' "$run"/../../../../../../logs/*.out 2>/dev/null | head -1)
    case "$name" in
      gripper_cube) task=TaskV2-BHL-CubeToShelfGrip-Blind-v0 ;;
      welded_cube)  task=TaskV2-BHL-CubeToShelf-Blind-v0 ;;
    esac
    echo "=== $name  <- $(basename "$run")  task=$task ==="
    "$PY" scripts/train_play.py --task "$task" --num_envs 4 \
        --video --video_length "${VIDEO_LEN:-400}" --headless --enable_cameras \
        --load_run "$(basename "$run")" || echo "  play failed for $name"
done
find "$L" -name "*.mp4" -newermt "-2 hours" 2>/dev/null | head
