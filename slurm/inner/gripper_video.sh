#!/bin/bash
# Clips of the gripper arms against the welded-hand ones, recorded in Isaac.
#
# The MuJoCo replay cannot render these: it builds its crew from the 22-DoF
# MJCF and the gripper asset is a 24-DoF URDF. Isaac 6.0 is where these
# policies trained and where RTX works, so the clips come from there and carry
# no cross-simulator caveat.
#
# The previous version died in four seconds with an empty log. It resolved the
# run directory through a chain of `..` from the checkpoint and grepped a log
# glob for the task id; under `set -euo pipefail` that is several ways to fail
# before printing anything. This names the task directly and lets train_play
# pick the newest matching run.
set -uo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
L="$UPSTREAM/logs/rsl_rl/task_v2"

record() {   # task run_glob label
    local task=$1 glob=$2 label=$3
    local run
    run=$(ls -dt "$L"/*"$glob" 2>/dev/null | head -1)
    if [ -z "$run" ]; then echo "SKIP $label: no run matching *$glob"; return; fi
    echo "=== $label  task=$task  run=$(basename "$run") ==="
    "$PY" scripts/train_play.py \
        --task "$task" --num_envs 4 --headless --enable_cameras \
        --video --video_length "${VIDEO_LEN:-400}" \
        --load_run "$(basename "$run")" \
      && echo "  ok" || echo "  FAILED $label"
}

record TaskV2-BHL-CubeToShelfGrip-Blind-v0 grip-cubetoshelfgrip-blind-s0 gripper_cube
record TaskV2-BHL-CubeToShelf-Blind-v0     v2-cubetoshelf-blind-s0       welded_cube

echo "=== videos produced ==="
find "$L" -name "*.mp4" -newermt "-3 hours" 2>/dev/null | head -10
