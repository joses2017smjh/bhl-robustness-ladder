#!/bin/bash
# The lift that survives the cross-check, and the view the robot had.
#
# Two debts close here. The first is that the README's carry clips all showed
# the cube, whose payload never leaves the floor, while the arm that reported a
# 21 cm lift had no clip at all. That arm turns out not to transfer -- it falls
# in 0.72 s in MuJoCo, in either scene -- so the clip it gets is the honest one:
# the fall, beside the cube that stands.
#
# The lift worth showing is the third row: the cube-trained policy dropped into
# the ball scene, which it never saw in training. It reaches 9.1 cm, half again
# what it manages on the cube, and it does that in the second engine. That is
# the geometry argument standing on its own.
#
# The second debt is that the README claims two camera modalities and shows
# neither from the robot's head. `--pov` appends a column: colour on top, the
# policy's own depth image below. One camera, one pose, one instant.
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
L=$UPSTREAM/logs/rsl_rl/coop_lift
pick() { ls -d "$L"/*_"$1" 2>/dev/null | tail -1; }

BALL=$(pick coop-ball-staged-s0)
CUBE=${CLIP_RUN:-$(pick coop-cube-staged-r2-s0)}
[ -n "$BALL" ] && [ -n "$CUBE" ] || { echo "pov_gifs: missing a run under $L" >&2; exit 1; }

clip() {  # name run payload seed extra...
    local name=$1 run=$2 pay=$3 seed=$4; shift 4
    echo "--- $name ---"
    "$PY" scripts/render_carry.py \
        --run-dir "$run" --robots 2 --payload "$pay" --seed "$seed" \
        --upstream "$UPSTREAM" --cache-dir "$CACHE_DIR" \
        --out "$REPO/docs/gifs/${name}.mp4" \
        --gif "$REPO/docs/gifs/${name}.gif" \
        --csv "$REPO/results/coop_clips_pov.csv" \
        --steps "${CARRY_STEPS:-300}" --gif-fps 12 --gif-width 1000 "$@"
}

# The headline: a policy trained on a box, lifting a ball it has never seen.
clip carry_ball_transfer_pov "$CUBE" ball 4 --pov
# The control it has to be read against.
clip carry_cube_pov          "$CUBE" cube 11 --pov
# The 21 cm arm, failing its cross-check.
clip carry_ball_native_pov   "$BALL" ball 0 --pov

ls -la "$REPO/docs/gifs/carry_ball_transfer_pov.gif" \
       "$REPO/docs/gifs/carry_cube_pov.gif" \
       "$REPO/docs/gifs/carry_ball_native_pov.gif"
