#!/bin/bash
# Every POV clip in the README, rendered in one pass under one depth mapping.
#
# They have to be one job. The false-colour ramp lives in the replay module, so
# a clip rendered before an edit and a clip rendered after it disagree about
# what a given brightness means -- and the whole claim of the strip is that
# brightness means the same distance in every frame of every clip. Two earlier
# rounds were thrown away for exactly that.
#
# carry_2/3/4 are not here: they carry no POV column, so the ramp does not
# reach them, and re-rendering them would cost twenty minutes for no change.
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
L=$UPSTREAM/logs/rsl_rl/coop_lift
pick() { ls -d "$L"/*_"$1" 2>/dev/null | tail -1; }

CUBE=${CLIP_RUN:-$(pick coop-cube-staged-r2-s0)}
BALL=$(pick coop-ball-staged-s0)
LADDER=$(pick coop-ladder-staged-s0)
SWAP=$(pick coop-cube-depthswap-s0)
BOTH=$(pick coop-cube-depthboth-s0)
for v in CUBE BALL LADDER SWAP BOTH; do
    [ -n "${!v}" ] || { echo "pov_gifs: no run for $v under $L" >&2; exit 1; }
done

best() {  # run robots payload criterion -> seed
    "$PY" scripts/bench/pick_seed.py --run-dir "$1" --robots "$2" --payload "$3" \
        --criterion "$4" --upstream "$UPSTREAM" --cache-dir "$CACHE_DIR" \
        --seeds "${PICK_SEEDS:-12}" --steps "${CARRY_STEPS:-300}" 2>&1 \
        | tee /dev/stderr | sed -n 's/^BEST_SEED=\([0-9]*\).*/\1/p'
}

clip() {  # name run payload robots seed
    local name=$1 run=$2 pay=$3 n=$4 seed=$5
    echo "--- $name (seed $seed) ---"
    "$PY" scripts/render_carry.py \
        --run-dir "$run" --robots "$n" --payload "$pay" --seed "$seed" --pov \
        --upstream "$UPSTREAM" --cache-dir "$CACHE_DIR" \
        --out "$REPO/docs/gifs/${name}.mp4" \
        --gif "$REPO/docs/gifs/${name}.gif" \
        --csv "$REPO/results/coop_clips_pov.csv" \
        --steps "${CARRY_STEPS:-300}" --gif-fps 12 --gif-width 1000
}

echo "=== the sighted arms, showing what they read ==="
s=$(best "$SWAP" 2 cube lift); clip carry_vision_swap_2 "$SWAP" cube 2 "${s:-0}"
s=$(best "$SWAP" 3 cube lift); clip carry_vision_swap_3 "$SWAP" cube 3 "${s:-0}"
s=$(best "$BOTH" 2 cube lift); clip carry_vision_both_2 "$BOTH" cube 2 "${s:-0}"
s=$(best "$BOTH" 4 cube lift); clip carry_vision_both_4 "$BOTH" cube 4 "${s:-0}"

echo "=== the three objects ==="
clip carry_ladder_pov        "$LADDER" ladder 2 0
s=$(best "$CUBE" 2 cube hold);  clip carry_cube_pov          "$CUBE" cube 2 "${s:-11}"
s=$(best "$CUBE" 2 ball lift);  clip carry_ball_transfer_pov "$CUBE" ball 2 "${s:-4}"
clip carry_ball_native_pov   "$BALL" ball 2 0

ls -la "$REPO/docs/gifs/carry_"*pov*.gif "$REPO/docs/gifs/carry_vision"*.gif
