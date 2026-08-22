#!/bin/bash
# Blind lift against vision-in-the-loop lift, at 2, 3 and 4 robots.
#
# The comparison only means something if both sides are scored by the same
# judge, so both go through the same MuJoCo replay: same crate, same seeds, same
# crew geometry, same pinch gate. The only difference is what the policy is
# allowed to know. The blind arms read the crate's pose straight out of the
# simulator as a privileged 3-vector; `depthswap` has that taken away and gets
# two 8x8 depth images instead; `depthboth` gets both and is the control that
# separates "vision helps" from "an extra 128 inputs help".
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
L=$UPSTREAM/logs/rsl_rl/coop_lift

pick() { ls -d "$L"/*_"$1" 2>/dev/null | tail -1; }

BLIND=$(pick coop-cube-staged-r2-s0)
SWAP=$(pick coop-cube-depthswap-s0)
BOTH=$(pick coop-cube-depthboth-s0)

echo "=== runs under comparison ==="
RUNS=()
for r in "$BLIND" "$SWAP" "$BOTH"; do
    if [ -n "$r" ]; then
        n=$(ls "$r"/model_*.pt 2>/dev/null | wc -l)
        echo "  $(basename "$r")  ($n checkpoints)"
        RUNS+=("$r")
    fi
done
if [ "${#RUNS[@]}" -lt 2 ]; then
    echo "FATAL: need at least the blind baseline and one vision arm" >&2
    exit 1
fi

echo
echo "=== 1. sim2sim scores, crews of 2, 3 and 4 ==="
"$PY" scripts/bench/coop_sim2sim.py \
    --run-dir "${RUNS[@]}" \
    --upstream "$UPSTREAM" --cache-dir "$CACHE_DIR" \
    --seeds "${N_SEEDS:-8}" --crews 2 3 4 \
    --csv "$REPO/results/coop_vision_sim2sim.csv"

echo
echo "=== 2. clips for whichever vision arms trained ==="
for arm in swap both; do
    case $arm in
        swap) RD=$SWAP ;;
        both) RD=$BOTH ;;
    esac
    [ -z "$RD" ] && { echo "skip $arm: no run directory"; continue; }
    for n in 2 3 4; do
        echo "--- $arm, $n robots ---"
        "$PY" scripts/render_carry.py \
            --run-dir "$RD" --robots "$n" \
            --upstream "$UPSTREAM" --cache-dir "$CACHE_DIR" \
            --out "$REPO/docs/gifs/carry_vision_${arm}_${n}.mp4" \
            --gif "$REPO/docs/gifs/carry_vision_${arm}_${n}.gif" \
            --csv "$REPO/results/coop_vision_clips.csv" \
            --steps "${CARRY_STEPS:-300}" --gif-fps 12 --gif-width 860
    done
done

ls -la "$REPO/docs/gifs/carry_vision_"*.gif 2>/dev/null || true
