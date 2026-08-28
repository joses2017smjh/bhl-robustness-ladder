#!/bin/bash
# Re-sweep and re-render the ladder alone, after the slot-order fix.
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
L=$UPSTREAM/logs/rsl_rl/coop_lift
LADDER=$(ls -d "$L"/*_coop-ladder-staged-s0 | tail -1)
"$PY" scripts/bench/pick_seed.py --run-dir "$LADDER" --robots 2 --payload ladder \
    --criterion lift --upstream "$UPSTREAM" --cache-dir "$CACHE_DIR" \
    --seeds 12 --steps 300
"$PY" scripts/render_carry.py --run-dir "$LADDER" --robots 2 --payload ladder \
    --seed 0 --pov --upstream "$UPSTREAM" --cache-dir "$CACHE_DIR" \
    --out "$REPO/docs/gifs/carry_ladder_pov.mp4" \
    --gif "$REPO/docs/gifs/carry_ladder_pov.gif" \
    --csv "$REPO/results/coop_clips_pov.csv" --steps 300 --gif-fps 12 --gif-width 1000
