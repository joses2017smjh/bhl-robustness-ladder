#!/bin/bash
# The cooperative-lift clips §5 never had, plus the MuJoCo scores behind them.
#
# The bench runs before the clips on purpose. A clip is an existence proof and
# nothing more: it can show that a pinch forms, it cannot show that it forms
# reliably. The table is what licenses the sentence in the README; the GIF is
# what makes it readable.
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
L=$UPSTREAM/logs/rsl_rl/coop_lift

# `tail -1`, not `head -1`. Two directories match `*_coop-cube-pinch-s0`: a
# smoke run that stopped at model_400 with the older 150-wide observation, and
# the real 4,000-iteration one. The lexically later timestamp is the real one,
# and the bench skips a width it cannot assemble anyway.
pick() { ls -d "$L"/*_"$1" 2>/dev/null | tail -1; }

STAGED=$(pick coop-cube-staged-r2-s0)
PICK0=$(pick coop-cube-pickfirst-s0)
PICK1=$(pick coop-cube-pickfirst-s1)
PICK2=$(pick coop-cube-pickfirst-s2)
CTRL0=$(pick coop-cube-pinch-s0)
CTRL1=$(pick coop-cube-control-s1)
CTRL2=$(pick coop-cube-control-s2)
NOTILT=$(pick coop-cube-notilt-s0)
NODRIFT=$(pick coop-cube-nodrift-s0)
UNGATED=$(pick coop-cube-ungated-s0)

echo "=== 1. MuJoCo scores: the arms §5 ranked from PhysX scalars ==="
RUNS=()
for r in "$STAGED" "$PICK0" "$PICK1" "$PICK2" "$CTRL0" "$CTRL1" "$CTRL2" \
         "$NOTILT" "$NODRIFT" "$UNGATED"; do
    [ -n "$r" ] && RUNS+=("$r")
done
"$PY" scripts/bench/coop_sim2sim.py \
    --run-dir "${RUNS[@]}" \
    --upstream "$UPSTREAM" --cache-dir "$CACHE_DIR" \
    --seeds "${N_SEEDS:-8}" --crews 2 3 4 \
    --csv "$REPO/results/coop_sim2sim.csv"

# Clips come off the best-formed pinch, which is the staged rerun. Naming it
# here rather than globbing means the README caption and the file agree.
CLIP_RUN=${CLIP_RUN:-$STAGED}
echo
echo "=== 2. clips: 2, 3 and 4 robots, run=$(basename "$CLIP_RUN") ==="
for n in 2 3 4; do
    echo "--- $n robots ---"
    "$PY" scripts/render_carry.py \
        --run-dir "$CLIP_RUN" --robots "$n" \
        --upstream "$UPSTREAM" --cache-dir "$CACHE_DIR" \
        --out "$REPO/docs/gifs/carry_${n}.mp4" \
        --gif "$REPO/docs/gifs/carry_${n}.gif" \
        --csv "$REPO/results/coop_clips.csv" \
        --steps "${CARRY_STEPS:-300}" --gif-fps 12 --gif-width 860
done

ls -la "$REPO/docs/gifs/carry_"*.gif
