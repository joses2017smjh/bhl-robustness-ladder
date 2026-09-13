#!/bin/bash
# One process per (task, fabric mode): a SimulationContext cannot be rebuilt
# with a different use_fabric inside the same interpreter.
set -uo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
OUT="$REPO/results/render_probe"
mkdir -p "$OUT"
fails=0
for task in Velocity-BHL-Maze-Blind-v0 ClothSort-BHL-Rigid-Oracle-v0 TaskV2-BHL-CubeToShelf-Blind-v0; do
    for mode in fabric usd; do
        flag=""; [ "$mode" = usd ] && flag="--disable_fabric"
        log="$OUT/${task}_${mode}.log"
        echo "=== $task  $mode ==="
        "$PY" "$REPO/scripts/bench/render_probe.py" --task "$task" --out-dir "$OUT" $flag > "$log" 2>&1
        if grep -q "render probe done" "$log"; then
            grep -E '^\{"step"' "$log" | cut -c1-400
        else
            echo "  FAILED -- tail:"; tail -5 "$log"; fails=$((fails+1))
        fi
    done
done
ls -la "$OUT"/*.png 2>/dev/null | wc -l | sed 's/^/pngs written: /'
[ "$fails" -eq 0 ] || echo "render probe: $fails of 6 runs failed"
