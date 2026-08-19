#!/bin/bash
set -euo pipefail
cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}"
A=$UPSTREAM/source/berkeley_humanoid_lite_assets/data/robots/berkeley_humanoid/berkeley_humanoid_lite

echo "===== AUDIT biped ====="
"$PY" -m bhl_robust.audit.asset_consistency \
    --assets "$A" --variant biped \
    --upstream "$UPSTREAM" --cache-dir "$CACHE_DIR" \
    --json-out "$REPO/results/audit/biped.json"

echo "===== AUDIT humanoid ====="
"$PY" -m bhl_robust.audit.asset_consistency \
    --assets "$A" --variant humanoid \
    --upstream "$UPSTREAM" --cache-dir "$CACHE_DIR" \
    --json-out "$REPO/results/audit/humanoid.json"

echo "===== USD eval terrain ====="
"$PY" -m bhl_robust.usd.eval_terrain \
    --out "$REPO/results/usd/eval_terrain.usdc" --seed 12345

echo "===== USD lab scene ====="
"$PY" -m bhl_robust.usd.lab_scene \
    --out "$REPO/results/usd/lab_scene.usda"
