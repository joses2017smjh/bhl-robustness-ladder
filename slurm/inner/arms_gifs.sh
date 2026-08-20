#!/bin/bash
# Film the 22-DoF comparison rollouts, then composite the three pair GIFs.
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
LOGROOT=$UPSTREAM/logs/rsl_rl/humanoid

deploy_for() {
    local label=$1
    local run
    run=$(ls -d "$LOGROOT"/*_"$label" 2>/dev/null | head -1)
    [ -n "$run" ] || { echo "missing run $label"; return 1; }
    [ -f "$run/exported/deploy.yaml" ] || { echo "missing deploy $label"; return 1; }
    echo "$run/exported/deploy.yaml"
}

film() {
    local label=$1 push=$2 d=$3 sub=$4
    export LABEL=$label PUSH_SPEED=$push TERRAIN_D=$d
    export DEPLOY_CFG
    DEPLOY_CFG=$(deploy_for "$label")
    export VIDEO_DIR=$REPO/results/demo_arms/$sub
    export OUT_CSV=$VIDEO_DIR/${label}.csv
    mkdir -p "$VIDEO_DIR"
    echo "=== film $label push=$push d=$d -> $sub ==="
    "$PY" -m bhl_robust.eval.run_eval \
        --deploy-cfg "$DEPLOY_CFG" --upstream "$UPSTREAM" --cache-dir "$CACHE_DIR" \
        --out "$OUT_CSV" --label "$LABEL" --variant humanoid \
        --episode-s "${EPISODE_S:-10}" --seeds "${N_SEEDS:-1}" \
        --push-speed "$PUSH_SPEED" --terrain-difficulty "$TERRAIN_D" \
        --video-dir "$VIDEO_DIR"
}

film arms-dr1.0-s0 0 0 dr
film arms-dr0.0-s0 0 0 dr
film arms-push-s0 0.5 0 push
film arms-dr1.0-s0 0.5 0 push
film arms-terrain-s0 0 0.80 terrain
film arms-dr1.0-s0 0 0.80 terrain

"$PY" "$REPO/scripts/make_gifs.py"
ls -lh "$REPO/docs/gifs"/arms_*.gif
