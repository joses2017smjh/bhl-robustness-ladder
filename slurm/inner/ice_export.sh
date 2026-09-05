#!/bin/bash
# Export deploy configs (and Isaac clips) for the ice arms.
#
# The B3 finding -- depth beats blind by 10.6% on friction patches flush with
# the floor -- has no clip, because rendering needs a deploy.yaml per robot and
# the export that writes one has been broken since these runs finished. It is
# fixed now, and the biped locomotion task is exactly the shape that export code
# was written for, so this should be the easy case.
set -uo pipefail
cd "$UPSTREAM"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
L="$UPSTREAM/logs/rsl_rl/biped"

run_one() {   # task run_glob label
    local task=$1 glob=$2 label=$3 run
    run=$(ls -dt "$L"/*"$glob" 2>/dev/null | head -1)
    if [ -z "$run" ]; then echo "SKIP $label: no run matching *$glob"; return; fi
    echo "=== $label  task=$task  run=$(basename "$run") ==="
    local marker marker2; marker=$(mktemp); marker2=$(mktemp)
    "$PY" "$REPO/scripts/train_play.py" \
        --task "$task" --num_envs 4 --headless --enable_cameras \
        --video --video_length "${VIDEO_LEN:-300}" \
        --load_run "$(basename "$run")" || true
    # Count artefacts, never the exit code -- train_play exits 0 on failure.
    local nv nc
    nv=$(find "$L" -name '*.mp4' -newer "$marker" 2>/dev/null | wc -l)
    nc=$(find "$UPSTREAM/configs" "$run" -name '*.yaml' -o -name '*.onnx' 2>/dev/null \
         | xargs -r ls -t 2>/dev/null | head -1)
    echo "  video: $nv new mp4"
    [ -n "$nc" ] && echo "  newest export artefact: $nc"
    rm -f "$marker" "$marker2"
    # Keep the deploy config under a name that says which arm it came from,
    # because train_play always writes configs/policy_latest.yaml.
    # Only if this run wrote it. train_play always writes the same filename, so
    # an unconditional copy silently saves a previous arm's config under this
    # arm's name -- which is how the first attempt "saved" a deploy config from
    # a run that had just died with a KeyError.
    if [ "$UPSTREAM/configs/policy_latest.yaml" -nt "$marker2" ] 2>/dev/null; then
        cp "$UPSTREAM/configs/policy_latest.yaml" "$REPO/results/deploy_${label}.yaml"
        echo "  saved results/deploy_${label}.yaml"
    else
        echo "  no deploy config written by this run"
    fi
}

mkdir -p "$REPO/results"
run_one Velocity-BHL-Biped-Ice-v0       ppo-ice-blind-s0   ice_blind
run_one Velocity-BHL-Biped-Ice-Depth-v0 ppo-ice-depth-s0   ice_depth

echo "=== artefacts ==="
ls -la "$REPO"/results/deploy_ice_*.yaml 2>/dev/null
find "$L" -name '*.mp4' -newermt '-2 hours' 2>/dev/null | head
