#!/bin/bash
set -euo pipefail
mode=${1:?train or eval}; seed=${2:?seed}; shift 2
base=$(dirname "$REPO")
campaign="$REPO/results/weekend-20260919"
if [ "$mode" = train ]; then
    steps=${1:-1500}; label=${2:-adapt}; shift 2
    mkdir -p "${TMPDIR:-/tmp}/bhl-fold-cache-s${seed}"
    exec "$PY" "$REPO/scripts/cloth/finetune_fold.py" \
        --policy-path "$base/lehome-data/outputs/train/bc_smolvla_seed0/checkpoints/030000/pretrained_model" \
        --captures "$base/lehome-data/storm_capture100/*.npz" \
        --out "$campaign/fold-${label}-s${seed}" \
        --cache "${TMPDIR:-/tmp}/bhl-fold-cache-s${seed}" \
        --steps "$steps" --save-every 100 --batch-size 4 \
        --unfreeze vision+action --seed "$seed" "$@"
elif [ "$mode" = eval ]; then
    policy=${1:?policy path}; label=${2:?output label}; episodes=${3:-2}; steps=${4:-600}; shift 4
    lehome="$base/lehome-fold-repro"
    export PYTHONPATH="$base/lehome51-site:$lehome/external/lehome-challenge/source/lehome:$lehome/external/lehome-challenge:$lehome/src:$REPO/src"
    libs=("$base"/venv/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-*)
    export PXR_PLUGINPATH_NAME="${libs[0]}/bin/usd"
    export HF_HOME="$base/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
    exec "$PY" "$REPO/scripts/cloth/eval_fold.py" \
        --lehome-repo "$lehome" --assets "$base/lehome-data/Assets" \
        --dataset-root "$base/lehome-data/Datasets/example/four_types_merged" \
        --policy-path "$policy" --out "$campaign/fold-${label}-s${seed}.json" \
        --garment-type pant_short --episodes "$episodes" --max-steps "$steps" --seed "$seed" "$@"
else
    echo "Unknown fold mode: $mode" >&2; exit 2
fi
