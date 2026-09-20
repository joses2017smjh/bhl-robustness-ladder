#!/bin/bash
set -euo pipefail
condition=${1:?baseline or adapted}; training_seed=${2:-0}
base=$(dirname "$REPO")
classes=(pant_short top_short top_long pant_long)
garment=${classes[${SLURM_ARRAY_TASK_ID:?array index}]}
case "$condition" in
    baseline) policy="$base/lehome-data/outputs/train/bc_smolvla_raster_ft_full"; label="baseline-$garment" ;;
    adapted) policy="$REPO/results/weekend-20260919/fold-adapt-s${training_seed}/best.json"; label="adapt${training_seed}-$garment" ;;
    *) echo "Unknown condition $condition" >&2; exit 2 ;;
esac
# Every checkpoint receives the same initialization seed and full fold horizon.
exec bash "$REPO/slurm/inner/weekend_fold.sh" eval 100 "$policy" "$label" 2 600 --garment-type "$garment"
