#!/bin/bash
#SBATCH --signal=B:TERM@30
# Deterministic native archetypes: recorded v1 frames with an all-black window.
set -euo pipefail
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
R=/nfs/hpc/share/sanchej7/humanoid-research-20261011
V=$R/processing-schedule-v1-20261012
REPO=$R/followups-20261014-stereo/repo
COMMIT=8d31e5b
SIF=/nfs/hpc/share/sanchej7/Humanoid_Lite/container/bhl.sif
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
NEW=$R/followups-20261014-stereo/native-atlas-build-v1/runtime.tar.gz
NEW_SHA=df8f927939837cde5edc7c81d75b41c94d27be06f44636e5aa362cb48a37c72c
DEST=$R/followups-20261014-stereo/native-replay-blackout-v1
WORK=$(mktemp -d /tmp/bhl-stereo-blackout-${SLURM_JOB_ID}-XXXXXX)
trap 'rm -rf "$WORK"' EXIT
echo "host=$(hostname) job=$SLURM_JOB_ID"
mkdir -p "$WORK/source" "$WORK/home" "$DEST"
git -C "$REPO" archive "$COMMIT" scripts src | tar -x -C "$WORK/source"
SHARD=$V/schedule-s750000-nominal-straight-stereo-21801949/outputs.tar.gz
EP=bounded_latest-straight-s750000-nominal-stereo
for spec in "early_reset 2.5 3.5" "short_late_loss 16.0 17.0" "long_late_loss 16.0 19.6"; do
  set -- $spec
  apptainer exec --containall --cleanenv --home "$WORK/home" \
    --bind /nfs/hpc/share/sanchej7:/nfs/hpc/share/sanchej7:ro --bind "$WORK:$WORK" --pwd "$WORK" "$SIF" \
    "$PY" "$WORK/source/scripts/native/orb_atlas_replay.py" --shard-archive "$SHARD" --episode "$EP" \
      --runtime atlas_telemetry "$NEW" "$NEW_SHA" --output "$WORK/out/$1" --pace 1.5 --blackout "$2" "$3"
  mkdir -p "$DEST/$1/atlas_telemetry"
  cp "$WORK/out/$1/replay_result.json" "$DEST/$1/"
  cp "$WORK/out/$1/atlas_telemetry/native_frames.jsonl" "$WORK/out/$1/atlas_telemetry/process/native-orb.log" "$DEST/$1/atlas_telemetry/"
done
echo "job=$SLURM_JOB_ID host=$(hostname) commit=$COMMIT" > "$DEST/job.txt"
