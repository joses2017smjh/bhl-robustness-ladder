#!/bin/bash
#SBATCH --signal=B:TERM@30
# Open-loop native smoke: recorded v1 bounded stereo frames through the new
# Atlas-telemetry runtime and the previous runtime. CPU only.
set -euo pipefail
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
R=/nfs/hpc/share/sanchej7/humanoid-research-20261011
V=$R/processing-schedule-v1-20261012
REPO=$R/followups-20261014-stereo/repo
COMMIT=685ca3c
SIF=/nfs/hpc/share/sanchej7/Humanoid_Lite/container/bhl.sif
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
NEW=$R/followups-20261014-stereo/native-atlas-build-v1/runtime.tar.gz
NEW_SHA=df8f927939837cde5edc7c81d75b41c94d27be06f44636e5aa362cb48a37c72c
OLD_SHA=91ca4b6e4ea8f318998f400d4f7dfcd6fe8ae88b375399c79d5d633c39d71516
DEST=$R/followups-20261014-stereo/native-replay-smoke-v1
WORK=$(mktemp -d /tmp/bhl-stereo-replay-${SLURM_JOB_ID}-XXXXXX)
trap 'rm -rf "$WORK"' EXIT
echo "host=$(hostname) job=$SLURM_JOB_ID"
mkdir -p "$WORK/source" "$WORK/input" "$WORK/home" "$DEST"
git -C "$REPO" archive "$COMMIT" scripts src | tar -x -C "$WORK/source"
tar -xzf "$R/native-sensor-controls-v2/frozen-campaign.tar.gz" -C "$WORK/input" inputs/runtime/orb-runtime.tar.gz
OLD=$WORK/input/inputs/runtime/orb-runtime.tar.gz
for spec in \
  "same_map_relocalization schedule-s750000-camera_10hz-straight-stereo bounded_latest-straight-s750000-camera_10hz-stereo" \
  "active_map_reset_same_id schedule-s750000-nominal-occluders-stereo bounded_latest-occluders-s750000-nominal-stereo" \
  "new_atlas_map schedule-s750000-nominal-straight-stereo bounded_latest-straight-s750000-nominal-stereo"; do
  set -- $spec
  label=$1; shard=$V/$2-21801949/outputs.tar.gz; episode=$3
  apptainer exec --containall --cleanenv --home "$WORK/home" \
    --bind /nfs/hpc/share/sanchej7:/nfs/hpc/share/sanchej7:ro --bind "$WORK:$WORK" --pwd "$WORK" "$SIF" \
    "$PY" "$WORK/source/scripts/native/orb_atlas_replay.py" --shard-archive "$shard" --episode "$episode" \
      --runtime atlas_telemetry "$NEW" "$NEW_SHA" --runtime previous "$OLD" "$OLD_SHA" \
      --output "$WORK/out/$label" --pace 1.5
  mkdir -p "$DEST/$label"
  cp "$WORK/out/$label/replay_result.json" "$DEST/$label/"
  for run in atlas_telemetry previous; do
    mkdir -p "$DEST/$label/$run"
    cp "$WORK/out/$label/$run/native_frames.jsonl" "$DEST/$label/$run/"
    cp "$WORK/out/$label/$run/process/native-orb.log" "$DEST/$label/$run/"
  done
done
"$PY" - "$DEST" "$COMMIT" "$REPO" <<'PYEOF'
import json, os, pathlib, subprocess, sys
from datetime import datetime, timezone
dest = pathlib.Path(sys.argv[1])
summary = {}
for result in sorted(dest.glob('*/replay_result.json')):
    r = json.loads(result.read_text())
    summary[result.parent.name] = {"episode": r["episode"], "frames": r["frames"], "original": {k: r["original_closed_loop"][k] for k in ("log_relocalized", "log_new_maps", "log_active_map_resets")},
        "runs": {k: {x: v[x] for x in ("responses", "responses_with_atlas_telemetry", "log_relocalized", "log_new_maps", "log_active_map_resets", "wrapper_reset_flags", "map_changed_flags", "inconsistent_snapshots", "legacy_vs_atlas_map_disagreements", "native_compute_p95_ms")} | {"legacy_invalidations": v["legacy_epochs"]["coordinate_invalidations"], "atlas_invalidations": None if v["atlas_epochs"] is None else [e["reasons"] for e in v["atlas_epochs"]["events"]], "tracked_frames": v["tracking"]["tracked_frames"]} for k, v in r["runs"].items()}}
receipt = {"schema": "bhl-stereo-recovery-replay-smoke-receipt-v1", "job_id": os.environ["SLURM_JOB_ID"], "host": os.uname().nodename,
           "source_commit": subprocess.check_output(["git", "-C", sys.argv[3], "rev-parse", sys.argv[2]], text=True).strip(),
           "finished_utc": datetime.now(timezone.utc).isoformat(), "episodes": summary}
(dest / "replay-smoke-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=1))
PYEOF
