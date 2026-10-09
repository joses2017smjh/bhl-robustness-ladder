#!/bin/bash
# Resources and output destinations are supplied by h34_campaign.py submit.
# The bootstrap reads its own implementation from the already pinned archive,
# never the mutable original checkout. No shared cache/path helper is sourced.
#SBATCH --signal=B:TERM@45
set -euo pipefail
PERSIST=$1
JOB_NAME=$2
EXPECTED_SHA=$3
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1
BOOT=$(mktemp -d /tmp/bhl-h34-bootstrap-${SLURM_JOB_ID:-local}-XXXXXX)
"$PY" - "$PERSIST/frozen-campaign.tar.gz" "$EXPECTED_SHA" "$BOOT" <<'PY'
import hashlib,pathlib,sys,tarfile
archive=pathlib.Path(sys.argv[1]); h=hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda:stream.read(1048576),b''):h.update(block)
if h.hexdigest()!=sys.argv[2]:raise SystemExit('Frozen archive changed; refusing launch')
with tarfile.open(archive,'r:gz') as tar:
    member=tar.getmember('source/scripts/bench/h34_campaign.py')
    if not member.isfile():raise SystemExit('Invalid bootstrap member')
    with tar.extractfile(member) as stream:
        (pathlib.Path(sys.argv[3])/'h34_campaign.py').write_bytes(stream.read())
PY
exec "$PY" "$BOOT/h34_campaign.py" execute --persistent-dir "$PERSIST" --job "$JOB_NAME" --archive-sha "$EXPECTED_SHA"
