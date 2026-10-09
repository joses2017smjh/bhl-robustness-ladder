#!/bin/bash
#SBATCH --job-name=bhl-h34-finalize
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --signal=B:TERM@60
# Root submits --dependency=afterany:<full array> --export=NONE and durable logs.
set -euo pipefail
PERSIST=$1
OBSERVER=$2
OBSERVER_SHA=$3
OUT=$4
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp
cd /tmp
BOOT=$(mktemp -d /tmp/bhl-h34-finalizer-bootstrap-${SLURM_JOB_ID:-local}-XXXXXX)
"$PY" - "$OBSERVER" "$OBSERVER_SHA" "$BOOT" <<'PY'
import hashlib,pathlib,sys,tarfile
archive=pathlib.Path(sys.argv[1]); h=hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda:stream.read(1048576),b''):h.update(block)
if h.hexdigest()!=sys.argv[2]:raise SystemExit('Pinned observer archive changed; refusing execution')
with tarfile.open(archive,'r:gz') as tar:
    member=tar.getmember('scripts/bench/h34_finalize.py')
    if not member.isfile() or member.size>2097152:raise SystemExit('Invalid finalizer bootstrap')
    with tar.extractfile(member) as stream:
        (pathlib.Path(sys.argv[3])/'h34_finalize.py').write_bytes(stream.read())
PY
exec "$PY" "$BOOT/h34_finalize.py" finalize --persistent-dir "$PERSIST" --observer-archive "$OBSERVER" --observer-sha "$OBSERVER_SHA" --out "$OUT"
