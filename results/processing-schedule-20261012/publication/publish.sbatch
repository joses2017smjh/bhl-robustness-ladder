#!/bin/bash
#SBATCH --signal=B:TERM@60
set -euo pipefail
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OPS=/nfs/hpc/share/sanchej7/humanoid-research-20261011/processing-schedule-publication-20261012
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
cd "$OPS"
"$PY" - "$OPS" "$1" <<'VERIFY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); manifest=root/'source-manifest.json'
if hashlib.sha256(manifest.read_bytes()).hexdigest()!=sys.argv[2]:
    raise SystemExit('Publisher source manifest changed')
for name,expected in json.loads(manifest.read_text()).items():
    if pathlib.Path(name).name!=name or hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
        raise SystemExit('Publisher source file changed: '+name)
VERIFY
"$PY" -m unittest -v test_publisher.py
exec "$PY" "$OPS/publish.py"
