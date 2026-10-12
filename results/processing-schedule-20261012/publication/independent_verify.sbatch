#!/bin/bash
set -euo pipefail
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
OPS=/nfs/hpc/share/sanchej7/humanoid-research-20261011/processing-schedule-publication-20261012
"$PY" - "$OPS/independent_verify.py" "$1" <<'VERIFY'
import hashlib,pathlib,sys
p=pathlib.Path(sys.argv[1])
if hashlib.sha256(p.read_bytes()).hexdigest()!=sys.argv[2]:
 raise SystemExit('Independent verifier changed after submission')
VERIFY
exec "$PY" "$OPS/independent_verify.py"
