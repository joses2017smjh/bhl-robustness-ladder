#!/bin/bash
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
set -euo pipefail
repo=${1:?}; snapshot=${2:?}; campaign=${3:?}; mode=${4:?}
cd "$repo"
PY="${PYTHON:-python3}"
export PYTHONPATH="$snapshot/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$PY" - "$snapshot" <<'PY'
import sys,pathlib,json,hashlib
p=pathlib.Path(sys.argv[1])
for name,expected in json.loads((p/'sha256.json').read_text()).items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==expected,name
PY
exec "$PY" "$snapshot/scripts/mission7_debug.py" "$mode" --repo "$repo" --campaign "$campaign" --index "${SLURM_ARRAY_TASK_ID:-0}"
