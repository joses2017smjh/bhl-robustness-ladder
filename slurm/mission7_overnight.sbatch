#!/bin/bash
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
set -euo pipefail
repo=${1:?repository}; snapshot=${2:?source snapshot}; campaign=${3:?campaign}
cd "$repo"
PY="${PYTHON:-python3}"
export PYTHONPATH="$snapshot/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$PY" - "$snapshot" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
for rel,expected in json.loads((root/'sha256.json').read_text()).items():
    assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==expected,rel
PY
exec "$PY" "$snapshot/scripts/mission7_overnight.py" --repo "$repo" --campaign "$campaign" --index "$SLURM_ARRAY_TASK_ID"
