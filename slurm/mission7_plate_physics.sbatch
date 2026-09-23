#!/bin/bash
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
set -euo pipefail
repo=${1:?}; snapshot=${2:?}; campaign=${3:?}; out=${4:?}
cd "$repo"; PY="${PYTHON:-python3}"
export PYTHONPATH="$snapshot/src:$snapshot/scripts" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$PY" - "$snapshot" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1])
for n,e in json.loads((p/'sha256.json').read_text()).items(): assert hashlib.sha256((p/n).read_bytes()).hexdigest()==e,n
PY
exec "$PY" "$snapshot/scripts/mission7_plate_physics.py" --repo "$repo" --campaign "$campaign" --out "$out"
