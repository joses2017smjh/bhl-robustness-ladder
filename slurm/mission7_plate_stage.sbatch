#!/bin/bash
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
set -euo pipefail
repo=${1:?repo}; snapshot=${2:?snapshot}; campaign=${3:?campaign}; baseline=${4:?baseline}; out=${5:?out}
cd "$repo"
PY="${PYTHON:-python3}"
export PYTHONPATH="$snapshot/src:$snapshot/scripts" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$PY" - "$snapshot" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for name, expected in json.loads((root / "sha256.json").read_text()).items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
PY
exec "$PY" "$snapshot/scripts/mission7_plate_stage.py" \
  --repo "$repo" --campaign "$campaign" --baseline "$baseline" --out "$out"
