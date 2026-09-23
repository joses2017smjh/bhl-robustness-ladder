#!/bin/bash
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
set -euo pipefail
# Five positional paths; everything after them is forwarded verbatim.
repo=${1:?repo}; snapshot=${2:?snapshot}; campaign=${3:?campaign}; baseline=${4:?baseline}; out=${5:?out}; shift 5
cd "$repo"
PY="${PYTHON:-/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python}"
export PYTHONPATH="$snapshot/src:$snapshot/scripts" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
script="$snapshot/scripts/mission7_plate_stage.py"
"$PY" - "$snapshot" <<'PYEOF'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for name, expected in json.loads((root / "sha256.json").read_text()).items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
PYEOF
"$PY" "$script" --repo "$repo" --campaign "$campaign" --baseline "$baseline" --out "$out" "$@" --preflight
exec "$PY" "$script" --repo "$repo" --campaign "$campaign" --baseline "$baseline" --out "$out" "$@"
