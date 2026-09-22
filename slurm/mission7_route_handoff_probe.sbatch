#!/bin/bash
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
set -euo pipefail
repo=${1:?repo}; snapshot=${2:?snapshot}; out=${3:?out}; stage=${4:?stage}; indices=${5:?indices}; handoff=${6:-switch}
cd "$repo"
# cn-c22's bare python3 does not provide the validated MuJoCo/numpy stack.
# Keep the environment override for portability, but default to the shared
# Humanoid Lite venv used by the completed Mission 7 route jobs.
PY="${PYTHON:-/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python}"
export PYTHONPATH="$snapshot/src:$snapshot/scripts" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$PY" - "$snapshot" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for name, expected in json.loads((root / "sha256.json").read_text()).items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
PY
exec "$PY" "$snapshot/scripts/mission7_route_handoff_probe.py" \
  --repo "$repo" --out "$out" --stage "$stage" --indices "$indices" --handoff "$handoff"
