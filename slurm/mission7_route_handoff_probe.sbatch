#!/bin/bash
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
set -euo pipefail

# Only the three paths are positional.  Everything after them is forwarded to
# the probe verbatim.
#
# This used to take eight positional arguments, and adding a parameter meant
# editing the submitter's command list, this file's ${N} parsing and the
# runner's argparse in lockstep with nothing checking they agreed.  Job
# 21399502 is what that costs: it ran, exited 0, and recorded an intervention
# that never happened.  Pass-through removes the failure mode rather than
# documenting it.
repo=${1:?repo}; snapshot=${2:?snapshot}; out=${3:?out}; shift 3
cd "$repo"

# cn-c22's bare python3 does not provide the validated MuJoCo/numpy stack.
# Keep the environment override for portability, but default to the shared
# Humanoid Lite venv used by the completed Mission 7 route jobs.
PY="${PYTHON:-/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python}"
export PYTHONPATH="$snapshot/src:$snapshot/scripts" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

probe="$snapshot/scripts/mission7_route_handoff_probe.py"

# 1. Source integrity: the snapshot must be the bytes that were submitted.
"$PY" - "$snapshot" <<'PYEOF'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for name, expected in json.loads((root / "sha256.json").read_text()).items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
PYEOF

# 2. Preflight on the EXACT argument list the science will use, before spending
#    queue time.  Importing the probe proves this interpreter has numpy and
#    mujoco and that the snapshot is import-closed; parsing proves every
#    forwarded flag is accepted.  Between them these are the causes of
#    21396684, 21397985, 21398501, 21399179 and 21399201.
"$PY" "$probe" --repo "$repo" --out "$out" "$@" --preflight

# 3. Science.
exec "$PY" "$probe" --repo "$repo" --out "$out" "$@"
