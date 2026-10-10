#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/tmp
/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python /nfs/stak/users/sanchej7/humanoid-methods-20261010/finalization/v2/methods_finalize_bootstrap.py --plan /nfs/stak/users/sanchej7/humanoid-methods-20261010/finalization/v2/bootstrap-plan.json --plan-sha256 5f44f261b68f1dc44eb232d3f44ce726fd119691fa869f21daa136b2b27d5496
