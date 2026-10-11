#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/tmp
/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python /nfs/stak/users/sanchej7/humanoid-methods-20261010/finalization/methods_finalize_bootstrap.py --plan /nfs/stak/users/sanchej7/humanoid-methods-20261010/finalization/bootstrap-plan.json --plan-sha256 d63c6cfbe1b6e827b4d153ff0b439a199d4abbcce440f2101a270349f1ff9574
