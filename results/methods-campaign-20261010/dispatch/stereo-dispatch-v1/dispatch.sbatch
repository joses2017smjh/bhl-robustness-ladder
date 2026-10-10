#!/bin/bash
set -euo pipefail
umask 077
export TMPDIR=/tmp
export PYTHONDONTWRITEBYTECODE=1
export PATH="$PATH:/nfs/stak/users/sanchej7/.local/bin"
exec /nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python /nfs/stak/users/sanchej7/humanoid-methods-20261010/stereo-dispatch-v1/methods_dispatch.py --plan /nfs/stak/users/sanchej7/humanoid-methods-20261010/stereo-dispatch-v1/plan.json --plan-sha256 7e3a7db98f8f96eb882b7574e7ada8af530585ab9e71b5be3148e4f9b1c57080
