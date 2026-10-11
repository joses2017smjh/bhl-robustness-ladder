#!/bin/bash
set -euo pipefail
umask 077
export TMPDIR=/tmp
export PYTHONDONTWRITEBYTECODE=1
export PATH="$PATH:/nfs/stak/users/sanchej7/.local/bin"
exec /nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python /nfs/stak/users/sanchej7/humanoid-methods-20261010/terrain-dispatch-v1/methods_dispatch.py --plan /nfs/stak/users/sanchej7/humanoid-methods-20261010/terrain-dispatch-v1/plan.json --plan-sha256 ca1551302b518f7b352a94560b3e4ba5623a65efaaf51885ed330c01584479a4
