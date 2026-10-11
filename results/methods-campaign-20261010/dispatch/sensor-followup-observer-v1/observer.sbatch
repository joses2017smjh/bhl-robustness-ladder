#!/bin/bash
set -euo pipefail
export TMPDIR=/tmp
exec /nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python /nfs/stak/users/sanchej7/humanoid-methods-20261010/sensor-followup-observer-v1/sensor_followup_observer.py --plan /nfs/stak/users/sanchej7/humanoid-methods-20261010/sensor-followup-observer-v1/plan.json --plan-sha256 c7687fdd60ec915e8c938bfba81562f57700d5be4c1342e43835a034090019f7
