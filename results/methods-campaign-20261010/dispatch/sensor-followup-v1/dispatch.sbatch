#!/bin/bash
set -euo pipefail
export TMPDIR=/tmp
exec /nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python /nfs/stak/users/sanchej7/humanoid-methods-20261010/sensor-followup-v1/sensor_followup_deferred.py --plan /nfs/stak/users/sanchej7/humanoid-methods-20261010/sensor-followup-v1/plan.json --plan-sha256 c63ae34d02ce2c178cfd9ca38c7dab146e74db4164c8233e6a671798fb2706c9
