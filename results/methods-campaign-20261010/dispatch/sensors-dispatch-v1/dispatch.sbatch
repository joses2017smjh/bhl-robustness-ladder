#!/bin/bash
set -euo pipefail
exec /nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python /nfs/stak/users/sanchej7/humanoid-methods-20261010/sensors-dispatch-v1/methods_dispatch.py --plan /nfs/stak/users/sanchej7/humanoid-methods-20261010/sensors-dispatch-v1/plan.json --plan-sha256 7035bb51fc41d72fa15c484093a4626e9e279838367f3c974f1777823cc202ef
