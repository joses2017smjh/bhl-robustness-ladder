#!/bin/bash
set -euo pipefail
exec /nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python /nfs/stak/users/sanchej7/humanoid-methods-20261010/terrain-dispatch-v2/methods_dispatch.py --plan /nfs/stak/users/sanchej7/humanoid-methods-20261010/terrain-dispatch-v2/plan.json --plan-sha256 5f121dc88fa942535060c61f4c5caccb264fb6c2cdc39e8e473fcd35ddc7bfdc
