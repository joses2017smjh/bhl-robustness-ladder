#!/bin/bash
# The one package that has kept RGB training from ever running.
#
# RTX rendering on 6.0 was solved and measured: rgb 256x256, mean 222.2,
# std 24.4, 1,902 unique colours. What was never done is training *with* it --
# job 21036909 died in four minutes on `ModuleNotFoundError: No module named
# 'rsl_rl'`, because the v60 venv was built with Isaac Sim and Isaac Lab but no
# RL library. Both arms failed the same way and it was read as an RGB problem.
set -euo pipefail
V="$UV_PROJECT_ENVIRONMENT"
echo "target venv: $V"
"$V/bin/python" -c "import sys; print('python', sys.version.split()[0])"
uv pip install --python "$V/bin/python" "rsl-rl-lib==3.0.1"
echo "=== verify ==="
"$V/bin/python" - <<'PY'
import rsl_rl, importlib.metadata as md
print("rsl_rl", md.version("rsl-rl-lib"))
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
print("runners and modules import OK")
PY
