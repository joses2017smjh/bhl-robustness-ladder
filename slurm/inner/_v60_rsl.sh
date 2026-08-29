#!/bin/bash
# Isaac Lab 3.0.0b2 pins rsl-rl-lib==5.0.1, not the 3.0.1 the v51 stack runs.
#
# I installed 3.0.1 to match v51, on the assumption that matching the other
# stack was the conservative choice. It is not: isaaclab_rl's setup.py names
# the version it was written against, and 3.0.1's PPO has no `optimizer`
# keyword, so all nine v2 arms died on
#   TypeError: PPO.__init__() got an unexpected keyword argument 'optimizer'
#
# Worth recording for the results: v51 trains on rsl-rl 3.0.1 and v60 on 5.0.1,
# so a v60 number and a v51 number differ by RL library as well as by simulator.
# The nine v2 cells are all on v60 and internally comparable; nothing from them
# belongs in a table with a v51 figure.
set -euo pipefail
V="$UV_PROJECT_ENVIRONMENT"
echo "target venv: $V"
uv pip install --python "$V/bin/python" "rsl-rl-lib==5.0.1" "onnxscript>=0.5"
echo "=== verify ==="
"$V/bin/python" - <<'PY'
import importlib.metadata as md, inspect
from rsl_rl.algorithms import PPO
print("rsl_rl", md.version("rsl-rl-lib"))
sig = inspect.signature(PPO.__init__)
print("PPO accepts 'optimizer':", "optimizer" in sig.parameters)
from rsl_rl.runners import OnPolicyRunner  # noqa: F401
print("runner imports OK")
PY
