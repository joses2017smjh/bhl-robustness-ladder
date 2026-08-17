#!/bin/bash
# Reports the resolved stack. Does NOT bare-import isaaclab: Carbonite requires
# SimulationApp to be instantiated first, so a plain `import isaaclab` aborts.
# Task registration is verified by the smoke train instead.
set -euo pipefail
cd "$UPSTREAM"
echo "python: $($PY -V)"
$PY - <<'PYEOF'
import importlib.metadata as md
import torch
print("torch      :", torch.__version__)
print("cuda avail :", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
for p in ["isaacsim", "isaaclab", "isaaclab-tasks", "isaaclab-rl",
          "rsl-rl-lib", "mujoco", "onnxruntime"]:
    try:
        print(f"{p:14s}:", md.version(p))
    except Exception:
        print(f"{p:14s}: NOT INSTALLED")
for p in ["berkeley_humanoid_lite", "berkeley_humanoid_lite_assets",
          "berkeley_humanoid_lite_lowlevel"]:
    print(f"{p:32s}:", md.version(p))
PYEOF
