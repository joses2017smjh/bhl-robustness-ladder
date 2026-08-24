#!/bin/bash
# Create the 6.0 venv and answer the only question that matters: does RTX render.
set -euo pipefail
V="$UV_PROJECT_ENVIRONMENT"

if [ ! -x "$V/bin/python" ]; then
    echo "=== creating $V (python 3.12) ==="
    uv venv --python 3.12 "$V"
fi

echo "=== installing isaacsim 6.0.0.1 + isaaclab 3.0.0b2 ==="
# --index-strategy unsafe-best-match: several isaacsim sub-packages resolve on
# pypi.org rather than the NVIDIA index, and uv refuses to cross indexes by
# default. The pinning is exact, so the dependency-confusion risk the default
# guards against does not apply here.
uv pip install --python "$V/bin/python" \
    --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match \
    "isaacsim[all,extscache]==6.0.0.1" "isaaclab[isaacsim]==3.0.0b2"

# The overlays import upstream's asset and lowlevel packages, which are source
# trees in external/ rather than wheels. The port audit's second round failed
# entirely on their absence -- ModuleNotFoundError, not an API change -- because
# the first build installed Isaac Sim and Isaac Lab and nothing else. Editable,
# matching how the v51 stack has them, so external/ stays the single source.
echo "=== installing the upstream BHL packages (editable) ==="
for pkg in berkeley_humanoid_lite_assets berkeley_humanoid_lite_lowlevel berkeley_humanoid_lite; do
    uv pip install --python "$V/bin/python" --no-deps -e "$UPSTREAM/source/$pkg"
done

# The MuJoCo evaluation harness drives policies through upstream's RlController,
# which loads an ONNX export. Round three of the port audit failed on its
# absence and nothing else.
uv pip install --python "$V/bin/python" onnxruntime

echo "=== version check ==="
"$V/bin/python" - <<'PY'
import importlib.metadata as md
for pkg in ("isaacsim", "isaaclab", "torch"):
    try:
        print(f"  {pkg} {md.version(pkg)}")
    except Exception as e:
        print(f"  {pkg} MISSING ({e})")
PY

echo "=== the question: does the RTX renderer produce pixels on 6.0? ==="
"$V/bin/python" "$REPO/scripts/bench/rtx60_probe.py" || echo "RTX PROBE FAILED"
