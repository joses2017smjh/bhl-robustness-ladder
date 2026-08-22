#!/bin/bash
# Build the 6.0.1 environment beside the locked 5.1 one, then probe the renderer.
#
# `venv60` is a second interpreter, not a second set of packages in the first.
# It cannot be anything else: Isaac Sim 6.0 requires Python 3.12 and the locked
# stack is 3.11, and the distribution name is `isaacsim` in both cases, so one
# environment can hold exactly one of them. That constraint is the whole answer
# to "can I add 6.0.1 without touching 5.1", and it is worth having the script
# state it rather than discovering it in a resolver error.
set -euo pipefail

V60=$WORKSPACE/venv60
PY60=$V60/bin/python

if [ ! -x "$PY60" ] || ! "$PY60" -c "import isaacsim" 2>/dev/null; then
    echo "=== 0. installing isaacsim 6.0.1 into venv60 (python 3.12) ==="
    uv venv --python 3.12 "$V60"
    # `extscache` matters: the extension cache wheels are what carry the RTX
    # plugins that 5.1 dies inside. Installing `all` without them gets a Kit
    # that cannot make a Hydra engine at all, which would look like the same
    # failure for a completely different reason.
    VIRTUAL_ENV="$V60" uv pip install --python "$PY60" \
        --extra-index-url https://pypi.nvidia.com \
        --index-strategy unsafe-best-match --prerelease=allow \
        "isaacsim[all,extscache]==6.0.1.0"
    echo "install done: $(du -sh "$V60" | cut -f1)"
else
    echo "=== 0. venv60 already has isaacsim; skipping install ==="
fi

"$PY60" - <<'PY'
import importlib.metadata as md
for p in ("isaacsim", "isaacsim-kernel", "isaacsim-extscache-kit"):
    try:
        print(f"  {p:26} {md.version(p)}")
    except md.PackageNotFoundError:
        print(f"  {p:26} (absent)")
PY

echo
echo "=== the probe ==="
cd "$REPO"
exec "$PY60" scripts/bench/rtx_probe.py \
    --counts ${RTX_COUNTS:-1 4 16 64} \
    --res ${RTX_RES:-64 128} \
    --json "$REPO/results/rtx_probe.json"
