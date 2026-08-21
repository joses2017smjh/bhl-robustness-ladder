#!/bin/bash
# Inside the container: does the stack this project needs actually work here?
set -uo pipefail
cd "$UPSTREAM"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
import os, socket
line = [f"HOST {socket.gethostname()}"]
try:
    import torch
    d = torch.cuda.get_device_properties(0)
    cap = f"sm_{d.major}{d.minor}"
    arch_ok = cap in [a.replace("sm_", "sm_") for a in torch.cuda.get_arch_list()]
    x = torch.randn(4096, 4096, device="cuda")
    _ = (x @ x).sum().item()
    line.append(f"torch {torch.__version__} | {d.name} {d.total_memory//2**20}MiB {cap} "
                f"| arch_list={torch.cuda.get_arch_list()} | supported={arch_ok} | matmul OK")
except Exception as e:
    line.append(f"TORCH FAIL {type(e).__name__}: {e}")
try:
    import warp as wp
    wp.init()
    line.append(f"warp {wp.config.version} OK on {wp.get_device()}")
except Exception as e:
    line.append(f"WARP FAIL {type(e).__name__}: {e}")
print("PROBE | " + " || ".join(line), flush=True)
PY
