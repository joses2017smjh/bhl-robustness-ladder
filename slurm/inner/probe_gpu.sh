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
    archs = torch.cuda.get_arch_list()
    # Exact membership in arch_list is NOT the test, and reporting it as one is
    # actively misleading: cubins are forward-compatible across minor revisions
    # within a major version, so an sm_86 binary runs on an sm_89 L40S even
    # though "sm_89" never appears in the list. What does not work is going
    # below the lowest major -- sm_70 against a floor of sm_75 -- which is why
    # the V100 nodes fail with "no kernel image is available". The matmul is
    # the real test; the list is only context for reading a failure.
    same_major = [a for a in archs if a.startswith(f"sm_{d.major}")]
    note = "exact" if cap in archs else (
        f"minor-compat via {min(same_major)}" if same_major else "NO COMPATIBLE CUBIN")
    x = torch.randn(4096, 4096, device="cuda")
    _ = (x @ x).sum().item()
    line.append(f"torch {torch.__version__} | {d.name} {d.total_memory//2**20}MiB {cap} "
                f"| arch_list={archs} | cubin={note} | matmul OK")
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
