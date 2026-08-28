#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
import numpy as np, sys
from pathlib import Path
sys.path.insert(0, "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/src")
from bhl_robust.eval.coop_replay import build_crew, CoopActor, CrewRunner, DEPTH_RANGE
UP = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/external/Berkeley-Humanoid-Lite")
CD = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/mjcf_cache")
L = UP/"logs/rsl_rl/coop_lift"
def ck(pat):
    r = sorted(L.glob(pat))[-1]
    return sorted(r.glob("model_*.pt"), key=lambda q:int(q.stem.split("_")[1]))[-1]
for scene, pat, seed in (("cube","*coop-cube-staged-r2-s0",11),
                         ("ball","*coop-cube-staged-r2-s0",4),
                         ("ladder","*coop-ladder-staged-s0",0)):
    m, slots, crates = build_crew(UP, CD, 2, ego_camera=True, payload=scene)
    r = CrewRunner(m, slots, crates, CoopActor(ck(pat))); r.enable_pov(184)
    r.reset(np.random.default_rng(seed))
    acc=[]
    for t in range(150):
        r.step()
        if t % 5 == 0:
            d = r.depth_image(0)
            acc.append(np.nan_to_num(d, nan=DEPTH_RANGE, posinf=DEPTH_RANGE,
                                     neginf=DEPTH_RANGE).ravel())
    a = np.concatenate(acc); fin = a[a < DEPTH_RANGE-1e-3]
    print(f"{scene:7} finite {100*len(fin)/len(a):5.1f}%  "
          f"p1={np.percentile(fin,1):.3f} p25={np.percentile(fin,25):.3f} "
          f"p50={np.percentile(fin,50):.3f} p75={np.percentile(fin,75):.3f} "
          f"p95={np.percentile(fin,95):.3f} p99={np.percentile(fin,99):.3f}")
    r.close()
PY
