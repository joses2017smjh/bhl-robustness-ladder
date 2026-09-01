#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Is the cloth in G-C1 actually being simulated, or just drawn?

2,357 steps/s at 16 envs and 2,036 at 1,024 is 14% slower for 64x the cloth.
Cloth does not scale like that, so the null hypothesis is that the prims loaded
as static meshes and the solver has nothing to integrate -- the same shape as
the one-step episodes, where a job reported healthy numbers while measuring
nothing.

The test is whether the vertices move. A simulated cloth dropped above the
ground deforms; a static mesh does not move by a micron.
"""
import argparse, glob
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args([]); a.headless = True
app = AppLauncher(a)

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from pxr import Usd, UsdGeom, PhysxSchema

cloth = sorted(glob.glob("/nfs/hpc/share/sanchej7/Humanoid_Lite/venv-isaac60/lib/"
                         "python3.12/site-packages/warp/examples/assets/square_cloth.usd"))[0]
sim = SimulationContext(SimulationCfg(dt=1/60.0, device=a.device if hasattr(a,'device') else "cuda:0"))
cfg = sim_utils.UsdFileCfg(usd_path=cloth)
cfg.func("/World/cloth_0", cfg, translation=(0.0, 0.0, 2.0))
sim.reset()

stage = sim.stage
def verts():
    for pr in stage.Traverse():
        if pr.GetPath().pathString.startswith("/World/cloth_0") and pr.IsA(UsdGeom.Mesh):
            pts = UsdGeom.Mesh(pr).GetPointsAttr().Get()
            return np.asarray(pts, dtype=float) if pts else None
    return None

v0 = verts()
print(f"mesh found: {v0 is not None}; vertices: {0 if v0 is None else len(v0)}")
# Is any physics schema applied at all?
sch = []
for pr in stage.Traverse():
    if pr.GetPath().pathString.startswith("/World/cloth_0"):
        sch += [s for s in pr.GetAppliedSchemas() if "Physx" in s or "Physics" in s]
print(f"physics schemas on the cloth prim: {sorted(set(sch)) or 'NONE'}")

for _ in range(120):
    sim.step()
v1 = verts()
if v0 is not None and v1 is not None:
    d = float(np.abs(v1 - v0).max())
    print(f"max vertex displacement after 120 steps: {d:.6f} m")
    print("VERDICT:", "cloth is simulating" if d > 1e-4 else
          "NOT SIMULATED -- the throughput number is meaningless")
else:
    print("VERDICT: could not read mesh points")
app.app.close()
PY
