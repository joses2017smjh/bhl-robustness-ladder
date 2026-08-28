#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Base height relative to the feet: is the pelvis above them or on the floor?

tilt() is arccos(R[2,2]) of the base -- it measures whether the torso is level,
not whether the robot is standing. A machine that sits straight down with its
torso level passes it. Foot-to-pelvis height is the check that cannot.
"""
import numpy as np, sys, mujoco
from pathlib import Path
sys.path.insert(0, "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/src")
from bhl_robust.eval.coop_replay import build_crew, CoopActor, CrewRunner, POLICY_DT
UP = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/external/Berkeley-Humanoid-Lite")
CD = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/mjcf_cache")
L = UP/"logs/rsl_rl/coop_lift"
def ck(pat):
    r = sorted(L.glob(pat))[-1]
    return sorted(r.glob("model_*.pt"), key=lambda q:int(q.stem.split("_")[1]))[-1]

for scene, pat, seed in (("cube","*coop-cube-staged-r2-s0",11),
                         ("ladder","*coop-ladder-staged-s0",0)):
    m, slots, crates = build_crew(UP, CD, 2, ego_camera=True, payload=scene)
    r = CrewRunner(m, slots, crates, CoopActor(ck(pat)))
    r.reset(np.random.default_rng(seed))
    # Lowest geom of robot 0, and its pelvis, both in world z.
    pre = slots[0].prefix
    gids = [g for g in range(m.ngeom)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                  int(m.geom_bodyid[g])) or "").startswith(pre)]
    def stat():
        mujoco.mj_forward(r.m, r.d)
        low = min(float(r.d.geom_xpos[g][2]) for g in gids)
        pel = float(r.d.xpos[slots[0].body_id][2])
        return low, pel
    print(f"\n{scene}: pelvis height above lowest body geom, robot 0")
    for t in range(300):
        if t % 60 == 0:
            low, pel = stat()
            print(f"  t={t*POLICY_DT:5.2f}s  lowest geom z={low:+.3f}  "
                  f"pelvis z={pel:+.3f}  pelvis above lowest={pel-low:+.3f} m  "
                  f"tilt={r.tilt(0):.2f}")
        r.step()
    low, pel = stat()
    print(f"  t=12.00s  lowest geom z={low:+.3f}  pelvis z={pel:+.3f}  "
          f"pelvis above lowest={pel-low:+.3f} m  tilt={r.tilt(0):.2f}")
    r.close()
PY
