#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Standing reach envelope vs collapsed reach, in world z above the floor.

The design question is narrow: is there a band of heights a standing robot can
reach and a collapsed one cannot? If yes, putting the payload in that band makes
standing instrumentally necessary and the collapse stops paying, without adding
a single penalty term.
"""
import numpy as np, mujoco, itertools, sys
from pathlib import Path
sys.path.insert(0, "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/src")
from bhl_robust.eval.coop_replay import build_crew, JOINTS, PINCH_POSE

UP = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/external/Berkeley-Humanoid-Lite")
CD = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/mjcf_cache")
m, slots, crates = build_crew(UP, CD, 2, ego_camera=False, payload="cube")
d = mujoco.MjData(m)
s = slots[0]
jid = lambda j: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, s.prefix + j)
adr = {j: int(m.jnt_qposadr[jid(j)]) for j in JOINTS}
rng = {j: m.jnt_range[jid(j)] for j in JOINTS}
q0 = {j: PINCH_POSE.get(j, 0.0) for j in JOINTS}

def fk(over, root_z):
    d.qpos[:] = 0.0
    d.qpos[s.qpos_adr + 2] = root_z
    d.qpos[s.qpos_adr + 3] = 1.0
    for j, v in {**q0, **over}.items():
        d.qpos[adr[j]] = v
    mujoco.mj_forward(m, d)
    lowest = min(float(d.geom_xpos[g][2]) for g in range(m.ngeom)
                 if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                     int(m.geom_bodyid[g])) or "").startswith(s.prefix))
    return d.xpos[s.hands][:, 2].copy(), lowest

# Plant the feet: shift the root until the lowest body geom sits on the floor.
_, low = fk({}, 0.0)
root_stand = -low
hz, low2 = fk({}, root_stand)
print(f"standing: root_z={root_stand:+.3f}  lowest geom={low2:+.3f}  "
      f"hands z={hz.mean():+.3f}")

SH = [j for j in JOINTS if "shoulder_pitch" in j]
RO = [j for j in JOINTS if "shoulder_roll" in j]
EL = [j for j in JOINTS if "elbow_pitch" in j]
def sweep(root_z, label):
    zs = []
    grids = {}
    for group in (SH, RO, EL):
        for j in group:
            lo, hi = rng[j]
            grids[j] = np.linspace(lo, hi, 7) if hi > lo else np.linspace(-2.0, 2.0, 7)
    keys = SH[:1] + RO[:1] + EL[:1]      # one arm; the pair is symmetric
    for combo in itertools.product(*[grids[k] for k in keys]):
        hz, _ = fk(dict(zip(keys, combo)), root_z)
        zs.append(float(hz.mean()))
    zs = np.array(zs)
    print(f"{label:22} hands z: min={zs.min():+.3f}  max={zs.max():+.3f}  "
          f"(n={len(zs)})")
    return zs.min(), zs.max()

st = sweep(root_stand, "standing")
co = sweep(root_stand - 0.41, "collapsed (-41 cm)")
lo_band, hi_band = max(co[1], 0.0), st[1]
print()
print(f"collapsed hands top out at {co[1]:+.3f} m")
print(f"standing hands reach up to {st[1]:+.3f} m")
print(f"==> payload grasp band: {co[1]:.2f} m .. {st[1]:.2f} m "
      f"({100*(st[1]-co[1]):.0f} cm wide)")
PY
