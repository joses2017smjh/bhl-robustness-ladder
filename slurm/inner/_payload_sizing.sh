#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""What mass can these arms actually hold, and where?

The lift has never been sized against the actuators. The cube is 0.5 kg and the
plank 1.1 kg, both chosen by eye. Arm joints are 4 Nm; legs are 6 Nm. Whether
0.5 kg is trivially easy or already marginal depends entirely on the moment arm,
and nobody has measured the moment arm.

Computed from the built model: the horizontal distance from each arm joint to
the hand, in the pinch pose and across the reach envelope. Static holding torque
for a mass m at horizontal offset d is m*g*d, so the feasible mass at a joint is
tau_limit / (g * d).
"""
import numpy as np, mujoco, sys
from pathlib import Path
sys.path.insert(0, "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/src")
from bhl_robust.eval.coop_replay import build_crew, JOINTS, PINCH_POSE

UP = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/external/Berkeley-Humanoid-Lite")
CD = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/mjcf_cache")
m, slots, _ = build_crew(UP, CD, 2, ego_camera=False, payload="cube")
d = mujoco.MjData(m); s = slots[0]
G = 9.81
ARM_TAU, LEG_TAU = 4.0, 6.0

jid = lambda j: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, s.prefix + j)
adr = {j: int(m.jnt_qposadr[jid(j)]) for j in JOINTS}
d.qpos[:] = 0.0
d.qpos[s.qpos_adr + 3] = 1.0
for j, v in PINCH_POSE.items():
    if j in adr: d.qpos[adr[j]] = v
mujoco.mj_forward(m, d)

hand = d.xpos[s.hands].mean(axis=0)
print("horizontal moment arm from each arm joint to the hand (pinch pose):")
worst = 0.0
for j in [x for x in JOINTS if x.startswith("arm_left")]:
    bid = int(m.jnt_bodyid[jid(j)])
    pos = d.xpos[bid]
    arm = float(np.linalg.norm(hand[:2] - pos[:2]))
    cap = ARM_TAU / (G * max(arm, 1e-3))
    worst = max(worst, arm)
    print(f"  {j:34} d={arm:.3f} m   -> {cap:5.2f} kg at 4 Nm")

print(f"\nlimiting joint moment arm: {worst:.3f} m")
print(f"one arm can hold      {ARM_TAU/(G*worst):5.2f} kg statically")
print(f"two arms (one robot)  {2*ARM_TAU/(G*worst):5.2f} kg")
print(f"four arms (a pair)    {4*ARM_TAU/(G*worst):5.2f} kg")
print()
print("current payloads: cube 0.5 kg, ball 0.7 kg, plank 1.1 kg")
print("NOTE: this is the *hanging* case -- load on the hand, gravity only.")
print("A friction pinch needs normal force too, which is where the budget goes.")
PY
