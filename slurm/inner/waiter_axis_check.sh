#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
import argparse, json
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); a = p.parse_args([]); a.headless = True
app = AppLauncher(a)
import torch, numpy as np
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext
from berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite import HUMANOID_LITE_CFG
from bhl_robust import waiter_asset as W
sim = SimulationContext(SimulationCfg(dt=0.005, device="cuda:0", gravity=(0.0, 0.0, 0.0)))
sh = Articulation(HUMANOID_LITE_CFG.replace(prim_path="/World/sh", init_state=HUMANOID_LITE_CFG.init_state.replace(pos=(0.0, 0.0, 1.0))))
wc = W.make_waiter_cfg(); wa = Articulation(wc.replace(prim_path="/World/wa", init_state=wc.init_state.replace(pos=(0.0, 3.0, 1.0))))
sim.reset()
rng = np.random.default_rng(0)
qn = {n: float(v) for n, v in zip(W.JOINT_ORDER, rng.uniform(-0.3, 0.3, 24))}
for n in ("leg_left_knee_pitch_joint", "leg_right_knee_pitch_joint"): qn[n] = 0.6
for n in ("arm_left_elbow_pitch_joint",): qn[n] = 0.5
for n in ("arm_right_elbow_pitch_joint",): qn[n] = -0.5
for n in ("arm_left_shoulder_roll_joint",): qn[n] = 0.2
for n in ("arm_right_shoulder_roll_joint",): qn[n] = -0.2
def setq(r):
    q = torch.zeros_like(r.data.default_joint_pos)
    for i, n in enumerate(r.joint_names):
        q[0, i] = qn.get(n, 0.0)
    r.write_joint_state_to_sim(q, torch.zeros_like(q))
setq(sh); setq(wa); sim.step(); sh.update(0.005); wa.update(0.005)
out = {}
for r, off, tag in ((sh, 0.0, "shipped"), (wa, 3.0, "waiter")):
    pos = r.data.body_pos_w[0].cpu().numpy().copy(); pos[:, 1] -= off
    out[tag] = {n: pos[i].round(4).tolist() for i, n in enumerate(r.body_names)}
diffs = {n: float(np.abs(np.array(out["shipped"][n]) - np.array(out["waiter"][n])).max()) for n in out["shipped"] if n in out["waiter"]}
bad = {n: round(d, 4) for n, d in diffs.items() if d > 0.002}
print("AXIS CHECK:", "PASS (all shared bodies within 2 mm at a random pose)" if not bad else f"FAIL {bad}")
print(json.dumps({k: out["shipped"][k] for k in ("leg_left_ankle_roll", "arm_left_hand_link")}), json.dumps({k: out["waiter"][k] for k in ("leg_left_ankle_roll", "arm_left_hand_link")}))
app.close()
PY
