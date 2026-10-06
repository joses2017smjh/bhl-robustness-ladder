#!/bin/bash
# Waiter phase 1 asset (docs/WAITER_PROGRAM.md): URDF (scripts/waiter/make_waiter_urdf.py) -> USD with Isaac Lab's
# UrdfConverter on this stack, then a geometry check against the shipped 22-DoF USD before anything trains on it.
# CHECK (all must hold; prints WAITER ASSET CHECK: PASS|FAIL):
#   joints: the 22 upstream names + 2 grippers, revolute; palm collision prims under both hand links;
#   upright at identity spawn: torso-above-ankles within 0.01 m of the shipped USD's (an earlier conversion came out
#   lying down: SLURM_JOBS.md "The spawn quaternion was a roll"); hands symmetric at zero arm angles (|y_L + y_R|
#   and |x_L - x_R|, |z_L - z_R| < 0.005 m) and within 0.01 m of the shipped USD's hand positions; total mass within
#   0.2 kg of shipped + 2 x 0.03 kg fingers.
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" scripts/waiter/make_waiter_urdf.py
"$PY" - <<'PY'
import argparse, json, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args([]); a.headless = True
app = AppLauncher(a)

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from bhl_robust import waiter_asset as W
cfg = UrdfConverterCfg(asset_path=W.WAITER_URDF, usd_dir=W.WAITER_USD_DIR,
                       usd_file_name="berkeley_humanoid_lite_waiter.usd", fix_base=False,
                       merge_fixed_joints=False, force_usd_conversion=True,
                       # gains are set by the task's actuator cfg at spawn (make_waiter_cfg); 0 here, as required
                       joint_drive=UrdfConverterCfg.JointDriveCfg(
                           gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)))
conv = UrdfConverter(cfg)
usd = W.find_waiter_usd()
print("usd ->", conv.usd_path, "| resolved", usd, flush=True)

from pxr import Usd, UsdPhysics
stage = Usd.Stage.Open(usd)
revolute = sorted(pr.GetName() for pr in stage.Traverse() if pr.IsA(UsdPhysics.RevoluteJoint))
palm_cols = sorted(str(pr.GetPath()) for pr in stage.Traverse(Usd.TraverseInstanceProxies())
                   if pr.HasAPI(UsdPhysics.CollisionAPI) and "hand_link" in str(pr.GetPath()))
print("revolute joints:", len(revolute), "| palm collision prims:", palm_cols, flush=True)

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext
from berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite import HUMANOID_LITE_CFG
sim = SimulationContext(SimulationCfg(dt=0.005, device="cuda:0", gravity=(0.0, 0.0, 0.0)))
shipped = Articulation(HUMANOID_LITE_CFG.replace(prim_path="/World/shipped",
                       init_state=HUMANOID_LITE_CFG.init_state.replace(pos=(0.0, 0.0, 1.0))))
wcfg = W.make_waiter_cfg(usd)
waiter = Articulation(wcfg.replace(prim_path="/World/waiter", init_state=wcfg.init_state.replace(pos=(0.0, 3.0, 1.0))))
sim.reset()
for r in (shipped, waiter):
    r.write_joint_state_to_sim(torch.zeros_like(r.data.default_joint_pos), torch.zeros_like(r.data.default_joint_vel))
sim.step(); shipped.update(0.005); waiter.update(0.005)

def geo(r, off_y):
    names = r.body_names
    pos = r.data.body_pos_w[0].cpu().numpy().copy(); pos[:, 1] -= off_y
    b = lambda n: pos[names.index(n)]
    ank = 0.5 * (b("leg_left_ankle_roll") + b("leg_right_ankle_roll"))
    return {"torso_above_ankles": float(b("base")[2] - ank[2]), "hand_l": b("arm_left_hand_link").tolist(),
            "hand_r": b("arm_right_hand_link").tolist(),
            "mass": float(r.root_physx_view.get_masses()[0].sum())}
gs, gw = geo(shipped, 0.0), geo(waiter, 3.0)
hl, hr = gw["hand_l"], gw["hand_r"]
checks = {
    "joints_24": len(waiter.joint_names) == 24 and set(waiter.joint_names) == set(W.JOINT_ORDER),
    "usd_revolute_24": len(revolute) == 24,
    "palm_collisions_2": sum(("arm_left_hand_link" in c) for c in palm_cols) >= 1
                         and sum(("arm_right_hand_link" in c) for c in palm_cols) >= 1,
    "upright_like_shipped": abs(gw["torso_above_ankles"] - gs["torso_above_ankles"]) < 0.01,
    "hands_symmetric": abs(hl[1] + hr[1]) < 0.005 and abs(hl[0] - hr[0]) < 0.005 and abs(hl[2] - hr[2]) < 0.005,
    "hands_match_shipped": max(abs(x - y) for x, y in zip(hl + hr, gs["hand_l"] + gs["hand_r"])) < 0.01,
    "mass_plausible": abs(gw["mass"] - (gs["mass"] + 0.06)) < 0.2,
}
out = {"usd": usd, "revolute": revolute, "palm_collision_prims": palm_cols, "joint_names": list(waiter.joint_names),
       "shipped": gs, "waiter": gw, "checks": checks,
       "actuators": {k: {"effort": getattr(v, "effort_limit", None), "stiffness": getattr(v, "stiffness", None)}
                     for k, v in wcfg.actuators.items()}}
os.makedirs(os.environ["WAITER_RES"], exist_ok=True)
with open(os.path.join(os.environ["WAITER_RES"], "asset_check.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({"shipped": gs, "waiter": gw, "checks": checks}, indent=1), flush=True)
print("WAITER ASSET CHECK:", "PASS" if all(checks.values()) else "FAIL " + str([k for k, v in checks.items() if not v]),
      flush=True)
app.close()
PY
