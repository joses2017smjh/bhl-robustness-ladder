#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Spawn the robot under each candidate quaternion and photograph it.

Every numeric probe in this investigation has disagreed with the render at some
point, and the render has been right every time. So this removes everything
between the two: no policy, no checkpoint, no training config, no reset events.
Spawn the articulation on a ground plane, take a picture, write a PNG.

If a candidate stands the robot on the floor, it will be visible in its own
image, and no interpretation is required.
"""
import argparse, os
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args([]); a.headless = True; a.enable_cameras = True
app = AppLauncher(a)

import numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite import HUMANOID_LITE_CFG
from bhl_robust.tasks import coop_lift_env_cfg as C

OUT = os.environ.get("SHOT_DIR", f"{os.environ['REPO']}/results/spawn_shots")
os.makedirs(OUT, exist_ok=True)

CANDS = {
    "y180": (0.0, 0.0, 1.0, 0.0),
    "y180_then_z90": (0.0, 0.70710678, 0.70710678, 0.0),
    "z90_then_y180": (0.0, -0.70710678, 0.70710678, 0.0),
    "y180_then_z180": (0.0, 1.0, 0.0, 0.0),
    "z180_then_y180": (0.0, -1.0, 0.0, 0.0),
    "y180_then_z270": (0.0, 0.70710678, -0.70710678, 0.0),
    "z270_then_y180": (0.0, -0.70710678, -0.70710678, 0.0),
}

sim = SimulationContext(SimulationCfg(dt=0.005, device="cuda:0"))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))

arts, cams = {}, {}
for i, (name, rot) in enumerate(CANDS.items()):
    cfg = HUMANOID_LITE_CFG.replace(prim_path=f"/World/r{i}")
    cfg.init_state = cfg.init_state.replace(
        pos=(i * 4.0, 0.0, C._PINCH_ROOT_Z), rot=rot,
        joint_pos=dict(C._PINCH_JOINT_POS))
    arts[name] = Articulation(cfg)
    cams[name] = Camera(CameraCfg(
        prim_path=f"/World/cam{i}", height=360, width=480,
        data_types=["rgb"],
        offset=CameraCfg.OffsetCfg(pos=(i * 4.0 + 1.4, 1.4, 0.7),
                                   rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0),
    ))
sim.reset()

# Force the joint pose through the sim, then settle briefly so the picture shows
# what training would actually see rather than a pre-physics placement.
for name, art in arts.items():
    jp = art.data.default_joint_pos.clone()
    names = art.joint_names
    for j, v in C._PINCH_JOINT_POS.items():
        if j in names:
            jp[:, names.index(j)] = v
    art.write_joint_state_to_sim(jp, torch.zeros_like(jp))
for _ in range(3):
    sim.step()
for i, (name, c) in enumerate(cams.items()):
    c.set_world_poses_from_view(
        torch.tensor([[i * 4.0 + 1.4, 1.4, 0.7]], device=sim.device),
        torch.tensor([[i * 4.0, 0.0, 0.3]], device=sim.device))
    c.update(dt=0.0)

try:
    from PIL import Image
except ImportError:
    Image = None

i_of = {n: k for k, n in enumerate(CANDS)}
print(f"{'candidate':12} {'ankle_z':>9} {'shoulder_z':>11} {'below':>7}  drift from spawn / image")
for name, art in arts.items():
    art.update(dt=0.0)
    nm = art.body_names
    b = art.data.body_pos_w; b = b.torch if hasattr(b, "torch") else b
    h = lambda k: float(b[0, [i for i, n in enumerate(nm) if k in n], 2].mean())
    below = int((b[0, :, 2] < 0).sum())
    path = f"{OUT}/{name}.png"
    rgb = cams[name].data.output["rgb"]
    rgb = rgb.torch if hasattr(rgb, "torch") else rgb
    arr = rgb[0, ..., :3].detach().cpu().numpy().astype(np.uint8)
    if Image is not None:
        Image.fromarray(arr).save(path)
    rp = art.data.root_pos_w; rp = rp.torch if hasattr(rp, "torch") else rp
    dx = float(rp[0, 0]) - i_of[name] * 4.0
    print(f"{name:12} {h('ankle_roll'):9.3f} {h('shoulder_pitch'):11.3f} "
          f"{below:3d}/{len(nm):<3d}  root drift ({dx:+.2f}, {float(rp[0,1]):+.2f}, "
          f"{float(rp[0,2]):+.2f})  {path}")
print("\nMuJoCo reference: ankle +0.140, shoulder +0.737, 1 of 26 below.")
app.close()
PY
