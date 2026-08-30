#!/bin/bash
# Convert the gripper-equipped URDF to USD, then load it and report the joints.
#
# Isaac spawns from USD, not URDF, so adding joints to the URDF is only half the
# job. Isaac Lab ships a UrdfConverter for exactly this. Output lands in the
# workspace; external/ stays pristine and re-pinnable.
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args([]); a.headless = True
app = AppLauncher(a)

from pathlib import Path
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

SRC = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/assets/gripper/berkeley_humanoid_lite_gripper.urdf")
OUT = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/assets/gripper/usd")
OUT.mkdir(parents=True, exist_ok=True)

cfg = UrdfConverterCfg(
    asset_path=str(SRC),
    usd_dir=str(OUT),
    usd_file_name="berkeley_humanoid_lite_gripper.usd",
    fix_base=False,
    merge_fixed_joints=False,   # keep the hand link addressable
    force_usd_conversion=True,
)
conv = UrdfConverter(cfg)
print("usd ->", conv.usd_path)

# Load it back and list what actually made it through the conversion.
from pxr import Usd, UsdPhysics
stage = Usd.Stage.Open(conv.usd_path)
joints = [pr for pr in stage.Traverse() if pr.IsA(UsdPhysics.RevoluteJoint)]
names = sorted(pr.GetName() for pr in joints)
print(f"revolute joints in the USD: {len(names)}")
for n in names:
    if "gripper" in n or "finger" in n:
        print("   ", n)
app.app.close()
PY
