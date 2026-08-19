"""Re-convert the biped URDF with convex-decomposition collision from visuals.

Upstream's convert_urdf_to_usd.py keeps the primitive collision geoms from the
URDF (`collision_from_visuals=False`). This is the other setting: take the
visual meshes, convex-decompose them, and use those as the colliding geometry.

The output is written outside `external/` so the submodule stays unmodified.
Isaac Sim's shutdown can hard-exit before Python flushes stdout, so the
success line is appended to a file before anything is torn down.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import argparse

from isaaclab.app import AppLauncher

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_UPSTREAM = Path(os.environ.get(
    "UPSTREAM", _REPO / "external" / "Berkeley-Humanoid-Lite",
))
_OUT_DIR = Path(os.environ.get(
    "BHL_CONVEX_USD_DIR",
    "/nfs/hpc/share/sanchej7/Humanoid_Lite/assets/bhl_biped_convex",
))
_MARKER = Path(os.environ.get("BENCH_OUT", "/tmp/convex_usd.txt"))

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

urdf = (
    _UPSTREAM / "source/berkeley_humanoid_lite_assets"
    / "data/robots/berkeley_humanoid/berkeley_humanoid_lite"
    / "urdf/berkeley_humanoid_lite_biped.urdf"
)
if not urdf.is_file():
    raise FileNotFoundError(urdf)

_OUT_DIR.mkdir(parents=True, exist_ok=True)
cfg = UrdfConverterCfg(
    asset_path=str(urdf),
    usd_dir=str(_OUT_DIR),
    usd_file_name="berkeley_humanoid_lite_biped.usd",
    fix_base=False,
    merge_fixed_joints=False,
    force_usd_conversion=True,
    collision_from_visuals=True,
    collider_type="convex_decomposition",
    self_collision=False,
    replace_cylinders_with_capsules=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=100.0, damping=1.0,
        ),
        target_type="position",
    ),
)
print("URDF importer config:", flush=True)
print_dict(cfg.to_dict(), nesting=0)
conv = UrdfConverter(cfg)
line = f"CONVEX USD OK | {conv.usd_path}"
_MARKER.parent.mkdir(parents=True, exist_ok=True)
with open(_MARKER, "a") as f:
    f.write(line + "\n")
print(line, flush=True)
simulation_app.close()
sys.exit(0)
