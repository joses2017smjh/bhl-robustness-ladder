#!/usr/bin/env python3
"""Write the Waiter program's robot URDF (docs/WAITER_PROGRAM.md, phase 1). `external/` stays pristine.

= upstream `berkeley_humanoid_lite.urdf` + the two 1-DoF grippers of `scripts/add_gripper.py` (unchanged geometry)
+ three declared changes:
  1. a <collision> on each hand link: the hand's own visual mesh with the visual's origin (Isaac's URDF converter
     and MuJoCo both collide with its convex hull), so a handle can be held between finger and palm (the gripper
     asset had a finger collider only);
  2. arm joint effort limits 8 Nm (WAITER_ARM_EFFORT; upstream URDF says 20, the upstream training cap is 4);
  3. mesh paths made absolute (the upstream `package://../meshes/...` only resolves next to the upstream URDF).
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = (REPO / "external/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_assets/data/robots/"
                "berkeley_humanoid/berkeley_humanoid_lite")
UPSTREAM_URDF = UPSTREAM_DIR / "urdf/berkeley_humanoid_lite.urdf"
DEFAULT_OUT = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/assets/waiter/berkeley_humanoid_lite_waiter.urdf")
WAITER_ARM_EFFORT = 8.0


def _add_gripper_module():
    spec = importlib.util.spec_from_file_location("add_gripper", REPO / "scripts/add_gripper.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def patch(root: ET.Element, meshes_dir: Path) -> dict:
    report = {"palm_collisions": [], "arm_efforts": [], "meshes_rewritten": 0}
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        if "meshes/" in fn:
            mesh.set("filename", str(meshes_dir / fn.split("meshes/")[-1]))
            report["meshes_rewritten"] += 1
    for side in ("left", "right"):
        link = root.find(f"./link[@name='arm_{side}_hand_link']")
        if link is None:
            raise SystemExit(f"arm_{side}_hand_link not found")
        if link.find("collision") is not None:
            raise SystemExit(f"arm_{side}_hand_link already has a collision")
        vis = link.find("visual")
        col = ET.SubElement(link, "collision")
        col.append(copy.deepcopy(vis.find("origin")))
        col.append(copy.deepcopy(vis.find("geometry")))
        report["palm_collisions"].append(f"arm_{side}_hand_link")
    for j in root.findall("joint"):
        name = j.get("name", "")
        if name.startswith("arm_") and "gripper" not in name and j.get("type") == "revolute":
            j.find("limit").set("effort", f"{WAITER_ARM_EFFORT:g}")
            report["arm_efforts"].append(name)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".gripper-only.urdf")
    added = _add_gripper_module().add_grippers(UPSTREAM_URDF, tmp)
    tree = ET.parse(tmp)
    rep = patch(tree.getroot(), UPSTREAM_DIR / "meshes")
    tree.write(a.out, encoding="utf-8", xml_declaration=True)
    tmp.unlink()
    joints = [j.get("name") for j in tree.getroot().findall("joint") if j.get("type") != "fixed"]
    assert len(joints) == 24 and len(rep["arm_efforts"]) == 10 and len(rep["palm_collisions"]) == 2, (joints, rep)
    print(f"wrote {a.out}: grippers added {added}; actuated joints {len(joints)}; palm collisions "
          f"{rep['palm_collisions']}; arm effort {WAITER_ARM_EFFORT:g} Nm on {len(rep['arm_efforts'])} joints; "
          f"{rep['meshes_rewritten']} mesh paths made absolute")


if __name__ == "__main__":
    main()
