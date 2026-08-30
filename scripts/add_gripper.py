"""Generate a gripper-equipped copy of the robot URDF. `external/` stays pristine.

The hardware has two grippers; the shipped asset welds both hands shut
(`arm_left_hand_l` and `arm_hand_r` are `type="fixed"`), so no policy in this
project has ever closed a hand. See `docs/GRIPPER.md`.

This does not edit upstream. It reads the shipped URDF and writes a modified
copy into the workspace, which is the same arrangement `prepare_mjcf` and
`convert_convex_usd.py` already use -- upstream is a pinned submodule and has to
stay re-pinnable.

**One DoF per hand, finger closing against the palm.** The driver sends a single
scalar per gripper (`gripper_left_target` in [0, 1], mapped to a raw [0.2, 0.8],
0.2 open and 0.85 closed), so one actuated DoF per hand is the faithful model,
not two. A two-finger jaw would need a `mimic` joint, which the URDF importer
handles poorly and which would put a second, unactuated DoF in the observation
for no gain.

The existing hand link is the opposing jaw. That is also the grasp the hardware
performs: lay the open hand over the object, close, and let the closed finger
and palm form a manifold that retains it geometrically rather than by friction.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

#: Joint names follow the arm convention so `JOINTS` ordering stays predictable.
GRIPPER_JOINTS = ("arm_left_gripper_joint", "arm_right_gripper_joint")

#: Finger sits at the distal end of the hand. The hand's inertial origin is
#: 54 mm along -z from the wrist, and its mesh is 136 mm long on that axis, so
#: the fingertip region is about 110 mm out.
FINGER_ORIGIN_Z = -0.110
FINGER_SIZE = (0.020, 0.050, 0.070)
FINGER_MASS = 0.03

#: 0 rad open, closing inward. The driver's [0, 1] maps onto this range.
FINGER_RANGE = (0.0, 1.20)
FINGER_EFFORT = 2.0        # Nm; the arm joints are 4 and this is a smaller motor
FINGER_VELOCITY = 6.0


def _finger_link(name: str) -> ET.Element:
    link = ET.Element("link", {"name": name})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 -0.035", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": str(FINGER_MASS)})
    # Thin box about its own centroid; exact enough for a 30 g link and it keeps
    # the solver from seeing a degenerate inertia.
    ET.SubElement(inertial, "inertia", {
        "ixx": "2.0e-5", "ixy": "0", "ixz": "0",
        "iyy": "1.5e-5", "iyz": "0", "izz": "1.0e-5"})
    sx, sy, sz = FINGER_SIZE
    for tag in ("visual", "collision"):
        el = ET.SubElement(link, tag)
        ET.SubElement(el, "origin", {"xyz": "0 0 -0.035", "rpy": "0 0 0"})
        geom = ET.SubElement(el, "geometry")
        ET.SubElement(geom, "box", {"size": f"{sx} {sy} {sz}"})
        if tag == "visual":
            mat = ET.SubElement(el, "material", {"name": f"{name}_material"})
            ET.SubElement(mat, "color", {"rgba": "0.922 0.408 0.204 1"})
    return link


def _finger_joint(name: str, parent: str, child: str, sign: float) -> ET.Element:
    j = ET.Element("joint", {"name": name, "type": "revolute"})
    ET.SubElement(j, "origin", {"xyz": f"0 0 {FINGER_ORIGIN_Z}", "rpy": "0 0 0"})
    ET.SubElement(j, "parent", {"link": parent})
    ET.SubElement(j, "child", {"link": child})
    # Closing rotates about the hand's y, so the finger sweeps toward the palm.
    ET.SubElement(j, "axis", {"xyz": f"0 {sign:g} 0"})
    ET.SubElement(j, "limit", {
        "lower": str(FINGER_RANGE[0]), "upper": str(FINGER_RANGE[1]),
        "effort": str(FINGER_EFFORT), "velocity": str(FINGER_VELOCITY)})
    return j


def add_grippers(src: Path, dst: Path) -> list[str]:
    tree = ET.parse(src)
    root = tree.getroot()
    existing = {j.get("name") for j in root.findall("joint")}
    added = []
    for side, sign in (("left", +1.0), ("right", -1.0)):
        jname = f"arm_{side}_gripper_joint"
        if jname in existing:
            continue
        hand = f"arm_{side}_hand_link"
        if root.find(f".//link[@name='{hand}']") is None:
            raise SystemExit(f"{hand} not found in {src}")
        finger = f"arm_{side}_finger_link"
        root.append(_finger_link(finger))
        root.append(_finger_joint(jname, hand, finger, sign))
        added.append(jname)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return added


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    added = add_grippers(args.src, args.out)
    tree = ET.parse(args.out)
    joints = [j.get("name") for j in tree.getroot().findall("joint")
              if j.get("type") != "fixed"]
    print(f"wrote {args.out}")
    print(f"  added: {added or '(already present)'}")
    print(f"  actuated joints now: {len(joints)}")
    for j in joints:
        if "gripper" in j:
            print(f"    {j}")


if __name__ == "__main__":
    main()
