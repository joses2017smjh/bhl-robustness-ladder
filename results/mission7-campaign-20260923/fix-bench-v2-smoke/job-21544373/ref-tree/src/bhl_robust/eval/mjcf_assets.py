"""Repair the upstream MJCF asset paths.

The shipped MJCF declares `meshdir="assets"` and references meshes as
`merged/<name>.stl`, i.e. it expects `<mjcf_dir>/assets/merged/<name>.stl`.
The assets submodule actually stores them flat in `<robot_dir>/meshes/<name>.stl`,
so `MjModel.from_xml_path` dies on the first mesh.

The meshes are visual-only (`contype="0" conaffinity="0" group="2"`); every
collision geom is a primitive box/cylinder. Physics is therefore unaffected by
this bug -- but the model will not load at all, so it still has to be fixed.

Rather than mutating `external/`, which must stay re-pinnable, a patched copy of
the XML pair is materialised into a cache directory. Anyone cloning this repo
gets the same repair automatically.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

# Upstream layout, relative to the assets submodule root.
_ROBOT_SUBDIR = Path("data/robots/berkeley_humanoid/berkeley_humanoid_lite")

_VARIANTS = {
    "biped": ("bhl_biped_scene.xml", "berkeley_humanoid_lite_biped.xml"),
    "humanoid": ("bhl_scene.xml", "berkeley_humanoid_lite.xml"),
}


# MuJoCo's offscreen framebuffer defaults to 640x480 and rendering above that
# raises rather than downscaling. Video is written at 960x540, so the scene's
# <global> clause is widened when the patched copy is materialised.
OFFSCREEN_W, OFFSCREEN_H = 1920, 1080

# Height-field asset injected when evaluating on rough ground. Dimensions must
# agree with bhl_robust.eval.terrain_field (HALF_EXTENT, ELEVATION_M, GRID).
HFIELD_NAME = "bhl_terrain"

# Egocentric depth camera, injected into the base body on request. Pose and
# field of view are matched to the Isaac Lab ray-cast camera in
# bhl_robust.tasks.depth_env_cfg so the two depth paths see the same thing:
# 12 cm forward, 30 cm up, pitched 20 deg down, 60.5 deg vertical FOV.
#
# MuJoCo cameras look down their own -Z with +Y up, so the rotation is given as
# `xyaxes` (camera +X then camera +Y) rather than a quaternion: +X is the
# robot's -Y (image right), +Y is world up tilted back by 20 deg.
EGO_CAM_NAME = "ego_depth"
EGO_CAM = (
    f'<camera name="{EGO_CAM_NAME}" mode="fixed" pos="0.12 0 0.30" '
    'xyaxes="0 -1 0  0.342 0 0.940" fovy="60.5"/>'
)


#: Finger geometry, matching `scripts/add_gripper.py` so the MuJoCo replay and
#: the Isaac asset describe the same hand. Changing one without the other makes
#: the sim2sim comparison a comparison of two robots.
GRIPPER_PIVOT_Z = -0.110
GRIPPER_FINGER = (0.020, 0.050, 0.070)
GRIPPER_RANGE = (0.0, 1.20)
GRIPPER_EFFORT = 2.0


def _gripper_body(side: str, sign: float) -> str:
    """One finger, hinged on the hand, closing across the palm."""
    sx, sy, sz = (v / 2.0 for v in GRIPPER_FINGER)
    return (
        f'<body name="arm_{side}_finger_link" pos="0 0 {GRIPPER_PIVOT_Z}">'
        f'<joint name="arm_{side}_gripper_joint" type="hinge" axis="0 {sign:g} 0" '
        f'range="{GRIPPER_RANGE[0]} {GRIPPER_RANGE[1]}"/>'
        f'<inertial pos="0 0 -0.035" mass="0.03" '
        f'diaginertia="2.0e-5 1.5e-5 1.0e-5"/>'
        f'<geom name="arm_{side}_finger_geom" type="box" '
        f'size="{sx} {sy} {sz}" pos="0 0 -0.035" rgba="0.922 0.408 0.204 1"/>'
        f'</body>'
    )


def prepare_mjcf(upstream: Path, cache_dir: Path, variant: str = "biped",
                 terrain: bool = False, ego_camera: bool = False,
                 gripper: bool = False) -> Path:
    """Materialise a loadable copy of the MJCF and return the scene path.

    Args:
        upstream: path to the Berkeley-Humanoid-Lite checkout.
        cache_dir: writable directory for the patched copy.
        variant: "biped" (12 DoF) or "humanoid" (22 DoF).
        terrain: replace the flat floor plane with a height field. The elevation
            data itself is written into `model.hfield_data` at load time, so one
            patched XML serves every difficulty.
        ego_camera: add a base-mounted depth camera (`EGO_CAM_NAME`). Off by
            default -- an extra camera changes nothing physical, but the scored
            runs and the depth clips should not share a cache directory.
        gripper: add one hinged finger per hand, giving the 24-DoF layout the
            hardware actually has. The shipped MJCF welds both hands, so
            without this the replay harness cannot load a 24-DoF checkpoint at
            all -- which is why the gripper arms, the largest effect in the
            project, had no clips. Geometry mirrors `scripts/add_gripper.py`;
            the two must agree or trainer and judge describe different robots.

    Returns:
        Path to the patched scene XML, ready for `MjModel.from_xml_path`.
    """
    if variant not in _VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(_VARIANTS)}")

    scene_name, robot_name = _VARIANTS[variant]
    robot_dir = upstream / "source/berkeley_humanoid_lite_assets" / _ROBOT_SUBDIR
    mjcf_dir = robot_dir / "mjcf"
    mesh_dir = robot_dir / "meshes"

    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"mesh directory missing: {mesh_dir}")

    out_dir = cache_dir / (f"mjcf_{variant}_hfield" if terrain else f"mjcf_{variant}")
    if ego_camera:
        out_dir = out_dir.with_name(out_dir.name + "_ego")
    if gripper:
        out_dir = out_dir.with_name(out_dir.name + "_grip")
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_out = out_dir / scene_name
    robot_out = out_dir / robot_name

    # Scene file only <include>s the robot, but its <global> clause caps the
    # offscreen framebuffer, so it needs one edit rather than a plain copy.
    scene_xml = (mjcf_dir / scene_name).read_text()
    if "offwidth" not in scene_xml:
        scene_xml, n = re.subn(
            r"(<global\b[^/>]*)(/?>)",
            rf'\1 offwidth="{OFFSCREEN_W}" offheight="{OFFSCREEN_H}"\2',
            scene_xml,
            count=1,
        )
        if n == 0:
            # No <global> to extend; add a <visual> block of our own.
            scene_xml = re.sub(
                r"(<mujoco[^>]*>)",
                rf'\1\n  <visual><global offwidth="{OFFSCREEN_W}" offheight="{OFFSCREEN_H}"/></visual>',
                scene_xml,
                count=1,
            )
    if terrain:
        from bhl_robust.eval.terrain_field import GRID, HALF_EXTENT, ELEVATION_M
        # A geom of type "hfield" replaces the infinite plane. `size` is
        # (x_radius, y_radius, z_elevation, z_base); z_base only has to be thick
        # enough that the solid body beneath the surface is never penetrated.
        hf = (f'<hfield name="{HFIELD_NAME}" nrow="{GRID}" ncol="{GRID}" '
              f'size="{HALF_EXTENT} {HALF_EXTENT} {ELEVATION_M} 0.5"/>')
        scene_xml = re.sub(r"(<asset>)", r"\1\n      " + hf, scene_xml, count=1)
        scene_xml = re.sub(
            r'<geom name="floor"[^/]*/>',
            f'<geom name="floor" type="hfield" hfield="{HFIELD_NAME}" pos="0 0 0" '
            f'material="groundplane"/>',
            scene_xml, count=1)
        if HFIELD_NAME not in scene_xml:
            raise RuntimeError("failed to inject the height field into the scene XML")

    scene_out.write_text(scene_xml)

    xml = (mjcf_dir / robot_name).read_text()

    # Point meshdir at the real, absolute mesh location...
    xml, n_dir = re.subn(
        r'meshdir="[^"]*"',
        f'meshdir="{mesh_dir.resolve()}"',
        xml,
    )
    # ...and flatten the phantom `merged/` prefix.
    xml, n_mesh = re.subn(r'file="merged/', 'file="', xml)

    if ego_camera:
        xml, n_cam = re.subn(
            r'(<body name="base"[^>]*>)',
            r"\1\n      " + EGO_CAM,
            xml,
            count=1,
        )
        if n_cam == 0:
            raise RuntimeError(f'no <body name="base"> in {robot_name}; upstream layout changed')

    if gripper:
        # A finger body inside each hand, plus a motor for it. The hand bodies
        # are self-closing in the shipped XML (the fixed joint is a comment, not
        # an element), so the finger is inserted just before each hand body's
        # closing tag rather than appended after it.
        for side, sign in (("left", +1.0), ("right", -1.0)):
            tag = f'<body name="arm_{side}_hand_link"'
            i = xml.find(tag)
            if i < 0:
                raise RuntimeError(
                    f'no <body name="arm_{side}_hand_link"> in {robot_name}; '
                    "the gripper patch assumes upstream's hand layout"
                )
            depth, j = 0, i
            while j < len(xml):
                if xml.startswith("<body", j):
                    depth += 1
                elif xml.startswith("</body>", j):
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if depth != 0:
                raise RuntimeError(f"unbalanced <body> around arm_{side}_hand_link")
            xml = xml[:j] + _gripper_body(side, sign) + xml[j:]

        motors = "".join(
            f'<motor class="berkeley-humanoid-lite" name="arm_{s}_gripper_joint" '
            f'joint="arm_{s}_gripper_joint" '
            f'forcerange="-{GRIPPER_EFFORT} {GRIPPER_EFFORT}"/>'
            for s in ("left", "right")
        )
        xml, n_act = re.subn(r"(</actuator>)", motors + r"\1", xml, count=1)
        if n_act == 0:
            raise RuntimeError(f"no </actuator> in {robot_name}; cannot add gripper motors")

    if n_dir == 0:
        raise RuntimeError(f"no meshdir attribute found in {robot_name}; upstream layout changed")

    robot_out.write_text(xml)

    # Fail loudly and specifically rather than letting MuJoCo report only the
    # first missing file.
    referenced = set(re.findall(r'<mesh file="([^"]+)"', xml))
    missing = sorted(m for m in referenced if not (mesh_dir / m).is_file())
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} mesh(es) referenced by {robot_name} are absent from {mesh_dir}: "
            + ", ".join(missing[:5])
            + ("..." if len(missing) > 5 else "")
        )

    return scene_out
