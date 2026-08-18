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


def prepare_mjcf(upstream: Path, cache_dir: Path, variant: str = "biped") -> Path:
    """Materialise a loadable copy of the MJCF and return the scene path.

    Args:
        upstream: path to the Berkeley-Humanoid-Lite checkout.
        cache_dir: writable directory for the patched copy.
        variant: "biped" (12 DoF) or "humanoid" (22 DoF).

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

    out_dir = cache_dir / f"mjcf_{variant}"
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
