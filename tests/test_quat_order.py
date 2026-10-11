"""Quaternion order across the two Isaac Lab stacks. No Isaac Sim, no numpy."""

from __future__ import annotations

import ast
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from bhl_robust.quat_order import WXYZ, XYZW, reorder  # noqa: E402

#: The depth rung's and the stereo pair's pose, as written: 20 deg of down-pitch.
PITCH_DOWN_20 = (0.9848, 0.0, 0.1736, 0.0)


def rotate_xyzw(q, v):
    """Rotate `v` by a quaternion stored (x, y, z, w), the way Isaac Lab 3.0 does."""
    x, y, z, w = q
    ux, uy, uz = v
    # t = 2 q_vec x v ; v' = v + w t + q_vec x t
    tx, ty, tz = 2 * (y * uz - z * uy), 2 * (z * ux - x * uz), 2 * (x * uy - y * ux)
    return (ux + w * tx + (y * tz - z * ty),
            uy + w * ty + (z * tx - x * tz),
            uz + w * tz + (x * ty - y * tx))


class ReorderTests(unittest.TestCase):
    def test_wxyz_is_identity(self):
        self.assertEqual(reorder(PITCH_DOWN_20, WXYZ), PITCH_DOWN_20)

    def test_xyzw_moves_w_last(self):
        self.assertEqual(reorder((1.0, 0.0, 0.0, 0.0), XYZW), (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(reorder(PITCH_DOWN_20, XYZW), (0.0, 0.1736, 0.0, 0.9848))

    def test_rejects_unknown_order(self):
        with self.assertRaises(ValueError):
            reorder(PITCH_DOWN_20, "zyxw")


class PitchTests(unittest.TestCase):
    """The bug itself, in numbers: world convention, forward is +x."""

    def pitch_deg(self, q_xyzw):
        fx, fy, fz = rotate_xyzw(q_xyzw, (1.0, 0.0, 0.0))
        return math.degrees(math.asin(max(-1.0, min(1.0, fz))))

    def test_reordered_literal_pitches_down(self):
        self.assertAlmostEqual(self.pitch_deg(reorder(PITCH_DOWN_20, XYZW)), -20.0, delta=0.1)

    def test_raw_literal_on_xyzw_pitches_up_and_flips(self):
        raw = PITCH_DOWN_20  # what v60 received before the fix
        self.assertAlmostEqual(self.pitch_deg(raw), +20.0, delta=0.1)
        ux, uy, uz = rotate_xyzw(raw, (0.0, 0.0, 1.0))
        self.assertLess(uz, -0.9, "camera up-axis should point down: image upside down")


class ClothProxySpawnTests(unittest.TestCase):
    """Execute the real proxy factory with Isaac config containers on CPU.

    Only the factory and its quaternion import are compiled: importing the
    complete task module registers simulator scenes. The conversion is the
    real native_quat; both storage orders are selected at its probe boundary.
    """

    @staticmethod
    def proxy_factory():
        path = _REPO / "src/bhl_robust/tasks/cloth_sort_env_cfg.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_proxy")
        quat_import = next(n for n in tree.body if isinstance(n, ast.ImportFrom)
                           and n.module == "bhl_robust.quat_order")
        body = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")],
                                              level=0), quat_import, factory], type_ignores=[])
        ast.fix_missing_locations(body)
        rigid_cfg = type("RigidObjectCfg", (SimpleNamespace,), {"InitialStateCfg": SimpleNamespace})
        spec = SimpleNamespace(proxy_size=(0.10, 0.08, 0.015), mass=0.016,
                               rgb=(0.75, 0.22, 0.18), friction=0.55)
        namespace = {
            "RigidObjectCfg": rigid_cfg,
            "GARMENT_BY_NAME": {"shirt_a": spec},
            "TABLE_TOP_Z": 0.70,
            "default_spawn_xy": lambda garment: (-0.28, 0.29),
            "parking_xy": lambda index: (1.0 + index, -0.5),
            "_RIGID": object(),
            "_COLLISION": object(),
            "sim_utils": SimpleNamespace(**{name: SimpleNamespace for name in (
                "CuboidCfg", "MassPropertiesCfg", "PreviewSurfaceCfg", "RigidBodyMaterialCfg")}),
        }
        exec(compile(body, str(path), "exec"), namespace)
        return namespace["_proxy"], namespace, spec

    def test_spawn_is_physical_identity_on_both_stacks(self):
        proxy, namespace, spec = self.proxy_factory()
        for order in (WXYZ, XYZW):
            for prim, xy, expected_xy in (
                ("garment_0", None, (-0.28, 0.29)),
                ("garment_3", None, (4.0, -0.5)),
                ("custom_proxy", (0.12, 0.24), (0.12, 0.24)),
            ):
                with self.subTest(order=order, prim=prim):
                    with patch("bhl_robust.quat_order.quat_order", return_value=order):
                        cfg = proxy("shirt_a", prim, xy)
                    # Read it the way the selected stack will: every local basis
                    # vector must retain its direction, especially the cloth normal.
                    q = cfg.init_state.rot
                    q_xyzw = (q[1], q[2], q[3], q[0]) if order == WXYZ else q
                    for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
                        self.assertEqual(rotate_xyzw(q_xyzw, axis), axis)
                    self.assertEqual(cfg.prim_path, f"{{ENV_REGEX_NS}}/{prim}")
                    self.assertEqual(cfg.init_state.pos[:2], expected_xy)
                    self.assertAlmostEqual(cfg.init_state.pos[2], 0.7075)
                    self.assertEqual(cfg.spawn.size, spec.proxy_size)
                    self.assertIs(cfg.spawn.rigid_props, namespace["_RIGID"])
                    self.assertIs(cfg.spawn.collision_props, namespace["_COLLISION"])
                    self.assertEqual(cfg.spawn.mass_props.mass, spec.mass)
                    self.assertEqual(cfg.spawn.visual_material.diffuse_color, spec.rgb)
                    self.assertEqual(cfg.spawn.physics_material.static_friction, spec.friction)
                    self.assertEqual(cfg.spawn.physics_material.dynamic_friction, 0.85 * spec.friction)
                    self.assertEqual(cfg.spawn.physics_material.restitution, 0.0)

    def test_original_literal_inverts_the_normal_on_xyzw(self):
        self.assertEqual(rotate_xyzw((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                         (0.0, 0.0, -1.0))


class CameraOffsetGuard(unittest.TestCase):
    """Every camera OffsetCfg in the package passes its rot through native_quat.

    A bare literal is correct on one stack and silently wrong on the other, so
    the guard is on the call shape rather than on the value.
    """

    def test_offset_rots_are_native(self):
        bad = []
        for path in sorted((_REPO / "src" / "bhl_robust").rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                if not name.endswith("OffsetCfg"):
                    continue
                for kw in node.keywords:
                    if kw.arg != "rot":
                        continue
                    ok = (isinstance(kw.value, ast.Call)
                          and getattr(kw.value.func, "id", "") == "native_quat")
                    if not ok:
                        bad.append(f"{path.relative_to(_REPO)}:{node.lineno}")
        self.assertEqual(bad, [], "OffsetCfg rot not wrapped in native_quat")


def _enclosing_functions(tree):
    """Map every node id to the name of the innermost def enclosing it ('<module>' if none)."""
    owner = {}

    def visit(node, name):
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else name
            owner[id(child)] = inner
            visit(child, inner)

    visit(tree, "<module>")
    return owner


def _is_native_quat_call(value) -> bool:
    return isinstance(value, ast.Call) and getattr(value.func, "id", "") == "native_quat"


#: Pose literals that are allowed NOT to go through `native_quat`, keyed by
#: (repo-relative path, enclosing function, `ast.unparse` of the rot value) --
#: never by line number, so an edit elsewhere in the file cannot move a hit onto
#: or off an entry. Each entry must match exactly one call (a stale entry fails).
INITIAL_STATE_ALLOWLIST = {
    # `_robot` converts by default: q = native_quat(rot) unless BHL_LEGACY_YAW=1,
    # which deliberately writes the raw 4-tuples every FINDINGS Isaac number
    # trained on (they bury the robot on v60; see the comment above `_robot`).
    ("src/bhl_robust/tasks/coop_lift_env_cfg.py", "_robot", "q"):
        "native_quat by default; raw only under the documented BHL_LEGACY_YAW=1",
    # Pass-through of `_box(rot=...)`, default (1, 0, 0, 0). Allowed ONLY while
    # no caller passes a rotation (`FurnitureBoxCallers` enforces that): on v60
    # the default is 180 deg about x, which maps a centred, uniformly coloured
    # cuboid onto itself, and a static AssetBaseCfg has no pose data anyone
    # reads. A caller that needs a real rotation must first change `_box` to
    # `rot=native_quat(rot)` (furniture.py is shared with cloth-sort and maze).
    ("src/bhl_robust/tasks/furniture.py", "_box", "rot"):
        "identity-only pass-through; see FurnitureBoxCallers",
}


def _rot_hits_in_tree(tree, rel: str):
    """Every rot in `tree` that reaches an InitialStateCfg other than through native_quat.

    Covers `...InitialStateCfg(rot=...)`, a positional rot (the second field on
    both stacks), `.replace(rot=...)` (how a config edits an existing
    init_state) and `<x>.init_state.rot = ...`. Returns (key, where) pairs,
    key = (rel, enclosing function, unparsed rot value).
    """
    hits = []
    owner = _enclosing_functions(tree)
    for node in ast.walk(tree):
        where = f"{rel}:{getattr(node, 'lineno', '?')}"
        fn = owner.get(id(node), "<module>")
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name.endswith("InitialStateCfg") and len(node.args) >= 2:
                hits.append(((rel, fn, "<positional rot>"), where))
            if name.endswith("InitialStateCfg") or name == "replace":
                for kw in node.keywords:
                    if kw.arg == "rot" and not _is_native_quat_call(kw.value):
                        hits.append(((rel, fn, ast.unparse(kw.value)), where))
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Attribute) and tgt.attr == "rot"
                        and isinstance(tgt.value, ast.Attribute) and tgt.value.attr == "init_state"
                        and not _is_native_quat_call(node.value)):
                    hits.append(((rel, fn, ast.unparse(node.value)), where))
    return hits


def _initial_state_rot_hits():
    """`_rot_hits_in_tree` over every module of the package (the configs live there)."""
    hits = []
    for path in sorted((_REPO / "src" / "bhl_robust").rglob("*.py")):
        rel = str(path.relative_to(_REPO))
        hits += _rot_hits_in_tree(ast.parse(path.read_text(), filename=str(path)), rel)
    return hits


class InitialStateGuard(unittest.TestCase):
    """Every asset InitialStateCfg rot in the package passes through native_quat.

    The 2026-09-28 bug: the coop-lift / TaskV2 / crew payload spawned with the
    raw literal rot=(1, 0, 0, 0), which Isaac Lab 3.0 reads as (x, y, z, w) =
    180 deg about x. Robots and cameras had been converted; the object never
    was. Same shape of guard as the OffsetCfg one: on the call, not the value.
    """

    def test_initial_state_rots_are_native(self):
        hits = _initial_state_rot_hits()
        bad = [f"{w}  rot={k[2]}  (in {k[1]})" for k, w in hits if k not in INITIAL_STATE_ALLOWLIST]
        self.assertEqual(bad, [], "InitialStateCfg rot not wrapped in native_quat")

    def test_allowlist_is_not_stale(self):
        counts = {k: 0 for k in INITIAL_STATE_ALLOWLIST}
        for k, _ in _initial_state_rot_hits():
            if k in counts:
                counts[k] += 1
        self.assertEqual({k: n for k, n in counts.items() if n != 1}, {},
                         "each allowlist entry must match exactly one call")

    def test_guard_sees_the_shapes_it_claims(self):
        src = ("def f():\n"
               "    a = RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1.0, 0.0, 0.0, 0.0))\n"
               "    b = AssetBaseCfg.InitialStateCfg((0, 0, 0), (1.0, 0.0, 0.0, 0.0))\n"
               "    c = cfg.init_state.replace(rot=(0.0, 0.0, 0.0, 1.0))\n"
               "    cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)\n"
               "    ok1 = RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=native_quat((1.0, 0.0, 0.0, 0.0)))\n"
               "    ok2 = cfg.init_state.replace(pos=(0, 0, 1))\n"
               "    ok3 = RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0))\n")
        hits = _rot_hits_in_tree(ast.parse(src), "x.py")
        self.assertEqual(sorted(w for _, w in hits), ["x.py:2", "x.py:3", "x.py:4", "x.py:5"])
        self.assertEqual({k[1] for k, _ in hits}, {"f"})
        self.assertIn(("x.py", "f", "<positional rot>"), [k for k, _ in hits])


class FurnitureBoxCallers(unittest.TestCase):
    """The furniture `_box` allowlist entry holds only while every box is unrotated."""

    def test_no_caller_passes_a_rotation(self):
        bad = []
        for path in sorted((_REPO / "src" / "bhl_robust").rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                is_box = ((isinstance(f, ast.Name) and f.id == "_box" and path.name == "furniture.py")
                          or (isinstance(f, ast.Attribute) and f.attr == "_box"
                              and getattr(f.value, "id", "") == "furniture"))
                if is_box and (len(node.args) >= 5 or any(kw.arg == "rot" for kw in node.keywords)):
                    bad.append(f"{path.relative_to(_REPO)}:{node.lineno}")
        self.assertEqual(bad, [], "furniture._box given a rotation: wrap it in native_quat inside _box first")

    def test_box_default_is_the_identity_literal(self):
        tree = ast.parse((_REPO / "src/bhl_robust/tasks/furniture.py").read_text())
        box = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_box")
        names = [a.arg for a in box.args.args]
        default = box.args.defaults[names.index("rot") - (len(names) - len(box.args.defaults))]
        self.assertEqual(ast.literal_eval(default), (1.0, 0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
