"""Quaternion order across the two Isaac Lab stacks. No Isaac Sim, no numpy."""

from __future__ import annotations

import ast
import math
import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
