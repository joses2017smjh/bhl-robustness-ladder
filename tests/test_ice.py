"""Ice patches have to fit on the tile the robot actually walks."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from bhl_robust.terrains.ice_layout import (  # noqa: E402
    TILE_SIZE,
    ice_local_offsets,
    patch_fits_tile,
)


class PackingTests(unittest.TestCase):
    def test_six_patches_fit_the_tile(self):
        offs = ice_local_offsets()
        self.assertEqual(len(offs), 6)
        for pos in offs:
            self.assertTrue(patch_fits_tile(pos), pos)

    def test_alternating_y_and_spread_x(self):
        xs = [p[0] for p in ice_local_offsets()]
        ys = [p[1] for p in ice_local_offsets()]
        self.assertEqual(len(set(xs)), 6)
        self.assertEqual({y > 0 for y in ys[0::2]}, {False})
        self.assertEqual({y > 0 for y in ys[1::2]}, {True})

    def test_tile_size_matches_the_bumpy_generator(self):
        """A silent 8 vs 10 would pack the patches onto the wrong square."""
        src = (_REPO / "src" / "bhl_robust" / "terrains" / "bumpy.py").read_text()
        tree = ast.parse(src)
        found = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(getattr(t, "id", None) == "BUMPY_TERRAINS_CFG" for t in node.targets):
                continue
            for kw in node.value.keywords:
                if kw.arg == "size":
                    found = (kw.value.elts[0].value, kw.value.elts[1].value)
        self.assertEqual(found, (TILE_SIZE, TILE_SIZE), found)


if __name__ == "__main__":
    unittest.main()
