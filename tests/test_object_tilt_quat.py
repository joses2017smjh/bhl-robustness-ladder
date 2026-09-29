"""The 2026-09-28 payload-quaternion fix, on stored tensors. No Isaac Sim.

Two bugs, fixed together (SLURM_JOBS.md 2026-09-28 re-verification entry):

* the coop-lift / TaskV2 / crew payload spawned with the raw literal
  rot=(1, 0, 0, 0), which Isaac Lab 3.0 (v60) stores (x, y, z, w) -- a 180 deg
  turn about x -- and now goes through `native_quat`;
* `coop_lift_mdp.object_tilt_l2` indexed `q[:, 1]`, `q[:, 2]` as x, y, which
  on v60 are y, z; it now reads through `unpack_wxyz`.

At rest they cancelled, so fixing one alone is worse than neither (the test
`test_fixing_one_alone_is_worse` pins that). v51 (w, x, y, z) must not move:
`native_quat` is the identity there and the reward reads the same two columns.

`coop_lift_mdp` imports three Isaac Lab names at module level; they are stubbed
inside `mock.patch.dict(sys.modules)` for the load only, so nothing leaks into
other tests. Every v60 run after this fix is a NEW configuration, never
compared with or used to re-score Stand / Stand2 / Stand3 / CoopLift results.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bhl_robust import quat_order as qo  # noqa: E402
from bhl_robust.quat_order import WXYZ, XYZW, native_quat, reorder  # noqa: E402

COOP_MDP = REPO / "src/bhl_robust/tasks/coop_lift_mdp.py"
LIFT_CFG = REPO / "src/bhl_robust/tasks/coop_lift_env_cfg.py"


_STUBS: dict = {}


def _load_coop_mdp(path=COOP_MDP):
    class SceneEntityCfg:                       # evaluated as a default argument at def time
        def __init__(self, name, **kw):
            self.name = name

    isaaclab = types.ModuleType("isaaclab")
    isaaclab.__path__ = []
    utils = types.ModuleType("isaaclab.utils")
    utils.__path__ = []
    assets = types.ModuleType("isaaclab.assets")
    assets.Articulation = assets.RigidObject = object
    managers = types.ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = SceneEntityCfg
    umath = types.ModuleType("isaaclab.utils.math")
    umath.subtract_frame_transforms = None
    stubs = {"isaaclab": isaaclab, "isaaclab.utils": utils, "isaaclab.assets": assets,
             "isaaclab.managers": managers, "isaaclab.utils.math": umath}
    _STUBS.update(stubs)
    with mock.patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location("coop_lift_mdp_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


coop = _load_coop_mdp()


def axis_angle_wxyz(axis, deg):
    """Rotation of `deg` about a WORLD axis, (w, x, y, z)."""
    n = math.sqrt(sum(a * a for a in axis))
    s = math.sin(math.radians(deg) / 2.0)
    return (math.cos(math.radians(deg) / 2.0), *(a / n * s for a in axis))


def qmul_wxyz(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def body_z_in_world_wxyz(q):
    """World z component of the body z axis, by rotating (0, 0, 1) -- independent of R22's closed form."""
    w, x, y, z = q
    p = qmul_wxyz(qmul_wxyz(q, (0.0, 0.0, 0.0, 1.0)), (w, -x, -y, -z))
    return p[3]


def _pre_fix_object_tilt(q):
    """`object_tilt_l2` as it was before 2026-09-28, verbatim indices."""
    up_z = 1.0 - 2.0 * (q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2])
    return torch.square(1.0 - up_z)


def _env(stored):
    obj = types.SimpleNamespace(data=types.SimpleNamespace(root_quat_w=stored))
    return types.SimpleNamespace(scene={"object": obj})


def _stored(wxyz_rows, order):
    return torch.tensor([reorder(q, order) for q in wxyz_rows], dtype=torch.float32)


#: The fixed spawn: the payload's (w, x, y, z) literal from coop_lift_env_cfg.
SPAWN_WXYZ = (1.0, 0.0, 0.0, 0.0)


class _Order:
    """Pin the running stack's quaternion order for native_quat and unpack_wxyz."""

    def __init__(self, order):
        self.patches = [mock.patch.object(qo, "quat_order", lambda: order),
                        mock.patch.object(qo, "_CACHED_ORDER", order)]

    def __enter__(self):
        for p in self.patches:
            p.start()

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()


class SpawnLiteralTests(unittest.TestCase):
    def test_object_literal_is_the_wxyz_identity(self):
        tree = ast.parse(LIFT_CFG.read_text())
        val = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and getattr(n.targets[0], "id", "") == "_OBJECT_ROT_WXYZ")
        self.assertEqual(val, SPAWN_WXYZ)

    def test_native_quat_is_the_identity_on_both_stacks(self):
        with _Order(XYZW):
            self.assertEqual(native_quat(SPAWN_WXYZ), (0.0, 0.0, 0.0, 1.0))   # v60 identity
        with _Order(WXYZ):
            # v51: the literal passes through unchanged -- no v51 number moves
            self.assertEqual(native_quat(SPAWN_WXYZ), (1.0, 0.0, 0.0, 0.0))
            self.assertEqual(native_quat(SPAWN_WXYZ), SPAWN_WXYZ)

    def test_the_old_raw_literal_on_v60_was_a_half_turn_about_x(self):
        raw = (1.0, 0.0, 0.0, 0.0)                        # read as (x, y, z, w) on v60
        x, y, z, w = raw
        self.assertAlmostEqual(body_z_in_world_wxyz((w, x, y, z)), -1.0)


class ObjectTiltV60Tests(unittest.TestCase):
    """Stored (x, y, z, w), as Isaac Lab 3.0 keeps root_quat_w."""

    def tilt(self, *rotations_wxyz):
        rows = [qmul_wxyz(r, SPAWN_WXYZ) for r in rotations_wxyz]   # world-frame turn of the spawn
        with _Order(XYZW):
            return coop.object_tilt_l2(_env(_stored(rows, XYZW))).tolist()

    def test_zero_at_the_fixed_spawn(self):
        with _Order(XYZW):
            spawn = torch.tensor([native_quat(SPAWN_WXYZ)])
            self.assertEqual(coop.object_tilt_l2(_env(spawn)).tolist(), [0.0])

    def test_side_face_is_one_about_world_x_and_y(self):
        got = self.tilt(axis_angle_wxyz((1, 0, 0), 90), axis_angle_wxyz((0, 1, 0), 90),
                        axis_angle_wxyz((1, 0, 0), -90), axis_angle_wxyz((0, 1, 0), -90))
        for v in got:
            self.assertAlmostEqual(v, 1.0, places=5)

    def test_yaw_is_free(self):
        got = self.tilt(axis_angle_wxyz((0, 0, 1), 90), axis_angle_wxyz((0, 0, 1), -135))
        for v in got:
            self.assertAlmostEqual(v, 0.0, places=6)

    def test_upside_down_is_four(self):
        self.assertAlmostEqual(self.tilt(axis_angle_wxyz((1, 0, 0), 180))[0], 4.0, places=5)

    def test_matches_body_z_for_any_rotation(self):
        g = torch.Generator().manual_seed(1234)
        q = torch.nn.functional.normalize(torch.randn(256, 4, generator=g, dtype=torch.float64), dim=-1)
        rows = [tuple(float(v) for v in r) for r in q]
        want = torch.tensor([(1.0 - body_z_in_world_wxyz(r)) ** 2 for r in rows], dtype=torch.float64)
        stored = torch.tensor([reorder(r, XYZW) for r in rows], dtype=torch.float64)
        with _Order(XYZW):
            got = coop.object_tilt_l2(_env(stored))
        self.assertTrue(torch.allclose(got, want, atol=1e-9))


class PreFixDocumentationTests(unittest.TestCase):
    """What the pre-fix code did on v60, so the ledger's diagnosis stays checkable."""

    def test_pre_fix_read_the_wrong_axes(self):
        rows = [axis_angle_wxyz((1, 0, 0), 90), axis_angle_wxyz((0, 0, 1), 90)]
        old = _pre_fix_object_tilt(_stored(rows, XYZW)).tolist()
        self.assertAlmostEqual(old[0], 0.0, places=6)     # a roll about x was free
        self.assertAlmostEqual(old[1], 1.0, places=5)     # a pure yaw was charged as a side-lie

    def test_fixing_one_alone_is_worse(self):
        flipped = torch.tensor([[1.0, 0.0, 0.0, 0.0]])     # the old raw spawn, stored xyzw
        fixed_spawn = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        with _Order(XYZW):
            new = coop.object_tilt_l2
            self.assertEqual(_pre_fix_object_tilt(flipped).tolist(), [0.0])    # both bugs: cancel
            self.assertEqual(new(_env(flipped)).tolist(), [4.0])               # reward only: -4 x weight per step
            self.assertEqual(_pre_fix_object_tilt(fixed_spawn).tolist(), [0.0])  # spawn only: 0 at rest ...
            self.assertEqual(new(_env(fixed_spawn)).tolist(), [0.0])             # both fixed: 0 at rest
        # ... but spawn-only still reads the wrong axes once the cube moves (test above).


class V51UnchangedTests(unittest.TestCase):
    """Stored (w, x, y, z): the fixed term is bit-for-bit the pre-fix term."""

    def test_bitwise_identical_on_wxyz(self):
        g = torch.Generator().manual_seed(51)
        for dtype in (torch.float32, torch.float64):
            q = torch.nn.functional.normalize(torch.randn(4096, 4, generator=g, dtype=dtype), dim=-1)
            with _Order(WXYZ):
                new = coop.object_tilt_l2(_env(q))
            self.assertTrue(torch.equal(new, _pre_fix_object_tilt(q)), dtype)

    def test_v51_known_rotations(self):
        rows = [SPAWN_WXYZ, axis_angle_wxyz((1, 0, 0), 90), axis_angle_wxyz((0, 1, 0), 90),
                axis_angle_wxyz((0, 0, 1), 90)]
        with _Order(WXYZ):
            spawn_stored = torch.tensor([native_quat(SPAWN_WXYZ)])
            got = coop.object_tilt_l2(_env(_stored(rows, WXYZ))).tolist()
            self.assertEqual(coop.object_tilt_l2(_env(spawn_stored)).tolist(), [0.0])
        for v, want in zip(got, (0.0, 1.0, 1.0, 0.0)):
            self.assertAlmostEqual(v, want, places=5)


SMOKE = REPO / "slurm/repo20260923/gpu_quatfix_smoke.sbatch"
SMOKE_INNER = REPO / "slurm/repo20260923/inner_quatfix_smoke.sh"


class SmokeLauncherTests(unittest.TestCase):
    """The smoke's header states the predeclared rule its verdict code applies."""

    def test_header_states_the_rule_and_scope(self):
        head = SMOKE.read_text().split("set -euo pipefail")[0]
        for s in ("PREDECLARED RULE", "A1", "A2", "A3", "A4", "B1", "B2",
                  "Curriculum/cube_tilted <= 0.01", "|Episode_Reward/object_tilt| <= 0.01",
                  "within 1e-4", "0 / 1 / 1 / 0 / 4", "INCOMPLETE", "never read",
                  "NEW", "never compared with", "re-score", "SCRIPTED", "seed 100",
                  "reward NOT fixed"):
            self.assertIn(s, head, s)

    def test_verdict_code_applies_the_header_thresholds(self):
        body = SMOKE.read_text().split("set -euo pipefail")[1]
        for s in ('["cfg_object_rot"] == [0.0, 0.0, 0.0, 1.0]', '"quat_order") == "xyzw"',
                  '["object_tilt_max"] <= 1e-4', '["cube_tilt_deg_max"] <= 1.0',
                  '"want_object_tilt"], 1e-3', '"want_cube_tilt_deg"], 0.5',
                  '["max_abs_err_object_tilt"] <= 1e-4', '["max_abs_err_tilt_deg"] <= 0.5',
                  '[TAG_TILTED]["value"] <= 0.01', 'abs(it0[TAG_OT]["value"]) <= 0.01',
                  "refusing to overwrite", "0|1)", "MAX_ITER=20", "NUM_ENVS=1024"):
            self.assertIn(s, body, s)

    def test_probe_wants_match_the_rule(self):
        inner = SMOKE_INNER.read_text()
        self.assertIn('"want_object_tilt": [0.0, 1.0, 1.0, 0.0, 4.0]', inner)
        self.assertIn('"want_cube_tilt_deg": [0.0, 90.0, 90.0, 0.0, 180.0]', inner)
        self.assertIn("refusing to overwrite", inner)


class StubsDoNotLeakTest(unittest.TestCase):
    def test_stubs_lived_only_for_the_load(self):
        self.assertTrue(_STUBS)
        self.assertEqual([k for k, v in _STUBS.items() if sys.modules.get(k) is v], [])


if __name__ == "__main__":
    unittest.main()
