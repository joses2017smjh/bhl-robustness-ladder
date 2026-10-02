"""CubeToShelfStand4 (roll-proof lift) terms, rules and launchers, without Isaac Lab.

`stand4_mdp` is loaded from its file path, as the launchers load it: the
`bhl_robust.tasks` package registers gym ids and needs a running simulator.
Synthetic cube poses are written (w, x, y, z), stored in the stack's layout with
`quat_order.reorder` and read back with `quat_order.unpack_wxyz` (the 2976f36
helpers), xyzw as on v60 unless a test says otherwise.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import math
import re
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

REPO = Path(__file__).resolve().parents[1]
S4_PATH = REPO / "src/bhl_robust/tasks/stand4_mdp.py"
STAND_MDP = REPO / "src/bhl_robust/tasks/stand_mdp.py"
TASK_V2_CFG = REPO / "src/bhl_robust/tasks/task_v2_env_cfg.py"
TASKS_INIT = REPO / "src/bhl_robust/tasks/__init__.py"
OVERLAY = REPO / "assets/cloth/berkeley_humanoid_lite_hand_colliders.usda"
TRAIN3 = REPO / "slurm/repo20260923/gpu_v2_stand3_train.sbatch"
TRAIN4 = REPO / "slurm/repo20260923/gpu_v2_stand4_train.sbatch"
SMOKE4 = REPO / "slurm/repo20260923/gpu_v2_stand4_smoke.sbatch"
INNER4 = REPO / "slurm/repo20260923/inner_v2_stand4_smoke.sh"
STAND3_DUMP = (REPO / "external/Berkeley-Humanoid-Lite/logs/rsl_rl/task_v2/"
               "2026-09-28_23-39-39_quatfix-smoke-stand3-blind-s100-j21463685/params")
STEP0_COLLIDERS = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/"
                       "campaign-stand4/step0/usd_colliders.json")

#: task_v2_env_cfg.py before Stand4 was appended (HEAD 094797e): byte length and sha256.
TASK_V2_PRE_STAND4 = (35686, "0dc6e7fbe7647b38ffdf104d05a8321720d14419fcdc6854255deb4f2de74fc5")
#: stand_mdp.py at HEAD 094797e; Stand4 does not touch it.
STAND_MDP_SHA = "53081846295bb72e64ff05de726e8a18ea8e7284844c9eff43434ee57ee3aaea"
STAND4_MARKER = "\n\n# --------------------------------------------- roll-proof lift cube, Stand4"

#: The frozen rule, verbatim from the workstream task (2026-10-01).
RULE_TEXT = (
    "PREDECLARED RULE (frozen; Stand3's structure): the kill rule at model_1000 exactly as "
    "Stand3's launcher applies it (copy its condition and window verbatim); a seed is COMPLETE "
    "iff it reaches iteration 7999; Stand4 PASSES iff >= 1 of 2 seeds has last-200-iteration "
    "success >= 0.10, where success is the Stand4 termination above (seated, tilt <= 8 deg, "
    "released). Otherwise NEGATIVE; INCOMPLETE if a seed is neither complete nor stopped by the "
    "kill rule. Stand4 is a different task from Stand3 and CubeToShelf and is never compared "
    "with their numbers. Labels: LEARNED crew policies; ORACLE privileged observations exactly "
    "as Stand3 uses them (say which); MODIFIED hand colliders (the overlay) relative to Stand3.")


def _load():
    spec = importlib.util.spec_from_file_location("stand4_mdp_under_test", S4_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s4 = _load()
sm = s4.sm
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
from bhl_robust import quat_order as qo  # noqa: E402  (import-safe off-simulator)

S2 = math.sqrt(2.0)
S3 = math.sqrt(3.0)
H = s4.CUBE_HALF


def q_axis(axis, deg):
    """(w, x, y, z) of a rotation by `deg` about `axis`."""
    n = math.sqrt(sum(a * a for a in axis))
    h = math.radians(deg) / 2.0
    return (math.cos(h),) + tuple(math.sin(h) * a / n for a in axis)


IDENT = (1.0, 0.0, 0.0, 0.0)
EDGE_45 = q_axis((1, 0, 0), 45.0)                          # rolled onto an edge
CORNER = q_axis((1, -1, 0), math.degrees(math.acos(1 / S3)))   # body diagonal vertical


def cols(rows_wxyz, order="xyzw", dtype=torch.float32):
    """Store (w, x, y, z) rows in `order` and read them back as (w, x, y, z) columns."""
    stored = torch.tensor([qo.reorder(r, order) for r in rows_wxyz], dtype=dtype)
    return qo.unpack_wxyz(stored, order=order)


def P(*rows):
    return torch.tensor(rows, dtype=torch.float32)


def gate(p, rows, order="xyzw"):
    return s4.roll_proof_mask(p, *cols(rows, order)).tolist()


def clear(p, rows, order="xyzw"):
    return s4.support_clearance(p, *cols(rows, order)).tolist()


def _norm(text: str) -> str:
    return " ".join(text.split())


def _header(path: Path) -> str:
    """The comment header (before `set -euo pipefail`), '# ' stripped, normalised."""
    head = path.read_text().split("set -euo pipefail")[0]
    return _norm(" ".join(ln[1:] for ln in head.splitlines() if ln.startswith("#")))


def _kill_block(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("#   kill      Stand2's rule:"))
    j = next(k for k in range(i + 1, len(lines)) if lines[k].startswith("#   complete"))
    return lines[i:j]


# ----------------------------------------------------------------- imports


class ImportAndConstantTests(unittest.TestCase):
    def test_imports_without_isaaclab(self):
        had = {k for k in sys.modules if k == "isaaclab" or k.startswith("isaaclab.")}
        _load()
        now = {k for k in sys.modules if k == "isaaclab" or k.startswith("isaaclab.")}
        self.assertEqual(now - had, set())

    def test_frozen_constants(self):
        self.assertEqual((s4.CORNER_CLEARANCE, s4.LIFT_TILT_MAX_DEG, s4.SEAT_TILT_MAX_DEG,
                          s4.RELEASE_FORCE_N), (0.02, 15.0, 8.0, 1.0))
        self.assertEqual((s4.STAND4_MAX_ITER, s4.COMPLETE_ITER, s4.RESULT_WINDOW,
                          s4.RESULT_MIN_SUCCESS, s4.STAND4_SEEDS), (8000, 7999, 200, 0.10, (0, 1)))
        self.assertEqual(sm.SEAT_HOLD_STEPS, 12)            # held as Stand3 holds seating
        self.assertEqual(s4.SMOKE_SEED, 100)
        self.assertNotIn(s4.SMOKE_SEED, s4.STAND4_SEEDS)

    def test_tilt_definition_is_stated_for_both_clauses(self):
        for note in (s4.TILT_NOTE, s4.SUCCESS4_NOTE):
            self.assertIn("own z axis and world up", note)
            self.assertIn("reads 90 deg", note)
        self.assertIn("15-deg lift clause and the 8-deg success clause", s4.TILT_NOTE)
        self.assertIn("is not used", s4.TILT_NOTE)
        src = S4_PATH.read_text()
        self.assertIn("The same definition serves BOTH tilt clauses", src)
        self.assertIn("Tilt as in 1 (the cube's own z", src)
        # both clauses call the one function
        self.assertEqual(src.count("tilt = cube_tilt_deg(w, x, y, z)"), 2)
        self.assertIn("Stand3's completeness check", s4.COMPLETE_NOTE)

    def test_rule_text_is_verbatim(self):
        self.assertEqual(s4.PREDECLARED_RULE, RULE_TEXT)
        for s in ("object_pos_a/b", "base_lin_vel_a/b", "object_lin_vel", "object_ang_vel",
                  "LEARNED", "ORACLE", "MODIFIED hand colliders"):
            self.assertIn(s, s4.LABELS_NOTE)

    def test_supports_are_stand3_geometry(self):
        src = TASK_V2_CFG.read_text()
        self.assertIn('furniture._box("plinth", (0.26, 0.26, h), (0.0, 0.0, h / 2.0))', src)
        self.assertIn("h = STAND_CUBE_Z - 0.14", src)
        by = {name: (x0, x1, y0, y1, top) for name, x0, x1, y0, y1, top in s4.SUPPORTS}
        self.assertEqual(by["plinth"], (-0.13, 0.13, -0.13, 0.13, sm.PLINTH_TOP))
        self.assertAlmostEqual(sm.PLINTH_TOP, 0.41)
        self.assertAlmostEqual(sm.DECK_TOP, 0.43)
        for name, sgn in (("deck_pos", 1.0), ("deck_neg", -1.0)):
            x0, x1, y0, y1, top = by[name]
            self.assertAlmostEqual((x0 + x1) / 2.0, sgn * sm.DECK_CENTER)
            self.assertAlmostEqual(x1 - x0, sm.DECK_LEN)
            self.assertAlmostEqual(y1 - y0, sm.DECK_WIDTH)
            self.assertEqual(top, sm.DECK_TOP)


# ------------------------------------------------- roll-proof lift geometry


class RollProofGateTests(unittest.TestCase):
    """The four synthetic poses the task names, plus their boundaries."""

    def test_flat_cube_resting_on_the_plinth_pays_zero(self):
        p = P([0.0, 0.0, sm.PLINTH_TOP + H])
        self.assertAlmostEqual(clear(p, [IDENT])[0], 0.0, places=6)
        self.assertEqual(gate(p, [IDENT]), [False])
        # even with the centre clause forced open, the gate keeps the pay at 0
        pay = s4.lift_pay(p[:, 2], p, *cols([IDENT]), spawn_z=0.55, minimal_height=-1.0,
                          pinch_kernel=torch.ones(1), upright_gate=torch.ones(1))
        self.assertEqual(pay.tolist(), [0.0])

    def test_cube_rolled_onto_an_edge_on_the_plinth_pays_zero(self):
        z = sm.PLINTH_TOP + H * S2                     # resting on an edge: +0.058 m
        p = P([0.0, 0.0, z])
        self.assertAlmostEqual(clear(p, [EDGE_45])[0], 0.0, places=5)
        self.assertAlmostEqual(s4.cube_tilt_deg(*cols([EDGE_45])).item(), 45.0, places=3)
        self.assertEqual(gate(p, [EDGE_45]), [False])
        # what Stand3 paid on: its centre clause is met at the 0.04 level
        self.assertTrue(bool(s4.centre_clause(p[:, 2], 0.55, 0.04)[0]))
        pay = s4.lift_pay(p[:, 2], p, *cols([EDGE_45]), spawn_z=0.55, minimal_height=0.04,
                          pinch_kernel=torch.ones(1), upright_gate=torch.ones(1))
        self.assertEqual(pay.tolist(), [0.0])

    def test_cube_standing_on_a_corner_pays_zero(self):
        z = sm.PLINTH_TOP + H * S3                     # +0.102 m: above both levels
        p = P([0.0, 0.0, z])
        self.assertAlmostEqual(s4.lowest_corner_z(p, *cols([CORNER])).item(), sm.PLINTH_TOP, places=5)
        # its corners' xy AABB reaches |x|, |y| = 0.22 > the deck edge 0.17, so the
        # (conservative) footprint rule measures it against the higher deck top too
        self.assertAlmostEqual(clear(p, [CORNER])[0], sm.PLINTH_TOP - sm.DECK_TOP, places=5)
        self.assertTrue(bool(s4.centre_clause(p[:, 2], 0.55, 0.06)[0]))
        self.assertEqual(gate(p, [CORNER]), [False])

    def test_cube_lifted_3cm_and_flat_pays(self):
        p = P([0.0, 0.0, sm.PLINTH_TOP + H + 0.03])
        self.assertAlmostEqual(clear(p, [IDENT])[0], 0.03, places=5)
        self.assertEqual(gate(p, [IDENT]), [True])
        # the whole term at a curriculum level: Stand3's centre clause (spawn 0.55 +
        # 0.04) x the gate x the pinch kernel x the upright gate
        p5 = P([0.0, 0.0, 0.55 + 0.05])
        pay = s4.lift_pay(p5[:, 2], p5, *cols([IDENT]), spawn_z=0.55, minimal_height=0.04,
                          pinch_kernel=torch.tensor([0.8]), upright_gate=torch.tensor([0.5]))
        self.assertAlmostEqual(pay.item(), 0.4, places=6)

    def test_cube_lifted_but_tilted_20deg_pays_zero(self):
        p = P([0.0, 0.0, 0.66])
        q = q_axis((0, 1, 0), 20.0)
        self.assertGreater(clear(p, [q])[0], s4.CORNER_CLEARANCE)   # only the tilt fails
        self.assertEqual(gate(p, [q]), [False])
        pay = s4.lift_pay(p[:, 2], p, *cols([q]), spawn_z=0.55, minimal_height=0.04,
                          pinch_kernel=torch.ones(1), upright_gate=torch.ones(1))
        self.assertEqual(pay.tolist(), [0.0])

    def test_tilt_boundary_15deg(self):
        p = P([0.0, 0.0, 0.70], [0.0, 0.0, 0.70])
        self.assertEqual(gate(p, [q_axis((1, 0, 0), 14.9), q_axis((1, 0, 0), 15.1)]), [True, False])
        self.assertEqual(gate(p, [q_axis((1, 1, 0), 14.9), q_axis((1, -1, 0), 15.1)]), [True, False])
        # a pure yaw never tilts the cube
        self.assertEqual(gate(p, [q_axis((0, 0, 1), 40.0), q_axis((0, 0, 1), 90.0)]), [True, True])

    def test_clearance_boundary_2cm(self):
        base = sm.PLINTH_TOP + H
        p = P([0.0, 0.0, base + 0.0199], [0.0, 0.0, base + 0.0201])
        self.assertEqual(gate(p, [IDENT, IDENT]), [False, True])

    def test_supports_under_the_footprint_only(self):
        seated = sm.DECK_TOP + H
        # over each deck: measured from the deck top (0.43), not the plinth
        p = P([sm.DECK_CENTER, 0.0, seated + 0.03], [-sm.DECK_CENTER, 0.0, seated + 0.01],
              [sm.DECK_CENTER, 0.0, seated])
        self.assertEqual(gate(p, [IDENT] * 3), [True, False, False])
        # straddling plinth and deck (x = 0.15): the higher top (deck) counts
        p = P([0.15, 0.0, seated + 0.015])
        self.assertAlmostEqual(clear(p, [IDENT])[0], 0.015, places=5)
        # far from every raised support: the floor only
        p = P([1.0, 1.0, 0.30])
        self.assertAlmostEqual(clear(p, [IDENT])[0], 0.30 - H, places=5)
        # a rotated cube's footprint grows: 1 cm beyond the deck's side edge, flat it
        # is over the floor only; yawed 45 deg its corner reaches over the deck
        p = P([sm.DECK_CENTER, sm.DECK_WIDTH / 2.0 + H + 0.01, sm.DECK_TOP + H + 0.015])
        self.assertAlmostEqual(clear(p, [IDENT])[0], sm.DECK_TOP + 0.015, places=5)
        self.assertAlmostEqual(clear(p, [q_axis((0, 0, 1), 45.0)])[0], 0.015, places=5)

    def test_lowest_corner_and_clearance_against_brute_force(self):
        g = torch.Generator().manual_seed(7)
        q = torch.randn(256, 4, generator=g, dtype=torch.float64)
        q = q / q.norm(dim=-1, keepdim=True)
        w, x, y, z = q.unbind(-1)
        p = torch.cat([torch.rand(256, 2, generator=g, dtype=torch.float64) * 1.2 - 0.6,
                       torch.rand(256, 1, generator=g, dtype=torch.float64) * 0.5 + 0.35], dim=-1)
        R = torch.stack([
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)
        signs = torch.tensor([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)],
                             dtype=torch.float64) * H
        corners = p[:, None, :] + torch.einsum("nij,kj->nki", R, signs)      # (N, 8, 3)
        low = corners[..., 2].min(dim=1).values
        self.assertTrue(torch.allclose(s4.lowest_corner_z(p, w, x, y, z), low, atol=1e-9))
        lo_xy, hi_xy = corners[..., :2].min(dim=1).values, corners[..., :2].max(dim=1).values
        want = low.clone()
        for _n, x0, x1, y0, y1, top in s4.SUPPORTS:
            under = (hi_xy[:, 0] > x0) & (lo_xy[:, 0] < x1) & (hi_xy[:, 1] > y0) & (lo_xy[:, 1] < y1)
            want = torch.where(under, torch.minimum(want, low - top), want)
        self.assertTrue(torch.allclose(s4.support_clearance(p, w, x, y, z), want, atol=1e-9))
        tilt = torch.rad2deg(torch.acos(R[:, 2, 2].clamp(-1, 1)))
        self.assertTrue(torch.allclose(s4.cube_tilt_deg(w, x, y, z), tilt, atol=1e-6))

    def test_v60_xyzw_and_v51_wxyz_agree_and_the_order_matters(self):
        p = P([0.0, 0.0, 0.70], [0.0, 0.0, 0.70], [0.0, 0.0, 0.60])
        rows = [EDGE_45, q_axis((0, 1, 0), 20.0), IDENT]
        self.assertEqual(gate(p, rows, "xyzw"), gate(p, rows, "wxyz"))
        self.assertEqual(gate(p, rows, "xyzw"), [False, False, True])
        # reading v60's xyzw storage as if it were wxyz turns the 45 deg roll into a
        # yaw: tilt 0 and an open gate. Hence quat_order.unpack_wxyz, never indices.
        stored = torch.tensor([qo.reorder(EDGE_45, "xyzw")])
        wrong = qo.unpack_wxyz(stored, order="wxyz")
        self.assertAlmostEqual(s4.cube_tilt_deg(*wrong).item(), 0.0, places=4)
        self.assertEqual(s4.roll_proof_mask(P([0.0, 0.0, 0.70]), *wrong).tolist(), [True])

    def test_nan_pose_never_pays(self):
        p = P([0.0, 0.0, float("nan")])
        self.assertEqual(gate(p, [IDENT]), [False])

    def test_no_raw_quaternion_literals_in_stand4(self):
        tree = ast.parse(S4_PATH.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Tuple) and len(node.elts) == 4 and all(
                    isinstance(e, ast.Constant) and isinstance(e.value, (int, float)) for e in node.elts):
                self.fail(f"numeric 4-tuple literal at line {node.lineno}")
        src = S4_PATH.read_text()
        self.assertIn("from bhl_robust.quat_order import unpack_wxyz", src)
        self.assertNotIn("root_quat_w[:", src)
        self.assertIn("native_quat((1.0, 0.0, 0.0, 0.0))", INNER4.read_text())   # smoke teleport


# ----------------------------------------------------------------- release


class ReleaseTests(unittest.TestCase):
    N_COL = s4.N_RELEASE_BODIES

    def fm(self, *col_forces):
        """(N, 1, 26, 3) with force (0, 0, f) on column c for each (c, f) per env."""
        out = torch.zeros(len(col_forces), 1, self.N_COL, 3)
        for i, cf in enumerate(col_forces):
            for c, f in cf:
                out[i, 0, c] = torch.tensor([0.0, f * 0.6, f * 0.8])
        return out

    def test_release_predicate_and_threshold(self):
        hand = s4.RELEASE_FILTER_EXPRS.index("{ENV_REGEX_NS}/robot_a/arm_left_hand_link")
        m = self.fm([], [(hand, 0.99)], [(hand, 1.0)], [(0, 1.5)], [(25, 50.0), (3, 0.2)])
        self.assertEqual(s4.released_mask(m).tolist(), [True, True, False, False, False])
        self.assertTrue(torch.allclose(s4.robot_force_max(m), torch.tensor([0.0, 0.99, 1.0, 1.5, 50.0])))

    def test_nan_is_contact(self):
        m = self.fm([], [])
        m[1, 0, 7, 0] = float("nan")
        self.assertEqual(s4.released_mask(m).tolist(), [True, False])
        self.assertTrue(math.isnan(s4.robot_force_max(m)[1].item()))

    def test_degenerate_matrices_raise(self):
        with self.assertRaises(ValueError):
            s4.robot_force_max(None)
        with self.assertRaises(ValueError):
            s4.robot_force_max(torch.zeros(4, 1, 0, 3))

    def test_filters_one_explicit_link_per_pattern(self):
        exprs = s4.RELEASE_FILTER_EXPRS
        self.assertEqual(len(exprs), 26)
        self.assertEqual(len(set(exprs)), 26)
        for e in exprs:
            self.assertTrue(e.startswith("{ENV_REGEX_NS}/robot_"), e)
            self.assertNotIn("*", e)
            self.assertNotIn("(", e)
        for r in ("robot_a", "robot_b"):
            for side in ("left", "right"):
                self.assertIn(f"{{ENV_REGEX_NS}}/{r}/arm_{side}_hand_link", exprs)

    @unittest.skipUnless(STEP0_COLLIDERS.is_file(), "step-0 collider inventory not available")
    def test_filters_are_exactly_the_collider_bearing_links(self):
        import json
        inv = json.loads(STEP0_COLLIDERS.read_text())["base_usd"]["bodies_with_colliders"]
        base = {k.rsplit("/", 1)[-1] for k in inv}
        self.assertEqual(set(s4.RELEASE_LINKS), base | {"arm_left_hand_link", "arm_right_hand_link"})

    def test_overlay_adds_a_collider_to_both_hand_links_only(self):
        txt = OVERLAY.read_text()
        self.assertEqual(re.findall(r'over "(arm_\w+_hand_link)"', txt),
                         ["arm_right_hand_link", "arm_left_hand_link"])
        self.assertEqual(txt.count('def Mesh "hand_collider"'), 2)
        self.assertEqual(txt.count('"PhysicsCollisionAPI"'), 2)
        self.assertEqual(txt.count('physics:approximation = "convexHull"'), 2)
        self.assertNotRegex(txt, r"physics:mass|PhysicsMassAPI|PhysicsRevoluteJoint|xformOp")
        sub = re.search(r"subLayers = \[\s*@([^@]+)@", txt).group(1)
        self.assertTrue(sub.endswith("berkeley_humanoid_lite/usd/berkeley_humanoid_lite.usd"), sub)
        base = (OVERLAY.parent / sub).resolve()
        if base.parent.is_dir():
            self.assertTrue(base.is_file(), base)
        self.assertEqual(s4.HAND_COLLIDER_USD, OVERLAY.resolve())
        self.assertEqual(s4.OVERLAY_NAME, OVERLAY.name)


class PlacedMaskTests(unittest.TestCase):
    def test_seated_flat_and_released(self):
        p = P(*[[sm.DECK_CENTER, 0.0, sm.DECK_SEATED_Z]] * 5)
        speed = torch.zeros(5)
        tilt = torch.tensor([0.0, 7.9, 8.1, 0.0, 90.0])
        rel = torch.tensor([True, True, True, False, True])
        self.assertEqual(s4.placed_mask(p, speed, tilt, rel).tolist(), [True, True, False, False, False])
        # Stand3's seated test is unchanged and still required
        self.assertEqual(s4.placed_mask(P([0.0, 0.0, 0.55]), torch.zeros(1), torch.zeros(1),
                                        torch.tensor([True])).tolist(), [False])


# ------------------------------------------------- env-facing terms (fakes)


class _FakeCoop:
    """coop_lift_mdp stand-in: Stand3's object_is_lifted (centre clause x pinch kernel)."""

    def __init__(self, n, kernel=0.9):
        self.kern = torch.full((n,), kernel)

    def _t(self, v):
        return v

    def _tilt_from_quat(self, robot):
        return torch.zeros_like(self.kern)

    def _pinch_weight(self, env, std=0.12):
        return self.kern

    def object_is_lifted(self, env, minimal_height):
        z = env.scene["object"].data.root_pos_w[:, 2]
        return (z > env.cfg.object_spawn_z + minimal_height).float() * self.kern


class _FakeV2:
    def _obj_local(self, env, name="object"):
        return env.scene["object"].data.root_pos_w

    def _held(self, env, key, now, steps):
        buf = getattr(env, key, None)
        if buf is None:
            buf = torch.zeros_like(now, dtype=torch.long)
        buf = torch.where(now, buf + 1, torch.zeros_like(buf))
        setattr(env, key, buf)
        return buf >= steps


def _env(p, rows, forces=None, order="xyzw"):
    stored = torch.tensor([qo.reorder(r, order) for r in rows], dtype=torch.float32)
    obj = types.SimpleNamespace(data=types.SimpleNamespace(
        root_pos_w=p, root_quat_w=stored, root_lin_vel_w=torch.zeros_like(p)))
    scene = {"object": obj, "robot_a": "robot_a", "robot_b": "robot_b"}
    if forces is not None:
        scene[s4.CUBE_SENSOR] = types.SimpleNamespace(data=types.SimpleNamespace(force_matrix_w=forces))
    return types.SimpleNamespace(scene=scene, cfg=types.SimpleNamespace(object_spawn_z=0.55),
                                 extras={}, num_envs=p.shape[0], device="cpu",
                                 _bhl_pinch_d=torch.full((p.shape[0],), 0.05))


class EnvTermTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load()
        self.patch = mock.patch.object(qo, "_CACHED_ORDER", "xyzw")
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def fakes(self, n):
        self.mod.sm._coop = lambda: _FakeCoop(n)
        self.mod.sm._v2 = lambda: _FakeV2()

    def test_lift_term_is_stand3_times_the_gate_and_cached_for_replay(self):
        m = self.mod
        self.fakes(4)
        p = P([0.0, 0.0, 0.60], [0.0, 0.0, sm.PLINTH_TOP + H * S2], [0.0, 0.0, 0.66],
              [0.0, 0.0, sm.PLINTH_TOP + H * S3])
        env = _env(p, [IDENT, EDGE_45, q_axis((0, 1, 0), 20.0), CORNER])
        stand3 = m.sm.gated_object_is_lifted(env, minimal_height=0.04)
        self.assertTrue(torch.allclose(stand3, torch.full((4,), 0.9)))     # Stand3 paid all four
        pay = m.stand4_object_is_lifted(env, minimal_height=0.04)
        self.assertTrue(torch.allclose(pay, torch.tensor([0.9, 0.0, 0.0, 0.0])))
        c = env._bhl_s4
        self.assertIs(env.extras["stand4"], c)
        self.assertEqual(set(m.LIFT_CACHE_KEYS) - set(c), set())
        self.assertEqual(c["minimal_height"], 0.04)
        for k in m.LIFT_CACHE_KEYS[1:]:
            self.assertEqual(tuple(c[k].shape), (4,), k)
        replay = c["centre_ok"].float() * c["roll_ok"].float() * c["pinch_kernel"] * c["upright_gate"]
        self.assertTrue(torch.equal(replay, c["lift_pay"]))
        self.assertEqual(c["roll_ok"].tolist(), [True, False, False, False])
        self.assertEqual(c["centre_ok"].tolist(), [True, True, True, True])

    def test_lift_term_signature_keeps_the_curriculum_parameter(self):
        sig = inspect.signature(s4.stand4_object_is_lifted).parameters
        self.assertIn("minimal_height", sig)
        self.assertIs(sig["minimal_height"].default, inspect.Parameter.empty)

    def test_success_needs_seated_flat_and_released_held_12_steps(self):
        m = self.mod
        self.fakes(3)
        hand = m.RELEASE_FILTER_EXPRS.index("{ENV_REGEX_NS}/robot_b/arm_right_hand_link")
        f = torch.zeros(3, 1, m.N_RELEASE_BODIES, 3)
        f[1, 0, hand, 2] = 2.0                          # a hand still pushing on the cube
        p = P(*[[sm.DECK_CENTER, 0.0, sm.DECK_SEATED_Z]] * 3)
        env = _env(p, [IDENT, IDENT, q_axis((1, 0, 0), 10.0)], forces=f)
        outs = [m.stand4_cube_placed(env, hold_steps=3).tolist() for _ in range(3)]
        self.assertEqual(outs, [[False] * 3, [False] * 3, [True, False, False]])
        c = env._bhl_s4
        self.assertEqual(set(m.SUCCESS_CACHE_KEYS) - set(c), set())
        self.assertEqual(c["released"].tolist(), [True, False, True])
        self.assertEqual(c["seated"].tolist(), [True, True, True])
        self.assertEqual(c["hold"].tolist(), [3, 0, 0])
        # release env 1: its count starts from zero, once per call
        f[1].zero_()
        outs = [m.stand4_cube_placed(env, hold_steps=3).tolist()[1] for _ in range(3)]
        self.assertEqual(outs, [False, False, True])
        self.assertEqual(inspect.signature(m.stand4_cube_placed).parameters["hold_steps"].default,
                         sm.SEAT_HOLD_STEPS)

    def test_diagnostics(self):
        m = self.mod
        self.fakes(3)
        f = torch.zeros(3, 1, m.N_RELEASE_BODIES, 3)
        f[1, 0, 0, 0] = 3.0
        p = P([sm.DECK_CENTER, 0.0, sm.DECK_SEATED_Z], [0.0, 0.0, 0.60], [0.0, 0.0, 0.70])
        env = _env(p, [IDENT, IDENT, q_axis((1, 0, 0), 30.0)], forces=f)
        self.assertTrue(math.isnan(m.lift_paid_share(env, None)))      # before any reward pass
        m.stand4_object_is_lifted(env, minimal_height=0.04)
        self.assertAlmostEqual(m.cube_tilt_mean(env, None), 10.0, places=3)
        self.assertAlmostEqual(m.roll_ok_share(env, None), 1 / 3, places=6)
        self.assertAlmostEqual(m.lift_paid_share(env, None), 1 / 3, places=6)
        self.assertAlmostEqual(m.released_share(env, None), 2 / 3, places=6)
        self.assertAlmostEqual(m.robot_force_max_mean(env, None), 1.0, places=6)
        self.assertAlmostEqual(m.pinch_distance_mean(env, None), 0.05, places=6)
        cc = m.corner_clearance_mean(env, None)
        want = (0.0 + 0.05 + (0.70 - H * (math.sin(math.radians(30)) + math.cos(math.radians(30)))
                              - sm.PLINTH_TOP)) / 3
        self.assertAlmostEqual(cc, want, places=5)


# ------------------------------------------------------------------- rules


def _sc(last, success=0.0, time_out=0.6, value=10.0, nan_success_at=None, ep_len=400.0):
    it = range(0, last + 1)
    succ = [(i, math.nan if i == nan_success_at else success) for i in it]
    out = {sm.TAG_LEN: [(i, ep_len) for i in it], sm.TAG_TIMEOUT: [(i, time_out) for i in it],
           sm.TAG_SUCCESS: succ, sm.TAG_FALLEN: [(i, 1 - time_out) for i in it],
           sm.TAG_VALUE: [(i, value) for i in it], s4.TAG_LIFT_LEVEL: [(i, 0.04) for i in it],
           s4.TAG_GATE: [(i, 0.9) for i in it]}
    for t in s4.DIAG_TAGS:
        out[t] = [(i, 0.1) for i in it]
    return out


class KillRuleTests(unittest.TestCase):
    def test_kill_is_stand3s_function(self):
        for args in ((70.0, 0.0, 0.9), (70.0, 0.05, 0.0), (326.1, 0.02, 0.05), (150.0, 0.2, 0.09),
                     (100.0, 0.10, 0.0), (99.9, 0.5, 0.0999), (math.nan, math.nan, math.nan)):
            self.assertEqual(s4.kill_verdict(*args)["verdict"], sm.kill_verdict3(*args)["verdict"], args)
        for sc in (_sc(1000, success=0.0, time_out=0.05), _sc(1000, success=0.5, ep_len=50.0),
                   _sc(1000, time_out=0.3, ep_len=99.0), _sc(1000, time_out=0.3)):
            self.assertEqual(s4.evaluate_kill(sc), sm.evaluate_kill3(sc))
        self.assertEqual((sm.KILL_AT_ITER, sm.KILL_WINDOW, sm.KILL_MIN_EP_LEN, sm.KILL_MIN_NONFALL),
                         (1000, 100, 100.0, 0.10))

    def test_kill_window_is_901_to_1000(self):
        sc = _sc(1100, time_out=0.6)
        for i in range(901, 1001):
            sc[sm.TAG_TIMEOUT][i] = (i, 0.0)
            sc[sm.TAG_FALLEN][i] = (i, 1.0)
        self.assertEqual(s4.evaluate_kill(sc)["verdict"], "KILL")
        sc = _sc(1100, time_out=0.6)
        for i in list(range(800, 901)) + list(range(1001, 1101)):
            sc[sm.TAG_TIMEOUT][i] = (i, 0.0)
        self.assertEqual(s4.evaluate_kill(sc)["verdict"], "CONTINUE")


class SeedRuleTests(unittest.TestCase):
    def test_complete_boundary_7999(self):
        self.assertFalse(s4.seed_result4(7998, 0.5)["complete"])
        self.assertFalse(s4.seed_result4(7998, 0.5)["decided"])
        self.assertTrue(s4.seed_result4(7999, 0.5)["complete"])
        self.assertTrue(s4.seed_result4(8000, 0.5)["passes"])

    def test_success_boundary_0_10(self):
        self.assertFalse(s4.seed_result4(7999, 0.0999)["passes"])
        self.assertTrue(s4.seed_result4(7999, 0.10)["passes"])
        self.assertFalse(s4.seed_result4(7999, math.nan)["passes"])
        self.assertTrue(s4.seed_result4(7999, math.nan)["decided"])

    def test_killed_and_bad_config(self):
        k = s4.seed_result4(1016, 0.9, killed=True)
        self.assertTrue(k["decided"] and k["killed"] and not k["passes"])
        b = s4.seed_result4(7999, 0.9, bad_config=True)
        self.assertFalse(b["complete"] or b["decided"] or b["passes"] or b["killed"])

    def test_pair_verdict_precedence(self):
        ok, neg = s4.seed_result4(7999, 0.2), s4.seed_result4(7999, 0.0)
        inc, kil = s4.seed_result4(4468, 0.9), s4.seed_result4(1016, 0.0, killed=True)
        bad = s4.seed_result4(7999, 0.9, bad_config=True)
        V = s4.pair_verdict4
        self.assertEqual(V([ok, inc]), f"{s4.PASS_LABEL} (1/2 seeds)")    # PASS whatever the other did
        self.assertEqual(V([ok, ok]), f"{s4.PASS_LABEL} (2/2 seeds)")
        self.assertTrue(V([neg, neg]).startswith("NEGATIVE (0/2"))
        self.assertTrue(V([kil, neg]).startswith("NEGATIVE") and "1 killed" in V([kil, neg]))
        self.assertTrue(V([neg, inc]).startswith("INCOMPLETE"))
        self.assertTrue(V([kil, inc]).startswith("INCOMPLETE"))
        self.assertTrue(V([bad, neg]).startswith("INCOMPLETE"))

    def test_evaluate_seed4_from_scalars(self):
        r = s4.evaluate_seed4(_sc(7999, success=0.12))
        self.assertTrue(r["complete"] and r["passes"])
        self.assertEqual(r["last_iter"], 7999)
        self.assertAlmostEqual(r["success"], 0.12)
        self.assertEqual(set(r["diagnostics_last200"]), set(s4.DIAG_TAGS))
        self.assertAlmostEqual(r["lift_level"], 0.04)
        # last 200 only: 0.5 before iteration 7800 does not count
        sc = _sc(7999, success=0.0)
        for i in range(0, 7800):
            sc[sm.TAG_SUCCESS][i] = (i, 0.5)
        self.assertFalse(s4.evaluate_seed4(sc)["passes"])
        sc = _sc(7999, success=0.0)
        for i in range(7800, 8000):
            sc[sm.TAG_SUCCESS][i] = (i, 0.125)
        self.assertTrue(s4.evaluate_seed4(sc)["passes"])
        # short run: never a result
        r = s4.evaluate_seed4(_sc(4468, success=0.5))
        self.assertFalse(r["complete"] or r["decided"] or r["passes"])

    def test_finite_window_and_value_gate_complete(self):
        """Stand3's completeness check (decided on review 2026-10-02): a seed that
        reached 7999 but went non-finite in its last 200 iterations is INCOMPLETE."""
        r = s4.evaluate_seed4(_sc(7999, success=0.2, nan_success_at=7990))
        self.assertFalse(r["window_finite"])
        self.assertFalse(r["complete"] or r["decided"] or r["passes"])
        # inf Loss/value inside the window, finite success >= 0.10: not a PASS
        sc = _sc(7999, success=0.2)
        sc[sm.TAG_VALUE][7950] = (7950, math.inf)
        r = s4.evaluate_seed4(sc)
        self.assertFalse(r["complete"] or r["decided"] or r["passes"])
        # NaN time_out on the last iteration
        sc = _sc(7999, success=0.2)
        sc[sm.TAG_TIMEOUT][7999] = (7999, math.nan)
        self.assertFalse(s4.evaluate_seed4(sc)["complete"])
        # the window boundary: 7799 is outside (last - 200, last], 7800 inside
        sc = _sc(7999, success=0.2)
        sc[sm.TAG_VALUE][7799] = (7799, math.inf)
        r = s4.evaluate_seed4(sc)
        self.assertTrue(r["complete"] and r["passes"])
        self.assertEqual(r["reported_not_gated"]["unstable_iters"], 1)       # reported only
        sc[sm.TAG_VALUE][7800] = (7800, math.inf)
        self.assertFalse(s4.evaluate_seed4(sc)["complete"])
        # no Loss/value in the window at all: not complete
        sc = _sc(7999, success=0.2)
        sc[sm.TAG_VALUE] = [(i, v) for i, v in sc[sm.TAG_VALUE] if i <= 7799]
        r = s4.evaluate_seed4(sc)
        self.assertFalse(r["has_value"])
        self.assertFalse(r["complete"] or r["passes"])
        # a runaway earlier in the run is reported, never gating
        sc = _sc(7999, success=0.2)
        sc[sm.TAG_VALUE][4357] = (4357, math.inf)
        r = s4.evaluate_seed4(sc)
        self.assertTrue(r["complete"] and r["passes"])
        self.assertEqual(r["reported_not_gated"]["unstable_iters"], 1)

    def test_complete_agrees_with_stand3_evaluate_seed3(self):
        def mutate(f):
            sc = _sc(7999, success=0.2)
            f(sc)
            return sc
        cases = {
            "clean": mutate(lambda sc: None),
            "nan success @7990": _sc(7999, success=0.2, nan_success_at=7990),
            "inf value @7950": mutate(lambda sc: sc[sm.TAG_VALUE].__setitem__(7950, (7950, math.inf))),
            "inf value @4357": mutate(lambda sc: sc[sm.TAG_VALUE].__setitem__(4357, (4357, math.inf))),
            "nan time_out @7999": mutate(lambda sc: sc[sm.TAG_TIMEOUT].__setitem__(7999, (7999, math.nan))),
            "no value in window": mutate(lambda sc: sc.__setitem__(
                sm.TAG_VALUE, [(i, v) for i, v in sc[sm.TAG_VALUE] if i <= 7799])),
            "short run": _sc(4468, success=0.5),
            "7998 only": _sc(7998, success=0.5),
        }
        for name, sc in cases.items():
            r3, r4 = sm.evaluate_seed3(sc), s4.evaluate_seed4(sc)
            self.assertEqual(r4["complete"], r3["complete"], name)
            self.assertEqual(r4["window_finite"], r3["window_finite"], name)
            self.assertEqual(r4["passes"], r3["placed"], name)

    def test_seed_result4_window_flags(self):
        self.assertTrue(s4.seed_result4(7999, 0.2)["passes"])
        for kw in ({"window_finite": False}, {"has_value": False}):
            r = s4.seed_result4(7999, 0.2, **kw)
            self.assertFalse(r["complete"] or r["decided"] or r["passes"], kw)
        # a killed seed is decided whatever its window
        k = s4.seed_result4(1016, 0.0, killed=True, window_finite=False)
        self.assertTrue(k["decided"] and not k["passes"])
        # pair: a NaN-window seed beside a clean negative is INCOMPLETE, as in Stand3
        nan_seed = s4.evaluate_seed4(_sc(7999, success=0.0, nan_success_at=7990))
        neg = s4.evaluate_seed4(_sc(7999, success=0.0))
        self.assertTrue(s4.pair_verdict4([nan_seed, neg]).startswith("INCOMPLETE"))
        self.assertTrue(sm.pair_result3([sm.evaluate_seed3(_sc(7999, success=0.0, nan_success_at=7990)),
                                         sm.evaluate_seed3(_sc(7999, success=0.0))]).startswith("INCOMPLETE"))


# -------------------------------------------------------- config check


def _dump(d) -> str:
    import yaml
    return yaml.dump(d, default_flow_style=False, sort_keys=False)    # Isaac Lab's dump_yaml


def _stand4_params():
    ns = "/World/envs/env_.*"

    def robot(r):
        return {"class_type": "isaaclab.assets.articulation.articulation:Articulation",
                "prim_path": f"{ns}/{r}",
                "spawn": {"func": "isaaclab.sim.spawners.from_files.from_files:spawn_from_usd",
                          "activate_contact_sensors": True, "scale": None,
                          "usd_path": str(s4.HAND_COLLIDER_USD), "variants": None}}

    env = {
        "seed": 0, "decimation": 8,
        "scene": {"num_envs": 1024, "robot_a": robot("robot_a"), "robot_b": robot("robot_b"),
                  "object": {"prim_path": f"{ns}/object",
                             "spawn": {"func": "isaaclab.sim.spawners.shapes.shapes:spawn_cuboid",
                                       "size": (0.28, 0.28, 0.28), "activate_contact_sensors": True}},
                  "contact_a": {"prim_path": f"{ns}/robot_a/.*", "filter_prim_paths_expr": []},
                  s4.CUBE_SENSOR: {
                      "class_type": "isaaclab.sensors.contact_sensor.contact_sensor:ContactSensor",
                      "prim_path": f"{ns}/object", "update_period": 0.005, "history_length": 0,
                      "filter_prim_paths_expr": [e.replace("{ENV_REGEX_NS}", ns)
                                                 for e in s4.RELEASE_FILTER_EXPRS],
                      "visualizer_cfg": {"prim_path": "/Visuals/ContactSensor",
                                         "markers": {"contact": {"radius": 0.02}}}}},
        "rewards": {
            "lift_progress": {"func": "bhl_robust.tasks.stand_mdp:gated_lift_progress",
                              "params": {"gate_free": 0.1, "gate_std": 0.25}, "weight": 2.0},
            "lifting_object": {"func": s4.FUNC_LIFT4,
                               "params": {"minimal_height": 0.04, "clearance": 0.02,
                                          "tilt_max_deg": 15.0, "gate_free": 0.1, "gate_std": 0.25},
                               "weight": 15.0},
            "object_xy": None,
            "placed": {"func": s4.FUNC_PLACED, "params": {"term_name": "success"}, "weight": 5000.0},
            "action_rate_clipped": {"func": s4.FUNC_ACTION_RATE, "params": {"bound": 10.0},
                                    "weight": -0.01}},
        "terminations": {
            "time_out": {"func": "isaaclab.envs.mdp.terminations:time_out", "params": {},
                         "time_out": True},
            "success": {"func": s4.FUNC_SUCCESS4,
                        "params": {"hold_steps": 12, "tilt_max_deg": 8.0, "release_force_n": 1.0,
                                   "sensor_name": s4.CUBE_SENSOR},
                        "time_out": False}},
        "curriculum": {
            "lift_height": {"func": "bhl_robust.tasks.coop_lift_mdp:lift_height_curriculum",
                            "params": {"term_name": "lifting_object", "step": 0.02,
                                       "min_height": 0.04, "max_height": 0.06}},
            "upright_gate": {"func": "bhl_robust.tasks.stand_mdp:upright_gate_mean",
                             "params": {"gate_free": 0.1, "gate_std": 0.25}},
            **{n: {"func": f"bhl_robust.tasks.stand4_mdp:{n}", "params": {}} for n in s4.DIAG_TERMS}},
        "commands": None,
    }
    agent = {"seed": 0, "experiment_name": "task_v2",
             "actor": {"distribution_cfg": {"init_std": 1.0, "std_type": "log"}},
             "algorithm": {"entropy_coef": 0.001, "gamma": 0.99}}
    return agent, env


class ConfigCheckTests(unittest.TestCase):
    def test_a_stand4_dump_passes(self):
        agent, env = _stand4_params()
        r = s4.config_check(_dump(agent), _dump(env))
        self.assertTrue(r["ok"], r["failed"])
        self.assertTrue(all(r["checks"].values()))

    def test_each_mutation_fails_its_clause(self):
        cases = [
            (lambda a, e: e["rewards"]["lifting_object"].update(func="bhl_robust.tasks.stand_mdp:gated_object_is_lifted"), "lifting_object_stand4_x15"),
            (lambda a, e: e["rewards"]["lifting_object"].update(weight=30.0), "lifting_object_stand4_x15"),
            (lambda a, e: e["rewards"]["lifting_object"]["params"].update(clearance=0.01), "lift_gate_0.02m_15deg"),
            (lambda a, e: e["rewards"]["placed"].update(weight=2500.0), "placed_success_bonus_x5000"),
            (lambda a, e: e["rewards"].pop("action_rate_clipped"), "action_rate_clipped"),
            (lambda a, e: e["terminations"]["success"].update(func="bhl_robust.tasks.stand_mdp:cube_on_side_deck"), "success_stand4"),
            (lambda a, e: e["terminations"]["success"]["params"].update(tilt_max_deg=9.0), "success_12steps_8deg_1N"),
            (lambda a, e: e["terminations"]["success"]["params"].update(hold_steps=6), "success_12steps_8deg_1N"),
            (lambda a, e: e["scene"]["robot_b"]["spawn"].update(usd_path="/x/berkeley_humanoid_lite.usd"), "robot_b_overlay_usd"),
            (lambda a, e: e["scene"]["object"]["spawn"].update(activate_contact_sensors=False), "cube_contact_reporting"),
            (lambda a, e: e["scene"][s4.CUBE_SENSOR].update(filter_prim_paths_expr=["/World/envs/env_.*/robot_a/.*"]), "cube_sensor_26_filters"),
            (lambda a, e: e["scene"].pop(s4.CUBE_SENSOR), "cube_sensor_26_filters"),
            (lambda a, e: e["curriculum"].pop("roll_ok"), "curriculum_terms"),
            (lambda a, e: e["curriculum"]["lift_height"]["params"].update(max_height=0.22), "lift_curriculum_cap_0.06"),
            (lambda a, e: a["algorithm"].update(entropy_coef=0.005), "agent_entropy_coef_0.001"),
            (lambda a, e: a["actor"]["distribution_cfg"].update(std_type="scalar"), "agent_std_type_log"),
        ]
        for mutate, clause in cases:
            agent, env = _stand4_params()
            mutate(agent, env)
            r = s4.config_check(_dump(agent), _dump(env))
            self.assertFalse(r["ok"], clause)
            self.assertIn(clause, r["failed"])

    @unittest.skipUnless((STAND3_DUMP / "env.yaml").is_file(), "Stand3 smoke run params not on disk")
    def test_a_real_stand3_dump_is_rejected_on_the_stand4_clauses_only(self):
        r = s4.config_check((STAND3_DUMP / "agent.yaml").read_text(), (STAND3_DUMP / "env.yaml").read_text())
        self.assertFalse(r["ok"])
        self.assertEqual(set(r["failed"]), {
            "lifting_object_stand4_x15", "lift_gate_0.02m_15deg", "success_stand4",
            "success_12steps_8deg_1N", "robot_a_overlay_usd", "robot_b_overlay_usd",
            "cube_contact_reporting", "cube_sensor_26_filters", "curriculum_terms"})

    def test_yaml_helpers(self):
        text = "a:\n  b:\n    c: 1\n    l:\n    - x\n    - 'y'\n  d: null\ne: []\n"
        self.assertEqual(s4.yaml_block(text, ("a", "b")), ["  b:", "    c: 1", "    l:", "    - x", "    - 'y'"])
        self.assertEqual(s4.yaml_value(s4.yaml_block(text, ("a", "b")), "c", 2), "1")
        self.assertEqual(s4.yaml_list(s4.yaml_block(text, ("a", "b")), "l", 2), ["x", "y"])
        self.assertEqual(s4.yaml_block(text, ("a", "d")), ["  d: null"])
        self.assertIsNone(s4.yaml_block(text, ("a", "zz")))
        self.assertIsNone(s4.yaml_value(None, "c", 2))


# ---------------------------------------------------------- smoke verdict


def _probe_ok(n=16):
    return {
        "status": "ok", "num_envs": n,
        "managers": {"reward_terms": ["reaching_coarse", "lifting_object", "placed"],
                     "termination_terms": ["time_out", "fallen", "success"],
                     "curriculum_terms": ["stage_lift"] + list(s4.CURRICULUM4_TERMS),
                     "lifting_object": {"func": s4.FUNC_LIFT4, "weight": 15.0},
                     "placed": {"func": s4.FUNC_PLACED, "weight": 5000.0},
                     "success": {"func": s4.FUNC_SUCCESS4}},
        "cfg_diff_keys": [k for k in s4.CFG_DIFF_ALLOWED if k != f"scene.{s4.CUBE_SENSOR}"]
                         + [f"scene.{s4.CUBE_SENSOR}.prim_path", f"scene.{s4.CUBE_SENSOR}.filter_prim_paths_expr"],
        "hand_colliders": {f"/World/envs/env_0/{r}/arm_{s}_hand_link/hand_collider":
                           {"valid": True, "collision_api": True}
                           for r in ("robot_a", "robot_b") for s in ("left", "right")},
        "usd_paths": {"stand4": {r: "/r/assets/cloth/" + s4.OVERLAY_NAME for r in ("robot_a", "robot_b")},
                      "stand3": {r: "/r/usd/berkeley_humanoid_lite.usd" for r in ("robot_a", "robot_b")}},
        "sensor": {"force_matrix_shape": [n, 1, 26, 3], "filter_count": 26},
        "cache": {"n_steps": 30, "keys_missing": [], "in_extras": True, "shapes_ok": True},
        "free_space": {"n_valid": n, "max_force_valid": 0.0, "all_released_valid": True},
        "overlap": {"n_valid": n, "n_hand_contact": 5, "inconsistent": 0, "cache_mismatch": 0},
        "seat": _seat_block(n),
    }


def _seat_records(groups, steps=None, fire_at=12):
    """What a correct env produces in the seat stage, one record per step.

    flat: seated, released, flat from step 1; hold = step; success, placed 5000
    and reset on step `fire_at`. After that reset the env is in a new episode and
    its records are deliberately garbage (they must be ignored). rolled90:
    seated and released at tilt 90 for every step, never fires. tilt10: falls
    flat over 3 steps (moving: not seated), then seated and flat, fires 12 steps
    later.
    """
    steps = s4.SMOKE_SEAT_STEPS if steps is None else steps
    hold_steps = sm.SEAT_HOLD_STEPS
    recs = []
    for k in range(1, steps + 1):
        r = {key: [] for key in s4.SEAT_RECORD_KEYS}
        for g in groups:
            if g == "flat":
                if k <= fire_at:
                    vals = dict(success=k >= hold_steps, reset=k == fire_at, hold=k, seated=True,
                                released=True, placed_now=True, tilt_deg=0.01,
                                placed_reward=5000.0 if k >= hold_steps else 0.0)
                else:      # new episode after the success reset: never read
                    vals = dict(success=True, reset=False, hold=0, seated=False, released=False,
                                placed_now=True, tilt_deg=float("nan"), placed_reward=0.0)
            elif g == "rolled90":
                vals = dict(success=False, reset=False, hold=0, seated=True, released=True,
                            placed_now=False, tilt_deg=90.0, placed_reward=0.0)
            else:          # tilt10
                h = max(0, k - 3)
                vals = dict(success=h >= hold_steps, reset=h >= hold_steps, hold=h, seated=k > 3,
                            released=True, placed_now=k > 3, tilt_deg=10.0 - 3.0 * k if k <= 3 else 0.2,
                            placed_reward=5000.0 if h >= hold_steps else 0.0)
            for key in s4.SEAT_RECORD_KEYS:
                r[key].append(vals[key])
        recs.append(r)
    return recs


def _seat_block(n=16, **kw):
    groups = [s4.seat_group(i) for i in range(n)]
    return {"steps": s4.SMOKE_SEAT_STEPS, "groups": groups, "records": _seat_records(groups, **kw)}


LOG_OK = "".join(f"\x1b[1m   Learning iteration {i}/20   \x1b[0m\n  Curriculum/roll_ok: 0.0100\n"
                 for i in range(20)) + "train.sh: reached 20 logged iterations\n"


class SmokeVerdictTests(unittest.TestCase):
    def v(self, probe=None, tags=None, log=LOG_OK, cfg=None, mutate=None):
        probe = _probe_ok() if probe is None else probe
        if mutate:
            mutate(probe)
        return s4.smoke_verdict(probe, set(s4.SMOKE_TAGS) if tags is None else tags, log,
                                {"ok": True} if cfg is None else cfg)

    def test_pass(self):
        r = self.v()
        self.assertEqual(r["verdict"], "PASS", r)
        self.assertEqual(set(r["clauses"]), set(s4.SMOKE_CLAUSES))

    def test_each_clause_fails(self):
        cases = {
            "A1_terms_in_managers": lambda p: p["managers"]["lifting_object"].update(
                func="bhl_robust.tasks.stand_mdp:gated_object_is_lifted"),
            "A2_cfg_diff_declared_only": lambda p: p["cfg_diff_keys"].append("rewards.lift_progress.weight"),
            "A3_hand_colliders": lambda p: next(iter(p["hand_colliders"].values())).update(collision_api=False),
            "A4_sensor_26_filters": lambda p: p["sensor"].update(force_matrix_shape=[16, 1, 2, 3]),
            "A5_step_cache_keys": lambda p: p["cache"].update(keys_missing=["released"]),
            "A6_free_cube_released": lambda p: p["free_space"].update(max_force_valid=1.0),
            "A7_hand_contact_sensed": lambda p: p["overlap"].update(n_hand_contact=0),
            "A8_success_chain_fires": lambda p: _set(p["seat"]["records"], 12, p["seat"]["groups"].index(
                "rolled90"), success=True, hold=12, placed_reward=5000.0, reset=True),
        }
        for clause, mutate in cases.items():
            r = self.v(mutate=mutate)
            self.assertEqual(r["verdict"], f"FAIL ({clause})", clause)

    def test_more_failures(self):
        self.assertIn("A2_cfg_diff_declared_only",
                      self.v(mutate=lambda p: p["cfg_diff_keys"].remove("curriculum.roll_ok"))["verdict"])
        self.assertIn("A3_hand_colliders", self.v(mutate=lambda p: p["usd_paths"]["stand3"].update(
            robot_a="/r/assets/cloth/" + s4.OVERLAY_NAME))["verdict"])
        self.assertIn("A6_free_cube_released",
                      self.v(mutate=lambda p: p["free_space"].update(n_valid=0, max_force_valid=None))["verdict"])
        self.assertIn("A7_hand_contact_sensed",
                      self.v(mutate=lambda p: p["overlap"].update(cache_mismatch=2))["verdict"])
        self.assertIn("B1_logged_tags", self.v(tags=set(s4.SMOKE_TAGS) - {"Curriculum/released"})["verdict"])
        self.assertIn("B2_training_ran", self.v(log=LOG_OK.replace("Learning iteration 19/20", ""))["verdict"])
        self.assertIn("B3_no_failure_markers",
                      self.v(log=LOG_OK + "train.sh: FAILED -- no training iteration\n")["verdict"])
        self.assertIn("B3_no_failure_markers", self.v(
            log=LOG_OK + "[Error] [omni.physx.plugin] PhysX error: GPU contact buffer overflow detected\n")["verdict"])
        self.assertIn("B3_no_failure_markers",
                      self.v(log=LOG_OK + "Traceback (most recent call last):\n")["verdict"])
        self.assertIn("B4_config_check", self.v(cfg={"ok": False, "failed": ["x"]})["verdict"])

    def test_incomplete_when_an_input_is_missing(self):
        self.assertTrue(s4.smoke_verdict(None, set(s4.SMOKE_TAGS), LOG_OK, {"ok": True})["verdict"]
                        .startswith("INCOMPLETE"))
        self.assertTrue(self.v(mutate=lambda p: p.update(status="error"))["verdict"].startswith("INCOMPLETE"))
        self.assertTrue(self.v(log=None)["verdict"].startswith("INCOMPLETE"))
        self.assertTrue(s4.smoke_verdict(_probe_ok(), set(s4.SMOKE_TAGS), LOG_OK, None)["verdict"]
                        .startswith("INCOMPLETE"))

    def test_console_parsing(self):
        self.assertEqual(s4.logged_iterations(LOG_OK), set(range(20)))
        self.assertEqual(s4.console_tags("\x1b[1m Curriculum/pinch_dist: nan\x1b[0m\n"
                                         "  Episode_Termination/success: 0.0000\n"),
                         {"Curriculum/pinch_dist", "Episode_Termination/success"})

    def test_required_tags_cover_the_spec(self):
        for t in ("Curriculum/lift_height", "Curriculum/upright_gate", "Curriculum/pinch_dist",
                  "Curriculum/cube_tilt_deg", "Curriculum/corner_clear", "Curriculum/released",
                  "Episode_Termination/success", "Episode_Reward/lifting_object"):
            self.assertIn(t, s4.SMOKE_TAGS)

    def test_cfg_diff_check(self):
        keys = _probe_ok()["cfg_diff_keys"]
        self.assertTrue(s4.cfg_diff_check(keys)["ok"])
        r = s4.cfg_diff_check(keys + ["scene.robot_a.init_state.pos"])
        self.assertEqual(r["unexpected"], ["scene.robot_a.init_state.pos"])
        r = s4.cfg_diff_check([k for k in keys if not k.startswith("terminations.success.func")])
        self.assertEqual(r["missing"], ["terminations.success.func"])
        # an entry covers its subtree, never a sibling with the same prefix
        self.assertFalse(s4.cfg_diff_check(keys + ["scene.cube_contact_extra.x"])["ok"])

    def test_a8_from_the_raw_seat_records(self):
        self.assertTrue(self.v()["clauses"]["A8_success_chain_fires"])
        for mutate in (lambda p: p.pop("seat"),
                       lambda p: p["seat"].update(steps=12),
                       lambda p: p["seat"].update(records=p["seat"]["records"][:12]),
                       lambda p: p["seat"].update(records="garbage"),
                       lambda p: p["seat"]["records"][3].pop("released"),
                       # a summary claiming ok is not trusted: the records decide
                       lambda p: p["seat"].update(summary={"ok": True},
                                                  records=_seat_records(p["seat"]["groups"], fire_at=99))):
            r = self.v(mutate=mutate)
            self.assertEqual(r["verdict"], "FAIL (A8_success_chain_fires)", r["clauses"])


# ------------------------------------------------------ seat stage (A8)


def _set(recs, step, env, **vals):
    for k, v in vals.items():
        recs[step - 1][k][env] = v


class SeatStageTests(unittest.TestCase):
    G = [s4.seat_group(i) for i in range(16)]

    def ev(self, recs, groups=None):
        return s4.seat_stage_eval(self.G if groups is None else groups, recs)

    def test_groups_poses_and_constants(self):
        self.assertGreaterEqual(s4.SMOKE_SEAT_STEPS, sm.SEAT_HOLD_STEPS + 1)
        self.assertEqual({g: self.G.count(g) for g in set(self.G)}, {"flat": 10, "rolled90": 4, "tilt10": 2})
        for g in ("flat", "rolled90"):
            self.assertEqual({s4.seat_side(i) for i in range(16) if self.G[i] == g}, {1.0, -1.0}, g)
        for i in range(16):
            (x, y, z), q = s4.seat_pose_local(i)
            w_, x_, y_, z_ = cols([q])
            p = P([x, y, z])
            self.assertAlmostEqual(abs(x), sm.DECK_CENTER)
            self.assertEqual(math.copysign(1.0, x), s4.seat_side(i))
            # resting: the lowest corner on the deck top, so no lift pay
            self.assertAlmostEqual(s4.lowest_corner_z(p, w_, x_, y_, z_).item(), sm.DECK_TOP, places=5)
            self.assertAlmostEqual(s4.support_clearance(p, w_, x_, y_, z_).item(), 0.0, places=5)
            self.assertEqual(s4.roll_proof_mask(p, w_, x_, y_, z_).tolist(), [False])
            tilt = s4.cube_tilt_deg(w_, x_, y_, z_)
            placed = s4.placed_mask(p, torch.zeros(1), tilt, torch.tensor([True])).item()
            g = self.G[i]
            if g == "flat":
                self.assertAlmostEqual(z, sm.DECK_SEATED_Z, places=6)
                self.assertAlmostEqual(tilt.item(), 0.0, places=3)
                self.assertTrue(placed)                            # it must fire
            elif g == "rolled90":
                self.assertAlmostEqual(z, sm.DECK_SEATED_Z, places=6)
                self.assertTrue(sm.seated_mask(p, torch.zeros(1)).item())   # seated ...
                self.assertAlmostEqual(tilt.item(), 90.0, places=3)        # ... but tilted
                self.assertFalse(placed)                           # the control
            else:
                h = math.radians(10.0)
                self.assertAlmostEqual(z, sm.DECK_TOP + H * (math.sin(h) + math.cos(h)), places=6)
                self.assertAlmostEqual(tilt.item(), 10.0, places=3)
                self.assertFalse(placed)
        # the probe writes through native_quat, never a literal
        self.assertIn("native_quat(q) for _, q in poses", INNER4.read_text())

    def test_pass_and_alive_logic(self):
        r = self.ev(_seat_records(self.G))
        self.assertTrue(r["ok"], r)
        self.assertEqual((r["n_flat"], r["n_flat_fired"], r["n_inconsistent"], r["n_fire_bad"],
                          r["n_control_fired"]), (10, 10, 0, 0, 0))
        self.assertEqual([r["fire_step"][i] for i in range(16) if self.G[i] == "flat"], [12] * 10)
        self.assertEqual([r["fire_step"][i] for i in range(16) if self.G[i] == "tilt10"], [15, 15])
        self.assertEqual(r["control_blocked_by_tilt_steps"], 4 * s4.SMOKE_SEAT_STEPS)
        ev = r["fire_events"][0]
        self.assertEqual((ev["step"], ev["hold"], ev["placed_reward"]), (12, 12, 5000.0))

    def test_an_earlier_reset_ends_the_evidence(self):
        recs = _seat_records(self.G)
        _set(recs, 5, 0, reset=True)                      # env 0 (flat) falls at step 5
        for k in range(6, 17):                            # then garbage, never read
            _set(recs, k, 0, success=True, hold=0, placed_now=False, placed_reward=0.0)
        r = self.ev(recs)
        self.assertTrue(r["ok"], r)
        self.assertEqual((r["n_flat_fired"], r["fire_step"][0], r["n_inconsistent"]), (9, None, 0))
        # every flat env reset before firing: nothing fired, FAIL
        recs = _seat_records(self.G)
        for i, g in enumerate(self.G):
            if g == "flat":
                _set(recs, 2, i, reset=True)
        r = self.ev(recs)
        self.assertEqual(r["n_flat_fired"], 0)
        self.assertFalse(r["ok"])

    def test_failures(self):
        flat0, rolled = 0, self.G.index("rolled90")
        cases = {
            "control fires": lambda recs: _set(recs, 12, rolled, success=True, hold=12,
                                               placed_reward=5000.0),
            "success before hold 12": lambda recs: _set(recs, 11, flat0, success=True,
                                                        placed_reward=5000.0),
            "no placed reward on the firing step": lambda recs: _set(recs, 12, flat0, placed_reward=0.0),
            "placed pays without success": lambda recs: _set(recs, 6, flat0, placed_reward=5000.0),
            "placed_now while not released": lambda recs: _set(recs, 6, flat0, released=False),
            "placed_now with tilt > 8": lambda recs: _set(recs, 6, flat0, tilt_deg=8.1),
            "placed_now with a NaN tilt": lambda recs: _set(recs, 6, flat0, tilt_deg=float("nan")),
            "hold 12 without success": lambda recs: _set(recs, 12, flat0, success=False,
                                                         placed_reward=0.0),
        }
        for name, f in cases.items():
            recs = _seat_records(self.G)
            f(recs)
            self.assertFalse(self.ev(recs)["ok"], name)
        # a firing that is not well-formed (the counter fires while the cube is not
        # seated this step; the counter itself is then out of step too)
        recs = _seat_records(self.G)
        _set(recs, 12, flat0, seated=False, placed_now=False)
        r = self.ev(recs)
        self.assertEqual((r["n_inconsistent"], r["n_fire_bad"]), (1, 1))
        self.assertFalse(r["ok"])
        # the hold counter must be _held's: previous + 1 while placed_now, else 0
        for k, h in ((6, 7), (1, 0), (3, 0)):
            recs = _seat_records(self.G)
            _set(recs, k, flat0, hold=h)
            self.assertFalse(self.ev(recs)["ok"], (k, h))
        recs = _seat_records(self.G)
        _set(recs, 2, self.G.index("rolled90"), hold=1)
        self.assertFalse(self.ev(recs)["ok"])
        # success that does not reset its env
        recs = _seat_records(self.G)
        _set(recs, 12, flat0, reset=False)
        self.assertFalse(self.ev(recs)["ok"])
        # a flat cube a robot still touches does not fire: not a failure by itself
        recs = _seat_records(self.G)
        _set(recs, 12, flat0, released=False, placed_now=False, hold=0, success=False,
             placed_reward=0.0)
        r = self.ev(recs)
        self.assertTrue(r["ok"] and r["n_flat_fired"] == 9)
        # the control is never seated and released: nothing shows the tilt clause blocking
        recs = _seat_records(self.G)
        for i, g in enumerate(self.G):
            if g == "rolled90":
                for k in range(1, 17):
                    _set(recs, k, i, released=False)
        r = self.ev(recs)
        self.assertEqual(r["control_blocked_by_tilt_steps"], 0)
        self.assertFalse(r["ok"])
        # too few steps, wrong shapes
        self.assertFalse(self.ev(_seat_records(self.G, steps=12))["ok"])
        recs = _seat_records(self.G)
        recs[0]["hold"].append(0)
        r = self.ev(recs)
        self.assertFalse(r["shapes_ok"] or r["ok"])
        self.assertFalse(self.ev([])["ok"])
        self.assertFalse(s4.seat_stage_eval([], _seat_records([]))["ok"])


# ------------------------------------------------ config class (AST / text)


def _s4_class_src() -> str:
    src = TASK_V2_CFG.read_text()
    return src[src.index("class CubeToShelfStand4Cfg"):src.index("CUBE_STAND4_VARIANTS = ")]


def _s4_terms():
    """(kind, func name, param keys) of every s4.* term the Stand4 class builds."""
    tree = ast.parse(TASK_V2_CFG.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CubeToShelfStand4Cfg")
    out = []
    for call in ast.walk(cls):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id in ("RewTerm", "CurrTerm", "DoneTerm")):
            continue
        kw = {k.arg: k.value for k in call.keywords}
        f = kw["func"]
        assert isinstance(f, ast.Attribute) and f.value.id == "s4", ast.dump(f)
        keys = set()
        p = kw.get("params")
        if isinstance(p, ast.Dict):
            for k, v in zip(p.keys, p.values):
                if k is None:
                    keys |= {"gate_free", "gate_std"}          # **gate
                else:
                    keys.add(k.value)
        out.append((call.func.id, f.attr, keys))
    return out


class CfgClassTests(unittest.TestCase):
    def test_terms_match_their_signatures(self):
        terms = _s4_terms()
        self.assertEqual({f for _, f, _ in terms}, {
            "stand4_object_is_lifted", "stand4_cube_placed", "pinch_distance_mean", "cube_tilt_mean",
            "corner_clearance_mean", "roll_ok_share", "lift_paid_share", "released_share",
            "robot_force_max_mean"})
        for kind, fname, keys in terms:
            sig = inspect.signature(getattr(s4, fname)).parameters
            args = list(sig)
            required = {a for a in args[(2 if kind == "CurrTerm" else 1):]
                        if sig[a].default is inspect.Parameter.empty}
            self.assertTrue(keys <= set(args), (fname, keys))
            self.assertTrue(required <= keys, (fname, required))

    def test_the_class_changes_only_what_it_declares(self):
        cls = _s4_class_src()
        self.assertIn("class CubeToShelfStand4Cfg(CubeToShelfStand3Cfg):", cls)
        self.assertIn("super().__post_init__()", cls)
        self.assertEqual(re.findall(r"\br\.(\w+) = ", cls), ["lifting_object"])
        self.assertEqual(re.findall(r"self\.terminations\.(\w+) = ", cls), ["success"])
        self.assertEqual(re.findall(r"\bc\.(\w+) = CurrTerm", cls), list(s4.DIAG_TERMS))
        self.assertEqual(re.findall(r"self\.scene\.([\w.]+) = ", cls), ["object.spawn"])
        self.assertIn("rc.spawn = rc.spawn.replace(usd_path=str(s4.HAND_COLLIDER_USD))", cls)
        self.assertIn('for name in ("robot_a", "robot_b"):', cls)
        self.assertIn("self.scene.object.spawn.replace(activate_contact_sensors=True)", cls)
        self.assertIn("setattr(self.scene, s4.CUBE_SENSOR, ContactSensorCfg(", cls)
        self.assertIn('prim_path="{ENV_REGEX_NS}/object"', cls)
        self.assertIn("filter_prim_paths_expr=list(s4.RELEASE_FILTER_EXPRS)", cls)
        self.assertIn('r.lifting_object.params["minimal_height"]', cls)
        self.assertIn("weight=r.lifting_object.weight", cls)
        self.assertIn('"hold_steps": stand.SEAT_HOLD_STEPS', cls)
        for banned in ("self.observations", "self.actions", "self.events", "self.rewards.",
                       "r.placed", "rewards.placed", ".weight = ", "episode_length_s",
                       "self.sim.dt =", "init_state", "self.curriculum.lift_height"):
            self.assertNotIn(banned, cls, banned)

    def test_variants_and_registration(self):
        src = TASK_V2_CFG.read_text()
        self.assertIn('CUBE_STAND4_VARIANTS = _variants(CubeToShelfStand4Cfg, "CubeToShelfStand4")', src)
        init = TASKS_INIT.read_text()
        self.assertEqual(init.count("CubeToShelfStand4"), 2)            # comment + the id
        self.assertEqual(init.count('id="TaskV2-BHL-CubeToShelfStand4-Blind-v0"'), 1)
        block = init[init.index("# Roll-proof lift cube (CubeToShelfStand4"):]
        block = block[:block.index("\n)\n") + 3]
        self.assertIn('"env_cfg_entry_point": task_v2_env_cfg.CUBE_STAND4_VARIANTS["blind"]', block)
        self.assertIn('"rsl_rl_cfg_entry_point": task_v2_env_cfg._STAND3_RUNNER', block)
        self.assertIn('entry_point="isaaclab.envs:ManagerBasedRLEnv"', block)
        self.assertNotIn("Depth", block)
        self.assertNotIn("Rgb", block)


class DefaultPathTests(unittest.TestCase):
    """Stand3 and every other task: byte-identical code paths."""

    def test_task_v2_cfg_is_append_only(self):
        data = TASK_V2_CFG.read_bytes()
        n, sha = TASK_V2_PRE_STAND4
        self.assertEqual(hashlib.sha256(data[:n]).hexdigest(), sha)
        self.assertTrue(data[n:].decode().startswith(STAND4_MARKER))

    def test_stand_mdp_untouched(self):
        self.assertEqual(hashlib.sha256(STAND_MDP.read_bytes()).hexdigest(), STAND_MDP_SHA)

    def test_stand3_registration_untouched(self):
        init = TASKS_INIT.read_text()
        self.assertIn("for _vis, _cls in task_v2_env_cfg.CUBE_STAND3_VARIANTS.items():", init)
        stand3 = init[init.index("for _vis, _cls in task_v2_env_cfg.CUBE_STAND3_VARIANTS.items():"):
                      init.index("# Roll-proof lift cube (CubeToShelfStand4")]
        self.assertIn('"rsl_rl_cfg_entry_point": task_v2_env_cfg._STAND3_RUNNER', stand3)
        self.assertNotIn("STAND4", stand3)

    def test_overlay_is_used_by_stand4_only(self):
        hits = []
        for path in (REPO / "src").rglob("*.py"):
            if "berkeley_humanoid_lite_hand_colliders" in path.read_text(errors="ignore"):
                hits.append(path.name)
        self.assertEqual(sorted(hits), ["cloth_sort_env_cfg.py", "stand4_mdp.py"])
        self.assertNotIn("HAND_COLLIDER_USD", TASK_V2_CFG.read_text()[:TASK_V2_PRE_STAND4[0]])


# ----------------------------------------------------------------- launchers


def _sbatch_flags(path: Path) -> dict:
    return dict(re.findall(r"^#SBATCH --([\w-]+)=?(.*)$", path.read_text(), re.M))


class TrainLauncherTests(unittest.TestCase):
    def test_header_carries_the_frozen_rule_verbatim(self):
        self.assertIn(_norm(RULE_TEXT), _header(TRAIN4))

    def test_kill_rule_block_copied_verbatim_from_stand3(self):
        b3, b4 = _kill_block(TRAIN3), _kill_block(TRAIN4)
        self.assertEqual(len(b3), 9)
        self.assertEqual(b4, b3)
        self.assertIn("sm.evaluate_kill3(sc)", TRAIN3.read_text())
        self.assertIn("s4.evaluate_kill(sc)", TRAIN4.read_text())
        self.assertIn('res["ready"] = last >= s4.sm.KILL_AT_ITER', TRAIN4.read_text())
        self.assertIn("KILL_AT=1000", TRAIN4.read_text())

    def test_resources_and_flags_are_stand3s(self):
        f3, f4 = _sbatch_flags(TRAIN3), _sbatch_flags(TRAIN4)
        self.assertEqual(f4.pop("job-name"), "v2-stand4-train")
        f3.pop("job-name")
        self.assertEqual(f4, f3)
        self.assertEqual((f4["array"], f4["time"], f4["mem"], f4["gres"], f4["cpus-per-task"]),
                         ("0-1", "40:00:00", "64G", "gpu:1", "8"))
        body = TRAIN4.read_text()
        for s in ('export BHL_STACK=v60 ENABLE_CAMERAS=1', 'export TASK="TaskV2-BHL-CubeToShelfStand4-Blind-v0"',
                  "MAX_ITER=${MAX_ITER:-8000} NUM_ENVS=${NUM_ENVS:-1024}",
                  "export SEED=${SLURM_ARRAY_TASK_ID:-${SEED:-0}}",
                  'case "$SEED" in 0|1) ;;', 'export RUN_NAME="v2-cubetoshelfstand4-blind-s${SEED}"',
                  "OUTDIR=$REPO/results/repo-gpu-20260923/stand4_2026-10-01",
                  ': > "$OVERRIDE_FILE"'):
            self.assertIn(s, body)

    def test_verdicts_from_json_never_exit_codes_and_no_overwrite(self):
        body = TRAIN4.read_text()
        for s in ("s4.evaluate_seed4(sc, killed=killed, bad_config=bad)", "s4.pair_verdict4(seeds)",
                  "s4.config_check(", 'open(out, "x")', "refusing to overwrite",
                  'result_s"${SEED}"_*.json', "refusing to start", 'jget "$js" decided',
                  "runner_check_s${SEED}_${JOBTAG}.json", 'mv "$tmp" "$js"'):
            self.assertIn(s, body)
        self.assertIn('[ "$killed" = 1 ] && exit 0', body)          # a rule kill is an outcome
        self.assertNotIn("Stand3 smoke", body)

    def test_header_states_the_rest(self):
        head = _header(TRAIN4)
        for s in ("iteration 7999", "INCOMPLETE", ">= 0.10", "tilt <= 8 deg",
                  "released", "1.0 N", "0.02 m", "15 deg", "DIFFERENT task", "never compared",
                  "afterok on gpu_v2_stand4_smoke.sbatch", "stand4 OK override it", "61 %",
                  "object_pos_a/b", "base_lin_vel_a/b", "LEARNED", "ORACLE", "MODIFIED",
                  "fresh runs", "RUNNER-CHECK", "never from exit codes", "26 collider-bearing",
                  # review 2026-10-02: Stand3's completeness check gates COMPLETE
                  "COMPLETE iff its events reach iteration 7999 (8000 iterations) AND every "
                  "Loss/value, success and time_out value in its last 200 iterations is finite "
                  "AND that window holds a Loss/value value",
                  "neither a PASS nor a NEGATIVE",
                  # review 2026-10-02: one tilt definition for both clauses
                  "both tilt clauses (lift <= 15 deg, success <= 8 deg) measure the angle between "
                  "the cube's OWN z axis and world up",
                  "the cube's own z axis vs world up, so a cube rolled 90 deg onto another face "
                  "reads 90 deg and never seats",
                  # review 2026-10-02: the smoke's bytes
                  "BYTES-CHECK", "--export=ALL,STAND4_SMOKE_JOB=<smoke>"):
            self.assertIn(s, head, s)
        self.assertNotIn("REPORTED, NOT GATING", head)

    def test_bytes_check_against_the_chained_smoke(self):
        body = TRAIN4.read_text()
        for s in ("check_smoke_bytes() {", 'check_smoke_bytes "${STAND4_SMOKE_JOB:-}"',
                  "sha256sum -c --strict -", 'verdict_${job}.json', 'sha256_${job}.txt',
                  '!= "PASS"', "SMOKE_OUT=$REPO/results/repo-gpu-20260923/stand4_2026-10-01-smoke",
                  'rm -f "$STAMP"', "BYTES-CHECK $RUN_NAME: FAIL -- refusing to start"):
            self.assertIn(s, body, s)
        # before anything trains: the check precedes the boot gate and the training
        self.assertLess(body.index('check_smoke_bytes "${STAND4_SMOKE_JOB:-}"'),
                        body.index("\nv60_boot_gate\n"))
        self.assertLess(body.index('check_smoke_bytes "${STAND4_SMOKE_JOB:-}"'),
                        body.index('bhl_exec "$REPO/slurm/inner/train.sh" &'))
        # the smoke hashes exactly the required set + the three excluded files
        required = re.search(r'^BYTES_REQUIRED="([^"]+)"$', body, re.M).group(1).split()
        excluded = re.search(r"grep -v -E '  \(([^)]+)\)\$'", body).group(1).replace("\\.", ".").split("|")
        self.assertEqual(set(excluded), {"src/bhl_robust/tasks/__init__.py",
                                         "slurm/repo20260923/gpu_v2_stand4_smoke.sbatch",
                                         "slurm/repo20260923/inner_v2_stand4_smoke.sh"})
        smoke = SMOKE4.read_text()
        cmd = smoke[smoke.index("sha256sum src/bhl_robust/tasks/stand4_mdp.py"):]
        cmd = cmd[:cmd.index('> "$HASHES"')]
        hashed = [w for w in cmd.replace("\\", " ").replace(")", " ").split() if "/" in w]
        self.assertEqual(sorted(hashed), sorted(required + list(excluded)))
        self.assertEqual(len(required), 6)


class SmokeLauncherTests(unittest.TestCase):
    def test_smoke_flags(self):
        f = _sbatch_flags(SMOKE4)
        self.assertIn("smoke", f["job-name"])
        hh, mm, ss = (int(v) for v in f["time"].split(":"))
        self.assertLessEqual(hh * 3600 + mm * 60 + ss, 3600)
        self.assertEqual((f["gres"], f["mem"]), ("gpu:1", "64G"))

    def test_smoke_outputs_seed_and_exit(self):
        body = SMOKE4.read_text()
        self.assertIn("STAND4_OUT=$REPO/results/repo-gpu-20260923/stand4_2026-10-01-smoke", body)
        self.assertIn('RUN_NAME="v2-cubetoshelfstand4-blind-s${SEED}-j${JOB}-smoke"', body)
        self.assertIn("SEED=100", body)
        self.assertIn('case "$SEED" in 0|1)', body)
        self.assertIn("NUM_ENVS=1024 MAX_ITER=20", body)
        self.assertIn('sys.exit(0 if v["verdict"] == "PASS" else 1)', body)
        self.assertIn("exit $rc", body)
        self.assertIn("s4.smoke_verdict(probe, tags, train_log, cfg)", body)
        self.assertIn("s4.config_check(", body)
        self.assertIn("refusing to overwrite", body)
        self.assertIn("sha256sum", body)
        self.assertEqual(s4.SMOKE_ITERS, 20)
        head = _header(SMOKE4)
        for clause in s4.SMOKE_CLAUSES:
            self.assertIn(clause.split("_")[0], head)
        for s in ("SCRIPTED", "Never a result", "(16, 1, 26, 3)", "afterok"):
            self.assertIn(s, head)

    def test_inner_probe(self):
        body = INNER4.read_text()
        for s in ("refusing to overwrite", "hand_collider", "PhysicsCollisionAPI", "s4.cfg_diff_check",
                  "write_root_pose_to_sim_index", "arm_left_hand_link", "u.reset_buf", "json.dump(res",
                  "STAND4-PROBE", "timeout 1500",
                  # stage 8 (review 2026-10-02): the success chain
                  'res["stage"] = "seat"', "s4.seat_pose_local(i)", "range(s4.SMOKE_SEAT_STEPS)",
                  'tm.get_term("success")', "rm._step_reward[:, i_placed]",
                  'list(rm.active_terms).index("placed")', "for k in s4.SEAT_RECORD_KEYS",
                  "s4.seat_stage_eval(groups, records)"):
            self.assertIn(s, body, s)
        # a fresh reset precedes the seat teleport, which precedes its steps
        i = body.index('res["stage"] = "seat"')
        self.assertLess(i, body.index("env.reset()", i))
        self.assertLess(body.index("env.reset()", i), body.index("torch.cat([seat_pos, seat_quat]", i))
        self.assertLess(body.index("torch.cat([seat_pos, seat_quat]", i),
                        body.index("range(s4.SMOKE_SEAT_STEPS)", i))
        # the host verdict recomputes A8 from the raw records
        self.assertIn('"seat", "traceback"', SMOKE4.read_text())
        self.assertIn("s4.seat_stage_eval(probe[\"seat\"].get(\"groups\", [])", SMOKE4.read_text())
        # the JSON is written before env.close() / simulation_app.close()
        self.assertLess(body.index("json.dump(res, f"), body.rindex("env.close()"))
        self.assertLess(body.index("json.dump(res, f"), body.rindex("simulation_app.close()"))


if __name__ == "__main__":
    unittest.main()
