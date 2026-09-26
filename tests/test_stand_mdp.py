"""CubeToShelfStand2 terms and predeclared rules, without Isaac Lab.

`stand_mdp` is loaded from its file path rather than through `bhl_robust.tasks`,
whose `__init__` registers every gym id and needs a running simulator.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import math
import sys
import types
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
STAND_MDP = REPO / "src/bhl_robust/tasks/stand_mdp.py"
TASK_V2_CFG = REPO / "src/bhl_robust/tasks/task_v2_env_cfg.py"
TRAIN_SBATCH = REPO / "slurm/repo20260923/gpu_v2_stand2_train.sbatch"
V1_RUN = (REPO / "external/Berkeley-Humanoid-Lite/logs/rsl_rl/task_v2/"
          "2026-09-24_09-54-29_v2-cubetoshelfstand-blind-s0")


def _load():
    spec = importlib.util.spec_from_file_location("stand_mdp_under_test", STAND_MDP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sm = _load()


class ImportTests(unittest.TestCase):
    def test_imports_without_isaaclab(self):
        had = {k for k in sys.modules if k == "isaaclab" or k.startswith("isaaclab.")}
        _load()
        now = {k for k in sys.modules if k == "isaaclab" or k.startswith("isaaclab.")}
        self.assertEqual(now - had, set())


class KernelTests(unittest.TestCase):
    def k(self, t):
        return float(sm.upright_kernel(torch.tensor([t]))[0])

    def test_flat_inside_free_band(self):
        for t in (0.0, 0.05, sm.GATE_FREE):
            self.assertAlmostEqual(self.k(t), 1.0, places=6)

    def test_anchors_from_v1_data(self):
        # Ordinary standing sway (v1 standing window: tilt_rms 0.16-0.20 rad) keeps >= 0.8.
        self.assertGreaterEqual(self.k(0.20), 0.80)
        # Tipping is unpaid well before the 0.78 rad termination.
        self.assertLessEqual(self.k(0.50), 0.10)
        self.assertLess(self.k(sm.FALL_LIMIT), 0.01)

    def test_monotone_and_bounded(self):
        t = torch.linspace(0.0, 3.2, 400)
        g = sm.upright_kernel(t)
        self.assertTrue(bool((g[1:] <= g[:-1] + 1e-7).all()))
        self.assertTrue(bool(((g >= 0) & (g <= 1)).all()))

    def test_pair_is_product(self):
        a = torch.tensor([0.0, 0.3, 0.6, 0.0])
        b = torch.tensor([0.0, 0.0, 0.3, 0.9])
        g = sm.pair_upright_gate(a, b)
        self.assertTrue(torch.allclose(g, sm.upright_kernel(a) * sm.upright_kernel(b)))
        # One robot tipping voids the pair's pay.
        self.assertLess(float(g[3]), 0.01)


class ShareTests(unittest.TestCase):
    def test_share_over_limit(self):
        tilt = torch.tensor([0.1, 0.9, 0.8, 0.2])
        self.assertAlmostEqual(sm.share_over_limit(tilt, None), 0.5)
        self.assertAlmostEqual(sm.share_over_limit(tilt, torch.tensor([1, 2])), 1.0)
        self.assertAlmostEqual(sm.share_over_limit(tilt, [0, 3]), 0.0)
        self.assertEqual(sm.share_over_limit(tilt, torch.tensor([], dtype=torch.long)), 0.0)


class _TermMgr:
    def __init__(self, terms):
        self.terms = terms

    def get_term(self, name):
        return self.terms[name]


class FallPenaltyTests(unittest.TestCase):
    def test_fires_on_fall_not_on_success(self):
        env = types.SimpleNamespace(termination_manager=_TermMgr({
            "fallen": torch.tensor([True, False, False]),
            "success": torch.tensor([False, True, False]),
            "time_out": torch.tensor([False, False, True]),
        }))
        out = sm.fall_penalty(env, "fallen")
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.tolist(), [1.0, 0.0, 0.0])

    def test_prices(self):
        fall = -sm.FALL_PENALTY_WEIGHT * sm.STEP_DT
        self.assertAlmostEqual(fall, 20.0)
        # anchored on a full episode of still_alive
        self.assertAlmostEqual(fall, 1.0 * sm.STEP_DT * sm.EPISODE_STEPS)
        # above v1's break-even (0.4 already paid + 14.2 extra) ...
        self.assertAlmostEqual(sm.V1_BREAK_EVEN_EXTRA, 14.16, places=2)
        self.assertGreater(fall, sm.V1_FALL + sm.V1_BREAK_EVEN_EXTRA)
        # ... and well below the point where standing-still beats standing-and-reaching
        self.assertLess(0.5 * fall, sm.V1_STAND_SHAPING_INCOME)
        # v1's own price ratio, the diagnosis in one line
        self.assertGreater(sm.V1_LIFT_STEP, sm.V1_FALL)
        # a success is not charged the fall price: placed (200 x dt = 8) stays positive
        self.assertGreater(200.0 * sm.STEP_DT, 0.0)


class _FakeCoop:
    """Stands in for coop_lift_mdp: fixed term values, records the cache write."""

    def __init__(self, tilt_a, tilt_b):
        self.tilt = {"robot_a": tilt_a, "robot_b": tilt_b}
        self.lifted_calls = []

    def _tilt_from_quat(self, robot):
        return self.tilt[robot]

    def constellation_reach(self, env, std, robot_a_cfg, robot_b_cfg):
        env._bhl_pinch_d = torch.tensor([0.05, 0.05])
        return torch.tensor([0.8, 0.8])

    def opposing_clamp(self, env, robot_a_cfg, robot_b_cfg):
        return torch.tensor([0.5, 0.5])

    def object_lift_progress(self, env):
        return torch.tensor([1.0, 1.0])

    def object_is_lifted(self, env, minimal_height):
        self.lifted_calls.append(minimal_height)
        return torch.tensor([1.0, 1.0])


class _FakeV2:
    def carry_progress(self, env, target_x, std=0.8):
        return torch.tensor([0.4, 0.4])


class GatedWrapperTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load()
        self.coop = _FakeCoop(torch.tensor([0.0, 0.6]), torch.tensor([0.0, 0.0]))
        self.mod._coop = lambda: self.coop
        self.mod._v2 = lambda: _FakeV2()
        self.env = types.SimpleNamespace(scene={"robot_a": "robot_a", "robot_b": "robot_b"})
        self.g = sm.pair_upright_gate(torch.tensor([0.0, 0.6]), torch.tensor([0.0, 0.0]))

    def test_gate_applied_once_per_term(self):
        m, e, g = self.mod, self.env, self.g
        cases = [
            (m.gated_constellation_reach(e, 0.4, "a", "b"), 0.8),
            (m.gated_opposing_clamp(e, "a", "b"), 0.5),
            (m.gated_lift_progress(e), 1.0),
            (m.gated_object_is_lifted(e, minimal_height=0.04), 1.0),
            (m.gated_carry_progress(e, target_x=1.2), 0.4),
        ]
        for out, base in cases:
            self.assertTrue(torch.allclose(out, base * g), (out, base * g))
        # upright env pays in full, tipping env nearly nothing
        self.assertAlmostEqual(float(g[0]), 1.0)
        self.assertLess(float(g[1]), 0.1)

    def test_pinch_cache_left_ungated(self):
        self.mod.gated_constellation_reach(self.env, 0.12, "a", "b")
        self.assertTrue(torch.equal(self.env._bhl_pinch_d, torch.tensor([0.05, 0.05])))

    def test_minimal_height_forwarded(self):
        self.mod.gated_object_is_lifted(self.env, minimal_height=0.07)
        self.assertEqual(self.coop.lifted_calls, [0.07])

    def test_fell_share_and_gate_mean(self):
        self.assertAlmostEqual(self.mod.fell_share(self.env, slice(None), "robot_a", 0.5), 0.5)
        self.assertAlmostEqual(self.mod.fell_share(self.env, torch.tensor([0]), "robot_a"), 0.0)
        self.assertAlmostEqual(self.mod.upright_gate_mean(self.env, None),
                               float(self.g.mean()), places=6)


def _stand2_terms():
    """(kind, func name, param keys) for every term CubeToShelfStand2Cfg sets."""
    tree = ast.parse(TASK_V2_CFG.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CubeToShelfStand2Cfg")
    gate_keys = {"gate_free", "gate_std"}
    out = []
    for call in ast.walk(cls):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id in ("RewTerm", "CurrTerm")):
            continue
        kw = {k.arg: k.value for k in call.keywords}
        func = kw["func"]
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id == "stand"):
            continue
        p = kw.get("params")
        keys = set()
        if isinstance(p, ast.Dict):
            for k in p.keys:
                if k is None:
                    keys |= gate_keys           # **gate
                else:
                    keys.add(k.value)
        elif isinstance(p, ast.Call) and getattr(p.func, "id", "") == "dict":
            keys |= gate_keys                   # dict(gate)
        out.append((call.func.id, func.attr, keys))
    return out


class CfgContractTests(unittest.TestCase):
    """The reward/curriculum managers' own check, applied statically.

    `ManagerBase._resolve_common_term_cfg` requires
    set(args[min_argc:]) == set(params) | set(args with defaults), with
    min_argc 1 for rewards and 2 for curriculum terms.
    """

    def test_every_stand2_term_matches_its_signature(self):
        terms = _stand2_terms()
        names = {f for _, f, _ in terms}
        self.assertTrue({"gated_constellation_reach", "gated_opposing_clamp", "gated_lift_progress",
                         "gated_object_is_lifted", "gated_carry_progress", "fall_penalty",
                         "upright_gate_mean", "fell_share"} <= names, names)
        for kind, fname, keys in terms:
            sig = inspect.signature(getattr(sm, fname)).parameters
            args = list(sig)
            with_def = [a for a in args if sig[a].default is not inspect.Parameter.empty]
            without = [a for a in args if sig[a].default is inspect.Parameter.empty]
            min_argc = 1 if kind == "RewTerm" else 2
            ordered = without + with_def
            self.assertEqual(set(ordered[min_argc:]), keys | set(with_def), (fname, keys))
            self.assertTrue(keys <= set(args), (fname, keys))

    def test_lifting_object_keeps_curriculum_param_name(self):
        self.assertIn("minimal_height", inspect.signature(sm.gated_object_is_lifted).parameters)

    def test_v1_classes_untouched_and_variants_bound(self):
        src = TASK_V2_CFG.read_text()
        self.assertIn('CUBE_STAND_VARIANTS = _variants(CubeToShelfStandCfg, "CubeToShelfStand")', src)
        self.assertIn('CUBE_STAND2_VARIANTS = _variants(CubeToShelfStand2Cfg, "CubeToShelfStand2")', src)
        self.assertIn("class CubeToShelfStand2Cfg(CubeToShelfStandCfg):", src)
        # placed keeps v1's ungated function; only its weight is raised so placing is
        # worth more than hovering (2026-09-26 review); termination_penalty is removed
        cls_src = src[src.index("class CubeToShelfStand2Cfg"):]
        self.assertNotIn("r.placed = ", cls_src)
        self.assertIn("r.placed.weight = stand.PLACED_WEIGHT", cls_src)
        self.assertGreater(sm.PLACED_WEIGHT * sm.STEP_DT, 93.0)   # > the ~93 a full episode of hovering is worth
        self.assertIn("r.termination_penalty = None", cls_src)


class RuleTests(unittest.TestCase):
    def test_v1_at_its_kill_point_is_killed(self):
        # v1, iterations 900-1000: len 177.5, time_out 0.053, success 0.
        self.assertEqual(sm.kill_verdict(177.5, 0.053, 0.0)["verdict"], "KILL")

    def test_v1_standing_phase_continues(self):
        # v1, iterations 300-400: len 326, time_out 0.447.
        self.assertEqual(sm.kill_verdict(326.1, 0.447, 0.0)["verdict"], "CONTINUE")

    def test_cannot_stand_is_killed(self):
        v = sm.kill_verdict(60.0, 0.30, 0.0)
        self.assertEqual(v["verdict"], "KILL")
        self.assertEqual(len(v["reasons"]), 1)

    def test_nan_is_a_kill(self):
        self.assertEqual(sm.kill_verdict(math.nan, math.nan, math.nan)["verdict"], "KILL")

    def test_result(self):
        a = sm.seed_result(success=0.12, time_out=0.5)
        b = sm.seed_result(success=0.02, time_out=0.7)
        self.assertTrue(a["placed"] and a["stands"])
        self.assertFalse(b["placed"])
        self.assertEqual(sm.pair_result([a, b]), sm.POSITIVE_LABEL)
        self.assertEqual(sm.pair_result([b, b]), "NEGATIVE (stands on 2/2 seeds)")
        killed = sm.seed_result(success=0.5, time_out=0.5, killed=True)
        self.assertFalse(killed["placed"])
        self.assertEqual(sm.pair_result([killed, sm.seed_result(math.nan, math.nan)]),
                         "NEGATIVE (stands on 0/2 seeds)")

    def test_windows(self):
        sc = {sm.TAG_LEN: [(i, float(i)) for i in range(0, 1001)],
              sm.TAG_TIMEOUT: [(i, 0.5) for i in range(0, 1001)],
              sm.TAG_SUCCESS: [(i, 0.0) for i in range(0, 1001)],
              sm.TAG_FALLEN: [(i, 0.5) for i in range(0, 1001)]}
        m = sm.window_means(sc, [sm.TAG_LEN], 1000, 100)
        self.assertAlmostEqual(m[sm.TAG_LEN], sum(range(901, 1001)) / 100)
        k = sm.evaluate_kill(sc)
        self.assertEqual(k["verdict"], "CONTINUE")
        r = sm.evaluate_seed(sc)
        self.assertEqual(r["last_iter"], 1000)
        self.assertFalse(r["placed"])
        self.assertTrue(r["stands"])


@unittest.skipUnless(V1_RUN.is_dir() and importlib.util.find_spec("tensorboard"),
                     "v1 run dir or tensorboard not available")
class V1ReplayTest(unittest.TestCase):
    """The kill rule, run on the real v1 run, kills it at model_1000."""

    def test_v1_events(self):
        sc = sm.read_scalars(str(V1_RUN))
        k = sm.evaluate_kill(sc)
        self.assertEqual(k["verdict"], "KILL", k)
        self.assertGreater(k["ep_len"], sm.KILL_MIN_EP_LEN)   # length alone would not have
        self.assertLess(k["nonfall"], sm.KILL_MIN_NONFALL)
        # and would have let it continue at its standing peak
        self.assertEqual(sm.evaluate_kill(sc, at_iter=400)["verdict"], "CONTINUE")


@unittest.skipUnless(TRAIN_SBATCH.is_file(), "launcher not written")
class LauncherHeaderTest(unittest.TestCase):
    def test_header_states_the_rules(self):
        head = TRAIN_SBATCH.read_text()
        for s in ("model_1000", "< 100 steps", "time_out + success", "< 0.10",
                  ">= 0.10", ">= 0.50", "DIFFERENT, EASIER task", "never compared"):
            self.assertIn(s, head, s)


if __name__ == "__main__":
    unittest.main()
