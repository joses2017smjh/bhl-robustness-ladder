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



# ============================================================ Stand3 (side deck)
TRAIN3_SBATCH = REPO / "slurm/repo20260923/gpu_v2_stand3_train.sbatch"
COOP_LIFT_CFG = REPO / "src/bhl_robust/tasks/coop_lift_env_cfg.py"
SMOKE3_INNER = REPO / "slurm/repo20260923/inner_v2_stand3_smoke.sh"


def _cfg_terms(class_name):
    """(kind, func name, param keys) for every stand.* term a cfg class sets."""
    tree = ast.parse(TASK_V2_CFG.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    gate_keys = {"gate_free", "gate_std"}
    out = []
    for call in ast.walk(cls):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id in ("RewTerm", "CurrTerm", "DoneTerm")):
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
                    keys |= gate_keys
                else:
                    keys.add(k.value)
        elif isinstance(p, ast.Call) and getattr(p.func, "id", "") == "dict":
            keys |= gate_keys
        out.append((call.func.id, func.attr, keys))
    return out


class Stand3GeometryTests(unittest.TestCase):
    def test_deck_clear_of_spawned_cube(self):
        # far face of a spawned cube at the jitter limit, plus clearance
        self.assertGreaterEqual(sm.DECK_EDGE - (sm.CUBE_HALF + sm.OBJ_X_JITTER), 0.02 - 1e-9)

    def test_lip_stops_a_slide_and_the_curriculum_clears_it(self):
        # the lip stops a flat slide; it does not stop a push-and-tip, which the
        # docstrings say and the tilt diagnostics record
        self.assertGreater(sm.DECK_TOP, sm.PLINTH_TOP)
        self.assertGreater(sm.PLINTH_TOP + 0.04, sm.DECK_TOP)      # first lift stage clears it
        self.assertGreater(sm.PLINTH_TOP + sm.STAND3_LIFT_MAX, sm.DECK_TOP)
        self.assertLess(sm.STAND3_LIFT_MAX, 0.19)                  # Stand2 climbed to +0.19

    def test_shift_is_stated_honestly(self):
        # the required shift exceeds the measured feet-planted envelope; the
        # docstrings and the launcher say so rather than "within reach"
        self.assertAlmostEqual(sm.REQUIRED_SHIFT, 0.19, places=6)
        self.assertGreater(sm.REQUIRED_SHIFT, sm.MEASURED_SHIFT_ARMS_TWIST)
        self.assertLess(sm.REQUIRED_SHIFT, 1.2 / 6.0 + 1e-9)       # 6x shorter than Stand2
        head = TRAIN3_SBATCH.read_text().split("set -euo")[0]
        self.assertIn("NOT", head)
        self.assertIn("mechanism not asserted", head)
        self.assertNotIn("lift required", TASK_V2_CFG.read_text())

    def test_income_weights_match_the_configs(self):
        # Every positive non-terminal term Stand3 runs: the base coop-lift
        # RewardsCfg's positive weights (read from source) plus deck_progress.
        tree = ast.parse(COOP_LIFT_CFG.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RewardsCfg")
        pos = {}
        for st in cls.body:
            if isinstance(st, ast.Assign) and isinstance(st.value, ast.Call):
                w = {k.arg: k.value for k in st.value.keywords}.get("weight")
                v = ast.literal_eval(w)
                if v > 0:
                    pos[st.targets[0].id] = v
        base = {k: v for k, v in sm.STAND3_INCOME_WEIGHTS.items() if k != "deck_progress"}
        self.assertEqual(pos, base)
        # the only runtime weight writer on these terms (stage_lift_on_pinch)
        # restores exactly the base weights, never more
        mdp = ast.parse((COOP_LIFT_CFG.parent / "coop_lift_mdp.py").read_text())
        fn = next(n for n in mdp.body if isinstance(n, ast.FunctionDef) and n.name == "stage_lift_on_pinch")
        defaults = dict(zip([a.arg for a in fn.args.args][-len(fn.args.defaults):],
                            [ast.literal_eval(d) for d in fn.args.defaults]))
        self.assertEqual((defaults["progress_weight"], defaults["bonus_weight"]),
                         (base["lift_progress"], base["lifting_object"]))
        # no ancestor in task_v2_env_cfg raises a weight by assignment; the one
        # `.weight =` is Stand2's placed, which Stand3 replaces
        import re
        self.assertEqual(re.findall(r"\.weight = .*", TASK_V2_CFG.read_text()),
                         [".weight = stand.PLACED_WEIGHT"])
        self.assertEqual(sm.STAND3_INCOME_WEIGHTS["deck_progress"], sm.DECK_PROGRESS_WEIGHT)
        # Stand3 adds no other positive term than deck_progress and placed, and
        # keeps Stand2's lift terms (it does not reassign them)
        src = TASK_V2_CFG.read_text()
        s3 = src[src.index("class CubeToShelfStand3Cfg"):src.index("CUBE_STAND3_VARIANTS = ")]
        self.assertNotIn("r.lift_progress =", s3)
        self.assertNotIn("r.lifting_object =", s3)
        self.assertNotIn("deck_edge", s3)
        self.assertIn("weight=stand.STAND3_PLACED_WEIGHT", s3)
        self.assertNotIn("weight=stand.PLACED_WEIGHT", s3)
        runner = src[src.index("class TaskV2Stand3PPORunnerCfg"):src.index("_STAND3_RUNNER = TaskV2Stand3PPORunnerCfg")]
        self.assertIn(f"gamma={sm.STAND3_GAMMA}", runner)

    def test_placing_beats_every_hover(self):
        # 2026-09-27 review: `success` terminates, so the bonus must beat the
        # discounted continuation of the RICHEST non-terminal state, not just
        # the over-deck hover. With the lift terms ungated, that is lifted +
        # pinched over a deck centre: 24.5 weight = 0.98/step <= 98 discounted.
        per_step = sum(sm.STAND3_INCOME_WEIGHTS.values()) * sm.STEP_DT
        self.assertAlmostEqual(sum(sm.STAND3_INCOME_WEIGHTS.values()), 24.5)
        self.assertAlmostEqual(sm.STAND3_HOVER_BOUND, per_step / (1 - sm.STAND3_GAMMA))
        self.assertAlmostEqual(sm.STAND3_HOVER_BOUND, 98.0)
        bonus = sm.STAND3_PLACED_WEIGHT * sm.STEP_DT
        self.assertAlmostEqual(bonus, 200.0)
        self.assertGreaterEqual(bonus, 2.0 * sm.STAND3_HOVER_BOUND)
        # the plinth hover Stand2 s0 learned (deck_progress at x = 0) is below
        # the bound, so it is covered too
        k0 = float(sm.deck_progress_kernel(torch.tensor([[0.0, 0.0, 0.62]]))[0])
        plinth = (per_step - sm.DECK_PROGRESS_WEIGHT * sm.STEP_DT * (1 - k0)) / (1 - sm.STAND3_GAMMA)
        self.assertLess(plinth, sm.STAND3_HOVER_BOUND)
        self.assertGreater(bonus, 2.0 * plinth)
        # lowering + 12-step hold forfeits lifting_object (seated centre is below
        # every lift threshold) and at most all of lift_progress: small vs the bonus
        self.assertLess(sm.DECK_SEATED_Z, sm.STAND3_CUBE_Z + 0.04)
        forfeit = (15.0 + 2.0) * sm.STEP_DT * 25
        self.assertLess(forfeit + sm.STAND3_HOVER_BOUND, bonus)
        # and the bonus is itself discounted over those ~25 steps (review round 2):
        # 200 x 0.99^25 = 156 still clears the 98 bound by > 50
        self.assertGreater(bonus * sm.STAND3_GAMMA ** 25 - sm.STAND3_HOVER_BOUND, 50.0)
        # Stand2's weight is untouched
        self.assertEqual(sm.PLACED_WEIGHT, 2500.0)


class Stand3KillRuleTests(unittest.TestCase):
    """2026-09-27 review: success ends the episode, so a seed that places quickly has
    short episodes; Stand2's length clause would kill it and record it as not placed."""

    def test_placing_seed_is_not_killed_for_short_episodes(self):
        v = sm.kill_verdict3(70.0, 0.0, 0.9)
        self.assertEqual(v["verdict"], "CONTINUE")
        self.assertTrue(v["length_clause_skipped"])
        self.assertEqual(sm.kill_verdict(70.0, 0.0, 0.9)["verdict"], "KILL")   # Stand2's rule would

    def test_falling_seed_is_still_killed(self):
        self.assertEqual(sm.kill_verdict3(70.0, 0.05, 0.0)["verdict"], "KILL")
        v = sm.kill_verdict3(70.0, 0.50, 0.05)             # success under the bar: length clause applies
        self.assertEqual(v["verdict"], "KILL")
        self.assertFalse(v["length_clause_skipped"])
        self.assertEqual(sm.kill_verdict3(326.1, 0.02, 0.05)["verdict"], "KILL")   # nonfall clause kept
        self.assertEqual(sm.kill_verdict3(math.nan, math.nan, math.nan)["verdict"], "KILL")

    def test_matches_stand2_below_the_success_bar(self):
        for args in ((177.5, 0.053, 0.0), (326.1, 0.447, 0.0), (60.0, 0.30, 0.0), (150.0, 0.2, 0.09)):
            self.assertEqual(sm.kill_verdict3(*args)["verdict"], sm.kill_verdict(*args)["verdict"])

    def test_launcher_uses_the_stand3_kill_rule(self):
        text = (REPO / "slurm/repo20260923/gpu_v2_stand3_train.sbatch").read_text()
        self.assertIn("sm.evaluate_kill3(sc)", text)
        self.assertNotIn("sm.evaluate_kill(sc)", text)
        self.assertIn("kill_verdict3", text)


class Stand3KernelTests(unittest.TestCase):
    def p(self, x, y=0.0, z=None):
        return torch.tensor([[x, y, sm.DECK_SEATED_Z if z is None else z]])

    def seated(self, x, y=0.0, z=None, v=0.0):
        return bool(sm.seated_mask(self.p(x, y, z), torch.tensor([v]))[0])

    def test_seated_mask(self):
        self.assertTrue(self.seated(sm.DECK_CENTER))
        self.assertTrue(self.seated(-sm.DECK_CENTER))              # either deck
        self.assertFalse(self.seated(0.0, z=sm.STAND3_CUBE_Z))     # spawn
        self.assertFalse(self.seated(sm.DECK_CENTER, z=sm.DECK_SEATED_Z + 0.05))   # hovering
        # release is not checked: still, 1.5 cm above the deck counts (SEATED_NOTE)
        self.assertTrue(self.seated(sm.DECK_CENTER, z=sm.DECK_SEATED_Z + 0.015))
        self.assertIn("hands may still be on it", sm.SEATED_NOTE)
        self.assertFalse(self.seated(sm.DECK_CENTER, v=0.2))       # moving
        self.assertFalse(self.seated(sm.DECK_CENTER, y=0.15))      # off the side
        self.assertFalse(self.seated(sm.DECK_CENTER, z=0.14))      # on the floor
        self.assertFalse(self.seated(sm.DECK_EDGE))                # COM on the edge
        self.assertFalse(self.seated(0.50))                        # past the far margin

    def test_over_deck(self):
        self.assertFalse(hasattr(sm, "off_deck"))                  # lift terms not position-gated
        p = torch.tensor([[0.3, 0.0, 0.6], [-0.3, 0.0, 0.14], [0.0, 0.0, 0.55]])
        self.assertEqual(sm.over_deck_mask(p).tolist(), [True, False, False])

    def test_deck_progress(self):
        xs = torch.linspace(0.0, sm.DECK_CENTER, 50)
        p = torch.stack([xs, torch.zeros(50), torch.full((50,), 0.55)], 1)
        k = sm.deck_progress_kernel(p)
        self.assertTrue(bool((k[1:] > k[:-1]).all()))              # rises toward the deck
        self.assertAlmostEqual(float(k[-1]), 1.0, places=5)
        self.assertTrue(torch.allclose(k, sm.deck_progress_kernel(p * torch.tensor([-1.0, 1, 1]))))
        self.assertEqual(float(sm.deck_progress_kernel(torch.tensor([[0.3, 0.0, 0.14]]))[0]), 0.0)
        # steeper at the plinth than Stand2's carry (0.68 per metre at x = 0)
        slope = sm.DECK_PROGRESS_WEIGHT * (1 - math.tanh(sm.DECK_CENTER / sm.DECK_PROGRESS_STD) ** 2) \
            / sm.DECK_PROGRESS_STD
        self.assertGreater(slope, 3.0 * 0.68)

    def test_clipped_action_rate(self):
        a, b = torch.randn(8, 44), torch.randn(8, 44)
        self.assertTrue(torch.allclose(sm.clipped_action_rate(a, b),
                                       torch.sum((a - b) ** 2, dim=-1)))
        huge = torch.full((2, 44), 1e30)
        r = sm.clipped_action_rate(huge, -huge)
        self.assertTrue(bool(torch.isfinite(r).all()))
        self.assertAlmostEqual(float(r[0]), 44 * (2 * sm.ACTION_CLIP) ** 2, places=1)

    def test_body_z_tilt(self):
        s2 = math.sqrt(0.5)
        w, x, y, z = (torch.tensor(v) for v in ([1.0, s2, s2, math.cos(0.2)],
                                                 [0.0, s2, 0.0, math.sin(0.2)],
                                                 [0.0, 0.0, 0.0, 0.0],
                                                 [0.0, 0.0, s2, 0.0]))
        t = sm.body_z_tilt_deg(w, x, y, z)
        # upright; tipped 90 deg about x (a cube tipped onto a side face);
        # yawed 90 deg (still upright); tilted 22.9 deg about x
        self.assertTrue(torch.allclose(t, torch.tensor([0.0, 90.0, 0.0, 22.918]), atol=1e-3))
        self.assertTrue(bool(t[1] > sm.TILT_TIPPED_DEG) and bool(t[3] < sm.TILT_TIPPED_DEG))

    def test_clip_bounds(self):
        self.assertGreaterEqual(0.25 * sm.ACTION_CLIP, 2.5)        # past every joint range
        self.assertGreater(sm.TARGET_CLIP, sm.MAX_JOINT_LIMIT)
        self.assertGreater(sm.OBS_CLIP, 2.0 * 10)                  # far above obs noise
        self.assertEqual((sm.STAND3_ENTROPY_COEF, sm.STAND3_STD_TYPE), (0.001, "log"))


class _FakeV2Stand3(_FakeV2):
    def __init__(self, p, speed):
        self.p, self.speed = p, speed

    def _obj_local(self, env, name="object"):
        return self.p

    def _held(self, env, key, now, steps):
        buf = getattr(env, key, torch.zeros_like(now, dtype=torch.long))
        buf = torch.where(now, buf + 1, torch.zeros_like(buf))
        setattr(env, key, buf)
        return buf >= steps


class _FakeCoopStand3(_FakeCoop):
    def __init__(self, tilt_a, tilt_b, vel):
        super().__init__(tilt_a, tilt_b)
        self.vel = vel

    def _t(self, v):
        return v


class Stand3WrapperTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load()
        p = torch.tensor([[0.0, 0.0, 0.62], [sm.DECK_CENTER, 0.0, sm.DECK_SEATED_Z]])
        vel = torch.zeros(2, 3)
        self.coop = _FakeCoopStand3(torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]), vel)
        self.v2 = _FakeV2Stand3(p, vel.norm(dim=-1))
        self.mod._coop = lambda: self.coop
        self.mod._v2 = lambda: self.v2
        obj = types.SimpleNamespace(data=types.SimpleNamespace(root_lin_vel_w=vel))
        self.env = types.SimpleNamespace(scene={"robot_a": "robot_a", "robot_b": "robot_b",
                                                "object": obj})

    def test_lift_terms_not_gated_by_position(self):
        # Stand3 inherits Stand2's gated_* lift terms: same pay over a deck as
        # over the plinth (2026-09-27 review), so crossing the deck edge costs nothing
        m, e = self.mod, self.env
        self.assertFalse(hasattr(m, "deck_gated_object_is_lifted"))
        self.assertFalse(hasattr(m, "deck_gated_lift_progress"))
        self.assertEqual(m.gated_object_is_lifted(e, minimal_height=0.04).tolist(), [1.0, 1.0])

    def test_success_holds_once_per_call(self):
        m, e = self.mod, self.env
        outs = [m.cube_on_side_deck(e, hold_steps=3).tolist() for _ in range(3)]
        self.assertEqual(outs, [[False, False], [False, False], [False, True]])

    def test_success_bonus_reads_the_termination(self):
        env = types.SimpleNamespace(termination_manager=_TermMgr({
            "success": torch.tensor([False, True])}))
        self.assertEqual(self.mod.success_bonus(env).tolist(), [0.0, 1.0])

    def test_diagnostics(self):
        self.assertAlmostEqual(self.mod.over_deck_share(self.env, None), 0.5)
        self.assertAlmostEqual(self.mod.cube_abs_x_mean(self.env, None), sm.DECK_CENTER / 2, places=6)

    def test_action_rate_wrapper(self):
        am = types.SimpleNamespace(action=torch.full((1, 44), 1e20), prev_action=torch.zeros(1, 44))
        r = self.mod.action_rate_clipped_l2(types.SimpleNamespace(action_manager=am))
        self.assertAlmostEqual(float(r[0]), 44 * sm.ACTION_CLIP ** 2, places=1)


MJCF = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/mjcf_cache/mjcf_humanoid/berkeley_humanoid_lite.xml")


@unittest.skipUnless(MJCF.is_file(), "MJCF cache not available")
class TargetClipTest(unittest.TestCase):
    def test_target_clip_never_binds_inside_the_joint_limits(self):
        import re
        joints = re.findall(r'<joint [^>]*type="hinge"[^>]*>', MJCF.read_text())
        self.assertEqual(len(joints), 22)
        lims = [abs(float(v)) for j in joints
                for a, b in re.findall(r'range="([-0-9.e]+) ([-0-9.e]+)"', j) for v in (a, b)]
        self.assertAlmostEqual(max(lims), sm.MAX_JOINT_LIMIT, places=2)
        self.assertLess(max(lims), sm.TARGET_CLIP)


class Stand3CfgContractTests(unittest.TestCase):
    def test_every_stand3_term_matches_its_signature(self):
        terms = _cfg_terms("CubeToShelfStand3Cfg")
        names = {f for _, f, _ in terms}
        self.assertEqual(names, {"gated_deck_progress", "cube_on_side_deck", "success_bonus",
                                 "action_rate_clipped_l2", "over_deck_share", "cube_abs_x_mean",
                                 "cube_tilted_share", "tipped_over_deck_share"})
        for kind, fname, keys in terms:
            sig = inspect.signature(getattr(sm, fname)).parameters
            args = list(sig)
            with_def = [a for a in args if sig[a].default is not inspect.Parameter.empty]
            without = [a for a in args if sig[a].default is inspect.Parameter.empty]
            min_argc = 2 if kind == "CurrTerm" else 1
            ordered = without + with_def
            self.assertEqual(set(ordered[min_argc:]), keys | set(with_def), (fname, keys))
            self.assertTrue(keys <= set(args), (fname, keys))
        # the lift terms Stand3 inherits are Stand2's
        s2 = {f for _, f, _ in _cfg_terms("CubeToShelfStand2Cfg")}
        self.assertTrue({"gated_lift_progress", "gated_object_is_lifted"} <= s2)

    def test_stand3_is_additive(self):
        src = TASK_V2_CFG.read_text()
        self.assertIn("class CubeToShelfStand3Cfg(CubeToShelfStand2Cfg):", src)
        self.assertIn('CUBE_STAND3_VARIANTS = _variants(CubeToShelfStand3Cfg, "CubeToShelfStand3")', src)
        self.assertIn("class TaskV2Stand3PPORunnerCfg(TaskV2PPORunnerCfg):", src)
        self.assertIn("_STAND3_RUNNER = _V2_RUNNER", src)          # v51 fallback
        head = src[:src.index("# ------------------------------------------------- side-deck cube, Stand3")]
        self.assertNotIn("Stand3", head)                           # appended only
        self.assertEqual(src.count("_V2_RUNNER = "), 1)            # never reassigned
        base = src[src.index("class TaskV2PPORunnerCfg"):src.index("_V2_RUNNER = TaskV2PPORunnerCfg")]
        self.assertIn("entropy_coef=0.005", base)
        # Stand2's constants, which its published result was trained on
        self.assertEqual((sm.FALL_PENALTY_WEIGHT, sm.PLACED_WEIGHT, sm.GATE_FREE, sm.GATE_STD),
                         (-500.0, 2500.0, 0.10, 0.25))

    def test_runner_uses_the_declared_constants(self):
        src = TASK_V2_CFG.read_text()
        cls = src[src.index("class TaskV2Stand3PPORunnerCfg"):src.index("_STAND3_RUNNER = TaskV2Stand3PPORunnerCfg")]
        self.assertIn("entropy_coef=stand.STAND3_ENTROPY_COEF", cls)
        self.assertIn("std_type=stand.STAND3_STD_TYPE", cls)
        self.assertNotIn("clip_actions", cls.split('"""', 2)[2])   # dead in scripts/train.py


def _sc(last, success=0.0, time_out=0.6, over=0.2, value=10.0, nan_at=None):
    it = range(0, last + 1)
    val = [(i, (math.nan if i == nan_at else value)) for i in it]
    return {sm.TAG_LEN: [(i, 400.0) for i in it], sm.TAG_TIMEOUT: [(i, time_out) for i in it],
            sm.TAG_SUCCESS: [(i, success) for i in it], sm.TAG_FALLEN: [(i, 1 - time_out) for i in it],
            sm.TAG_OVER_DECK: [(i, over) for i in it], sm.TAG_ABS_X: [(i, 0.1) for i in it],
            sm.TAG_VALUE: val}


class Stand3RuleTests(unittest.TestCase):
    def test_complete_positive(self):
        r = sm.evaluate_seed3(_sc(7999, success=0.2))
        self.assertTrue(r["complete"] and r["placed"] and r["stands"] and r["shifts"] and r["stable"])
        self.assertTrue(sm.pair_result3([r, sm.evaluate_seed3(_sc(4468))]).startswith(sm.POSITIVE_LABEL3))

    def test_short_run_is_incomplete_never_a_result(self):
        r = sm.evaluate_seed3(_sc(4468, success=0.5))              # Stand2 s1's last iteration
        self.assertFalse(r["complete"] or r["placed"] or r["decided"])
        neg = sm.evaluate_seed3(_sc(7999))
        self.assertTrue(sm.pair_result3([neg, r]).startswith("INCOMPLETE"))
        self.assertTrue(sm.pair_result3([neg, neg]).startswith("NEGATIVE"))

    def test_nan_in_window_is_incomplete(self):
        r = sm.evaluate_seed3(_sc(7999, success=0.5, nan_at=7990))
        self.assertFalse(r["complete"] or r["placed"])
        self.assertFalse(r["stable"])

    def test_transient_spike_is_reported_not_voiding(self):
        sc = _sc(7999, success=0.0)
        sc[sm.TAG_VALUE][4357] = (4357, math.inf)
        sc[sm.TAG_VALUE][4207] = (4207, 2.8e6)
        r = sm.evaluate_seed3(sc)
        self.assertTrue(r["complete"])
        self.assertEqual(r["unstable_iters"], 2)
        self.assertFalse(r["stable"])

    def test_killed_is_decided_not_placed(self):
        k = sm.evaluate_seed3(_sc(1016, success=0.5), killed=True)
        self.assertTrue(k["decided"] and not k["placed"])
        self.assertTrue(sm.pair_result3([k, sm.evaluate_seed3(_sc(7999))]).startswith("NEGATIVE"))

    def test_budget_is_8000(self):
        self.assertEqual(sm.STAND3_MAX_ITER, 8000)
        self.assertFalse(sm.evaluate_seed3(_sc(7998))["complete"])


@unittest.skipUnless(TRAIN3_SBATCH.is_file(), "launcher not written")
class Stand3LauncherTests(unittest.TestCase):
    def test_header_states_the_rules(self):
        head = TRAIN3_SBATCH.read_text().split("set -euo pipefail")[0]
        for s in ("model_1000", "< 100 steps", "(time_out + success) < 0.10", "iteration 7999",
                  "finite", "INCOMPLETE", ">= 0.10 on at least one", ">= 0.50",
                  "Curriculum/over_deck >= 0.10", "Loss/value > 1000", "DIFFERENT, EASIER task",
                  "never compared", "fresh runs", "0.119", "0.19", "200 units",
                  "98-unit", "NOT gated", "hands may", "Loss/value", "1e-5"):
            self.assertIn(s, head, s)

    def test_runner_check_matches_the_constants(self):
        body = TRAIN3_SBATCH.read_text()
        self.assertIn(f"entropy_coef: {sm.STAND3_ENTROPY_COEF}".replace(".", r"\."), body)
        self.assertIn(f"std_type: {sm.STAND3_STD_TYPE}$", body)
        self.assertIn("TaskV2-BHL-CubeToShelfStand3-Blind-v0", body)
        self.assertIn("stand3_2026-09-27", body)
        w = f"{sm.STAND3_PLACED_WEIGHT}".replace(".", r"\.")
        self.assertIn(f"weight: {w}$", body)                       # placed weight checked at start
        self.assertIn(r"stand_mdp:success_bonus$", body)
        self.assertIn("evaluate_seed3", body)
        self.assertIn("pair_result3", body)

    def test_smoke_checks_terms(self):
        body = SMOKE3_INNER.read_text()
        for t in ("deck_progress", "action_rate_clipped", "over_deck", "cube_tilted",
                  "tipped_over_deck", "carry", "object_xy", "termination_penalty", "194 322 578"):
            self.assertIn(t, body)


# ---- 2026-09-28 payload-quaternion fix (appended). The body-z tilt diagnostic's
# definition is unchanged; with the spawn fixed (coop_lift_env_cfg._object through
# native_quat) it reads 0 at spawn and 90 on a side face, as designed. Stand3's
# recorded cube_tilted / tipped_over_deck (1.0 from iteration 0) came from the
# flipped spawn and stay uninformative; every later v60 run is a NEW configuration.

class _QuatOrder:
    """Pin the stack's quaternion order for bhl_robust.quat_order (native_quat, unpack_wxyz)."""

    def __init__(self, order):
        if str(REPO / "src") not in sys.path:
            sys.path.insert(0, str(REPO / "src"))
        from unittest import mock
        from bhl_robust import quat_order as qo
        self.qo = qo
        self.patches = [mock.patch.object(qo, "quat_order", lambda: order),
                        mock.patch.object(qo, "_CACHED_ORDER", order)]

    def __enter__(self):
        for p in self.patches:
            p.start()
        return self.qo

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()


class CubeTiltQuatFixTests(unittest.TestCase):
    S2 = math.sqrt(0.5)

    def setUp(self):
        self.mod = _load()
        self.mod._coop = lambda: _FakeCoopStand3(torch.zeros(1), torch.zeros(1), torch.zeros(1, 3))

    def env(self, stored):
        obj = types.SimpleNamespace(data=types.SimpleNamespace(root_quat_w=stored))
        return types.SimpleNamespace(scene={"object": obj})

    def tilt(self, rows_wxyz, order):
        with _QuatOrder(order) as qo:
            stored = torch.tensor([qo.reorder(r, order) for r in rows_wxyz])
            return self.mod._cube_tilt_deg(self.env(stored)).tolist()

    def test_zero_at_the_fixed_spawn_and_90_after_a_roll_v60(self):
        with _QuatOrder("xyzw") as qo:
            spawn = torch.tensor([qo.native_quat((1.0, 0.0, 0.0, 0.0))])     # what the cfg now writes
            self.assertEqual(spawn.tolist(), [[0.0, 0.0, 0.0, 1.0]])
            self.assertAlmostEqual(self.mod._cube_tilt_deg(self.env(spawn)).tolist()[0], 0.0, places=4)
        s = self.S2
        got = self.tilt([(s, s, 0.0, 0.0), (s, 0.0, s, 0.0), (s, 0.0, 0.0, s)], "xyzw")
        for v, want in zip(got, (90.0, 90.0, 0.0)):        # roll x, pitch y, yaw z
            self.assertAlmostEqual(v, want, places=3)

    def test_old_raw_spawn_read_180_on_v60(self):
        # the pre-fix literal stored as-is on v60: why cube_tilted was 1.0 from iteration 0
        with _QuatOrder("xyzw"):
            old = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            self.assertAlmostEqual(self.mod._cube_tilt_deg(self.env(old)).tolist()[0], 180.0, places=3)
            self.assertEqual(self.mod.cube_tilted_share(self.env(old), None), 1.0)

    def test_share_at_spawn_is_zero_v60(self):
        s = self.S2
        with _QuatOrder("xyzw") as qo:
            rows = [qo.native_quat((1.0, 0.0, 0.0, 0.0))] * 3 + [qo.reorder((s, s, 0.0, 0.0), "xyzw")]
            self.assertEqual(self.mod.cube_tilted_share(self.env(torch.tensor(rows[:3])), None), 0.0)
            self.assertEqual(self.mod.cube_tilted_share(self.env(torch.tensor(rows)), None), 0.25)

    def test_v51_unchanged(self):
        s = self.S2
        got = self.tilt([(1.0, 0.0, 0.0, 0.0), (s, s, 0.0, 0.0), (s, 0.0, 0.0, s)], "wxyz")
        for v, want in zip(got, (0.0, 90.0, 0.0)):
            self.assertAlmostEqual(v, want, places=3)
        with _QuatOrder("wxyz") as qo:
            self.assertEqual(qo.native_quat((1.0, 0.0, 0.0, 0.0)), (1.0, 0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
