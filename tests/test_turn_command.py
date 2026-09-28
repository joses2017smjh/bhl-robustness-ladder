"""TurnCmd command mix (src/bhl_robust/tasks/turn_command.py) without Isaac Sim.

`import isaaclab` bootstraps Kit (EULA prompt) on a login node, so the Isaac
modules turn_command.py needs are stubbed in sys.modules -- EXCEPT upstream's
real `velocity_command.py` (UniformVelocityCommand: `_resample_command` and
`_update_command`) and the real `configclass`, which are loaded from the
installed isaaclab source. The subclass is thus exercised against the actual
upstream heading/standing logic it has to coexist with. Skips if the installed
source is not found.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import types
import unittest
from dataclasses import MISSING
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder")
for _p in (Path(__file__).resolve().parents[1], REPO):
    if (_p / "src/bhl_robust/tasks/turn_command.py").exists():
        REPO = _p
        break
ISAACLAB = Path(sys.prefix) / "lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab"
if not ISAACLAB.exists():
    ISAACLAB = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab")


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _wrap_to_pi(angles):          # verbatim logic of isaaclab.utils.math.wrap_to_pi
    wrapped = (angles + torch.pi) % (2 * torch.pi)
    return torch.where((wrapped == 0) & (angles > 0), torch.pi, wrapped - torch.pi)


class _CommandTerm:               # the parts of isaaclab.managers.CommandTerm the subclass touches
    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env
        self.num_envs, self.device = env.num_envs, env.device
        self.metrics = {}


def _install():
    saved = {k: v for k, v in sys.modules.items() if k == "isaaclab" or k.startswith("isaaclab.")}
    for k in saved:
        del sys.modules[k]
    _module("isaaclab", __path__=[])
    utils = _module("isaaclab.utils", __path__=[str(ISAACLAB / "utils")])
    configclass = importlib.import_module("isaaclab.utils.configclass").configclass   # real one
    utils.configclass = configclass
    _module("isaaclab.utils.math", wrap_to_pi=_wrap_to_pi)
    utils.math = sys.modules["isaaclab.utils.math"]
    _module("isaaclab.assets", Articulation=object)
    _module("isaaclab.managers", CommandTerm=_CommandTerm)
    _module("isaaclab.markers", VisualizationMarkers=object)
    _module("isaaclab.envs", __path__=[])
    _module("isaaclab.envs.mdp", __path__=[])
    _module("isaaclab.envs.mdp.commands", __path__=[])
    vc = _load("isaaclab.envs.mdp.commands.velocity_command", ISAACLAB / "envs/mdp/commands/velocity_command.py")

    @configclass
    class UniformVelocityCommandCfg:          # field-for-field stand-in (no markers)
        class_type: type = vc.UniformVelocityCommand
        resampling_time_range: tuple[float, float] = MISSING
        debug_vis: bool = False
        asset_name: str = MISSING
        heading_command: bool = False
        heading_control_stiffness: float = 1.0
        rel_standing_envs: float = 0.0
        rel_heading_envs: float = 1.0

        @configclass
        class Ranges:
            lin_vel_x: tuple[float, float] = MISSING
            lin_vel_y: tuple[float, float] = MISSING
            ang_vel_z: tuple[float, float] = MISSING
            heading: tuple[float, float] | None = None

        ranges: Ranges = MISSING

    _module("isaaclab.envs.mdp.commands.commands_cfg", UniformVelocityCommandCfg=UniformVelocityCommandCfg)
    tc = _load("turn_command_under_test", REPO / "src/bhl_robust/tasks/turn_command.py")
    return vc, UniformVelocityCommandCfg, tc


@unittest.skipUnless((ISAACLAB / "envs/mdp/commands/velocity_command.py").exists(), "installed isaaclab source not found")
class TurnCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vc, cls.BaseCfg, cls.tc = _install()

    def cfg(self, **kw):
        # Upstream humanoid CommandsCfg.base_velocity, field for field, as HumanoidTurnCmdCfg copies it.
        ranges = self.BaseCfg.Ranges(lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.5, 1.5),
                                     heading=(-math.pi, math.pi))
        args = dict(resampling_time_range=(10.0, 10.0), debug_vis=False, asset_name="robot", heading_command=True,
                    heading_control_stiffness=0.5, rel_standing_envs=0.02, rel_heading_envs=1.0, ranges=ranges)
        args.update(kw)
        return self.tc.TurnMixVelocityCommandCfg(**args)

    def make(self, n, seed=0, cfg=None, cls=None):
        torch.manual_seed(seed)
        self.heading = torch.empty(n).uniform_(-math.pi, math.pi)
        robot = SimpleNamespace(data=SimpleNamespace(heading_w=self.heading))
        env = SimpleNamespace(num_envs=n, device="cpu", scene={"robot": robot})
        cls = cls or self.tc.TurnMixVelocityCommand
        cmd = cls(cfg or self.cfg(), env)
        cmd._resample_command(torch.arange(n))
        return cmd

    # --- pure functions --------------------------------------------------
    def test_draw_modes_boundaries(self):
        tc = self.tc
        u = torch.tensor([0.0, 0.249, 0.25, 0.499, 0.5, 0.999])
        self.assertEqual(tc.draw_modes(u, 0.25, 0.25).tolist(),
                         [tc.PURE_TURN, tc.PURE_TURN, tc.DIRECT, tc.DIRECT, tc.UPSTREAM, tc.UPSTREAM])
        self.assertTrue((tc.draw_modes(u, 0.0, 0.0) == tc.UPSTREAM).all())

    def test_mix_rows_and_no_input_mutation(self):
        tc = self.tc
        vel = torch.tensor([[0.8, -0.3, 1.2], [0.8, -0.3, 1.2], [0.8, -0.3, 1.2], [0.8, -0.3, 1.2]])
        head = torch.ones(4, dtype=torch.bool)
        modes = torch.tensor([tc.PURE_TURN, tc.PURE_TURN, tc.DIRECT, tc.UPSTREAM])
        u = torch.tensor([[0.1, 0.0, 0.5], [0.9, 1.0, 0.5], [0.0, 1.0, 0.5], [0.3, 0.3, 0.3]])
        v2, h2 = tc.mix_turn_commands(vel, head, modes, u, pure_abs=(0.3, 1.0), direct_x=(-0.5, 0.5),
                                      direct_y=(-0.25, 0.25), direct_wz=(-1.0, 1.0))
        torch.testing.assert_close(v2, torch.tensor([[0.0, 0.0, -0.3], [0.0, 0.0, 1.0], [-0.5, 0.25, 0.0],
                                                     [0.8, -0.3, 1.2]]))
        self.assertEqual(h2.tolist(), [False, False, False, True])
        self.assertTrue((vel == 0.8).any() and head.all())          # inputs untouched

    def test_check_mix_rejects_bad_configs(self):
        tc = self.tc
        ok = dict(pure_abs=(0.3, 1.0), direct_x=(-0.5, 0.5), direct_y=(-0.25, 0.25), direct_wz=(-1.0, 1.0))
        tc.check_mix(0.25, 0.25, *ok.values())
        for bad in ((0.7, 0.4, *ok.values()), (-0.1, 0.2, *ok.values()),
                    (0.25, 0.25, (0.0, 1.0), *list(ok.values())[1:]),
                    (0.25, 0.25, (0.3, 1.0), (0.5, -0.5), (-0.25, 0.25), (-1.0, 1.0))):
            with self.assertRaises(ValueError):
                tc.check_mix(*bad)

    # --- the subclass against upstream's real _resample/_update ------------
    def test_cfg_under_real_configclass(self):
        c = self.cfg()
        self.assertIs(c.class_type, self.tc.TurnMixVelocityCommand)
        self.assertEqual((c.rel_pure_turn_envs, c.rel_direct_envs), (0.25, 0.25))
        self.assertEqual(tuple(c.pure_turn_ang_vel_abs), (0.3, 1.0))
        self.assertEqual((c.heading_control_stiffness, c.rel_standing_envs, c.rel_heading_envs), (0.5, 0.02, 1.0))

    def test_mode_fractions_and_heading_exclusion(self):
        tc, n = self.tc, 40000
        cmd = self.make(n)
        before = cmd.vel_command_b.clone()
        cmd._update_command()
        v, mode, stand = cmd.vel_command_b, cmd.turn_mode, cmd.is_standing_env
        for m, target in ((tc.PURE_TURN, 0.25), (tc.DIRECT, 0.25), (tc.UPSTREAM, 0.50)):
            self.assertAlmostEqual(float((mode == m).float().mean()), target, delta=0.015)
        self.assertAlmostEqual(float(stand.float().mean()), 0.02, delta=0.005)
        # standing envs are zero whatever their mode (upstream's rule, unchanged)
        self.assertTrue((v[stand] == 0).all())
        # heading control only on UPSTREAM envs
        self.assertTrue((cmd.is_heading_env == (mode == tc.UPSTREAM)).all())
        live = ~stand
        pure, direct, up = live & (mode == tc.PURE_TURN), live & (mode == tc.DIRECT), live & (mode == tc.UPSTREAM)
        self.assertTrue((v[pure, :2] == 0).all())
        self.assertTrue(((v[pure, 2].abs() >= 0.3) & (v[pure, 2].abs() <= 1.0)).all())
        self.assertAlmostEqual(float((v[pure, 2] > 0).float().mean()), 0.5, delta=0.03)
        self.assertTrue((v[pure | direct] == before[pure | direct]).all())       # _update_command left them alone
        self.assertTrue(((v[direct, 0].abs() <= 0.5) & (v[direct, 1].abs() <= 0.25) & (v[direct, 2].abs() <= 1.0)).all())
        err = _wrap_to_pi(cmd.heading_target[up] - self.heading[up])
        torch.testing.assert_close(v[up, 2], torch.clip(0.5 * err, -1.5, 1.5))
        self.assertAlmostEqual(float(cmd.metrics["pure_turn_env"].mean()), float((mode == tc.PURE_TURN).float().mean()))

    def test_pure_turn_is_sustained_across_steps(self):
        tc, n = self.tc, 4000
        cmd = self.make(n, seed=1)
        cmd._update_command()
        pure = (cmd.turn_mode == tc.PURE_TURN) & ~cmd.is_standing_env
        wz0 = cmd.vel_command_b[pure, 2].clone()
        for k in range(50):                              # robot turns; heading error changes every step
            self.heading += 0.05
            cmd._update_command()
            self.assertTrue((cmd.vel_command_b[pure, 2] == wz0).all())
            self.assertTrue((cmd.vel_command_b[pure, :2] == 0).all())

    def test_upstream_generator_almost_never_commands_a_pure_turn(self):
        # The diagnosis this arm acts on: with the mix off (== upstream), (0, 0, |wz| >= 0.3) does not occur.
        n = 40000
        def pure_share(cmd):
            cmd._update_command()
            v = cmd.vel_command_b
            return float(((v[:, :2].norm(dim=1) < 0.05) & (v[:, 2].abs() >= 0.3)).float().mean())
        up = self.make(n, seed=2, cfg=self.cfg(rel_pure_turn_envs=0.0, rel_direct_envs=0.0))
        mixed = self.make(n, seed=2)
        self.assertLess(pure_share(up), 0.005)
        self.assertAlmostEqual(pure_share(mixed), 0.25 * 0.98, delta=0.015)

    def test_partial_resample_touches_only_those_envs(self):
        tc, n = self.tc, 1000
        cmd = self.make(n, seed=3)
        v0, h0, m0 = cmd.vel_command_b.clone(), cmd.is_heading_env.clone(), cmd.turn_mode.clone()
        ids = torch.arange(0, n, 7)
        cmd._resample_command(ids)
        other = torch.ones(n, dtype=torch.bool)
        other[ids] = False
        self.assertTrue((cmd.vel_command_b[other] == v0[other]).all())
        self.assertTrue((cmd.is_heading_env[other] == h0[other]).all() and (cmd.turn_mode[other] == m0[other]).all())
        self.assertTrue((cmd.is_heading_env[ids] == (cmd.turn_mode[ids] == tc.UPSTREAM)).all())
        cmd._resample_command(torch.tensor([], dtype=torch.long))      # no-op, no error

    def test_command_dimension_unchanged(self):
        cmd = self.make(16)
        self.assertEqual(tuple(cmd.command.shape), (16, 3))              # -> 75-obs layout unchanged

    def test_bad_cfg_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            self.make(8, cfg=self.cfg(rel_pure_turn_envs=0.8, rel_direct_envs=0.4))


@unittest.skipUnless((ISAACLAB / "envs/mdp/commands/velocity_command.py").exists(), "installed isaaclab source not found")
class TurnRestCommandTests(unittest.TestCase):
    """Arm TurnRest (2026-09-27): rest-then-turn envs on top of the TurnCmd mix."""

    DT = 0.04

    @classmethod
    def setUpClass(cls):
        cls.vc, cls.BaseCfg, cls.tc = _install()

    def cfg(self, **kw):
        ranges = self.BaseCfg.Ranges(lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.5, 1.5),
                                     heading=(-math.pi, math.pi))
        args = dict(resampling_time_range=(10.0, 10.0), debug_vis=False, asset_name="robot", heading_command=True,
                    heading_control_stiffness=0.5, rel_standing_envs=0.02, rel_heading_envs=1.0, ranges=ranges)
        args.update(kw)
        return self.tc.TurnRestMixVelocityCommandCfg(**args)

    def make(self, n, seed=0, cfg=None):
        torch.manual_seed(seed)
        self.heading = torch.empty(n).uniform_(-math.pi, math.pi)
        robot = SimpleNamespace(data=SimpleNamespace(heading_w=self.heading))
        env = SimpleNamespace(num_envs=n, device="cpu", scene={"robot": robot}, step_dt=self.DT)
        cmd = self.tc.TurnRestMixVelocityCommand(cfg or self.cfg(), env)
        cmd._resample_command(torch.arange(n))
        return cmd

    def test_draw_modes_rest_boundaries(self):
        tc = self.tc
        u = torch.tensor([0.0, 0.149, 0.15, 0.399, 0.40, 0.599, 0.60, 0.999])
        self.assertEqual(tc.draw_modes_rest(u, 0.15, 0.25, 0.20).tolist(),
                         [tc.PURE_TURN, tc.PURE_TURN, tc.REST_TURN, tc.REST_TURN, tc.DIRECT, tc.DIRECT,
                          tc.UPSTREAM, tc.UPSTREAM])
        self.assertTrue((tc.draw_modes_rest(u, 0.0, 0.0, 0.0) == tc.UPSTREAM).all())

    def test_rest_turn_update_pure_function(self):
        tc = self.tc
        vel = torch.tensor([[0.5, 0.1, 0.3]] * 5)
        modes = torch.tensor([tc.REST_TURN, tc.REST_TURN, tc.REST_TURN, tc.PURE_TURN, tc.UPSTREAM])
        left = torch.tensor([1.0, 0.0, -0.3, 0.0, 0.0])
        wz = torch.tensor([0.7, -0.6, 0.9, 0.0, 0.0])
        standing = torch.tensor([False, False, True, False, False])
        out = tc.rest_turn_update(vel, modes, left, wz, standing)
        torch.testing.assert_close(out, torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -0.6], [0.0, 0.0, 0.0],
                                                      [0.5, 0.1, 0.3], [0.5, 0.1, 0.3]]))
        self.assertTrue((vel == 0.5).all(dim=0)[0])                     # input untouched

    def test_check_rest_mix(self):
        tc = self.tc
        tc.check_rest_mix(0.15, 0.25, 0.20, (1.5, 4.0))
        for bad in ((0.5, 0.4, 0.2, (1.5, 4.0)), (0.1, -0.1, 0.2, (1.5, 4.0)), (0.1, 0.2, 0.2, (4.0, 1.5)),
                    (0.1, 0.2, 0.2, (-1.0, 1.0))):
            with self.assertRaises(ValueError):
                tc.check_rest_mix(*bad)
        with self.assertRaises(ValueError):
            self.make(8, cfg=self.cfg(rel_pure_turn_envs=0.5, rel_rest_turn_envs=0.4, rel_direct_envs=0.2))

    def test_cfg_defaults_and_class(self):
        c = self.cfg()
        self.assertIs(c.class_type, self.tc.TurnRestMixVelocityCommand)
        self.assertEqual((c.rel_pure_turn_envs, c.rel_rest_turn_envs, c.rel_direct_envs), (0.15, 0.25, 0.20))
        self.assertEqual(tuple(c.rest_time_range), (1.5, 4.0))
        # the TurnCmd cfg's defaults are untouched by the subclass
        base = self.tc.TurnMixVelocityCommandCfg(resampling_time_range=(10.0, 10.0), asset_name="robot", ranges=c.ranges)
        self.assertEqual((base.rel_pure_turn_envs, base.rel_direct_envs), (0.25, 0.25))
        self.assertIs(base.class_type, self.tc.TurnMixVelocityCommand)

    def test_fractions_rest_then_sustained_turn(self):
        tc, n = self.tc, 40000
        cmd = self.make(n, seed=5)
        mode, stand = cmd.turn_mode.clone(), cmd.is_standing_env.clone()
        for m, target in ((tc.PURE_TURN, 0.15), (tc.REST_TURN, 0.25), (tc.DIRECT, 0.20), (tc.UPSTREAM, 0.40)):
            self.assertAlmostEqual(float((mode == m).float().mean()), target, delta=0.015)
        self.assertTrue((cmd.is_heading_env == (mode == tc.UPSTREAM)).all())
        rest = (mode == tc.REST_TURN) & ~stand
        rest_s = cmd.rest_left[rest].clone()
        self.assertTrue(((rest_s >= 1.5) & (rest_s <= 4.0)).all())
        wz = cmd.rest_wz[rest].clone()
        self.assertTrue(((wz.abs() >= 0.3) & (wz.abs() <= 1.0)).all())
        self.assertAlmostEqual(float((wz > 0).float().mean()), 0.5, delta=0.03)
        pure = (mode == tc.PURE_TURN) & ~stand
        pure_wz = None
        # step through 6 s: each REST row is zero until its rest time has elapsed, then (0, 0, wz), held
        for k in range(1, 151):
            self.heading += 0.03
            cmd._update_command()
            v = cmd.vel_command_b
            if pure_wz is None:
                pure_wz = v[pure, 2].clone()
            t = k * self.DT
            resting = rest_s > t + 1e-6
            vr = v[rest]
            self.assertTrue((vr[resting] == 0).all())
            self.assertTrue((vr[~resting & (rest_s < t - 1e-6), :2] == 0).all())
            torch.testing.assert_close(vr[~resting & (rest_s < t - 1e-6), 2], wz[~resting & (rest_s < t - 1e-6)])
            self.assertTrue((v[stand] == 0).all())
            self.assertTrue((v[pure, 2] == pure_wz).all() and (v[pure, :2] == 0).all())
        self.assertTrue(((cmd.vel_command_b[rest, 2] == wz)).all())      # all past rest by 6 s
        self.assertAlmostEqual(float(cmd.metrics["rest_turn_env"].mean()), float((mode == tc.REST_TURN).float().mean()))

    def test_partial_resample_and_dimension(self):
        tc, n = self.tc, 1000
        cmd = self.make(n, seed=6)
        for _ in range(10):
            cmd._update_command()
        v0, l0, m0 = cmd.vel_command_b.clone(), cmd.rest_left.clone(), cmd.turn_mode.clone()
        ids = torch.arange(0, n, 5)
        cmd._resample_command(ids)
        other = torch.ones(n, dtype=torch.bool)
        other[ids] = False
        self.assertTrue((cmd.vel_command_b[other] == v0[other]).all() and (cmd.rest_left[other] == l0[other]).all())
        self.assertTrue((cmd.turn_mode[other] == m0[other]).all())
        self.assertTrue((cmd.vel_command_b[ids][cmd.turn_mode[ids] == tc.REST_TURN] == 0).all())
        self.assertEqual(tuple(cmd.command.shape), (n, 3))
        cmd._resample_command(torch.tensor([], dtype=torch.long))


class TurnDiagnoseHelperTests(unittest.TestCase):
    """scripts/bench/turn_diagnose.py pure helpers."""

    @classmethod
    def setUpClass(cls):
        cls.td = _load("turn_diagnose_under_test", REPO / "scripts/bench/turn_diagnose.py")

    def test_count_liftoffs(self):
        c = self.td.count_liftoffs
        dt = 0.005
        self.assertEqual(c([True] * 10, dt), (0, 0.0))
        n, air = c([True] * 3 + [False] * 10 + [True] * 3 + [False] * 2 + [True], dt)   # 50 ms bout counts, 10 ms chatter not
        self.assertEqual(n, 1)
        self.assertAlmostEqual(air, 0.05)
        self.assertEqual(c([False] * 10 + [True] * 3, dt)[0], 0)          # airborne before first contact is not a lift-off
        self.assertEqual(c([True] + [False] * 4, dt)[0], 1)               # open bout at the end, 20 ms, counts

    def test_tb_normalisers(self):
        # error_vel_yaw sum 0.877 over a 419.8-step episode, max command 250 steps -> 0.522 rad/s per step
        self.assertAlmostEqual(self.td.per_step(0.877, 419.8, 250.0), 0.5223, places=3)
        # Episode_Reward 0.455 /s over 20 s, 419.8 steps of 0.04 s, weight 2.0 -> mean kernel 0.271
        self.assertAlmostEqual(self.td.kernel_mean(0.455, 20.0, 419.8, 0.04, 2.0), 0.2709, places=3)


class TurnTestV2RuleTests(unittest.TestCase):
    """scripts/bench/turn_test.py v2_judge: the predeclared v2 rule, sign-aware."""

    @classmethod
    def setUpClass(cls):
        cls.tt = _load("turn_test_under_test", REPO / "scripts/bench/turn_test.py")

    @staticmethod
    def turns(yaws, falls=None):
        falls = falls or [None] * len(yaws)
        return [{"cmd": [0.0, 0.0, 0.6 if i % 2 == 0 else -0.6], "yaw_deg": y, "fell_at_s": f}
                for i, (y, f) in enumerate(zip(yaws, falls))]

    def test_pass_and_sign(self):
        j = self.tt.v2_judge(self.turns([150.0, -150.0] * 3), {"yaw_deg": -15.0, "fell_at_s": None}, 150.0, 15.0)
        self.assertEqual((j["verdict"], j["n_turn_ok"]), ("PASS", 6))
        # turning 300 deg the WRONG way under a -0.6 command does not count
        j = self.tt.v2_judge(self.turns([290.0, 300.0] * 3), {"yaw_deg": 0.0, "fell_at_s": None}, 150.0, 15.0)
        self.assertEqual((j["verdict"], j["n_turn_ok"]), ("FAIL", 3))

    def test_one_weak_turn_a_fall_or_drift_fails(self):
        walk = {"yaw_deg": 0.0, "fell_at_s": None}
        # TurnBoth-s0 as measured 2026-09-26 under v2
        j = self.tt.v2_judge(self.turns([290.7, -244.8, 218.0, -11.4, 296.1, -239.4]), dict(walk), 150.0, 15.0)
        self.assertEqual((j["verdict"], j["n_turn_ok"]), ("FAIL", 5))
        j = self.tt.v2_judge(self.turns([200.0, -200.0] * 3, [None, 4.2] + [None] * 4), dict(walk), 150.0, 15.0)
        self.assertEqual(j["verdict"], "FAIL")
        j = self.tt.v2_judge(self.turns([200.0, -200.0] * 3), {"yaw_deg": 15.1, "fell_at_s": None}, 150.0, 15.0)
        self.assertEqual((j["verdict"], j["turn_ok"], j["walk_ok"]), ("FAIL", True, False))
        j = self.tt.v2_judge(self.turns([200.0, -200.0] * 3), {"yaw_deg": 1.0, "fell_at_s": 5.0}, 150.0, 15.0)
        self.assertEqual(j["verdict"], "FAIL")
        j = self.tt.v2_judge([], dict(walk), 150.0, 15.0)
        self.assertEqual(j["verdict"], "FAIL")


class TurnTestV2xRuleTests(unittest.TestCase):
    """scripts/bench/turn_test.py v2x_judge: the single-checkpoint qualification count rule (2026-09-27)."""

    @classmethod
    def setUpClass(cls):
        cls.tt = _load("turn_test_v2x_under_test", REPO / "scripts/bench/turn_test.py")

    @staticmethod
    def turns(yaws, falls=None):
        falls = falls or [None] * len(yaws)
        return [{"cmd": [0.0, 0.0, 0.6 if i % 2 == 0 else -0.6], "yaw_deg": y, "fell_at_s": f}
                for i, (y, f) in enumerate(zip(yaws, falls))]

    @staticmethod
    def walks(drifts, falls=None):
        falls = falls or [None] * len(drifts)
        return [{"yaw_deg": d, "fell_at_s": f} for d, f in zip(drifts, falls)]

    def judge(self, turns, walks):
        return self.tt.v2x_judge(turns, walks, 150.0, 15.0, 9, 2)

    def test_defaults_are_fresh_seeds(self):
        self.assertEqual(self.tt.V2X_TURN_SEEDS, (10, 11, 12, 13, 14))
        self.assertEqual(self.tt.V2X_WALK_SEEDS, (10, 11, 12))
        self.assertFalse(set(self.tt.V2X_TURN_SEEDS) & set(self.tt.V2_TURN_SEEDS))

    def test_nine_of_ten_passes_eight_fails(self):
        good = [200.0, -200.0] * 5
        j = self.judge(self.turns(good[:9] + [-11.4]), self.walks([1.0, -20.0, 3.0]))
        self.assertEqual((j["verdict"], j["n_turn_ok"], j["n_walk_ok"]), ("PASS", 9, 2))
        j = self.judge(self.turns(good[:8] + [100.0, -11.4]), self.walks([1.0, 2.0, 3.0]))
        self.assertEqual((j["verdict"], j["n_turn_ok"]), ("FAIL", 8))

    def test_wrong_direction_and_falls_do_not_count(self):
        j = self.judge(self.turns([200.0, 200.0] * 5), self.walks([0.0] * 3))       # -0.6 runs turn the wrong way
        self.assertEqual((j["verdict"], j["n_turn_ok"]), ("FAIL", 5))
        j = self.judge(self.turns([200.0, -200.0] * 5, [None] * 8 + [3.0, 2.0]), self.walks([0.0] * 3))
        self.assertEqual((j["verdict"], j["n_turn_ok"], j["n_fell"]), ("FAIL", 8, 2))

    def test_walk_clause(self):
        t = self.turns([200.0, -200.0] * 5)
        self.assertEqual(self.judge(t, self.walks([16.0, -15.1, 2.0]))["verdict"], "FAIL")
        self.assertEqual(self.judge(t, self.walks([1.0, 2.0, 3.0], [None, 4.0, 5.0]))["verdict"], "FAIL")
        self.assertEqual(self.judge(t, self.walks([15.0, -15.0, 40.0]))["verdict"], "PASS")
        self.assertEqual(self.judge(t, [])["verdict"], "FAIL")
        self.assertEqual(self.judge([], self.walks([0.0] * 3))["verdict"], "FAIL")

if __name__ == "__main__":
    unittest.main()


class TurnGaitV5WiringTests(unittest.TestCase):
    """2026-09-27 follow-up to TurnRest's failed push regression: the TurnRestPush arm and
    the continued-TurnBoth control (source-level; the configs need Isaac to import)."""

    def test_turnrestpush_is_turnrest_plus_the_adaptive_push(self):
        import ast
        tree = ast.parse((REPO / "src/bhl_robust/tasks/arms_env_cfg.py").read_text())
        cls = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        c = cls["HumanoidTurnRestPushCfg"]
        self.assertEqual([b.id for b in c.bases], ["HumanoidTurnRestCfg"])
        ann = {s.target.id: s.annotation.id for s in c.body if isinstance(s, ast.AnnAssign)}
        self.assertEqual(ann, {"events": "ArmsPushEventsCfg", "curriculum": "ArmsPushCurriculumCfg"})
        # the same two members as the push-adaptive task, nothing else changed
        ref = {s.target.id: s.annotation.id for s in cls["HumanoidPushAdaptiveCfg"].body if isinstance(s, ast.AnnAssign)}
        self.assertEqual(ann, ref)
        self.assertFalse(any(isinstance(s, ast.FunctionDef) for s in c.body))

    def test_registered(self):
        text = (REPO / "src/bhl_robust/tasks/__init__.py").read_text()
        self.assertIn('("Velocity-BHL-Arms-TurnRestPush-v0", arms_env_cfg.HumanoidTurnRestPushCfg)', text)

    def test_v5_launcher_arms_and_rules(self):
        text = (REPO / "slurm/repo20260923/gpu_turngait_v5.sbatch").read_text()
        self.assertIn("#SBATCH --array=0-5%4", text)
        self.assertIn("cont) PREFIX=arms-turn-turnboth-cont;   ARM_TASK=Velocity-BHL-Arms-TurnBoth-v0;", text)
        self.assertIn("push) PREFIX=arms-turn-turnrestpush-ft; ARM_TASK=Velocity-BHL-Arms-TurnRestPush-v0;", text)
        self.assertIn("SEED=$((IDX % 3))", text)
        self.assertIn('TURNQ_RUN="$RUN_NAME" bash "$REPO/slurm/repo20260923/cpu_turn_qualify.sbatch"', text)
        self.assertIn("PUSH arm PASSES iff >= 2 of its 3 seeds both PASS turn_test v2 AND are QUALIFIED", text)
        self.assertIn('k = sum(v[0] == "PASS" and v[1] == "QUALIFIED" for v in rows.values())', text)
        self.assertIn("hi, lo = sum(r > 0.30 for r in rates), sum(r <= 0.15 for r in rates)", text)
        self.assertNotIn("Isaac probe (GPU)", text)
        self.assertEqual(text.count('arm_guard "$TLOG"'), 2)     # smoke and full-run guards
