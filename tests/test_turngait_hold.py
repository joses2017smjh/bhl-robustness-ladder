"""Turning follow-up R1H (Velocity-BHL-Arms-TurnGaitClockHold-v0; SLURM_JOBS.md "(C') revised design") without Isaac.

Covers: the explicit-command mix (turn_command.TurnHoldMixVelocityCommand) on synthetic draws against upstream's real
`velocity_command.py` and a CommandTerm stand-in carrying isaaclab's real reset / compute / _resample logic (fractions,
the wz = 0 share, is_heading_env cleared exactly on the explicit envs, psi_ref at a time-out resample and at an episode
reset); the heading_hold term (gait_clock_mdp, loaded by path) on synthetic yaw and command sequences (active and
inactive, the wz boundary, wrap, psi_ref reset); its weight against R1's track_ang_vel_z_exp; the env.yaml recipe
check (turngait_hold_verdict.py check-recipe: exactly R1 + the two declared changes); the arm verdict at its boundaries; the launcher
header, arm table, log guard and training-record gate; and byte-identity of every pre-existing line of the touched
modules against the commit this workstream started from (BASE), so existing tasks keep their behaviour. Files this
workstream does not own are checked only for its own names, never pinned whole to BASE.
"""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import io
import json
import math
import os
import re
import subprocess
import sys
import sysconfig
import types
from contextlib import redirect_stdout
from dataclasses import MISSING
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
TURN_CMD = REPO / "src/bhl_robust/tasks/turn_command.py"
MDP_PATH = REPO / "src/bhl_robust/tasks/gait_clock_mdp.py"
ARMS_CFG = REPO / "src/bhl_robust/tasks/arms_env_cfg.py"
TASKS_INIT = REPO / "src/bhl_robust/tasks/__init__.py"
EVAL_GC = REPO / "src/bhl_robust/eval/gait_clock.py"
VERDICT_PATH = REPO / "scripts/bench/turngait_hold_verdict.py"
R12_VERDICT_PATH = REPO / "scripts/bench/turngait_r12_verdict.py"
SBATCH = REPO / "slurm/repo20260923/gpu_turngait_hold.sbatch"
R12_SBATCH = REPO / "slurm/repo20260923/gpu_turngait_r12.sbatch"
LOGROOT = REPO / "external/Berkeley-Humanoid-Lite/logs/rsl_rl/humanoid"
BASE = "46c0e0e"            # the branch commit this workstream started from (every touched module unchanged since)
MARK, END = "# --- turning-hold ---", "# --- end turning-hold ---"
TASK = "Velocity-BHL-Arms-TurnGaitClockHold-v0"

ISAACLAB = Path(os.environ.get("BHL_TEST_ISAACLAB_SOURCE",
    str(Path(sysconfig.get_paths()["purelib"]) / "isaaclab/source/isaaclab/isaaclab")))
HAVE_ISAACLAB_SRC = (ISAACLAB / "envs/mdp/commands/velocity_command.py").exists()

# The frozen texts, word for word as the task states them.
FROZEN_RULE = ("PASS iff >= 2/3 seeds both PASS turn_test v2 AND are QUALIFIED by cpu_turn_qualify's unchanged rule "
               "(v2x >= 9/10 on reset seeds 10-14, walk <= 15 deg on >= 2/3, push <= 9/60); else FAIL; INCOMPLETE if "
               "any JSON is missing. A seed that fails v2 does not count even if it qualifies through v2x.")
FROZEN_TRAINING = ("Training counts only if the run dir holds model_5999.pt and the training log shows the arm's task "
                   "id, the feet_gait term, the heading_hold term, the explicit-command mix and the push event.")
FROZEN_NEW_TASK = "A new task, never presented as a fine-tune of R1."
FROZEN_LABELS = "Labels: LEARNED gait; MuJoCo gates."
FROZEN_DISCLOSURE = ("the arm changes the command mix (the training distribution), so it tests explicit straight-walk "
                     "commands + the hold reward together, not the reward alone.")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gm = _load("gait_clock_mdp_hold_under_test", MDP_PATH)
vd = _load("turngait_hold_verdict_under_test", VERDICT_PATH)
from bhl_robust.eval import gait_clock as gc  # noqa: E402  (numpy only at import)


# =============================================================================== Isaac stand-ins

def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


def _wrap_to_pi(angles):          # verbatim logic of isaaclab.utils.math.wrap_to_pi
    wrapped = (angles + torch.pi) % (2 * torch.pi)
    return torch.where((wrapped == 0) & (angles > 0), torch.pi, wrapped - torch.pi)


class _CommandTerm:
    """isaaclab.managers.CommandTerm's buffers and its reset / compute / _resample, logic verbatim from
    isaaclab 2.3.2 managers/command_manager.py (CommandTerm.reset, .compute, ._resample), without debug vis."""

    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env
        self.num_envs, self.device = env.num_envs, env.device
        self.metrics = {}
        self.time_left = torch.zeros(self.num_envs, device=self.device)
        self.command_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            metric_value[env_ids] = 0.0
        self.command_counter[env_ids] = 0
        self._resample(env_ids)
        return extras

    def compute(self, dt):
        self._update_metrics()
        self.time_left -= dt
        resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
        if len(resample_env_ids) > 0:
            self._resample(resample_env_ids)
        self._update_command()

    def _resample(self, env_ids):
        if len(env_ids) != 0:
            self.time_left[env_ids] = self.time_left[env_ids].uniform_(*self.cfg.resampling_time_range)
            self._resample_command(env_ids)
            self.command_counter[env_ids] += 1


def _install():
    """turn_command.py on stubbed isaaclab modules, EXCEPT upstream's real velocity_command.py and the real
    configclass (as tests/test_turn_command.py does), with the CommandTerm stand-in above."""
    for k in [k for k in sys.modules if k == "isaaclab" or k.startswith("isaaclab.")]:
        del sys.modules[k]
    _module("isaaclab", __path__=[])
    utils = _module("isaaclab.utils", __path__=[str(ISAACLAB / "utils")])
    configclass = importlib.import_module("isaaclab.utils.configclass").configclass   # the real one
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
    tc = _load("turn_command_hold_under_test", TURN_CMD)
    return vc, UniformVelocityCommandCfg, tc


@pytest.fixture(scope="module")
def stack():
    if not HAVE_ISAACLAB_SRC:
        pytest.skip("installed isaaclab source not found")
    saved = {k: v for k, v in sys.modules.items() if k == "isaaclab" or k.startswith("isaaclab.")}
    vc, base_cfg, tc = _install()
    yield SimpleNamespace(vc=vc, BaseCfg=base_cfg, tc=tc)
    for k in [k for k in sys.modules if k == "isaaclab" or k.startswith("isaaclab.")]:   # leave no stub behind
        del sys.modules[k]
    sys.modules.update(saved)


DT = 0.04


def _r1_ranges(stack):
    # Upstream humanoid CommandsCfg.base_velocity = R1's (its env.yaml): lin_vel_x (-1, 1), lin_vel_y (-0.5, 0.5),
    # ang_vel_z (-1.5, 1.5), heading (-pi, pi)
    return stack.BaseCfg.Ranges(lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.5, 1.5),
                                heading=(-math.pi, math.pi))


def _hold_cfg(stack, **kw):
    """TurnHoldMixVelocityCommandCfg exactly as HumanoidTurnGaitClockHoldCfg builds it from R1's base_velocity."""
    ranges = _r1_ranges(stack)
    args = dict(resampling_time_range=(10.0, 10.0), debug_vis=False, asset_name="robot", heading_command=True,
                heading_control_stiffness=0.5, rel_standing_envs=0.02, rel_heading_envs=1.0, ranges=ranges,
                rel_pure_turn_envs=0.0, rel_direct_envs=0.30, direct_lin_vel_x=tuple(ranges.lin_vel_x),
                direct_lin_vel_y=tuple(ranges.lin_vel_y), direct_ang_vel_z=tuple(ranges.ang_vel_z),
                direct_zero_wz_prob=0.5)
    args.update(kw)
    return stack.tc.TurnHoldMixVelocityCommandCfg(**args)


class _World:
    """A batch of robots: heading_w is a live tensor the test moves; velocities are zero (metrics only)."""

    def __init__(self, n, seed):
        torch.manual_seed(seed)
        self.heading = torch.empty(n).uniform_(-math.pi, math.pi)
        data = SimpleNamespace(heading_w=self.heading, root_lin_vel_b=torch.zeros(n, 3), root_ang_vel_b=torch.zeros(n, 3))
        self.env = SimpleNamespace(num_envs=n, device="cpu", scene={"robot": SimpleNamespace(data=data)}, step_dt=DT)


def _make(stack, n, seed=0, cfg=None, quiet=True):
    w = _World(n, seed)
    buf = io.StringIO()
    with redirect_stdout(buf if quiet else sys.stdout):
        cmd = stack.tc.TurnHoldMixVelocityCommand(cfg or _hold_cfg(stack), w.env)
        cmd.reset(torch.arange(n))                     # what ManagerBasedRLEnv._reset_idx does for every env at start
    return cmd, w, buf.getvalue()


# =============================================================================== the command mix

def test_hold_cfg_defaults_and_class_and_older_defaults_untouched(stack):
    tc = stack.tc
    c = _hold_cfg(stack)
    assert c.class_type is tc.TurnHoldMixVelocityCommand
    assert (c.rel_pure_turn_envs, c.rel_direct_envs, c.direct_zero_wz_prob) == (0.0, 0.30, 0.5)
    assert (tuple(c.direct_lin_vel_x), tuple(c.direct_lin_vel_y), tuple(c.direct_ang_vel_z)) == \
        ((-1.0, 1.0), (-0.5, 0.5), (-1.5, 1.5))
    d = tc.TurnHoldMixVelocityCommandCfg(resampling_time_range=(10.0, 10.0), asset_name="robot", ranges=c.ranges)
    assert (d.rel_pure_turn_envs, d.rel_direct_envs, d.direct_zero_wz_prob) == (0.0, 0.30, 0.5)
    # the TurnCmd / TurnRest cfgs keep their defaults and classes (no new field on them)
    base = tc.TurnMixVelocityCommandCfg(resampling_time_range=(10.0, 10.0), asset_name="robot", ranges=c.ranges)
    assert base.class_type is tc.TurnMixVelocityCommand and not hasattr(base, "direct_zero_wz_prob")
    assert (base.rel_pure_turn_envs, base.rel_direct_envs, tuple(base.pure_turn_ang_vel_abs)) == (0.25, 0.25, (0.3, 1.0))
    assert (tuple(base.direct_lin_vel_x), tuple(base.direct_lin_vel_y), tuple(base.direct_ang_vel_z)) == \
        ((-0.5, 0.5), (-0.25, 0.25), (-1.0, 1.0))
    rest = tc.TurnRestMixVelocityCommandCfg(resampling_time_range=(10.0, 10.0), asset_name="robot", ranges=c.ranges)
    assert rest.class_type is tc.TurnRestMixVelocityCommand and not hasattr(rest, "direct_zero_wz_prob")
    assert (rest.rel_pure_turn_envs, rest.rel_rest_turn_envs, rest.rel_direct_envs) == (0.15, 0.25, 0.20)


def test_zero_direct_wz_pure_function(stack):
    tc = stack.tc
    vel = torch.tensor([[0.5, -0.2, 1.1]] * 5)
    modes = torch.tensor([tc.DIRECT, tc.DIRECT, tc.DIRECT, tc.UPSTREAM, tc.PURE_TURN])
    coin = torch.tensor([0.0, 0.4999, 0.5, 0.1, 0.1])
    out, zero = tc.zero_direct_wz(vel, modes, coin, 0.5)
    assert zero.tolist() == [True, True, False, False, False]                 # strict: coin < p, DIRECT rows only
    assert out[:2, 2].tolist() == [0.0, 0.0] and out[2:, 2].tolist() == pytest.approx([1.1, 1.1, 1.1])
    assert torch.equal(out[:, :2], vel[:, :2])                                 # vx, vy never touched
    assert torch.allclose(vel[:, 2], torch.full((5,), 1.1)) and vel.data_ptr() != out.data_ptr()   # input untouched
    assert not tc.zero_direct_wz(vel, modes, coin, 0.0)[1].any()               # p = 0: no zero-wz env
    assert tc.zero_direct_wz(vel, modes, coin, 1.0)[1].tolist() == [True, True, True, False, False]


def test_check_zero_wz_prob(stack):
    tc = stack.tc
    for ok in (0.0, 0.5, 1.0):
        tc.check_zero_wz_prob(ok)
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError):
            tc.check_zero_wz_prob(bad)
    with pytest.raises(ValueError):
        _make(stack, 8, cfg=_hold_cfg(stack, direct_zero_wz_prob=1.5))


def test_mix_fractions_wz_zero_share_and_heading_cleared(stack):
    """40,000 synthetic draws: about 30 % explicit, about half of them at wz = 0 exactly, is_heading_env cleared
    exactly on them, no PURE_TURN env, R1's ranges, the 2 % standing draw upstream's."""
    tc, n = stack.tc, 40000
    cmd, w, out = _make(stack, n, seed=0)
    mode, head, v, zero, stand = cmd.turn_mode, cmd.is_heading_env, cmd.vel_command_b.clone(), cmd.zero_wz_env, \
        cmd.is_standing_env
    explicit = mode == tc.DIRECT
    assert not bool((mode == tc.PURE_TURN).any())
    assert float(explicit.float().mean()) == pytest.approx(0.30, abs=0.015)
    assert float(zero[explicit].float().mean()) == pytest.approx(0.5, abs=0.02)
    assert torch.equal(~head, explicit) and torch.equal(cmd.explicit_env, explicit)   # cleared exactly there
    assert not bool((zero & ~explicit).any()) and bool((v[zero, 2] == 0.0).all())     # wz = 0 EXACTLY
    nz = explicit & ~zero
    assert bool((v[nz, 2].abs() <= 1.5).all()) and float(v[nz, 2].min()) < -1.4 and float(v[nz, 2].max()) > 1.4
    assert float(v[nz, 2].abs().lt(0.05).float().mean()) == pytest.approx(0.1 / 3.0, abs=0.01)   # U(-1.5, 1.5)
    assert bool((v[explicit, 0].abs() <= 1.0).all()) and bool((v[explicit, 1].abs() <= 0.5).all())
    assert float(v[explicit, 0].min()) < -0.95 and float(v[explicit, 0].max()) > 0.95   # vx over R1's whole range
    assert float(stand.float().mean()) == pytest.approx(0.02, abs=0.005)
    assert float(cmd.metrics["zero_wz_env"].mean()) == pytest.approx(float(zero.float().mean()))
    assert float(cmd.metrics["direct_wz_env"].mean()) == pytest.approx(float(explicit.float().mean()))
    assert "[turn_command] TurnHoldMixVelocityCommand: explicit 0.3 zero_wz_p 0.5" in out
    assert "heading_cleared_iff_explicit True" in out
    # the per-step update (upstream's _update_command, unchanged): heading envs follow the heading loop,
    # explicit commands survive, standing envs are zeroed
    cmd._update_command()
    v2 = cmd.vel_command_b
    keep = explicit & ~stand
    assert torch.equal(v2[keep], v[keep])
    assert bool((v2[stand] == 0).all())
    up = ~explicit & ~stand
    err = _wrap_to_pi(cmd.heading_target[up] - w.heading[up])
    torch.testing.assert_close(v2[up, 2], torch.clip(0.5 * err, -1.5, 1.5))


def test_explicit_commands_are_sustained_while_the_robot_turns(stack):
    tc, n = stack.tc, 4000
    cmd, w, _ = _make(stack, n, seed=1)
    cmd._update_command()
    keep = cmd.explicit_env & ~cmd.is_standing_env
    v0 = cmd.vel_command_b[keep].clone()
    zero = keep & cmd.zero_wz_env
    assert bool(zero.any())
    for _ in range(50):                              # the robot turns; heading errors change every step
        w.heading += 0.05
        cmd._update_command()
        assert torch.equal(cmd.vel_command_b[keep], v0)
        assert bool((cmd.vel_command_b[zero, 2] == 0.0).all())


def test_psi_ref_at_time_out_resample_and_at_episode_reset(stack):
    """psi_ref (yaw_at_resample) is written by the command term's own _resample_command, which CommandTerm._resample
    runs for the time-out resample (compute) and for the episode reset (reset)."""
    n = 1000
    cmd, w, _ = _make(stack, n, seed=2)
    assert torch.equal(cmd.yaw_at_resample, w.heading)            # the reset of every env at start
    assert bool((cmd.command_counter == 1).all())
    # time passes: the robot turns, nothing is resampled before 10 s
    for _ in range(100):
        w.heading += 0.01
        cmd.compute(DT)
    assert torch.allclose(cmd.yaw_at_resample, w.heading - 1.0, atol=1e-4)   # still the yaw at the start
    # a time-out for some envs: those (and only those) take the current yaw
    ids = torch.arange(0, n, 3)
    cmd.time_left[ids] = 0.01
    ref_before = cmd.yaw_at_resample.clone()
    cmd.compute(DT)
    other = torch.ones(n, dtype=torch.bool)
    other[ids] = False
    assert torch.equal(cmd.yaw_at_resample[ids], w.heading[ids])
    assert torch.equal(cmd.yaw_at_resample[other], ref_before[other])
    assert bool((cmd.command_counter[ids] == 2).all()) and bool((cmd.command_counter[other] == 1).all())
    # an episode reset: the reset events write the new root pose first, then command_manager.reset resamples
    rid = torch.arange(1, n, 5)
    w.heading[rid] = torch.empty(rid.numel()).uniform_(-math.pi, math.pi)      # the reset pose's yaw
    before = cmd.yaw_at_resample.clone()
    cmd.reset(rid)
    assert torch.equal(cmd.yaw_at_resample[rid], w.heading[rid])
    rest = torch.ones(n, dtype=torch.bool)
    rest[rid] = False
    assert torch.equal(cmd.yaw_at_resample[rest], before[rest])
    # command_counter cannot mark the reset: reset zeroes it, then the resample makes it 1 again
    assert bool((cmd.command_counter[rid] == 1).all())


def test_partial_resample_touches_only_those_envs(stack):
    tc, n = stack.tc, 1000
    cmd, w, _ = _make(stack, n, seed=3)
    snap = {k: getattr(cmd, k).clone() for k in ("vel_command_b", "is_heading_env", "turn_mode", "zero_wz_env",
                                                  "yaw_at_resample")}
    w.heading += 0.3
    ids = torch.arange(0, n, 7)
    cmd._resample(ids)
    other = torch.ones(n, dtype=torch.bool)
    other[ids] = False
    for k, v in snap.items():
        assert torch.equal(getattr(cmd, k)[other], v[other]), k
    assert torch.equal(cmd.is_heading_env[ids], cmd.turn_mode[ids] != tc.DIRECT)
    cmd._resample(torch.tensor([], dtype=torch.long))           # no-op, no error
    assert tuple(cmd.command.shape) == (n, 3)                   # -> R1's 77-obs layout unchanged


# =============================================================================== heading_hold

def test_hold_constants_and_scale_against_r1():
    assert (gm.HOLD_STD_RAD, gm.HOLD_WZ_THRESHOLD, gm.HOLD_WEIGHT, gm.HOLD_COMMAND_NAME) == \
        (0.2, 0.05, 1.0, "base_velocity")
    # R1's own constants untouched
    assert gm.GAIT_PERIOD_S == 0.8 and gm.FEET_GAIT_WEIGHT == 0.5 and gm.SWING_HEIGHT_WEIGHT == -20.0
    r = vd.HOLD_RECIPE
    assert r["heading_hold"]["weight"] == gm.HOLD_WEIGHT
    assert r["heading_hold"]["params"] == {"command_name": gm.HOLD_COMMAND_NAME, "std": gm.HOLD_STD_RAD,
                                           "wz_threshold": gm.HOLD_WZ_THRESHOLD}
    assert (r["rel_direct_envs"], r["direct_zero_wz_prob"], r["rel_pure_turn_envs"]) == (0.3, 0.5, 0.0)
    assert vd.RECIPE_SECTIONS == gc.RECIPE_SECTIONS                 # the same sections R1/R2's check compares
    # Scale, as stated in the header: R1's track_ang_vel_z_exp (weight 2.0, std 0.25) pays at most 2.0 per step,
    # heading_hold at most 1.0 -- half -- and only in about 16 % of envs.
    ref = _r1_env()["rewards"]["track_ang_vel_z_exp"]
    assert (ref["weight"], ref["params"]["std"]) == (2.0, 0.25)
    assert gm.HOLD_WEIGHT / ref["weight"] == 0.5
    p_active = 0.30 * (1.0 - 0.98 * 0.5 * (1.0 - 0.1 / 3.0))
    assert p_active == pytest.approx(0.158, abs=0.001)
    src = SBATCH.read_text()
    assert "R1's track_ang_vel_z_exp has weight 2.0 (std 0.25 rad/s)" in src and "half the yaw-rate term's" in src


def test_wrap_to_pi_is_isaaclabs():
    a = torch.tensor([0.0, 0.1, -0.1, math.pi, -math.pi, 3 * math.pi, -3 * math.pi, 2 * math.pi - 0.1,
                      -2 * math.pi + 0.1, 7.0, -7.0, 100.0])
    assert torch.equal(gm.wrap_to_pi(a), _wrap_to_pi(a))
    assert bool(((gm.wrap_to_pi(a) > -math.pi - 1e-6) & (gm.wrap_to_pi(a) <= math.pi + 1e-6)).all())


def _r(yaw, ref, wz, explicit):
    t = lambda x: torch.as_tensor(x, dtype=torch.float32)  # noqa: E731
    return gm.heading_hold_reward(t(yaw), t(ref), t(wz), torch.as_tensor(explicit), 0.2, 0.05).tolist()


def test_heading_hold_reward_active_inactive_and_wrap():
    # dpsi 0 -> 1.0; dpsi 0.2 rad -> exp(-1); dpsi -0.1 -> exp(-0.25)
    got = _r([0.5, 0.7, 0.4], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [True, True, True])
    assert got == pytest.approx([1.0, math.exp(-1.0), math.exp(-0.25)], rel=1e-5)
    # inactive: not explicit, or |wz_cmd| >= 0.05 (strict), whatever the yaw error
    got = _r([0.5, 0.5, 0.5, 0.5, 0.5], [0.5] * 5, [0.0, 0.05, -0.05, 0.0499, -0.0499], [False, True, True, True, True])
    assert got == pytest.approx([0.0, 0.0, 0.0, 1.0, 1.0])
    # wrap: yaw 3.1 vs psi_ref -3.1 is 0.083 rad apart, not 6.2
    d = 2 * math.pi - 6.2
    assert _r([3.1], [-3.1], [0.0], [True]) == pytest.approx([math.exp(-(d / 0.2) ** 2)], rel=1e-4)
    assert _r([-3.1], [3.1], [0.0], [True]) == pytest.approx([math.exp(-(d / 0.2) ** 2)], rel=1e-4)
    assert _r([0.1 + 4 * math.pi], [0.0], [0.0], [True]) == pytest.approx([math.exp(-0.25)], rel=1e-3)
    # large errors pay ~0, never negative
    assert _r([math.pi / 2], [0.0], [0.0], [True])[0] == pytest.approx(0.0, abs=1e-20)


class _HoldEnv:
    def __init__(self, term):
        self.command_manager = SimpleNamespace(get_term=lambda name: {"base_velocity": term}[name])


def test_heading_hold_term_on_a_synthetic_sequence(stack):
    """yaw drifts at 0.05 rad/step from a resample: the term pays exp(-(0.05 k / 0.2)^2) in the active envs and 0
    elsewhere; a time-out resample and an episode reset each restart psi_ref."""
    n = 2000
    cmd, w, _ = _make(stack, n, seed=4)
    cmd._update_command()                                         # standing envs zeroed, as every step does
    env = _HoldEnv(cmd)
    ex, wz = cmd.explicit_env, cmd.vel_command_b[:, 2]
    active = ex & (wz.abs() < 0.05)
    assert bool(active.any()) and bool((ex & ~active).any()) and bool((~ex).any())
    assert float(active.float().mean()) == pytest.approx(0.158, abs=0.03)
    buf = io.StringIO()
    with redirect_stdout(buf):
        r0 = gm.heading_hold(env, "base_velocity", 0.2, 0.05)
    assert "[gait_clock_mdp] heading_hold command base_velocity (TurnHoldMixVelocityCommand) std 0.2" in buf.getvalue() \
        or "heading_hold" in gm._LOGGED
    assert bool((r0[active] == 1.0).all()) and bool((r0[~active] == 0.0).all())
    for k in range(1, 6):
        w.heading += 0.05
        cmd._update_command()
        r = gm.heading_hold(env, "base_velocity", 0.2, 0.05)
        want = math.exp(-((0.05 * k) / 0.2) ** 2)
        torch.testing.assert_close(r[active], torch.full((int(active.sum()),), want), rtol=1e-4, atol=1e-5)
        assert bool((r[~active] == 0.0).all())
    # time-out resample of the active envs whose new draw is again active: psi_ref = the current yaw -> 1.0
    ids = active.nonzero().flatten()
    cmd.time_left[ids] = 0.0
    cmd.compute(DT)
    r = gm.heading_hold(env, "base_velocity", 0.2, 0.05)
    again = cmd.explicit_env[ids] & (cmd.vel_command_b[ids, 2].abs() < 0.05)
    assert bool(again.any())
    assert bool((r[ids[again]] == 1.0).all())
    assert bool((r[ids[~again]] == 0.0).all())                     # re-drawn into heading mode or |wz| >= 0.05
    # episode reset: a new pose (yaw + 1 rad) then the reset resample -> psi_ref follows the new yaw
    w.heading[ids] += 1.0
    cmd.reset(ids)
    cmd._update_command()
    r = gm.heading_hold(env, "base_velocity", 0.2, 0.05)
    now = cmd.explicit_env[ids] & (cmd.vel_command_b[ids, 2].abs() < 0.05)
    assert bool((r[ids[now]] == 1.0).all()) and bool((r[ids[~now]] == 0.0).all())
    # a plain upstream command term (R1's) is refused: the term needs the hold mix's psi_ref
    plain = SimpleNamespace(vel_command_b=cmd.vel_command_b, robot=cmd.robot)
    with pytest.raises(TypeError):
        gm.heading_hold(_HoldEnv(plain), "base_velocity", 0.2, 0.05)


def test_heading_hold_signature_matches_its_params():
    """Isaac's reward manager calls func(env, **params): the params must be exactly the signature's."""
    import inspect
    sig = list(inspect.signature(gm.heading_hold).parameters)
    assert sig == ["env", "command_name", "std", "wz_threshold"]
    assert sorted(vd.HOLD_RECIPE["heading_hold"]["params"]) == sorted(sig[1:])


def test_gait_clock_mdp_still_imports_without_isaaclab():
    had = {k for k in sys.modules if k == "isaaclab" or k.startswith("isaaclab.")}
    _load("gait_clock_mdp_hold_reload", MDP_PATH)
    assert {k for k in sys.modules if k == "isaaclab" or k.startswith("isaaclab.")} - had == set()


# =============================================================================== source: append-only, byte-identical

def _git_show(rev, path):
    try:
        return subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{path}"], capture_output=True, text=True,
                              check=True).stdout
    except Exception:                                               # noqa: BLE001
        return None


@pytest.mark.parametrize("rel", ["src/bhl_robust/tasks/turn_command.py", "src/bhl_robust/tasks/gait_clock_mdp.py",
                                 "src/bhl_robust/tasks/arms_env_cfg.py"])
def test_every_pre_existing_line_is_byte_identical(rel):
    """Append-only: the file at BASE is a byte-identical prefix of the file now (robust to later appended blocks),
    and the turning-hold block comes after it."""
    now = (REPO / rel).read_text()
    assert now.count(MARK) == 1 and now.count(END) == 1 and now.index(MARK) < now.index(END)
    base = _git_show(BASE, rel)
    if base is None:
        pytest.skip(f"git or commit {BASE} unavailable")
    assert now.startswith(base) and now.index(MARK) >= len(base)


# Names only this workstream introduced (its block markers, task, command class, reward term, recipe check, launcher
# and verdict script). None may appear in a file this workstream does not own.
R1H_IDENTIFIERS = (MARK, END, "TurnGaitClockHold", "TurnHoldMix", "heading_hold", "HOLD_RECIPE", "check_recipe_hold",
                   "recipe_diff_hold", "check_hold_reference", "turngait-hold", "turngait_hold")


@pytest.mark.parametrize("path", [EVAL_GC, R12_VERDICT_PATH, R12_SBATCH], ids=lambda p: p.name)
def test_untouched_files_carry_no_turning_hold_code(path):
    """The deploy-side clock (a hash-frozen source of the Mission 7 clocks2 replay snapshot), R1/R2's verdict and
    R1/R2's launcher: this workstream adds nothing to them (its recipe check lives in turngait_hold_verdict.py).
    Checked by the workstream's own names, not by a whole-file pin to BASE, so a later edit by their owners does
    not fail this test."""
    text = path.read_text()
    for ident in R1H_IDENTIFIERS:
        assert ident not in text, f"{ident!r} in {path.relative_to(REPO)}"


def test_registration_block():
    src = TASKS_INIT.read_text()
    assert src.count(MARK) == 1 and src.count(END) == 1
    block = src[src.index(MARK):src.index(END)]
    assert f'id="{TASK}"' in block and "arms_env_cfg.HumanoidTurnGaitClockHoldCfg" in block
    assert '"rsl_rl_cfg_entry_point": _ARM_PPO_CFG' in block                 # R1's runner
    assert block.count("gym.register(") == 1 and "try:" in block and "except Exception" in block
    assert "# --- " not in block[len(MARK):]                                  # self-contained: no other block inside
    # R1's registration untouched; ours is appended (its place among other appended blocks is not asserted)
    assert '("Velocity-BHL-Arms-TurnGaitClock-v0", arms_env_cfg.HumanoidTurnGaitClockCfg),' in src
    base = _git_show(BASE, "src/bhl_robust/tasks/__init__.py")
    if base is not None:
        assert src.startswith(base)                                            # appended only


def test_arms_env_cfg_hold_classes_are_r1_plus_the_two_changes():
    src = ARMS_CFG.read_text()
    tree = ast.parse(src)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    for name in ("TurnGaitHoldRewardsCfg", "HumanoidTurnGaitClockHoldCfg"):
        assert src.index(f"class {name}(") > src.index(MARK) > src.index("class HumanoidTurnGaitCriticCfg(")
    base = lambda c: [ast.unparse(b) for b in classes[c].bases]  # noqa: E731
    assert base("HumanoidTurnGaitClockHoldCfg") == ["HumanoidTurnGaitClockCfg"]          # R1 itself
    assert base("TurnGaitHoldRewardsCfg") == ["TurnGaitRewardsCfg"]                      # R1's rewards
    rew = classes["TurnGaitHoldRewardsCfg"]
    fields = [s for s in rew.body if isinstance(s, (ast.Assign, ast.AnnAssign))]
    assert [ast.unparse(s.targets[0]) for s in fields] == ["heading_hold"]               # one new term, nothing else
    rsrc = ast.get_source_segment(src, rew)
    assert "func=gait_clock_mdp.heading_hold" in rsrc and "weight=gait_clock_mdp.HOLD_WEIGHT" in rsrc
    for k in ("HOLD_COMMAND_NAME", "HOLD_STD_RAD", "HOLD_WZ_THRESHOLD"):
        assert f"gait_clock_mdp.{k}" in rsrc
    hold = classes["HumanoidTurnGaitClockHoldCfg"]
    ann = {s.target.id: ast.unparse(s.annotation) for s in hold.body if isinstance(s, ast.AnnAssign)}
    assert ann == {"rewards": "TurnGaitHoldRewardsCfg"}                                  # observations, events: R1's
    post = [s for s in hold.body if isinstance(s, ast.FunctionDef)]
    assert [f.name for f in post] == ["__post_init__"]
    call = [n for n in ast.walk(post[0]) if isinstance(n, ast.Call) and ast.unparse(n.func) == "TurnHoldMixVelocityCommandCfg"]
    assert len(call) == 1
    kw = {k.arg: ast.unparse(k.value) for k in call[0].keywords}
    # every field of R1's command copied from R1's own cfg (as HumanoidTurnCmdCfg does), plus the declared mix
    for f in ("resampling_time_range", "debug_vis", "asset_name", "heading_command", "heading_control_stiffness",
              "rel_standing_envs", "rel_heading_envs", "ranges"):
        assert kw.pop(f) == f"old.{f}", f
    assert kw == {"rel_pure_turn_envs": "0.0", "rel_direct_envs": "0.3", "direct_lin_vel_x": "tuple(old.ranges.lin_vel_x)",
                  "direct_lin_vel_y": "tuple(old.ranges.lin_vel_y)", "direct_ang_vel_z": "tuple(old.ranges.ang_vel_z)",
                  "direct_zero_wz_prob": "0.5"}
    psrc = ast.get_source_segment(src, post[0])
    assert psrc.index("super().__post_init__()") < psrc.index("old = self.commands.base_velocity")
    # the same copied-field list as HumanoidTurnCmdCfg (TurnCmd), the repo's existing DIRECT task
    tcall = [n for n in ast.walk(classes["HumanoidTurnCmdCfg"]) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "TurnMixVelocityCommandCfg"][0]
    copied = {k.arg for k in tcall.keywords if ast.unparse(k.value) == f"old.{k.arg}"}
    assert copied == {"resampling_time_range", "debug_vis", "asset_name", "heading_command",
                      "heading_control_stiffness", "rel_standing_envs", "rel_heading_envs", "ranges"}


# =============================================================================== env.yaml: exactly R1 + two changes

def _term(func, params=None, **kw):
    return {"func": func, "params": params or {}, **kw}


def _r1_env():
    """A compact R1 env.yaml (the sections the checks read), as an R1 run's params/env.yaml shows them."""
    obs = {k: _term(f"isaaclab.envs.mdp.observations:{k}") for k in
           ("velocity_commands", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions")}
    clock = _term("bhl_robust.tasks.gait_clock_mdp:gait_clock", {"period": 0.8})
    policy = {"concatenate_terms": True, "enable_corruption": True, **obs, "gait_clock": clock}
    critic = {"concatenate_terms": True, "enable_corruption": False, **obs,
              "base_lin_vel": _term("isaaclab.envs.mdp.observations:base_lin_vel"), "gait_clock": copy.deepcopy(clock)}
    feet = {"name": "contact_forces", "body_names": [".*_left_ankle_roll", ".*_right_ankle_roll"], "preserve_order": True}
    return {
        "sim": {"dt": 0.005}, "decimation": 8, "seed": 0,
        "observations": {"policy": policy, "critic": critic},
        "rewards": {
            "track_ang_vel_z_exp": {"func": "x:track_ang_vel_z_world_exp", "params": {"std": 0.25}, "weight": 2.0},
            "feet_air_time": {"func": "bhl_robust.tasks.arms_env_cfg:feet_air_time_positive_biped_turn",
                              "params": {"command_name": "base_velocity", "threshold": 0.5}, "weight": 0.0},
            "feet_gait": {"func": "bhl_robust.tasks.gait_clock_mdp:feet_gait", "weight": 0.5,
                          "params": {"period": 0.8, "offset": [0.0, 0.5], "threshold": 0.55, "sensor_cfg": feet}},
            "feet_swing_height": {"func": "bhl_robust.tasks.gait_clock_mdp:feet_swing_height", "weight": -20.0,
                                  "params": {"target_height": 0.05, "foot_height_offset": 0.06, "force_threshold": 1.0,
                                             "sensor_cfg": feet}}},
        "events": {"push_robot": {"func": "isaaclab.envs.mdp.events:push_by_setting_velocity", "mode": "interval",
                                  "interval_range_s": [5.0, 9.0],
                                  "params": {"velocity_range": {"x": [-0.5, 0.5], "y": [-0.5, 0.5]}}},
                   "physics_material": {"func": "x:randomize_rigid_body_material",
                                        "params": {"static_friction_range": [0.4, 1.2]}, "mode": "startup"}},
        "commands": {"base_velocity": {
            "class_type": "isaaclab.envs.mdp.commands.velocity_command:UniformVelocityCommand",
            "resampling_time_range": [10.0, 10.0], "debug_vis": True, "asset_name": "robot", "heading_command": True,
            "heading_control_stiffness": 0.5, "rel_standing_envs": 0.02, "rel_heading_envs": 1.0,
            "ranges": {"lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-0.5, 0.5], "ang_vel_z": [-1.5, 1.5],
                       "heading": [-3.141592653589793, 3.141592653589793]},
            "goal_vel_visualizer_cfg": {"prim_path": "/Visuals/Command/velocity_goal"}}},
        "terminations": {"time_out": {"func": "x:time_out"}}, "actions": {"joint_pos": {"scale": 0.25}},
        "curriculum": None,
    }


def _hold_env(ref=None):
    env = copy.deepcopy(ref if ref is not None else _r1_env())
    c = env["commands"]["base_velocity"]
    c["class_type"] = "bhl_robust.tasks.turn_command:TurnHoldMixVelocityCommand"
    c.update(rel_pure_turn_envs=0.0, rel_direct_envs=0.3, pure_turn_ang_vel_abs=[0.3, 1.0],
             direct_lin_vel_x=list(c["ranges"]["lin_vel_x"]), direct_lin_vel_y=list(c["ranges"]["lin_vel_y"]),
             direct_ang_vel_z=list(c["ranges"]["ang_vel_z"]), direct_zero_wz_prob=0.5)
    env["rewards"]["heading_hold"] = {"func": "bhl_robust.tasks.gait_clock_mdp:heading_hold", "weight": 1.0,
                                      "params": {"command_name": "base_velocity", "std": 0.2, "wz_threshold": 0.05}}
    return env


def test_recipe_hold_accepts_the_recipe_and_rejects_r1():
    ref, env = _r1_env(), _hold_env()
    assert gc.check_recipe(ref, "R1") == [] and vd.check_hold_reference(ref) == []
    assert vd.check_recipe_hold(env) == [] and vd.recipe_diff_hold(env, ref) == []
    assert vd.check_recipe_hold(ref) != []                                     # R1 itself is not R1H
    d = vd.recipe_diff_hold(ref, ref)
    assert any("heading_hold (declared change missing)" in x for x in d)
    assert any("direct_zero_wz_prob (declared change missing)" in x for x in d)
    assert vd.check_hold_reference(env) != []                                  # R1H is not a valid reference


@pytest.mark.parametrize("mutate, needle", [
    (lambda e: e["commands"]["base_velocity"].update(class_type="isaaclab.envs.mdp.commands.velocity_command:UniformVelocityCommand"), "class_type"),
    (lambda e: e["commands"]["base_velocity"].update(rel_direct_envs=0.25), "rel_direct_envs"),
    (lambda e: e["commands"]["base_velocity"].update(rel_pure_turn_envs=0.1), "rel_pure_turn_envs"),
    (lambda e: e["commands"]["base_velocity"].update(direct_zero_wz_prob=0.4), "direct_zero_wz_prob"),
    (lambda e: e["commands"]["base_velocity"].update(direct_lin_vel_x=[-0.5, 0.5]), "direct_lin_vel_x"),
    (lambda e: e["commands"]["base_velocity"].update(direct_ang_vel_z=[-1.0, 1.0]), "direct_ang_vel_z"),
    (lambda e: e["commands"]["base_velocity"].update(heading_command=False), "heading_command"),
    (lambda e: e["commands"]["base_velocity"].update(rel_heading_envs=0.7), "rel_heading_envs"),
    (lambda e: e["rewards"]["heading_hold"].update(weight=2.0), "heading_hold weight"),
    (lambda e: e["rewards"]["heading_hold"]["params"].update(std=0.3), "heading_hold std"),
    (lambda e: e["rewards"]["heading_hold"]["params"].update(wz_threshold=0.1), "heading_hold wz_threshold"),
    (lambda e: e["rewards"]["heading_hold"]["params"].update(asset_cfg="robot"), "heading_hold params"),
    (lambda e: e["rewards"]["heading_hold"].update(func="x:other"), "heading_hold func"),
    (lambda e: e["rewards"]["feet_gait"].update(weight=1.0), "R1 recipe: feet_gait weight"),
])
def test_recipe_hold_catches_each_change(mutate, needle):
    env = _hold_env()
    mutate(env)
    probs = vd.check_recipe_hold(env)
    assert any(needle in p for p in probs), probs


def test_recipe_diff_hold_catches_anything_outside_the_two_changes():
    ref = _r1_env()
    for mutate, needle in (
            (lambda e: e["rewards"]["track_ang_vel_z_exp"].update(weight=3.0), "track_ang_vel_z_exp.weight"),
            (lambda e: e["events"]["physics_material"]["params"].update(static_friction_range=[0.2, 1.4]),
             "static_friction_range"),
            (lambda e: e["commands"]["base_velocity"]["ranges"].update(lin_vel_x=[-0.5, 0.5]), "ranges.lin_vel_x"),
            (lambda e: e["commands"]["base_velocity"].update(resampling_time_range=[5.0, 5.0]), "resampling_time_range"),
            (lambda e: e["commands"]["base_velocity"].update(rel_standing_envs=0.1), "rel_standing_envs"),
            (lambda e: e["observations"]["policy"].pop("gait_clock"), "policy.gait_clock"),
            (lambda e: e["rewards"].update(extra_term={"func": "x:y", "weight": 1.0}), "rewards.extra_term (new)"),
            (lambda e: e["commands"]["base_velocity"].update(rest_time_range=[1.5, 4.0]), "rest_time_range (new)"),
            (lambda e: e.update(curriculum={"push_levels": {"func": "x"}}), "curriculum")):
        env = _hold_env()
        mutate(env)
        d = vd.recipe_diff_hold(env, ref)
        assert any(needle in x for x in d), (needle, d)


def _real_r1_env_yaml():
    runs = sorted(LOGROOT.glob("*_arms-turngait-clock-s0"))
    if not runs or not (runs[-1] / "params/env.yaml").is_file():
        pytest.skip("needs the arms-turngait-clock-s0 run's params/env.yaml")
    pytest.importorskip("yaml")
    return runs[-1] / "params/env.yaml"


def test_recipe_on_the_real_r1_env_yaml_and_the_cli(tmp_path, capsys):
    import yaml
    ref_path = _real_r1_env_yaml()
    ref = gc.load_env_yaml(ref_path)
    assert vd.check_hold_reference(ref) == []
    env = _hold_env(ref)
    assert vd.check_recipe_hold(env) == [] and vd.recipe_diff_hold(env, ref) == []
    out = tmp_path / "env.yaml"
    out.write_text(yaml.safe_dump(env, sort_keys=False))          # key order kept (the clock stays last)
    assert vd.main(["check-recipe", "--env-yaml", str(out), "--ref-env-yaml", str(ref_path)]) == 0
    assert capsys.readouterr().out.startswith("HOLD RECIPE: OK ")
    assert vd.main(["check-recipe", "--env-yaml", str(ref_path), "--ref-env-yaml", str(ref_path)]) == 1
    assert capsys.readouterr().out.startswith("HOLD RECIPE: FAIL ")
    env["rewards"]["track_ang_vel_z_exp"]["weight"] = 1.0
    out.write_text(yaml.safe_dump(env, sort_keys=False))
    assert vd.main(["check-recipe", "--env-yaml", str(out), "--ref-env-yaml", str(ref_path)]) == 1
    assert "differs from R1: rewards.track_ang_vel_z_exp.weight" in capsys.readouterr().out


# =============================================================================== the arm verdict

def _seed(training=True, v2="PASS", qualify="QUALIFIED"):
    return {"training": training, "v2": v2, "qualify": qualify}


def test_rule_texts_are_the_frozen_ones():
    assert vd.RULE == FROZEN_RULE and vd.TRAINING_RULE == FROZEN_TRAINING
    assert vd.LABEL == "LEARNED gait; MuJoCo gates" and vd.DISCLOSURE == FROZEN_DISCLOSURE
    assert vd.FINAL_CKPT == "model_5999.pt" and vd.NEED == 2 and vd.SEEDS == (0, 1, 2)
    assert vd.ARM["task"] == TASK and vd.ARM["prefix"] == "arms-turngait-hold" and vd.ARM["name"] == "R1H"
    r12 = _load("turngait_r12_verdict_for_hold", R12_VERDICT_PATH)
    assert vd.V2_RULE == r12.V2_RULE and vd.RULE == r12.RULE          # the unchanged gate and v5's rule


def test_verdict_boundaries():
    v = vd.arm_verdict
    assert v([_seed(), _seed(), _seed()])["verdict"] == "PASS"
    r = v([_seed(), _seed(), _seed(v2="FAIL")])
    assert r["verdict"] == "PASS" and r["n_counted"] == 2                         # exactly 2/3
    assert v([_seed(), _seed(v2="FAIL"), _seed(v2="FAIL")])["verdict"] == "FAIL"   # 1/3
    assert v([_seed(v2="FAIL")] * 3)["verdict"] == "FAIL"                          # 0/3
    # v2 PASS but NOT QUALIFIED does not count; QUALIFIED but v2 FAIL does not count (v2x cannot rescue it)
    assert v([_seed(), _seed(qualify="NOT QUALIFIED"), _seed(v2="FAIL")])["verdict"] == "FAIL"
    assert v([_seed(), _seed(v2="FAIL", qualify="QUALIFIED"), _seed(qualify="NOT QUALIFIED")])["verdict"] == "FAIL"
    # training that does not count makes the seed not count, whatever its gates say
    assert v([_seed(), _seed(training=False), _seed(v2="FAIL")])["verdict"] == "FAIL"
    # any missing JSON -> INCOMPLETE, even with 2 seeds already counted
    r = v([_seed(), _seed(), _seed(qualify=None)])
    assert r["verdict"] == "INCOMPLETE" and r["n_counted"] == 2 and r["missing_seeds"] == [2]
    assert v([_seed(), _seed(), None])["verdict"] == "INCOMPLETE"
    assert v([_seed(training=None), _seed(), _seed()])["verdict"] == "INCOMPLETE"
    assert v([_seed(v2=None), _seed(), _seed()])["verdict"] == "INCOMPLETE"
    with pytest.raises(ValueError):
        v([_seed(), _seed()])


def _training(**kw):
    t = {"run": "r", "final_ckpt": "model_5999.pt", "final_ckpt_present": True, "task_id_in_log": True,
         "feet_gait_in_log": True, "heading_hold_in_log": True, "command_mix_in_log": True, "push_robot_in_log": True}
    t.update(kw)
    return t


def test_training_clause_needs_all_six_flags():
    assert vd.training_counts(_training())
    for bad in ({"final_ckpt": "model_2.pt"}, {"final_ckpt_present": False}, {"task_id_in_log": False},
                {"feet_gait_in_log": False}, {"heading_hold_in_log": False}, {"command_mix_in_log": False},
                {"push_robot_in_log": False}, {"heading_hold_in_log": "true"}):
        assert not vd.training_counts(_training(**bad)), bad
    r1_style = _training()
    del r1_style["heading_hold_in_log"], r1_style["command_mix_in_log"]        # an R1 record does not count here
    assert not vd.training_counts(r1_style)


def _v2_json(run, verdict="PASS", **rule):
    r = dict(vd.V2_RULE)
    r.update(rule)
    return {"deploy": f"/u/logs/rsl_rl/humanoid/2026-10-03_00-00-00_{run}/exported/deploy.yaml", "protocol": "v2",
            "verdict": verdict, "rule": r, "turns": [], "walk": {}}


def _write_seed(res, run, v2="PASS", qualify="QUALIFIED", training=None, v2json=None):
    for sub in ("training", "turn-test-v2", "qualify"):
        (res / sub).mkdir(parents=True, exist_ok=True)
    (res / "training" / f"{run}.json").write_text(json.dumps(training or _training(run=run)))
    (res / "turn-test-v2" / f"{run}.json").write_text(json.dumps(v2json or _v2_json(run, v2)))
    (res / "qualify" / f"{run}__qualify.json").write_text(json.dumps({"run": run, "verdict": qualify,
                                                                      "clauses": {"push": {"rate": 0.1}}}))


def test_read_seed_accepts_only_this_run_and_the_declared_protocol(tmp_path):
    run = "arms-turngait-hold-s0"
    _write_seed(tmp_path, run)
    assert vd.read_seed(tmp_path, run) == {"run": run, "training": True, "v2": "PASS", "qualify": "QUALIFIED",
                                           "push": {"rate": 0.1}}
    _write_seed(tmp_path, run, v2json=_v2_json("arms-turngait-clock-s0"))           # R1's v2 JSON, not this run's
    assert vd.read_seed(tmp_path, run)["v2"] is None
    _write_seed(tmp_path, run, v2json=_v2_json(run, seconds=5.0))                  # not the declared protocol
    assert vd.read_seed(tmp_path, run)["v2"] is None
    _write_seed(tmp_path, run, v2json={**_v2_json(run), "protocol": "v2x"})
    assert vd.read_seed(tmp_path, run)["v2"] is None
    _write_seed(tmp_path, run, training=_training(run="arms-turngait-clock-s0"))
    assert vd.read_seed(tmp_path, run)["training"] is None
    _write_seed(tmp_path, run, training=_training(run=run, heading_hold_in_log=False))
    assert vd.read_seed(tmp_path, run)["training"] is False
    _write_seed(tmp_path, run, qualify="INCOMPLETE")
    assert vd.read_seed(tmp_path, run)["qualify"] is None


def test_verdict_cli_writes_once_and_never_overwrites(tmp_path, capsys):
    for s in (0, 1):
        _write_seed(tmp_path, f"arms-turngait-hold-s{s}")
    assert vd.main(["verdict", "--res", str(tmp_path)]) == 0
    assert not (tmp_path / "verdict" / "R1H.json").exists()                         # INCOMPLETE writes nothing
    assert capsys.readouterr().out.startswith("ARM-HOLD R1H (Velocity-BHL-Arms-TurnGaitClockHold-v0): INCOMPLETE")
    _write_seed(tmp_path, "arms-turngait-hold-s2", v2="FAIL", qualify="QUALIFIED")
    vd.main(["verdict", "--res", str(tmp_path)])
    rec = json.loads((tmp_path / "verdict" / "R1H.json").read_text())
    assert rec["verdict"] == "PASS" and rec["n_counted"] == 2 and rec["rule"] == FROZEN_RULE
    assert rec["training_rule"] == FROZEN_TRAINING and rec["disclosure"] == FROZEN_DISCLOSURE
    assert rec["task"] == TASK and "never presented as a fine-tune of R1" in rec["note"]
    _write_seed(tmp_path, "arms-turngait-hold-s0", v2="FAIL")                      # results change afterwards...
    vd.main(["verdict", "--res", str(tmp_path)])
    assert json.loads((tmp_path / "verdict" / "R1H.json").read_text()) == rec        # ...the recorded verdict stays
    assert "not overwritten" in capsys.readouterr().out


# =============================================================================== launcher

def _header():
    lines = []
    for ln in SBATCH.read_text().splitlines()[1:]:
        if not ln.startswith("#"):
            break
        if not ln.startswith("#SBATCH"):
            lines.append(ln[1:].strip())
    return " ".join(" ".join(lines).split())


def _body():
    return "\n".join(ln for ln in SBATCH.read_text().splitlines() if not ln.lstrip().startswith("#"))


def test_launcher_header_holds_the_frozen_rule_verbatim():
    h = _header()
    for frozen in (FROZEN_RULE, FROZEN_TRAINING, FROZEN_NEW_TASK, FROZEN_LABELS, "Disclosure: " + FROZEN_DISCLOSURE):
        assert frozen in h, frozen
    assert "v5's joint rule verbatim, SLURM_JOBS.md line 3055" in h
    assert "R1H = R1 EXACTLY + these two changes ONLY, both chosen, not tuned" in h
    assert "No other change: rewards, observations (the 77-obs clock), events (the fixed pushes), DR and runner are R1's." in h
    sb = SBATCH.read_text()
    assert "#SBATCH --array=0-2%3" in sb and "#SBATCH --gres=gpu:1" in sb and "#SBATCH --job-name=turngait-hold" in sb


def test_launcher_arm_table():
    body = _body()
    rows = re.findall(r"^\s+(\S+)\) ARM=", body, re.M)
    assert rows == ["0|1|2"]                                                        # this arm only
    m = re.search(r"0\|1\|2\) ARM=(\w+); ARM_TASK=(\S+); +PREFIX=(\S+); +OBS_POLICY=(\d+); OBS_CRITIC=(\d+); EXPECT=(\w+);;", body)
    assert m and m.groups() == ("R1H", TASK, "arms-turngait-hold", "77", "80", "clock")
    assert vd.ARM["name"] == m.group(1) and vd.ARM["task"] == m.group(2) and vd.ARM["prefix"] == m.group(3)
    assert "SEED=$((IDX % 3))" in body and "FULL_ITERS=6000" in body
    assert 'export MAX_ITER=$FULL_ITERS RUN_NAME="$PREFIX-s${SEED}"' in body
    assert 'export MAX_ITER=3 NUM_ENVS=64 RUN_NAME="$PREFIX-s${SEED}-smoke"' in body
    assert "*-smoke) ;;" in body and re.search(r"SMOKE_RES=\S+/campaign-turning-hold/turngait-hold-smoke\n", body)
    assert "RES=$REPO/results/repo-gpu-20260923/turngait-hold-20261002" in body
    # from scratch: no resume, no parent checkpoint
    assert "agent.resume" not in body and "load_run" not in body and "load_checkpoint" not in body
    assert 'TASK="$ARM_TASK" bhl_exec "$REPO/slurm/inner/export_one_humanoid.sh"' in body
    assert 'TURNQ_RUN="$RUN_NAME" TURNQ_OUT_DIR="$RES/qualify" bash "$REPO/slurm/repo20260923/cpu_turn_qualify.sbatch"' in body
    assert "dr_overrides_scale 1.0" in body and "turngait_hold_verdict.py verdict --res" in body
    assert "gait_clock stamp --deploy" in body                                      # the 77-obs export, clock stamped
    assert 'REF_RUN=$(ls -d "$LOGROOT"/*_arms-turngait-clock-s0 2>/dev/null' in body   # recipe reference = R1
    assert "turngait_hold_verdict.py check-recipe" in body and '"HOLD RECIPE: OK"*' in body
    assert 'EXPORT_LOCK="$WORKSPACE/.turngait-r12-export.lock"' in body              # shared with R1/R2, platecross
    # R1/R2's launcher carries no R1H code: test_untouched_files_carry_no_turning_hold_code


def _bash_regexes():
    sb = SBATCH.read_text()
    return "\n".join(re.search(rf"^{k}=.*$", sb, re.M).group(0)
                     for k in ("TASK_RE", "GAIT_RE", "HOLD_RE", "MIXROW_RE", "MIX_RE", "PUSH_RE"))


def _guard(tmp_path, log_text):
    """The launcher's own arm_guard and clause regexes (extracted from the sbatch), run in bash on a log."""
    sb = SBATCH.read_text()
    fn = re.search(r"^arm_guard\(\) \{.*?^\}$", sb, re.S | re.M).group(0)
    log = tmp_path / "train.log"
    log.write_text(log_text)
    script = (f"set -euo pipefail\nARM_TASK={TASK}\nRUN_NAME=arms-turngait-hold-s0\nSEED=0\nOBS_POLICY=77\n"
              f"OBS_CRITIC=80\n{_bash_regexes()}\n{fn}\n"
              'why="$(arm_guard "$1")"\n'
              'grep -qE "$TASK_RE" "$1" || why="$why task"\n'
              'grep -qE "$GAIT_RE" "$1" || why="$why gait"\n'
              'grep -qE "$HOLD_RE" "$1" || why="$why hold"\n'
              '{ grep -qE "$MIXROW_RE" "$1" && grep -qE "$MIX_RE" "$1"; } || why="$why mix"\n'
              'grep -qE "$PUSH_RE" "$1" || why="$why push"\n'
              'echo "WHY:$why"\n')
    out = subprocess.run(["bash", "-c", script, "guard", str(log)], capture_output=True, text=True, check=True)
    return out.stdout.strip().split("WHY:", 1)[1].split()


MIX_LINE = ("[turn_command] TurnHoldMixVelocityCommand: explicit 0.3 zero_wz_p 0.5 pure 0.0 vx (-1.0, 1.0) "
            "vy (-0.5, 0.5) wz (-1.5, 1.5) heading_command True rel_heading 1.0 stiffness 0.5 rel_standing 0.02")
GOOD_LOG = f"""\
task={TASK} run=arms-turngait-hold-s0 seed=0 envs=cfg iters=6000
{MIX_LINE}
|   0   | base_velocity | TurnHoldMixVelocityCommand |
|   0   | push_robot |        (5.0, 9.0)       |
| Active Observation Terms in Group: 'policy' (shape: (77,)) |
| Active Observation Terms in Group: 'critic' (shape: (80,)) |
|   10  | feet_air_time              |    0.0 |
|   17  | feet_gait                  |    0.5 |
|   18  | feet_swing_height          |  -20.0 |
|   19  | heading_hold               |    1.0 |
[turn_command] TurnHoldMix first resample: n 4096 explicit 0.301 zero_wz 0.152 (of explicit 0.505) heading_cleared_iff_explicit True standing 0.019
[gait_clock_mdp] feet_gait bodies ['leg_left_ankle_roll', 'leg_right_ankle_roll'] offsets [0.0, 0.5] period 0.8 threshold 0.55 step_dt 0.04
[gait_clock_mdp] feet_swing_height bodies ['leg_left_ankle_roll', 'leg_right_ankle_roll']: link-origin z of feet in contact median 0.0600 m (n=8)
[gait_clock_mdp] heading_hold command base_velocity (TurnHoldMixVelocityCommand) std 0.2 wz_threshold 0.05: active in 640 of 4096 envs (explicit 1233)
train.sh: reached 6000 logged iterations
"""


def test_launcher_log_guard_accepts_the_r1h_log(tmp_path):
    assert _guard(tmp_path, GOOD_LOG) == []


def test_launcher_log_guard_rejects_r1_and_other_wrong_logs(tmp_path):
    r1_log = (GOOD_LOG.replace("TurnHoldMixVelocityCommand |", "UniformVelocityCommand |")
              .replace(MIX_LINE + "\n", "").replace("|   19  | heading_hold               |    1.0 |\n", ""))
    r1_log = "\n".join(ln for ln in r1_log.splitlines() if "heading_hold" not in ln and "TurnHoldMix" not in ln) + "\n"
    why = _guard(tmp_path, r1_log)
    for reason in ("command-class-is-not-TurnHoldMixVelocityCommand", "command-class-is-UniformVelocityCommand",
                   "heading_hold-did-not-run", "command-mix-did-not-resample-as-declared", "hold", "mix"):
        assert reason in why, (reason, why)
    assert "hold" in _guard(tmp_path, GOOD_LOG.replace("heading_hold               |    1.0", "heading_hold | 2.0"))
    assert "mix" in _guard(tmp_path, GOOD_LOG.replace("explicit 0.3 zero_wz_p 0.5", "explicit 0.25 zero_wz_p 0.5"))
    assert "mix" in _guard(tmp_path, GOOD_LOG.replace("rel_standing 0.02", "rel_standing 0.025"))
    assert "mix" in _guard(tmp_path, GOOD_LOG.replace(MIX_LINE + "\n", ""))
    assert "command-mix-did-not-resample-as-declared" in _guard(
        tmp_path, GOOD_LOG.replace("heading_cleared_iff_explicit True", "heading_cleared_iff_explicit False"))
    assert "heading_hold-did-not-run" in _guard(tmp_path, GOOD_LOG.replace("std 0.2 wz_threshold", "std 0.3 wz_threshold"))
    assert "task" in _guard(tmp_path, GOOD_LOG.replace(TASK, "Velocity-BHL-Arms-TurnGaitClock-v0"))
    assert "push" in _guard(tmp_path, GOOD_LOG.replace("(5.0, 9.0)", "(10.0, 15.0)"))
    assert "gait" in _guard(tmp_path, GOOD_LOG.replace("feet_gait                  |    0.5", "feet_gait | 1.0"))
    assert "policy-obs-not-77" in _guard(tmp_path, GOOD_LOG.replace("(shape: (77,))", "(shape: (75,))"))
    assert "unexpected-push_levels-curriculum" in _guard(tmp_path, GOOD_LOG + "|    0    | push_levels    |\n")


def test_guard_regexes_match_the_terms_real_prints(stack, tmp_path):
    """The log lines the guard looks for are the ones the code really prints (command term and reward term)."""
    cmd, w, out = _make(stack, 4096, seed=5)
    cmd._update_command()
    gm._LOGGED.discard("heading_hold")
    buf = io.StringIO()
    with redirect_stdout(buf):
        gm.heading_hold(_HoldEnv(cmd), "base_velocity", gm.HOLD_STD_RAD, gm.HOLD_WZ_THRESHOLD)
    real = out + buf.getvalue()
    assert MIX_LINE + "\n" in real
    log = GOOD_LOG
    for ln in GOOD_LOG.splitlines():
        if ln.startswith("[turn_command]") or ln.startswith("[gait_clock_mdp] heading_hold"):
            log = log.replace(ln + "\n", "")
    assert _guard(tmp_path, log + real) == []


def _record(rd, /, **kw):
    rec = {"run": "arms-turngait-hold-s0", "arm": "R1H", "task": TASK, "seed": 0, "final_ckpt": "model_5999.pt",
           "final_ckpt_present": True, "task_id_in_log": True, "feet_gait_in_log": True, "heading_hold_in_log": True,
           "command_mix_in_log": True, "push_robot_in_log": True, "run_dir": str(rd), "job": "1",
           "wrong_arm_guard": [], "recipe_check": "HOLD RECIPE: OK"}
    rec.update(kw)
    return rec


def _record_gate(tmp_path, fresh, record, why_train="", ckpt=True):
    """The launcher's own non-smoke training gate with its training_record_problems function, in bash."""
    sb = SBATCH.read_text()
    fn = re.search(r"^training_record_problems\(\) \{.*?^\}$", sb, re.S | re.M).group(0)
    block = re.search(r'^if \[ "\$SMOKE" != 1 \]; then\n    \{ \[ -n "\$run" \].*?^fi$', sb, re.S | re.M).group(0)
    run_dir = tmp_path / "2026-10-03_03-00-00_arms-turngait-hold-s0"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "model_5999.pt").unlink(missing_ok=True)
    if ckpt:
        (run_dir / "model_5999.pt").write_text("")
    tjson = tmp_path / "training" / "arms-turngait-hold-s0.json"
    tjson.parent.mkdir(exist_ok=True)
    tjson.unlink(missing_ok=True)
    if record is not None:
        tjson.write_text(record if isinstance(record, str) else json.dumps(record))
    script = (f"set -euo pipefail\nPYH={sys.executable}\nSMOKE=0\nARM=R1H\nRUN_NAME=arms-turngait-hold-s0\n"
              f"ARM_TASK={TASK}\nFINAL_CKPT=model_5999.pt\n"
              f'FRESH="$1"\nwhy_train="$2"\nrun="$3"\nTJSON="$4"\n{fn}\n{block}\necho GATES-WOULD-RUN\n')
    out = subprocess.run(["bash", "-c", script, "gate", str(int(fresh)), why_train, str(run_dir), str(tjson)],
                         capture_output=True, text=True)
    return out.returncode, out.stdout + out.stderr


def test_launcher_training_record_gate(tmp_path):
    rd = tmp_path / "2026-10-03_03-00-00_arms-turngait-hold-s0"
    for fresh in (0, 1):
        rc, out = _record_gate(tmp_path, fresh, _record(rd))
        assert rc == 0 and "GATES-WOULD-RUN" in out, out
    for change in ({"heading_hold_in_log": False}, {"command_mix_in_log": False}, {"task_id_in_log": False},
                   {"feet_gait_in_log": False}, {"push_robot_in_log": False}, {"final_ckpt_present": False},
                   {"final_ckpt": "model_2.pt"}, {"run_dir": str(tmp_path / "x")}, {"run": "arms-turngait-clock-s0"},
                   {"task": "Velocity-BHL-Arms-TurnGaitClock-v0"}, {"wrong_arm_guard": ["heading_hold-did-not-run"]},
                   {"heading_hold_in_log": "true"}):
        rc, out = _record_gate(tmp_path, 0, _record(rd, **change))
        assert rc == 1 and "GATES-WOULD-RUN" not in out and "seed stays INCOMPLETE" in out, (change, out)
    r1_style = _record(rd)
    del r1_style["heading_hold_in_log"], r1_style["command_mix_in_log"]
    rc, out = _record_gate(tmp_path, 0, r1_style)
    assert rc == 1 and "record-heading_hold_in_log-not-true" in out and "record-command_mix_in_log-not-true" in out
    rc, out = _record_gate(tmp_path, 0, "{not json")
    assert rc == 1 and "unreadable-training-record" in out
    rc, out = _record_gate(tmp_path, 0, None)
    assert rc == 1 and "no training record" in out
    rc, out = _record_gate(tmp_path, 0, _record(rd), why_train=" recipe-check-failed")
    assert rc == 1 and "seed stays INCOMPLETE" in out
    rc, out = _record_gate(tmp_path, 1, _record(rd), why_train=" heading_hold-did-not-run")
    assert rc == 1 and "no gates run, seed stays INCOMPLETE" in out
    rc, out = _record_gate(tmp_path, 0, _record(rd), ckpt=False)
    assert rc == 1 and "training incomplete" in out


def test_launcher_training_record_writer_names_the_six_flags():
    sb = SBATCH.read_text()
    for flag in vd.TRAINING_FLAGS:
        assert f'"{flag}": b(' in sb, flag
    assert f'"clause": "{FROZEN_TRAINING[:60]}' in sb.replace("\n", " ") or FROZEN_TRAINING[:60] in sb


def test_launcher_smoke_touches_no_scored_seed_and_probes_the_mix():
    sb = SBATCH.read_text()
    smoke = re.search(r'^if \[ "\$SMOKE" = 1 \]; then\n    why=.*?^fi$', sb, re.S | re.M).group(0)
    seeds = [int(s) for m in re.findall(r"--seed(?:s|0)?((?: \d+(?![\d>]))+)", smoke) for s in m.split()]
    assert len(seeds) == 4 and min(seeds) >= gc.EXPLORATION_SEED0 == 100, seeds   # clock smoke, v1, diagnose, run-eval
    assert "-m bhl_robust.eval.run_eval" not in smoke and "-m bhl_robust.eval.gait_clock run-eval" in smoke
    assert 'training_record_problems "$TJSON" "$run"' in smoke
    assert '--expect "$EXPECT"' in smoke and 'why="$why_train$probe_why$why_export"' in smoke
    probe = sb[sb.index('cat > "$PROBE_PY" <<\'EOF\''):sb.index('cat > "$PROBE_SH" <<\'EOF\'')]
    for needle in ('abs(E / N - 0.30) <= 0.02', 'abs(Z / E - 0.5) <= 0.03', 'abs(S / N - 0.02) <= 0.01',
                   'torch.equal(~head, direct)', '(v[zero, 2] == 0.0).all()', 'float(hc.weight) == 1.0',
                   'names == R1_TERMS + ["heading_hold"]', 'u._reset_idx(ids)', 'term._resample(ids2)',
                   'rm._step_reward[:, idx]', 'with open(a.out, "w")'):
        assert needle in probe, needle
    assert probe.index('with open(a.out, "w")') < probe.index("app.close()")      # verdict before the hard exit
    assert '--num_envs 64 --draws 300' in sb
    assert '"R1H-PROBE JSON: PASS"*' in sb
    assert "S=$(slurm_clean sbatch --parsable --job-name=turngait-hold-smoke --array=0 --time=01:00:00" in sb
    assert "--dependency=afterok:${S%%;*}" in sb


def test_launcher_bash_syntax():
    out = subprocess.run(["bash", "-n", str(SBATCH)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
