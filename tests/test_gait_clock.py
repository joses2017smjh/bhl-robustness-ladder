"""R1 / R2 turning recipe (docs/SOLUTIONS_2026-10-01.md section 2) without Isaac Sim.

Covers: the phase and clock math (Isaac side, `tasks/gait_clock_mdp.py`, loaded by path as
tests/test_stand_mdp.py does), `feet_gait` on synthetic contact sequences (zero command
included), the swing-height term, the deploy-side clock controller (`eval/gait_clock.py`)
against an independent reference and against the Isaac clock, byte-identity of the unchanged
75-observation path, the env.yaml recipe checks, the per-arm verdict at its boundaries, and
the launcher header / arm table / log guards. The MuJoCo tests at the end replay a stand-in
77-wide ONNX through turn_test.py's and run_eval's real paths; they skip without the
upstream assets or the arms-turn-turnboth-s0 export.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
MDP_PATH = REPO / "src/bhl_robust/tasks/gait_clock_mdp.py"
ARMS_CFG = REPO / "src/bhl_robust/tasks/arms_env_cfg.py"
TASKS_INIT = REPO / "src/bhl_robust/tasks/__init__.py"
VERDICT_PATH = REPO / "scripts/bench/turngait_r12_verdict.py"
SBATCH = REPO / "slurm/repo20260923/gpu_turngait_r12.sbatch"
QUALIFY = REPO / "slurm/repo20260923/cpu_turn_qualify.sbatch"
UPSTREAM = REPO / "external/Berkeley-Humanoid-Lite"
LOGROOT = UPSTREAM / "logs/rsl_rl/humanoid"

# The frozen rule, word for word as the task states it.
FROZEN_RULE = ("PASS iff >= 2/3 seeds both PASS turn_test v2 AND are QUALIFIED by cpu_turn_qualify's unchanged rule "
               "(v2x >= 9/10 on reset seeds 10-14, walk <= 15 deg on >= 2/3, push <= 9/60); else FAIL; INCOMPLETE if "
               "any JSON is missing. A seed that fails v2 does not count even if it qualifies through v2x.")
FROZEN_TRAINING = ("Training counts only if the run dir holds model_5999.pt and the training log shows the arm's task "
                   "id, the feet_gait term and the push event.")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gm = _load("gait_clock_mdp_under_test", MDP_PATH)
vd = _load("turngait_r12_verdict_under_test", VERDICT_PATH)
from bhl_robust.eval import gait_clock as gc  # noqa: E402  (numpy only at import)


# =============================================================================== Isaac-side terms

class _Sensor:
    def __init__(self, contact_time=None, forces=None, names=("leg_left_ankle_roll", "leg_right_ankle_roll")):
        self.data = SimpleNamespace(current_contact_time=contact_time, net_forces_w=forces)
        self.body_names = list(names)


class _Asset:
    def __init__(self, z, names=("leg_left_ankle_roll", "leg_right_ankle_roll")):
        pos = torch.zeros(z.shape[0], z.shape[1], 3)
        pos[..., 2] = z
        self.data = SimpleNamespace(body_pos_w=pos)
        self.body_names = list(names)


class _Scene:
    def __init__(self, sensors=None, assets=None):
        self.sensors = sensors or {}
        self._assets = assets or {}

    def __getitem__(self, key):
        return self._assets[key]


class _Env:
    """episode_length_buf / step_dt / scene, and NO command manager: a term that read the
    command would raise, so passing these tests proves the reward is not command-gated."""

    def __init__(self, steps, scene=None, step_dt=0.04):
        self.episode_length_buf = torch.as_tensor(steps, dtype=torch.long)
        self.step_dt = step_dt
        self.num_envs = int(self.episode_length_buf.shape[0])
        self.device = "cpu"
        self.scene = scene

    @property
    def command_manager(self):
        raise AssertionError("the R1/R2 terms must not read the command")


FEET = SimpleNamespace(name="contact_forces", body_ids=[0, 1])
ROBOT_FEET = SimpleNamespace(name="robot", body_ids=[0, 1])


def test_mdp_imports_without_isaaclab():
    had = {k for k in sys.modules if k == "isaaclab" or k.startswith("isaaclab.")}
    _load("gait_clock_mdp_reload", MDP_PATH)
    assert {k for k in sys.modules if k == "isaaclab" or k.startswith("isaaclab.")} - had == set()


def test_frozen_constants():
    assert gm.GAIT_PERIOD_S == 0.8 and gm.GAIT_OFFSETS == (0.0, 0.5) and gm.STANCE_THRESHOLD == 0.55
    assert gm.FEET_GAIT_WEIGHT == 0.5 and gm.SWING_HEIGHT_WEIGHT == -20.0 and gm.SWING_HEIGHT_TARGET_M == 0.05
    assert gm.CONTACT_FORCE_THRESHOLD_N == 1.0 and gm.PUSH_VELOCITY_MPS == 0.5
    assert gm.FOOT_ORIGIN_ABOVE_SOLE_M == 0.060
    # the deploy side and the env.yaml recipe check carry the same numbers
    assert gc.EXPECTED_PERIOD_S == gm.GAIT_PERIOD_S and gc.EXPECTED_PHASE_OFFSET == gm.GAIT_OFFSETS[0]
    r = gc.RECIPE
    assert r["feet_gait"]["weight"] == gm.FEET_GAIT_WEIGHT and r["feet_gait"]["period"] == gm.GAIT_PERIOD_S
    assert r["feet_gait"]["offset"] == list(gm.GAIT_OFFSETS) and r["feet_gait"]["threshold"] == gm.STANCE_THRESHOLD
    assert r["feet_swing_height"]["weight"] == gm.SWING_HEIGHT_WEIGHT
    assert r["feet_swing_height"]["target_height"] == gm.SWING_HEIGHT_TARGET_M
    assert r["feet_swing_height"]["foot_height_offset"] == gm.FOOT_ORIGIN_ABOVE_SOLE_M
    assert r["feet_swing_height"]["force_threshold"] == gm.CONTACT_FORCE_THRESHOLD_N
    assert r["push_robot"]["velocity_range"] == {"x": [-0.5, 0.5], "y": [-0.5, 0.5]}
    assert r["push_robot"]["interval_range_s"] == [5.0, 9.0] and r["feet_air_time_weight"] == 0.0


def _circ(a, b):
    d = (torch.as_tensor(a, dtype=torch.float64) - torch.as_tensor(b, dtype=torch.float64)).abs() % 1.0
    return torch.minimum(d, 1.0 - d)


def test_global_phase_and_leg_phases():
    steps = torch.tensor([0, 5, 10, 15, 19, 20, 25, 500])
    ph = gm.global_phase(steps, 0.04, 0.8)
    want = [0.0, 0.25, 0.5, 0.75, 0.95, 0.0, 0.25, 0.0]       # 20 policy steps per 0.8 s period
    # float32, as Isaac computes it: at k = 20 the product rounds just below 0.8, so the phase reads
    # 1 - 6e-8 instead of 0 -- the same point on the circle (the clock observation is continuous).
    assert float(_circ(ph, want).max()) < 1e-5
    assert bool(((ph >= 0) & (ph < 1)).all())
    legs = gm.leg_phases(torch.tensor([0.25, 0.6, 0.0]), (0.0, 0.5))
    assert torch.allclose(legs, torch.tensor([[0.25, 0.75], [0.6, 0.1], [0.0, 0.5]]), atol=1e-6)


def test_stance_threshold_is_strict():
    ph = torch.tensor([[0.5499, 0.55]])
    # left at 0.5499 is stance, right at 0.55 is swing: contact on both pays 1 (left) + 0 (right)
    assert float(gm.schedule_reward(ph, torch.tensor([[True, True]]), 0.55)[0]) == 1.0
    assert float(gm.schedule_reward(ph, torch.tensor([[True, False]]), 0.55)[0]) == 2.0


def test_gait_clock_observation_values():
    env = _Env([0, 5, 10, 15, 20])
    out = gm.gait_clock(env, 0.8)
    assert out.shape == (5, 2)
    ref = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0], [0.0, 1.0]])
    assert torch.allclose(out, ref, atol=1e-5)


def test_gait_clock_without_episode_buffer_reads_zero():
    env = SimpleNamespace(num_envs=3, device="cpu", step_dt=0.04)       # a non-RL env (load-time probe)
    out = gm.gait_clock(env, 0.8)
    assert torch.allclose(out, torch.tensor([[0.0, 1.0]] * 3)) and env.episode_length_buf.dtype == torch.long


def _schedule(k):
    """(left, right) stance of step k on Isaac's own float32 clock (the exact-arithmetic schedule
    differs from it only at period boundaries, where float32 rounding moves a step across)."""
    legs = gm.leg_phases(gm.global_phase(torch.tensor([k]), 0.04, 0.8), (0.0, 0.5))[0]
    return bool(legs[0] < 0.55), bool(legs[1] < 0.55)


def _gait_reward(k, left, right):
    ct = torch.tensor([[0.1 if left else 0.0, 0.1 if right else 0.0]])
    env = _Env([k], scene=_Scene(sensors={"contact_forces": _Sensor(contact_time=ct)}))
    return float(gm.feet_gait(env, 0.8, [0.0, 0.5], 0.55, FEET)[0])


def test_feet_gait_follows_the_schedule_and_pays_at_zero_command():
    on_schedule, anti, standing, hopping, n_stance = [], [], [], [], []
    for k in range(40):                                              # two periods, phase 0 at k = 0
        sl, sr = _schedule(k)
        n_stance.append(sl + sr)
        on_schedule.append(_gait_reward(k, sl, sr))
        anti.append(_gait_reward(k, not sl, not sr))
        standing.append(_gait_reward(k, True, True))                  # a standing policy at zero command
        hopping.append(_gait_reward(k, False, False))
    assert on_schedule == [2.0] * 40 and anti == [0.0] * 40
    # standing is paid only for the feet the schedule puts in stance: 1.1 per step on average
    # (stance 0.55 of the period per foot), against 2.0 for following the schedule
    assert standing == [float(n) for n in n_stance] and hopping == [2.0 - n for n in n_stance]
    assert abs(np.mean(standing) - 1.1) <= 0.051 and np.mean(on_schedule) - np.mean(standing) > 0.85
    # mid-phase: left stance / right swing at 0.12 s, the reverse at 0.52 s
    assert _schedule(3) == (True, False) and _schedule(13) == (False, True)
    assert _gait_reward(3, True, False) == 2.0 and _gait_reward(13, False, True) == 2.0
    # the reward reads only the clock and the contacts: the env has no command manager at all,
    # so the same contacts are paid identically whatever the command (zero included)


def test_feet_gait_rejects_a_foot_count_mismatch():
    ct = torch.zeros(1, 3)
    env = _Env([0], scene=_Scene(sensors={"contact_forces": _Sensor(contact_time=ct, names=("a", "b", "c"))}))
    with pytest.raises(ValueError):
        gm.feet_gait(env, 0.8, [0.0, 0.5], 0.55, SimpleNamespace(name="contact_forces", body_ids=[0, 1, 2]))


def test_swing_height_penalty_math():
    z = torch.tensor([[0.05, 0.10], [0.0, 0.0], [0.15, 0.15]])
    c = torch.tensor([[False, False], [True, True], [False, True]])
    pen = gm.swing_height_penalty(z, c, 0.05)
    assert torch.allclose(pen, torch.tensor([0.0025, 0.0, 0.01]), atol=1e-7)


def _swing(z_origin, force_n):
    f = torch.zeros(1, 2, 3)
    f[0, :, 2] = torch.tensor(force_n)
    env = _Env([0], scene=_Scene(sensors={"contact_forces": _Sensor(forces=f)},
                                 assets={"robot": _Asset(torch.tensor([z_origin]))}))
    return float(gm.feet_swing_height(env, 0.05, FEET, ROBOT_FEET, foot_height_offset=0.06, force_threshold=1.0)[0])


def test_feet_swing_height_reads_the_sole_and_the_contact_force():
    assert _swing([0.11, 0.06], [0.0, 50.0]) == pytest.approx(0.0, abs=1e-9)           # swing sole at h; stance foot free
    assert _swing([0.06, 0.06], [0.0, 0.0]) == pytest.approx(2 * 0.05 ** 2, rel=1e-5)  # both unloaded, origin at 0.06 m
    assert _swing([0.16, 0.06], [0.5, 2.0]) == pytest.approx(0.05 ** 2, rel=1e-5)      # 0.5 N is not contact
    assert _swing([0.16, 0.06], [1.0, 1.0001]) == pytest.approx(0.05 ** 2, rel=1e-4)   # contact is > 1 N strictly
    # the literal unitree reading (offset 0) measures the origin itself: (0.06 - 0.05)^2 per unloaded foot
    f = torch.zeros(1, 2, 3)
    env = _Env([0], scene=_Scene(sensors={"contact_forces": _Sensor(forces=f)},
                                 assets={"robot": _Asset(torch.tensor([[0.06, 0.06]]))}))
    lit = float(gm.feet_swing_height(env, 0.05, FEET, ROBOT_FEET, foot_height_offset=0.0)[0])
    assert lit == pytest.approx(2 * 0.01 ** 2, rel=1e-4)


# =============================================================================== task config (source)

def test_arms_env_cfg_appends_only_and_wires_the_frozen_terms():
    src = ARMS_CFG.read_text()
    marker = src.index("# --- Turning gait R1 / R2")
    assert src.index("class HumanoidTurnRestPushCfg") < marker                 # everything new comes after
    tree = ast.parse(src)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    new = ["TurnGaitRewardsCfg", "ArmsFixedPushEventsCfg", "_GaitClockPolicyObsCfg", "_GaitClockCriticObsCfg",
           "GaitClockActorObservationsCfg", "GaitClockCriticObservationsCfg", "_HumanoidTurnGaitBaseCfg",
           "HumanoidTurnGaitClockCfg", "HumanoidTurnGaitCriticCfg"]
    for name in new:
        assert src.index(f"class {name}(") > marker
    base = lambda c: [ast.unparse(b) for b in classes[c].bases]  # noqa: E731
    assert base("_HumanoidTurnGaitBaseCfg") == ["HumanoidTurnBothCfg"]
    assert base("HumanoidTurnGaitClockCfg") == ["_HumanoidTurnGaitBaseCfg"]
    assert base("HumanoidTurnGaitCriticCfg") == ["_HumanoidTurnGaitBaseCfg"]
    base_src = ast.get_source_segment(src, classes["_HumanoidTurnGaitBaseCfg"])
    assert "self.rewards.feet_air_time.weight = 0.0" in base_src and "curriculum" not in base_src
    push_src = ast.get_source_segment(src, classes["ArmsFixedPushEventsCfg"])
    assert "interval_range_s=PUSH_INTERVAL_S" in push_src and "PUSH_VELOCITY_MPS" in push_src
    rew_src = ast.get_source_segment(src, classes["TurnGaitRewardsCfg"])
    assert "command_name" not in rew_src                                         # never command-gated
    for k in ("GAIT_PERIOD_S", "GAIT_OFFSETS", "STANCE_THRESHOLD", "FEET_GAIT_WEIGHT", "SWING_HEIGHT_WEIGHT",
              "SWING_HEIGHT_TARGET_M", "FOOT_ORIGIN_ABOVE_SOLE_M", "CONTACT_FORCE_THRESHOLD_N"):
        assert f"gait_clock_mdp.{k}" in rew_src
    assert '_FEET_LR = [".*_left_ankle_roll", ".*_right_ankle_roll"]' in src
    # R1: clock in the actor and the critic; R2: the critic only (the actor keeps upstream's PolicyCfg)
    assert "policy: _GaitClockPolicyObsCfg" in ast.get_source_segment(src, classes["GaitClockActorObservationsCfg"])
    r2 = ast.get_source_segment(src, classes["GaitClockCriticObservationsCfg"])
    assert "critic: _GaitClockCriticObsCfg" in r2 and "policy:" not in r2


def test_task_registrations():
    src = TASKS_INIT.read_text()
    assert '("Velocity-BHL-Arms-TurnGaitClock-v0", arms_env_cfg.HumanoidTurnGaitClockCfg),' in src
    assert '("Velocity-BHL-Arms-TurnGaitCritic-v0", arms_env_cfg.HumanoidTurnGaitCriticCfg),' in src
    loop = src[src.index('("Velocity-BHL-Arms-TurnHip-v0"'):]
    loop = loop[:loop.index("gym.register(") + 400]
    assert "TurnGaitClock" in loop and '"rsl_rl_cfg_entry_point": _ARM_PPO_CFG' in loop   # TurnBoth's runner


# =============================================================================== deploy-side controller

N_ACT = 22
DEFAULT_Q = np.linspace(-0.3, 0.3, N_ACT).astype(np.float32)


def _cfg(width=75, clock=False, history=0):
    from omegaconf import OmegaConf
    d = {"policy_checkpoint_path": "/nonexistent/policy.onnx", "policy_dt": 0.04, "num_joints": N_ACT,
         "num_actions": N_ACT, "num_observations": width, "history_length": history, "action_scale": 0.25,
         "action_limit_lower": -10000, "action_limit_upper": 10000, "command_velocity": [0.0, 0.0, 0.0],
         "default_joint_positions": DEFAULT_Q.tolist()}
    if clock:
        d["gait_clock"] = {"period_s": 0.8, "phase_offset": 0.0}
    return OmegaConf.create(d)


class _RecPolicy:
    def __init__(self, width, seed=0):
        self.W = np.random.default_rng(seed).normal(0.0, 0.1, (width, N_ACT)).astype(np.float32)
        self.inputs = []

    def forward(self, obs):
        self.inputs.append(np.array(obs, copy=True))
        return np.tanh(obs @ self.W).astype(np.float32)


def _robot_obs(n, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        q = rng.normal(0, 1, 4); q[0] += 4.0; q /= np.linalg.norm(q)           # near upright, unit
        out.append(np.concatenate([q, rng.normal(0, 0.5, 3), DEFAULT_Q + rng.normal(0, 0.1, N_ACT),
                                   rng.normal(0, 1.0, N_ACT), [3.0], rng.uniform(-1, 1, 3)]).astype(np.float32))
    return out


def _gravity_ref(q):
    w, x, y, z = (float(v) for v in q)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R.T @ np.array([0.0, 0.0, -1.0])                                     # world gravity in the body frame


def test_has_clock():
    from omegaconf import OmegaConf
    assert not gc.has_clock(_cfg()) and gc.has_clock(_cfg(77, clock=True))
    assert not gc.has_clock(OmegaConf.create({"gait_clock": None}))


def test_75_obs_path_is_upstream_rlcontroller_byte_identical():
    pytest.importorskip("berkeley_humanoid_lite_lowlevel")
    from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
    obs = _robot_obs(60)
    new, old = gc.make_controller(_cfg()), RlController(_cfg())                # after vs before the change
    assert type(new) is RlController
    new.policy, old.policy = _RecPolicy(75), _RecPolicy(75)
    for i, o in enumerate(obs):
        if i == 30:                                                            # a harness reset mid-sequence
            for c in (new, old):
                c.prev_actions[:] = 0.0
                c.policy_observations[:] = 0.0
        assert np.array_equal(new.update(o.copy()), old.update(o.copy()))
    assert all(np.array_equal(a, b) for a, b in zip(new.policy.inputs, old.policy.inputs))
    # a 77-wide config WITHOUT the stamped block is not silently given a clock
    assert type(gc.make_controller(_cfg(77))) is RlController


def test_harnesses_build_through_make_controller():
    for path, line in ((REPO / "scripts/bench/turn_test.py", "ctrl = make_controller(cfg)"),
                       (REPO / "scripts/bench/turn_diagnose.py", "ctrl = make_controller(cfg)"),
                       (REPO / "src/bhl_robust/eval/run_eval.py", "controller = make_controller(cfg)")):
        src = path.read_text()
        code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())          # comments stripped
        assert line in code and "RlController(cfg)" not in code, path
        assert "from bhl_robust.eval.gait_clock import make_controller" in code
    q = QUALIFY.read_text()                       # qualify runs exactly those two entry points, unchanged
    assert "scripts/bench/turn_test.py --protocol v2x" in q and "-m bhl_robust.eval.run_eval" in q


def test_clock_math_matches_isaac_term():
    env = _Env(list(range(0, 520)))
    isaac = gm.gait_clock(env, 0.8).numpy()
    deploy = np.stack([gc.clock_features(k, 0.04, 0.8) for k in range(0, 520)])
    assert np.max(np.abs(isaac - deploy)) < 2e-6
    # the same float32 phase as Isaac, bit for bit (k = 20 reads 1 - 6e-8, not 0: see test_global_phase...)
    isaac_phase = gm.global_phase(torch.arange(0, 520), 0.04, 0.8).numpy()
    assert all(np.float32(gc.clock_phase(k, 0.04, 0.8)) == isaac_phase[k] for k in range(0, 520))
    assert float(_circ(gc.clock_phase(20, 0.04, 0.8), 0.0)) < 1e-6
    assert gc.clock_phase(7, 0.04, 0.8) == pytest.approx(0.35, abs=1e-6)
    assert gc.base_obs_width(22) == 75


def test_77_obs_controller_against_reference():
    pytest.importorskip("berkeley_humanoid_lite_lowlevel")
    from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
    ctrl = gc.make_controller(_cfg(77, clock=True))
    assert type(ctrl).__name__ == "GaitClockRlController" and isinstance(ctrl, RlController)
    ctrl.policy = _RecPolicy(77)
    obs = _robot_obs(45, seed=3)
    prev = np.zeros(N_ACT, np.float32)
    outs = []
    for k, o in enumerate(obs):
        if k == 33:                                                            # MultiRunner.reset / run_episode
            ctrl.prev_actions[:] = 0.0
            ctrl.policy_observations[:] = 0.0
            prev = np.zeros(N_ACT, np.float32)
        step = k if k < 33 else k - 33
        out = ctrl.update(o.copy())
        frame = ctrl.policy.inputs[-1][0]
        n = N_ACT
        ph = ((step * 0.04) % 0.8) / 0.8
        ref = np.concatenate([o[7 + 2 * n + 1:7 + 2 * n + 4], o[4:7], _gravity_ref(o[0:4]), o[7:7 + n] - DEFAULT_Q,
                              o[7 + n:7 + 2 * n], prev, [math.sin(2 * math.pi * ph), math.cos(2 * math.pi * ph)]])
        assert frame.shape == (77,)
        np.testing.assert_allclose(frame, ref, atol=2e-5)
        assert ctrl.clock_step == step + 1
        act = np.tanh(ctrl.policy.inputs[-1] @ ctrl.policy.W)[0]
        np.testing.assert_allclose(out, act * 0.25 + DEFAULT_Q, atol=1e-6)
        prev = act.astype(np.float32)
        outs.append(out)
    # the first 75 entries are exactly what upstream assembles from the same inputs and history
    plain = RlController(_cfg(75))
    replay = iter([np.tanh(x @ ctrl.policy.W) for x in ctrl.policy.inputs])

    class _Replay:
        def forward(self, x):
            return next(replay)
    plain.policy = _Replay()
    frames75 = []
    for k, o in enumerate(obs):
        if k == 33:
            plain.prev_actions[:] = 0.0
            plain.policy_observations[:] = 0.0
        plain.update(o.copy())
        frames75.append(plain.policy_observations[0].copy())
    for k, f in enumerate(frames75):
        assert np.array_equal(f, ctrl.policy.inputs[k][0][:75])


def test_clock_controller_refuses_bad_configs():
    pytest.importorskip("berkeley_humanoid_lite_lowlevel")
    with pytest.raises(ValueError):
        gc.make_controller(_cfg(76, clock=True))
    with pytest.raises(ValueError):
        gc.make_controller(_cfg(77, clock=True, history=1))


# =============================================================================== env.yaml: stamp and recipe

def _term(func, params=None, **kw):
    return {"func": func, "params": params or {}, **kw}


def _turnboth_env():
    obs = {k: _term(f"isaaclab.envs.mdp.observations:{k}") for k in
           ("velocity_commands", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions")}
    policy = {"concatenate_terms": True, "enable_corruption": True, **obs}
    critic = {"concatenate_terms": True, "enable_corruption": False, **obs,
              "base_lin_vel": _term("isaaclab.envs.mdp.observations:base_lin_vel")}
    feet = {"name": "contact_forces", "body_names": ".*_ankle_roll", "preserve_order": False}
    return {
        "sim": {"dt": 0.005}, "decimation": 8, "seed": 0,
        "observations": {"policy": policy, "critic": critic},
        "rewards": {"track_ang_vel_z_exp": {"func": "x:track_ang_vel_z_world_exp", "params": {"std": 0.25}, "weight": 2.0},
                    "feet_air_time": {"func": "bhl_robust.tasks.arms_env_cfg:feet_air_time_positive_biped_turn",
                                      "params": {"command_name": "base_velocity", "sensor_cfg": feet, "threshold": 0.5},
                                      "weight": 2.0},
                    "joint_deviation_hip": {"func": "x:joint_deviation_l1", "params": {}, "weight": -0.2}},
        "events": {"physics_material": {"func": "x:randomize_rigid_body_material",
                                        "params": {"static_friction_range": [0.4, 1.2]}, "mode": "startup"}},
        "commands": {"base_velocity": {"class_type": "isaaclab.envs.mdp.commands.velocity_command:UniformVelocityCommand",
                                       "rel_standing_envs": 0.02}},
        "terminations": {"time_out": {"func": "x:time_out"}}, "actions": {"joint_pos": {"scale": 0.25}},
        "curriculum": None,
    }


def _arm_env(arm="R1"):
    import copy
    env = copy.deepcopy(_turnboth_env())
    clock = _term("bhl_robust.tasks.gait_clock_mdp:gait_clock", {"period": 0.8})
    if arm == "R1":
        env["observations"]["policy"]["gait_clock"] = clock
    env["observations"]["critic"]["gait_clock"] = copy.deepcopy(clock)
    feet = {"name": "contact_forces", "body_names": [".*_left_ankle_roll", ".*_right_ankle_roll"], "preserve_order": True}
    env["rewards"]["feet_air_time"]["weight"] = 0.0
    env["rewards"]["feet_gait"] = {"func": "bhl_robust.tasks.gait_clock_mdp:feet_gait", "weight": 0.5,
                                   "params": {"period": 0.8, "offset": [0.0, 0.5], "threshold": 0.55, "sensor_cfg": feet}}
    env["rewards"]["feet_swing_height"] = {"func": "bhl_robust.tasks.gait_clock_mdp:feet_swing_height", "weight": -20.0,
                                           "params": {"target_height": 0.05, "foot_height_offset": 0.06,
                                                      "force_threshold": 1.0, "sensor_cfg": feet}}
    env["events"]["push_robot"] = {"func": "isaaclab.envs.mdp.events:push_by_setting_velocity", "mode": "interval",
                                   "interval_range_s": [5.0, 9.0],
                                   "params": {"velocity_range": {"x": [-0.5, 0.5], "y": [-0.5, 0.5]}}}
    return env


def test_recipe_checks_accept_the_recipe():
    for arm in ("R1", "R2"):
        env = _arm_env(arm)
        assert gc.check_recipe(env, arm) == []
        assert gc.recipe_diff(env, _turnboth_env(), arm) == []
        assert gc.check_env(env, arm)["step_dt"] == pytest.approx(0.04)
    assert gc.check_recipe(_arm_env("R1"), "R2") != []                # R1 layout is not R2
    assert gc.check_recipe(_arm_env("R2"), "R1") != []


@pytest.mark.parametrize("mutate, needle", [
    (lambda e: e["rewards"]["feet_gait"].update(weight=1.0), "feet_gait weight"),
    (lambda e: e["rewards"]["feet_gait"]["params"].update(period=0.6), "feet_gait period"),
    (lambda e: e["rewards"]["feet_gait"]["params"].update(offset=[0.5, 0.0]), "feet_gait offset"),
    (lambda e: e["rewards"]["feet_gait"]["params"].update(command_name="base_velocity"), "command-gated"),
    (lambda e: e["rewards"]["feet_swing_height"]["params"].update(target_height=0.08), "target_height"),
    (lambda e: e["rewards"]["feet_swing_height"].update(weight=-10.0), "feet_swing_height weight"),
    (lambda e: e["rewards"]["feet_air_time"].update(weight=2.0), "feet_air_time weight"),
    (lambda e: e["events"]["push_robot"]["params"]["velocity_range"].update(x=[-1.0, 1.0]), "velocity_range"),
    (lambda e: e["events"]["push_robot"].update(interval_range_s=[10.0, 15.0]), "push_robot interval"),
    (lambda e: e.update(curriculum={"push_levels": {"func": "bhl_robust.curricula.push:push_levels_adaptive"}}), "curriculum"),
])
def test_recipe_check_catches_each_change(mutate, needle):
    env = _arm_env("R1")
    mutate(env)
    probs = gc.check_recipe(env, "R1")
    assert any(needle in p for p in probs), probs


def test_recipe_diff_catches_a_change_outside_the_recipe():
    env = _arm_env("R2")
    env["events"]["physics_material"]["params"]["static_friction_range"] = [0.2, 1.4]     # different DR
    env["commands"]["base_velocity"]["rel_standing_envs"] = 0.1                          # different command mix
    env["rewards"]["joint_deviation_hip"]["weight"] = -1.0                               # not TurnBoth's rewards
    d = gc.recipe_diff(env, _turnboth_env(), "R2")
    assert any("static_friction_range" in x for x in d) and any("rel_standing_envs" in x for x in d)
    assert any("joint_deviation_hip.weight" in x for x in d)
    env = _arm_env("R2")
    env["observations"]["policy"]["gait_clock"] = env["observations"]["critic"]["gait_clock"]
    assert any("policy.gait_clock" in x for x in gc.recipe_diff(env, _turnboth_env(), "R2"))


ENV_YAML_TAGS = """\
sim:
  dt: 0.005
decimation: 8
observations:
  policy:
    concatenate_terms: true
    actions:
      func: isaaclab.envs.mdp.observations:last_action
      params: {}
    gait_clock:
      func: bhl_robust.tasks.gait_clock_mdp:gait_clock
      params:
        period: 0.8
  critic:
    base_lin_vel:
      func: isaaclab.envs.mdp.observations:base_lin_vel
      params: {}
    gait_clock:
      func: bhl_robust.tasks.gait_clock_mdp:gait_clock
      params:
        period: 0.8
rewards:
  feet_gait:
    func: bhl_robust.tasks.gait_clock_mdp:feet_gait
    params:
      period: 0.8
      offset:
      - 0.0
      - 0.5
      threshold: 0.55
      sensor_cfg:
        name: contact_forces
        body_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
    weight: 0.5
events:
  push_robot:
    interval_range_s: !!python/tuple
    - 5.0
    - 9.0
"""


def test_load_env_yaml_reads_isaac_tags(tmp_path):
    p = tmp_path / "env.yaml"
    p.write_text(ENV_YAML_TAGS)
    env = gc.load_env_yaml(p)
    assert env["events"]["push_robot"]["interval_range_s"] == [5.0, 9.0]
    assert gc.check_env(env, "R1") == {"period_s": 0.8, "phase_offset": 0.0, "step_dt": pytest.approx(0.04)}


def test_stamp_appends_once_and_leaves_the_export_byte_identical(tmp_path):
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text(ENV_YAML_TAGS)
    deploy = tmp_path / "deploy.yaml"
    text = "policy_checkpoint_path: /x/policy.onnx\npolicy_dt: 0.04\nnum_actions: 22\nnum_observations: 77\n"
    deploy.write_text(text)
    assert gc.stamp(deploy, env_yaml).startswith("stamped")
    after = deploy.read_text()
    assert after.startswith(text) and "\ngait_clock:\n  period_s: 0.8\n  phase_offset: 0.0\n" in after
    assert gc.stamp(deploy, env_yaml).startswith("already stamped") and deploy.read_text() == after
    deploy.write_text(text.replace("77", "75"))
    with pytest.raises(ValueError):
        gc.stamp(deploy, env_yaml)                                  # a 75-wide export is not an R1 export
    r2 = tmp_path / "env_r2.yaml"
    r2.write_text(ENV_YAML_TAGS.replace("    gait_clock:\n      func: bhl_robust.tasks.gait_clock_mdp:gait_clock\n"
                                        "      params:\n        period: 0.8\n  critic:", "  critic:"))
    deploy.write_text(text)
    with pytest.raises(ValueError):
        gc.stamp(deploy, r2)                                        # no clock in the actor: R2, never stamped


# =============================================================================== per-arm verdict

def _seed(training=True, v2="PASS", qualify="QUALIFIED"):
    return {"training": training, "v2": v2, "qualify": qualify}


def test_rule_text_is_the_frozen_rule():
    assert vd.RULE == FROZEN_RULE and vd.TRAINING_RULE == FROZEN_TRAINING and vd.LABEL == "LEARNED gait; MuJoCo gates"
    assert vd.FINAL_CKPT == "model_5999.pt" and vd.NEED == 2 and vd.SEEDS == (0, 1, 2)


def test_verdict_boundaries():
    v = vd.arm_verdict
    assert v([_seed(), _seed(), _seed()])["verdict"] == "PASS"
    r = v([_seed(), _seed(), _seed(v2="FAIL")])
    assert r["verdict"] == "PASS" and r["n_counted"] == 2                         # exactly 2/3
    assert v([_seed(), _seed(v2="FAIL"), _seed(v2="FAIL")])["verdict"] == "FAIL"   # 1/3
    assert v([_seed(v2="FAIL")] * 3)["verdict"] == "FAIL"                          # 0/3
    # v2 PASS but NOT QUALIFIED does not count; QUALIFIED but v2 FAIL does not count
    assert v([_seed(), _seed(qualify="NOT QUALIFIED"), _seed(v2="FAIL")])["verdict"] == "FAIL"
    # training that does not count makes the seed not count, whatever its gates say
    assert v([_seed(), _seed(training=False), _seed(v2="FAIL")])["verdict"] == "FAIL"
    # any missing JSON -> INCOMPLETE, even with 2 seeds already counted
    r = v([_seed(), _seed(), _seed(qualify=None)])
    assert r["verdict"] == "INCOMPLETE" and r["n_counted"] == 2 and r["missing_seeds"] == [2]
    assert v([_seed(), _seed(), None])["verdict"] == "INCOMPLETE"
    assert v([_seed(training=None), _seed(), _seed()])["verdict"] == "INCOMPLETE"
    with pytest.raises(ValueError):
        v([_seed(), _seed()])


def _training(**kw):
    t = {"run": "r", "final_ckpt": "model_5999.pt", "final_ckpt_present": True, "task_id_in_log": True,
         "feet_gait_in_log": True, "push_robot_in_log": True}
    t.update(kw)
    return t


def test_training_clause():
    assert vd.training_counts(_training())
    for bad in ({"final_ckpt": "model_2.pt"}, {"final_ckpt_present": False}, {"task_id_in_log": False},
                {"feet_gait_in_log": False}, {"push_robot_in_log": False}):
        assert not vd.training_counts(_training(**bad)), bad


def _v2_json(run, verdict="PASS", **rule):
    r = dict(vd.V2_RULE)
    r.update(rule)
    return {"deploy": f"/u/logs/rsl_rl/humanoid/2026-10-02_00-00-00_{run}/exported/deploy.yaml", "protocol": "v2",
            "verdict": verdict, "rule": r, "turns": [], "walk": {}}


def _write_seed(res, run, v2="PASS", qualify="QUALIFIED", training=None, v2json=None):
    for sub in ("training", "turn-test-v2", "qualify"):
        (res / sub).mkdir(parents=True, exist_ok=True)
    (res / "training" / f"{run}.json").write_text(json.dumps(training or _training(run=run)))
    (res / "turn-test-v2" / f"{run}.json").write_text(json.dumps(v2json or _v2_json(run, v2)))
    (res / "qualify" / f"{run}__qualify.json").write_text(json.dumps({"run": run, "verdict": qualify,
                                                                      "clauses": {"push": {"rate": 0.1}}}))


def test_read_seed_accepts_only_this_run_and_the_declared_protocol(tmp_path):
    run = "arms-turngait-clock-s0"
    _write_seed(tmp_path, run)
    assert vd.read_seed(tmp_path, run) == {"run": run, "training": True, "v2": "PASS", "qualify": "QUALIFIED",
                                           "push": {"rate": 0.1}}
    _write_seed(tmp_path, run, v2json=_v2_json("arms-turngait-clock-s1"))          # another run's v2 JSON
    assert vd.read_seed(tmp_path, run)["v2"] is None
    _write_seed(tmp_path, run, v2json=_v2_json(run, seconds=5.0))                  # not the declared protocol
    assert vd.read_seed(tmp_path, run)["v2"] is None
    _write_seed(tmp_path, run, v2json={**_v2_json(run), "protocol": "v2x"})
    assert vd.read_seed(tmp_path, run)["v2"] is None
    _write_seed(tmp_path, run, training=_training(run="other"))
    assert vd.read_seed(tmp_path, run)["training"] is None
    _write_seed(tmp_path, run, qualify="INCOMPLETE")
    assert vd.read_seed(tmp_path, run)["qualify"] is None


def test_verdict_cli_writes_once_and_never_overwrites(tmp_path, capsys):
    for s in (0, 1):
        _write_seed(tmp_path, f"arms-turngait-critic-s{s}")
    assert vd.main(["--arm", "R2", "--res", str(tmp_path)]) == 0
    assert not (tmp_path / "verdict" / "R2.json").exists()                          # INCOMPLETE writes nothing
    _write_seed(tmp_path, "arms-turngait-critic-s2", v2="FAIL")
    vd.main(["--arm", "R2", "--res", str(tmp_path)])
    rec = json.loads((tmp_path / "verdict" / "R2.json").read_text())
    assert rec["verdict"] == "PASS" and rec["n_counted"] == 2 and rec["rule"] == FROZEN_RULE
    assert rec["task"] == "Velocity-BHL-Arms-TurnGaitCritic-v0" and "not a fine-tune" in rec["note"]
    _write_seed(tmp_path, "arms-turngait-critic-s0", v2="FAIL")                    # results change afterwards...
    vd.main(["--arm", "R2", "--res", str(tmp_path)])
    assert json.loads((tmp_path / "verdict" / "R2.json").read_text()) == rec        # ...the recorded verdict stays
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
    assert FROZEN_RULE in h and FROZEN_TRAINING in h
    assert "The two arms are new tasks, never presented as fine-tunes of TurnBoth-s0." in h
    assert "Labels: LEARNED gait; MuJoCo gates." in h
    assert "v5's joint rule verbatim, SLURM_JOBS.md line 3055" in h
    assert "DISCLOSED INTERPRETATION" in h and "0.060 m" in h
    sb = SBATCH.read_text()
    assert "#SBATCH --array=0-5%3" in sb and "#SBATCH --gres=gpu:1" in sb


def test_launcher_arm_table():
    body = _body()
    m1 = re.search(r"0\|1\|2\) ARM=R1; ARM_TASK=(\S+); +PREFIX=(\S+); +OBS_POLICY=(\d+); OBS_CRITIC=(\d+); EXPECT=(\w+);;", body)
    m2 = re.search(r"3\|4\|5\) ARM=R2; ARM_TASK=(\S+); +PREFIX=(\S+); +OBS_POLICY=(\d+); OBS_CRITIC=(\d+); EXPECT=(\w+);;", body)
    assert m1 and m2
    assert m1.groups() == ("Velocity-BHL-Arms-TurnGaitClock-v0", "arms-turngait-clock", "77", "80", "clock")
    assert m2.groups() == ("Velocity-BHL-Arms-TurnGaitCritic-v0", "arms-turngait-critic", "75", "80", "plain")
    for arm, m in (("R1", m1), ("R2", m2)):                        # the verdict reads the same names
        assert vd.ARMS[arm]["task"] == m.group(1) and vd.ARMS[arm]["prefix"] == m.group(2)
    assert "SEED=$((IDX % 3))" in body and "FULL_ITERS=6000" in body
    assert 'export MAX_ITER=$FULL_ITERS RUN_NAME="$PREFIX-s${SEED}"' in body
    assert 'export MAX_ITER=3 NUM_ENVS=64 RUN_NAME="$PREFIX-s${SEED}-smoke"' in body
    assert "*-smoke) ;;" in body and re.search(r"SMOKE_RES=\S+-smoke\n", body)
    # from scratch: no resume, no parent checkpoint
    assert "agent.resume" not in body and "load_run" not in body and "load_checkpoint" not in body
    # exported with the arm's own task; qualified into this launcher's results directory
    assert 'TASK="$ARM_TASK" bhl_exec "$REPO/slurm/inner/export_one_humanoid.sh"' in body
    assert 'TURNQ_RUN="$RUN_NAME" TURNQ_OUT_DIR="$RES/qualify" bash "$REPO/slurm/repo20260923/cpu_turn_qualify.sbatch"' in body
    assert "dr_overrides_scale 1.0" in body and "turngait_r12_verdict.py --arm" in body


GOOD_LOG = """\
task=Velocity-BHL-Arms-TurnGaitClock-v0 run=arms-turngait-clock-s0 seed=0 envs=cfg iters=6000
|   0   | base_velocity | UniformVelocityCommand |
|   0   | push_robot |        (5.0, 9.0)       |
| Active Observation Terms in Group: 'policy' (shape: (77,)) |
| Active Observation Terms in Group: 'critic' (shape: (80,)) |
|   10  | feet_air_time              |    0.0 |
|   17  | feet_gait                  |    0.5 |
|   18  | feet_swing_height          |  -20.0 |
[gait_clock_mdp] feet_gait bodies ['leg_left_ankle_roll', 'leg_right_ankle_roll'] offsets [0.0, 0.5] period 0.8 threshold 0.55 step_dt 0.04
[gait_clock_mdp] feet_swing_height bodies ['leg_left_ankle_roll', 'leg_right_ankle_roll']: link-origin z of feet in contact median 0.0600 m (n=8)
train.sh: reached 6000 logged iterations
"""


def _guard(tmp_path, log_text, arm="R1"):
    """Run the launcher's own arm_guard and TASK/GAIT/PUSH regexes (extracted from the sbatch) in bash."""
    sb = SBATCH.read_text()
    fn = re.search(r"^arm_guard\(\) \{.*?^\}$", sb, re.S | re.M).group(0)
    regs = "\n".join(re.search(rf"^{k}=.*$", sb, re.M).group(0) for k in ("TASK_RE", "GAIT_RE", "PUSH_RE"))
    log = tmp_path / "train.log"
    log.write_text(log_text)
    vals = {"R1": ("Velocity-BHL-Arms-TurnGaitClock-v0", "arms-turngait-clock-s0", 77),
            "R2": ("Velocity-BHL-Arms-TurnGaitCritic-v0", "arms-turngait-critic-s0", 75)}[arm]
    script = (f"set -euo pipefail\nARM_TASK={vals[0]}\nRUN_NAME={vals[1]}\nSEED=0\nOBS_POLICY={vals[2]}\nOBS_CRITIC=80\n"
              f"{regs}\n{fn}\n"
              'why="$(arm_guard "$1")"\n'
              'grep -qE "$TASK_RE" "$1" || why="$why task"\n'
              'grep -qE "$GAIT_RE" "$1" || why="$why gait"\n'
              'grep -qE "$PUSH_RE" "$1" || why="$why push"\n'
              'echo "WHY:$why"\n')
    out = subprocess.run(["bash", "-c", script, "guard", str(log)], capture_output=True, text=True, check=True)
    return out.stdout.strip().split("WHY:", 1)[1].split()


def test_launcher_log_guard_accepts_the_r1_log(tmp_path):
    assert _guard(tmp_path, GOOD_LOG) == []


def test_launcher_log_guard_rejects_wrong_arms(tmp_path):
    assert "policy-obs-not-75" in _guard(tmp_path, GOOD_LOG, arm="R2")                  # R1's log is not R2
    assert "task" in _guard(tmp_path, GOOD_LOG, arm="R2")
    bad = GOOD_LOG.replace("|    0.0 |", "|    2.0 |")
    assert "feet_air_time-weight-not-0.0" in _guard(tmp_path, bad)
    assert "gait" in _guard(tmp_path, GOOD_LOG.replace("feet_gait                  |    0.5", "feet_gait | 1.0"))
    assert "push" in _guard(tmp_path, GOOD_LOG.replace("(5.0, 9.0)", "(10.0, 15.0)"))
    assert "unexpected-push_levels-curriculum" in _guard(tmp_path, GOOD_LOG + "|    0    | push_levels    |\n")
    swapped = GOOD_LOG.replace("['leg_left_ankle_roll', 'leg_right_ankle_roll'] offsets",
                               "['leg_right_ankle_roll', 'leg_left_ankle_roll'] offsets")
    assert "feet_gait-did-not-run-on-(left,right)" in _guard(tmp_path, swapped)
    assert "no-logged-iterations" in _guard(tmp_path, GOOD_LOG.replace("train.sh: reached 6000", "train.sh: none"))


RUN0 = "arms-turngait-clock-s0"


def _record(rd, /, **kw):
    """A training record as the launcher writes it from a clean fresh log for run dir rd."""
    rec = {"run": RUN0, "arm": "R1", "task": "Velocity-BHL-Arms-TurnGaitClock-v0", "seed": 0,
           "final_ckpt": "model_5999.pt", "final_ckpt_present": True, "task_id_in_log": True,
           "feet_gait_in_log": True, "push_robot_in_log": True, "run_dir": str(rd), "job": "1",
           "wrong_arm_guard": [], "recipe_check": "GAIT-CLOCK RECIPE: OK R1 (= TurnBoth + the declared changes only)"}
    rec.update(kw)
    return rec


def _record_gate(tmp_path, fresh, record, why_train="", ckpt=True):
    """Run the launcher's own non-smoke training gate (the `if [ "$SMOKE" != 1 ]` block after the
    training evidence) with its training_record_problems function, both extracted from the sbatch,
    in bash. Returns (exit code, output); GATES-WOULD-RUN = the job would go on to export and gates."""
    sb = SBATCH.read_text()
    fn = re.search(r"^training_record_problems\(\) \{.*?^\}$", sb, re.S | re.M).group(0)
    block = re.search(r'^if \[ "\$SMOKE" != 1 \]; then\n    \{ \[ -n "\$run" \].*?^fi$', sb, re.S | re.M).group(0)
    run_dir = tmp_path / f"2026-10-02_03-00-00_{RUN0}"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "model_5999.pt").unlink(missing_ok=True)
    if ckpt:
        (run_dir / "model_5999.pt").write_text("")
    tjson = tmp_path / "training" / f"{RUN0}.json"
    tjson.parent.mkdir(exist_ok=True)
    tjson.unlink(missing_ok=True)
    if record is not None:
        tjson.write_text(record if isinstance(record, str) else json.dumps(record))
    script = (f"set -euo pipefail\nPYH={sys.executable}\nSMOKE=0\nARM=R1\nRUN_NAME={RUN0}\n"
              "ARM_TASK=Velocity-BHL-Arms-TurnGaitClock-v0\nFINAL_CKPT=model_5999.pt\n"
              f'FRESH="$1"\nwhy_train="$2"\nrun="$3"\nTJSON="$4"\n{fn}\n{block}\necho GATES-WOULD-RUN\n')
    out = subprocess.run(["bash", "-c", script, "gate", str(int(fresh)), why_train, str(run_dir), str(tjson)],
                         capture_output=True, text=True)
    return out.returncode, out.stdout + out.stderr


def test_launcher_resubmit_rereads_the_wrong_arm_guard(tmp_path):
    rd = tmp_path / f"2026-10-02_03-00-00_{RUN0}"
    rc, out = _record_gate(tmp_path, 0, _record(rd))                    # resubmit, clean record: gates run
    assert rc == 0 and "GATES-WOULD-RUN" in out, out
    # The review's scenario: the first job reached model_5999.pt but failed the wrong-arm guard, so its
    # record holds the four clause flags true and a non-empty guard. The frozen clause alone would count
    # it; a resubmit (no fresh log) must still run no gate.
    bad = _record(rd, wrong_arm_guard=["policy-obs-not-77"])
    assert vd.training_counts(bad)
    rc, out = _record_gate(tmp_path, 0, bad)
    assert rc == 1 and "GATES-WOULD-RUN" not in out and "seed stays INCOMPLETE" in out and "policy-obs-not-77" in out
    for change in ({"task_id_in_log": False}, {"feet_gait_in_log": False}, {"push_robot_in_log": False},
                   {"final_ckpt_present": False}, {"final_ckpt": "model_2.pt"}, {"run_dir": str(tmp_path / "x")},
                   {"run": "arms-turngait-clock-s1"}, {"task": "Velocity-BHL-Arms-TurnGaitCritic-v0"},
                   {"wrong_arm_guard": "policy-obs-not-77"}, {"task_id_in_log": "true"}):
        rc, out = _record_gate(tmp_path, 0, _record(rd, **change))
        assert rc == 1 and "GATES-WOULD-RUN" not in out and "seed stays INCOMPLETE" in out, (change, out)
    rec = _record(rd)
    del rec["wrong_arm_guard"]
    rc, out = _record_gate(tmp_path, 0, rec)
    assert rc == 1 and "record-has-no-wrong_arm_guard-list" in out
    rc, out = _record_gate(tmp_path, 0, "{not json")
    assert rc == 1 and "unreadable-training-record" in out and "GATES-WOULD-RUN" not in out
    rc, out = _record_gate(tmp_path, 0, None)
    assert rc == 1 and "no training record" in out
    rc, out = _record_gate(tmp_path, 0, _record(rd), why_train=" recipe-check-failed")
    assert rc == 1 and "seed stays INCOMPLETE" in out
    rc, out = _record_gate(tmp_path, 0, _record(rd), ckpt=False)
    assert rc == 1 and "training incomplete" in out


def test_launcher_fresh_job_checks_the_record_too(tmp_path):
    rd = tmp_path / f"2026-10-02_03-00-00_{RUN0}"
    rc, out = _record_gate(tmp_path, 1, _record(rd))
    assert rc == 0 and "GATES-WOULD-RUN" in out, out
    rc, out = _record_gate(tmp_path, 1, _record(rd), why_train=" policy-obs-not-77")
    assert rc == 1 and "no gates run, seed stays INCOMPLETE" in out
    # a retrain after the old run dir was moved aside but its record left in place ("already exists, not
    # overwritten"): the record belongs to another run dir, so no gate runs on the new one
    rc, out = _record_gate(tmp_path, 1, _record(tmp_path / f"2026-10-01_00-00-00_{RUN0}"))
    assert rc == 1 and "record-is-for-another-run-dir" in out and "GATES-WOULD-RUN" not in out


def test_launcher_smoke_touches_no_scored_seed():
    sb = SBATCH.read_text()
    smoke = re.search(r'^if \[ "\$SMOKE" = 1 \]; then\n    why=.*?^fi$', sb, re.S | re.M).group(0)
    # every number after --seed / --seeds / --seed0 (not the "2" of a following "2>&1")
    seeds = [int(s) for m in re.findall(r"--seed(?:s|0)?((?: \d+(?![\d>]))+)", smoke) for s in m.split()]
    assert len(seeds) == 4 and min(seeds) >= gc.EXPLORATION_SEED0 == 100, seeds   # clock smoke, v1, diagnose, run-eval
    assert "-m bhl_robust.eval.run_eval" not in smoke and "-m bhl_robust.eval.gait_clock run-eval" in smoke
    assert '"GAIT-CLOCK RUN-EVAL: PASS"*' in smoke
    assert 'training_record_problems "$TJSON" "$run"' in smoke                    # the smoke runs the record check
    assert 'TJSON="$RES/training/$RUN_NAME.$JOBTAG.json"' in sb                   # on its own fresh record
    gate = re.search(r'^if \[ "\$SMOKE" != 1 \]; then\n    \{ \[ -n "\$run" \].*?^fi$', sb, re.S | re.M).group(0)
    assert gate.count('training_record_problems "$TJSON" "$run"') == 1
    assert "S=$(slurm_clean sbatch --parsable --job-name=turngait-r12-smoke --array=0,3 --time=01:00:00" in sb
    assert "--dependency=afterok:${S%%;*}" in sb and "afterok:<smoke id>" not in sb


# =============================================================================== run_eval on exploration seeds

def test_run_eval_exploration_moves_the_seeds_and_restores_run_eval(tmp_path, monkeypatch):
    """run_eval.main itself with its MuJoCo pieces faked: the eval seeds become 100+, the CSV proves it,
    run_eval.EvalConfig is restored (also after an exception), and a scored seed is refused."""
    pytest.importorskip("mujoco")
    pytest.importorskip("berkeley_humanoid_lite_lowlevel")
    from bhl_robust.eval import harness, run_eval
    (tmp_path / "p.onnx").write_bytes(b"")
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text(f"policy_checkpoint_path: {tmp_path / 'p.onnx'}\npolicy_dt: 0.04\n")
    seen = []

    def episode(env, ctrl, command, seed, cfg, recorder=None):
        seen.append((seed, cfg.seeds, cfg.push_speed, cfg.episode_s))
        return harness.EpisodeResult(command_vx=command[0], command_vy=command[1], command_wz=command[2], seed=seed,
                                     fell=False, survival_s=cfg.episode_s, lin_vel_err=0.0, yaw_rate_err=0.0,
                                     distance_m=0.0, mean_height_m=0.5, mean_tilt_rad=0.0)

    monkeypatch.setattr(run_eval, "prepare_mjcf", lambda *a, **k: tmp_path / "scene.xml")
    monkeypatch.setattr(run_eval, "HeadlessMujocoEnv",
                        lambda *a, **k: SimpleNamespace(cfg=SimpleNamespace(policy_dt=0.04)))
    monkeypatch.setattr(run_eval, "make_controller", lambda cfg: SimpleNamespace(load_policy=lambda: None))
    monkeypatch.setattr(run_eval, "run_episode", episode)
    res = gc.run_eval_smoke(deploy, tmp_path, tmp_path / "cache", tmp_path / "a.csv", "fake")
    assert res["verdict"] == "PASS" and res["rows"] == 6 and res["seeds"] == [100], res
    assert seen == [(100, (100,), 0.5, 2.0)] * 6                       # six commands, all on seed 100
    assert run_eval.EvalConfig is harness.EvalConfig                    # restored
    res = gc.run_eval_smoke(deploy, tmp_path, tmp_path / "cache", tmp_path / "b.csv", "fake", n_seeds=2, seed0=200)
    assert res["verdict"] == "PASS" and res["seeds"] == [200, 201] and res["rows"] == 12
    with pytest.raises(FileExistsError):                                # never overwrites a CSV
        gc.run_eval_smoke(deploy, tmp_path, tmp_path / "cache", tmp_path / "a.csv", "fake")
    with pytest.raises(ValueError):                                     # a scored seed is refused
        gc.run_eval_smoke(deploy, tmp_path, tmp_path / "cache", tmp_path / "c.csv", "fake", seed0=0)
    with pytest.raises(ValueError):
        gc.run_eval_exploration(["--help"], seed0=99)
    # a wrap that did not take effect (episodes on seed 0) is caught by the CSV check
    monkeypatch.setattr(run_eval, "run_episode",
                        lambda env, ctrl, command, seed, cfg, recorder=None: episode(env, ctrl, command, 0, cfg))
    res = gc.run_eval_smoke(deploy, tmp_path, tmp_path / "cache", tmp_path / "d.csv", "fake")
    assert res["verdict"] == "FAIL" and any("exploration seeds" in p for p in res["problems"]), res

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(run_eval, "run_episode", boom)
    with pytest.raises(RuntimeError):
        gc.run_eval_smoke(deploy, tmp_path, tmp_path / "cache", tmp_path / "e.csv", "fake")
    assert run_eval.EvalConfig is harness.EvalConfig                    # restored after an exception too


def test_run_eval_cli_prints_one_verdict_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gc, "run_eval_smoke", lambda *a: {"verdict": "PASS", "rows": 6, "seeds": [100], "rc": 0,
                                                          "falls": 0, "problems": [], "csv": str(a[3])})
    rc = gc.main(["run-eval", "--deploy", "d.yaml", "--upstream", "u", "--cache-dir", "c", "--out",
                  str(tmp_path / "x.csv"), "--label", "l"])
    assert rc == 0 and capsys.readouterr().out.startswith("GAIT-CLOCK RUN-EVAL: PASS (6 rows, seeds [100]")
    monkeypatch.setattr(gc, "run_eval_smoke", lambda *a: {"verdict": "FAIL", "rows": 6, "seeds": [0], "rc": 0,
                                                          "falls": 0, "problems": ["seeds"], "csv": str(a[3])})
    assert gc.main(["run-eval", "--deploy", "d.yaml", "--upstream", "u", "--cache-dir", "c", "--out",
                    str(tmp_path / "x.csv"), "--label", "l", "--seed0", "100"]) == 1


# =============================================================================== MuJoCo: the 77 deploy path

def _turnboth_deploy():
    runs = sorted(LOGROOT.glob("*_arms-turn-turnboth-s0"))
    if not runs or not (runs[-1] / "exported/deploy.yaml").is_file() or not (UPSTREAM / "source").is_dir():
        pytest.skip("needs the upstream assets and the arms-turn-turnboth-s0 export")
    pytest.importorskip("mujoco")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("berkeley_humanoid_lite_lowlevel")
    return runs[-1] / "exported/deploy.yaml"


def _stand_in_r1_deploy(tmp_path):
    """TurnBoth-s0's deploy.yaml re-pointed at a small 77-input MLP exported to ONNX, stamped like an R1 export."""
    from omegaconf import OmegaConf
    src = _turnboth_deploy()
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(77, 32), torch.nn.ELU(), torch.nn.Linear(32, 22))
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(0.05)
    onnx_path = tmp_path / "policy.onnx"
    kw = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    try:
        torch.onnx.export(net, torch.zeros(1, 77), str(onnx_path), input_names=["obs"], output_names=["actions"],
                          opset_version=11, **kw)
    except Exception as exc:                                                        # noqa: BLE001
        pytest.skip(f"torch.onnx.export unavailable here: {exc!r}")
    cfg = OmegaConf.load(src)
    cfg.policy_checkpoint_path = str(onnx_path)
    cfg.num_observations = 77
    deploy = tmp_path / "deploy.yaml"
    OmegaConf.save(cfg, deploy)
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text(ENV_YAML_TAGS)
    gc.stamp(deploy, env_yaml)
    return deploy


def test_smoke_rollout_plain_on_the_shipped_75_export(tmp_path):
    res = gc.smoke(_turnboth_deploy(), UPSTREAM, tmp_path / "cache", "plain", steps=12, seed=100)
    assert res["verdict"] == "PASS", res["problems"]
    assert res["controller"] == "RlController" and res["width"] == res["onnx_width"] == 75


def test_smoke_rollout_clock_through_turn_test_path(tmp_path):
    deploy = _stand_in_r1_deploy(tmp_path)
    res = gc.smoke(deploy, UPSTREAM, tmp_path / "cache", "clock", steps=30, seed=100)
    assert res["verdict"] == "PASS", res["problems"]
    assert res["controller"] == "GaitClockRlController" and res["onnx_width"] == 77
    assert 20 <= res["distinct_clock_values_episode0"] <= 30                     # a full 0.8 s period of values
    assert gc.smoke(deploy, UPSTREAM, tmp_path / "cache", "plain")["verdict"] == "FAIL"


def test_turn_test_and_run_eval_accept_the_77_deploy(tmp_path):
    deploy = _stand_in_r1_deploy(tmp_path)
    tt = _load("turn_test_under_test", REPO / "scripts/bench/turn_test.py")
    r = tt.run_command(deploy, UPSTREAM, tmp_path / "cache", "humanoid", (0.0, 0.0, 0.6), 0.3, 0.2, 100)
    assert set(r) >= {"cmd", "fell_at_s", "yaw_deg"}
    # run_eval's real main() on the 77 deploy, on exploration seed 100 only (never the scored push seeds 0-9)
    out = tmp_path / "eval.csv"
    res = gc.run_eval_smoke(deploy, UPSTREAM, tmp_path / "c2", out, "r1-standin", episode_s=0.4, push_speed=0.5)
    assert res["verdict"] == "PASS", res["problems"]
    assert res["rc"] == 0 and res["rows"] == 6 and res["seeds"] == [100]
    lines = out.read_text().strip().splitlines()
    seed_col = lines[0].split(",").index("seed")
    assert len(lines) == 1 + 6 and {ln.split(",")[seed_col] for ln in lines[1:]} == {"100"}
