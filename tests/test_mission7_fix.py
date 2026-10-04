"""Mission 7 workstream m7-fix-opt: F2 "cross budget" and F3 "yaw cap 0.60" as opt-in options, their plumbing, the
m7-fix release arm and the fix bench launcher.

Frozen design: SLURM_JOBS.md, "User approval recorded 2026-10-03 09:55", item (F).  No bench, replay or route episode
runs here: the stage logic runs on a kinematic stub with upstream's controllers, the replay loop on a kinematic stub
runner, the committed code before this workstream (3a67bc2) is loaded from git for the default-path comparisons,
MissionEnv steps a few decisions on exploration train layout 112 (the fix smoke's route layout), and the launchers run
as extracted functions, in plan / dry modes or on a fake tree.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import mission7_gates as gates  # noqa: E402
import mission7_plate_bench as v1  # noqa: E402
import mission7_plate_bench_v2 as v2  # noqa: E402
from bhl_robust.mission.layout import generate  # noqa: E402

UPSTREAM = _REPO / "external/Berkeley-Humanoid-Lite"
LAUNCHER = _REPO / "slurm/repo20260923/cpu_m7_plate_bench_v2_fix.sbatch"
CLOCKS2_LAUNCHER = _REPO / "slurm/repo20260923/cpu_m7_plate_bench_v2_clocks2.sbatch"
FOLLOWUP = _REPO / "slurm/repo20260923/m7_replay_gate_followup.sbatch"
REPLAY_GATES = _REPO / "slurm/repo20260923/submit_m7_replay_gates.sh"
STAGE_SUBMITTER = _REPO / "scripts/submit_mission7_plate_stage.py"
ROUTE_SUBMITTER = _REPO / "scripts/submit_mission7_route_handoff.py"
PROBE = _REPO / "scripts/mission7_route_handoff_probe.py"
STAGE = _REPO / "scripts/mission7_plate_stage.py"
ENV = _REPO / "src/bhl_robust/mission/env.py"
REPLAY_SOURCE = "results/mission7-replay-smoke-20260921"
REPLAY_BASELINE = "results/mission7-approach-followup-20260922/replay-diagnose-cn-c22"
SIM_STACK = ("mujoco", "torch", "onnxruntime", "omegaconf", "rsl_rl", "berkeley_humanoid_lite_lowlevel")
# The committed code before this workstream (HEAD when it started); the launcher's smoke pins the same.
REF_COMMIT = "3a67bc2"
REF_SHA256 = {"scripts/mission7_plate_stage.py": "100ce2147dc4bbb541b412579a72be46ca2b467628c05f40a11454d136b3944b",
              "scripts/mission7_plate_bench_v2.py": "4e4a9dcb5f63ef6d67b2bbef505c2d9f1f74b71cd93b4931fc033e0c5bed2025",
              "scripts/mission7_route_handoff_probe.py":
                  "cf77fab82f17328562fb5d2827b2f4498108bda707ef7d73508cc5afa156bf5e",
              "src/bhl_robust/mission/env.py": "99cd21167c01d1aa7fd014e2a40b32f13ad06a04bccc1048433d20749f959f9b",
              "scripts/mission7_plate_bench.py": "3b98e47114eed2ceb82f5b437917d7135a5b6e4e0ded30dca106405f78683182"}
CLOCKS2_SHA256 = "c1862f1e4429c1e6ac2e4fae1391f3f40ccc353382c8bbf4b88efee4c6c8d8de"
FROZEN_RULE = (
    "Bench v2: with N crossings run (64 minus the dropped set) and N_h per heading, bench v2 PASSES iff clears >= "
    "N - 2, 0 falls, and clears >= N_h - 1 for every heading. Otherwise FAIL; INCOMPLETE if any declared crossing is "
    "missing. The clear definition is unchanged: mission7_gates' real clear (base along >= +0.35 m past the plate "
    "centre along the door direction), within 10 s of takeover.")
DELIVERED_060 = .6 * np.tanh(np.arctanh(.999999))   # 0.5999994: what a 0.60 command becomes through the interface
N_ACT = 22
DEFAULT_Q = np.linspace(-0.3, 0.3, N_ACT).astype(np.float32)
SMOKE_LAYOUTS = (101, 103, 104, 112, 113, 116)


def _stage_module():
    for module in SIM_STACK:
        pytest.importorskip(module)
    import mission7_plate_stage
    return mission7_plate_stage


@pytest.fixture(autouse=True)
def _no_export_selection():
    """No test leaks a --stage-gait export selection into another."""
    if "mission7_plate_stage" in sys.modules:
        sys.modules["mission7_plate_stage"]._EXPORT_SELECTION.clear()
    yield
    if "mission7_plate_stage" in sys.modules:
        sys.modules["mission7_plate_stage"]._EXPORT_SELECTION.clear()


# ---- the kinematic stub with upstream's controllers (as tests/test_mission7_clocks2.py) ----------------------------

def _cfg(width=75, clock=False):
    from omegaconf import OmegaConf
    d = {"policy_checkpoint_path": "/nonexistent/policy.onnx", "policy_dt": 0.04, "num_joints": N_ACT,
         "num_actions": N_ACT, "num_observations": width, "history_length": 0, "action_scale": 0.25,
         "action_limit_lower": -10000, "action_limit_upper": 10000, "command_velocity": [0.0, 0.0, 0.0],
         "default_joint_positions": DEFAULT_Q.tolist()}
    if clock:
        d["gait_clock"] = {"period_s": 0.8, "phase_offset": 0.0}
    return OmegaConf.create(d)


class _Policy:
    """A deterministic stand-in for the ONNX policy (records what it is fed)."""

    def __init__(self, width, seed=0):
        self.W = np.random.default_rng(seed).normal(0.0, 0.1, (width, N_ACT)).astype(np.float32)
        self.inputs = []

    def forward(self, obs):
        self.inputs.append(np.array(obs, copy=True))
        return np.tanh(obs @ self.W).astype(np.float32)


class _Data:
    def __init__(self):
        self.time = 0.
        self.xpos = np.zeros((2, 3))
        self.qpos = np.zeros(7)


class _Slot:
    body_id, qpos_adr = 1, 0


class StubEnv:
    """runner.d / slot / layout / state / controller as PlateStage reads them; a kinematic plant."""

    def __init__(self, layout, xy, yaw, shipped=None):
        from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
        self.layout = layout
        self.runner = argparse.Namespace(d=_Data())
        self.slot = _Slot()
        self.state = argparse.Namespace(open=[False, False])
        self.upstream = UPSTREAM
        if shipped is None:
            shipped = RlController(_cfg(75))
            shipped.policy = "shipped-policy"
            shipped.prev_actions[:] = .5
            shipped.policy_observations[:] = .1
        self.controller = shipped
        self.set_pose(np.asarray(xy, dtype=float), float(yaw))

    def set_pose(self, xy, yaw):
        d = self.runner.d
        d.xpos[1, :2] = xy
        d.qpos[:2] = xy
        d.qpos[3:7] = [np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]

    def yaw(self):
        q = self.runner.d.qpos[3:7]
        return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))

    def step(self, command, dt=.2, move=True, translate=True):
        yaw = self.yaw()
        vx, vy, wz = np.asarray(command, dtype=float)
        world = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]) @ [vx, vy]
        if move:
            xy = self.runner.d.xpos[1, :2] + (world * dt if translate else 0.)
            self.set_pose(xy, yaw + wz * dt)
        self.runner.d.time = round(self.runner.d.time + dt, 10)
        self.controller.prev_actions[:] = .25 + .01 * (self.runner.d.time % 1)


def _export_gait(clock=True, seed=0):
    width = 77 if clock else 75
    return _Policy(width, seed), _cfg(width, clock), {"stage_gait": "stub", "num_observations": width}


def _bench_like(m, stage_gait="clocks2", layout_index=7, door=1, plate="round", heading=90, export_gait=None,
                **kwargs):
    layout = generate("train", layout_index)
    g = v2.crossing_geometry(layout, door, plate, heading, "standstill")
    env = StubEnv(layout, g["pre_point"], g["entry_yaw_rad"] + 1e-3)
    if stage_gait in ("clocks2", "export"):
        kwargs["export_gait"] = export_gait or _export_gait()
    elif stage_gait == "turnboth":
        kwargs["turnboth_policy"] = "turnboth-policy"
    return env, m.PlateStage(env, stage_gait=stage_gait, **kwargs), g


def _run(stage, env, steps=600, stall_in=(), freeze_in=()):
    """Drive the stage on the stub; phases in ``stall_in`` turn but do not translate, in ``freeze_in`` do not move."""
    log = []
    for _ in range(steps):
        command, phase = stage.command(np.zeros(3))
        log.append((phase, np.asarray(command, dtype=float).round(12).tolist(), float(env.runner.d.time)))
        if phase == "recorded":
            break
        env.step(command, move=phase not in freeze_in, translate=phase not in stall_in)
    return log


def _git_blob(path):
    try:
        blob = subprocess.run(["git", "-C", str(_REPO), "show", f"{REF_COMMIT}:{path}"], capture_output=True,
                              check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history not available")
    assert hashlib.sha256(blob).hexdigest() == REF_SHA256[path], path
    return blob


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_module(tmp_path, path, name):
    _stage_module()
    target = tmp_path / f"{name}.py"
    target.write_bytes(_git_blob(path))
    return _load(target, name)


# ---- F3 on the stub: the in-place turn at 0.60, the crossing hold at 0.40, the turn bound ---------------------------

@pytest.mark.parametrize("layout_index, door, heading, plate", [(7, 1, 90, "round"), (11, 1, 180, "square"),
                                                                 (6, 0, -90, "round")])
def test_yaw_cap_turns_in_place_at_060_and_keeps_the_crossing_hold_at_040(layout_index, door, heading, plate):
    m = _stage_module()
    env, stage, g = _bench_like(m, layout_index=layout_index, door=door, heading=heading, plate=plate, yaw_cap=.6)
    stage._start(door, g["side"], env.runner.d.time)
    log = _run(stage, env)
    assert [h["phase"] for h in stage.history] == ["approach", "settle", "turn", "cross", "turn_back", "recorded"]
    turns = [c for p, c, _ in log if p in ("turn", "turn_back")]
    assert turns and all(c[0] == 0. and c[1] == 0. and abs(c[2]) == .60 for c in turns)
    crossing = [c for p, c, _ in log if p == "cross"]
    assert crossing and all(0. <= c[0] <= m.TURNBOTH_CRUISE and c[1] == 0. and abs(c[2]) <= .40 for c in crossing)
    assert stage._turn_rate_and_bound() == (.60, 2 * np.pi / .60)
    assert m.FIX_TURN_MAX_S == pytest.approx(10.471975511965978) and m.TURNBOTH_TURN_MAX_S == pytest.approx(15.70796)


def test_without_the_cap_the_clocks2_law_still_turns_at_040():
    m = _stage_module()
    env, stage, g = _bench_like(m)
    stage._start(1, g["side"], env.runner.d.time)
    log = _run(stage, env)
    turns = [c for p, c, _ in log if p in ("turn", "turn_back")]
    assert turns and all(abs(c[2]) == .40 for c in turns)
    assert stage._turn_rate_and_bound() == (m.TURNBOTH_TURN_RATE, m.TURNBOTH_TURN_MAX_S)
    assert (stage.cross_budget, stage.yaw_cap, stage.fix_takeover_s) == (None, None, None)


@pytest.mark.parametrize("yaw_cap, bound", [(None, 2 * np.pi / .40), (.6, 2 * np.pi / .60)])
def test_a_turn_that_never_converges_ends_at_a_full_revolution_at_the_commanded_rate(yaw_cap, bound):
    m = _stage_module()
    env, stage, g = _bench_like(m, heading=180, yaw_cap=yaw_cap)
    stage._start(1, g["side"], env.runner.d.time)
    _run(stage, env, steps=300, freeze_in=("turn",))     # the stub never rotates in the turn
    turn, cross = (next(h for h in stage.history if h["phase"] == p) for p in ("turn", "cross"))
    assert cross["turn_timed_out"] is True
    assert bound - 1e-9 <= cross["time_s"] - turn["time_s"] < bound + .2 + 1e-9


# ---- F2 on the stub: the crossing budget ------------------------------------------------------------------------

def _window_cross(m, stall=True, heading=90, yaw_cap=.6, **kwargs):
    env, stage, g = _bench_like(m, heading=heading, cross_budget="window", yaw_cap=yaw_cap, **kwargs)
    takeover = env.runner.d.time
    stage._start(1, g["side"], takeover)
    log = _run(stage, env, stall_in=("cross",) if stall else ())
    cross = next(h for h in stage.history if h["phase"] == "cross")
    back = next(h for h in stage.history if h["phase"] == "turn_back")
    return env, stage, log, takeover, cross, back


def test_cross_budget_keeps_crossing_past_4s_until_02s_of_the_window_remain():
    m = _stage_module()
    env, stage, log, takeover, cross, back = _window_cross(m, stall=True)
    assert stage.fix_takeover_s == takeover
    assert back["cross_ended_by"] == "budget" and back["cleared"] is False and back["along_m"] < .35
    assert back["cross_s"] == pytest.approx(back["time_s"] - cross["time_s"]) and back["cross_s"] > 4.0
    assert 0. < back["window_left_s"] <= .2 + 1e-9                                     # the first poll at <= 0.2 s
    assert 9.8 - 1e-9 <= back["time_s"] - takeover < 10.0
    crossing = [t for p, _, t in log if p == "cross"]
    assert max(crossing) - takeover < 9.8 - 1e-9                                       # no crossing command after 9.8 s


def test_without_the_budget_the_same_stall_ends_at_the_fixed_4s():
    m = _stage_module()
    env, stage, g = _bench_like(m, yaw_cap=.6)
    stage._start(1, g["side"], env.runner.d.time)
    _run(stage, env, stall_in=("cross",))
    cross = next(h for h in stage.history if h["phase"] == "cross")
    back = next(h for h in stage.history if h["phase"] == "turn_back")
    assert 4.0 <= back["time_s"] - cross["time_s"] < 4.2 + 1e-9 and "cross_ended_by" not in back


def test_cross_budget_ends_at_the_clear_point():
    m = _stage_module()
    env, stage, log, takeover, cross, back = _window_cross(m, stall=False, heading=0, layout_index=8)
    assert back["cross_ended_by"] == "clear" and back["cleared"] is True and back["along_m"] >= gates.CLEAR_ALONG_M
    assert back["window_left_s"] > .2 and back["cross_s"] < 4.0
    samples = []
    env2, stage2, g2 = _bench_like(m, heading=0, layout_index=8, cross_budget="window", yaw_cap=.6)
    stage2._start(1, g2["side"], env2.runner.d.time)
    for _ in range(400):
        command, phase = stage2.command(np.zeros(3))
        if phase == "recorded":
            break
        env2.step(command)
        samples.append({"time_s": float(env2.runner.d.time), "xy": env2.runner.d.xpos[1, :2].tolist(), "tilt": .05})
    row = {"layout_index": 8, "layout_seed": env2.layout.seed, "stage_history": stage2.history, "samples": samples}
    [crossing] = gates.crossing_outcomes(row, env2.layout)
    assert crossing["cleared"] and crossing["real_clear"]


def test_cross_budget_boundaries():
    m = _stage_module()
    env, stage, g = _bench_like(m, cross_budget="window")
    for takeover in (0., 3.0, 12.34, 1234.5678):
        stage.fix_takeover_s = takeover
        assert stage._window_left(takeover + 9.8) == pytest.approx(.2)
        assert stage._crossing_continues(takeover + 9.8 - 1e-6)
        assert stage._crossing_continues(takeover + 9.8 - 3e-9)
        assert not stage._crossing_continues(takeover + 9.8)
        assert not stage._crossing_continues(takeover + 9.8 - 5e-10)
        assert not stage._crossing_continues(takeover + 9.8 + 5e-10)
        assert not stage._crossing_continues(takeover + 10.0)
        assert stage._crossing_continues(takeover)                                 # the cross bound plays no part
        assert stage._crossing_continues(takeover + 9.0)
    assert (m.FIX_CLEAR_WINDOW_S, m.FIX_WINDOW_MARGIN_S, m.FIX_TIME_EPS_S) == (v1.CLEAR_WINDOW_S, .2, v2.TIME_EPS_S)


def test_a_crossing_that_starts_after_the_budget_ends_at_once():
    m = _stage_module()
    env, stage, g = _bench_like(m, heading=180, cross_budget="window", yaw_cap=.6)
    takeover = env.runner.d.time
    stage._start(1, g["side"], takeover)
    _run(stage, env, freeze_in=("turn",))                 # the turn runs to its 10.47 s bound
    cross = next(h for h in stage.history if h["phase"] == "cross")
    back = next(h for h in stage.history if h["phase"] == "turn_back")
    assert cross["time_s"] - takeover > 9.8 and back["time_s"] == cross["time_s"]
    assert back["cross_ended_by"] == "budget" and back["cross_s"] == 0. and back["window_left_s"] < .2


def test_the_budget_counts_from_the_stages_own_takeover_at_capture():
    """Replay and route: _start fires at capture (0.78 m), so the approach runs inside the same budget."""
    m = _stage_module()
    layout = generate("validation", 4)
    door = 1
    plate = np.asarray(layout.plate(door, layout.correct_sides[door]))
    _, direction = layout.door(door)
    xy = plate - np.asarray(direction, dtype=float) * .70            # inside the 0.78 m capture radius
    env = StubEnv(layout, xy, float(np.arctan2(direction[1], direction[0])))
    stage = m.PlateStage(env, stage_gait="clocks2", export_gait=_export_gait(), cross_budget="window", yaw_cap=.6)
    env.runner.d.time = 31.2
    log = []
    for _ in range(400):
        command, phase = stage.command(np.array([.1, 0., 0.]))
        log.append(phase)
        if phase == "recorded" and len(log) > 1:
            break
        env.step(command, translate=phase != "cross")
    assert stage.history[0]["phase"] == "approach" and stage.fix_takeover_s == 31.2 == stage.history[0]["time_s"]
    settle = next(h for h in stage.history if h["phase"] == "settle")
    assert settle["time_s"] > 31.2                                   # the approach took time inside the budget
    back = next(h for h in stage.history if h["phase"] == "turn_back")
    assert back["cross_ended_by"] == "budget" and 31.2 + 9.8 - 1e-9 <= back["time_s"] < 31.2 + 10.0


def test_options_compose_with_the_export_gaits_only_and_validate():
    m = _stage_module()
    env = StubEnv(generate("train", 0), [0., 0.], 0.)
    for gait, extra in (("shipped", {}), ("turnboth", {"turnboth_policy": "p"}), ("m3", {})):
        for kwargs in ({"cross_budget": "window"}, {"yaw_cap": .6}, {"cross_budget": "window", "yaw_cap": .6}):
            with pytest.raises(ValueError, match="compose with --stage-gait"):
                m.PlateStage(env, stage_gait=gait, **extra, **kwargs)
    for kwargs in ({"cross_budget": "always"}, {"yaw_cap": .5}, {"yaw_cap": .4}, {"yaw_cap": .61},
                   {"cross_budget": "window", "cross_clear_m": .35}):
        with pytest.raises(ValueError):
            m.PlateStage(env, stage_gait="clocks2", export_gait=_export_gait(), **kwargs)
    for gait in ("clocks2", "export"):
        for kwargs in ({"cross_budget": "window"}, {"yaw_cap": .6}, {"yaw_cap": "0.60"}):
            stage = m.PlateStage(env, stage_gait=gait, export_gait=_export_gait(), **kwargs)
            assert stage.cross_budget == kwargs.get("cross_budget")
            assert stage.yaw_cap == (.6 if "yaw_cap" in kwargs else None)
    stage = m.PlateStage(env, stage_gait="clocks2", export_gait=_export_gait(), yaw_cap=.6, cross_clear_m=.30)
    assert stage.turnboth_clear_m == .30                              # F3 alone keeps --cross-clear
    default = m.PlateStage(env)
    assert (default.cross_budget, default.yaw_cap, default.fix_takeover_s) == (None, None, None)
    assert not hasattr(default, "gait_events") and not hasattr(default, "m3_events")


def test_declared_constants_and_texts():
    m = _stage_module()
    assert m.FIX_CROSS_BUDGETS == ("window",) == v2.FIX_CROSS_BUDGETS_V2
    assert m.FIX_YAW_CAPS == (.60,) == v2.FIX_YAW_CAPS_V2 and m.FIX_YAW_CAP_RPS == m.FIX_TURN_RATE == .60
    assert m.FIX_TURN_MAX_S == 2 * np.pi / .60
    # the pinned M2 / clock-s2 constants are untouched (tests/test_mission7_plate_bench.py, test_mission7_clocks2.py)
    assert m.TURNBOTH_TURN_RATE == .40 == m.TURNBOTH_WZ_WALK and m.TURNBOTH_TURN_MAX_S == 2 * np.pi / .40
    assert m.fix_constants() is None
    both = m.fix_constants("window", .6)
    assert (both["cross_budget"], both["yaw_cap_rps"], both["turn_rate_rps"], both["wz_walk_rps"]) == ("window", .6, .6, .4)
    assert both["command_scales"] == [.4, .35, .6] and both["clear_along_m"] == gates.CLEAR_ALONG_M
    assert both["clear_window_s"] == 10.0 and both["window_margin_s"] == .2
    only_f2 = m.fix_constants("window", None)
    assert only_f2["turn_rate_rps"] == .4 and only_f2["command_scales"] == [.4, .35, .4]
    only_f3 = m.fix_constants(None, .6)
    assert "cross_max_s" in only_f3["crossing_end"]
    text = m.fix_description("window", .6)
    for clause in ("0.2 s of the 10 s clear window", "instead of at cross_max_s", "0.60 rad/s", "+-0.40",
                   "cap semantics"):
        assert clause in text, clause
    assert m.fix_kwargs(argparse.Namespace()) == {} == m.fix_kwargs(argparse.Namespace(cross_budget=None, yaw_cap=None))
    assert m.fix_kwargs(argparse.Namespace(cross_budget="window", yaw_cap=.6)) == {"cross_budget": "window", "yaw_cap": .6}


# ---- default paths against the committed code before this workstream (3a67bc2) ----------------------------------

def _ns_env(layout, xy, yaw):
    env = StubEnv(layout, xy, yaw)
    env.controller = argparse.Namespace(policy="shipped-policy", prev_actions=np.full(22, .5, dtype=np.float32))
    return env


def _drive(module, kwargs, layout_index, door, open_at, steps=260):
    layout = generate("validation", layout_index)   # a kinematic stub drive, no episode (the M2/M3 tests' layouts)
    plate = np.asarray(layout.plate(door, layout.correct_sides[door]))
    env = StubEnv(layout, layout.xy(layout.route[layout.door_indices[door] - 1]), 0.)
    if kwargs.get("stage_gait") not in ("clocks2", "export"):
        env.controller = argparse.Namespace(policy="shipped-policy", prev_actions=np.full(22, .5, dtype=np.float32))
    stage = module.PlateStage(env, **kwargs)
    trace = []
    for n in range(steps):
        if open_at is not None and n == open_at:
            env.state.open[door] = True
        xy = env.runner.d.xpos[1, :2]
        delta = plate - xy
        yaw = env.yaw()
        body = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]) @ (
            delta / max(np.linalg.norm(delta), 1e-9) * .3)
        command, phase = stage.command(np.array([body[0], body[1], -.5 * yaw]))
        trace.append((phase, np.asarray(command, dtype=float).round(12).tolist(),
                      np.asarray(env.controller.prev_actions).copy().tolist(), type(env.controller).__name__))
        env.step(command)
    return trace, stage


@pytest.mark.parametrize("kwargs", [{}, {"cross_clear_m": .35}, {"align_yaw": True}, {"cross_kick": True},
                                    {"press_hold": True, "wait_open_s": 2.0}, {"stage_lateral_m": .35},
                                    {"stage_gait": "shipped"},
                                    {"stage_gait": "turnboth", "turnboth_policy": "turnboth-policy"},
                                    {"stage_gait": "m3"}, {"stage_gait": "m3", "cross_kick": True},
                                    {"stage_gait": "clocks2"}, {"stage_gait": "export"},
                                    {"stage_gait": "clocks2", "cross_clear_m": .30, "cross_kick": True}])
@pytest.mark.parametrize("layout_index, door, open_at", [(0, 0, None), (4, 1, 40), (13, 1, None)])
def test_every_existing_stage_path_matches_the_committed_stage(tmp_path, kwargs, layout_index, door, open_at):
    reference = _git_module(tmp_path, "scripts/mission7_plate_stage.py", "m7_stage_ref_3a67bc2")
    m = _stage_module()
    out = []
    for module in (reference, m):
        run = dict(kwargs)
        if run.get("stage_gait") in ("clocks2", "export"):
            run["export_gait"] = _export_gait()
        out.append(_drive(module, run, layout_index, door, open_at))
    (expected, ref_stage), (actual, new_stage) = out
    assert actual == expected and new_stage.history == ref_stage.history and new_stage.done == ref_stage.done
    for attr in ("gait_events", "m3_events", "kick_events", "align_events"):
        assert getattr(new_stage, attr, None) == getattr(ref_stage, attr, None), attr


@pytest.mark.parametrize("gait", ["turnboth", "m3", "clocks2", "export"])
@pytest.mark.parametrize("layout_index, door, heading", [(7, 1, 90), (8, 1, 0), (11, 1, 180)])
def test_the_bench_like_drive_matches_the_committed_stage(tmp_path, gait, layout_index, door, heading):
    reference = _git_module(tmp_path, "scripts/mission7_plate_stage.py", "m7_stage_ref_bench")
    m = _stage_module()
    out = []
    for module in (reference, m):
        env, stage, g = _bench_like(module, gait, layout_index=layout_index, door=door, heading=heading)
        if gait in ("turnboth", "m3"):
            env.controller = argparse.Namespace(policy="shipped-policy", prev_actions=np.full(22, .5, dtype=np.float32))
        stage._start(door, g["side"], env.runner.d.time)
        events = getattr(stage, "gait_events", None)
        out.append((_run(stage, env), stage.history, getattr(stage, "m3_events", None),
                    [{k: v for k, v in e.items()} for e in events] if events is not None else None))
    assert out[0] == out[1]


def test_existing_descriptions_and_constants_match_the_committed_stage(tmp_path):
    reference = _git_module(tmp_path, "scripts/mission7_plate_stage.py", "m7_stage_ref_texts")
    m = _stage_module()
    for gait in ("turnboth", "m3", "clocks2"):
        assert m.stage_gait_description(gait) == reference.stage_gait_description(gait)
    assert m.m3_constants() == reference.m3_constants() and m.turnboth_law_constants() == reference.turnboth_law_constants()
    for name in ("STAGE_GAITS", "EXPORT_STAGE_GAITS", "TURNBOTH_LAW_GAITS", "TURNBOTH_TURN_RATE", "TURNBOTH_TURN_EXIT",
                 "TURNBOTH_TURN_MAX_S", "TURNBOTH_CRUISE", "TURNBOTH_K_YAW", "TURNBOTH_WZ_WALK", "TURNBOTH_CROSS_CLEAR_M",
                 "CLOCKS2_EXPORT", "CLOCKS2_POLICY_SHA256", "EXPORT_FREE_DEPLOY_KEYS"):
        assert getattr(m, name) == getattr(reference, name), name
    for name in ("_m3_command", "_swap_controller", "_swap_gait", "_kick", "_with_yaw", "_m3_stall", "_m3_stalled"):
        assert inspect.getsource(getattr(m.PlateStage, name)) == inspect.getsource(getattr(reference.PlateStage, name))


# ---- the transforms: default identity, the 0.60 yaw scale, cap semantics -----------------------------------------

def _commands(n=400, seed=3):
    rng = np.random.default_rng(seed)
    out = list(rng.uniform(-.8, .8, (n, 3)))
    out += [np.array(c, dtype=float) for c in ([0., 0., .4], [0., 0., -.4], [.4, .35, .6], [-.4, -.35, -.6], [0., 0., .6],
                                               [.3, 0., .3999], [0., 0., 0.], [1e-12, -1e-12, 1e-9], [5., -5., 5.])]
    return out


def test_capped_command_without_the_cap_is_bitwise_v1s_route_transform():
    m = _stage_module()
    for c in _commands():
        np.testing.assert_array_equal(m.capped_command(c), v1._route_transform(c))
        np.testing.assert_array_equal(m.capped_command(list(c), None), v1._route_transform(list(c)))


def test_capped_command_with_the_cap_changes_the_yaw_scale_only():
    m = _stage_module()
    for c in _commands():
        capped, base = m.capped_command(c, .6), v1._route_transform(c)
        np.testing.assert_array_equal(capped[:2], base[:2])
        assert capped[2] == pytest.approx(np.tanh(np.arctanh(np.clip(c[2] / .6, -.999999, .999999))) * .6, abs=1e-15)
        if abs(c[2]) < .6 * .999999:
            assert capped[2] == pytest.approx(c[2], abs=1e-15)          # cap semantics: below the cap, unchanged
    assert m.capped_command([0., 0., .6], .6)[2] == pytest.approx(DELIVERED_060, abs=1e-15)
    assert DELIVERED_060 == pytest.approx(.5999994, abs=1e-9)
    assert m.capped_command([0., 0., -.6], .6)[2] == pytest.approx(-DELIVERED_060, abs=1e-15)
    assert v1._route_transform([0., 0., .6])[2] == pytest.approx(.3999996, abs=1e-9)   # what 0.60 became before
    np.testing.assert_array_equal(m.fix_command_scales(), [.4, .35, .4])
    np.testing.assert_array_equal(m.fix_command_scales(.6), [.4, .35, .6])
    np.testing.assert_array_equal(m.fix_command_scales(), v1.COMMAND_SCALES)


def _ref_probe_functions():
    text = _git_blob("scripts/mission7_route_handoff_probe.py").decode()
    namespace = {"np": np}
    for name in ("COMMAND_SCALES",):
        exec(re.search(rf"^{name} = .*$", text, re.M).group(0), namespace)
    for name in ("action_to_physical", "physical_to_action"):
        exec(re.search(rf"^def {name}\(.*?\n(?=\n\n)", text, re.S | re.M).group(0), namespace)
    return namespace


def test_probe_conversions_default_to_the_committed_ones_bitwise():
    m = _stage_module()
    import mission7_route_handoff_probe as probe
    ref = _ref_probe_functions()
    rng = np.random.default_rng(5)
    for c in _commands():
        base = rng.normal(0, 1, 5)
        np.testing.assert_array_equal(probe.physical_to_action(c, base), ref["physical_to_action"](c, base))
        np.testing.assert_array_equal(probe.action_to_physical(base), ref["action_to_physical"](base))
        np.testing.assert_array_equal(probe.physical_to_action(c, base, probe.COMMAND_SCALES),
                                      ref["physical_to_action"](c, base))
    np.testing.assert_array_equal(probe.COMMAND_SCALES, ref["COMMAND_SCALES"])
    assert m.FIX_COMMAND_CLIP == .999999


def test_probe_encoding_under_the_cap_delivers_060_turns_and_the_routes_own_commands_unchanged():
    m = _stage_module()
    import mission7_route_handoff_probe as probe
    from bhl_robust.mission.approach_debug import command_action
    scales = m.fix_command_scales(.6)

    def decode(action):            # MissionEnv.step with yaw_scale=0.6
        bounded = np.tanh(np.asarray(action, dtype=float))
        out = bounded[:3] * [.4, .35, .4]
        out[2] = bounded[2] * .6
        return out
    turn = probe.physical_to_action([0., 0., .6], np.zeros(5), scales)
    assert decode(turn)[2] == pytest.approx(DELIVERED_060, abs=1e-12)
    hold = probe.physical_to_action([.3, 0., .4], np.zeros(5), scales)
    assert decode(hold)[2] == pytest.approx(.4, abs=1e-12)
    rng = np.random.default_rng(9)
    for _ in range(300):           # the route's own actions: command_action at the 0.40 scale + activate / acquire
        physical = [rng.uniform(-.3, .3), rng.uniform(-.3, .35), rng.uniform(-.35, .35)]
        base = command_action(physical)
        base[3], base[4] = rng.choice([0., 2.]), rng.choice([0., 2., -2.])
        under = probe.route_action_under_cap(base, scales)
        np.testing.assert_array_equal(under[[0, 1, 3, 4]], base[[0, 1, 3, 4]])
        np.testing.assert_allclose(decode(under), np.tanh(base[:3]) * [.4, .35, .4], atol=1e-12, rtol=0)
        np.testing.assert_allclose(decode(under), physical, atol=1e-6)
    np.testing.assert_array_equal(probe.route_action_under_cap(np.zeros(5), scales), np.zeros(5))


def _decision_env(module, tmp_path, name, **kwargs):
    env = module.MissionEnv(_REPO, tmp_path / name, stage="doors", split="train", seed=4112, **kwargs)
    env.reset(112)
    seen = []
    update = env.controller.update

    def recording(robot_observations):
        seen.append(np.asarray(robot_observations, dtype=np.float32)[-3:].copy())
        return update(robot_observations)
    env.controller.update = recording
    return env, seen


def test_missionenv_default_path_matches_the_committed_env_and_the_yaw_scale_decodes_060(tmp_path):
    _stage_module()
    from bhl_robust.mission import env as new_env
    reference = _git_module(tmp_path, "src/bhl_robust/mission/env.py", "m7_env_ref_3a67bc2")
    rng = np.random.default_rng(11)
    actions = [rng.normal(0, .8, 5) for _ in range(3)] + [np.array([0., 0., 20., 0., 0.])]
    ref_env, ref_seen = _decision_env(reference, tmp_path, "ref")
    new, new_seen = _decision_env(new_env, tmp_path, "new")
    capped, cap_seen = _decision_env(new_env, tmp_path, "cap", yaw_scale=.6)
    assert new.yaw_scale is None and capped.yaw_scale == .6
    for action in actions:
        a = ref_env.step(action.copy())
        b = new.step(action.copy())
        capped.step(action.copy())
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:3] == b[1:3]
        np.testing.assert_array_equal(ref_env.runner.d.qpos, new.runner.d.qpos)
    np.testing.assert_array_equal(np.array(ref_seen), np.array(new_seen))
    seen, cap = np.array(new_seen), np.array(cap_seen)
    np.testing.assert_array_equal(seen[:5, :2], cap[:5, :2])          # first decision: same state, vx and vy unchanged
    for k, action in enumerate(actions):
        bounded = np.tanh(action)
        assert cap[5 * k, 2] == np.float32(bounded[2] * .6) and seen[5 * k, 2] == np.float32(bounded[2] * .4)
    assert cap[-1, 2] == np.float32(.6) and seen[-1, 2] == np.float32(.4)   # saturated: the scale is the cap
    with pytest.raises(ValueError):
        new_env.MissionEnv(_REPO, tmp_path / "bad", stage="doors", yaw_scale=0.)
    assert "bounded[:3]*[.4, .35, .4]" in ENV.read_text()


# ---- the exact replay's loop (_episode) on a stub runner -----------------------------------------------------------

class _ReplayRunner:
    """The runner surface _episode uses: d, contact_trace, observe (a full robot observation, command last), step."""

    def __init__(self, env):
        self.env, self.d, self.contact_trace, self.commands = env, env.runner.d, None, []
        self.d.warning = argparse.Namespace(number=np.zeros(8, dtype=int))

    def observe(self, i, command):
        self.commands.append(np.asarray(command, dtype=float).copy())
        return np.concatenate([[1., 0., 0., 0.], np.zeros(3), DEFAULT_Q, np.zeros(N_ACT), [3.],
                               np.asarray(command, dtype=float)]).astype(np.float32)

    def step(self, targets):
        self.env.step(self.commands[-1], dt=.04)


def _replay_harness(module, stage_gait, layout_index=4, door=1, n=420, **fix):
    from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
    layout = generate("validation", layout_index)
    side = layout.correct_sides[door]
    plate = np.asarray(layout.plate(door, side))
    _, direction = layout.door(door)
    direction = np.asarray(direction, dtype=float)
    start = plate - direction * .70
    shipped = RlController(_cfg(75))
    shipped.policy = _Policy(75, seed=4)
    env = StubEnv(layout, start, float(np.arctan2(direction[1], direction[0])) + np.pi / 2, shipped=shipped)
    env._gates = lambda: None
    runner = _ReplayRunner(env)
    env.runner = runner
    trace = [{"command": [.2, 0., .1] if k % 7 else [0., .1, -.2], "time_s": round(.04 * (k + 1), 10),
              "xy": start.tolist(), "tilt": 0., "yaw": 0.} for k in range(n)]
    source = {"diagnostic_trace": trace, "activation_s": [None, None], "layout": {"seed": layout.seed},
              "elapsed_s": trace[-1]["time_s"]}

    def fake_sample(env_, command, phase):
        return {"time_s": float(env_.runner.d.time), "xy": env_.runner.d.xpos[1, :2].tolist(), "tilt": 0.,
                "yaw": env_.yaw(), "phase": phase, "command": np.asarray(command, dtype=float).tolist(),
                "contacts": []}
    module.physical_sample = fake_sample
    export = _export_gait() if stage_gait in ("clocks2", "export") else None
    original = module.load_export_gait
    module.load_export_gait = lambda env_, gait: export
    try:
        row = module._episode(env, layout_index, source, {}, stage_gait=stage_gait, **fix)
    finally:
        module.load_export_gait = original
    return row, runner.commands


def test_the_replay_loop_without_options_matches_the_committed_one(tmp_path):
    reference = _git_module(tmp_path, "scripts/mission7_plate_stage.py", "m7_stage_ref_replay")
    m = _stage_module()
    saved = m.physical_sample
    try:
        for gait in ("shipped", "clocks2"):
            expected, sent_ref = _replay_harness(reference, gait)
            actual, sent_new = _replay_harness(m, gait)
            assert json.dumps(actual, sort_keys=True, default=str) == json.dumps(expected, sort_keys=True, default=str)
            np.testing.assert_array_equal(np.array(sent_new), np.array(sent_ref))
            assert "stage_fix" not in actual
            if gait == "clocks2":
                turns = [s["effective_command"] for s in actual["samples"] if s["phase"] in ("turn", "turn_back")]
                assert turns and all(abs(c[2]) == .40 for c in turns)   # delivered raw, as before
    finally:
        m.physical_sample = saved


def test_the_replay_loop_with_the_cap_delivers_the_stages_commands_through_the_060_interface():
    m = _stage_module()
    saved, original, calls = m.physical_sample, m.capped_command, []

    def recording(command, yaw_cap=None):
        out = original(command, yaw_cap)
        calls.append((np.asarray(command, dtype=float).copy(), yaw_cap, out.copy()))
        return out
    m.capped_command = recording
    try:
        row, sent = _replay_harness(m, "clocks2", cross_budget="window", yaw_cap=.6)
    finally:
        m.physical_sample, m.capped_command = saved, original
    samples = row["samples"]
    assert row["stage_fix"] == m.fix_constants("window", .6) and list(row)[-1] == "samples"
    staged = [s for s in samples if s["phase"] != "recorded"]
    assert len(staged) == len(calls) and all(cap == .6 for _, cap, _ in calls)
    for s, (raw, _, out) in zip(staged, calls):                                 # exactly capped_command of the stage's own
        np.testing.assert_array_equal(s["effective_command"], out)
        if s["phase"] in ("turn", "turn_back"):
            assert raw[0] == raw[1] == 0. and abs(raw[2]) == .60
            assert abs(s["effective_command"][2]) == pytest.approx(DELIVERED_060, abs=1e-15)
    for s, command in zip(samples, sent):
        np.testing.assert_array_equal(s["effective_command"], command)          # what the controller was given
        if s["phase"] == "recorded":
            assert s["effective_command"] == s["recorded_command"]               # the replayed route, untouched
    assert any(s["phase"] == "recorded" for s in samples) and any(s["phase"] == "turn" for s in samples)
    phases = [h["phase"] for h in row["stage_history"]]
    assert phases[:4] == ["approach", "settle", "turn", "cross"] and "turn_back" in phases


# ---- bench v2: run_crossing_fix, the fix smoke parser, the CLI, provenance, verdict ------------------------------

DECLARED_FIX_DIFF = [
    ("-", 'def run_crossing(repo, cache, spec, stage_gait):'),
    ("-", '    """One v2 crossing; returns the full row (trace included)."""'),
    ("+", 'def run_crossing_fix(repo, cache, spec, stage_gait, cross_budget=None, yaw_cap=None):'),
    ("+", '    """--- m7-fix --- one v2 crossing with --cross-budget / --yaw-cap (F2 / F3): run_crossing with exactly these'),
    ("+", '    changes (tests/test_mission7_fix.py diffs the two sources): the stage gets the options, every command passes'),
    ("+", '    through capped_command (bitwise v1._route_transform without --yaw-cap; the 0.60 yaw scale with it), each sample'),
    ("+", "    records the command the active controller received (policy_command = its observation's first three entries),"),
    ("+", '    and the result records the options and its fix_summary (computed here: v1.read_results drops the samples)."""'),
    ("-", '    from mission7_plate_stage import PlateStage, _yaw'),
    ("+", '    from mission7_plate_stage import PlateStage, _yaw, capped_command, fix_constants'),
    ("-", '    stage = PlateStage(env, stage_gait=stage_gait)'),
    ("+", '    stage = PlateStage(env, stage_gait=stage_gait, cross_budget=cross_budget, yaw_cap=yaw_cap)'),
    ("-", '        applied = v1._route_transform(command)'),
    ("+", '        applied = capped_command(command, yaw_cap)'),
    ("+", '            received = env.controller.policy_observations[0, :3].astype(float).tolist()'),
    ("+", '            sample["policy_command"] = received'),
    ("-", '        "stage_kwargs": {"stage_gait": stage_gait},'),
    ("+", '        "stage_kwargs": {"stage_gait": stage_gait, "cross_budget": cross_budget, "yaw_cap": yaw_cap},'),
    ("+", '        "stage_fix": fix_constants(cross_budget, yaw_cap),'),
    ("+", '        "fix_summary": fix_crossing_summary({"stage_history": stage.history, "samples": samples}),'),
]


def test_run_crossing_fix_is_run_crossing_with_exactly_the_declared_changes():
    old = inspect.getsource(v2.run_crossing).splitlines()
    new = inspect.getsource(v2.run_crossing_fix).splitlines()
    diff = [(line[0], line[1:]) for line in difflib.unified_diff(old, new, lineterm="", n=0)
            if line[:1] in "+-" and not line.startswith(("+++", "---"))]
    assert sorted(diff) == sorted(DECLARED_FIX_DIFF)


def test_the_pinned_bench_v2_functions_match_the_committed_ones(tmp_path):
    reference = _git_module(tmp_path, "scripts/mission7_plate_bench_v2.py", "m7_bench_v2_ref_3a67bc2")
    for name in ("run_crossing", "bench_v2_verdict", "crossing_geometry", "declared_crossings", "parse_smoke_specs",
                 "check_rescore_m2", "dropped_crossings", "chain_for"):
        assert inspect.getsource(getattr(v2, name)) == inspect.getsource(getattr(reference, name)), name
    assert v2.RULE_V2 == reference.RULE_V2 == FROZEN_RULE and v2.DROPPED == reference.DROPPED
    for gait in ("turnboth", "m3", "clocks2", "export"):
        assert v2.labels(gait) == reference.labels(gait) and v2.chain_for(gait) == reference.chain_for(gait)
        assert v2.source_files(_REPO, gait) == reference.source_files(_REPO, gait)
    result = {"layout": 8, "door": 1, "stage_gait": "clocks2", "events": [], "stage_history": [], "geometry": {},
              "plate_activation": {"target_pressed_during_stage": False}, "wall_contacts": {"stage_samples": 0},
              "spawn": {"contacts": []}, "m3_events": None,
              "gait_events": [{"event": "swap_to_clocks2", "time_s": 3.0, "controller": "GaitClockRlController"}]}
    assert v2.compact_result(result) == reference.compact_result(result)


def test_fix_smoke_parser_admits_exploration_layouts_32_to_169_except_ds_smoke_layouts():
    specs = v2.parse_fix_smoke_specs("103:0,113:1,116:0")
    assert [(s["layout"], s["door"], s["heading_deg"], s["plate"], s["entry"]) for s in specs] == [
        (103, 0, 180, "square", "standstill"), (113, 1, 90, "round", "walking"), (116, 0, 0, "square", "standstill")]
    assert v2.parse_fix_smoke_specs("34:0") and v2.parse_fix_smoke_specs("37:0") and v2.parse_fix_smoke_specs("169:1")
    for bad in ("31:0", "0:0", "32:0", "33:1", "38:0", "170:0", "171:1", "200:0", "249:1", "250:0", "253:1", "255:1",
                "256:0", "290:0", "103:2", "103", "103:0:45:round:standstill", "103:0:90:oval:standstill", "",
                "103:0,103:0", "100:0:180:round:walking"):
        with pytest.raises(ValueError):
            v2.parse_fix_smoke_specs(bad)
    with pytest.raises(ValueError):
        v2.parse_smoke_specs("103:0")                     # the pinned bench-v2 smoke parser keeps 250-255
    assert v2.FIX_SMOKE_LAYOUTS == (32, 170) and v2.FIX_SMOKE_EXCLUDED == (32, 33, 38)
    admitted = [layout_index for layout_index in range(256) if v2.FIX_SMOKE_LAYOUTS[0] <= layout_index
                < v2.FIX_SMOKE_LAYOUTS[1] and layout_index not in v2.FIX_SMOKE_EXCLUDED]
    assert admitted[0] == 34 and admitted[-1] == 169 and 38 not in admitted and not set(admitted) & set(range(170, 256))


def test_the_smoke_layouts_were_chosen_as_the_header_says():
    """The geometry picks are the lowest layouts >= 100 with each property (pure geometry; no episode); the two
    development-smoke picks have the properties the header names."""
    def first(predicate):
        for layout_index in range(100, 200):
            for door in (0, 1):
                heading, plate, entry = v1.cell_of(layout_index, door)
                g = v2.crossing_geometry(generate("train", layout_index), door, plate, heading, entry)
                if predicate(heading, entry, g):
                    return layout_index, door
    assert first(lambda h, e, g: h == 180 and e == "standstill" and g["pre_point_wall_clearance_m"] >= .316) == (103, 0)
    assert first(lambda h, e, g: h != 0 and e == "walking" and g["walking_geometrically_clear"]) == (113, 1)
    assert first(lambda h, e, g: h == 90 and e == "standstill") == (101, 0)
    assert v1.cell_of(116, 0) == (0, "square", "standstill")
    assert tuple(np.round(generate("train", 112).door(0)[1], 6)) == (0., 1.)          # the stage must turn in place
    candidates = [layout_index for layout_index in range(105, 118)
                  if tuple(np.round(generate("train", layout_index).door(0)[1], 6)) != (1., 0.)]
    assert candidates[:7] == [106, 108, 109, 110, 112, 113, 114] and 117 in candidates   # 113 is a bench pick


def test_labels_chain_disclosure_and_source_files_of_the_fix():
    fix = v2.fix_options("window", .6)
    assert fix == {"cross_budget": "window", "yaw_cap": .6} and v2.fix_options() == {}
    labels = v2.labels("clocks2", fix)
    assert labels["stage"] == "SCRIPTED (PlateStage, stage_gait=clocks2, cross_budget=window, yaw_cap=0.6)"
    assert labels["gaits"] == v2.labels("clocks2")["gaits"] and "arms-turngait-clock-s2" in labels["gaits"]
    assert "export" in v2.labels("export", fix)["gaits"] and "ORACLE" in labels["layout_and_plate_pose"]
    for clause in ("--cross-budget window --yaw-cap 0.6", "release arm m7-fix", "same stage gait and options",
                   "Doors and Transport each >= 16/16 successes on validation layouts 0-15"):
        assert clause in v2.CHAIN_V2_FIX, clause
    for clause in ("Bundling means a PASS is not attributable to one change.",
                   "Faster turns in narrow cells add wall-contact and fall risk (6 of 8 180-deg turns already touched "
                   "walls at 0.40", "Crossing until clear produced a replay fall before (9/10)."):
        assert clause in v2.DISCLOSURE_FIX, clause
    plain = set(v2.source_files(_REPO, "clocks2"))
    assert set(v2.source_files(_REPO, "clocks2", fix=True)) - plain == {v2.V2_FIX_LAUNCHER}
    assert "src/bhl_robust/mission/env.py" in plain and v2.V2_FIX_LAUNCHER == str(LAUNCHER.relative_to(_REPO))


def test_compact_result_reports_the_fix_evidence():
    result = {"layout": 103, "door": 0, "stage_gait": "clocks2", "events": [], "geometry": {}, "spawn": {"contacts": []},
              "plate_activation": {"target_pressed_during_stage": False}, "wall_contacts": {"stage_samples": 0},
              "m3_events": None, "gait_events": [], "stage_fix": {"cross_budget": "window"},
              "stage_history": [{"phase": "cross", "time_s": 8.0},
                                {"phase": "turn_back", "time_s": 12.8, "cross_ended_by": "budget",
                                 "window_left_s": .2, "cross_s": 4.8}],
              "samples": [{"phase": "stage_turn", "command": [0, 0, DELIVERED_060], "policy_command": [0, 0, .6]},
                          {"phase": "stage_cross", "command": [.3, 0, -.4], "policy_command": [.3, 0, -.4]}]}
    line = v2.compact_result(result)
    assert line["fix"] == {"cross_s": 4.8, "cross_ended_by": "budget", "window_left_s": .2,
                           "turn_wz_sent_max": DELIVERED_060, "turn_wz_received_max": .6, "cross_wz_sent_max": .4}
    json.dumps(line)
    result.pop("stage_fix")
    assert "fix" not in v2.compact_result(result)


def test_the_verdict_line_keeps_the_fix_evidence_after_read_results_drops_the_samples(tmp_path):
    """bench v2 writes compact_result(r) for r in v1.read_results(out), and read_results drops the samples: the |wz|
    maxima of the verdict line come from the fix_summary run_crossing_fix stores while it has them (computed from the
    sample-less result they were null: smokes 21532778 / 21532779).  Synthetic record, no episode."""
    layout = generate("train", 103)
    spec = {"layout": 103, "door": 0, **dict(zip(("heading_deg", "plate", "entry"), v1.cell_of(103, 0)))}
    side = v1.plate_side(layout, 0, spec["plate"])
    plate = np.asarray(layout.plate(0, side))
    _, direction = layout.door(0)
    t = np.round(np.arange(1, 251) * .04, 10)
    along = np.where(t < 5., -.30, np.where(t < 7., 0., .40))
    received_060 = float(np.float32(DELIVERED_060))

    def phase_and_command(ti):
        if ti < 3.:
            return "bench_settle", [0., 0., 0.], [0., 0., 0.]
        if ti < 3.4:
            return "stage_settle", [0., 0., 0.], [0., 0., 0.]
        if ti < 5.:
            return "stage_turn", [0., 0., float(DELIVERED_060)], [0., 0., received_060]
        if ti < 7.2:
            wz = -.25 if ti == 5. else .1
            return "stage_cross", [.3, 0., wz], [.3, 0., wz]
        if ti < 8.:
            return "stage_turn_back", [0., 0., -float(DELIVERED_060)], [0., 0., -received_060]
        return "bench_post", [0., 0., 0.], [0., 0., 0.]
    samples = []
    for ti, a in zip(t, along):
        phase, command, received = phase_and_command(float(ti))
        samples.append({"time_s": float(ti), "xy": (plate + np.asarray(direction) * a).tolist(), "tilt": .05,
                        "phase": phase, "command": command, "policy_command": received})
    history = [{"time_s": 3.0, "door": 0, "side": side, "phase": "approach"}, {"time_s": 3.0, "door": 0, "phase": "settle"},
               {"time_s": 3.4, "door": 0, "phase": "turn"}, {"time_s": 5.0, "door": 0, "phase": "cross"},
               {"time_s": 7.2, "door": 0, "phase": "turn_back", "along_m": .40, "cleared": True,
                "cross_ended_by": "clear", "window_left_s": 5.8, "cross_s": 2.2},
               {"time_s": 8.0, "door": 0, "side": side, "phase": "recorded"}]
    row = {"layout_index": 103, "layout_seed": layout.seed, "replay_elapsed_s": samples[-1]["time_s"],
           "first_fall_s": None, "maximum_tilt": .05, "stage_history": history, "samples": samples}
    scored = v2.score_crossing(row, layout, 0, 3.0)
    assert scored["clear"]
    result = {**spec, "complete": True, "end_reason": "complete", "clear": scored["clear"], "real_clear": True,
              "fell": False, "takeover_s": 3.0, "handback_s": 8.0, "events": [{"event": "takeover", "speed_mps": 0.}],
              "crossing": scored["crossing"], "stage_history": history, "stage_gait": "clocks2",
              "spawn": {"contacts": []}, "m3_events": None, "gait_events": [], "geometry": {},
              "plate_activation": {"target_pressed_during_stage": False}, "wall_contacts": {"stage_samples": 0},
              "clear_after_takeover_s": scored["clear_after_takeover_s"],
              "stage_kwargs": {"stage_gait": "clocks2", "cross_budget": "window", "yaw_cap": .6},
              "stage_fix": {"cross_budget": "window", "yaw_cap_rps": .6},
              # the line run_crossing_fix carries (DECLARED_FIX_DIFF pins it there)
              "fix_summary": v2.fix_crossing_summary({"stage_history": history, "samples": samples}),
              "trace_header": {k: row[k] for k in ("layout_index", "layout_seed", "replay_elapsed_s", "first_fall_s",
                                                   "maximum_tilt")},
              "samples": samples}
    expected = {"cross_s": 2.2, "cross_ended_by": "clear", "window_left_s": 5.8,
                "turn_wz_sent_max": float(DELIVERED_060), "turn_wz_received_max": received_060,
                "cross_wz_sent_max": .25}
    assert result["fix_summary"] == expected
    (tmp_path / "crossings").mkdir()
    v1.write_crossing(tmp_path, result)
    [back] = v1.read_results(tmp_path)
    assert "samples" not in back
    stripped = v2.fix_crossing_summary(back)          # the verdict line's |wz| maxima before the stored summary
    assert stripped["turn_wz_sent_max"] is stripped["turn_wz_received_max"] is stripped["cross_wz_sent_max"] is None
    line = v2.compact_result(back)
    assert line["fix"] == expected and line["clear"]
    json.dumps(line)
    assert v2.compact_result(result)["fix"] == expected   # with the samples present: the same line


def _need_exports(m):
    for d in (UPSTREAM / m.CLOCKS2_EXPORT, UPSTREAM / m.SHIPPED_EXPORT):
        if not (d / "policy.onnx").is_file():
            pytest.skip(f"export not on disk: {d}")


def test_bench_v2_provenance_records_the_fix(tmp_path):
    m = _stage_module()
    _need_exports(m)
    fix = v2.fix_options("window", .6)
    prov = v2.provenance(_REPO, "smoke", "clocks2", [(103, 0)], fix=fix)
    assert prov["stage_fix"] == m.fix_constants("window", .6) and prov["chain"] == v2.CHAIN_V2_FIX
    assert prov["disclosure_fix"] == v2.DISCLOSURE_FIX and prov["labels"] == v2.labels("clocks2", fix)
    assert prov["weights_sha256"]["clocks2"] == CLOCKS2_SHA256 and prov["rule"] == FROZEN_RULE
    assert {v2.V2_FIX_LAUNCHER, "src/bhl_robust/mission/env.py", "src/bhl_robust/eval/gait_clock.py"} <= set(
        prov["sources_sha256"])
    plain = v2.provenance(_REPO, "smoke", "clocks2", [(103, 0)])
    assert "stage_fix" not in plain and plain["chain"] == v2.CHAIN_V2_CLOCKS2 and v2.V2_FIX_LAUNCHER not in plain[
        "sources_sha256"]


def test_bench_v2_cli_accepts_the_options_with_the_export_gaits_only(tmp_path, capsys):
    m = _stage_module()
    _need_exports(m)
    base = ["--repo", str(_REPO), "--out", str(tmp_path / "never"), "--mode", "smoke", "--crossings", "103:0,113:1"]
    fix = ["--cross-budget", "window", "--yaw-cap", "0.6"]
    assert v2.main([*base, "--stage-gait", "clocks2", *fix, "--preflight"]) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["stage_fix"] == m.fix_constants("window", .6) and line["crossings"] == 2
    export = str(UPSTREAM / m.CLOCKS2_EXPORT)
    assert v2.main([*base, "--stage-gait", "export", "--stage-gait-export", export, "--stage-gait-export-sha256",
                    CLOCKS2_SHA256, *fix, "--preflight"]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["stage_gait"] == "export"
    assert v2.main([*base, "--stage-gait", "clocks2", "--yaw-cap", "0.60", "--preflight"]) == 0
    capsys.readouterr()
    scored = ["--repo", str(_REPO), "--out", str(tmp_path / "never"), "--mode", "scored"]
    assert v2.main([*scored, "--stage-gait", "clocks2", *fix, "--preflight"]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["crossings"] == 41
    for argv in ([*base, "--stage-gait", "m3", *fix], [*base, "--stage-gait", "turnboth", "--yaw-cap", "0.6"],
                 [*base, "--stage-gait", "clocks2", "--yaw-cap", "0.5"],
                 [*base, "--stage-gait", "clocks2", "--cross-budget", "always"],
                 ["--repo", str(_REPO), "--out", str(tmp_path / "never"), "--mode", "smoke", "--crossings", "253:1",
                  "--stage-gait", "clocks2", *fix],
                 ["--repo", str(_REPO), "--out", str(tmp_path / "never"), "--mode", "smoke", "--crossings", "31:0",
                  "--stage-gait", "clocks2", *fix],
                 ["--repo", str(_REPO), "--out", str(tmp_path / "never"), "--mode", "rescore-m2", *fix]):
        with pytest.raises(SystemExit) as info:
            v2.main([*argv, "--preflight"])
        assert info.value.code == 2, argv
    assert not (tmp_path / "never").exists()


# ---- plumbing: the stage and probe CLIs, both submitters, the follow-up, the release script --------------------------

def _cli(script, *extra, tmp_path):
    env = {**os.environ, "PYTHONPATH": f"{_REPO / 'src'}:{_REPO / 'scripts'}"}
    if script == "stage":
        args = [str(STAGE), "--repo", str(_REPO), "--campaign", REPLAY_SOURCE, "--baseline", REPLAY_BASELINE,
                "--out", str(tmp_path / "never"), "--preflight"]
    else:
        args = [str(PROBE), "--repo", str(_REPO), "--out", str(_REPO / "results/never-created-fix-probe"),
                "--stage", "doors", "--indices", "0", "--preflight"]
    return subprocess.run([sys.executable, *args, *extra], cwd=_REPO, capture_output=True, text=True, timeout=300,
                          env=env)


@pytest.mark.parametrize("script", ["stage", "probe"])
def test_clis_accept_the_options_with_the_export_gaits_and_refuse_the_rest(tmp_path, script):
    m = _stage_module()
    _need_exports(m)
    out = _cli(script, "--stage-gait", "clocks2", "--cross-budget", "window", "--yaw-cap", "0.6", tmp_path=tmp_path)
    assert out.returncode == 0, out.stderr
    line = json.loads(out.stdout.strip().splitlines()[-1])
    assert line["stage_fix"] == m.fix_constants("window", .6) and line["stage_gait"] == "clocks2"
    out = _cli(script, "--stage-gait", "export", "--stage-gait-export", str(UPSTREAM / m.CLOCKS2_EXPORT),
               "--yaw-cap", "0.6", tmp_path=tmp_path)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1])["stage_fix"]["cross_budget"] is None
    default = _cli(script, "--stage-gait", "clocks2", tmp_path=tmp_path)
    assert default.returncode == 0 and "stage_fix" not in json.loads(default.stdout.strip().splitlines()[-1])
    for extra in (["--cross-budget", "window"], ["--stage-gait", "m3", "--yaw-cap", "0.6"],
                  ["--stage-gait", "turnboth", "--cross-budget", "window"],
                  ["--stage-gait", "clocks2", "--yaw-cap", "0.5"], ["--stage-gait", "clocks2", "--cross-budget", "x"],
                  ["--stage-gait", "clocks2", "--cross-budget", "window", "--cross-clear", "0.35"]):
        bad = _cli(script, *extra, tmp_path=tmp_path)
        assert bad.returncode == 2, (extra, bad.stderr)
    assert not (tmp_path / "never").exists() and not (_REPO / "results/never-created-fix-probe").exists()


def test_the_pinned_call_lines_stay_and_the_options_are_inserted_before_them():
    stage, probe = STAGE.read_text(), PROBE.read_text()
    for text in (stage, probe):
        assert 'parser.add_argument("--stage-gait", choices=STAGE_GAITS, default="shipped"' in text
        assert "align_yaw=align_yaw, stage_gait=stage_gait)" in text
        assert "stage_gait_action.choices = STAGE_GAITS + EXPORT_STAGE_GAITS" in text
    assert ("                             **fix_kwargs(args),   # --- m7-fix --- {} unless --cross-budget / --yaw-cap\n"
            "                             stage_gait=getattr(args, 'stage_gait', 'shipped')))") in stage
    assert ("                **fix_kwargs(args),   # --- m7-fix --- {} unless --cross-budget / --yaw-cap\n"
            '                align_yaw=args.align_yaw, stage_gait=getattr(args, "stage_gait", "shipped"))') in probe
    assert "                       cross_budget=cross_budget, yaw_cap=yaw_cap,\n" in stage
    assert "cross_budget=cross_budget, yaw_cap=yaw_cap,   # --- m7-fix --- None: the default path\n" in probe
    assert 'yaw_scale=getattr(args, "yaw_cap", None))' in probe


def test_the_probe_controller_refuses_an_env_whose_yaw_scale_differs():
    _stage_module()
    import mission7_route_handoff_probe as probe
    for env_scale, cap in ((None, .6), (.6, None), (.4, .6)):
        env = argparse.Namespace(yaw_scale=env_scale)
        with pytest.raises(ValueError, match="yaw_scale"):
            probe.RouteHandoffController(env, yaw_cap=cap, stage_gait="clocks2")


def _plan(script, *args, code=0):
    out = subprocess.run([sys.executable, str(script), *args], cwd=_REPO, capture_output=True, text=True, timeout=120)
    assert out.returncode == code, out.stderr
    return json.loads(out.stdout) if code == 0 else out.stderr


def test_replay_submitter_forwards_the_options():
    m = _stage_module()
    _need_exports(m)
    if not ((_REPO / REPLAY_SOURCE / "fullroute/legacy-doors.json").is_file()
            and (_REPO / REPLAY_BASELINE / "result.json").is_file()):
        pytest.skip("untracked replay inputs not on disk")
    common = ["--out-campaign", "results/mission7-campaign-20260923", "--output-name", "replay-gate-test-fix-never"]
    fix = ["--cross-budget", "window", "--yaw-cap", "0.6"]
    assert _plan(STAGE_SUBMITTER, *common, "--stage-gait", "clocks2", *fix)["probe_args"] == [
        "--stage-gait=clocks2", "--cross-budget=window", "--yaw-cap=0.6"]
    export = str(UPSTREAM / m.CLOCKS2_EXPORT)
    assert _plan(STAGE_SUBMITTER, *common, "--stage-gait", "export", "--stage-gait-export", export,
                 "--stage-gait-export-sha256", CLOCKS2_SHA256, *fix)["probe_args"] == [
        "--stage-gait=export", f"--stage-gait-export={Path(export).resolve()}",
        f"--stage-gait-export-sha256={CLOCKS2_SHA256}", "--cross-budget=window", "--yaw-cap=0.6"]
    assert _plan(STAGE_SUBMITTER, *common, "--stage-gait", "clocks2")["probe_args"] == ["--stage-gait=clocks2"]
    for extra in (["--stage-gait", "m3", *fix], fix, ["--stage-gait", "clocks2", "--yaw-cap", "0.5"],
                  ["--stage-gait", "clocks2", "--cross-budget", "window", "--cross-clear", "0.35"]):
        _plan(STAGE_SUBMITTER, *common, *extra, code=2)
    assert not (_REPO / "results/mission7-campaign-20260923/replay-gate-test-fix-never").exists()


def test_route_submitter_forwards_the_options():
    m = _stage_module()
    _need_exports(m)
    common = ["--campaign", "results/mission7-campaign-20260923", "--stage", "doors", "--indices", "0,1",
              "--output-name", "route-gate-test-fix-never", "--handoff", "early", "--stage-activate",
              "--rejoin-advance", "--exit-ramp-center", "--exit-ramp", "1.2", "--stage-wait-open", "0.0"]
    fix = ["--cross-budget", "window", "--yaw-cap", "0.6"]
    plan = _plan(ROUTE_SUBMITTER, *common, "--stage-gait", "clocks2", *fix)
    assert plan["probe_args"][-3:] == ["--stage-gait=clocks2", "--cross-budget=window", "--yaw-cap=0.6"]
    assert set(_plan(ROUTE_SUBMITTER, *common)) == {"planned_output", "stage", "indices", "node", "constraint"}
    assert _plan(ROUTE_SUBMITTER, *common, "--stage-gait", "clocks2")["probe_args"][-1] == "--stage-gait=clocks2"
    for extra in (["--stage-gait", "m3", *fix], fix, ["--stage-gait", "clocks2", "--yaw-cap", "0.61"],
                  ["--stage-gait", "clocks2", "--cross-budget", "window", "--cross-clear", "0.35"]):
        _plan(ROUTE_SUBMITTER, *common, *extra, code=2)
    module = _load(ROUTE_SUBMITTER, "m7_route_submitter_fix_args")
    args = argparse.Namespace(stage="doors", indices="0", handoff="switch", rejoin_fix="none", rejoin_diagnostic=False,
                              chain_trace=False, stage_lateral=None, stage_activate=False, stage_press_hold=False,
                              stage_wait_open=None, pre_point=None, cross_clear=None, stall_min_s=None, settle_s=None,
                              cross_kick=False, rejoin_advance=False, exit_ramp_center=False, align_yaw=False,
                              exit_ramp=0., allow_inactive_intervention=False, stage_gait="clocks2")
    assert module._probe_args(args)[-1] == "--stage-gait=clocks2"          # a Namespace without the new attributes
    args.cross_budget, args.yaw_cap = "window", .6
    assert module._probe_args(args)[-2:] == ["--cross-budget=window", "--yaw-cap=0.6"]
    assert not (_REPO / "results/mission7-campaign-20260923/route-gate-test-fix-never").exists()


def _translation_heredoc():
    text = FOLLOWUP.read_text()
    match = re.search(r'^"\$PY" - "\$gate_dir" > "\$flags_file" <<\'PYEOF\'\n(.*?)\nPYEOF$', text, re.S | re.M)
    assert match, "step-3 heredoc not found"
    return match.group(1)


@pytest.mark.parametrize("probe_args, expected", [
    (["--stage-gait=clocks2", "--cross-budget=window", "--yaw-cap=0.6"],
     ["--stage-gait", "clocks2", "--cross-budget", "window", "--yaw-cap", "0.6", "--stage-wait-open", "0.0"]),
    (["--stage-gait=export", "--stage-gait-export=/x/exported", "--stage-gait-export-sha256=ab", "--cross-budget=window",
      "--yaw-cap=0.6"],
     ["--stage-gait", "export", "--stage-gait-export", "/x/exported", "--stage-gait-export-sha256", "ab",
      "--cross-budget", "window", "--yaw-cap", "0.6", "--stage-wait-open", "0.0"]),
    (["--stage-gait=clocks2"], ["--stage-gait", "clocks2", "--stage-wait-open", "0.0"])])
def test_followup_translates_the_options(tmp_path, probe_args, expected):
    (tmp_path / "submission.json").write_text(json.dumps({"probe_args": probe_args}))
    script = tmp_path / "translate.py"
    script.write_text(_translation_heredoc())
    out = subprocess.run([sys.executable, str(script), str(tmp_path)], capture_output=True, text=True, check=True)
    assert out.stdout.split("\n")[:-1] == expected


def test_followup_dry_run_releases_a_fix_route_gate_from_a_fix_pass(tmp_path):
    _stage_module()
    campaign = tmp_path / "campaign"
    gate = campaign / "replay-gate-fix-synthetic"
    gate.mkdir(parents=True)
    rows = [{"layout_index": i, "fall": False, "stage_history": [], "stage_gait": "clocks2", "gait_events": []}
            for i in (0, 1, 4, 5, 7, 9, 12, 13, 14, 15)]
    (gate / "result.json").write_text(json.dumps({
        "complete": True, "status": "COMPLETED_INTERVENTION", "episodes": 10, "falls": 0, "upright": 10,
        "exact_replay_gate_passed": True, "stage_gait": "clocks2", "episode_comparison": rows}))
    (gate / "submission.json").write_text(json.dumps({"job_id": "0", "source_sha256": {},
                                                      "probe_args": ["--stage-gait=clocks2", "--cross-budget=window",
                                                                     "--yaw-cap=0.6"]}))
    route_campaign = "results/mission7-campaign-20260923"
    locks_before = sorted(p.name for p in (_REPO / route_campaign).glob("replay-gate-route-lock-*"))
    env = {**os.environ, "DRY_RUN": "1", "PYTHON": sys.executable, "M7_REPO": str(_REPO),
           "M7_ROUTE_CAMPAIGN": route_campaign}
    out = subprocess.run(["bash", str(FOLLOWUP), str(gate), "fix-synthetic", "1", str(campaign)], env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "M7 REPLAY GATE fix-synthetic: PASS" in out.stdout and "ROUTE_GATE_READY" in out.stdout
    composition = next(line for line in out.stdout.splitlines() if line.startswith("route gate composition"))
    assert "--stage-gait clocks2 --cross-budget window --yaw-cap 0.6 --stage-wait-open 0.0" in composition
    plans = [json.loads(block) for block in re.findall(r"^\{\n.*?^\}$", out.stdout, re.S | re.M)]
    assert [p["stage"] for p in plans] == ["doors", "transport"]
    assert all(p["probe_args"][-3:] == ["--stage-gait=clocks2", "--cross-budget=window", "--yaw-cap=0.6"]
               and p["indices"] == ",".join(map(str, range(16))) for p in plans)
    assert sorted(p.name for p in (_REPO / route_campaign).glob("replay-gate-route-lock-*")) == locks_before
    assert not (_REPO / route_campaign / "route-gate-fix-synthetic-doors").exists()


def test_replay_gates_script_has_the_fix_arm_gate_and_placement():
    text = REPLAY_GATES.read_text()
    assert re.search(r'^\s*"m7-fix\s+--stage-gait clocks2 --cross-budget window --yaw-cap 0\.6"\s*$', text, re.M)
    assert re.search(r'^\s*"m7-clocks2\s+--stage-gait clocks2"\s*$', text, re.M)      # the clocks2 arm is untouched
    subprocess.run(["bash", "-n", str(REPLAY_GATES)], check=True)
    assert "FIX_BENCH_VERDICT=${M7_FIX_BENCH_VERDICT:-$CAMPAIGN/fix-plate-bench-v2/verdict.json}" in text
    calls = [mt.start() for mt in re.finditer(
        r'^\s*fix_bench_gate "\$FIX_BENCH_VERDICT" "\$FIX_GAIT" "\$FIX_POLICY_SHA256" "\$FIX_POLICY" '
        r'"\$fix_export_arg" "\$\{clocks2_gate_files\[@\]\}" \|\| exit 3$', text, re.M)]
    assert len(calls) == 1
    assert text.index('echo "DRY RUN: nothing submitted.') < calls[0] < text.index("real_sbatch=$(command -v sbatch)")
    assert text.rindex('if [ "$fix_selected" = 1 ]; then', 0, calls[0]) > text.index(
        'clocks2_bench_gate "$CLOCKS2_BENCH_VERDICT"')
    assert text.count("fix_selected=1") == 1 and text.count("c2_selected=1") == 1
    assert 'if [ "$fix_selected" = 1 ] && ! git diff --quiet -- src/bhl_robust/eval/gait_clock.py; then' in text
    replay = re.search(r"^snapshot_files=\((.*?)\)\n", text, re.S | re.M)
    assert replay and "src/bhl_robust/mission/*.py" in replay.group(1) and "gait_clock" not in replay.group(1)


def _arm_rewrite_block():
    text = REPLAY_GATES.read_text()
    match = re.search(r'^if \[ -n "\$FIX_EXPORT" \] \|\| \[ -n "\$FIX_EXPORT_SHA256" \]; then\n.*?^fi\n', text,
                      re.S | re.M)
    assert match
    return match.group(0)


@pytest.mark.parametrize("export, sha, code, arm", [
    ("", "", 0, "m7-fix --stage-gait clocks2 --cross-budget window --yaw-cap 0.6 clocks2"),
    ("EXPORT", "a" * 64, 0, "m7-fix --stage-gait export --stage-gait-export EXPORT --stage-gait-export-sha256 "
                            + "a" * 64 + " --cross-budget window --yaw-cap 0.6 export"),
    ("EXPORT", "", 2, None), ("", "a" * 64, 2, None), ("EXPORT", "A" * 64, 2, None), ("EXPORT", "a" * 63, 2, None),
    ("MISSING", "a" * 64, 2, None), ("WITH SPACE", "a" * 64, 2, None)])
def test_the_fix_arm_follows_the_bench_export_env(tmp_path, export, sha, code, arm):
    exported = tmp_path / "exp"
    exported.mkdir()
    (exported / "policy.onnx").write_text("weights\n")
    spaced = tmp_path / "with space"
    spaced.mkdir()
    (spaced / "policy.onnx").write_text("weights\n")
    value = {"": "", "EXPORT": str(exported), "MISSING": str(tmp_path / "nowhere"), "WITH SPACE": str(spaced)}[export]
    script = tmp_path / "rewrite.sh"
    script.write_text('set -euo pipefail\nARMS=(\n  "m7-clocks2            --stage-gait clocks2"\n'
                      '  "m7-fix                --stage-gait clocks2 --cross-budget window --yaw-cap 0.6"\n)\n'
                      'CLOCKS2_POLICY=c2.onnx; CLOCKS2_POLICY_SHA256=c2sha\n'
                      f'FIX_EXPORT="{value}"; FIX_EXPORT_SHA256="{sha}"\n' + _arm_rewrite_block()
                      + 'for a in "${ARMS[@]}"; do read -r tag flags <<<"$a"; echo "$tag $flags"; done\n'
                      + 'echo "$FIX_GAIT $FIX_POLICY $FIX_POLICY_SHA256"\n')
    run = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=60)
    assert run.returncode == code, run.stderr
    if code == 0:
        lines = run.stdout.splitlines()
        assert lines[0] == "m7-clocks2 --stage-gait clocks2"
        assert lines[1] == arm.replace("EXPORT", str(exported)).rsplit(" ", 1)[0]
        gait = arm.rsplit(" ", 1)[1]
        assert lines[2] == (f"{gait} {exported}/policy.onnx {sha}" if gait == "export" else "clocks2 c2.onnx c2sha")


def _fix_gate_function():
    match = re.search(r"^fix_bench_gate\(\) \{\n.*?^\}\n", REPLAY_GATES.read_text(), re.S | re.M)
    assert match
    return match.group(0)


def _fix_bench_dir(tmp_path, gait="clocks2", verdict=None, provenance=None, record=None, weights=None,
                   stage_fix=None, export_dir=None):
    files = {"scripts/mission7_plate_stage.py": "stage\n", "scripts/mission7_gates.py": "gates\n",
             "src/bhl_robust/mission/env.py": "env\n", "src/bhl_robust/eval/gait_clock.py": "clock\n"}
    for name, body in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(body)
    exported = tmp_path / "exported"
    exported.mkdir(exist_ok=True)
    (exported / "policy.onnx").write_text(f"{gait} weights\n")
    pinned = hashlib.sha256(f"{gait} weights\n".encode()).hexdigest()
    out = tmp_path / "results/fix-plate-bench-v2"
    out.mkdir(parents=True, exist_ok=True)
    options = {"cross_budget": "window", "yaw_cap_rps": .6, "turn_rate_rps": .6} if stage_fix is None else stage_fix
    verdict = ({"bench": "v2", "mode": "scored", "stage_gait": gait, "verdict": "PASS", "stage_fix": options}
               if verdict is None else verdict)
    (out / "verdict.json").write_text(json.dumps(verdict))
    recorded = {name: hashlib.sha256(body.encode()).hexdigest() for name, body in files.items()}
    prov = {"bench": "v2", "mode": "scored", "stage_gait": gait, "stage_fix": options,
            "sources_sha256": recorded if record is None else record,
            "weights_sha256": {"shipped": "s", gait: pinned} if weights is None else weights,
            "stage_gait_export": str(exported if export_dir is None else export_dir)}
    (out / "provenance.json").write_text(json.dumps(prov if provenance is None else {**prov, **provenance}))
    return out, pinned, sorted(files), exported


def _run_fix_gate(tmp_path, verdict_path, gait, pinned, policy, export_dir, files):
    script = tmp_path / "gate.sh"
    script.write_text("set -euo pipefail\n" + _fix_gate_function() + 'fix_bench_gate "$@"\n')
    return subprocess.run(["bash", str(script), str(verdict_path), gait, pinned, str(policy), export_dir, *files],
                          cwd=tmp_path, capture_output=True, text=True, timeout=60, env={**os.environ, "PY": sys.executable})


@pytest.mark.parametrize("gait", ["clocks2", "export"])
def test_fix_gate_passes_a_scored_v2_pass_with_both_options_and_the_pinned_weights(tmp_path, gait):
    out, pinned, files, exported = _fix_bench_dir(tmp_path, gait=gait)
    run = _run_fix_gate(tmp_path, out / "verdict.json", gait, pinned, exported / "policy.onnx",
                        str(exported) if gait == "export" else "", files)
    assert run.returncode == 0, run.stderr
    assert f"scored bench-v2 {gait} PASS with --cross-budget window --yaw-cap 0.6" in run.stdout
    assert "4 snapshot sources identical" in run.stdout


@pytest.mark.parametrize("change", [
    {"verdict": {"bench": "v2", "mode": "scored", "stage_gait": "clocks2", "verdict": "FAIL",
                 "stage_fix": {"cross_budget": "window", "yaw_cap_rps": .6}}},
    {"verdict": {"bench": "v2", "mode": "scored", "stage_gait": "clocks2", "verdict": "INCOMPLETE",
                 "stage_fix": {"cross_budget": "window", "yaw_cap_rps": .6}}},
    {"verdict": {"bench": "v2", "mode": "smoke", "stage_gait": "clocks2", "verdict": "PASS",
                 "stage_fix": {"cross_budget": "window", "yaw_cap_rps": .6}}},
    {"verdict": {"bench": "v2", "mode": "scored", "stage_gait": "export", "verdict": "PASS",
                 "stage_fix": {"cross_budget": "window", "yaw_cap_rps": .6}}},
    {"verdict": {"bench": "v2", "mode": "scored", "stage_gait": "clocks2", "verdict": "PASS"}},     # the clocks2 bench
    {"verdict": {"bench": "v2", "mode": "scored", "stage_gait": "clocks2", "verdict": "PASS",
                 "stage_fix": {"cross_budget": "window", "yaw_cap_rps": None}}},
    {"verdict": {"bench": "v2", "mode": "scored", "stage_gait": "clocks2", "verdict": "PASS",
                 "stage_fix": {"cross_budget": None, "yaw_cap_rps": .6}}},
    {"provenance": {"stage_fix": None}}, {"provenance": {"stage_fix": {"cross_budget": "window", "yaw_cap_rps": .5}}},
    {"provenance": {"mode": "smoke"}}, {"provenance": {"stage_gait": "export"}},
    {"weights": {"shipped": "s", "clocks2": "0" * 64}}, {"record": {"scripts/mission7_gates.py": "0" * 64}}])
def test_fix_gate_refuses_anything_but_that(tmp_path, change):
    out, pinned, files, exported = _fix_bench_dir(tmp_path, **change)
    run = _run_fix_gate(tmp_path, out / "verdict.json", "clocks2", pinned, exported / "policy.onnx", "", files)
    assert run.returncode == 3 and "RELEASE_BLOCKED m7-fix" in run.stderr, (run.stdout, run.stderr)


def test_fix_gate_refuses_changed_weights_sources_exports_and_missing_modules(tmp_path):
    out, pinned, files, exported = _fix_bench_dir(tmp_path / "a")
    (exported / "policy.onnx").write_text("retrained\n")
    run = _run_fix_gate(tmp_path / "a", out / "verdict.json", "clocks2", pinned, exported / "policy.onnx", "", files)
    assert run.returncode == 3 and "pinned sha256" in run.stderr
    for module in ("gait_clock.py", "env.py"):
        out, pinned, files, exported = _fix_bench_dir(tmp_path / f"b-{module}")
        run = _run_fix_gate(tmp_path / f"b-{module}", out / "verdict.json", "clocks2", pinned, exported / "policy.onnx",
                            "", [f for f in files if not f.endswith(module)])
        assert run.returncode == 3 and module in run.stderr
    out, pinned, files, exported = _fix_bench_dir(tmp_path / "c")
    (tmp_path / "c/src/bhl_robust/mission/env.py").write_text("env, edited after the bench\n")
    run = _run_fix_gate(tmp_path / "c", out / "verdict.json", "clocks2", pinned, exported / "policy.onnx", "", files)
    assert run.returncode == 3 and "env.py" in run.stderr
    out, pinned, files, exported = _fix_bench_dir(tmp_path / "d", gait="export", export_dir=tmp_path / "elsewhere")
    run = _run_fix_gate(tmp_path / "d", out / "verdict.json", "export", pinned, exported / "policy.onnx",
                        str(exported), files)
    assert run.returncode == 3 and "ran the export" in run.stderr
    out, pinned, files, exported = _fix_bench_dir(tmp_path / "e", gait="export")
    run = _run_fix_gate(tmp_path / "e", out / "verdict.json", "export", pinned, exported / "policy.onnx", "", files)
    assert run.returncode == 3                                                  # export run, no export directory given
    out, pinned, files, exported = _fix_bench_dir(tmp_path / "f")
    run = _run_fix_gate(tmp_path / "f", out / "verdict.json", "clocks2", pinned, exported / "policy.onnx",
                        str(exported), files)
    assert run.returncode == 3 and "clocks2 preset" in run.stderr
    run = _run_fix_gate(tmp_path / "f", out / "verdict.json", "m3", pinned, exported / "policy.onnx", "", files)
    assert run.returncode == 3
    run = _run_fix_gate(tmp_path / "f", tmp_path / "nowhere/verdict.json", "clocks2", pinned, exported / "policy.onnx",
                        "", files)
    assert run.returncode == 3
    run = _run_fix_gate(tmp_path / "f", out / "verdict.json", "clocks2", pinned, exported / "policy.onnx", "", [])
    assert run.returncode == 3


# ---- the launcher ---------------------------------------------------------------------------------------------------

def _header():
    return LAUNCHER.read_text().split("set -euo pipefail")[0]


def _flat(text):
    return " ".join(line.lstrip("#").strip() for line in text.splitlines())


def test_launcher_header_carries_the_frozen_rule_chain_labels_budget_and_disclosures():
    flat = _flat(_header())
    for clause in (
            FROZEN_RULE,
            "PREDECLARED RULE (frozen before any episode; bench v2's rule verbatim, unchanged; the chain as M3's",
            "Then the unchanged exact ten-fall replay once with the same stage gait and --cross-budget window "
            "--yaw-cap 0.6, through the existing replay-gate machinery (complete, 10 episodes, 10 upright, 0 falls; "
            "release arm m7-fix).",
            "Then the route gate as coded by M1 with the same stage gait and options: Doors and Transport each >= 16/16 "
            "successes on validation layouts 0-15 (32 episodes).",
            "Here N = 41 (0 deg: 12, +90: 11, -90: 10, 180: 8), so PASS needs >= 39/41 clears, 0 falls, and >= 11, 10, 9, "
            "7 clears at 0, +90, -90, 180 deg.",
            "LEARNED gaits: the shipped arms-dr1.0-s0 gait outside the stage + the LEARNED stage gait",
            "SCRIPTED stage", "ORACLE layout and plate pose", "never from an exit code",
            "41 bench-v2 crossings (this job) + 10 exact-replay episodes + 32 route-gate episodes = 83 more Mission 7 "
            "episodes", "NOT part of the line",
            "Bundling means a PASS is not attributable to one change.",
            "Faster turns in narrow cells add wall-contact and fall risk (6 of 8 180-deg turns already touched walls at "
            "0.40); the clocks2 bench's only fall (21517668: L22 d0, -90 deg, square, standstill) was during the in-place "
            "turn.",
            "Crossing until clear produced a replay fall before (9/10)",
            "when 0.2 s of the 10 s clear window counted from takeover remains, whichever comes first, instead of at the "
            "fixed cross_max_s = 4.0 s.", "The clear definition is unchanged.",
            "its capture at 0.78 m from the plate", "the crossing can get less than the default 4.0 s",
            "CAP semantics -- every command below the cap is delivered unchanged.",
            "re-encodes their yaw channel only and they keep their physical values",
            "the crossing heading hold stays clip(1.2 x error, +-0.40)", "0.5999994 rad/s",
            "2 pi / 0.60 = 10.47 s (15.71 s at 0.40)", "(pi - 0.15)/0.60 = 4.99 s",
            "records wz at the 0.40 scale", "the route gate's verdict reads only each episode's success",
            "a forced deviation from the instruction \"exploration layouts >= 290\"",
            'SPLITS["train"] = (0, 256)', "The crossing diagnosis (D) runs on train layouts 170-249",
            "and its smoke on 32, 33 and 38",
            "restricted to train layouts 32-169 except 32, 33 and 38",
            "never runs a crossing on 170-249",
            "F1's selected PlateCross v2 seed if F1 ran and selected one, else clock-s2",
            "M7_FIX_STAGE_GAIT_EXPORT", "DROPPED (23 of the 32 walking crossings)", "Run: 32 standstill + 9 walking",
            "Route-gate snapshot (a gap inherited from M2/M3/clocks2; disclosed, not closed)",
            "None of those files is to be edited between this bench and the route-gate submission",
            "ONE run: the output directory is fixed whatever the stage gait, and refused if it exists."):
        assert clause in flat, clause
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    dropped = ", ".join(f"L{layout_index} d{door}" for layout_index, door in v2.DROPPED)
    assert dropped in flat
    assert abs((np.pi - .15) / .60 - 4.99) < .005 and abs(2 * np.pi / .6 - 10.47) < .005


def test_launcher_bench_mode_sources_options_and_release():
    text = LAUNCHER.read_text()
    bench = text[text.index('if [ "$mode" = bench ]; then'):text.index('[ "$mode" = smoke ] || {')]
    assert '[ ! -e "$OUT" ] || { echo "refusing: $OUT exists' in bench
    assert 'git diff --quiet HEAD -- "${sources[@]}"' in bench
    assert 'FIX_ARGS=(--cross-budget window --yaw-cap 0.6)' in text
    assert '--mode scored "${GAIT_ARGS[@]}" "${FIX_ARGS[@]}" --preflight' in bench
    assert '--mode scored "${GAIT_ARGS[@]}" "${FIX_ARGS[@]}"\n' in bench
    assert "OUT=${M7_BENCH_V2_FIX_OUT:-$CAMPAIGN/fix-plate-bench-v2}" in bench
    sources = re.search(r"^  sources=\((.*?)\)\n", text, re.S | re.M)
    expanded = set()
    for item in sources.group(1).split():
        expanded.update({str(p.relative_to(_REPO)) for p in _REPO.glob(item)} if "*" in item else
                        ({str(p.relative_to(_REPO)) for p in (_REPO / item).glob("*.py")} if (_REPO / item).is_dir()
                         else {item}))
    assert expanded == set(v2.source_files(_REPO, "clocks2", fix=True)) == set(v2.source_files(_REPO, "export", fix=True))
    assert f"CLOCKS2_POLICY_SHA256={CLOCKS2_SHA256}" in text
    m = _stage_module()
    assert re.search(r"^CLOCKS2_EXPORT=(\S+)$", text, re.M).group(1) == f"external/Berkeley-Humanoid-Lite/{m.CLOCKS2_EXPORT}"


def _release_function():
    match = re.search(r"^release_if_pass\(\) \{\n.*?^\}\n", LAUNCHER.read_text(), re.S | re.M)
    assert match
    return match.group(0)


@pytest.mark.parametrize("export", ["", "/some/export"])
@pytest.mark.parametrize("verdict, release_rc, released, rc, text", [
    ("PASS", 0, True, 0, "releasing the exact ten-fall replay with --stage-gait"),
    ("PASS", 3, True, 1, "RELEASE_BLOCKED"),
    ("PASS", 1, True, 1, "M7 BENCH V2 FIX RELEASE FAILED (exit 1)"),
    ("FAIL", 0, False, 0, "chain stops"),
    ("INCOMPLETE", 0, False, 0, "nothing released")])
def test_launcher_releases_the_fix_replay_only_on_a_pass_read_from_json(tmp_path, verdict, release_rc, released, rc,
                                                                        text, export):
    out = tmp_path / "bench"
    out.mkdir()
    (out / "verdict.json").write_text(json.dumps({"verdict": verdict}))
    calls = tmp_path / "calls"
    stub = tmp_path / "release.sh"
    stub.write_text(f'#!/bin/bash\necho "$@ V=$M7_FIX_BENCH_VERDICT E=$M7_FIX_STAGE_GAIT_EXPORT '
                    f'S=$M7_FIX_STAGE_GAIT_EXPORT_SHA256" >> {calls}\nexit {release_rc}\n')
    script = tmp_path / "run.sh"
    sha = "b" * 64 if export else ""
    script.write_text("set -euo pipefail\n" + _release_function() + f'release_if_pass "{out}"\n')
    env = {**os.environ, "PY": sys.executable, "RELEASE": str(stub), "DRY_RUN": "0"}
    for key in ("M7_FIX_BENCH_VERDICT", "M7_FIX_STAGE_GAIT_EXPORT", "M7_FIX_STAGE_GAIT_EXPORT_SHA256"):
        env.pop(key, None)
    gait = "export" if export else "clocks2"
    run = subprocess.run(["bash", "-c", f'REPO=/repo PY="$PY" RELEASE="$RELEASE" DRY_RUN=0 STAGE_GAIT={gait} '
                                        f'STAGE_GAIT_EXPORT="{export}" STAGE_GAIT_EXPORT_SHA256="{sha}" '
                                        f'FIX_ARGS=(--cross-budget window --yaw-cap 0.6); source "{script}"'],
                         env=env, capture_output=True, text=True, timeout=60)
    assert run.returncode == rc, run.stderr
    assert text in run.stdout and calls.exists() == released
    if released:
        assert calls.read_text().strip() == (f"--only m7-fix --submit V={out}/verdict.json E={export} S={sha}")


def _smoke_limits_function():
    match = re.search(r"^smoke_limits_ok\(\) \{\n.*?^\}\n", LAUNCHER.read_text(), re.S | re.M)
    assert match
    return match.group(0)


@pytest.mark.parametrize("name, job_id, limit, ok", [
    ("m7-plate-bench-v2-fix-smoke", "123", "1:00:00", True), ("m7-plate-bench-v2-fix-smoke", "123", "59:00", True),
    ("m7-plate-bench-v2-fix-smoke", "123", "1:00:01", False), ("m7-plate-bench-v2-fix-smoke", "123", "1-00:00:00", False),
    ("m7-plate-bench-v2-fix-smoke", "123", "UNLIMITED", False), ("m7-plate-bench-v2-fix-smoke", "", "1:00:00", False),
    ("m7-plate-bench-v2-fix", "123", "1:00:00", False)])
def test_launcher_smoke_mode_enforces_its_name_and_time_limit(tmp_path, name, job_id, limit, ok):
    stubs = tmp_path / "bin"
    stubs.mkdir()
    (stubs / "squeue").write_text(f'#!/bin/bash\nprintf "%s\\n" "{limit}"\n')
    (stubs / "sleep").write_text("#!/bin/bash\nexit 0\n")
    for stub in stubs.iterdir():
        stub.chmod(0o755)
    script = tmp_path / "smoke.sh"
    script.write_text("set -euo pipefail\n" + _smoke_limits_function() + "smoke_limits_ok\n")
    env = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "SLURM_JOB_NAME": name, "SLURM_JOB_ID": job_id}
    run = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=60)
    assert (run.returncode == 0) == ok, (run.stdout, run.stderr)


def test_launcher_smoke_never_runs_a_scored_or_reserved_layout_and_checks_identity_and_the_options():
    text = LAUNCHER.read_text()
    smoke = text[text.index('[ "$mode" = smoke ] || {'):]
    assert smoke.index("smoke_limits_ok || exit 2") < smoke.index("SMOKE=${M7_SMOKE_DIR") < smoke.index("mkdir -p")
    assert "--job-name=m7-plate-bench-v2-fix-smoke --time=01:00:00" in text
    crossings = re.search(r"^FIX_SMOKE_CROSSINGS=(\S+)", smoke, re.M).group(1)
    specs = v2.parse_fix_smoke_specs(crossings)
    assert {s["entry"] for s in specs} == {"standstill", "walking"} and 180 in {s["heading_deg"] for s in specs}
    identity = re.search(r"^IDENTITY_CROSSING=(\d+):(\d)\s", smoke, re.M)
    v1_crossing = re.search(r"^V1_SMOKE_CROSSING=(\d+):(\d)\s", smoke, re.M)
    route = int(re.search(r"^ROUTE_LAYOUT=(\d+)", smoke, re.M).group(1))
    used = {s["layout"] for s in specs} | {int(identity.group(1)), int(v1_crossing.group(1)), route}
    assert used == set(SMOKE_LAYOUTS) and all(100 <= layout_index <= 199 for layout_index in used)
    assert not used & ({32, 33, 38} | set(range(0, 32)) | set(range(170, 256)))   # D's smoke and run, the bench
    assert all(32 <= layout_index < 170 for layout_index in used)
    assert v1.parse_crossings(f"{v1_crossing.group(1)}:{v1_crossing.group(2)}")
    assert 'split="train", seed=4000 + layout' in smoke
    stage_runs = re.findall(r'"\$PY" scripts/mission7_plate_stage\.py(?:[^\n]*\\\n)*[^\n]*', smoke)
    probe_runs = re.findall(r'"\$PY" scripts/mission7_route_handoff_probe\.py(?:[^\n]*\\\n)*[^\n]*', smoke)
    assert len(stage_runs) == 2 and len(probe_runs) == 1
    assert all("--preflight" in run for run in stage_runs + probe_runs)
    for cli in re.findall(r'"\$scripts/mission7_(?:plate_stage|route_handoff_probe)\.py"(?:[^\n]*\\\n)*[^\n]*', smoke):
        assert "--preflight" in cli
    for step in (f"REFERENCE_COMMIT={REF_COMMIT}",
                 f"REFERENCE_STAGE_SHA256={REF_SHA256['scripts/mission7_plate_stage.py']}",
                 f"REFERENCE_BENCH_V2_SHA256={REF_SHA256['scripts/mission7_plate_bench_v2.py']}",
                 f"REFERENCE_PROBE_SHA256={REF_SHA256['scripts/mission7_route_handoff_probe.py']}",
                 f"REFERENCE_ENV_SHA256={REF_SHA256['src/bhl_robust/mission/env.py']}",
                 f"REFERENCE_V1_SHA256={REF_SHA256['scripts/mission7_plate_bench.py']}",
                 'REF_PATH="$SMOKE/ref-tree/src:$SMOKE/ref-tree/scripts"',
                 "for gait in shipped turnboth m3 clocks2; do",
                 'cmp "$SMOKE/identity-$gait-ref/crossing.json" "$SMOKE/identity-$gait-new/crossing.json"',
                 'cmp "$SMOKE/identity-v1-ref.json" "$SMOKE/identity-v1-new.json"',
                 'cmp "$SMOKE/identity-route-$gait-ref/route.json" "$SMOKE/identity-route-$gait-new/route.json"',
                 'cmp "$SMOKE/preflight-replay-$gait-ref.txt" "$SMOKE/preflight-replay-$gait-new.txt"',
                 'cmp "$SMOKE/preflight-probe-$gait-ref.txt" "$SMOKE/preflight-probe-$gait-new.txt"',
                 'route_run "$SMOKE/route-fix" "$STAGE_GAIT" 120 "$PYTHONPATH" 1 "$STAGE_GAIT_EXPORT" '
                 '"$STAGE_GAIT_EXPORT_SHA256"',
                 '--mode smoke "${GAIT_ARGS[@]}" \\\n      "${FIX_ARGS[@]}" --crossings "$FIX_SMOKE_CROSSINGS"\n',
                 'mission7_plate_stage.select_export_gait(sys.argv[7], sys.argv[8])',
                 'first.get("event") == f"swap_to_{gait}" and first.get("controller") == CTRL',
                 'turn_wz_sent_and_received_is_0.60', "route_fix_controller_receives_0.60_in_the_turns",
                 "route_fix_route_steps_deliver_the_routes_physical_command", "crossing_ends_at_clear_or_budget",
                 "f2_crossing_beyond_4s_reported", "sys.exit(0 if not problems else 1)"):
        assert step in text, step
    for path, sha in REF_SHA256.items():
        assert hashlib.sha256(_git_blob(path)).hexdigest() == sha


def _fake_tree(tmp_path, m):
    """A git repo with every source the bench mode checks, the clock-s2 policy at its pinned path and a fake bench."""
    repo = tmp_path / "repo"
    names = ["scripts/mission7_plate_bench_v2.py", "scripts/mission7_plate_stage.py", "scripts/mission7_gates.py",
             "src/bhl_robust/mission/env.py", "src/bhl_robust/__init__.py", "src/bhl_robust/sensor_io.py",
             "src/bhl_robust/eval/__init__.py", "src/bhl_robust/eval/multi_robot.py", "src/bhl_robust/eval/mjcf_assets.py",
             "src/bhl_robust/eval/livery.py", "src/bhl_robust/eval/team_sensors.py", "src/bhl_robust/eval/gait_clock.py",
             "slurm/repo20260923/cpu_m7_plate_bench.sbatch", "slurm/repo20260923/cpu_m7_plate_bench_v2.sbatch",
             "slurm/repo20260923/cpu_m7_plate_bench_v2_clocks2.sbatch", "slurm/repo20260923/cpu_m7_plate_bench_v2_fix.sbatch"]
    for name in names:
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(f"# {name}\n")
    (repo / "scripts/mission7_plate_bench_v2.py").write_text(
        "import json, os, pathlib, sys\n"
        "argv = sys.argv[1:]\n"
        "with open(os.environ['FAKE_LOG'], 'a') as log:\n"
        "    log.write(json.dumps(argv) + '\\n')\n"
        "if '--preflight' not in argv:\n"
        "    out = pathlib.Path(argv[argv.index('--out') + 1])\n"
        "    out.mkdir(parents=True)\n"
        "    (out / 'verdict.json').write_text(json.dumps({'verdict': os.environ['FAKE_VERDICT']}))\n")
    policy = repo / "external/Berkeley-Humanoid-Lite" / m.CLOCKS2_EXPORT / "policy.onnx"
    policy.parent.mkdir(parents=True)
    shutil.copyfile(UPSTREAM / m.CLOCKS2_EXPORT / "policy.onnx", policy)
    export = tmp_path / "f1-export"
    export.mkdir()
    (export / "policy.onnx").write_text("plate-cross v2 weights\n")
    git = ["git", "-C", str(repo)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", *names], check=True)
    subprocess.run([*git, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "fake"], check=True)
    release = tmp_path / "release.sh"
    release.write_text(f'#!/bin/bash\necho "$@ V=$M7_FIX_BENCH_VERDICT E=$M7_FIX_STAGE_GAIT_EXPORT '
                       f'S=$M7_FIX_STAGE_GAIT_EXPORT_SHA256" >> {tmp_path}/release-calls\nexit 0\n')
    return repo, export, hashlib.sha256(b"plate-cross v2 weights\n").hexdigest(), release


def _bench(tmp_path, repo, release, verdict="FAIL", **extra):
    env = {key: value for key, value in os.environ.items() if not key.startswith(("SLURM_", "M7_FIX_"))}
    env.update(M7_REPO=str(repo), PYTHON=sys.executable, M7_BENCH_RELEASE=str(release), FAKE_VERDICT=verdict,
               FAKE_LOG=str(tmp_path / "bench-calls"), **extra)
    return subprocess.run(["bash", str(LAUNCHER), "bench"], env=env, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("verdict", ["PASS", "FAIL"])
def test_launcher_bench_mode_runs_on_a_fake_tree_with_clock_s2(tmp_path, verdict):
    m = _stage_module()
    _need_exports(m)
    repo, export, sha, release = _fake_tree(tmp_path, m)
    run = _bench(tmp_path, repo, release, verdict=verdict)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "M7_PLATE_BENCH_V2_FIX_COMPLETE" in run.stdout and "gait=clocks2" in run.stdout
    calls = [json.loads(line) for line in (tmp_path / "bench-calls").read_text().splitlines()]
    out = "results/mission7-campaign-20260923/fix-plate-bench-v2"
    expected = ["--repo", str(repo), "--out", out, "--mode", "scored", "--stage-gait", "clocks2",
                "--cross-budget", "window", "--yaw-cap", "0.6"]
    assert calls == [expected + ["--preflight"], expected]
    assert (tmp_path / "release-calls").exists() == (verdict == "PASS")
    if verdict == "PASS":
        assert (tmp_path / "release-calls").read_text().strip() == (
            f"--only m7-fix --submit V={repo}/{out}/verdict.json E= S=")
    again = _bench(tmp_path, repo, release, verdict=verdict)        # ONE run: the output now exists
    assert again.returncode == 2 and "exists" in again.stderr


def test_launcher_bench_mode_runs_an_export_pinned_by_sha256(tmp_path):
    m = _stage_module()
    _need_exports(m)
    repo, export, sha, release = _fake_tree(tmp_path, m)
    run = _bench(tmp_path, repo, release, verdict="PASS", M7_FIX_STAGE_GAIT_EXPORT=str(export),
                 M7_FIX_STAGE_GAIT_EXPORT_SHA256=sha)
    assert run.returncode == 0, run.stdout + run.stderr
    calls = [json.loads(line) for line in (tmp_path / "bench-calls").read_text().splitlines()]
    assert calls[1][6:] == ["--stage-gait", "export", "--stage-gait-export", str(export), "--stage-gait-export-sha256",
                            sha, "--cross-budget", "window", "--yaw-cap", "0.6"]
    assert (tmp_path / "release-calls").read_text().strip().endswith(f"E={export} S={sha}")


@pytest.mark.parametrize("case", ["no_sha", "sha_only", "bad_sha", "wrong_sha", "missing_policy", "dirty", "uncommitted"])
def test_launcher_bench_mode_refuses(tmp_path, case):
    m = _stage_module()
    _need_exports(m)
    repo, export, sha, release = _fake_tree(tmp_path, m)
    extra = {"no_sha": {"M7_FIX_STAGE_GAIT_EXPORT": str(export)},
             "sha_only": {"M7_FIX_STAGE_GAIT_EXPORT_SHA256": sha},
             "bad_sha": {"M7_FIX_STAGE_GAIT_EXPORT": str(export), "M7_FIX_STAGE_GAIT_EXPORT_SHA256": "xyz"},
             "wrong_sha": {"M7_FIX_STAGE_GAIT_EXPORT": str(export), "M7_FIX_STAGE_GAIT_EXPORT_SHA256": "0" * 64},
             "missing_policy": {"M7_FIX_STAGE_GAIT_EXPORT": str(tmp_path / "nowhere"),
                                "M7_FIX_STAGE_GAIT_EXPORT_SHA256": sha}}.get(case, {})
    if case == "dirty":
        (repo / "src/bhl_robust/mission/env.py").write_text("# edited\n")
    if case == "uncommitted":
        subprocess.run(["git", "-C", str(repo), "rm", "-q", "--cached", "src/bhl_robust/eval/gait_clock.py"], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "drop"],
                       check=True)
    run = _bench(tmp_path, repo, release, verdict="PASS", **extra)
    assert run.returncode != 0 and not (tmp_path / "bench-calls").exists(), run.stdout + run.stderr
    assert not (repo / "results/mission7-campaign-20260923/fix-plate-bench-v2").exists()
    assert not (tmp_path / "release-calls").exists()
