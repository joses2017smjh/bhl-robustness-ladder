"""Mission 7 workstream m7-clocks2: arms-turngait-clock-s2 as the stage gait on bench v2 (the generic export override).

Frozen design: SLURM_JOBS.md, "User approval recorded 2026-10-02 14:15", item (A).  No MuJoCo episode runs here: the
stage logic runs on a kinematic stub with upstream's controllers (a real GaitClockRlController is swapped in), the
export check runs on copies of the real exports, the committed M3 code (e33e44f) is loaded from git for the default-path
comparisons, and the launchers run as extracted functions or in their plan / dry modes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
LAUNCHER = _REPO / "slurm/repo20260923/cpu_m7_plate_bench_v2_clocks2.sbatch"
M3_LAUNCHER = _REPO / "slurm/repo20260923/cpu_m7_plate_bench_v2.sbatch"
FOLLOWUP = _REPO / "slurm/repo20260923/m7_replay_gate_followup.sbatch"
REPLAY_GATES = _REPO / "slurm/repo20260923/submit_m7_replay_gates.sh"
STAGE_SUBMITTER = _REPO / "scripts/submit_mission7_plate_stage.py"
ROUTE_SUBMITTER = _REPO / "scripts/submit_mission7_route_handoff.py"
PROBE = _REPO / "scripts/mission7_route_handoff_probe.py"
STAGE = _REPO / "scripts/mission7_plate_stage.py"
M3_DIR = _REPO / "results/mission7-campaign-20260923/m3-plate-bench-v2"
REPLAY_SOURCE = "results/mission7-replay-smoke-20260921"
REPLAY_BASELINE = "results/mission7-approach-followup-20260922/replay-diagnose-cn-c22"
SIM_STACK = ("mujoco", "torch", "onnxruntime", "omegaconf", "rsl_rl", "berkeley_humanoid_lite_lowlevel")
# the committed M3 code: stage, bench v2 and probe as the m3 bench ran them (its provenance.json records the same)
M3_COMMIT = "e33e44f"
M3_STAGE_SHA256 = "22418834da39658ba4ff2120044bcab6f60cc5f6b5be04e34061f9f8997e7deb"
M3_BENCH_V2_SHA256 = "8a1b08987f277fa5b56b88cb382b7a4cd4c309607eeb55032bd3ae1879cf5e72"
M3_PROBE_SHA256 = "396bdbca86aa73ebdb01379feffee9df6d8e3305ad30c59c94c720f050da3056"
CLOCKS2_SHA256 = "c1862f1e4429c1e6ac2e4fae1391f3f40ccc353382c8bbf4b88efee4c6c8d8de"
FREE_KEYS = ["command_velocity", "gait_clock", "num_observations", "policy_checkpoint_path"]
FROZEN_RULE = (
    "Bench v2: with N crossings run (64 minus the dropped set) and N_h per heading, bench v2 PASSES iff clears >= "
    "N - 2, 0 falls, and clears >= N_h - 1 for every heading. Otherwise FAIL; INCOMPLETE if any declared crossing is "
    "missing. The clear definition is unchanged: mission7_gates' real clear (base along >= +0.35 m past the plate "
    "centre along the door direction), within 10 s of takeover.")
N_ACT = 22
DEFAULT_Q = np.linspace(-0.3, 0.3, N_ACT).astype(np.float32)


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


# ---- the kinematic stub with upstream's controllers ---------------------------------------------------------------

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


def _robot_obs(n, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        q = rng.normal(0, 1, 4)
        q[0] += 4.0
        q /= np.linalg.norm(q)
        out.append(np.concatenate([q, rng.normal(0, 0.5, 3), DEFAULT_Q + rng.normal(0, 0.1, N_ACT),
                                   rng.normal(0, 1.0, N_ACT), [3.0], rng.uniform(-1, 1, 3)]).astype(np.float32))
    return out


class _Data:
    def __init__(self):
        self.time = 0.
        self.xpos = np.zeros((2, 3))
        self.qpos = np.zeros(7)


class _Slot:
    body_id, qpos_adr = 1, 0


class StubEnv:
    """runner.d / slot / layout / state / controller as PlateStage reads them; a kinematic plant.  The shipped
    controller is upstream's RlController (75 wide) carrying a sentinel policy, as MissionEnv builds it."""

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

    def step(self, command, dt=.2, move=True):
        yaw = self.yaw()
        vx, vy, wz = np.asarray(command, dtype=float)
        world = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]) @ [vx, vy]
        if move:
            self.set_pose(self.runner.d.xpos[1, :2] + world * dt, yaw + wz * dt)
        self.runner.d.time = round(self.runner.d.time + dt, 10)
        self.controller.prev_actions[:] = .25 + .01 * (self.runner.d.time % 1)   # the active controller acted


def _export_gait(clock=True, seed=0):
    width = 77 if clock else 75
    return _Policy(width, seed), _cfg(width, clock), {"stage_gait": "stub", "num_observations": width}


def _bench_like(m, stage_gait, layout_index=7, door=1, plate="round", heading=90, export_gait=None, **kwargs):
    layout = generate("train", layout_index)
    g = v2.crossing_geometry(layout, door, plate, heading, "standstill")
    env = StubEnv(layout, g["pre_point"], g["entry_yaw_rad"] + 1e-3)
    if stage_gait in ("clocks2", "export"):
        kwargs["export_gait"] = export_gait or _export_gait()
    elif stage_gait == "turnboth":
        kwargs["turnboth_policy"] = "turnboth-policy"
    return env, m.PlateStage(env, stage_gait=stage_gait, **kwargs), g


def _run(stage, env, steps=400, move=True):
    log = []
    for _ in range(steps):
        command, phase = stage.command(np.zeros(3))
        log.append((phase, np.asarray(command, dtype=float).round(12).tolist(), float(env.runner.d.time)))
        if phase == "recorded":
            break
        env.step(command, move=move)
    return log


# ---- the generic override on the stub: controller swap, prev_actions, clock phase, swap back ---------------------

def test_takeover_swaps_in_a_fresh_clock_controller_at_phase_zero_and_handback_restores_the_shipped_one():
    m = _stage_module()
    from bhl_robust.eval.gait_clock import clock_features
    policy, cfg, info = _export_gait()
    env, stage, g = _bench_like(m, "clocks2", export_gait=(policy, cfg, info))
    shipped = env.controller
    stage._start(1, g["side"], env.runner.d.time)
    ctrl = env.controller
    assert type(ctrl).__name__ == "GaitClockRlController" and ctrl is not shipped and ctrl.policy is policy
    assert ctrl.policy_observations.shape == (1, 77) and not ctrl.policy_observations.any()
    assert not ctrl.prev_actions.any() and ctrl.clock_step == 0
    assert shipped.policy == "shipped-policy" and shipped.prev_actions.any()        # set aside untouched until hand-back
    swap_in = stage.gait_events[0]
    assert swap_in["event"] == "swap_to_clocks2" and swap_in["controller"] == "GaitClockRlController"
    assert (swap_in["obs_width"], swap_in["clock_step"], swap_in["policy_observations_zero"]) == (77, 0, True)
    assert swap_in["prev_actions_after_norm"] == 0. and swap_in["prev_actions_before_norm"] > 0
    assert swap_in["takeover_yaw_rad"] == pytest.approx(env.yaw())
    # the swapped-in controller's frames carry the gait clock from phase 0 (Isaac's episode start), 77 wide
    for k, obs in enumerate(_robot_obs(7)):
        ctrl.update(obs)
        assert policy.inputs[-1].shape == (1, 77)
        np.testing.assert_array_equal(ctrl.policy_observations[0, -2:], clock_features(k, .04, .8))
    log = _run(stage, env)
    assert log[-1][0] == "recorded" and env.controller is shipped and shipped.policy == "shipped-policy"
    assert not shipped.prev_actions.any()                                              # zeroed at hand-back
    swap_back = stage.gait_events[-1]
    assert [e["event"] for e in stage.gait_events] == ["swap_to_clocks2", "swap_to_shipped"]
    assert swap_back["controller"] == "RlController" and swap_back["obs_width"] == 75
    assert swap_back["stage_controller"] == "GaitClockRlController" and swap_back["stage_controller_obs_width"] == 77
    assert swap_back["stage_controller_clock_step"] == 7
    np.testing.assert_allclose(swap_back["stage_controller_last_clock"], clock_features(6, .04, .8), atol=1e-7)
    assert swap_back["prev_actions_after_norm"] == 0. and swap_back["prev_actions_before_norm"] > 0
    assert stage.shipped_controller is None and stage.done == {1}


def test_every_takeover_builds_a_new_controller_with_the_clock_at_zero():
    m = _stage_module()
    env, stage, g = _bench_like(m, "clocks2", layout_index=8, door=1, heading=0)
    stage._start(1, g["side"], env.runner.d.time)
    first = env.controller
    for obs in _robot_obs(5):
        first.update(obs)
    assert first.clock_step == 5
    _run(stage, env)
    shipped = env.controller
    stage._start(0, g["side"], env.runner.d.time)            # a second plate in the same episode
    second = env.controller
    assert second is not first and second is not shipped and second.clock_step == 0
    assert not second.policy_observations.any() and not second.prev_actions.any()
    assert [e["event"] for e in stage.gait_events] == ["swap_to_clocks2", "swap_to_shipped", "swap_to_clocks2"]


def test_the_stage_law_is_m2s_turnboth_law_unchanged():
    """Same stub, same drive: clocks2 / export issue exactly turnboth's commands and history; only the swap differs."""
    m = _stage_module()
    for layout_index, door, heading, plate in ((7, 1, 90, "round"), (8, 1, 0, "round"), (11, 1, 180, "square"),
                                              (6, 0, -90, "round")):
        runs = {}
        for gait in ("turnboth", "clocks2", "export"):
            env, stage, g = _bench_like(m, gait, layout_index=layout_index, door=door, heading=heading, plate=plate)
            stage._start(door, g["side"], env.runner.d.time)
            runs[gait] = (_run(stage, env), stage.history, stage.done)
        assert runs["clocks2"] == runs["turnboth"] == runs["export"], (layout_index, door)
        phases = [h["phase"] for h in runs["clocks2"][1]]
        assert phases == ["approach", "settle", "turn", "cross", "turn_back", "recorded"]
        turns = [c for p, c, _ in runs["clocks2"][0] if p in ("turn", "turn_back")]
        assert (turns or heading == 0)    # a 0-deg entry starts inside the 0.15 rad exit: no turn command
        assert all(c[0] == 0. and c[1] == 0. and abs(c[2]) == m.TURNBOTH_TURN_RATE for c in turns)
        crossing = [c for p, c, _ in runs["clocks2"][0] if p == "cross"]
        assert crossing and all(0. <= c[0] <= m.TURNBOTH_CRUISE and c[1] == 0. and abs(c[2]) <= m.TURNBOTH_WZ_WALK
                                for c in crossing)


def test_handback_restores_the_route_probes_wrapped_shipped_controller():
    """The route probe wraps the shipped controller's update (traced_update, an instance attribute); the swap keeps
    that object aside and gives the same object back, so the wrapper is in place again after the hand-back."""
    m = _stage_module()
    env, stage, g = _bench_like(m, "clocks2")
    shipped = env.controller
    calls = []

    def traced_update(robot_observations):
        calls.append(1)
        return np.zeros(N_ACT)
    shipped.update = traced_update
    stage._start(1, g["side"], env.runner.d.time)
    assert env.controller.update.__name__ == "update" and env.controller is not shipped
    _run(stage, env)
    assert env.controller is shipped and env.controller.update is traced_update
    env.controller.update(_robot_obs(1)[0])
    assert calls == [1]


def test_a_75_observation_export_swaps_in_upstreams_controller():
    m = _stage_module()
    policy, cfg, info = _export_gait(clock=False)
    env, stage, g = _bench_like(m, "export", export_gait=(policy, cfg, info))
    shipped = env.controller
    stage._start(1, g["side"], env.runner.d.time)
    ctrl = env.controller
    assert type(ctrl).__name__ == "RlController" and ctrl is not shipped and ctrl.policy is policy
    assert stage.gait_events[0]["clock_step"] is None and stage.gait_events[0]["obs_width"] == 75
    ctrl.update(_robot_obs(1)[0])
    assert policy.inputs[-1].shape == (1, 75)
    _run(stage, env)
    assert env.controller is shipped and stage.gait_events[-1]["stage_controller_last_clock"] is None


def test_the_crossing_is_scored_by_mission7_gates():
    m = _stage_module()
    env, stage, g = _bench_like(m, "clocks2", layout_index=8, door=1, heading=0)
    stage._start(1, g["side"], env.runner.d.time)
    samples = []
    for _ in range(400):
        command, phase = stage.command(np.zeros(3))
        if phase == "recorded":
            break
        env.step(command)
        samples.append({"time_s": float(env.runner.d.time), "xy": env.runner.d.xpos[1, :2].tolist(), "tilt": .05})
    row = {"layout_index": 8, "layout_seed": env.layout.seed, "stage_history": stage.history, "samples": samples}
    [crossing] = gates.crossing_outcomes(row, env.layout)
    assert crossing["cleared"] and crossing["real_clear"] and crossing["door"] == 1


def test_export_gaits_refuse_align_yaw_press_hold_a_handback_without_takeover_and_an_unselected_export():
    m = _stage_module()
    env = StubEnv(generate("train", 0), [0., 0.], 0.)
    for gait in ("clocks2", "export"):
        for kwargs in ({"align_yaw": True}, {"press_hold": True}):
            with pytest.raises(ValueError):
                m.PlateStage(env, stage_gait=gait, export_gait=_export_gait(), **kwargs)
    with pytest.raises(RuntimeError, match="needs --stage-gait-export"):
        m.PlateStage(env, stage_gait="export")                      # nothing selected: fail loudly
    stage = m.PlateStage(env, stage_gait="clocks2", export_gait=_export_gait())
    stage.door = 0
    with pytest.raises(RuntimeError, match="without a takeover"):
        stage._swap_gait(0., "shipped")
    with pytest.raises(ValueError):
        m.PlateStage(env, stage_gait="clocks3")
    assert m.STAGE_GAITS == ("shipped", "turnboth", "m3")          # the M2/M3 tuple is untouched
    assert m.EXPORT_STAGE_GAITS == ("clocks2", "export") == v2.EXPORT_STAGE_GAITS_V2
    assert m.TURNBOTH_LAW_GAITS == ("turnboth", "clocks2", "export")


def test_selection_and_export_dirs():
    m = _stage_module()
    assert m.export_dir_for(UPSTREAM, "clocks2") == (UPSTREAM / m.CLOCKS2_EXPORT, CLOCKS2_SHA256)
    assert m.CLOCKS2_POLICY_SHA256 == CLOCKS2_SHA256
    with pytest.raises(RuntimeError):
        m.export_dir_for(UPSTREAM, "export")
    with pytest.raises(ValueError):
        m.export_dir_for(UPSTREAM, "m3")
    m.select_export_gait("relative/export", "ab12")
    assert m.export_dir_for(UPSTREAM, "export") == (Path("relative/export").resolve(), "ab12")
    assert "relative/export" in m.stage_gait_description("export")
    assert "arms-turngait-clock-s2" in m.stage_gait_description("clocks2")
    assert "M2's turnboth law unchanged" in m.export_description("clocks2")
    assert m.stage_gait_description("turnboth") == m.turnboth_description()
    law = m.turnboth_law_constants()
    assert (law["turn_rate_rps"], law["turn_exit_rad"], law["cruise_mps"], law["k_yaw"], law["wz_walk_rps"],
            law["cross_clear_m"], law["lateral_command"]) == (.40, .15, .30, 1.2, .40, gates.CLEAR_ALONG_M, 0.)


# ---- export validation on copies of the real exports ----------------------------------------------------------------

def _need_exports(m):
    for d in (UPSTREAM / m.CLOCKS2_EXPORT, UPSTREAM / m.SHIPPED_EXPORT):
        if not (d / "policy.onnx").is_file():
            pytest.skip(f"export not on disk: {d}")


def _copy(m, tmp_path, name, source="clocks2", policy_from=None, edit=None, pointer=None):
    from omegaconf import OmegaConf
    src = UPSTREAM / (m.CLOCKS2_EXPORT if source == "clocks2" else m.SHIPPED_EXPORT)
    out = tmp_path / name
    out.mkdir()
    shutil.copyfile(UPSTREAM / ((m.CLOCKS2_EXPORT if policy_from == "clocks2" else m.SHIPPED_EXPORT)
                                if policy_from else (m.CLOCKS2_EXPORT if source == "clocks2" else m.SHIPPED_EXPORT))
                    / "policy.onnx", out / "policy.onnx")
    cfg = OmegaConf.load(src / "deploy.yaml")
    cfg.policy_checkpoint_path = str(pointer or (out / "policy.onnx"))
    if edit is not None:
        edit(cfg)
    OmegaConf.save(cfg, out / "deploy.yaml")
    return out


def test_the_clocks2_export_passes_its_pinned_check():
    m = _stage_module()
    _need_exports(m)
    info = m.export_check(UPSTREAM, UPSTREAM / m.CLOCKS2_EXPORT, CLOCKS2_SHA256)
    assert info["policy_sha256"] == CLOCKS2_SHA256 == info["pinned_sha256"]
    assert info["deploy_keys_differing"] == FREE_KEYS and info["num_observations"] == 77 == info["onnx_input_width"]
    assert info["controller"] == "GaitClockRlController" and info["gait_clock"] == {"period_s": .8, "phase_offset": 0.}
    assert info["stage_law"] == m.turnboth_law_constants()
    with pytest.raises(RuntimeError, match="sha256"):
        m.export_check(UPSTREAM, UPSTREAM / m.CLOCKS2_EXPORT, "0" * 64)


def test_a_plain_75_observation_export_passes(tmp_path):
    m = _stage_module()
    _need_exports(m)
    info = m.export_check(UPSTREAM, _copy(m, tmp_path, "plain", source="shipped"))
    assert info["controller"] == "RlController" and info["gait_clock"] is None and info["num_observations"] == 75
    assert info["deploy_keys_differing"] == ["policy_checkpoint_path"] and info["pinned_sha256"] is None


def _edits():
    def kp(cfg):
        cfg.joint_kp[0] = 11.0

    def scale(cfg):
        cfg.action_scale = 0.3

    def history(cfg):
        cfg.history_length = 1

    def no_clock(cfg):
        del cfg["gait_clock"]

    def period(cfg):
        cfg.gait_clock.period_s = 0.5

    def width(cfg):
        cfg.num_observations = 75

    return {"joint_kp": (kp, "differs from the shipped"), "action_scale": (scale, "differs from the shipped"),
            "history": (history, "history"), "no_clock_block": (no_clock, "observations"),
            "clock_period": (period, "frozen period"), "width_75_with_clock": (width, "observations")}


@pytest.mark.parametrize("case", sorted(_edits()))
def test_the_export_check_refuses_a_deploy_yaml_that_would_not_be_equivalent(tmp_path, case):
    m = _stage_module()
    _need_exports(m)
    edit, message = _edits()[case]
    with pytest.raises(RuntimeError, match=message):
        m.export_check(UPSTREAM, _copy(m, tmp_path, case, edit=edit))


def test_the_export_check_refuses_a_foreign_policy_path_a_wrong_onnx_width_and_missing_files(tmp_path):
    m = _stage_module()
    _need_exports(m)
    with pytest.raises(RuntimeError, match="policy_checkpoint_path"):
        m.export_check(UPSTREAM, _copy(m, tmp_path, "foreign", pointer=UPSTREAM / m.CLOCKS2_EXPORT / "policy.onnx"))
    with pytest.raises(RuntimeError, match="ONNX input 75"):
        m.export_check(UPSTREAM, _copy(m, tmp_path, "narrow", policy_from="shipped"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="not an export"):
        m.export_check(UPSTREAM, empty)


def test_load_export_gait_checks_once_and_feeds_the_stage(tmp_path):
    m = _stage_module()
    _need_exports(m)
    from omegaconf import OmegaConf
    env = argparse.Namespace(upstream=UPSTREAM, cfg=OmegaConf.load(UPSTREAM / m.SHIPPED_EXPORT / "deploy.yaml"))
    policy, cfg, info = m.load_export_gait(env, "clocks2")
    assert m.load_export_gait(env, "clocks2")[0] is policy and info["stage_gait"] == "clocks2"
    assert int(cfg.num_observations) == 77 and policy.session.get_inputs()[0].shape[-1] == 77
    m.select_export_gait(_copy(m, tmp_path, "sel"), None)
    assert m.load_export_gait(env, "export")[2]["stage_gait"] == "export"


# ---- default paths: shipped, turnboth and m3 against the committed M3 code ------------------------------------------

def _git_module(tmp_path, path, sha256, name):
    _stage_module()
    try:
        text = subprocess.run(["git", "-C", str(_REPO), "show", f"{M3_COMMIT}:{path}"], capture_output=True,
                              check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history not available")
    assert hashlib.sha256(text).hexdigest() == sha256
    target = tmp_path / f"{name}.py"
    target.write_bytes(text)
    spec = importlib.util.spec_from_file_location(name, target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ns_env(layout, xy, yaw):
    env = StubEnv(layout, xy, yaw)
    env.controller = argparse.Namespace(policy="shipped-policy", prev_actions=np.full(22, .5, dtype=np.float32))
    return env


def _drive(module, kwargs, layout_index, door, open_at, steps=260):
    layout = generate("validation", layout_index)   # a kinematic stub drive, no episode: the M2/M3 tests' own layouts
    plate = np.asarray(layout.plate(door, layout.correct_sides[door]))
    env = _ns_env(layout, layout.xy(layout.route[layout.door_indices[door] - 1]), 0.)
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
                      env.controller.prev_actions.copy().tolist(), env.controller.policy))
        env.step(command)
    return trace, stage


@pytest.mark.parametrize("kwargs", [{}, {"cross_clear_m": .35}, {"align_yaw": True}, {"cross_kick": True},
                                    {"press_hold": True, "wait_open_s": 2.0}, {"stage_lateral_m": .35},
                                    {"stage_gait": "shipped"},
                                    {"stage_gait": "turnboth", "turnboth_policy": "turnboth-policy"},
                                    {"stage_gait": "turnboth", "turnboth_policy": "p", "cross_clear_m": .30},
                                    {"stage_gait": "m3"}, {"stage_gait": "m3", "cross_kick": True}])
@pytest.mark.parametrize("layout_index, door, open_at", [(0, 0, None), (4, 1, 40), (13, 1, None)])
def test_shipped_turnboth_and_m3_paths_match_the_committed_m3_stage(tmp_path, kwargs, layout_index, door, open_at):
    reference = _git_module(tmp_path, "scripts/mission7_plate_stage.py", M3_STAGE_SHA256, "m7_stage_m3_commit")
    m = _stage_module()
    expected, ref_stage = _drive(reference, kwargs, layout_index, door, open_at)
    actual, new_stage = _drive(m, kwargs, layout_index, door, open_at)
    assert actual == expected and new_stage.history == ref_stage.history and new_stage.done == ref_stage.done
    for attr in ("gait_events", "m3_events", "kick_events", "align_events"):
        assert getattr(new_stage, attr, None) == getattr(ref_stage, attr, None), attr
    assert not hasattr(new_stage, "export_policy") and not hasattr(new_stage, "shipped_controller") or \
        kwargs.get("stage_gait") == "turnboth"


@pytest.mark.parametrize("move", [True, False])
@pytest.mark.parametrize("layout_index, door, heading", [(7, 1, 90), (8, 1, 0), (11, 1, 180)])
def test_the_m3_bench_path_matches_the_committed_m3_stage(tmp_path, move, layout_index, door, heading):
    reference = _git_module(tmp_path, "scripts/mission7_plate_stage.py", M3_STAGE_SHA256, "m7_stage_m3_bench")
    m = _stage_module()
    out = []
    for module in (reference, m):
        layout = generate("train", layout_index)
        g = v2.crossing_geometry(layout, door, "round", heading, "standstill")
        env = _ns_env(layout, g["pre_point"], g["entry_yaw_rad"] + 1e-3)
        stage = module.PlateStage(env, stage_gait="m3")
        stage._start(door, g["side"], env.runner.d.time)
        out.append((_run(stage, env, move=move), stage.history, stage.m3_events, stage.gait_events,
                    env.controller.prev_actions.tolist()))
    assert out[0] == out[1]


def test_the_m3_and_turnboth_texts_and_the_bench_v2_records_match_the_committed_code(tmp_path):
    ref_stage = _git_module(tmp_path, "scripts/mission7_plate_stage.py", M3_STAGE_SHA256, "m7_stage_m3_texts")
    ref_v2 = _git_module(tmp_path, "scripts/mission7_plate_bench_v2.py", M3_BENCH_V2_SHA256, "m7_bench_v2_m3_commit")
    m = _stage_module()
    for gait in ("turnboth", "m3"):
        assert m.stage_gait_description(gait) == ref_stage.stage_gait_description(gait)
        assert v2.labels(gait) == ref_v2.labels(gait) and v2.chain_for(gait) == ref_v2.CHAIN_V2
        assert v2.source_files(_REPO, gait) == ref_v2.source_files(_REPO) == v2.source_files(_REPO)
    assert m.m3_constants() == ref_stage.m3_constants() and v2.RULE_V2 == ref_v2.RULE_V2 == FROZEN_RULE
    assert (v2.DROPPED, v2.STANDSTILL_SPAWN_WALL_OVERLAP, v2.STAGE_GAITS_V2) == (
        ref_v2.DROPPED, ref_v2.STANDSTILL_SPAWN_WALL_OVERLAP, ref_v2.STAGE_GAITS_V2)
    import inspect
    for name in ("run_crossing", "bench_v2_verdict", "crossing_geometry", "declared_crossings", "parse_smoke_specs"):
        assert inspect.getsource(getattr(v2, name)) == inspect.getsource(getattr(ref_v2, name)), name
    m3_result = {"layout": 8, "door": 1, "stage_gait": "m3", "events": [], "stage_history": [], "geometry": {},
                 "plate_activation": {"target_pressed_during_stage": False}, "wall_contacts": {"stage_samples": 0},
                 "spawn": {"contacts": []}, "m3_events": [{"event": "stall_recover"}], "gait_events": []}
    assert v2.compact_result(m3_result) == ref_v2.compact_result(m3_result)
    clocks2 = v2.source_files(_REPO, "clocks2")
    assert set(clocks2) - set(v2.source_files(_REPO)) == {"src/bhl_robust/eval/gait_clock.py", v2.V2_CLOCKS2_LAUNCHER}


def test_the_m3_launcher_and_the_v1_bench_are_unchanged():
    recorded = json.loads((M3_DIR / "provenance.json").read_text())["sources_sha256"]
    for name in ("slurm/repo20260923/cpu_m7_plate_bench_v2.sbatch", "scripts/mission7_plate_bench.py",
                 "slurm/repo20260923/cpu_m7_plate_bench.sbatch"):
        assert hashlib.sha256((_REPO / name).read_bytes()).hexdigest() == recorded[name], name
    out = subprocess.run(["git", "-C", str(_REPO), "diff", "--quiet", "HEAD", "--",
                          "slurm/repo20260923/cpu_m7_plate_bench_v2.sbatch"])
    assert out.returncode == 0


# gait_clock.py as R1's qualification and the clocks2 smoke (21516422) used it; other workstreams may append to the
# file (turning-hold did on 2026-10-02 and then restored it), so the controller code this override builds is compared
# function by function. Factory dispatch may gain unrelated workstreams; its
# plain and mission7 clock paths are compared by controller inputs/outputs.
GAIT_CLOCK_COMMIT = ("563bb9a", "eedd6279fd74e9706e8015e4c048900f08570d26f54f9905af3346e54d8f6b98")


def test_the_clock_controller_the_override_builds_is_the_committed_one(tmp_path):
    pytest.importorskip("berkeley_humanoid_lite_lowlevel")
    import inspect
    from bhl_robust.eval import gait_clock as gc
    try:
        text = subprocess.run(["git", "-C", str(_REPO), "show", f"{GAIT_CLOCK_COMMIT[0]}:src/bhl_robust/eval/gait_clock.py"],
                              capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history not available")
    assert hashlib.sha256(text).hexdigest() == GAIT_CLOCK_COMMIT[1]
    (tmp_path / "gait_clock_ref.py").write_bytes(text)
    ref = _load(tmp_path / "gait_clock_ref.py", "gait_clock_ref")
    for name in ("clock_phase", "clock_features", "base_obs_width", "has_clock",
                 "_clock_controller_class"):
        assert inspect.getsource(getattr(gc, name)) == inspect.getsource(getattr(ref, name)), name
    assert (gc.CLOCK_KEY, gc.EXPECTED_PERIOD_S, gc.EXPECTED_PHASE_OFFSET, gc.N_BASE_TERMS_FIXED) == (
        ref.CLOCK_KEY, ref.EXPECTED_PERIOD_S, ref.EXPECTED_PHASE_OFFSET, ref.N_BASE_TERMS_FIXED)
    # A waiter_wbc branch is allowed to coexist, but must not redirect the
    # frozen plain or mission7 clock configurations or alter their behavior.
    from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
    for clock, width in ((False, 75), (True, 77)):
        actual = gc.make_controller(_cfg(width, clock))
        expected = ref.make_controller(_cfg(width, clock))
        if clock:
            assert type(actual) is gc._clock_controller_class()
        else:
            assert type(actual) is RlController
        assert type(actual).__name__ == type(expected).__name__
        actual.policy, expected.policy = _Policy(width, 17), _Policy(width, 17)
        for obs in _robot_obs(24, seed=29):
            np.testing.assert_array_equal(actual.update(obs), expected.update(obs))
            np.testing.assert_array_equal(actual.policy_observations, expected.policy_observations)
            np.testing.assert_array_equal(actual.prev_actions, expected.prev_actions)
        actual.policy_observations[:] = expected.policy_observations[:] = 0
        obs = _robot_obs(1, seed=30)[0]
        np.testing.assert_array_equal(actual.update(obs), expected.update(obs))
        if clock:
            assert actual.clock_step == expected.clock_step == 1


def test_the_pinned_default_call_lines_stay():
    stage, probe = STAGE.read_text(), PROBE.read_text()
    for text in (stage, probe):
        assert 'parser.add_argument("--stage-gait", choices=STAGE_GAITS, default="shipped"' in text
        assert "align_yaw=align_yaw, stage_gait=stage_gait)" in text
        assert "stage_gait_action.choices = STAGE_GAITS + EXPORT_STAGE_GAITS" in text
    assert "stage_gait=getattr(args, 'stage_gait', 'shipped')))" in stage
    assert 'stage_gait=getattr(args, "stage_gait", "shipped"))' in probe


# ---- bench v2: the gait option, its labels, chain and provenance -------------------------------------------------

def test_bench_v2_labels_chain_and_disclosure_name_clock_s2():
    labels = v2.labels("clocks2")
    assert "arms-turngait-clock-s2" in labels["gaits"] and "TurnBoth" not in labels["gaits"]
    assert labels["stage"] == "SCRIPTED (PlateStage, stage_gait=clocks2)" and "ORACLE" in labels["layout_and_plate_pose"]
    assert "--stage-gait clocks2" in v2.chain_for("clocks2") and "m7-clocks2" in v2.chain_for("clocks2")
    assert "m3" not in v2.chain_for("clocks2") and "no replay or route release arm" in v2.chain_for("export")
    for clause in ("qualified at +-0.6 rad/s", "untested at the interface's 0.40 rad/s", "(pi - 0.15)/0.40 = 7.48 s",
                   "0.65 m / 0.30 m/s = 2.17 s", "10.05-10.25 s (about 10.1-10.3 s)", ">= 7/8"):
        assert clause in v2.DISCLOSURE_CLOCKS2, clause
    assert abs((np.pi - .15) / .40 - 7.48) < .005 and abs(.65 / .30 - 2.17) < .005
    assert 10.0 < (np.pi - .15) / .40 + .40 + .65 / .30 < (np.pi - .15) / .40 + .60 + .65 / .30 < 10.3


def test_bench_v2_provenance_for_clocks2_and_m3(tmp_path):
    m = _stage_module()
    _need_exports(m)
    prov = v2.provenance(_REPO, "smoke", "clocks2", [(251, 1)])
    assert prov["weights_sha256"]["clocks2"] == CLOCKS2_SHA256 and prov["stage_constants"]["policy_sha256"] == CLOCKS2_SHA256
    assert prov["chain"] == v2.CHAIN_V2_CLOCKS2 and prov["disclosure_stage_gait"] == v2.DISCLOSURE_CLOCKS2
    assert {"src/bhl_robust/eval/gait_clock.py", v2.V2_CLOCKS2_LAUNCHER} <= set(prov["sources_sha256"])
    assert prov["rule"] == FROZEN_RULE and prov["labels"] == v2.labels("clocks2")
    m3 = v2.provenance(_REPO, "smoke", "m3", [(251, 1)])
    assert m3["chain"] == v2.CHAIN_V2 and m3["stage_constants"] == m.m3_constants()
    assert "disclosure_stage_gait" not in m3 and "stage_gait_export" not in m3 and set(m3["weights_sha256"]) == {"shipped"}
    assert set(m3["sources_sha256"]) == set(v2.source_files(_REPO)) - {"no such"}


def test_bench_v2_cli_accepts_clocks2_and_guards_the_export_flags(tmp_path, capsys):
    m = _stage_module()
    _need_exports(m)
    base = ["--repo", str(_REPO), "--out", str(tmp_path / "never"), "--mode", "smoke", "--crossings", "251:1"]
    assert v2.main([*base, "--stage-gait", "clocks2", "--preflight"]) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["stage_gait"] == "clocks2" and line["stage_gait_check"]["policy_sha256"] == CLOCKS2_SHA256
    export = str(UPSTREAM / m.CLOCKS2_EXPORT)
    assert v2.main([*base, "--stage-gait", "export", "--stage-gait-export", export, "--stage-gait-export-sha256",
                    CLOCKS2_SHA256, "--preflight"]) == 0
    for argv in (["--stage-gait", "export"], ["--stage-gait", "clocks2", "--stage-gait-export", export],
                 ["--stage-gait", "m3", "--stage-gait-export-sha256", CLOCKS2_SHA256],
                 ["--stage-gait", "export", "--stage-gait-export", export, "--stage-gait-export-sha256", "0" * 64]):
        with pytest.raises((SystemExit, RuntimeError)) as info:
            v2.main([*base, *argv, "--preflight"])
        if isinstance(info.value, SystemExit):
            assert info.value.code == 2
    assert not (tmp_path / "never").exists()


def _results(misses=(), falls=()):
    out = []
    for c in v2.declared_crossings():
        key = (c["layout"], c["door"])
        out.append({**c, "complete": True, "clear": key not in misses, "fell": key in falls})
    return out


def _by_heading():
    groups = {}
    for c in v2.declared_crossings():
        groups.setdefault(c["heading_deg"], []).append((c["layout"], c["door"]))
    return groups


def test_the_frozen_rule_at_its_boundaries_for_the_clocks2_run():
    """bench_v2_verdict is the clocks2 run's verdict, unchanged: N = 41, >= 39/41, 0 falls, >= 11/10/9/7."""
    groups = _by_heading()
    assert {h: len(v) for h, v in groups.items()} == {0: 12, 90: 11, -90: 10, 180: 8}
    passed = v2.bench_v2_verdict(_results())
    assert passed["verdict"] == "PASS" and (passed["N"], passed["required_clears"]) == (41, 39)
    assert {h: v["required"] for h, v in passed["per_heading"].items()} == {"0": 11, "90": 10, "-90": 9, "180": 7}
    two_headings = (groups[0][0], groups[180][0])                 # 39/41, every heading at N_h - 1 at most
    assert v2.bench_v2_verdict(_results(misses=two_headings))["verdict"] == "PASS"
    three = (groups[0][0], groups[90][0], groups[-90][0])          # 38/41
    assert v2.bench_v2_verdict(_results(misses=three))["verdict"] == "FAIL"
    one_heading = tuple(groups[180][:2])                           # 39/41 but 180 deg at 6/8 < 7
    assert v2.bench_v2_verdict(_results(misses=one_heading))["verdict"] == "FAIL"
    assert v2.bench_v2_verdict(_results(falls=(groups[90][0],)))["verdict"] == "FAIL"   # cleared, but a fall
    missing = _results()[1:]
    assert v2.bench_v2_verdict(missing)["verdict"] == "INCOMPLETE"
    incomplete = _results()
    incomplete[0]["complete"] = False
    assert v2.bench_v2_verdict(incomplete)["verdict"] == "INCOMPLETE"


def test_compact_result_reports_the_controller_swaps():
    result = {"layout": 8, "door": 1, "stage_gait": "clocks2", "events": [], "stage_history": [], "geometry": {},
              "plate_activation": {"target_pressed_during_stage": False}, "wall_contacts": {"stage_samples": 0},
              "spawn": {"contacts": []}, "m3_events": None,
              "gait_events": [{"event": "swap_to_clocks2", "time_s": 3.0, "controller": "GaitClockRlController",
                               "obs_width": 77, "clock_step": 0},
                              {"event": "swap_to_shipped", "time_s": 9.0, "controller": "RlController", "obs_width": 75,
                               "clock_step": None, "stage_controller_clock_step": 150}]}
    line = v2.compact_result(result)
    assert line["watchdog_recoveries"] == 0 and line["stage_gait"] == "clocks2"
    assert [s["event"] for s in line["controller_swaps"]] == ["swap_to_clocks2", "swap_to_shipped"]
    assert line["controller_swaps"][1]["stage_controller_clock_step"] == 150
    json.dumps(line)


# ---- plumbing: the stage and probe CLIs, both submitters, the follow-up, the release script -------------------------

@pytest.mark.parametrize("script", ["stage", "probe"])
def test_clis_accept_clocks2_and_export_in_preflight(tmp_path, script):
    m = _stage_module()
    _need_exports(m)
    upstream_python = _REPO / "external/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, (_REPO / "src", _REPO / "scripts", upstream_python)))}
    if script == "stage":
        args = [str(STAGE), "--repo", str(_REPO), "--campaign", REPLAY_SOURCE, "--baseline", REPLAY_BASELINE,
                "--out", str(tmp_path / "never"), "--preflight"]
    else:
        args = [str(PROBE), "--repo", str(_REPO), "--out", str(_REPO / "results/never-created-clocks2-probe"),
                "--stage", "doors", "--indices", "0", "--preflight"]

    def run(*extra):
        return subprocess.run([sys.executable, *args, *extra], cwd=_REPO, capture_output=True, text=True, timeout=300,
                              env=env)
    out = run("--stage-gait", "clocks2")
    assert out.returncode == 0, out.stderr
    line = json.loads(out.stdout.strip().splitlines()[-1])
    assert line["status"] == "PREFLIGHT_OK" and line["stage_gait"] == "clocks2"
    check = line["stage_gait_check"]
    assert check["policy_sha256"] == CLOCKS2_SHA256 and check["deploy_keys_differing"] == FREE_KEYS
    out = run("--stage-gait", "export", "--stage-gait-export", str(UPSTREAM / m.CLOCKS2_EXPORT))
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1])["stage_gait_check"]["pinned_sha256"] is None
    for extra in (["--stage-gait", "export"], ["--stage-gait", "clocks2", "--stage-gait-export", "x"],
                  ["--stage-gait", "clocks2", "--align-yaw"], ["--stage-gait-export-sha256", "ab"]):
        bad = run(*extra)
        assert bad.returncode == 2, (extra, bad.stderr)
    assert not (tmp_path / "never").exists() and not (_REPO / "results/never-created-clocks2-probe").exists()


def _plan(script, *args, code=0):
    out = subprocess.run([sys.executable, str(script), *args], cwd=_REPO, capture_output=True, text=True, timeout=120)
    assert out.returncode == code, out.stderr
    return json.loads(out.stdout) if code == 0 else out.stderr


def test_replay_submitter_forwards_clocks2_and_pins_an_export():
    m = _stage_module()
    _need_exports(m)
    if not ((_REPO / REPLAY_SOURCE / "fullroute/legacy-doors.json").is_file()
            and (_REPO / REPLAY_BASELINE / "result.json").is_file()):
        pytest.skip("untracked replay inputs not on disk")
    common = ["--out-campaign", "results/mission7-campaign-20260923", "--output-name",
              "replay-gate-test-clocks2-never-created"]
    assert _plan(STAGE_SUBMITTER, *common, "--stage-gait", "clocks2")["probe_args"] == ["--stage-gait=clocks2"]
    export = str(UPSTREAM / m.CLOCKS2_EXPORT)
    plan = _plan(STAGE_SUBMITTER, *common, "--stage-gait", "export", "--stage-gait-export", export)
    assert plan["probe_args"] == ["--stage-gait=export", f"--stage-gait-export={Path(export).resolve()}",
                                  f"--stage-gait-export-sha256={CLOCKS2_SHA256}"]
    assert _plan(STAGE_SUBMITTER, *common, "--stage-gait", "m3")["probe_args"] == ["--stage-gait=m3"]   # unchanged
    for extra in (["--stage-gait", "clocks2", "--align-yaw"], ["--stage-gait", "clocks2", "--stage-gait-export", export],
                  ["--stage-gait", "export"], ["--stage-gait", "m3", "--stage-gait-export", export],
                  ["--stage-gait", "export", "--stage-gait-export", export, "--stage-gait-export-sha256", "0" * 64]):
        _plan(STAGE_SUBMITTER, *common, *extra, code=2)
    assert not (_REPO / "results/mission7-campaign-20260923/replay-gate-test-clocks2-never-created").exists()


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_both_submitters_snapshot_gait_clock_for_the_export_gaits_only():
    for path, name in ((STAGE_SUBMITTER, "m7_stage_submitter_c2"), (ROUTE_SUBMITTER, "m7_route_submitter_c2")):
        module = _load(path, name)
        assert module.GAIT_CLOCK_MODULE == Path("src/bhl_robust/eval/gait_clock.py")
        assert (_REPO / module.GAIT_CLOCK_MODULE).is_file() and module.EXPORT_STAGE_GAITS == ("clocks2", "export")
        text = path.read_text()
        assert ("    if args.stage_gait in EXPORT_STAGE_GAITS:   # --- m7-clocks2 --- see GAIT_CLOCK_MODULE\n"
                "        files.append(ROOT / GAIT_CLOCK_MODULE)\n") in text
    # the default snapshot set the M2 test reads (double-quoted literals) is unchanged
    explicit = set(re.findall(r'"((?:src|slurm)/[^"]+\.(?:py|sbatch))"', STAGE_SUBMITTER.read_text()))
    assert explicit == set(v1.BENCH_SOURCES[:-1]) | {"slurm/mission7_plate_stage.sbatch"}


def test_route_submitter_forwards_clocks2_and_export():
    m = _stage_module()
    _need_exports(m)
    common = ["--campaign", "results/mission7-campaign-20260923", "--stage", "doors", "--indices", "0,1",
              "--output-name", "route-gate-test-clocks2-never-created", "--handoff", "early", "--stage-activate",
              "--rejoin-advance", "--exit-ramp-center", "--exit-ramp", "1.2", "--stage-wait-open", "0.0"]
    plan = _plan(ROUTE_SUBMITTER, *common, "--stage-gait", "clocks2")
    assert plan["stage_gait"] == "clocks2" and plan["probe_args"][-1] == "--stage-gait=clocks2"
    export = str(UPSTREAM / m.CLOCKS2_EXPORT)
    plan = _plan(ROUTE_SUBMITTER, *common, "--stage-gait", "export", "--stage-gait-export", export,
                 "--stage-gait-export-sha256", CLOCKS2_SHA256)
    assert plan["probe_args"][-3:] == ["--stage-gait=export", f"--stage-gait-export={Path(export).resolve()}",
                                       f"--stage-gait-export-sha256={CLOCKS2_SHA256}"]
    assert set(_plan(ROUTE_SUBMITTER, *common)) == {"planned_output", "stage", "indices", "node", "constraint"}
    for extra in (["--stage-gait", "clocks2", "--align-yaw"], ["--stage-gait", "export"],
                  ["--stage-gait", "clocks2", "--stage-gait-export", export],
                  ["--stage-gait", "export", "--stage-gait-export", export, "--stage-gait-export-sha256", "0" * 64]):
        _plan(ROUTE_SUBMITTER, *common, *extra, code=2)
    module = _load(ROUTE_SUBMITTER, "m7_route_submitter_c2_args")
    args = argparse.Namespace(stage="doors", indices="0", handoff="switch", rejoin_fix="none", rejoin_diagnostic=False,
                              chain_trace=False, stage_lateral=None, stage_activate=False, stage_press_hold=False,
                              stage_wait_open=None, pre_point=None, cross_clear=None, stall_min_s=None, settle_s=None,
                              cross_kick=False, rejoin_advance=False, exit_ramp_center=False, align_yaw=False,
                              exit_ramp=0., allow_inactive_intervention=False, stage_gait="clocks2")
    assert module._probe_args(args)[-1] == "--stage-gait=clocks2"          # a Namespace without the export attributes
    args.stage_gait = None
    assert module._probe_args(args) == ["--stage", "doors", "--indices", "0", "--handoff", "switch", "--rejoin-fix", "none"]
    assert not (_REPO / "results/mission7-campaign-20260923/route-gate-test-clocks2-never-created").exists()


def _translation_heredoc():
    text = FOLLOWUP.read_text()
    match = re.search(r'^"\$PY" - "\$gate_dir" > "\$flags_file" <<\'PYEOF\'\n(.*?)\nPYEOF$', text, re.S | re.M)
    assert match, "step-3 heredoc not found"
    return match.group(1)


@pytest.mark.parametrize("probe_args, expected", [
    (["--stage-gait=clocks2"], ["--stage-gait", "clocks2", "--stage-wait-open", "0.0"]),
    (["--stage-gait=export", "--stage-gait-export=/x/exported", "--stage-gait-export-sha256=ab"],
     ["--stage-gait", "export", "--stage-gait-export", "/x/exported", "--stage-gait-export-sha256", "ab",
      "--stage-wait-open", "0.0"]),
    (["--stage-gait=m3"], ["--stage-gait", "m3", "--stage-wait-open", "0.0"]),
    (["--stage-gait=turnboth", "--wait-open=2.0"], ["--stage-gait", "turnboth", "--stage-wait-open", "2.0"])])
def test_followup_translates_the_export_gaits(tmp_path, probe_args, expected):
    (tmp_path / "submission.json").write_text(json.dumps({"probe_args": probe_args}))
    script = tmp_path / "translate.py"
    script.write_text(_translation_heredoc())
    out = subprocess.run([sys.executable, str(script), str(tmp_path)], capture_output=True, text=True, check=True)
    assert out.stdout.split("\n")[:-1] == expected


def test_followup_dry_run_releases_a_clocks2_route_gate_from_a_clocks2_pass(tmp_path):
    _stage_module()
    campaign = tmp_path / "campaign"
    gate = campaign / "replay-gate-clocks2-synthetic"
    gate.mkdir(parents=True)
    rows = [{"layout_index": i, "fall": False, "stage_history": [], "stage_gait": "clocks2", "gait_events": []}
            for i in (0, 1, 4, 5, 7, 9, 12, 13, 14, 15)]
    (gate / "result.json").write_text(json.dumps({
        "complete": True, "status": "COMPLETED_INTERVENTION", "episodes": 10, "falls": 0, "upright": 10,
        "exact_replay_gate_passed": True, "stage_gait": "clocks2", "episode_comparison": rows}))
    (gate / "submission.json").write_text(json.dumps({"job_id": "0", "probe_args": ["--stage-gait=clocks2"],
                                                      "source_sha256": {}}))
    route_campaign = "results/mission7-campaign-20260923"
    locks_before = sorted(p.name for p in (_REPO / route_campaign).glob("replay-gate-route-lock-*"))
    env = {**os.environ, "DRY_RUN": "1", "PYTHON": sys.executable, "M7_REPO": str(_REPO),
           "M7_ROUTE_CAMPAIGN": route_campaign}
    out = subprocess.run(["bash", str(FOLLOWUP), str(gate), "clocks2-synthetic", "1", str(campaign)], env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "M7 REPLAY GATE clocks2-synthetic: PASS" in out.stdout and "ROUTE_GATE_READY" in out.stdout
    composition = next(line for line in out.stdout.splitlines() if line.startswith("route gate composition"))
    assert "--stage-gait clocks2" in composition and "--stage-wait-open 0.0" in composition
    plans = [json.loads(block) for block in re.findall(r"^\{\n.*?^\}$", out.stdout, re.S | re.M)]
    assert [p["stage"] for p in plans] == ["doors", "transport"]
    assert all(p["probe_args"][-1] == "--stage-gait=clocks2" and p["indices"] == ",".join(map(str, range(16)))
               for p in plans)
    assert sorted(p.name for p in (_REPO / route_campaign).glob("replay-gate-route-lock-*")) == locks_before
    assert not (_REPO / route_campaign / "route-gate-clocks2-synthetic-doors").exists()


def test_replay_gates_script_has_the_clocks2_arm_gate_and_placement():
    text = REPLAY_GATES.read_text()
    assert re.search(r'^\s*"m7-clocks2\s+--stage-gait clocks2"\s*$', text, re.M)
    assert re.search(r'^\s*"m3-shipped-step\s+--stage-gait m3"\s*$', text, re.M)
    subprocess.run(["bash", "-n", str(REPLAY_GATES)], check=True)
    assert "CLOCKS2_BENCH_VERDICT=${M7_CLOCKS2_BENCH_VERDICT:-$CAMPAIGN/clocks2-plate-bench-v2/verdict.json}" in text
    assert f"CLOCKS2_POLICY_SHA256={CLOCKS2_SHA256}" in text
    calls = [mt.start() for mt in re.finditer(
        r'^\s*clocks2_bench_gate "\$CLOCKS2_BENCH_VERDICT" "\$CLOCKS2_POLICY_SHA256" "\$CLOCKS2_POLICY" '
        r'"\$\{clocks2_gate_files\[@\]\}" \|\| exit 3$', text, re.M)]
    assert len(calls) == 1
    assert text.index('echo "DRY RUN: nothing submitted.') < calls[0] < text.index("real_sbatch=$(command -v sbatch)")
    assert text.rindex('if [ "$c2_selected" = 1 ]; then', 0, calls[0]) > text.index(
        'm3_bench_gate "$M3_BENCH_VERDICT" "${m2_gate_files[@]}" || exit 3')
    assert text.count("m2_selected=1") == 1 and text.count("m3_selected=1") == 1 and text.count("c2_selected=1") == 1
    assert 'clocks2_gate_files=("${m2_gate_files[@]}" src/bhl_robust/eval/gait_clock.py)' in text
    assert 'if [ "$c2_selected" = 1 ] && ! git diff --quiet -- src/bhl_robust/eval/gait_clock.py; then' in text
    policy = re.search(r"^CLOCKS2_POLICY=(\S+)$", text, re.M).group(1)
    m = _stage_module()
    assert policy == f"external/Berkeley-Humanoid-Lite/{m.CLOCKS2_EXPORT}/policy.onnx"
    replay = re.search(r"^snapshot_files=\((.*?)\)\n", text, re.S | re.M)
    assert replay and "gait_clock" not in replay.group(1)               # the default snapshot set is unchanged


def _clocks2_gate_function():
    match = re.search(r"^clocks2_bench_gate\(\) \{\n.*?^\}\n", REPLAY_GATES.read_text(), re.S | re.M)
    assert match
    return match.group(0)


def _bench_dir(tmp_path, verdict=None, provenance=None, record=None, weights=None):
    files = {"scripts/mission7_plate_stage.py": "stage\n", "scripts/mission7_gates.py": "gates\n",
             "src/bhl_robust/mission/env.py": "env\n", "src/bhl_robust/eval/gait_clock.py": "clock\n"}
    for name, body in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(body)
    (tmp_path / "policy.onnx").write_text("clock-s2 weights\n")
    pinned = hashlib.sha256(b"clock-s2 weights\n").hexdigest()
    out = tmp_path / "results/clocks2-plate-bench-v2"
    out.mkdir(parents=True)
    verdict = {"bench": "v2", "mode": "scored", "stage_gait": "clocks2", "verdict": "PASS"} if verdict is None else verdict
    (out / "verdict.json").write_text(json.dumps(verdict))
    recorded = {name: hashlib.sha256(body.encode()).hexdigest() for name, body in files.items()}
    prov = {"bench": "v2", "mode": "scored", "stage_gait": "clocks2",
            "sources_sha256": recorded if record is None else record,
            "weights_sha256": {"shipped": "s", "clocks2": pinned} if weights is None else weights}
    (out / "provenance.json").write_text(json.dumps(prov if provenance is None else {**prov, **provenance}))
    return out, pinned, sorted(files)


def _run_gate(tmp_path, verdict_path, pinned, files, policy="policy.onnx"):
    script = tmp_path / "gate.sh"
    script.write_text("set -euo pipefail\n" + _clocks2_gate_function() + 'clocks2_bench_gate "$@"\n')
    return subprocess.run(["bash", str(script), str(verdict_path), pinned, policy, *files], cwd=tmp_path,
                          capture_output=True, text=True, timeout=60, env={**os.environ, "PY": sys.executable})


def test_clocks2_gate_passes_a_scored_v2_clocks2_pass(tmp_path):
    out, pinned, files = _bench_dir(tmp_path)
    run = _run_gate(tmp_path, out / "verdict.json", pinned, files)
    assert run.returncode == 0, run.stderr
    assert "scored bench-v2 clocks2 PASS" in run.stdout and "4 snapshot sources identical" in run.stdout


@pytest.mark.parametrize("verdict, provenance", [
    ({"bench": "v2", "mode": "scored", "stage_gait": "clocks2", "verdict": "FAIL"}, None),
    ({"bench": "v2", "mode": "scored", "stage_gait": "clocks2", "verdict": "INCOMPLETE"}, None),
    ({"bench": "v2", "mode": "smoke", "stage_gait": "clocks2", "verdict": "PASS"}, None),
    ({"bench": "v2", "mode": "scored", "stage_gait": "m3", "verdict": "PASS"}, None),
    ({"bench": "v2", "mode": "scored", "stage_gait": "export", "verdict": "PASS"}, None),
    ({"mode": "scored", "verdict": "PASS"}, None),
    ({"bench": "v2", "mode": "scored", "stage_gait": "clocks2", "verdict": "PASS"}, {"stage_gait": "m3"}),
    ({"bench": "v2", "mode": "scored", "stage_gait": "clocks2", "verdict": "PASS"}, {"mode": "smoke"})])
def test_clocks2_gate_refuses_anything_but_a_scored_v2_clocks2_pass(tmp_path, verdict, provenance):
    out, pinned, files = _bench_dir(tmp_path, verdict=verdict, provenance=provenance)
    run = _run_gate(tmp_path, out / "verdict.json", pinned, files)
    assert run.returncode == 3 and "RELEASE_BLOCKED m7-clocks2" in run.stderr


def test_clocks2_gate_refuses_other_weights_changed_sources_and_a_missing_clock_module(tmp_path):
    out, pinned, files = _bench_dir(tmp_path, weights={"shipped": "s", "clocks2": "0" * 64})
    assert _run_gate(tmp_path, out / "verdict.json", pinned, files).returncode == 3          # bench ran other weights
    out, pinned, files = _bench_dir(tmp_path / "b")
    (tmp_path / "b/policy.onnx").write_text("retrained\n")
    run = _run_gate(tmp_path / "b", out / "verdict.json", pinned, files)
    assert run.returncode == 3 and "pinned sha256" in run.stderr                              # export changed on disk
    out, pinned, files = _bench_dir(tmp_path / "c")
    (tmp_path / "c/src/bhl_robust/eval/gait_clock.py").write_text("clock, edited after the bench\n")
    run = _run_gate(tmp_path / "c", out / "verdict.json", pinned, files)
    assert run.returncode == 3 and "gait_clock.py" in run.stderr
    out, pinned, files = _bench_dir(tmp_path / "d")
    run = _run_gate(tmp_path / "d", out / "verdict.json", pinned,
                    [f for f in files if not f.endswith("gait_clock.py")])
    assert run.returncode == 3 and "gait_clock.py" in run.stderr
    assert _run_gate(tmp_path / "d", out / "verdict.json", pinned, []).returncode == 3
    assert _run_gate(tmp_path / "d", tmp_path / "nowhere/verdict.json", pinned, files).returncode == 3
    out, pinned, files = _bench_dir(tmp_path / "e", record={"scripts/mission7_gates.py": "0" * 64})
    assert _run_gate(tmp_path / "e", out / "verdict.json", pinned, files).returncode == 3


# ---- the launcher -------------------------------------------------------------------------------------------------

def _header():
    return LAUNCHER.read_text().split("set -euo pipefail")[0]


def _flat(text):
    return " ".join(line.lstrip("#").strip() for line in text.splitlines())


def test_launcher_header_carries_the_frozen_rule_chain_labels_budget_and_disclosures():
    flat = _flat(_header())
    for clause in (
            FROZEN_RULE,
            "PREDECLARED RULE (frozen before any episode; bench v2's rule verbatim, unchanged; the chain as M3's",
            "Then the unchanged exact ten-fall replay once with --stage-gait clocks2, through the existing replay-gate "
            "machinery (complete, 10 episodes, 10 upright, 0 falls; release arm m7-clocks2).",
            "Then the route gate as coded by M1: Doors and Transport each >= 16/16 successes on validation layouts 0-15 "
            "(32 episodes).",
            "Here N = 41 (0 deg: 12, +90: 11, -90: 10, 180: 8), so PASS needs >= 39/41 clears, 0 falls, and >= 11, 10, 9, "
            "7 clears at 0, +90, -90, 180 deg.",
            "LEARNED gaits: the shipped arms-dr1.0-s0 gait outside the stage + the LEARNED clock-s2 stage gait",
            "SCRIPTED stage", "ORACLE layout and plate pose", "never from an exit code",
            "41 bench-v2 crossings (this job) + 10 exact-replay episodes + 32 route-gate episodes = 83 more Mission 7 "
            "episodes", "NOT part of the line",
            "clock-s2 was qualified at +-0.6 rad/s and is UNTESTED at 0.40 rad/s.",
            "(pi - 0.15)/0.40 = 7.48 s (about 7.5 s) of turning + the settle (0.40 or 0.60 s as polled) + 0.65 m / "
            "0.30 m/s = 2.17 s (about 2.2 s) of crossing = 10.05-10.25 s (about 10.1-10.3 s)",
            "the 180-deg clause (>= 7/8) is likely to fail on timing alone unless the gait turns faster than commanded",
            "swaps the CONTROLLER, not only the policy", "gait-clock phase starts at 0, as at an Isaac episode start",
            "prev_actions := 0", "c1862f1e4429c1e6ac2e4fae1391f3f40ccc353382c8bbf4b88efee4c6c8d8de",
            "M2's turnboth law, unchanged", "until |error| < 0.15 rad", "clip(1.2 x error, +-0.40)",
            "M2's FAIL (8/64) and M3's FAIL on bench v2 (21514946: 16/41, 5 falls) stand",
            "--chain-trace", "src/bhl_robust/eval/gait_clock.py",
            "DROPPED (23 of the 32 walking crossings", "Run: 32 standstill + 9 walking",
            # review fixes (2026-10-02, before any scored episode): coverage, the smoke's status, the route snapshot
            "Coverage (disclosed before any scored episode; M2's coverage note, same pattern)",
            "so a bench PASS does not cover clock-s2 driving the stage's capture-to-pre-point approach.",
            "at capture (0.78 m from the plate)", "can command lateral or backward steps",
            "6.8 s of approach on clock-s2 (18.8 -> 25.6 s, maximum tilt 0.18 rad, no fall)",
            "The exact replay (10 episodes) exercises that phase first, before the route gate.",
            "It is a MACHINERY smoke of the uncommitted working tree (HEAD 1bf3e3a plus uncommitted edits), not of the "
            "code a scored run uses.",
            "So the committed code is smoked again (mode smoke) and the bench runs afterok on it",
            "Route-gate snapshot (a gap inherited from M2/M3; disclosed, not closed; the chain stays M3's)",
            "None of those files is to be edited between this bench and the route-gate submission"):
        assert clause in flat, clause
    assert "edited after it: this paragraph and the replay-CLI bullet above only" not in flat
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def test_launcher_bench_refuses_existing_output_dirty_tree_and_releases_only_m7_clocks2():
    text = LAUNCHER.read_text()
    bench = text[text.index('if [ "$mode" = bench ]; then'):text.index('[ "$mode" = smoke ] || {')]
    assert '[ ! -e "$OUT" ] || { echo "refusing: $OUT exists' in bench
    assert 'git diff --quiet HEAD -- "${sources[@]}"' in bench and "STAGE_GAIT=clocks2" in text
    assert '--mode scored --stage-gait "$STAGE_GAIT" --preflight' in bench and "--mode scored --stage-gait \"$STAGE_GAIT\"\n" in bench
    sources = re.search(r"^  sources=\((.*?)\)\n", text, re.S | re.M)
    assert sources and set(sources.group(1).split()) == ({"scripts/mission7*.py", "src/bhl_robust/mission"}
                                                         | set(v1.BENCH_SOURCES) | {v2.V2_LAUNCHER, v2.V2_CLOCKS2_LAUNCHER,
                                                                                    v2.GAIT_CLOCK_SOURCE})
    expanded = set()
    for item in sources.group(1).split():
        expanded.update({str(p.relative_to(_REPO)) for p in _REPO.glob(item)} if "*" in item else
                        ({str(p.relative_to(_REPO)) for p in (_REPO / item).glob("*.py")} if (_REPO / item).is_dir()
                         else {item}))
    assert expanded == set(v2.source_files(_REPO, "clocks2"))
    assert "OUT=${M7_BENCH_V2_CLOCKS2_OUT:-$CAMPAIGN/clocks2-plate-bench-v2}" in bench


def _release_function():
    match = re.search(r"^release_if_pass\(\) \{\n.*?^\}\n", LAUNCHER.read_text(), re.S | re.M)
    assert match
    return match.group(0)


@pytest.mark.parametrize("verdict, release_rc, released, rc, text", [
    ("PASS", 0, True, 0, "releasing the exact ten-fall replay with --stage-gait clocks2"),
    ("PASS", 3, True, 1, "RELEASE_BLOCKED"),
    ("PASS", 1, True, 1, "M7 BENCH V2 CLOCKS2 RELEASE FAILED (exit 1)"),
    ("FAIL", 0, False, 0, "chain stops"),
    ("INCOMPLETE", 0, False, 0, "nothing released")])
def test_launcher_releases_the_clocks2_replay_only_on_a_pass_read_from_json(tmp_path, verdict, release_rc, released, rc,
                                                                            text):
    out = tmp_path / "bench"
    out.mkdir()
    (out / "verdict.json").write_text(json.dumps({"verdict": verdict}))
    calls = tmp_path / "calls"
    stub = tmp_path / "release.sh"
    stub.write_text(f'#!/bin/bash\necho "$@ M7_CLOCKS2_BENCH_VERDICT=$M7_CLOCKS2_BENCH_VERDICT" >> {calls}\n'
                    f'exit {release_rc}\n')
    script = tmp_path / "run.sh"
    script.write_text("set -euo pipefail\n" + _release_function() + f'release_if_pass "{out}"\n')
    env = {**os.environ, "PY": sys.executable, "RELEASE": str(stub), "DRY_RUN": "0"}
    env.pop("M7_CLOCKS2_BENCH_VERDICT", None)
    run = subprocess.run(["bash", "-c", f'PY="$PY" RELEASE="$RELEASE" DRY_RUN=0; source "{script}"'],
                         env=env, capture_output=True, text=True, timeout=60)
    assert run.returncode == rc, run.stderr
    assert text in run.stdout and calls.exists() == released
    if released:
        assert calls.read_text().strip() == f"--only m7-clocks2 --submit M7_CLOCKS2_BENCH_VERDICT={out}/verdict.json"


def _smoke_limits_function():
    match = re.search(r"^smoke_limits_ok\(\) \{\n.*?^\}\n", LAUNCHER.read_text(), re.S | re.M)
    assert match
    return match.group(0)


@pytest.mark.parametrize("name, job_id, limit, ok", [
    ("m7-plate-bench-v2-clocks2-smoke", "123", "1:00:00", True), ("m7-plate-bench-v2-clocks2-smoke", "123", "59:00", True),
    ("m7-plate-bench-v2-clocks2-smoke", "123", "1:00:01", False),
    ("m7-plate-bench-v2-clocks2-smoke", "123", "1-00:00:00", False),
    ("m7-plate-bench-v2-clocks2-smoke", "123", "UNLIMITED", False), ("m7-plate-bench-v2-clocks2-smoke", "", "1:00:00", False),
    ("m7-plate-bench-v2-clocks2", "123", "1:00:00", False)])
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


def test_launcher_smoke_never_runs_a_scored_layout_and_checks_identity_and_the_swap():
    text = LAUNCHER.read_text()
    smoke = text[text.index('[ "$mode" = smoke ] || {'):]
    assert smoke.index("smoke_limits_ok || exit 2") < smoke.index("SMOKE=${M7_SMOKE_DIR") < smoke.index("mkdir -p")
    assert "--job-name=m7-plate-bench-v2-clocks2-smoke --time=01:00:00" in text
    crossings = re.search(r"^SMOKE_CROSSINGS=(\S+)", smoke, re.M).group(1)
    specs = v2.parse_smoke_specs(crossings)                              # refuses any layout < 250
    assert {s["entry"] for s in specs} == {"standstill", "walking"} and {s["heading_deg"] for s in specs} & {90, 180}
    assert v2.parse_smoke_specs(re.search(r"^IDENTITY_CROSSING=(\S+)", smoke, re.M).group(1))
    v1_crossing = re.search(r"^V1_SMOKE_CROSSING=(\d+):(\d)\s", smoke, re.M)     # v1's own grid labels (-90 walking)
    assert v1_crossing and int(v1_crossing.group(1)) >= v2.SMOKE_MIN_LAYOUT
    assert v1.parse_crossings(f"{v1_crossing.group(1)}:{v1_crossing.group(2)}")
    assert int(re.search(r"^ROUTE_LAYOUT=(\d+)", smoke, re.M).group(1)) >= v2.SMOKE_MIN_LAYOUT
    assert 'split="train", seed=4000 + layout' in smoke
    # the replay CLI (validation layouts are scored) and the probe CLI run in their preflight only
    stage_runs = re.findall(r'"\$PY" scripts/mission7_plate_stage\.py(?:[^\n]*\\\n)*[^\n]*', smoke)
    probe_runs = re.findall(r'"\$PY" scripts/mission7_route_handoff_probe\.py(?:[^\n]*\\\n)*[^\n]*', smoke)
    assert len(stage_runs) == 2 and len(probe_runs) == 1
    assert all("--preflight" in run for run in stage_runs + probe_runs)
    assert 'check("no_replay_or_probe_episode"' in smoke
    for step in ("REFERENCE_COMMIT=e33e44f", f"REFERENCE_STAGE_SHA256={M3_STAGE_SHA256}",
                 f"REFERENCE_BENCH_V2_SHA256={M3_BENCH_V2_SHA256}", f"REFERENCE_PROBE_SHA256={M3_PROBE_SHA256}",
                 'cmp "$SMOKE/identity-$gait-ref.L253-d1.json" "$SMOKE/identity-$gait-new.L253-d1.json"',
                 'cmp "$SMOKE/identity-shipped-ref/crossing.json" "$SMOKE/identity-shipped-new/crossing.json"',
                 'cmp "$SMOKE/identity-v1-ref.L250-d0.json" "$SMOKE/identity-v1-new.L250-d0.json"',
                 'cmp "$SMOKE/identity-route-ref/route.json" "$SMOKE/identity-route-new/route.json"',
                 'for gait in turnboth m3; do', 'route_run "$SMOKE/route-clocks2" clocks2 120 "$PYTHONPATH"',
                 'stage_controller_clock_step") == updates', "clock_features(updates - 1, DT, PERIOD)",
                 'first.get("clock_step") == 0', "sys.exit(0 if not problems else 1)"):
        assert step in text, step
    for name, sha in (("scripts/mission7_plate_stage.py", M3_STAGE_SHA256),
                      ("scripts/mission7_plate_bench_v2.py", M3_BENCH_V2_SHA256),
                      ("scripts/mission7_route_handoff_probe.py", M3_PROBE_SHA256)):
        try:
            blob = subprocess.run(["git", "-C", str(_REPO), "show", f"{M3_COMMIT}:{name}"], capture_output=True,
                                  check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            pytest.skip("git history not available")
        assert hashlib.sha256(blob).hexdigest() == sha, name
    recorded = json.loads((M3_DIR / "provenance.json").read_text())["sources_sha256"]
    assert recorded["scripts/mission7_plate_stage.py"] == M3_STAGE_SHA256
    assert recorded["scripts/mission7_plate_bench_v2.py"] == M3_BENCH_V2_SHA256
