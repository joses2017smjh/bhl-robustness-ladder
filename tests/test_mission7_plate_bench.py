"""M2 Mission 7 plate bench: grid, frozen verdict, TurnBoth-s0 stage gait, default path, plumbing, headers.

docs/SOLUTIONS_2026-10-01.md section 3, row M2.  No MuJoCo episode runs here: the stage logic runs on a
kinematic stub (a body-frame command integrated in the world frame), the pre-M2 PlateStage is loaded from git
for the default-path comparison, and the launchers are exercised in their dry/plan modes only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import mission7_gates as gates  # noqa: E402
import mission7_plate_bench as bench  # noqa: E402
from bhl_robust.mission.layout import generate  # noqa: E402

LAUNCHER = _REPO / "slurm/repo20260923/cpu_m7_plate_bench.sbatch"
FOLLOWUP = _REPO / "slurm/repo20260923/m7_replay_gate_followup.sbatch"
REPLAY_GATES = _REPO / "slurm/repo20260923/submit_m7_replay_gates.sh"
STAGE_SUBMITTER = _REPO / "scripts/submit_mission7_plate_stage.py"
ROUTE_SUBMITTER = _REPO / "scripts/submit_mission7_route_handoff.py"
PROBE = _REPO / "scripts/mission7_route_handoff_probe.py"
STAGE = _REPO / "scripts/mission7_plate_stage.py"
REPLAY_SOURCE = "results/mission7-replay-smoke-20260921"
REPLAY_BASELINE = "results/mission7-approach-followup-20260922/replay-diagnose-cn-c22"
REFERENCE_COMMIT = "094797e"   # the pre-M2 stage, sha256 fdde7a55...5576
REFERENCE_SHA256 = "fdde7a559bc92d047cc10f6c66775eb24e444db7b211404378bd0e65d3555576"
SIM_STACK = ("mujoco", "torch", "onnxruntime", "omegaconf", "rsl_rl", "berkeley_humanoid_lite_lowlevel")


def _stage_module():
    for module in SIM_STACK:
        pytest.importorskip(module)
    import mission7_plate_stage
    return mission7_plate_stage


# ---- grid -----------------------------------------------------------------------------------------

def test_grid_is_sixteen_cells_of_four_with_two_doors_each():
    rows = bench.grid()
    assert len(rows) == 64 == bench.TOTAL
    assert len({(r["layout"], r["door"]) for r in rows}) == 64
    cells = {}
    for r in rows:
        cells.setdefault((r["heading_deg"], r["plate"], r["entry"]), []).append(r)
    assert len(cells) == 16
    for cell, members in cells.items():
        assert len(members) == 4, cell
        assert sorted(m["door"] for m in members) == [0, 0, 1, 1], cell


def test_grid_sixteen_per_heading_and_balanced_factors():
    rows = bench.grid()
    for heading in (0, 90, -90, 180):
        assert sum(r["heading_deg"] == heading for r in rows) == 16
    for plate in ("round", "square"):
        assert sum(r["plate"] == plate for r in rows) == 32
    for entry in ("standstill", "walking"):
        assert sum(r["entry"] == entry for r in rows) == 32


@pytest.mark.parametrize("layout_index, door, expected", [
    (0, 0, (0, "round", "standstill")), (0, 1, (0, "round", "walking")),
    (1, 0, (90, "round", "standstill")), (2, 0, (-90, "round", "standstill")), (3, 1, (180, "round", "walking")),
    (4, 0, (0, "square", "standstill")), (8, 0, (0, "round", "walking")), (8, 1, (0, "round", "standstill")),
    (15, 0, (180, "square", "walking")), (31, 1, (180, "square", "standstill"))])
def test_grid_formula(layout_index, door, expected):
    # heading by L mod 4, plate by floor(L/4) mod 2, entry by d XOR floor(L/8) mod 2
    assert bench.cell_of(layout_index, door) == expected


def test_bench_layouts_are_train_0_to_31_and_disjoint_from_the_gates():
    assert bench.BENCH_SPLIT == "train" and bench.BENCH_LAYOUTS == tuple(range(32))
    seeds = {generate("train", index).seed for index in bench.BENCH_LAYOUTS}
    assert seeds == set(range(32))
    route_gate = {generate("validation", index).seed for index in gates.ROUTE_GATE_LAYOUTS}
    assert route_gate == set(range(10000, 10016)) and not seeds & route_gate
    baseline = _REPO / REPLAY_BASELINE / "result.json"
    if not baseline.is_file():   # untracked replay inputs; present in the working tree the gates run from
        pytest.skip("replay baseline not on disk")
    source = json.loads(baseline.read_text())
    assert sorted(row["layout_index"] for row in source["episodes_detail"]) == [0, 1, 4, 5, 7, 9, 12, 13, 14, 15]


def test_geometry_uses_the_stage_pre_point_and_the_route_neighbour():
    layout = generate("train", 5)
    for door in (0, 1):
        for plate in ("round", "square"):
            g = bench.crossing_geometry(layout, door, plate, 90)
            side = layout.correct_sides[door] if plate == "round" else -layout.correct_sides[door]
            _, direction = layout.door(door)
            assert g["side"] == side
            assert np.allclose(g["pre_point"], np.asarray(layout.plate(door, side)) - np.asarray(direction) * .30)
            assert np.allclose(g["start"], layout.xy(layout.route[layout.door_indices[door] - 1]))
            yaw = np.arctan2(direction[1], direction[0]) + np.pi / 2
            assert np.isclose(np.cos(g["entry_yaw_rad"]), np.cos(yaw)) and np.isclose(np.sin(g["entry_yaw_rad"]), np.sin(yaw))


# ---- the frozen verdict -----------------------------------------------------------------------------

def _results(misses=(), falls=(), drop=(), extra=None):
    rows = []
    for spec in bench.grid():
        key = (spec["layout"], spec["door"])
        if key in drop:
            continue
        rows.append({**spec, "complete": True, "clear": key not in misses, "fell": key in falls})
    return rows + ([extra] if extra else [])


def _keys(heading, n):
    return [(r["layout"], r["door"]) for r in bench.grid() if r["heading_deg"] == heading][:n]


def test_verdict_64_of_64_passes():
    assert bench.bench_verdict(_results())["verdict"] == "PASS"


def test_verdict_62_of_64_in_two_headings_passes():
    misses = _keys(0, 1) + _keys(180, 1)
    v = bench.bench_verdict(_results(misses=misses))
    assert (v["verdict"], v["clears"], v["per_heading"]["0"]["clears"], v["per_heading"]["180"]["clears"]) == \
        ("PASS", 62, 15, 15)


def test_verdict_61_of_64_fails():
    misses = _keys(0, 1) + _keys(90, 1) + _keys(180, 1)
    v = bench.bench_verdict(_results(misses=misses))
    assert (v["verdict"], v["clears"]) == ("FAIL", 61)
    assert all(h["clears"] >= 15 for h in v["per_heading"].values())   # only the total fails


def test_verdict_14_of_16_in_one_heading_fails_at_62_of_64():
    v = bench.bench_verdict(_results(misses=_keys(-90, 2)))
    assert (v["verdict"], v["clears"], v["per_heading"]["-90"]["clears"]) == ("FAIL", 62, 14)


def test_verdict_15_of_16_in_every_heading_is_60_and_fails():
    misses = [k for h in (0, 90, -90, 180) for k in _keys(h, 1)]
    v = bench.bench_verdict(_results(misses=misses))
    assert (v["verdict"], v["clears"]) == ("FAIL", 60)


@pytest.mark.parametrize("cleared", [True, False])
def test_verdict_any_fall_fails(cleared):
    key = _keys(90, 1)
    v = bench.bench_verdict(_results(falls=key, misses=() if cleared else key))
    assert (v["verdict"], v["falls"]) == ("FAIL", 1)


def test_verdict_missing_or_incomplete_is_incomplete():
    key = _keys(0, 1)
    assert bench.bench_verdict(_results(drop=key))["verdict"] == "INCOMPLETE"
    rows = _results()
    rows[5]["complete"] = False
    v = bench.bench_verdict(rows)
    assert v["verdict"] == "INCOMPLETE" and v["missing"] == [[rows[5]["layout"], rows[5]["door"]]]
    # incomplete even when everything present would fail
    assert bench.bench_verdict(_results(drop=key, falls=_keys(90, 1)))["verdict"] == "INCOMPLETE"


def test_verdict_duplicate_foreign_or_mislabelled_is_incomplete():
    rows = _results()
    assert bench.bench_verdict(rows + [dict(rows[0])])["verdict"] == "INCOMPLETE"
    foreign = {"layout": 40, "door": 0, "heading_deg": 0, "plate": "square", "entry": "walking",
               "complete": True, "clear": True, "fell": False}
    assert bench.bench_verdict(rows + [foreign])["verdict"] == "INCOMPLETE"
    rows[3] = {**rows[3], "heading_deg": 0 if rows[3]["heading_deg"] != 0 else 90}
    assert bench.bench_verdict(rows)["verdict"] == "INCOMPLETE"


def test_verdict_on_a_subset_never_passes():
    specs = bench.grid([251, 250])
    rows = [{**s, "complete": True, "clear": True, "fell": False} for s in specs]
    assert bench.bench_verdict(rows, expected=specs)["verdict"] == "INCOMPLETE"


def test_rule_text_is_the_frozen_rule():
    assert bench.RULE == (
        "Bench: PASS iff >= 62/64 crossings clear (base along >= +0.35 m past the plate centre along the door "
        "direction, within 10 s of takeover, by mission7_gates' clear definition), 0 falls (tilt >= 0.78), and "
        ">= 15/16 clears per heading. Otherwise FAIL; INCOMPLETE if any of the 64 is missing.")
    assert (bench.REQUIRED_CLEARS, bench.REQUIRED_PER_HEADING, bench.CLEAR_WINDOW_S) == (62, 15, 10.0)
    assert (bench.CLEAR_ALONG_M, bench.FALL_TILT) == (gates.CLEAR_ALONG_M, gates.FALL_TILT) == (.35, .78)


# ---- per-crossing scoring on synthetic traces ------------------------------------------------------

def _trace(layout_index=7, door=0, takeover=4.0, cross=5.0, handback=11.0, clear_at=9.0, fall_at=None, end=13.0):
    """A bench row on the plate line: along -0.30 before the crossing, 0.0 during it, 0.40 from clear_at on."""
    layout = generate("train", layout_index)
    side = layout.correct_sides[door]
    plate = np.asarray(layout.plate(door, side))
    _, direction = layout.door(door)
    direction = np.asarray(direction, dtype=float)
    t = np.round(np.arange(1, int(round(end / .04)) + 1) * .04, 10)
    along = np.where(t < cross, -.30, np.where(t < clear_at - 1e-9, 0., .40))
    samples = []
    for ti, a in zip(t, along):
        tilt = 1.0 if fall_at is not None and ti >= fall_at - 1e-9 else .05
        samples.append({"time_s": float(ti), "xy": (plate + direction * a).tolist(), "tilt": tilt})
        if tilt >= .78:
            break
    history = [{"time_s": takeover, "door": door, "side": side, "phase": "approach"},
               {"time_s": takeover, "door": door, "side": side, "phase": "settle"},
               {"time_s": takeover + .4, "door": door, "phase": "turn"},
               {"time_s": cross, "door": door, "phase": "cross"}]
    if handback is not None and (fall_at is None or fall_at > handback):
        history += [{"time_s": handback - 1., "door": door, "phase": "turn_back"},
                    {"time_s": handback, "door": door, "side": side, "phase": "recorded"}]
    tilts = [s["tilt"] for s in samples]
    row = {"layout_index": layout_index, "layout_seed": layout.seed, "replay_elapsed_s": samples[-1]["time_s"],
           "first_fall_s": next((s["time_s"] for s in samples if s["tilt"] >= .78), None),
           "maximum_tilt": max(tilts), "stage_history": history, "samples": samples}
    return row, layout


def test_clear_within_ten_seconds_counts():
    row, layout = _trace(takeover=4.0, clear_at=9.0)
    s = bench.score_crossing(row, layout, 0, 4.0)
    assert s["clear"] and s["real_clear"] and abs(s["clear_after_takeover_s"] - 5.0) < 1e-9


def test_clear_one_sample_after_ten_seconds_is_a_miss():
    row, layout = _trace(takeover=1.0, cross=5.0, clear_at=11.04, handback=12.0, end=14.0)
    s = bench.score_crossing(row, layout, 0, 1.0)
    assert s["real_clear"] and not s["clear"] and abs(s["clear_after_takeover_s"] - 10.04) < 1e-9


def test_clear_at_exactly_ten_seconds_counts():
    row, layout = _trace(takeover=1.0, cross=5.0, clear_at=11.0, handback=12.0, end=14.0)
    s = bench.score_crossing(row, layout, 0, 1.0)
    assert abs(s["clear_after_takeover_s"] - 10.0) < 1e-9 and s["clear"]


def test_no_handback_or_fall_before_handback_is_not_a_clear():
    row, layout = _trace(handback=None)
    assert not bench.score_crossing(row, layout, 0, 4.0)["clear"]
    row, layout = _trace(clear_at=9.0, fall_at=10.0, handback=11.0)
    s = bench.score_crossing(row, layout, 0, 4.0)
    assert s["fell"] and not s["clear"] and not s["real_clear"]


def test_first_clear_time_agrees_with_mission7_gates():
    for kwargs in ({}, {"clear_at": 12.0, "handback": 14.0, "end": 15.0}, {"fall_at": 7.0}, {"handback": None}):
        row, layout = _trace(**kwargs)
        for crossing in gates.crossing_outcomes(row, layout):
            assert (bench.first_clear_time(row, layout, crossing) is not None) == crossing["cleared"]


def test_crossing_that_never_started_is_a_miss():
    row, layout = _trace()
    row["stage_history"] = row["stage_history"][:3]
    s = bench.score_crossing(row, layout, 0, 4.0)
    assert s["crossing"] is None and not s["clear"]
    assert not bench.score_crossing(_trace()[0], layout, 0, None)["clear"]


# ---- the stage gait on a kinematic stub -------------------------------------------------------------

class _Data:
    def __init__(self):
        self.time = 0.
        self.xpos = np.zeros((2, 3))
        self.qpos = np.zeros(7)


class _Slot:
    body_id, qpos_adr = 1, 0


class StubEnv:
    """runner.d / slot / layout / state / controller as PlateStage reads them; a kinematic plant."""

    def __init__(self, layout, xy, yaw):
        self.layout = layout
        self.runner = argparse.Namespace(d=_Data())
        self.slot = _Slot()
        self.state = argparse.Namespace(open=[False, False])
        self.controller = argparse.Namespace(policy="shipped-policy", prev_actions=np.full(22, .5, dtype=np.float32))
        self.set_pose(np.asarray(xy, dtype=float), float(yaw))

    def set_pose(self, xy, yaw):
        d = self.runner.d
        d.xpos[1, :2] = xy
        d.qpos[:2] = xy
        d.qpos[3:7] = [np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]

    def yaw(self):
        q = self.runner.d.qpos[3:7]
        return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))

    def step(self, command, dt=.2):
        yaw = self.yaw()
        vx, vy, wz = np.asarray(command, dtype=float)
        world = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]) @ [vx, vy]
        self.set_pose(self.runner.d.xpos[1, :2] + world * dt, yaw + wz * dt)
        self.runner.d.time = round(self.runner.d.time + dt, 10)
        self.controller.prev_actions[:] = .25 + .01 * (self.runner.d.time % 1)   # the policy wrote an action


def _bench_like(stage_module, layout_index=7, door=1, plate="round", heading=180, **kwargs):
    layout = generate("train", layout_index)
    g = bench.crossing_geometry(layout, door, plate, heading)
    env = StubEnv(layout, g["pre_point"], g["entry_yaw_rad"] + 1e-3)
    stage = stage_module.PlateStage(env, stage_gait="turnboth", turnboth_policy="turnboth-policy", **kwargs)
    return env, stage, g


def test_turnboth_swaps_policy_and_zeroes_prev_actions_at_takeover_and_handback():
    m = _stage_module()
    env, stage, g = _bench_like(m)
    shipped = env.controller.policy
    stage._start(1, g["side"], env.runner.d.time)
    assert env.controller.policy == "turnboth-policy" and not env.controller.prev_actions.any()
    assert stage.gait_events[0]["event"] == "swap_to_turnboth" and stage.gait_events[0]["prev_actions_before_norm"] > 0
    commands, phases = [], []
    for _ in range(400):
        command, phase = stage.command(np.zeros(3))
        commands.append(command)
        phases.append(phase)
        if phase == "recorded":
            break
        assert env.controller.policy == "turnboth-policy"
        env.step(command)
    assert phases[-1] == "recorded"
    assert env.controller.policy is shipped and not env.controller.prev_actions.any()
    assert [e["event"] for e in stage.gait_events] == ["swap_to_turnboth", "swap_to_shipped"]
    assert all(e["prev_actions_after_norm"] == 0. for e in stage.gait_events)
    assert stage.gait_events[1]["prev_actions_before_norm"] > 0
    assert [h["phase"] for h in stage.history] == ["approach", "settle", "turn", "cross", "turn_back", "recorded"]
    assert stage.history[-1]["door"] == 1 and stage.done == {1} and stage.phase == "recorded"


def test_turnboth_turns_in_place_crosses_forward_and_turns_back():
    m = _stage_module()
    env, stage, g = _bench_like(m, heading=90)
    takeover_yaw = env.yaw()
    stage._start(1, g["side"], env.runner.d.time)
    door_yaw, direction = g["door_yaw_rad"], np.asarray(g["direction"])
    log = []
    for _ in range(400):
        command, phase = stage.command(np.zeros(3))
        if phase == "recorded":
            break
        log.append((stage.phase, np.asarray(command, dtype=float), env.yaw(), env.runner.d.xpos[1, :2].copy()))
        env.step(command)
    turns = [c for p, c, _, _ in log if p in ("turn", "turn_back")]
    assert turns and all(c[0] == 0. and c[1] == 0. and abs(abs(c[2]) - m.TURNBOTH_TURN_RATE) < 1e-12 for c in turns)
    crossing = [(c, xy) for p, c, _, xy in log if p == "cross"]
    assert crossing and all(c[1] == 0. and 0. <= c[0] <= m.TURNBOTH_CRUISE + 1e-12 and abs(c[2]) <= m.TURNBOTH_WZ_WALK
                            for c, _ in crossing)
    cross_start = next(h for h in stage.history if h["phase"] == "cross")
    assert abs(cross_start["yaw_error_rad"]) < m.TURNBOTH_TURN_EXIT and not cross_start["turn_timed_out"]
    turn_back = next(h for h in stage.history if h["phase"] == "turn_back")
    assert turn_back["cleared"] and turn_back["along_m"] >= m.TURNBOTH_CROSS_CLEAR_M
    final = stage.history[-1]
    assert abs(final["yaw_error_rad"]) < m.TURNBOTH_TURN_EXIT and not final["turn_back_timed_out"]
    assert abs(np.arctan2(np.sin(env.yaw() - takeover_yaw), np.cos(env.yaw() - takeover_yaw))) < m.TURNBOTH_TURN_EXIT
    # the turn went the short way: from +90 deg it turns clockwise toward the door direction
    first_turn = next(c for p, c, _, _ in log if p == "turn")
    assert first_turn[2] < 0
    assert np.isclose(np.cos(door_yaw), direction[0]) and np.isclose(np.sin(door_yaw), direction[1])


def test_turnboth_takeover_through_the_stage_predicate_swaps_too():
    m = _stage_module()
    layout = generate("validation", 0)
    door = 0
    plate = np.asarray(layout.plate(door, layout.correct_sides[door]))
    _, direction = layout.door(door)
    env = StubEnv(layout, plate - np.asarray(direction) * .6, 0.)
    stage = m.PlateStage(env, stage_gait="turnboth", turnboth_policy="turnboth-policy")
    command, phase = stage.command(np.array([.3, 0., 0.]))
    assert phase == "approach" and env.controller.policy == "turnboth-policy" and not env.controller.prev_actions.any()
    assert stage.takeover_yaw == pytest.approx(0., abs=1e-12)


def test_turnboth_bounded_turn_proceeds_to_the_crossing():
    m = _stage_module()
    env, stage, g = _bench_like(m, heading=180)
    stage._start(1, g["side"], env.runner.d.time)
    while stage.phase in ("approach", "settle"):
        command, _ = stage.command(np.zeros(3))
        env.step(command)
    assert stage.phase == "turn"
    held = 0.
    while stage.phase == "turn":   # a gait that never turns: time passes, the pose does not change
        stage.command(np.zeros(3))
        env.runner.d.time = round(env.runner.d.time + .2, 10)
        held += .2
        assert held < m.TURNBOTH_TURN_MAX_S + 1.
    turn = next(h for h in stage.history if h["phase"] == "turn")
    cross = next(h for h in stage.history if h["phase"] == "cross")
    assert cross["turn_timed_out"] and cross["time_s"] - turn["time_s"] >= m.TURNBOTH_TURN_MAX_S - 1e-9
    assert cross["time_s"] - turn["time_s"] < m.TURNBOTH_TURN_MAX_S + .2 + 1e-9


def test_turnboth_rejects_align_yaw_press_hold_and_unknown_gaits():
    m = _stage_module()
    layout = generate("train", 0)
    env = StubEnv(layout, [0., 0.], 0.)
    for kwargs in ({"align_yaw": True}, {"press_hold": True}):
        with pytest.raises(ValueError):
            m.PlateStage(env, stage_gait="turnboth", turnboth_policy="p", **kwargs)
    with pytest.raises(ValueError):
        m.PlateStage(env, stage_gait="turnfast")
    assert m.PlateStage(env).stage_gait == "shipped"


def test_turnboth_declared_constants():
    m = _stage_module()
    assert m.STAGE_GAITS == ("shipped", "turnboth", "m3")   # m3 added 2026-10-02 (M3, opt-in)
    assert m.TURNBOTH_CROSS_CLEAR_M == gates.CLEAR_ALONG_M
    assert m.TURNBOTH_TURN_RATE == .40 == m.TURNBOTH_WZ_WALK      # Mission 7's |wz| bound (tanh x 0.4)
    assert (m.TURNBOTH_TURN_EXIT, m.TURNBOTH_CRUISE, m.TURNBOTH_K_YAW) == (.15, .30, 1.2)
    assert m.TURNBOTH_TURN_MAX_S == pytest.approx(2 * np.pi / .4)
    from bhl_robust.eval.random_maze import TurnWalkController
    walk = TurnWalkController()
    assert (walk.turn_exit, walk.cruise, walk.k_yaw, walk.wz_walk) == (m.TURNBOTH_TURN_EXIT, m.TURNBOTH_CRUISE,
                                                                       m.TURNBOTH_K_YAW, m.TURNBOTH_WZ_WALK)
    env_source = (_REPO / "src/bhl_robust/mission/env.py").read_text()
    assert m.SHIPPED_EXPORT in env_source and "bounded[:3]*[.4, .35, .4]" in env_source


def test_turnboth_export_check_on_the_real_files():
    m = _stage_module()
    upstream = _REPO / "external/Berkeley-Humanoid-Lite"
    if not (upstream / m.TURNBOTH_EXPORT / "policy.onnx").is_file():
        pytest.skip("TurnBoth-s0 export not on disk")
    info = m.turnboth_check(upstream)
    assert info["policy_sha256"] == m.TURNBOTH_POLICY_SHA256
    assert set(info["deploy_keys_differing"]) <= set(m.TURNBOTH_FREE_DEPLOY_KEYS)
    assert "policy_checkpoint_path" in info["deploy_keys_differing"]


# ---- the default path against the pre-M2 stage -------------------------------------------------------

def _reference_module(tmp_path):
    _stage_module()
    try:
        text = subprocess.run(["git", "-C", str(_REPO), "show", f"{REFERENCE_COMMIT}:scripts/mission7_plate_stage.py"],
                              capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history not available")
    import hashlib
    assert hashlib.sha256(text).hexdigest() == REFERENCE_SHA256
    path = tmp_path / "m7_stage_reference.py"
    path.write_bytes(text)
    spec = importlib.util.spec_from_file_location("m7_stage_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _drive(module, kwargs, layout_index, door, open_at, steps=180):
    layout = generate("validation", layout_index)
    plate = np.asarray(layout.plate(door, layout.correct_sides[door]))
    env = StubEnv(layout, layout.xy(layout.route[layout.door_indices[door] - 1]), 0.)
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
        recorded = np.array([body[0], body[1], -.5 * yaw])
        command, phase = stage.command(recorded)
        trace.append((phase, np.asarray(command, dtype=float).round(12).tolist(),
                      env.controller.prev_actions.copy().tolist()))
        env.step(command)
    return trace, stage


@pytest.mark.parametrize("kwargs", [
    {}, {"cross_clear_m": .35}, {"align_yaw": True}, {"cross_kick": True}, {"wait_open_s": 1.0},
    {"press_hold": True, "wait_open_s": 2.0}, {"stage_lateral_m": .35}, {"pre_point_m": .45, "settle_s": .8},
    {"pre_point_m": .45, "cross_clear_m": .35, "cross_kick": True, "align_yaw": True},
    {"stage_gait": "shipped"}])
@pytest.mark.parametrize("layout_index, door, open_at", [(0, 0, None), (4, 1, 40), (7, 0, 25)])
def test_default_path_matches_the_pre_m2_stage(tmp_path, kwargs, layout_index, door, open_at):
    reference = _reference_module(tmp_path)
    m = _stage_module()
    ref_kwargs = {k: v for k, v in kwargs.items() if k != "stage_gait"}
    expected, ref_stage = _drive(reference, ref_kwargs, layout_index, door, open_at)
    actual, new_stage = _drive(m, kwargs, layout_index, door, open_at)
    assert actual == expected
    assert new_stage.history == ref_stage.history and new_stage.done == ref_stage.done
    assert new_stage.kick_events == ref_stage.kick_events and new_stage.align_events == ref_stage.align_events
    assert any(h["phase"] == "cross" for h in new_stage.history)   # the drive did reach a crossing
    assert not hasattr(new_stage, "gait_events")                    # no stage-gait state on the default path


def test_default_rows_results_and_preflight_keep_their_keys():
    text = STAGE.read_text()
    assert 'if stage_gait != "shipped":   # the default row keeps its keys and their order' in text
    assert "if getattr(args, 'stage_gait', 'shipped') != \"shipped\":   # the default report is unchanged" in text
    assert 'if args.stage_gait != "shipped":   # the default preflight line is unchanged' in text
    probe = PROBE.read_text()
    for guard in ('if getattr(args, "stage_gait", "shipped") != "shipped":   # default rows keep their keys',
                  'if getattr(args, "stage_gait", "shipped") != "shipped":   # the default result is unchanged',
                  'if args.stage_gait != "shipped":   # the default preflight line is unchanged'):
        assert guard in probe, guard


# ---- flag plumbing -------------------------------------------------------------------------------------

def _plan(script, *args):
    out = subprocess.run([sys.executable, str(script), *args], cwd=_REPO, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_replay_submitter_forwards_stage_gait():
    if not ((_REPO / REPLAY_SOURCE / "fullroute/legacy-doors.json").is_file()
            and (_REPO / REPLAY_BASELINE / "result.json").is_file()):
        pytest.skip("untracked replay inputs not on disk (the submitter checks them before planning)")
    campaign = "results/mission7-campaign-20260923"
    plan = _plan(STAGE_SUBMITTER, "--out-campaign", campaign, "--output-name", "replay-gate-test-m2-never-created",
                 "--stage-gait", "turnboth")
    assert plan["probe_args"] == ["--stage-gait=turnboth"]
    default = _plan(STAGE_SUBMITTER, "--out-campaign", campaign, "--output-name", "replay-gate-test-m2-never-created",
                    "--pre-point", "0.45", "--cross-clear", "0.35", "--cross-kick", "--align-yaw")
    assert default["probe_args"] == ["--pre-point=0.45", "--cross-clear=0.35", "--cross-kick", "--align-yaw"]
    bad = subprocess.run([sys.executable, str(STAGE_SUBMITTER), "--out-campaign", campaign, "--output-name",
                          "replay-gate-test-m2-never-created", "--stage-gait", "turnboth", "--align-yaw"],
                         cwd=_REPO, capture_output=True, text=True, timeout=120)
    assert bad.returncode == 2 and "does not compose" in bad.stderr
    assert not (_REPO / campaign / "replay-gate-test-m2-never-created").exists()


def test_route_submitter_forwards_stage_gait():
    campaign = "results/mission7-campaign-20260923"
    common = ["--campaign", campaign, "--stage", "doors", "--indices", "0,1", "--output-name",
              "route-gate-test-m2-never-created", "--handoff", "early", "--stage-activate", "--rejoin-advance",
              "--exit-ramp-center", "--exit-ramp", "1.2", "--stage-wait-open", "0.0"]
    plan = _plan(ROUTE_SUBMITTER, *common, "--stage-gait", "turnboth")
    assert plan["stage_gait"] == "turnboth" and plan["probe_args"][-1] == "--stage-gait=turnboth"
    assert plan["probe_args"][:8] == ["--stage", "doors", "--indices", "0,1", "--handoff", "early", "--rejoin-fix", "none"]
    default = _plan(ROUTE_SUBMITTER, *common)
    assert set(default) == {"planned_output", "stage", "indices", "node", "constraint"}   # unchanged plan
    spec = importlib.util.spec_from_file_location("m7_route_submitter", ROUTE_SUBMITTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = argparse.Namespace(stage="doors", indices="0", handoff="switch", rejoin_fix="none", rejoin_diagnostic=False,
                              chain_trace=False, stage_lateral=None, stage_activate=False, stage_press_hold=False,
                              stage_wait_open=None, pre_point=None, cross_clear=None, stall_min_s=None, settle_s=None,
                              cross_kick=False, rejoin_advance=False, exit_ramp_center=False, align_yaw=False,
                              exit_ramp=0., allow_inactive_intervention=False, stage_gait=None)
    assert module._probe_args(args) == ["--stage", "doors", "--indices", "0", "--handoff", "switch",
                                        "--rejoin-fix", "none"]


def _translation_heredoc():
    text = FOLLOWUP.read_text()
    match = re.search(r'^"\$PY" - "\$gate_dir" > "\$flags_file" <<\'PYEOF\'\n(.*?)\nPYEOF$', text, re.S | re.M)
    assert match, "step-3 heredoc not found"
    return match.group(1)


@pytest.mark.parametrize("probe_args, expected", [
    (["--stage-gait=turnboth"], ["--stage-gait", "turnboth", "--stage-wait-open", "0.0"]),
    (["--pre-point=0.45", "--cross-clear=0.35", "--cross-kick", "--align-yaw"],
     ["--pre-point", "0.45", "--cross-clear", "0.35", "--cross-kick", "--align-yaw", "--stage-wait-open", "0.0"]),
    (["--stage-gait=turnboth", "--wait-open=2.0"], ["--stage-gait", "turnboth", "--stage-wait-open", "2.0"])])
def test_followup_translates_stage_gait_to_the_route_gate(tmp_path, probe_args, expected):
    (tmp_path / "submission.json").write_text(json.dumps({"probe_args": probe_args}))
    script = tmp_path / "translate.py"
    script.write_text(_translation_heredoc())
    out = subprocess.run([sys.executable, str(script), str(tmp_path)], capture_output=True, text=True, check=True)
    assert out.stdout.split("\n")[:-1] == expected


def test_followup_dry_run_releases_a_turnboth_route_gate_from_a_turnboth_pass(tmp_path):
    # bench PASS -> replay gate (probe_args --stage-gait=turnboth) PASS -> the follow-up's route gate, DRY_RUN=1:
    # nothing is locked or submitted; the route submitter runs in plan mode and must accept the flag.
    campaign = tmp_path / "campaign"
    gate = campaign / "replay-gate-m2-synthetic"
    gate.mkdir(parents=True)
    rows = [{"layout_index": i, "fall": False, "stage_history": [], "stage_gait": "turnboth", "gait_events": []}
            for i in (0, 1, 4, 5, 7, 9, 12, 13, 14, 15)]
    (gate / "result.json").write_text(json.dumps({
        "complete": True, "status": "COMPLETED_INTERVENTION", "episodes": 10, "falls": 0, "upright": 10,
        "exact_replay_gate_passed": True, "stage_gait": "turnboth", "episode_comparison": rows}))
    (gate / "submission.json").write_text(json.dumps({"job_id": "0", "probe_args": ["--stage-gait=turnboth"],
                                                      "source_sha256": {}}))
    route_campaign = "results/mission7-campaign-20260923"
    assert not (_REPO / route_campaign / "route-gate-m2-synthetic-doors").exists()
    locks_before = sorted(p.name for p in (_REPO / route_campaign).glob("replay-gate-route-lock-*"))
    env = {**os.environ, "DRY_RUN": "1", "PYTHON": sys.executable, "M7_REPO": str(_REPO),
           "M7_ROUTE_CAMPAIGN": route_campaign}
    out = subprocess.run(["bash", str(FOLLOWUP), str(gate), "m2-synthetic", "1", str(campaign)], env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "M7 REPLAY GATE m2-synthetic: PASS" in out.stdout and "ROUTE_GATE_READY" in out.stdout
    composition = next(line for line in out.stdout.splitlines() if line.startswith("route gate composition"))
    assert "--stage-gait turnboth" in composition and "--stage-wait-open 0.0" in composition
    plans = [json.loads(block) for block in re.findall(r"^\{\n.*?^\}$", out.stdout, re.S | re.M)]
    assert [p["stage"] for p in plans] == ["doors", "transport"]
    assert all(p["probe_args"][-1] == "--stage-gait=turnboth" and p["indices"] == ",".join(map(str, range(16)))
               for p in plans)
    # a dry run takes no lock (the real chain may hold slot 1 by the time this runs again)
    assert sorted(p.name for p in (_REPO / route_campaign).glob("replay-gate-route-lock-*")) == locks_before
    assert not (_REPO / route_campaign / "route-gate-m2-synthetic-doors").exists()


def test_followup_still_refuses_smoke_and_unknown_flags(tmp_path):
    script = tmp_path / "translate.py"
    script.write_text(_translation_heredoc())
    for args in (["--stage-gait=turnboth", "--smoke"], ["--stage-gait=turnboth", "--mystery"]):
        (tmp_path / "submission.json").write_text(json.dumps({"probe_args": args}))
        out = subprocess.run([sys.executable, str(script), str(tmp_path)], capture_output=True, text=True)
        assert out.returncode != 0


def test_followup_header_carries_the_106_episode_line():
    header = FOLLOWUP.read_text().split("set -euo pipefail")[0]
    assert "101 of 512 unspent): five gates" not in header
    assert "106 episodes" in header and "64 bench crossings" in header and "32\n#    route-gate episodes" in header
    assert "2026-10-01" in header


def test_replay_gates_script_has_the_m2_arm():
    text = REPLAY_GATES.read_text()
    assert re.search(r'^\s*"m2-turnboth\s+--stage-gait turnboth"\s*$', text, re.M)
    subprocess.run(["bash", "-n", str(REPLAY_GATES)], check=True)
    assert "106-episode line" in text


def test_probe_and_stage_plumb_stage_gait_to_platestage():
    probe = PROBE.read_text()
    assert "align_yaw=align_yaw, stage_gait=stage_gait)" in probe
    assert 'stage_gait=getattr(args, "stage_gait", "shipped"))' in probe
    assert 'parser.add_argument("--stage-gait", choices=STAGE_GAITS, default="shipped"' in probe
    stage = STAGE.read_text()
    assert "align_yaw=align_yaw, stage_gait=stage_gait)" in stage
    assert "stage_gait=getattr(args, 'stage_gait', 'shipped')))" in stage
    assert 'parser.add_argument("--stage-gait", choices=STAGE_GAITS, default="shipped"' in stage


@pytest.mark.parametrize("script", ["stage", "probe"])
def test_clis_accept_stage_gait_in_preflight(tmp_path, script):
    _stage_module()
    if not (_REPO / "external/Berkeley-Humanoid-Lite/logs/rsl_rl/humanoid/2026-09-25_18-54-44_arms-turn-turnboth-s0"
            "/exported/policy.onnx").is_file():
        pytest.skip("TurnBoth-s0 export not on disk")
    env = {**os.environ, "PYTHONPATH": f"{_REPO / 'src'}:{_REPO / 'scripts'}"}
    if script == "stage":
        args = [str(STAGE), "--repo", str(_REPO), "--campaign", "results/mission7-replay-smoke-20260921",
                "--baseline", "results/mission7-approach-followup-20260922/replay-diagnose-cn-c22",
                "--out", str(tmp_path / "never"), "--stage-gait", "turnboth", "--preflight"]
    else:
        args = [str(PROBE), "--repo", str(_REPO), "--out", str(_REPO / "results/never-created-m2-probe"),
                "--stage", "doors", "--indices", "0", "--stage-gait", "turnboth", "--preflight"]
    out = subprocess.run([sys.executable, *args], cwd=_REPO, capture_output=True, text=True, timeout=300, env=env)
    assert out.returncode == 0, out.stderr
    line = json.loads(out.stdout.strip().splitlines()[-1])
    assert line["status"] == "PREFLIGHT_OK" and line["stage_gait"] == "turnboth"
    assert line["stage_gait_check"]["policy_sha256"].startswith("562ceed7")
    bad = subprocess.run([sys.executable, *args, "--align-yaw"], cwd=_REPO, capture_output=True, text=True,
                         timeout=300, env=env)
    assert bad.returncode == 2


# ---- the bench CLI guards ------------------------------------------------------------------------------

def test_bench_cli_refuses_scored_layouts_in_smoke_and_subsets_in_scored(tmp_path):
    for argv in (["--mode", "smoke", "--crossings", "3:0"], ["--mode", "smoke", "--crossings", "251:2"],
                 ["--mode", "smoke"], ["--mode", "scored", "--crossings", "251:1"]):
        with pytest.raises(SystemExit) as exit_info:
            bench.main(["--repo", str(_REPO), "--out", str(tmp_path / "never"), *argv])
        assert exit_info.value.code == 2
    assert not (tmp_path / "never").exists()


def test_bench_refuses_an_existing_output(tmp_path):
    out = tmp_path / "bench"
    out.mkdir()
    with pytest.raises(SystemExit) as exit_info:
        bench.main(["--repo", str(_REPO), "--out", str(out), "--mode", "smoke", "--crossings", "251:1"])
    assert exit_info.value.code == 2


def test_bench_results_round_trip_and_rescore(tmp_path):
    row, layout = _trace(layout_index=7, door=0)
    spec = {"layout": 7, "door": 0, **dict(zip(("heading_deg", "plate", "entry"), bench.cell_of(7, 0)))}
    scored = bench.score_crossing(row, layout, 0, 4.0)
    result = {**spec, "complete": True, "clear": scored["clear"], "fell": scored["fell"], "takeover_s": 4.0,
              "stage_history": row["stage_history"], "samples": row["samples"],
              "trace_header": {k: row[k] for k in ("layout_index", "layout_seed", "replay_elapsed_s",
                                                    "first_fall_s", "maximum_tilt")}}
    (tmp_path / "crossings").mkdir()
    bench.write_crossing(tmp_path, result)
    with pytest.raises(FileExistsError):
        bench.write_crossing(tmp_path, result)
    [back] = bench.read_results(tmp_path)
    assert back["clear"] == scored["clear"] and "samples" not in back
    tampered = dict(result, clear=not result["clear"], layout=8, door=1)
    tampered["trace_header"] = dict(result["trace_header"])
    with gzip.open(bench.crossing_path(tmp_path, 8, 1), "wt") as stream:
        json.dump(tampered, stream)
    with pytest.raises(ValueError):   # layout 8's geometry is not this trace's (seed mismatch)
        bench.read_results(tmp_path)


def test_compact_result_and_verdict_on_saved_smoke_crossings():
    # the smoke's real crossing files (untracked, gzipped): re-scored, compacted and judged as main() does
    dirs = sorted((_REPO / "results/mission7-campaign-20260923/m2-bench-smoke").glob("job-*/bench"))
    dirs = [d for d in dirs if list((d / "crossings").glob("*.json.gz"))]
    if not dirs:
        pytest.skip("no saved smoke crossings on disk")
    results = bench.read_results(dirs[-1])
    compact = [bench.compact_result(r) for r in results]
    json.dumps(compact)
    for line, r in zip(compact, results):
        assert line["layout"] >= bench.SMOKE_MIN_LAYOUT and line["clear"] == r["clear"] and line["fell"] == r["fell"]
        assert line["phases"][0] == "approach" and isinstance(line["takeover_speed_mps"], float)
    specs = bench.grid(sorted({r["layout"] for r in results}))
    specs = [s for s in specs if (s["layout"], s["door"]) in {(r["layout"], r["door"]) for r in results}]
    assert bench.bench_verdict(results, expected=specs)["verdict"] == "INCOMPLETE"


def test_compact_result_without_a_takeover():
    spec = {"layout": 40, "door": 0, "heading_deg": 0, "plate": "round", "entry": "standstill"}
    result = {**spec, "complete": True, "end_reason": "runup_timeout", "clear": False, "real_clear": False,
              "fell": False, "events": [], "crossing": None, "stage_history": [],
              "plate_activation": {"target_pressed_during_stage": False}, "wall_contacts": {"stage_samples": 0}}
    line = bench.compact_result(result)
    assert line["takeover_speed_mps"] is None and line["along_start_m"] is None and line["phases"] == []
    json.dumps(line)


# ---- the launcher -----------------------------------------------------------------------------------------

def _header():
    return LAUNCHER.read_text().split("set -euo pipefail")[0]


def _flat(text):
    return " ".join(line.lstrip("#").strip() for line in text.splitlines())


def test_launcher_header_carries_the_frozen_rules_verbatim():
    flat = _flat(_header())
    assert bench.RULE in flat
    for clause in (
            "Then the unchanged exact ten-fall replay once with --stage-gait turnboth, through the existing replay-gate "
            "machinery (complete, 10 episodes, 10 upright, 0 falls).",
            "Then the route gate as coded by M1: Doors and Transport each >= 16/16 successes on validation layouts 0-15 "
            "(32 episodes).",
            "M3 (turn while stepping with the shipped gait + a stall watchdog) is predeclared as conditional on a bench "
            "FAIL"):
        assert clause in flat, clause
    for declared in ("0.40 rad/s", "106 episodes", "64 bench crossings", "NOT part of the 106 episodes",
                     "LEARNED gaits", "SCRIPTED stage", "ORACLE layout and plate pose", "unmeasured",
                     "never from an exit code"):
        assert declared in flat, declared
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def test_launcher_header_discloses_the_smoke_turns_runups_and_coverage():
    # review findings 1-3 (2026-10-02): disclosed before the bench, nothing tuned in response
    flat = _flat(_header())
    for disclosed in (
            "its turn at 0.4 rad/s was unmeasured before the protocol was declared; smoke showed 2/3 turns timing out",
            "= 10.05 s at nominal tracking, over the 10 s window",
            "turn 2.60 s at ~0.57 rad/s, no time-out",
            "timed out after 15.8 s with 0.68 rad left",
            "the turn timed out after 15.7 s with 0.85 rad left",
            "0 of 2 bench smoke crossings cleared inside 10 s, which predicts a bench FAIL",
            "a bench PASS does not cover TurnBoth-s0's capture-to-pre-point approach",
            "the exact replay (10 episodes) exercises that phase first, before the route gate",
            "In 17 of the 64 crossings the straight run-up line passes 0.114-0.131 m from the centre of the door's "
            "OPPOSITE plate",
            ", ".join(f"L{layout_index} d{door}" for layout_index, door in bench.RUNUP_OVER_OPPOSITE_PLATE),
            "by heading 0: 3, +90: 6, -90: 4, 180: 4",
            "forward in 15, sideways in 33 and backward in 16",
            "it counts as a bench fall and FAILS the bench"):
        assert disclosed in flat, disclosed
    assert "its turn at 0.4 rad/s was unmeasured before this protocol was declared" in STAGE.read_text().replace(
        "\n# ", " ")


def test_runup_over_opposite_plate_is_the_disclosed_seventeen():
    over, distances, kinds = [], [], {"forward": 0, "sideways": 0, "backward": 0}
    for spec in bench.grid():
        g = bench.crossing_geometry(generate("train", spec["layout"]), spec["door"], spec["plate"], spec["heading_deg"])
        assert not g["runup_over_target_plate"] and g["runup_line_to_plate_centre_m"]["target"] >= bench.PLATE_HALF_M
        if g["runup_over_opposite_plate"]:
            over.append((spec["layout"], spec["door"]))
            distances.append(g["runup_line_to_plate_centre_m"]["opposite"])
        move = np.asarray(g["pre_point"]) - np.asarray(g["start"])
        angle = (np.degrees(np.arctan2(move[1], move[0]) - g["entry_yaw_rad"]) + 180) % 360 - 180
        kinds["forward" if abs(angle) < 45 else "backward" if abs(angle) > 135 else "sideways"] += 1
    assert tuple(over) == bench.RUNUP_OVER_OPPOSITE_PLATE and len(over) == 17
    assert .113 < min(distances) and max(distances) < .132
    headings = [bench.cell_of(layout_index, door)[0] for layout_index, door in over]
    assert {h: headings.count(h) for h in bench.HEADINGS_DEG} == {0: 3, 90: 6, -90: 4, 180: 4}
    assert kinds == {"forward": 15, "sideways": 33, "backward": 16}
    assert bench.segment_distance([0., 1.], [-1., 0.], [1., 0.]) == pytest.approx(1.)
    assert bench.segment_distance([3., 0.], [-1., 0.], [1., 0.]) == pytest.approx(2.)
    assert bench.segment_distance([0., 0.], [1., 1.], [1., 1.]) == pytest.approx(np.sqrt(2.))


def test_runup_report_lists_the_opposite_plate_runups_and_every_fall_phase():
    results = [
        {"layout": 3, "door": 0, "fell": True, "fall_phase": "bench_runup", "takeover_s": None,
         "geometry": {"runup_over_opposite_plate": True}},
        {"layout": 3, "door": 1, "fell": False, "fall_phase": None, "takeover_s": 9.0,
         "geometry": {"runup_over_opposite_plate": True}},
        {"layout": 5, "door": 0, "fell": True, "fall_phase": "stage_cross", "takeover_s": 9.0,
         "geometry": {"runup_over_opposite_plate": True}},
        {"layout": 0, "door": 0, "fell": True, "fall_phase": "stage_turn", "takeover_s": 8.0,
         "geometry": {"runup_over_opposite_plate": False}},
        {"layout": 1, "door": 0, "fell": False, "fall_phase": None, "takeover_s": 8.0, "geometry": {}},
    ]
    report = bench.runup_report(results)
    assert report["runup_over_opposite_plate_not_gated"] == {
        "crossings": [[3, 0], [3, 1], [5, 0]], "count": 3, "fallen": [[3, 0], [5, 0]],
        "fallen_before_takeover": [[3, 0]]}
    assert report["falls_by_phase_not_gated"] == {"bench_runup": 1, "stage_cross": 1, "stage_turn": 1}
    json.dumps(report)
    # reported, not gated: a run-up fall is still a bench fall
    falls = [(layout_index, door) for layout_index, door in bench.RUNUP_OVER_OPPOSITE_PLATE[:1]]
    assert bench.bench_verdict(_results(falls=falls))["verdict"] == "FAIL"


def test_bench_sources_cover_the_replay_snapshot():
    # review finding 6: the bench's clean-tree check and provenance.json cover every file the replay snapshot
    # freezes (scripts/submit_mission7_plate_stage.py), minus the replay's own sbatch, plus the bench launcher
    explicit = set(re.findall(r'"((?:src|slurm)/[^"]+\.(?:py|sbatch))"', STAGE_SUBMITTER.read_text()))
    assert explicit == set(bench.BENCH_SOURCES[:-1]) | {"slurm/mission7_plate_stage.sbatch"}
    snapshot = (explicit - {"slurm/mission7_plate_stage.sbatch"}
                | {str(p.relative_to(_REPO)) for p in (_REPO / "src/bhl_robust/mission").glob("*.py")}
                | {str(p.relative_to(_REPO)) for p in (_REPO / "scripts").glob("mission7*.py")})
    sources = bench.source_files(_REPO)
    assert sources == sorted(set(sources))
    assert set(sources) == snapshot | {"slurm/repo20260923/cpu_m7_plate_bench.sbatch"}
    assert {"scripts/mission7_plate_stage.py", "scripts/mission7_gates.py", "scripts/mission7_diagnostic.py",
            "src/bhl_robust/eval/multi_robot.py", "src/bhl_robust/sensor_io.py"} <= set(sources)
    launcher = re.search(r"^  sources=\((.*?)\)\n", LAUNCHER.read_text(), re.S | re.M)
    assert launcher and set(launcher.group(1).split()) == (set(bench.BENCH_SOURCES)
                                                           | {"scripts/mission7*.py", "src/bhl_robust/mission"})
    replay = re.search(r"^snapshot_files=\((.*?)\)\n", REPLAY_GATES.read_text(), re.S | re.M)
    assert replay
    expanded = set()
    for item in replay.group(1).split():
        expanded.update({str(p.relative_to(_REPO)) for p in _REPO.glob(item)} if "*" in item else {item})
    assert expanded - {"slurm/mission7_plate_stage.sbatch"} <= set(sources)


def test_launcher_smoke_limits():
    text = LAUNCHER.read_text()
    assert "--job-name=m7-plate-bench-smoke --time=01:00:00" in text
    assert "SMOKE_CROSSINGS=251:1,250:0" in text
    assert f"REFERENCE_COMMIT={REFERENCE_COMMIT}" in text and f"REFERENCE_STAGE_SHA256={REFERENCE_SHA256}" in text
    for name, (layout_index, door) in (("251:1", (251, 1)), ("250:0", (250, 0))):
        assert layout_index >= bench.SMOKE_MIN_LAYOUT
    assert {bench.cell_of(251, 1)[2], bench.cell_of(250, 0)[2]} == {"standstill", "walking"}


def _release_function():
    text = LAUNCHER.read_text()
    match = re.search(r"^release_if_pass\(\) \{\n.*?^\}\n", text, re.S | re.M)
    assert match
    return match.group(0)


@pytest.mark.parametrize("verdict, release_rc, released, rc, text", [
    ("PASS", 0, True, 0, "releasing the exact ten-fall replay"),
    ("PASS", 3, True, 1, "RELEASE_BLOCKED"),            # the release script's m2_bench_gate refused
    ("PASS", 1, True, 1, "M7 BENCH RELEASE FAILED (exit 1)"),
    ("FAIL", 0, False, 0, "chain stops"),
    ("INCOMPLETE", 0, False, 0, "nothing released")])
def test_launcher_releases_the_replay_only_on_a_pass_read_from_json(tmp_path, verdict, release_rc, released, rc,
                                                                    text):
    out = tmp_path / "bench"
    out.mkdir()
    (out / "verdict.json").write_text(json.dumps({"verdict": verdict}))
    calls = tmp_path / "calls"
    stub = tmp_path / "release.sh"
    stub.write_text(f'#!/bin/bash\necho "$@ M7_BENCH_VERDICT=$M7_BENCH_VERDICT" >> {calls}\nexit {release_rc}\n')
    script = tmp_path / "run.sh"
    script.write_text("set -euo pipefail\n" + _release_function() + f'release_if_pass "{out}"\n')
    env = {**os.environ, "PY": sys.executable, "RELEASE": str(stub), "DRY_RUN": "0"}
    env.pop("M7_BENCH_VERDICT", None)
    out_run = subprocess.run(["bash", "-c", f'PY="$PY" RELEASE="$RELEASE" DRY_RUN=0; source "{script}"'],
                             env=env, capture_output=True, text=True, timeout=60)
    assert out_run.returncode == rc, out_run.stderr
    assert text in out_run.stdout
    assert calls.exists() == released
    if released:   # the release script is told which verdict (and provenance.json) to check
        assert calls.read_text().strip() == f"--only m2-turnboth --submit M7_BENCH_VERDICT={out}/verdict.json"
    if verdict == "FAIL":
        assert "chain stops" in out_run.stdout and "M3" in out_run.stdout


# ---- review finding 4 + 7: the release script refuses m2-turnboth without a scored bench PASS whose sources match

def _bench_gate_function():
    text = REPLAY_GATES.read_text()
    match = re.search(r"^m2_bench_gate\(\) \{\n.*?^\}\n", text, re.S | re.M)
    assert match
    return match.group(0)


def _bench_dir(tmp_path, verdict="PASS", mode="scored", record=None):
    files = {"scripts/mission7_plate_stage.py": "stage\n", "scripts/mission7_gates.py": "gates\n",
             "src/bhl_robust/mission/env.py": "env\n"}
    for name, body in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(body)
    out = tmp_path / "results/m2-plate-bench"
    out.mkdir(parents=True)
    if verdict is not None:
        (out / "verdict.json").write_text(json.dumps({"verdict": verdict, "mode": mode}))
    recorded = {name: hashlib.sha256(body.encode()).hexdigest() for name, body in files.items()}
    (out / "provenance.json").write_text(json.dumps({"sources_sha256": recorded if record is None else record}))
    return out, sorted(files)


def _run_gate(tmp_path, verdict_path, files):
    script = tmp_path / "gate.sh"
    script.write_text("set -euo pipefail\n" + _bench_gate_function() + 'm2_bench_gate "$@"\n')
    return subprocess.run(["bash", str(script), str(verdict_path), *files], cwd=tmp_path, capture_output=True,
                          text=True, timeout=60, env={**os.environ, "PY": sys.executable})


def test_bench_gate_passes_a_scored_pass_with_unchanged_sources(tmp_path):
    out, files = _bench_dir(tmp_path)
    run = _run_gate(tmp_path, out / "verdict.json", files)
    assert run.returncode == 0, run.stderr
    assert "scored PASS" in run.stdout and "3 snapshot sources identical" in run.stdout


@pytest.mark.parametrize("verdict, mode", [("FAIL", "scored"), ("INCOMPLETE", "scored"), ("PASS", "smoke"),
                                           ("SMOKE_RUN", "smoke"), (None, None)])
def test_bench_gate_refuses_anything_but_a_scored_pass(tmp_path, verdict, mode):
    out, files = _bench_dir(tmp_path, verdict=verdict, mode=mode)
    run = _run_gate(tmp_path, out / "verdict.json", files)
    assert run.returncode == 3 and "RELEASE_BLOCKED m2-turnboth" in run.stderr


def test_bench_gate_refuses_a_stage_changed_since_the_bench(tmp_path):
    out, files = _bench_dir(tmp_path)
    (tmp_path / "scripts/mission7_plate_stage.py").write_text("stage, edited and committed after the bench\n")
    run = _run_gate(tmp_path, out / "verdict.json", files)
    assert run.returncode == 3 and "scripts/mission7_plate_stage.py" in run.stderr


def test_bench_gate_refuses_sources_missing_from_the_bench_record_or_disk(tmp_path):
    out, files = _bench_dir(tmp_path, record={"scripts/mission7_gates.py": "0" * 64})
    run = _run_gate(tmp_path, out / "verdict.json", files)
    assert run.returncode == 3 and "scripts/mission7_plate_stage.py" in run.stderr
    out2, files2 = _bench_dir(tmp_path / "b")
    (tmp_path / "b/src/bhl_robust/mission/env.py").unlink()
    assert _run_gate(tmp_path / "b", out2 / "verdict.json", files2).returncode == 3
    assert _run_gate(tmp_path / "b", out2 / "verdict.json", []).returncode == 3   # nothing to compare
    (out2 / "provenance.json").write_text("not json")
    assert _run_gate(tmp_path / "b", out2 / "verdict.json", files2).returncode == 3


def test_replay_gates_script_applies_the_bench_gate_to_m2_only_and_last():
    text = REPLAY_GATES.read_text()
    assert "M2_BENCH_VERDICT=${M7_BENCH_VERDICT:-$CAMPAIGN/m2-plate-bench/verdict.json}" in text
    calls = [m.start() for m in re.finditer(r'^\s*m2_bench_gate "\$M2_BENCH_VERDICT" "\$\{m2_gate_files\[@\]\}" '
                                            r'\|\| exit 3$', text, re.M)]
    assert len(calls) == 1
    # after the dry-run exit, before the first sbatch, inside the m2-only guard
    assert text.index('echo "DRY RUN: nothing submitted.') < calls[0] < text.index("real_sbatch=$(command -v sbatch)")
    assert text.rindex('if [ "$m2_selected" = 1 ]; then', 0, calls[0]) > text.index("exit 0\nfi\n", text.index(
        'echo "DRY RUN: nothing submitted.'))
    assert ('m2_selected=0; for arm in "${selected[@]}"; do [ "${arm%% *}" != m2-turnboth ] || m2_selected=1; done'
            in text)
    assert text.count("m2_selected=1") == 1
    assert ('m2_gate_files=(); for f in "${snapshot_files[@]}"; do [ "$f" = slurm/mission7_plate_stage.sbatch ] '
            '|| m2_gate_files+=("$f"); done') in text


# ---- review finding 5: smoke mode enforces its own job name and time limit -------------------------------------

def _smoke_limits_function():
    text = LAUNCHER.read_text()
    match = re.search(r"^smoke_limits_ok\(\) \{\n.*?^\}\n", text, re.S | re.M)
    assert match
    return match.group(0)


@pytest.mark.parametrize("name, job_id, limit, ok", [
    ("m7-plate-bench-smoke", "123", "1:00:00", True),
    ("m7-plate-bench-smoke", "123", "01:00:00", True),
    ("m7-plate-bench-smoke", "123", "59:00", True),
    ("m7-plate-bench-smoke", "123", "0:30:00", True),
    ("m7-plate-bench-smoke", "123", "1:00:01", False),
    ("m7-plate-bench-smoke", "123", "4:00:00", False),
    ("m7-plate-bench-smoke", "123", "1-00:00:00", False),
    ("m7-plate-bench-smoke", "123", "UNLIMITED", False),
    ("m7-plate-bench-smoke", "123", "", False),
    ("m7-plate-bench-smoke", "123", "1:00:00:00", False),
    ("m7-plate-bench-smoke", "", "1:00:00", False),
    ("m7-plate-bench", "123", "1:00:00", False),
    ("", "123", "1:00:00", False)])
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
    if not ok:
        assert "refusing" in run.stderr


def test_launcher_smoke_branch_checks_its_limits_before_writing_anything():
    text = LAUNCHER.read_text()
    smoke = text[text.index('[ "$mode" = smoke ] || {'):]
    assert smoke.index("smoke_limits_ok || exit 2") < smoke.index("SMOKE=${M7_SMOKE_DIR") < smoke.index("mkdir -p")
