"""Bench v2 (run-up-free) and the M3 stage (shipped gait, turn while stepping, stall watchdog) for Mission 7.

docs/SOLUTIONS_2026-10-01.md section 3, row M3; bench v2 was introduced after the M2 bench FAIL (21507962).  No MuJoCo
episode runs here: the stage logic runs on a kinematic stub, the pre-M2 and M2 PlateStage files are loaded from git for
the default-path and turnboth-path comparisons, and the launchers run in their dry/plan modes or as extracted functions.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import inspect
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
import mission7_plate_bench as v1  # noqa: E402
import mission7_plate_bench_v2 as v2  # noqa: E402
from bhl_robust.mission.layout import generate  # noqa: E402

LAUNCHER = _REPO / "slurm/repo20260923/cpu_m7_plate_bench_v2.sbatch"
V1_LAUNCHER = _REPO / "slurm/repo20260923/cpu_m7_plate_bench.sbatch"
FOLLOWUP = _REPO / "slurm/repo20260923/m7_replay_gate_followup.sbatch"
REPLAY_GATES = _REPO / "slurm/repo20260923/submit_m7_replay_gates.sh"
STAGE_SUBMITTER = _REPO / "scripts/submit_mission7_plate_stage.py"
ROUTE_SUBMITTER = _REPO / "scripts/submit_mission7_route_handoff.py"
PROBE = _REPO / "scripts/mission7_route_handoff_probe.py"
STAGE = _REPO / "scripts/mission7_plate_stage.py"
M2_DIR = _REPO / "results/mission7-campaign-20260923/m2-plate-bench"
REPLAY_SOURCE = "results/mission7-replay-smoke-20260921"
REPLAY_BASELINE = "results/mission7-approach-followup-20260922/replay-diagnose-cn-c22"
PRE_M2 = ("094797e", "fdde7a559bc92d047cc10f6c66775eb24e444db7b211404378bd0e65d3555576")
M2 = ("a6c64e0", "adb200c509f00fa9cce503353ce5b2149d07715137051f50ba05ad7696c5a1a7")
SIM_STACK = ("mujoco", "torch", "onnxruntime", "omegaconf", "rsl_rl", "berkeley_humanoid_lite_lowlevel")
FROZEN_RULE = (
    "Bench v2: with N crossings run (64 minus the dropped set) and N_h per heading, bench v2 PASSES iff clears >= "
    "N - 2, 0 falls, and clears >= N_h - 1 for every heading. Otherwise FAIL; INCOMPLETE if any declared crossing is "
    "missing. The clear definition is unchanged: mission7_gates' real clear (base along >= +0.35 m past the plate "
    "centre along the door direction), within 10 s of takeover.")


def _stage_module():
    for module in SIM_STACK:
        pytest.importorskip(module)
    import mission7_plate_stage
    return mission7_plate_stage


# ---- grid and the dropped set ---------------------------------------------------------------------------

def test_grid_is_v1s_and_the_declared_set_is_41():
    assert v2.cell_of is v1.cell_of and v2.BENCH_SPLIT == "train" and v2.TOTAL_GRID == 64
    declared = v2.declared_crossings()
    assert len(declared) == 41 == 64 - len(v2.DROPPED)
    assert {h: sum(c["heading_deg"] == h for c in declared) for h in (0, 90, -90, 180)} == {0: 12, 90: 11, -90: 10, 180: 8}
    assert sum(c["entry"] == "standstill" for c in declared) == 32          # no standstill crossing is ever dropped
    walking = sorted((c["layout"], c["door"]) for c in declared if c["entry"] == "walking")
    assert walking == [(0, 1), (1, 1), (2, 1), (8, 0), (17, 1), (18, 1), (20, 1), (25, 0), (28, 0)]


def test_dropped_set_is_frozen_deterministic_and_walking_only():
    assert v2.dropped_crossings() == v2.DROPPED == v2.dropped_crossings()
    assert len(v2.DROPPED) == 23 and len(set(v2.DROPPED)) == 23
    assert all(v1.cell_of(*key)[2] == "walking" for key in v2.DROPPED)
    walking_180 = [(s["layout"], s["door"]) for s in v1.grid() if s["entry"] == "walking" and s["heading_deg"] == 180]
    assert len(walking_180) == 8 and set(walking_180) <= set(v2.DROPPED)


def test_each_drop_has_a_geometric_reason_and_each_run_walk_is_clear():
    for spec in v1.grid():
        if spec["entry"] != "walking":
            continue
        g = v2.crossing_geometry(generate("train", spec["layout"]), spec["door"], spec["plate"], spec["heading_deg"],
                                 "walking")
        blocked = g["walk_path_wall_clearance_m"] < v2.ROBOT_PLANAR_RADIUS_M or bool(g["walk_start_feet_on_plates"])
        assert blocked == ((spec["layout"], spec["door"]) in v2.DROPPED), spec
        assert np.allclose(g["spawn"], g["walk_start"])
        heading = np.array([np.cos(g["entry_yaw_rad"]), np.sin(g["entry_yaw_rad"])])
        assert np.allclose(np.asarray(g["pre_point"]) - np.asarray(g["walk_start"]), .30 * heading)
        if spec["heading_deg"] == 180:   # S = P + 0.30 m along the door direction = the plate centre
            assert np.allclose(g["walk_start"], g["plate_xy"])


def test_standstill_spawn_is_the_pre_point():
    for spec in [s for s in v1.grid() if s["entry"] == "standstill"][:6]:
        layout = generate("train", spec["layout"])
        g = v2.crossing_geometry(layout, spec["door"], spec["plate"], spec["heading_deg"], "standstill")
        side = v1.plate_side(layout, spec["door"], spec["plate"])
        _, direction = layout.door(spec["door"])
        assert np.allclose(g["spawn"], np.asarray(layout.plate(spec["door"], side)) - .30 * np.asarray(direction))
    assert set(v2.STANDSTILL_SPAWN_WALL_OVERLAP) <= {(s["layout"], s["door"]) for s in v2.declared_crossings()
                                                     if s["entry"] == "standstill"}


def test_geometry_helpers_against_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(40):
        centre, half = rng.uniform(-1, 1, 2), rng.uniform(.02, .5, 2)
        a, b = rng.uniform(-2, 2, 2), rng.uniform(-2, 2, 2)
        points = a + (b - a) * np.linspace(0, 1, 20001)[:, None]
        gap = np.maximum(np.abs(points - centre) - half, 0.)
        brute = float(np.hypot(gap[:, 0], gap[:, 1]).min())
        assert v2.segment_box_distance(a, b, centre, half) == pytest.approx(brute, abs=2e-4)
        assert v2.segment_box_distance(a, b, centre, half) <= brute + 1e-12
    assert v2.point_box_distance([0., 0.], [0., 0.], [.1, .1]) == 0.
    assert v2.point_box_distance([1., 1.], [0., 0.], [.5, .5]) == pytest.approx(np.sqrt(.5))
    feet = v2.foot_polygons([0., 0.], 0.)
    assert np.allclose(sorted(feet[0][:, 1]), [.021, .021, .093, .093]) and np.allclose(sorted(feet[1][:, 1]),
                                                                                       [-.093, -.093, -.021, -.021])
    assert feet[0][:, 0].min() == pytest.approx(-.080) and feet[0][:, 0].max() == pytest.approx(.143)
    square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    assert v2.polygon_hits_disc(square, [.5, .5], .01) and v2.polygon_hits_disc(square, [1.2, .5], .25)
    assert not v2.polygon_hits_disc(square, [1.3, .5], .25)
    assert v2.polygon_hits_box(square, [1.2, .5], [.25, .1]) and not v2.polygon_hits_box(square, [1.3, .5], [.25, .1])
    diamond = np.array([[0., -1.], [1., 0.], [0., 1.], [-1., 0.]])
    assert not v2.polygon_hits_box(diamond, [.9, .9], [.3, .3])   # separated only by the diamond's own edge axis


def test_launcher_lists_the_dropped_and_run_sets():
    flat = _flat(_header())
    assert ", ".join(f"L{a} d{b}" for a, b in v2.DROPPED) in flat
    walking = [(c["layout"], c["door"]) for c in v2.declared_crossings() if c["entry"] == "walking"]
    assert "(" + ", ".join(f"L{a} d{b}" for a, b in walking) + ") = 41" in flat
    assert "N = 41 (0 deg: 12, +90: 11, -90: 10, 180: 8), so PASS needs >= 39/41 clears, 0 falls, and >= 11, 10, 9, 7" in flat
    overlap = {"L4 d0 6.6 mm", "L5 d0 6.9", "L11 d1 1.3", "L22 d0 4.2", "L24 d1 7.0", "L30 d1 9.3 mm"}
    assert all(o in flat for o in overlap)
    assert [f"L{a} d{b}" for a, b in v2.STANDSTILL_SPAWN_WALL_OVERLAP] == [o.split(" ")[0] + " " + o.split(" ")[1]
                                                                         for o in sorted(overlap, key=lambda s: int(s.split(" ")[0][1:]))]


# ---- the frozen v2 rule ---------------------------------------------------------------------------------------

def _results(misses=(), falls=(), drop=(), extra=None, declared=None):
    declared = v2.declared_crossings() if declared is None else declared
    rows = [{**c, "complete": True, "clear": (c["layout"], c["door"]) not in misses,
             "fell": (c["layout"], c["door"]) in falls} for c in declared if (c["layout"], c["door"]) not in drop]
    return rows + ([extra] if extra else [])


def _keys(heading, n):
    return [(c["layout"], c["door"]) for c in v2.declared_crossings() if c["heading_deg"] == heading][:n]


def test_rule_text_is_frozen_verbatim():
    assert v2.RULE_V2 == FROZEN_RULE
    assert (v2.MAX_MISSES, v2.MAX_MISSES_PER_HEADING, v2.CLEAR_WINDOW_S, v2.FALL_TILT) == (2, 1, 10.0, .78)
    assert FROZEN_RULE in _flat(_header())


def test_v2_all_clear_passes_and_reports_requirements():
    v = v2.bench_v2_verdict(_results())
    assert (v["verdict"], v["N"], v["required_clears"], v["clears"]) == ("PASS", 41, 39, 41)
    assert {h: e["required"] for h, e in v["per_heading"].items()} == {"0": 11, "90": 10, "-90": 9, "180": 7}


def test_v2_n_minus_2_in_two_headings_passes():
    v = v2.bench_v2_verdict(_results(misses=_keys(0, 1) + _keys(180, 1)))
    assert (v["verdict"], v["clears"], v["per_heading"]["180"]["clears"]) == ("PASS", 39, 7)


def test_v2_n_minus_3_fails_even_with_every_heading_at_n_h_minus_1():
    v = v2.bench_v2_verdict(_results(misses=_keys(0, 1) + _keys(90, 1) + _keys(180, 1)))
    assert (v["verdict"], v["clears"]) == ("FAIL", 38)
    assert all(e["clears"] >= e["required"] for e in v["per_heading"].values())


@pytest.mark.parametrize("heading", [0, 90, -90, 180])
def test_v2_n_h_minus_2_in_one_heading_fails_at_n_minus_2(heading):
    v = v2.bench_v2_verdict(_results(misses=_keys(heading, 2)))
    assert (v["verdict"], v["clears"]) == ("FAIL", 39)
    assert v["per_heading"][str(heading)]["clears"] == v["per_heading"][str(heading)]["declared"] - 2


@pytest.mark.parametrize("cleared", [True, False])
def test_v2_any_fall_fails(cleared):
    key = _keys(-90, 1)
    v = v2.bench_v2_verdict(_results(falls=key, misses=() if cleared else key))
    assert (v["verdict"], v["falls"]) == ("FAIL", 1)


def test_v2_missing_incomplete_dropped_duplicate_mislabelled_or_empty_is_incomplete():
    assert v2.bench_v2_verdict(_results(drop=_keys(0, 1)))["verdict"] == "INCOMPLETE"
    rows = _results()
    rows[3]["complete"] = False
    assert v2.bench_v2_verdict(rows)["verdict"] == "INCOMPLETE"
    dropped = v2.DROPPED[0]
    extra = {"layout": dropped[0], "door": dropped[1], **dict(zip(("heading_deg", "plate", "entry"), v1.cell_of(*dropped))),
             "complete": True, "clear": True, "fell": False}
    v = v2.bench_v2_verdict(_results(extra=extra))
    assert v["verdict"] == "INCOMPLETE" and "not a declared crossing" in v["problems"][0]
    rows = _results()
    assert v2.bench_v2_verdict(rows + [dict(rows[0])])["verdict"] == "INCOMPLETE"
    rows[5] = {**rows[5], "plate": "square" if rows[5]["plate"] == "round" else "round"}
    assert v2.bench_v2_verdict(rows)["verdict"] == "INCOMPLETE"
    assert v2.bench_v2_verdict([], declared=[])["verdict"] == "INCOMPLETE"
    # incomplete even when everything present would fail
    assert v2.bench_v2_verdict(_results(drop=_keys(0, 1), falls=_keys(90, 1)))["verdict"] == "INCOMPLETE"


def test_v2_rule_is_scale_free():
    declared = [c for c in v1.grid() if c["layout"] < 4]           # 8 crossings, 2 per heading
    assert v2.bench_v2_verdict(_results(declared=declared), declared=declared)["verdict"] == "PASS"
    keys = [(c["layout"], c["door"]) for c in declared]
    one_each = [k for k in keys if v1.cell_of(*k)[0] in (0, 90)][::2]   # one miss in two headings: 6/8 = N - 2
    assert v2.bench_v2_verdict(_results(misses=one_each, declared=declared), declared=declared)["verdict"] == "PASS"
    same_heading = [k for k in keys if v1.cell_of(*k)[0] == 0]          # 2 misses at 0 deg: N_h - 2
    assert v2.bench_v2_verdict(_results(misses=same_heading, declared=declared), declared=declared)["verdict"] == "FAIL"


def test_clear_computation_is_v1s_unchanged():
    assert v2.score_crossing is v1.score_crossing
    assert inspect.getsource(v2.check_rescore_m2).count("score_crossing(") == 1


def test_rescore_of_the_38_m2_crossings_reproduces_their_clears():
    if not list((M2_DIR / "crossings").glob("L*-d*.json.gz")):
        pytest.skip("the M2 bench crossing records (gitignored) are not on disk")
    report = v2.check_rescore_m2(M2_DIR)
    assert (report["reached_stage"], report["matches"], report["all_match"]) == (38, 38, True)
    assert report["recorded_clears"] == report["rescored_clears"] == 8
    json.dumps(report)


# ---- the M3 stage on a kinematic stub -----------------------------------------------------------------

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

    def step(self, command, dt=.2, move=True):
        yaw = self.yaw()
        vx, vy, wz = np.asarray(command, dtype=float)
        world = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]) @ [vx, vy]
        if move:
            self.set_pose(self.runner.d.xpos[1, :2] + world * dt, yaw + wz * dt)
        self.runner.d.time = round(self.runner.d.time + dt, 10)
        self.controller.prev_actions[:] = .25 + .01 * (self.runner.d.time % 1)   # the policy wrote an action


def _m3_bench_like(m, layout_index=7, door=1, plate="round", heading=90, **kwargs):
    layout = generate("train", layout_index)
    g = v2.crossing_geometry(layout, door, plate, heading, "standstill")
    env = StubEnv(layout, g["pre_point"], g["entry_yaw_rad"] + 1e-3)
    stage = m.PlateStage(env, stage_gait="m3", **kwargs)
    return env, stage, g


def _run(stage, env, steps=400, move=True):
    log = []
    for _ in range(steps):
        command, phase = stage.command(np.zeros(3))
        log.append((phase, np.asarray(command, dtype=float).copy(), float(env.runner.d.time)))
        if phase == "recorded":
            break
        env.step(command, move=move)
    return log


def test_m3_keeps_the_shipped_policy_and_turns_while_stepping():
    m = _stage_module()
    env, stage, g = _m3_bench_like(m, heading=90)
    shipped = env.controller.policy
    stage._start(1, g["side"], env.runner.d.time)
    assert env.controller.policy is shipped and env.controller.prev_actions.any()   # no swap, no zeroing at takeover
    log = _run(stage, env)
    assert log[-1][0] == "recorded" and env.controller.policy is shipped and stage.gait_events == []
    assert [e["event"] for e in stage.m3_events] == ["takeover"]                     # the stub never stalls
    turns = [c for p, c, _ in log if p in ("turn", "turn_back")]
    assert turns and all(c[0] == m.M3_TURN_VX and c[1] == 0. and abs(c[2]) == m.M3_TURN_WZ for c in turns)
    crossing = [c for p, c, _ in log if p == "cross"]
    assert all(c[0] == m.M3_CROSS_VX and c[1] == 0. and abs(c[2]) <= m.M3_WZ_HOLD + 1e-12 for c in crossing)
    phases = [h["phase"] for h in stage.history]
    assert phases == ["approach", "settle", "turn", "cross", "turn_back", "recorded"]
    assert stage.history[-1]["door"] == 1 and stage.done == {1} and stage.phase == "recorded"
    # from +90 deg the short way is clockwise; the arc carries the base about one radius toward the door
    first_turn = next(c for p, c, _ in log if p == "turn")
    assert first_turn[2] < 0
    turn_back = next(h for h in stage.history if h["phase"] == "turn_back")
    assert turn_back["cleared"] and turn_back["along_m"] >= m.M3_CROSS_CLEAR_M
    assert not stage.history[-1]["turn_back_timed_out"]


def test_m3_crossing_is_scored_by_mission7_gates():
    m = _stage_module()
    env, stage, g = _m3_bench_like(m, heading=0, layout_index=8, door=1)
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
    assert [h["phase"] for h in stage.history].count("cross") == 1


@pytest.mark.parametrize("side_sign", [1, -1])
def test_m3_u_turn_goes_toward_the_door_centreline(side_sign):
    m = _stage_module()
    for layout_index, door in ((3, 0), (7, 0), (11, 1), (15, 1), (19, 0), (23, 0), (27, 1), (31, 1)):
        layout = generate("train", layout_index)
        for plate in ("round", "square"):
            if v1.plate_side(layout, door, plate) != side_sign:
                continue
            env, stage, g = _m3_bench_like(m, layout_index=layout_index, door=door, plate=plate, heading=180)
            stage._start(door, g["side"], env.runner.d.time)
            log = _run(stage, env, steps=12)
            first_turn = next(c for p, c, _ in log if p == "turn")
            assert np.sign(first_turn[2]) == side_sign            # wz sign = side: the arc ends toward the centreline
            return
    pytest.skip("no 180-deg crossing with that side among the probed layouts")


def test_m3_watchdog_recovers_twice_then_hands_back_in_the_turn():
    m = _stage_module()
    env, stage, g = _m3_bench_like(m, heading=90)
    shipped = env.controller.policy
    stage._start(1, g["side"], env.runner.d.time)
    log = _run(stage, env, move=False)                          # a gait that never steps: time passes, the pose does not
    assert log[-1][0] == "recorded" and env.controller.policy is shipped
    events = [e["event"] for e in stage.m3_events]
    assert events == ["takeover", "stall_recover", "resume", "stall_recover", "resume", "stall_handback"]
    recover = [c for p, c, _ in log if p == "recover"]
    assert recover and all(np.allclose(c, [-m.M3_RECOVER_VX, 0., 0.]) for c in recover)
    recoveries = [h for h in stage.history if h["phase"] == "recover"]
    assert [h["recovery"] for h in recoveries] == [1, 2] and all(h["resume"] == "turn" for h in recoveries)
    assert all(h["prev_actions_before_norm"] > 0 for h in recoveries)
    turn_start = next(h["time_s"] for h in stage.history if h["phase"] == "turn")
    times = [h["time_s"] - turn_start for h in recoveries]
    window, pulse = m.M3_STALL_WINDOW_S, m.M3_RECOVER_S
    assert times[0] == pytest.approx(window) and times[1] == pytest.approx(2 * window + pulse)
    last = stage.history[-1]
    assert last["phase"] == "recorded" and last["watchdog_handback"] and last["stalled_in"] == "turn"
    assert last["time_s"] - turn_start == pytest.approx(3 * window + 2 * pulse)
    assert "cross" not in [h["phase"] for h in stage.history] and stage.done == {1}


def test_m3_watchdog_in_the_crossing_keeps_one_cross_entry():
    m = _stage_module()
    env, stage, g = _m3_bench_like(m, heading=0, layout_index=8, door=1)
    stage._start(1, g["side"], env.runner.d.time)
    log = _run(stage, env, move=False)
    phases = [h["phase"] for h in stage.history]
    # two recoveries inside the 4.0 s cross bound, then the bound ends the crossing (a third stall would come later)
    assert phases == ["approach", "settle", "turn", "cross", "recover", "recover", "turn_back", "recorded"]
    assert all(h["resume"] == "cross" for h in stage.history if h["phase"] == "recover")
    assert [e["event"] for e in stage.m3_events] == ["takeover", "stall_recover", "resume", "stall_recover", "resume"]
    assert any(p == "recover" for p, _, _ in log) and not stage.history[-1].get("watchdog_handback")
    [crossing] = gates.crossing_outcomes({"layout_index": 8, "stage_history": stage.history,
                                          "samples": [{"time_s": t, "xy": env.runner.d.xpos[1, :2].tolist(), "tilt": .05}
                                                      for _, _, t in log]}, env.layout)
    assert not crossing["cleared"] and crossing["handback_s"] == stage.history[-1]["time_s"]


def test_m3_replay_trace_reads_through_the_replay_gate_reader(tmp_path):
    """The reader half of the smoke's removed step 6 (it ran the m3 replay on validation layout 0, a scored
    replay-gate layout): an m3 row shaped as _episode() writes it (m3_events before samples, recover entries in the
    stage history), written by mission7_overnight.write(), reads back through gates.replay_gate_crossings with
    result.json's rows.  Kinematic stub only: no episode, no policy, no physics."""
    m = _stage_module()
    layout, door = generate("validation", 0), 0
    g = v2.crossing_geometry(layout, door, "round", 0, "standstill")
    env = StubEnv(layout, g["pre_point"], g["entry_yaw_rad"] + 1e-3)
    stage = m.PlateStage(env, stage_gait="m3")
    stage._start(door, g["side"], env.runner.d.time)
    samples = []
    for _ in range(400):
        command, phase = stage.command(np.zeros(3))
        env.step(command, move=False)   # a gait that never steps: the watchdog recovers twice in the crossing
        samples.append({"time_s": float(env.runner.d.time), "xy": env.runner.d.xpos[1, :2].tolist(), "tilt": .05,
                        "phase": phase, "contacts": [], "contact_events": []})
        if phase == "recorded":
            break
    row = {"layout_index": 0, "layout_seed": int(layout.seed), "replay_elapsed_s": samples[-1]["time_s"], "fall": False,
           "first_fall_s": None, "maximum_tilt": .05, "stage_history": stage.history, "stage_gait": "m3",
           "gait_events": stage.gait_events, "m3_events": stage.m3_events, "samples": samples}
    m.write(tmp_path / "episodes.json", {"complete": True, "episodes": [row]})
    comparison = [{key: value for key, value in row.items() if key != "samples"}]
    traced = gates.replay_gate_crossings(gates.replay_trace_path(tmp_path), comparison)
    assert (traced["episodes"], traced["crossings"], traced["real_clears"]) == (1, 1, 0)
    [crossing] = traced["layouts"][0]["crossings"]
    assert crossing["handed_back"] and not crossing["cleared"] and crossing["door"] == door
    [lean] = gates.read_replay_episodes(tmp_path / "episodes.json")
    assert lean["m3_events"] == stage.m3_events and lean["gait_events"] == [] and lean["stage_gait"] == "m3"
    assert [h["phase"] for h in lean["stage_history"]].count("recover") == 2
    assert [e["event"] for e in lean["m3_events"]] == ["takeover", "stall_recover", "resume", "stall_recover", "resume"]


def test_m3_watchdog_does_not_fire_on_a_moving_base():
    m = _stage_module()
    stage = m.PlateStage(StubEnv(generate("train", 0), [0., 0.], 0.), stage_gait="m3")
    stage._m3_segment(0., [0., 0.])
    for k in range(1, 30):   # 0.06 m/s: 0.072 m per 1.2 s window, above the 0.05 m threshold
        assert not stage._m3_stalled(k * .2, [k * .2 * .06, 0.])
    stage._m3_segment(10., [0., 0.])
    assert not stage._m3_stalled(11.0, [0., 0.])                  # window not yet full
    assert stage._m3_stalled(11.2, [.049, 0.])                   # 1.2 s, 0.049 m < 0.05 m


def test_m3_rejects_align_yaw_and_press_hold_and_constants_are_declared():
    m = _stage_module()
    env = StubEnv(generate("train", 0), [0., 0.], 0.)
    for kwargs in ({"align_yaw": True}, {"press_hold": True}):
        with pytest.raises(ValueError):
            m.PlateStage(env, stage_gait="m3", **kwargs)
    assert m.STAGE_GAITS == ("shipped", "turnboth", "m3")
    assert (m.M3_TURN_VX, m.M3_TURN_WZ, m.M3_ALIGN_EXIT, m.M3_CROSS_VX, m.M3_K_YAW, m.M3_WZ_HOLD) == (
        .30, .40, .25, .30, 1.2, .35)
    assert m.M3_TURN_MAX_S == pytest.approx(2 * np.pi / .4) and m.M3_CROSS_CLEAR_M == gates.CLEAR_ALONG_M
    assert (m.M3_STALL_WINDOW_S, m.M3_STALL_MIN_M, m.M3_RECOVER_VX, m.M3_RECOVER_S, m.M3_MAX_RECOVERIES) == (
        1.2, .05, .30, .40, 2)
    from bhl_robust.mission import approach_debug
    assert inspect.signature(approach_debug.BearingController.__init__).parameters["heading_tolerance"].default == m.M3_ALIGN_EXIT
    source = inspect.getsource(approach_debug)
    assert "np.clip(-1.2*yaw,-.35,.35)" in source                                   # the route's heading hold
    assert "now-1.2<=r['time_s']<=now" in source and "<.05" in source               # RecoveryRouteController
    assert "Commands stay at the measured 0.30 m/s onset threshold" in source
    assert m.m3_constants()["turn_vx_mps"] == .30 and "never swapped" in m.m3_constants()["policy"]
    assert "turn while stepping" in m.stage_gait_description("m3")
    assert m.stage_gait_description("turnboth") == m.turnboth_description()


# ---- default and turnboth paths against git ------------------------------------------------------------------

def _git_module(tmp_path, commit, sha256, name):
    _stage_module()
    try:
        text = subprocess.run(["git", "-C", str(_REPO), "show", f"{commit}:scripts/mission7_plate_stage.py"],
                              capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history not available")
    assert hashlib.sha256(text).hexdigest() == sha256
    path = tmp_path / f"{name}.py"
    path.write_bytes(text)
    spec = importlib.util.spec_from_file_location(name, path)
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
        command, phase = stage.command(np.array([body[0], body[1], -.5 * yaw]))
        trace.append((phase, np.asarray(command, dtype=float).round(12).tolist(),
                      env.controller.prev_actions.copy().tolist(), env.controller.policy))
        env.step(command)
    return trace, stage


@pytest.mark.parametrize("kwargs", [{}, {"cross_clear_m": .35}, {"align_yaw": True}, {"cross_kick": True},
                                    {"press_hold": True, "wait_open_s": 2.0}, {"stage_lateral_m": .35},
                                    {"pre_point_m": .45, "cross_clear_m": .35, "cross_kick": True, "align_yaw": True},
                                    {"stage_gait": "shipped"}])
@pytest.mark.parametrize("layout_index, door, open_at", [(0, 0, None), (4, 1, 40), (7, 0, 25)])
def test_default_path_matches_the_pre_m2_stage(tmp_path, kwargs, layout_index, door, open_at):
    reference = _git_module(tmp_path, *PRE_M2, "m7_stage_pre_m2")
    m = _stage_module()
    expected, ref_stage = _drive(reference, {k: v for k, v in kwargs.items() if k != "stage_gait"}, layout_index, door,
                                 open_at)
    actual, new_stage = _drive(m, kwargs, layout_index, door, open_at)
    assert actual == expected and new_stage.history == ref_stage.history and new_stage.done == ref_stage.done
    assert not hasattr(new_stage, "gait_events") and not hasattr(new_stage, "m3_events")


@pytest.mark.parametrize("kwargs", [{}, {"cross_clear_m": .30}, {"wait_open_s": 1.0}, {"cross_kick": True}])
@pytest.mark.parametrize("layout_index, door, open_at", [(0, 0, None), (4, 1, 40), (7, 0, 25), (13, 1, None)])
def test_turnboth_path_matches_the_m2_stage(tmp_path, kwargs, layout_index, door, open_at):
    reference = _git_module(tmp_path, *M2, "m7_stage_m2")
    m = _stage_module()
    common = {"stage_gait": "turnboth", "turnboth_policy": "turnboth-policy", **kwargs}
    expected, ref_stage = _drive(reference, common, layout_index, door, open_at, steps=260)
    actual, new_stage = _drive(m, common, layout_index, door, open_at, steps=260)
    assert actual == expected
    assert new_stage.history == ref_stage.history and new_stage.gait_events == ref_stage.gait_events
    assert any(h["phase"] == "turn" for h in new_stage.history) and not hasattr(new_stage, "m3_events")


def test_v1_bench_and_launcher_are_the_files_the_m2_bench_ran():
    recorded = json.loads((M2_DIR / "provenance.json").read_text())["sources_sha256"]
    for name in ("scripts/mission7_plate_bench.py", "slurm/repo20260923/cpu_m7_plate_bench.sbatch"):
        assert hashlib.sha256((_REPO / name).read_bytes()).hexdigest() == recorded[name], name


def test_turnboth_and_default_outputs_keep_their_keys():
    text = STAGE.read_text()
    assert ('        if stage_gait == "m3":   # the turnboth row keeps its keys too\n'
            '            row["m3_events"] = stage.m3_events\n') in text
    assert ('        report["stage_gait_provenance"] = (load_turnboth_policy(env)[1] if args.stage_gait == "turnboth"\n'
            '                                           else m3_constants())') in text
    probe = PROBE.read_text()
    assert ('                if args.stage_gait == "m3":   # turnboth rows keep their keys\n'
            '                    row["m3_events"] = controller.stage.m3_events') in probe


# ---- plumbing --------------------------------------------------------------------------------------------------

def _plan(script, *args):
    out = subprocess.run([sys.executable, str(script), *args], cwd=_REPO, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_replay_submitter_forwards_m3():
    if not ((_REPO / REPLAY_SOURCE / "fullroute/legacy-doors.json").is_file()
            and (_REPO / REPLAY_BASELINE / "result.json").is_file()):
        pytest.skip("untracked replay inputs not on disk")
    campaign = "results/mission7-campaign-20260923"
    plan = _plan(STAGE_SUBMITTER, "--out-campaign", campaign, "--output-name", "replay-gate-test-m3-never-created",
                 "--stage-gait", "m3")
    assert plan["probe_args"] == ["--stage-gait=m3"]
    assert _plan(STAGE_SUBMITTER, "--out-campaign", campaign, "--output-name", "replay-gate-test-m3-never-created",
                 "--stage-gait", "turnboth")["probe_args"] == ["--stage-gait=turnboth"]
    bad = subprocess.run([sys.executable, str(STAGE_SUBMITTER), "--out-campaign", campaign, "--output-name",
                          "replay-gate-test-m3-never-created", "--stage-gait", "m3", "--press-hold"],
                         cwd=_REPO, capture_output=True, text=True, timeout=120)
    assert bad.returncode == 2 and "does not compose" in bad.stderr
    assert not (_REPO / campaign / "replay-gate-test-m3-never-created").exists()


def test_route_submitter_forwards_m3():
    common = ["--campaign", "results/mission7-campaign-20260923", "--stage", "doors", "--indices", "0,1",
              "--output-name", "route-gate-test-m3-never-created", "--handoff", "early", "--stage-activate",
              "--rejoin-advance", "--exit-ramp-center", "--exit-ramp", "1.2", "--stage-wait-open", "0.0"]
    plan = _plan(ROUTE_SUBMITTER, *common, "--stage-gait", "m3")
    assert plan["stage_gait"] == "m3" and plan["probe_args"][-1] == "--stage-gait=m3"
    assert set(_plan(ROUTE_SUBMITTER, *common)) == {"planned_output", "stage", "indices", "node", "constraint"}
    spec = importlib.util.spec_from_file_location("m7_route_submitter_v2", ROUTE_SUBMITTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = argparse.Namespace(stage="doors", indices="0", handoff="switch", rejoin_fix="none", rejoin_diagnostic=False,
                              chain_trace=False, stage_lateral=None, stage_activate=False, stage_press_hold=False,
                              stage_wait_open=None, pre_point=None, cross_clear=None, stall_min_s=None, settle_s=None,
                              cross_kick=False, rejoin_advance=False, exit_ramp_center=False, align_yaw=False,
                              exit_ramp=0., allow_inactive_intervention=False, stage_gait="m3")
    assert module._probe_args(args)[-1] == "--stage-gait=m3"
    args.stage_gait = None
    assert module._probe_args(args) == ["--stage", "doors", "--indices", "0", "--handoff", "switch", "--rejoin-fix", "none"]


def _translation_heredoc():
    text = FOLLOWUP.read_text()
    match = re.search(r'^"\$PY" - "\$gate_dir" > "\$flags_file" <<\'PYEOF\'\n(.*?)\nPYEOF$', text, re.S | re.M)
    assert match, "step-3 heredoc not found"
    return match.group(1)


def test_followup_translates_m3_to_the_route_gate(tmp_path):
    (tmp_path / "submission.json").write_text(json.dumps({"probe_args": ["--stage-gait=m3"]}))
    script = tmp_path / "translate.py"
    script.write_text(_translation_heredoc())
    out = subprocess.run([sys.executable, str(script), str(tmp_path)], capture_output=True, text=True, check=True)
    assert out.stdout.split("\n")[:-1] == ["--stage-gait", "m3", "--stage-wait-open", "0.0"]
    header = FOLLOWUP.read_text().split("set -euo pipefail")[0]
    assert "m3-shipped-step" in header and "--stage-gait m3" in header and "106 episodes" in header


def test_followup_dry_run_releases_an_m3_route_gate_from_an_m3_pass(tmp_path):
    campaign = tmp_path / "campaign"
    gate = campaign / "replay-gate-m3-synthetic"
    gate.mkdir(parents=True)
    rows = [{"layout_index": i, "fall": False, "stage_history": [], "stage_gait": "m3", "gait_events": [],
             "m3_events": []} for i in (0, 1, 4, 5, 7, 9, 12, 13, 14, 15)]
    (gate / "result.json").write_text(json.dumps({
        "complete": True, "status": "COMPLETED_INTERVENTION", "episodes": 10, "falls": 0, "upright": 10,
        "exact_replay_gate_passed": True, "stage_gait": "m3", "episode_comparison": rows}))
    (gate / "submission.json").write_text(json.dumps({"job_id": "0", "probe_args": ["--stage-gait=m3"],
                                                      "source_sha256": {}}))
    route_campaign = "results/mission7-campaign-20260923"
    locks_before = sorted(p.name for p in (_REPO / route_campaign).glob("replay-gate-route-lock-*"))
    env = {**os.environ, "DRY_RUN": "1", "PYTHON": sys.executable, "M7_REPO": str(_REPO),
           "M7_ROUTE_CAMPAIGN": route_campaign}
    out = subprocess.run(["bash", str(FOLLOWUP), str(gate), "m3-synthetic", "1", str(campaign)], env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "M7 REPLAY GATE m3-synthetic: PASS" in out.stdout and "ROUTE_GATE_READY" in out.stdout
    composition = next(line for line in out.stdout.splitlines() if line.startswith("route gate composition"))
    assert "--stage-gait m3" in composition and "--stage-wait-open 0.0" in composition
    plans = [json.loads(block) for block in re.findall(r"^\{\n.*?^\}$", out.stdout, re.S | re.M)]
    assert [p["stage"] for p in plans] == ["doors", "transport"]
    assert all(p["probe_args"][-1] == "--stage-gait=m3" and p["indices"] == ",".join(map(str, range(16))) for p in plans)
    assert sorted(p.name for p in (_REPO / route_campaign).glob("replay-gate-route-lock-*")) == locks_before
    assert not (_REPO / route_campaign / "route-gate-m3-synthetic-doors").exists()


def test_replay_gates_script_has_the_m3_arm_and_gate_placement():
    text = REPLAY_GATES.read_text()
    assert re.search(r'^\s*"m3-shipped-step\s+--stage-gait m3"\s*$', text, re.M)
    assert re.search(r'^\s*"m2-turnboth\s+--stage-gait turnboth"\s*$', text, re.M)
    subprocess.run(["bash", "-n", str(REPLAY_GATES)], check=True)
    assert "M3_BENCH_VERDICT=${M7_BENCH_V2_VERDICT:-$CAMPAIGN/m3-plate-bench-v2/verdict.json}" in text
    calls = [mt.start() for mt in re.finditer(r'^\s*m3_bench_gate "\$M3_BENCH_VERDICT" "\$\{m2_gate_files\[@\]\}" '
                                              r'\|\| exit 3$', text, re.M)]
    assert len(calls) == 1
    assert text.index('echo "DRY RUN: nothing submitted.') < calls[0] < text.index("real_sbatch=$(command -v sbatch)")
    assert text.rindex('if [ "$m3_selected" = 1 ]; then', 0, calls[0]) > text.index(
        'm2_bench_gate "$M2_BENCH_VERDICT" "${m2_gate_files[@]}" || exit 3')
    assert text.count("m2_selected=1") == 1 and text.count("m3_selected=1") == 1


def _m3_gate_function():
    match = re.search(r"^m3_bench_gate\(\) \{\n.*?^\}\n", REPLAY_GATES.read_text(), re.S | re.M)
    assert match
    return match.group(0)


def _v2_bench_dir(tmp_path, verdict=None, provenance=None, record=None):
    files = {"scripts/mission7_plate_stage.py": "stage\n", "scripts/mission7_gates.py": "gates\n",
             "src/bhl_robust/mission/env.py": "env\n"}
    for name, body in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(body)
    out = tmp_path / "results/m3-plate-bench-v2"
    out.mkdir(parents=True)
    verdict = {"bench": "v2", "mode": "scored", "stage_gait": "m3", "verdict": "PASS"} if verdict is None else verdict
    (out / "verdict.json").write_text(json.dumps(verdict))
    recorded = {name: hashlib.sha256(body.encode()).hexdigest() for name, body in files.items()}
    prov = {"bench": "v2", "mode": "scored", "stage_gait": "m3", "sources_sha256": recorded if record is None else record}
    (out / "provenance.json").write_text(json.dumps(prov if provenance is None else {**prov, **provenance}))
    return out, sorted(files)


def _run_m3_gate(tmp_path, verdict_path, files):
    script = tmp_path / "gate.sh"
    script.write_text("set -euo pipefail\n" + _m3_gate_function() + 'm3_bench_gate "$@"\n')
    return subprocess.run(["bash", str(script), str(verdict_path), *files], cwd=tmp_path, capture_output=True,
                          text=True, timeout=60, env={**os.environ, "PY": sys.executable})


def test_m3_gate_passes_a_scored_v2_m3_pass(tmp_path):
    out, files = _v2_bench_dir(tmp_path)
    run = _run_m3_gate(tmp_path, out / "verdict.json", files)
    assert run.returncode == 0, run.stderr
    assert "scored bench-v2 m3 PASS" in run.stdout and "3 snapshot sources identical" in run.stdout


@pytest.mark.parametrize("verdict, provenance", [
    ({"bench": "v2", "mode": "scored", "stage_gait": "m3", "verdict": "FAIL"}, None),
    ({"bench": "v2", "mode": "scored", "stage_gait": "m3", "verdict": "INCOMPLETE"}, None),
    ({"bench": "v2", "mode": "smoke", "stage_gait": "m3", "verdict": "PASS"}, None),
    ({"bench": "v2", "mode": "scored", "stage_gait": "turnboth", "verdict": "PASS"}, None),
    ({"mode": "scored", "verdict": "PASS"}, None),                                    # a v1 (M2) verdict
    ({"bench": "v2", "mode": "scored", "stage_gait": "m3", "verdict": "PASS"}, {"stage_gait": "turnboth"}),
    ({"bench": "v2", "mode": "scored", "stage_gait": "m3", "verdict": "PASS"}, {"mode": "smoke"})])
def test_m3_gate_refuses_anything_but_a_scored_v2_m3_pass(tmp_path, verdict, provenance):
    out, files = _v2_bench_dir(tmp_path, verdict=verdict, provenance=provenance)
    run = _run_m3_gate(tmp_path, out / "verdict.json", files)
    assert run.returncode == 3 and "RELEASE_BLOCKED m3-shipped-step" in run.stderr


def test_m3_gate_refuses_changed_missing_or_unrecorded_sources(tmp_path):
    out, files = _v2_bench_dir(tmp_path)
    (tmp_path / "scripts/mission7_plate_stage.py").write_text("stage, edited after the bench\n")
    run = _run_m3_gate(tmp_path, out / "verdict.json", files)
    assert run.returncode == 3 and "scripts/mission7_plate_stage.py" in run.stderr
    out2, files2 = _v2_bench_dir(tmp_path / "b", record={"scripts/mission7_gates.py": "0" * 64})
    assert _run_m3_gate(tmp_path / "b", out2 / "verdict.json", files2).returncode == 3
    assert _run_m3_gate(tmp_path / "b", out2 / "verdict.json", []).returncode == 3
    assert _run_m3_gate(tmp_path / "b", tmp_path / "nowhere/verdict.json", files2).returncode == 3


@pytest.mark.parametrize("script", ["stage", "probe"])
def test_clis_accept_m3_in_preflight(tmp_path, script):
    _stage_module()
    upstream_python = _REPO / "external/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, (_REPO / "src", _REPO / "scripts", upstream_python)))}
    if script == "stage":
        args = [str(STAGE), "--repo", str(_REPO), "--campaign", REPLAY_SOURCE, "--baseline", REPLAY_BASELINE,
                "--out", str(tmp_path / "never"), "--stage-gait", "m3", "--preflight"]
    else:
        args = [str(PROBE), "--repo", str(_REPO), "--out", str(_REPO / "results/never-created-m3-probe"),
                "--stage", "doors", "--indices", "0", "--stage-gait", "m3", "--preflight"]
    out = subprocess.run([sys.executable, *args], cwd=_REPO, capture_output=True, text=True, timeout=300, env=env)
    assert out.returncode == 0, out.stderr
    line = json.loads(out.stdout.strip().splitlines()[-1])
    assert line["status"] == "PREFLIGHT_OK" and line["stage_gait"] == "m3"
    assert line["stage_gait_check"]["turn_vx_mps"] == .30 and "policy_sha256" not in line["stage_gait_check"]
    bad = subprocess.run([sys.executable, *args, "--align-yaw"], cwd=_REPO, capture_output=True, text=True,
                         timeout=300, env=env)
    assert bad.returncode == 2 and "m3 does not compose" in bad.stderr


# ---- the v2 CLI guards -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("crossings, message", [
    ("31:1", "never a scored layout"), ("249:0", "never a scored layout"), ("251:2", "never a scored layout"),
    ("250:0", "fails the drop rule"),                                    # grid label: -90 deg walking, blocked
    ("250:0:45:round:walking", "outside the grid"), ("250:0:90:round", "is not L:d"),
    ("251:1,251:1", "duplicate"), ("", "needs --crossings")])
def test_smoke_specs_refused(crossings, message):
    with pytest.raises(ValueError, match=message):
        v2.parse_smoke_specs(crossings)


def test_smoke_specs_grid_and_explicit_labels():
    specs = v2.parse_smoke_specs("251:1,250:0:90:round:walking,253:1")
    assert specs == [{"layout": 251, "door": 1, "heading_deg": 180, "plate": "round", "entry": "standstill"},
                     {"layout": 250, "door": 0, "heading_deg": 90, "plate": "round", "entry": "walking"},
                     {"layout": 253, "door": 1, "heading_deg": 90, "plate": "square", "entry": "standstill"}]
    assert "SMOKE_CROSSINGS=251:1,250:0:90:round:walking,253:1" in LAUNCHER.read_text()


def test_bench_cli_guards(tmp_path):
    for argv in (["--mode", "scored", "--stage-gait", "m3", "--crossings", "251:1"], ["--mode", "scored"],
                 ["--mode", "smoke", "--stage-gait", "m3", "--crossings", "3:0"]):
        with pytest.raises(SystemExit) as exit_info:
            v2.main(["--repo", str(_REPO), "--out", str(tmp_path / "never"), *argv])
        assert exit_info.value.code == 2
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(SystemExit) as exit_info:
        v2.main(["--repo", str(_REPO), "--out", str(out), "--mode", "smoke", "--stage-gait", "m3", "--crossings", "251:1"])
    assert exit_info.value.code == 2 and not (tmp_path / "never").exists()
    (out / "rescore_m2.json").write_text("{}")
    with pytest.raises(SystemExit) as exit_info:
        v2.main(["--repo", str(_REPO), "--out", str(out), "--mode", "rescore-m2"])
    assert exit_info.value.code == 2


def test_compact_result_and_verdict_on_a_synthetic_v2_crossing(tmp_path):
    layout = generate("train", 8)
    spec = {"layout": 8, "door": 1, **dict(zip(("heading_deg", "plate", "entry"), v1.cell_of(8, 1)))}
    side = v1.plate_side(layout, 1, spec["plate"])
    plate = np.asarray(layout.plate(1, side))
    _, direction = layout.door(1)
    t = np.round(np.arange(1, 251) * .04, 10)
    along = np.where(t < 5., -.30, np.where(t < 7., 0., .40))
    samples = [{"time_s": float(ti), "xy": (plate + np.asarray(direction) * a).tolist(), "tilt": .05}
               for ti, a in zip(t, along)]
    history = [{"time_s": 3.0, "door": 1, "side": side, "phase": "approach"}, {"time_s": 3.0, "door": 1, "phase": "settle"},
               {"time_s": 3.4, "door": 1, "phase": "turn"}, {"time_s": 5.0, "door": 1, "phase": "cross"},
               {"time_s": 7.2, "door": 1, "phase": "turn_back"}, {"time_s": 8.0, "door": 1, "side": side, "phase": "recorded"}]
    row = {"layout_index": 8, "layout_seed": layout.seed, "replay_elapsed_s": samples[-1]["time_s"], "first_fall_s": None,
           "maximum_tilt": .05, "stage_history": history, "samples": samples}
    scored = v2.score_crossing(row, layout, 1, 3.0)
    assert scored["clear"] and scored["clear_after_takeover_s"] == pytest.approx(4.0)
    result = {**spec, "complete": True, "end_reason": "complete", "clear": scored["clear"], "real_clear": True,
              "fell": False, "takeover_s": 3.0, "handback_s": 8.0, "events": [{"event": "takeover", "speed_mps": 0.}],
              "crossing": scored["crossing"], "stage_history": history, "samples": samples, "stage_gait": "m3",
              "spawn": {"contacts": [{"world_geom": "wall_m7_3", "dist_m": -.001}]},
              "m3_events": [{"event": "takeover"}, {"event": "stall_recover"}], "geometry": {},
              "plate_activation": {"target_pressed_during_stage": False}, "wall_contacts": {"stage_samples": 0},
              "clear_after_takeover_s": scored["clear_after_takeover_s"],
              "trace_header": {k: row[k] for k in ("layout_index", "layout_seed", "replay_elapsed_s", "first_fall_s",
                                                   "maximum_tilt")}}
    (tmp_path / "crossings").mkdir()
    v1.write_crossing(tmp_path, result)
    [back] = v1.read_results(tmp_path)
    line = v2.compact_result(back)
    assert line["clear"] and line["watchdog_recoveries"] == 1 and line["spawn_wall_contacts"] == ["wall_m7_3"]
    assert "runup_over_opposite_plate" not in line and line["stage_gait"] == "m3"
    json.dumps(line)
    assert v2.bench_v2_verdict([back], declared=[spec])["verdict"] == "PASS"


# ---- the launcher ---------------------------------------------------------------------------------------------

def _header():
    return LAUNCHER.read_text().split("set -euo pipefail")[0]


def _flat(text):
    return " ".join(line.lstrip("#").strip() for line in text.splitlines())


def test_launcher_header_carries_rule_chain_labels_constants_and_disclosure():
    flat = _flat(_header())
    for clause in (
            FROZEN_RULE,
            "Then the unchanged exact ten-fall replay once with --stage-gait m3, through the existing replay-gate "
            "machinery (complete, 10 episodes, 10 upright, 0 falls; release arm m3-shipped-step).",
            "Then the route gate as coded by M1: Doors and Transport each >= 16/16 successes on validation layouts 0-15 "
            "(32 episodes).",
            "bench v2 was introduced AFTER the M2 bench FAIL (21507962: 8/64 clears, 8 falls) revealed the run-up "
            "confound", "M2's FAIL stands",
            "LEARNED gait", "SCRIPTED stage", "ORACLE layout and plate pose", "never from an exit code",
            "NOT part of the line", "41 bench-v2 crossings (this job) + 10 exact-replay episodes + 32 route-gate episodes",
            "[0.30 m/s, 0, +-0.40 rad/s]", "until |error| < 0.25 rad", "2 pi/0.40 = 15.7 s",
            "[0.30 m/s, 0, clip(1.2 x error, +-0.35)]", "base moved < 0.05 m over the last 1.2 s",
            "0.40 s at -0.30 m/s", "at most 2 recoveries per crossing", "srun 21508399",
            ".15/.20 never stepped, .25 stepped at +0.40 only, .30 stepped both ways",
            "0.039 m above the floor", "3.0 s at zero command", "within 0.13 m of P", "not arrived after 3.0 s",
            "predicted to FAIL", "Falls (tilt >= 0.78) count in EVERY phase after the spawn",
            # review disclosures: the settle and cross bound as polled, M3 not acting at takeover, no scored smoke
            "the declared 0.40 s settle lasts 0.40 or 0.60 s and the 4.0 s cross bound 4.0 or 4.2 s",
            "= 10.0-10.2 s sits at or past the 10 s window at nominal tracking",
            "a walking entry is brought to a standstill before the first M3 command",
            "This was a pre-read of 1 of the 10 replay-gate layouts", "no later smoke runs M3 on a scored layout",
            "it runs the m3 stage on exploration train layouts >= 250 only"):
        assert clause in flat, clause
    assert "0.40 s settle + (pi - 0.25)/0.40 = 7.2 s + 0.65 m / 0.27 m/s = 2.4 s = 10.0 s" not in flat
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def test_launcher_bench_refuses_existing_output_dirty_tree_and_releases_only_m3():
    text = LAUNCHER.read_text()
    bench = text[text.index('if [ "$mode" = bench ]; then'):text.index('[ "$mode" = smoke ] || {')]
    assert '[ ! -e "$OUT" ] || { echo "refusing: $OUT exists' in bench
    assert 'git diff --quiet HEAD -- "${sources[@]}"' in bench and "STAGE_GAIT=m3" in text
    assert "--mode scored --stage-gait \"$STAGE_GAIT\"" in bench
    sources = re.search(r"^  sources=\((.*?)\)\n", text, re.S | re.M)
    assert sources and set(sources.group(1).split()) == ({"scripts/mission7*.py", "src/bhl_robust/mission"}
                                                         | set(v1.BENCH_SOURCES) | {v2.V2_LAUNCHER})


def _release_function():
    match = re.search(r"^release_if_pass\(\) \{\n.*?^\}\n", LAUNCHER.read_text(), re.S | re.M)
    assert match
    return match.group(0)


@pytest.mark.parametrize("verdict, release_rc, released, rc, text", [
    ("PASS", 0, True, 0, "releasing the exact ten-fall replay with --stage-gait m3"),
    ("PASS", 3, True, 1, "RELEASE_BLOCKED"),
    ("PASS", 1, True, 1, "M7 BENCH V2 RELEASE FAILED (exit 1)"),
    ("FAIL", 0, False, 0, "chain stops"),
    ("INCOMPLETE", 0, False, 0, "nothing released")])
def test_launcher_releases_the_m3_replay_only_on_a_pass_read_from_json(tmp_path, verdict, release_rc, released, rc,
                                                                       text):
    out = tmp_path / "bench"
    out.mkdir()
    (out / "verdict.json").write_text(json.dumps({"verdict": verdict}))
    calls = tmp_path / "calls"
    stub = tmp_path / "release.sh"
    stub.write_text(f'#!/bin/bash\necho "$@ M7_BENCH_V2_VERDICT=$M7_BENCH_V2_VERDICT" >> {calls}\nexit {release_rc}\n')
    script = tmp_path / "run.sh"
    script.write_text("set -euo pipefail\n" + _release_function() + f'release_if_pass "{out}"\n')
    env = {**os.environ, "PY": sys.executable, "RELEASE": str(stub), "DRY_RUN": "0"}
    env.pop("M7_BENCH_V2_VERDICT", None)
    run = subprocess.run(["bash", "-c", f'PY="$PY" RELEASE="$RELEASE" DRY_RUN=0; source "{script}"'],
                         env=env, capture_output=True, text=True, timeout=60)
    assert run.returncode == rc, run.stderr
    assert text in run.stdout and calls.exists() == released
    if released:
        assert calls.read_text().strip() == f"--only m3-shipped-step --submit M7_BENCH_V2_VERDICT={out}/verdict.json"


def _smoke_limits_function():
    match = re.search(r"^smoke_limits_ok\(\) \{\n.*?^\}\n", LAUNCHER.read_text(), re.S | re.M)
    assert match
    return match.group(0)


@pytest.mark.parametrize("name, job_id, limit, ok", [
    ("m7-plate-bench-v2-smoke", "123", "1:00:00", True), ("m7-plate-bench-v2-smoke", "123", "59:00", True),
    ("m7-plate-bench-v2-smoke", "123", "1:00:01", False), ("m7-plate-bench-v2-smoke", "123", "1-00:00:00", False),
    ("m7-plate-bench-v2-smoke", "123", "UNLIMITED", False), ("m7-plate-bench-v2-smoke", "", "1:00:00", False),
    ("m7-plate-bench-v2", "123", "1:00:00", False)])
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


def test_launcher_smoke_checks_limits_first_and_covers_identity_rescore_and_m3():
    text = LAUNCHER.read_text()
    smoke = text[text.index('[ "$mode" = smoke ] || {'):]
    assert smoke.index("smoke_limits_ok || exit 2") < smoke.index("SMOKE=${M7_SMOKE_DIR") < smoke.index("mkdir -p")
    assert "--job-name=m7-plate-bench-v2-smoke --time=01:00:00" in text
    for step in ('echo "$REFERENCE_STAGE_SHA256  $SMOKE/reference/mission7_plate_stage.py" | sha256sum -c -',
                 'echo "$M2_STAGE_SHA256  $SMOKE/reference-m2/mission7_plate_stage.py" | sha256sum -c -',
                 'cmp "$SMOKE/turnboth-reference/episodes.json" "$SMOKE/turnboth-new/episodes.json"',
                 'cmp "$SMOKE/v1-reference.L250-d0.json" "$SMOKE/v1-new.L250-d0.json"',
                 '--mode rescore-m2', '--smoke --stage-gait m3', 'stage_gait="m3")'):
        assert step in smoke, step
    # the replay's --smoke layout is validation 0, a scored replay-gate layout: M3 reaches it only as a preflight
    stage_runs = re.findall(r'"\$PY" [^\n]*mission7_plate_stage\.py(?:[^\n]*\\\n)*[^\n]*', smoke)
    m3_runs = [run for run in stage_runs if "--stage-gait m3" in run]
    assert len(stage_runs) == 5 and m3_runs and all("--preflight" in run for run in m3_runs), stage_runs
    assert 'check("replay_m3_preflight"' in smoke and 'check("replay_m3_no_scored_episode"' in smoke
    assert f"REFERENCE_COMMIT={PRE_M2[0]}" in text and f"REFERENCE_STAGE_SHA256={PRE_M2[1]}" in text
    assert f"M2_COMMIT={M2[0]}" in text and f"M2_STAGE_SHA256={M2[1]}" in text
    assert "sys.exit(0 if not problems else 1)" in smoke
