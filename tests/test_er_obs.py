"""ER-OBS-1 (design G): labels at their boundaries, point hits, verdicts at their boundaries (INCOMPLETE included),
the mock pipeline end to end (malformed / refusal / timeout count as wrong), the request content, the API-key guard
(never echoes the key), and both launchers (headers, refusals, real-mode command lines on a fake tree).

No test touches the network: the run-mode transport is replaced by an in-process fake, sockets are disabled for the
mock paths, and no request is ever made to any Google endpoint.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import math
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bhl_robust.eval import er_obs as E                               # noqa: E402
from bhl_robust.eval.er_obs import client as C                        # noqa: E402
from bhl_robust.eval.er_obs import labels as L                        # noqa: E402
from bhl_robust.eval.er_obs import verdict as V                       # noqa: E402

PY = sys.executable
CLI = REPO / "scripts/bench/er_obs.py"
FRAMES_SH = REPO / "slurm/repo20260923/gpu_er_obs1_frames.sbatch"
CALLS_SH = REPO / "slurm/repo20260923/cpu_er_obs1_calls.sbatch"
FAKE_KEY = "AIzaFAKE-test-key_0123456789abcdefghijklmnopq"


def _load_cli():
    spec = importlib.util.spec_from_file_location("er_obs_cli_under_test", CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


# =================================================================== labels: pure kernels at the boundaries

PLINTH = [(-0.125, 0.125, -0.125, 0.125, 0.25)]            # binary-exact numbers


def test_lifted_clear_boundary_is_2cm_inclusive():
    assert L.lifted_clear(0.02) is True
    assert L.lifted_clear(float(np.nextafter(0.02, 0.0))) is False
    assert L.lifted_clear(0.5) is True and L.lifted_clear(-0.01) is False


def test_support_clearance_uses_the_plinth_only_when_the_footprint_strictly_overlaps():
    I = np.eye(3)
    # over the plinth: lowest corner 0.25 + 0.03125 -> clearance vs the plinth top
    c = [0.0, 0.0, 0.25 + 0.125 + 0.03125]
    assert L.support_clearance(c, I, 0.125, PLINTH) == 0.03125
    # beside it, AABB edge exactly touching the plinth edge (x - hx == x1): not under -> the floor only
    c = [0.25, 0.0, 0.125 + 0.0625]
    assert L.support_clearance(c, I, 0.125, PLINTH) == 0.0625
    # overlapping by any amount -> the plinth top counts (the cube is below it: negative clearance)
    c = [float(np.nextafter(0.25, 0.0)), 0.0, 0.125 + 0.0625]
    assert L.support_clearance(c, I, 0.125, PLINTH) < 0
    # a 45-deg roll lowers the lowest corner by half * (sqrt 2 - 1)
    R = rot_x(45.0)
    c = [2.0, 2.0, 1.0]
    assert L.lowest_corner_z(c, R, 0.14) == pytest.approx(1.0 - 0.14 * math.sqrt(2))
    assert L.support_clearance(c, R, 0.14, PLINTH) == pytest.approx(1.0 - 0.14 * math.sqrt(2))


def test_lifted_clear_on_a_hovering_cube_2cm_up():
    I = np.eye(3)
    above = L.support_clearance([0.0, 0.0, 0.25 + 0.125 + 0.03], I, 0.125, PLINTH)
    below = L.support_clearance([0.0, 0.0, 0.25 + 0.125 + 0.01], I, 0.125, PLINTH)
    assert L.lifted_clear(above) and not L.lifted_clear(below)


def test_face_tilt_reads_any_face_down_and_body_z_does_not():
    assert L.face_tilt_deg(np.eye(3)) == 0.0 and L.body_z_tilt_deg(np.eye(3)) == 0.0
    for R in (rot_x(90), rot_y(90), rot_x(180), rot_y(-90) @ rot_x(90)):
        assert L.face_tilt_deg(R) == pytest.approx(0.0, abs=1e-6)
    assert L.body_z_tilt_deg(rot_x(90)) == pytest.approx(90.0)
    assert L.body_z_tilt_deg(rot_x(180)) == pytest.approx(180.0)
    assert L.face_tilt_deg(rot_x(30)) == pytest.approx(30.0)
    assert L.face_tilt_deg(rot_x(60)) == pytest.approx(30.0)          # the -y face is 30 deg from down
    assert L.face_tilt_deg(rot_x(45)) == pytest.approx(45.0)


def test_seated_flat_boundaries_8deg_footprint_and_contact():
    assert L.seated_flat(True, (0.0, 0.0), 8.0) is True
    assert L.seated_flat(True, (0.0, 0.0), float(np.nextafter(8.0, 9.0))) is False
    assert L.seated_flat(True, (0.09, -0.09), 0.0) is True
    assert L.seated_flat(True, (float(np.nextafter(0.09, 1.0)), 0.0), 0.0) is False
    assert L.seated_flat(False, (0.0, 0.0), 0.0) is False
    # with real rotations, away from the boundary
    assert L.seated_flat(True, (0.0, 0.0), L.face_tilt_deg(rot_x(7.9))) is True
    assert L.seated_flat(True, (0.0, 0.0), L.face_tilt_deg(rot_x(8.1))) is False


def test_any_face_down_seats_while_strict_body_z_disagrees():
    for R in (rot_x(90), rot_y(90), rot_x(180)):
        assert L.seated_flat(True, (0, 0), L.face_tilt_deg(R)) is True
        assert L.seated_flat(True, (0, 0), L.body_z_tilt_deg(R)) is False
    assert L.seated_flat(True, (0, 0), L.body_z_tilt_deg(rot_x(7.9))) is True


def test_robot_contact_is_strictly_above_1N():
    assert L.robot_contact(1.0) is False
    assert L.robot_contact(float(np.nextafter(1.0, 2.0))) is True
    assert L.robot_contact(0.0) is False


def test_c2_status_quo_boundaries():
    assert L.c2_lifted(0.05) is True and L.c2_lifted(float(np.nextafter(0.05, 0.0))) is False
    assert L.c2_placed(0.03, (0, 0)) and L.c2_placed(-0.03, (0.09, 0.09))
    assert not L.c2_placed(float(np.nextafter(0.03, 1.0)), (0, 0))
    assert not L.c2_placed(0.0, (0.0, float(np.nextafter(0.09, 1.0))))
    # the status quo has no orientation clause: a rolled but height-matched cube is "placed"
    assert L.c2_placed(0.0, (0, 0)) is True


# =================================================================== labels: read-only MuJoCo extraction

WORLD = """<mujoco><option timestep="0.002"/>
<worldbody>
  <geom name="floor" type="plane" size="0 0 0.05"/>
  <geom name="plinth" type="box" pos="0 0.1 0.095" size="0.09 0.09 0.095"/>
  <body name="cube" pos="0 0.1 0.33"><freejoint name="cube_free"/>
    <geom name="cube_g" type="box" size="0.14 0.14 0.14" mass="0.5"/></body>
  <body name="r0_pad" pos="3 3 0.5"><freejoint name="pad_free"/>
    <geom name="r0_pad_g" type="box" size="0.04 0.04 0.02" mass="{mass}"/></body>
</worldbody></mujoco>"""


def _world(pad_mass=0.3):
    mujoco = pytest.importorskip("mujoco")
    from bhl_robust.eval import scripted_carry as sc
    m = mujoco.MjModel.from_xml_string(WORLD.format(mass=pad_mass))
    d = mujoco.MjData(m)
    gid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("floor", "plinth", "cube_g", "r0_pad_g")}
    owner = np.full(m.ngeom, -1)
    owner[gid["r0_pad_g"]] = 0                                       # robot 0 = the pair's robot b
    pair = SimpleNamespace(cube_body=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube"),
                           cube_geom=gid["cube_g"], plinth_geom=gid["plinth"], robot_b=0, robot_a=1)
    runner = SimpleNamespace(m=m, d=d, owner=owner)
    runner.cube_forces = lambda pr: sc.CarryRunner.cube_forces(runner, pr)    # the harness's own force code
    return mujoco, m, d, gid, pair, runner


def _place(mujoco, m, d, cube_pos, cube_quat=(1, 0, 0, 0), pad_pos=(3, 3, 0.5)):
    d.qpos[:3], d.qpos[3:7] = cube_pos, cube_quat
    d.qpos[7:10], d.qpos[10:14] = pad_pos, (1, 0, 0, 0)
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)                                         # the test's own data, not a simulation's


def test_frame_state_on_a_synthetic_world_reads_each_label():
    mujoco, m, d, gid, pair, runner = _world()
    # resting flat on the plinth (1 mm penetration)
    _place(mujoco, m, d, (0, 0.1, 0.19 + 0.14 - 0.001))
    s = L.frame_state(m, d, runner, pair, gid["floor"])
    assert s["labels"] == {"lifted_clear": False, "on_floor": False, "robot_contact": False, "seated_flat": True}
    assert s["state"]["plinth_contact"] and not s["body_z_disagrees"]
    # hovering 5 cm above the plinth: lifted clear, not seated
    _place(mujoco, m, d, (0, 0.1, 0.19 + 0.14 + 0.05))
    s = L.frame_state(m, d, runner, pair, gid["floor"])
    assert s["labels"] == {"lifted_clear": True, "on_floor": False, "robot_contact": False, "seated_flat": False}
    assert s["state"]["support_clearance_m"] == pytest.approx(0.05, abs=1e-6)
    # on the floor beside the plinth
    _place(mujoco, m, d, (0.6, 0.1, 0.14 - 0.001))
    s = L.frame_state(m, d, runner, pair, gid["floor"])
    assert s["labels"] == {"lifted_clear": False, "on_floor": True, "robot_contact": False, "seated_flat": False}
    # rolled 90 deg about x, resting on the plinth: any face down seats; strict body-z disagrees
    q = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0)
    _place(mujoco, m, d, (0, 0.1, 0.19 + 0.14 - 0.001), q)
    s = L.frame_state(m, d, runner, pair, gid["floor"])
    assert s["labels"]["seated_flat"] is True and s["seated_flat_strict_body_z"] is False and s["body_z_disagrees"]


@pytest.mark.parametrize("mass,expected", [(0.3, True), (0.03, False)])
def test_frame_state_robot_contact_uses_the_harness_normal_force(mass, expected):
    mujoco, m, d, gid, pair, runner = _world(pad_mass=mass)
    top = 0.19 + 0.28 - 0.001
    _place(mujoco, m, d, (0, 0.1, 0.19 + 0.14 - 0.001), pad_pos=(0, 0.1, top + 0.02 - 0.0005))
    s = L.frame_state(m, d, runner, pair, gid["floor"])
    f = s["state"]["robot_normal_force_n"]
    assert s["labels"]["robot_contact"] is expected, f
    assert s["state"]["robot_contact_geoms"] == 1
    assert (f > 1.0) is expected


def test_frame_state_never_writes_the_simulation_data():
    mujoco, m, d, gid, pair, runner = _world()
    _place(mujoco, m, d, (0, 0.1, 0.19 + 0.14 - 0.001), pad_pos=(0, 0.1, 0.19 + 0.28 + 0.0195))
    snap = {k: np.array(getattr(d, k)).copy() for k in ("qpos", "qvel", "qacc", "qacc_warmstart", "efc_force",
                                                         "xpos", "xmat", "geom_xpos")}
    ncon, time0 = int(d.ncon), float(d.time)
    L.frame_state(m, d, runner, pair, gid["floor"])
    for k, v in snap.items():
        assert np.array_equal(np.array(getattr(d, k)), v), k
    assert (int(d.ncon), float(d.time)) == (ncon, time0)


# =================================================================== C1 gates

def test_c1_gates_exact_scene_exclusion_and_the_noise_bound():
    pytest.importorskip("mujoco")
    from bhl_robust.eval.er_obs.render import c1_ok
    assert E.C1_NOISE_MAX_LEVELS == 2 and E.C1_NOISE_MAX_PX == 100
    assert c1_ok(0, 0, 0) and c1_ok(0, 2, 100)
    assert not c1_ok(1, 0, 0)            # the cube geom is still in the C1 scene
    assert not c1_ok(0, 3, 1)            # beyond 2 levels: a shadow or body leak, not rasterization noise
    assert not c1_ok(0, 1, 101)          # beyond 100 px


# =================================================================== point hit

def test_point_hit_maps_normalized_yx_to_the_pixel_and_needs_a_cube_pixel():
    mask = np.zeros((540, 960), dtype=bool)
    mask[270, 480] = True
    centre = [(270.5) * 1000 / 540, (480.5) * 1000 / 960]
    assert V.point_hit(centre, mask) is True
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        assert V.point_hit([(270.5 + dy) * 1000 / 540, (480.5 + dx) * 1000 / 960], mask) is False
    assert V.point_pixel([500.0, 500.0], 540, 960) == (270, 480)         # exact pixel boundary -> floor
    assert V.point_pixel([1000, 1000], 540, 960) == (539, 959)           # clipped to the last pixel
    assert V.point_pixel([0, 0], 540, 960) == (0, 0)
    assert V.point_hit(None, mask) is False
    full = np.ones((540, 960), dtype=bool)
    assert V.point_hit([1000, 1000], full) and V.point_hit([0, 0], full)


# =================================================================== verdict at its boundaries

def _frames(n=400, n_pos=100, visible=400):
    fr = []
    for i in range(n):
        labels = {b: ((i + 37 * j) % n) < n_pos for j, b in enumerate(E.BOOLEANS)}
        fr.append({"id": f"f{i:03d}", "labels": labels, "cube_visible": i < visible,
                   "c2": {"lifted_centre": labels["lifted_clear"], "placed_no_orientation": True},
                   "body_z_disagrees": i % 50 == 0})
    return fr


def _ok(ans):
    """An ok outcome as client.classify records it: the full answer plus the four booleans parsed on their own."""
    return {"kind": "ok", "answer": {**ans, "cube_point": [1.0, 1.0]}, "booleans": {b: ans[b] for b in E.BOOLEANS}}


def _outcomes(frames, main=None, c1=None):
    """main/c1: functions (frame, boolean) -> predicted bool; default main = label, c1 = BA exactly 0.60."""
    main = main or (lambda f, b: f["labels"][b])
    out = {}
    for f in frames:
        out[(f["id"], "main")] = _ok({b: main(f, b) for b in E.BOOLEANS})
    c1 = c1 or _c1_at(60, 180)
    for f in frames:
        out[(f["id"], "c1")] = _ok({b: c1(f, b) for b in E.BOOLEANS})
    return out


def _c1_at(tp, tn):
    """C1 answers with exactly `tp` true positives and `tn` true negatives per boolean (fresh counters per call)."""
    seen = {}

    def pred(f, b):
        y = f["labels"][b]
        k = seen.get((b, y), 0)
        seen[(b, y)] = k + 1
        return (k < tp) if y else not (k < tn)
    return pred


def _verdict(frames, outcomes, hits=None, expected=400):
    hits = {f["id"]: True for f in frames if f["cube_visible"]} if hits is None else hits
    return V.accuracy_verdict(frames, outcomes, hits, expected_frames=expected)


def test_accuracy_pass_at_every_boundary():
    fr = _frames()
    v = _verdict(fr, _outcomes(fr))
    assert v["verdict"] == "PASS", v["checks"]
    assert all(v["c1_balanced_accuracy"][b]["ba"] == 0.6 for b in E.BOOLEANS)
    assert all(v["main_balanced_accuracy"][b]["ba"] == 1.0 for b in E.BOOLEANS)


def test_main_ba_exactly_090_passes_and_just_below_fails():
    fr = _frames()

    def at(tp):
        seen = {}

        def main(f, b):
            if b != "robot_contact" or not f["labels"][b]:
                return f["labels"][b]
            seen[b] = seen.get(b, 0) + 1
            return seen[b] <= tp
        return main
    v = _verdict(fr, _outcomes(fr, main=at(80)))
    assert v["main_balanced_accuracy"]["robot_contact"]["ba"] == 0.9 and v["verdict"] == "PASS"
    v = _verdict(fr, _outcomes(fr, main=at(79)))
    assert v["main_balanced_accuracy"]["robot_contact"]["ba"] < 0.9 and v["verdict"] == "NEGATIVE"
    assert v["checks"]["main_ba_all_ge_0.90"] is False


def test_point_hit_rate_exactly_095_passes_and_just_below_fails():
    fr = _frames()
    out = _outcomes(fr)
    hits = {f["id"]: i < 380 for i, f in enumerate(fr)}
    v = _verdict(fr, out, hits)
    assert v["point"]["hit_rate"] == 0.95 and v["verdict"] == "PASS"
    hits = {f["id"]: i < 379 for i, f in enumerate(fr)}
    v = _verdict(fr, out, hits)
    assert v["verdict"] == "NEGATIVE" and not v["checks"]["point_hit_rate_ge_0.95"]


def test_c1_above_060_is_negative():
    fr = _frames()
    v = _verdict(fr, _outcomes(fr, c1=_c1_at(61, 180)))
    assert v["c1_balanced_accuracy"]["lifted_clear"]["ba"] > 0.6 and v["verdict"] == "NEGATIVE"
    assert v["checks"]["c1_ba_all_le_0.60"] is False


def test_incomplete_on_a_class_below_20_and_takes_precedence():
    fr = _frames(n_pos=19)
    v = _verdict(fr, _outcomes(fr, c1=lambda f, b: False))
    assert v["verdict"] == "INCOMPLETE" and "lifted_clear=true" in v["incomplete_reasons"][0]
    fr = _frames(n_pos=20)
    v = _verdict(fr, _outcomes(fr, c1=lambda f, b: False))
    assert v["verdict"] == "PASS" and not v["incomplete_reasons"]
    # a NEGATIVE run is INCOMPLETE too when a class is short (precedence)
    fr = _frames(n_pos=19)
    v = _verdict(fr, _outcomes(fr, main=lambda f, b: True))
    assert v["verdict"] == "INCOMPLETE"


def test_incomplete_on_a_missing_outcome_or_frame_count():
    fr = _frames()
    out = _outcomes(fr)
    out.pop((fr[5]["id"], "c1"))
    v = _verdict(fr, out)
    assert v["verdict"] == "INCOMPLETE" and v["n_missing_outcomes"] == 1
    v = _verdict(fr[:399], _outcomes(fr[:399]), expected=400)
    assert v["verdict"] == "INCOMPLETE" and "399 frames" in v["incomplete_reasons"][0]


@pytest.mark.parametrize("kind", ["timeout", "non_json", "refusal", "schema_violation", "http_error",
                                  "transport_error", "bad_finish"])
def test_failed_calls_count_as_wrong_on_every_boolean_and_the_point(kind):
    fr = _frames()
    out = _outcomes(fr)
    bad = [f["id"] for f in fr[:30]]
    for i in bad:
        out[(i, "main")] = {"kind": kind}
    hits = {f["id"]: f["id"] not in bad for f in fr}                 # a failed call has no point: a miss
    v = _verdict(fr, out, hits)
    for b in E.BOOLEANS:
        m = v["main_balanced_accuracy"][b]
        wrong_pos = sum(1 for f in fr[:30] if f["labels"][b])
        wrong_neg = 30 - wrong_pos
        assert (m["fn"], m["fp"]) == (wrong_pos, wrong_neg)
    assert v["point"]["hits"] == 370
    assert v["outcome_kinds"]["main"][kind] == 30


def test_balanced_accuracy_counts_none_as_wrong():
    r = V.balanced_accuracy([True, True, False, False], [True, None, False, None])
    assert (r["tp"], r["fn"], r["tn"], r["fp"]) == (1, 1, 1, 1) and r["ba"] == 0.5
    assert V.balanced_accuracy([True, True], [True, True])["ba"] is None


def test_latency_p95_nearest_rank_boundaries():
    def recs(vals):
        return [{"kind": "ok", "latency_s": v} if v is not None else {"kind": "timeout", "latency_s": None}
                for v in vals]
    assert V.latency_verdict(recs([0.5] * 47 + [1.0, 5.0, 5.0]))["verdict"] == "IN-LOOP-ELIGIBLE"
    v = V.latency_verdict(recs([0.5] * 47 + [1.0001, 1.0001, 1.0001]))
    assert v["verdict"] == "NOT-IN-LOOP-ELIGIBLE" and v["p95_nearest_rank_s"] == 1.0001
    v = V.latency_verdict(recs([0.5] * 48 + [None, None]))             # 2 failures sit at ranks 49-50
    assert v["verdict"] == "IN-LOOP-ELIGIBLE" and v["failed_calls"] == 2
    v = V.latency_verdict(recs([0.5] * 47 + [None, None, None]))       # the 48th is +inf
    assert v["verdict"] == "NOT-IN-LOOP-ELIGIBLE" and v["p95_is_inf"]
    assert V.latency_verdict(recs([0.1] * 49))["verdict"] == "INCOMPLETE"
    assert V.p95_nearest_rank(list(range(1, 51))) == 48


def test_verdicts_are_written_once(tmp_path):
    p = tmp_path / "v.json"
    V.write_once(p, {"verdict": "PASS"})
    with pytest.raises(FileExistsError):
        V.write_once(p, {"verdict": "NEGATIVE"})
    assert json.loads(p.read_text())["verdict"] == "PASS"


# =================================================================== response classification

def _env(text=None, **extra):
    """A response in the documented interactions shape (the answer in output_text), as client._interaction builds it."""
    resp = {"id": "i-1", "model": C.MODEL, "status": "completed",
            "usage": {"total_input_tokens": 1290, "total_output_tokens": 61, "total_thought_tokens": 37}}
    if text is not None:
        resp["output_text"] = text
    resp.update(extra)
    return json.dumps(resp).encode()


GOOD = '{"cube_point": [512, 300], "lifted_clear": true, "on_floor": false, "robot_contact": true, "seated_flat": false}'


def test_classify_ok_and_every_failure_kind():
    r = C.classify(200, _env(GOOD))
    assert r["kind"] == "ok" and r["answer"]["cube_point"] == [512.0, 300.0] and r["answer"]["robot_contact"]
    assert r["booleans"] == {"lifted_clear": True, "on_floor": False, "robot_contact": True, "seated_flat": False}
    assert r["thought_tokens"] == [["usage.total_thought_tokens", 37]]
    assert C.classify(200, _env("```json\n" + GOOD + "\n```"))["kind"] == "ok"           # one fenced block
    assert C.classify(200, _env('{"cube_point": [1, 2], "lifted_clear": tru'))["kind"] == "non_json"
    assert C.classify(200, _env("Sure! Here it is: " + GOOD))["kind"] == "non_json"
    assert C.classify(200, _env("I'm sorry, I can't help with that."))["kind"] == "non_json"   # a refusal in prose
    assert C.classify(200, _env(None))["kind"] == "no_answer"                                # no output_text
    assert C.classify(200, _env(None, output_text=["x"]))["kind"] == "bad_envelope"         # not a string
    assert C.classify(200, json.dumps([1, 2]).encode())["kind"] == "bad_envelope"
    assert C.classify(200, b"<html>")["kind"] == "bad_envelope"
    assert C.classify(500, b'{"error": {"message": "x"}}')["kind"] == "http_error"
    assert C.classify(429, b"not json")["kind"] == "http_error"
    for kind_body in (_env(None), b"<html>", b'{"error": {"message": "x"}}'):
        assert C.classify(200, kind_body)["booleans"] is None


@pytest.mark.parametrize("bad", [
    '{"cube_point": [true, 3], "lifted_clear": true, "on_floor": false, "robot_contact": true, "seated_flat": false}',
    '{"cube_point": [1, 2, 3], "lifted_clear": true, "on_floor": false, "robot_contact": true, "seated_flat": false}',
    '{"cube_point": [1000.5, 3], "lifted_clear": true, "on_floor": false, "robot_contact": true, "seated_flat": false}',
    '{"cube_point": [-1, 3], "lifted_clear": true, "on_floor": false, "robot_contact": true, "seated_flat": false}',
    '{"cube_point": ["500", 3], "lifted_clear": true, "on_floor": false, "robot_contact": true, "seated_flat": false}',
    '{"cube_point": [5, 3], "lifted_clear": "true", "on_floor": false, "robot_contact": true, "seated_flat": false}',
    '{"cube_point": [5, 3], "lifted_clear": true, "on_floor": false, "robot_contact": true}',
    '[1, 2]',
])
def test_schema_violations_are_wrong(bad):
    assert C.classify(200, _env(bad))["kind"] == "schema_violation"


def test_c1_booleans_survive_a_bad_point_but_not_a_bad_boolean():
    """Fix 1 (2026-10-03 review): C1 booleans are parsed on their own, whatever cube_point holds."""
    no_cube = '{"cube_point": [-1, -1], "lifted_clear": false, "on_floor": true, "robot_contact": false, "seated_flat": false}'
    r = C.classify(200, _env(no_cube))
    assert r["kind"] == "schema_violation" and r["booleans"] == {"lifted_clear": False, "on_floor": True,
                                                                 "robot_contact": False, "seated_flat": False}
    assert V.pred_c1(r, "on_floor") is True and V.pred_main(r, "on_floor") is None
    r = C.classify(200, _env('{"cube_point": [5, 3], "lifted_clear": "true", "on_floor": false, '
                             '"robot_contact": true, "seated_flat": false}'))
    assert r["booleans"] is None and V.pred_c1(r, "lifted_clear") is None


def test_c1_failed_calls_are_scored_correct_and_main_failed_calls_wrong():
    """Fix 1: failures can only make C1 HARDER to hold and a main PASS harder to reach."""
    y = [True] * 30 + [False] * 30
    assert V.balanced_accuracy(y, [None] * 60, failed="correct")["ba"] == 1.0
    assert V.balanced_accuracy(y, [None] * 60, failed="wrong")["ba"] == 0.0
    with pytest.raises(ValueError):
        V.balanced_accuracy(y, [None] * 60, failed="skip")
    fr = _frames()
    out = _outcomes(fr)
    for f in fr:                                  # every C1 call failed: C1 cannot hold
        out[(f["id"], "c1")] = {"kind": "timeout", "booleans": None}
    v = _verdict(fr, out)
    assert v["verdict"] == "NEGATIVE" and v["checks"]["c1_ba_all_le_0.60"] is False
    assert all(v["c1_balanced_accuracy"][b]["ba"] == 1.0 for b in E.BOOLEANS)
    assert v["c1_failures"]["failed"] == 400 and v["c1_failures"]["rate"] == 1.0
    assert v["c1_failures"]["failed_kinds"] == {"timeout": 400}


def test_retry_hint_from_retry_after_and_retry_info():
    body = C._err(429, "RESOURCE_EXHAUSTED", "q", retry_delay="30s")
    assert C.retry_hint_s({"retry_after": "2"}, body) == 30.0
    assert C.retry_hint_s({"retry_after": "45"}, C._err(429, "R", "q", retry_delay="1.5s")) == 45.0
    assert C.retry_hint_s({}, b"not json") is None and C.retry_hint_s(None, None) is None
    assert C.retry_hint_s({"retry_after": "Wed, 21 Oct 2015 07:28:10 GMT"}, None, wall=lambda: 1445412480.0) == 10.0


def test_call_once_timeouts_and_transport_errors():
    def raise_timeout(_body):
        raise C.TransportTimeout("t")

    def raise_conn(_body):
        raise C.TransportConnectionError("reset")

    def raise_err(_body):
        raise C.TransportError("boom")
    r = C.call_once(raise_timeout, b"png")
    assert r["kind"] == "timeout" and r["latency_s"] is None and r["retryable"] is False
    r = C.call_once(raise_conn, b"png")
    assert r["kind"] == "connection_error" and r["retryable"] is True
    assert C.call_once(raise_err, b"png")["kind"] == "transport_error"
    ticks = iter([0.0, 61.0])
    r = C.call_once(lambda b: (200, _env(GOOD), {}), b"png", clock=lambda: next(ticks))
    assert r["kind"] == "timeout" and r["late_result_kind"] == "ok" and r["retryable"] is False
    ticks = iter([0.0, 0.42])
    r = C.call_once(lambda b: (200, _env(GOOD), {}), b"png", clock=lambda: next(ticks))
    assert r["kind"] == "ok" and r["latency_s"] == 0.42 and r["retryable"] is False
    for code in (429, 503):
        ticks = iter([0.0, 0.1])
        r = C.call_once(lambda b, c=code: (c, C._err(c, "X", "busy"), {"retry_after": "3"}), b"png",
                        clock=lambda: next(ticks))
        assert r["kind"] == "http_error" and r["retryable"] is True and r["retry_hint_s"] == 3.0
    ticks = iter([0.0, 0.1])
    r = C.call_once(lambda b: (500, C._err(500, "INTERNAL", "x"), {}), b"png", clock=lambda: next(ticks))
    assert r["kind"] == "http_error" and r["retryable"] is False


def test_pacer_keeps_the_minimum_gap_and_honours_longer_waits():
    vc = C.VirtualClock()
    pacer = C.Pacer(clock=vc, sleep=vc.sleep)
    assert pacer.before_send() is None
    pacer.after()
    vc.advance(0.3)
    assert pacer.before_send() == C.MIN_GAP_S                 # slept 0.7 s more
    pacer.after()
    vc.advance(5.0)
    assert pacer.before_send() == 5.0                         # already past the gap: no sleep
    pacer.after()
    assert pacer.before_send(wait_s=12.0) == 12.0             # a requested wait longer than the gap
    assert C.MIN_GAP_S == 1.0 and C.MAX_RESENDS == 3 and C.RETRYABLE_HTTP == {429, 503}


def _scripted(responses, clock):
    """A transport returning the scripted (status, body, meta) or raising the scripted exception, in order."""
    it = iter(responses)

    def transport(_body):
        clock.advance(0.2)
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r
    return transport


def test_call_policy_resends_only_after_429_503_or_connection_errors():
    vc = C.VirtualClock()
    ok = (200, _env(GOOD), {})
    res = C.call_with_policy(_scripted([(429, C._err(429, "R", "q"), {"retry_after": "2"}),
                                        C.TransportConnectionError("reset"), (503, C._err(503, "U", "busy"), {}), ok],
                                       vc), b"png", C.Pacer(clock=vc, sleep=vc.sleep))
    assert res["final"]["kind"] == "ok" and res["stop"] is None and len(res["attempts"]) == 4
    assert [a["kind"] for a in res["attempts"]] == ["http_error", "connection_error", "http_error", "ok"]
    assert [a["waited_for_s"] for a in res["attempts"]] == [None, 2.0, C.BACKOFF_S[1], C.BACKOFF_S[2]]
    assert all(a["gap_before_s"] is None or a["gap_before_s"] >= C.MIN_GAP_S for a in res["attempts"])
    # every other outcome is final at its first attempt, failures included
    for first in ((500, C._err(500, "I", "x"), {}), (400, C._err(400, "I", "x"), {}), (200, _env("prose"), {}),
                  C.TransportTimeout("t"), C.TransportError("boom")):
        vc = C.VirtualClock()
        res = C.call_with_policy(_scripted([first, ok], vc), b"png", C.Pacer(clock=vc, sleep=vc.sleep))
        assert len(res["attempts"]) == 1 and res["final"]["kind"] != "ok" and res["stop"] is None
    # re-sends used up, or a wait over MAX_WAIT_S: no final outcome, the caller stops (resume later)
    vc = C.VirtualClock()
    busy = (429, C._err(429, "R", "q"), {"retry_after": "1"})
    res = C.call_with_policy(_scripted([busy] * 4, vc), b"png", C.Pacer(clock=vc, sleep=vc.sleep))
    assert res["final"] is None and len(res["attempts"]) == C.MAX_RESENDS + 1 and "re-sends used up" in res["stop"]
    vc = C.VirtualClock()
    res = C.call_with_policy(_scripted([(429, C._err(429, "R", "q", retry_delay="600s"), {})], vc), b"png",
                             C.Pacer(clock=vc, sleep=vc.sleep))
    assert res["final"] is None and len(res["attempts"]) == 1 and "600 s wait" in res["stop"]


def test_request_carries_only_the_frame_and_the_fixed_prompt():
    png = b"\x89PNG-fake-bytes"
    req = C.build_request(png)
    assert set(req) == {"model", "input", "generation_config"} and req["model"] == "gemini-robotics-er-2-preview"
    parts = req["input"]["parts"]
    assert set(req["input"]) == {"parts"} and len(parts) == 2
    assert base64.b64decode(parts[0]["inlineData"]["data"]) == png and parts[0]["inlineData"]["mimeType"] == "image/png"
    assert parts[1] == {"text": C.TEXT} and C.TEXT.startswith(C.PROMPT) and C.SCHEMA_LINE in C.TEXT
    assert req["generation_config"] == C.GENERATION_CONFIG == {"thinking_config": {"thinking_level": "low"}}
    assert C.ENDPOINT == "https://generativelanguage.googleapis.com/v1beta/interactions"
    assert C.PROMPT_SCHEMA_SHA256 == C.sha256_hex(C.canonical({"prompt_text": C.TEXT, "json_schema": C.SCHEMA}))
    assert C.REQUEST_SHA256 == C.sha256_hex(C.canonical(C.request_template()))
    tmpl = json.dumps(C.request_template())
    assert "<from the key file; never stored>" in tmpl and FAKE_KEY not in tmpl
    for word in ("seed", "/nfs", "sanchej7", "bhl", "repo", "oracle", "ORACLE"):
        assert word not in C.TEXT


# =================================================================== API-key guard

def _key(tmp_path, mode=0o600, text=FAKE_KEY, name="key"):
    p = tmp_path / name
    p.write_text(text + "\n")
    os.chmod(p, mode)
    return p


def test_key_guard_refuses_missing_relative_wrong_mode_symlink_and_inside_repo(tmp_path):
    with pytest.raises(C.KeyRefused, match="missing"):
        C.check_key_file(tmp_path / "nope", [REPO])
    with pytest.raises(C.KeyRefused, match="absolute"):
        C.check_key_file(Path("relative/key"), [REPO])
    p = _key(tmp_path, 0o644)
    with pytest.raises(C.KeyRefused, match="644") as e:
        C.check_key_file(p, [REPO])
    assert FAKE_KEY not in str(e.value)
    p = _key(tmp_path, 0o640, name="k2")
    with pytest.raises(C.KeyRefused, match="640"):
        C.check_key_file(p, [REPO])
    ok = _key(tmp_path, name="k3")
    link = tmp_path / "link"
    link.symlink_to(ok)
    with pytest.raises(C.KeyRefused, match="symlink"):
        C.check_key_file(link, [REPO])
    fake_repo = tmp_path / "repo"
    (fake_repo / "sub").mkdir(parents=True)
    inside = _key(fake_repo / "sub", name="key")
    with pytest.raises(C.KeyRefused, match="inside") as e:
        C.check_key_file(inside, [REPO, fake_repo])
    assert FAKE_KEY not in str(e.value)
    assert C.check_key_file(ok, [REPO, fake_repo]) == ok.resolve()
    assert C.read_key(ok) == FAKE_KEY


def test_key_guard_never_echoes_bad_content(tmp_path):
    p = _key(tmp_path, text="two tokens here")
    with pytest.raises(C.KeyRefused) as e:
        C.read_key(p)
    assert "two" not in str(e.value) and "tokens" not in str(e.value)
    assert C.redact(f"bad key {FAKE_KEY} rejected", FAKE_KEY) == "bad key [REDACTED] rejected"


def test_key_path_default_and_env(monkeypatch):
    monkeypatch.delenv(C.KEY_ENV, raising=False)
    assert C.key_file_path() == Path(os.path.expanduser("~/.config/bhl/gemini_api_key"))
    assert C.key_file_path({C.KEY_ENV: "/x/y"}) == Path("/x/y")


def test_http_transport_repr_hides_the_key():
    t = C.HttpTransport(FAKE_KEY)
    assert FAKE_KEY not in repr(t) and FAKE_KEY not in str(t)


# =================================================================== synthetic frames directories

def _png(arr):
    from bhl_robust.eval.er_obs.render import png_bytes
    return png_bytes(arr)


def make_frames_dir(root: Path, mode: str, seeds, frames_per_ep, hw=(54, 96), visible_every=1):
    """A FRAMES_OK manifest with tiny distinct PNGs (no rendering), labels cycling so every class occurs."""
    from bhl_robust.eval.er_obs.render import png_chunks, sha256_bytes
    root.mkdir(parents=True)
    rng = np.random.default_rng(len(seeds) * 1000 + frames_per_ep)
    frames = []
    for seed in seeds:
        for k in range(1, frames_per_ep + 1):
            fid = f"s{seed}_t{k:02d}"
            labels = {b: bool((k + j) % (2 + j) == 0) for j, b in enumerate(E.BOOLEANS)}
            mask = np.zeros(hw, dtype=bool)
            vis = (k % visible_every) == 0
            r0, r1, c0, c1 = hw[0] // 4, hw[0] // 2, hw[1] // 4, hw[1] // 2
            if vis:
                mask[r0:r1, c0:c1] = True
            rec = {"id": fid, "seed": seed, "k": k, "labels": labels, "cube_visible": vis,
                   "cube_pixels": int(mask.sum()), "mask_interior_rc": [(r0 + r1) // 2, (c0 + c1) // 2] if vis else None,
                   "c2": {"lifted_centre": labels["lifted_clear"], "placed_no_orientation": labels["seated_flat"]},
                   "body_z_disagrees": False, "files": {}, "sha256": {}}
            for name, arr in (("rgb", rng.integers(0, 255, hw + (3,), dtype=np.uint8)),
                              ("c1", rng.integers(0, 255, hw + (3,), dtype=np.uint8)),
                              ("cube_mask", mask.astype(np.uint8) * 255)):
                b = _png(arr)
                assert set(png_chunks(b)) <= {"IHDR", "IDAT", "IEND"}
                rel = f"frames/s{seed}/t{k:02d}_{name}.png"
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_bytes(b)
                rec["files"][name], rec["sha256"][name] = rel, sha256_bytes(b)
            frames.append(rec)
    man = {"schema": "er_obs1_frames_v1", "check": "FRAMES_OK", "mode": mode, "seeds": list(seeds),
           "episode_s": 20.0 if mode == "run" else 14.0, "expected_frames": len(frames),
           "image": {"height": hw[0], "width": hw[1]}, "frames": frames, "label": E.LABEL,
           "summary": {"classes_below_20": V.short_classes(V.class_counts(frames))}}
    (root / "manifest.json").write_text(json.dumps(man))
    return root


def _clean_env():
    """The caller's environment without Slurm job context or any ER-OBS-1 / key setting (tests are hermetic even
    inside a launcher's own smoke job, which exports ER_OBS_* variables)."""
    return {k: v for k, v in os.environ.items()
            if not k.startswith(("SLURM_", "ER_OBS_")) and k != C.KEY_ENV}


def _run_cli(*args, env_extra=None, check=True):
    env = _clean_env()
    env["PYTHONPATH"] = str(REPO / "src")
    env[C.KEY_ENV] = "/nonexistent/er-obs1-test-key"                 # mock must never need it
    env.update(env_extra or {})
    r = subprocess.run([PY, str(CLI), *map(str, args)], capture_output=True, text=True, env=env, timeout=600)
    if check:
        assert r.returncode == 0, r.stdout + r.stderr
    return r


# =================================================================== mock pipeline end to end

def test_mock_pipeline_end_to_end_failures_count_as_wrong(tmp_path):
    fr = make_frames_dir(tmp_path / "frames", "smoke", [900, 901], 14)
    out = tmp_path / "mock"
    r = _run_cli("calls", "--mode", "mock", "--frames", fr, "--preflight-frames", fr, "--out", out / "calls")
    assert "ER-OBS-1 PREFLIGHT: PASS" in r.stdout and "COMPLETE" in r.stdout
    calls = json.loads((out / "calls" / "calls.json").read_text())
    kinds = calls["outcome_kinds"]
    for k in ("ok", "non_json", "schema_violation", "no_answer", "bad_envelope", "http_error", "timeout"):
        assert kinds.get(k), (k, kinds)
    # fix 2: the re-sendable first attempts (429 / 503 / connection error) were re-sent and recorded; nothing else was
    by_id = {(rec["frame_id"], rec["variant"]): rec for rec in calls["records"]}
    resent = [rec for rec in calls["records"] if len(rec.get("attempts") or []) > 1]
    assert resent and all(rec["attempts"][0]["kind"] in ("http_error", "connection_error")
                          and rec["attempts"][0]["retryable"] for rec in resent)
    assert all(not a["retryable"] for rec in calls["records"] for a in (rec.get("attempts") or [])[-1:])
    # fix 1: a C1 answer with an out-of-range point is a schema violation that still carries its booleans
    c1_point = [rec for rec in calls["records"] if rec["variant"] == "c1" and rec["kind"] == "schema_violation"
                and rec.get("booleans")]
    assert c1_point, "the forced c1_point_out_of_range outcome"
    assert by_id
    assert calls["prompt_schema_sha256"] == C.PROMPT_SCHEMA_SHA256 and calls["mode"] == "mock"
    assert "not a result" in calls["note"]
    r = _run_cli("verdict", "accuracy", "--frames", fr, "--calls", out / "calls" / "calls.json",
                 "--out", out / "verdict_accuracy.json")
    assert "PIPELINE_CHECK (MOCK" in r.stdout
    va = json.loads((out / "verdict_accuracy.json").read_text())
    assert va["verdict"] == "PIPELINE_CHECK" and va["mode"] == "mock"
    # recount from the raw records: every failed main call is scored as the negation of its label
    man = json.loads((fr / "manifest.json").read_text())
    lab = {f["id"]: f["labels"] for f in man["frames"]}
    failed = [r for r in calls["records"] if r["variant"] == "main" and r["kind"] != "ok"]
    assert failed
    for b in E.BOOLEANS:
        tp = tn = 0
        for rec in calls["records"]:
            if rec["variant"] != "main":
                continue
            y = lab[rec["frame_id"]][b]
            p = rec["answer"][b] if rec["kind"] == "ok" else (not y)
            tp += y and p
            tn += (not y) and (not p)
        assert (tp, tn) == (va["main_balanced_accuracy"][b]["tp"], va["main_balanced_accuracy"][b]["tn"])
    # mock answers marked correct point inside the cube: every visible frame with an ok answer is a hit
    ok_visible = [r for r in calls["records"] if r["variant"] == "main" and r["kind"] == "ok"]
    assert va["point"]["hits"] == len(ok_visible)
    # the verdict refuses to be written twice
    r = _run_cli("verdict", "accuracy", "--frames", fr, "--calls", out / "calls" / "calls.json",
                 "--out", out / "verdict_accuracy.json", check=False)
    assert r.returncode != 0 and "REFUSED" in (r.stdout + r.stderr)
    # latency (mock): 50 calls, exactly 2 forced timeouts (+inf), finite nearest-rank p95
    _run_cli("latency", "--mode", "mock", "--frames", fr, "--preflight-frames", fr, "--out", out / "latency")
    _run_cli("verdict", "latency", "--latency", out / "latency" / "latency.json", "--out", out / "verdict_latency.json")
    vl = json.loads((out / "verdict_latency.json").read_text())
    assert vl["verdict"] == "PIPELINE_CHECK" and vl["failed_calls"] == 2 and not vl["p95_is_inf"]
    assert vl["calls"] == 50
    (out / "step_status.json").write_text(json.dumps({"pytest": 0, "calls": 0, "verdict_accuracy": 0, "latency": 0,
                                                      "verdict_latency": 0}))
    r = _run_cli("smoke-verdict", "calls", "--step-status", out / "step_status.json", "--frames", fr,
                 "--calls", out / "calls" / "calls.json", "--verdict-accuracy", out / "verdict_accuracy.json",
                 "--verdict-latency", out / "verdict_latency.json", "--out", out / "smoke_verdict.json")
    assert json.loads((out / "smoke_verdict.json").read_text())["verdict"] == "MOCK_SMOKE_PASS", r.stdout
    # the calls cache is final: a second calls run refuses (calls.json exists)
    r = _run_cli("calls", "--mode", "mock", "--frames", fr, "--out", out / "calls", check=False)
    assert r.returncode != 0 and "REFUSED" in (r.stdout + r.stderr)


def test_mock_mode_disables_the_network(monkeypatch):
    for name in ("create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, getattr(socket, name))
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, getattr(socket.socket, name))
    C.disable_network()
    with pytest.raises(RuntimeError, match="network disabled"):
        socket.create_connection(("generativelanguage.googleapis.com", 443))
    with pytest.raises(RuntimeError, match="network disabled"):
        socket.getaddrinfo("generativelanguage.googleapis.com", 443)


def test_mock_preflight_that_does_not_parse_stops_before_any_scored_call(tmp_path, monkeypatch, capsys):
    cli = _load_cli()
    fr = make_frames_dir(tmp_path / "frames", "smoke", [900], 6)
    monkeypatch.setattr(cli.C, "disable_network", lambda: None)
    real = cli.C.MockTransport

    class BadPreflight(real):
        def __call__(self, body):
            self.force = {}
            self.plan = lambda *a: "malformed_json"
            return super().__call__(body)
    monkeypatch.setattr(cli.C, "MockTransport", BadPreflight)
    rc = cli.main(["calls", "--mode", "mock", "--frames", str(fr), "--preflight-frames", str(fr),
                   "--out", str(tmp_path / "calls")])
    assert rc == 5 and "STOPPED BEFORE ANY SCORED CALL" in capsys.readouterr().out
    assert not list((tmp_path / "calls" / "cache").glob("*.json"))


# =================================================================== run mode with an in-process fake service

class FakeHttp:
    """Replaces C.HttpTransport: records headers and bodies, returns scripted responses, never opens a socket."""
    instances: list = []
    script = None

    def __init__(self, key):
        self.headers = {"Content-Type": "application/json", "x-goog-api-key": key}
        self.bodies = []
        FakeHttp.instances.append(self)

    def __call__(self, body):
        self.bodies.append(body)
        n = len(self.bodies)
        if FakeHttp.script is not None:
            r = FakeHttp.script(n, body)
            if isinstance(r, Exception):
                raise r
            return r if len(r) == 3 else (*r, {})
        return 200, _env(GOOD), {}


def _run_setup(tmp_path, monkeypatch):
    cli = _load_cli()
    FakeHttp.instances, FakeHttp.script = [], None
    monkeypatch.setattr(cli.C, "HttpTransport", FakeHttp)
    for name in ("create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network in tests")))
    run = make_frames_dir(tmp_path / "frames-run", "run", list(E.SCORED_SEEDS), 20, hw=(6, 8))
    pre = make_frames_dir(tmp_path / "frames-smoke", "smoke", [900], 2, hw=(6, 8))
    keydir = tmp_path / "secret"
    keydir.mkdir()
    key = _key(keydir)
    monkeypatch.setenv(C.KEY_ENV, str(key))
    monkeypatch.setenv(C.PAID_TIER_ENV, "1")
    vc = C.VirtualClock()                       # the 1.0 s pacing and re-send waits run on a virtual clock
    monkeypatch.setattr(C, "CLOCK", vc)
    monkeypatch.setattr(C, "SLEEP", vc.sleep)
    return cli, run, pre


def test_run_mode_sends_only_frame_and_prompt_and_never_stores_the_key(tmp_path, monkeypatch, capsys):
    cli, run, pre = _run_setup(tmp_path, monkeypatch)
    out = tmp_path / "calls-run"
    rc = cli.main(["calls", "--mode", "run", "--frames", str(run), "--preflight-frames", str(pre), "--out", str(out),
                   "--expect-prompt-sha256", C.PROMPT_SCHEMA_SHA256,
                   "--expect-request-sha256", C.REQUEST_SHA256])
    printed = capsys.readouterr()
    assert rc == 0, printed.out
    (t,) = FakeHttp.instances
    assert len(t.bodies) == 1 + 800                               # the preflight + 400 main + 400 C1
    assert t.headers["x-goog-api-key"] == FAKE_KEY
    for body in t.bodies:
        req = json.loads(body)
        assert set(req) == {"model", "input", "generation_config"} and req["generation_config"] == C.GENERATION_CONFIG
        parts = req["input"]["parts"]
        assert set(req["input"]) == {"parts"} and len(parts) == 2 and parts[1] == {"text": C.TEXT}
        assert set(parts[0]) == {"inlineData"} and set(parts[0]["inlineData"]) == {"mimeType", "data"}
        assert FAKE_KEY.encode() not in body
        parts[0]["inlineData"]["data"] = "<image>"
        rest = json.dumps(req)
        for word in ("/nfs", "s30", "900", "sanchej7", "seed"):
            assert word not in rest, word
    for f in out.rglob("*"):
        if f.is_file():
            assert FAKE_KEY not in f.read_text(errors="replace"), f
    assert FAKE_KEY not in printed.out + printed.err
    calls = json.loads((out / "calls.json").read_text())
    assert calls["mode"] == "run" and calls["outcome_kinds"] == {"ok": 800}
    assert calls["request_format_tested_live"] is False
    assert calls["pacing"]["min_gap_before_s"] >= C.MIN_GAP_S and calls["re_sent_records"] == 0
    v = out / "verdict.json"
    assert cli.main(["verdict", "accuracy", "--frames", str(run), "--calls", str(out / "calls.json"),
                     "--out", str(v)]) == 0
    assert json.loads(v.read_text())["verdict"] in ("PASS", "NEGATIVE", "INCOMPLETE")


def test_run_mode_refuses_wrong_hash_or_bad_key_before_reading_the_key(tmp_path, monkeypatch):
    cli, run, pre = _run_setup(tmp_path, monkeypatch)
    good = ["--expect-prompt-sha256", C.PROMPT_SCHEMA_SHA256, "--expect-request-sha256", C.REQUEST_SHA256]

    def calls(out, *extra, frames=run, preflight=pre, cmd="calls"):
        argv = [cmd, "--mode", "run", "--frames", str(frames), "--out", str(tmp_path / out), *extra]
        if preflight is not None:
            argv[5:5] = ["--preflight-frames", str(preflight)]
        return cli.main(argv)
    with pytest.raises(SystemExit, match="expect-prompt-sha256"):
        calls("o1", "--expect-prompt-sha256", "0" * 64, "--expect-request-sha256", C.REQUEST_SHA256)
    with pytest.raises(SystemExit, match="expect-request-sha256"):
        calls("o1b", "--expect-prompt-sha256", C.PROMPT_SCHEMA_SHA256, "--expect-request-sha256", "0" * 64)
    # fix 6: the paid-tier confirmation is required in run and latency modes, before the key is read
    monkeypatch.delenv(C.PAID_TIER_ENV)
    for cmd in ("calls", "latency"):
        with pytest.raises(SystemExit, match=C.PAID_TIER_ENV):
            calls("o1c", *good, cmd=cmd)
    monkeypatch.setenv(C.PAID_TIER_ENV, "yes")                   # only exactly "1" confirms
    with pytest.raises(SystemExit, match=C.PAID_TIER_ENV):
        calls("o1d", *good)
    monkeypatch.setenv(C.PAID_TIER_ENV, "1")
    monkeypatch.setenv(C.KEY_ENV, str(tmp_path / "missing-key"))
    with pytest.raises(SystemExit, match="API key file: key file missing"):
        calls("o2", *good)
    loose = _key(tmp_path / "secret", 0o644, name="loose")
    monkeypatch.setenv(C.KEY_ENV, str(loose))
    with pytest.raises(SystemExit, match="mode is 644"):
        calls("o3", *good, cmd="latency")
    inside = _key(run, name="key-in-frames")                      # inside the frames directory
    monkeypatch.setenv(C.KEY_ENV, str(inside))
    with pytest.raises(SystemExit, match="inside"):
        calls("o4", *good)
    assert FakeHttp.instances == []                               # no transport was ever built
    with pytest.raises(SystemExit, match="preflight-frames"):
        calls("o5", *good, preflight=None)
    with pytest.raises(SystemExit, match="not the scored run"):
        calls("o6", *good, frames=pre)


def test_run_mode_refuses_frames_with_a_class_below_20_before_any_call(tmp_path, monkeypatch):
    """Fix 5: the accuracy verdict would be INCOMPLETE whatever the answers, so no paid call is made."""
    cli, run, pre = _run_setup(tmp_path, monkeypatch)
    good = ["--expect-prompt-sha256", C.PROMPT_SCHEMA_SHA256, "--expect-request-sha256", C.REQUEST_SHA256]
    mpath = run / "manifest.json"
    man = json.loads(mpath.read_text())
    for f in man["frames"]:
        f["labels"]["on_floor"] = False                          # on_floor=true: 0 frames
    man["summary"]["classes_below_20"] = V.short_classes(V.class_counts(man["frames"]))
    mpath.write_text(json.dumps(man))
    with pytest.raises(SystemExit, match="label classes below 20 frames"):
        cli.main(["calls", "--mode", "run", "--frames", str(run), "--preflight-frames", str(pre),
                  "--out", str(tmp_path / "o"), *good])
    man["summary"]["classes_below_20"] = []                      # a summary that disagrees with the recount
    mpath.write_text(json.dumps(man))
    with pytest.raises(SystemExit, match="label classes below 20 frames"):
        cli.main(["calls", "--mode", "run", "--frames", str(run), "--preflight-frames", str(pre),
                  "--out", str(tmp_path / "o2"), *good])
    assert FakeHttp.instances == [] and not (tmp_path / "o").exists()


def test_run_mode_stops_after_5_transport_failures_and_resumes_with_failures_final(tmp_path, monkeypatch, capsys):
    cli, run, pre = _run_setup(tmp_path, monkeypatch)
    out = tmp_path / "calls-run"
    argv = ["calls", "--mode", "run", "--frames", str(run), "--preflight-frames", str(pre), "--out", str(out),
            "--expect-prompt-sha256", C.PROMPT_SCHEMA_SHA256, "--expect-request-sha256", C.REQUEST_SHA256]
    # HTTP 500 is a final transport-level failure (not re-sendable): 5 in a row stop the caller
    FakeHttp.script = lambda n, body: (200, _env(GOOD)) if n == 1 else (500, b'{"error": {"message": "down"}}')
    assert cli.main(argv) == 4
    cached = sorted((out / "cache").glob("*.json"))
    assert len(cached) == 5 and all(json.loads(p.read_text())["kind"] == "http_error" for p in cached)
    assert not list((out / "cache").glob("*.claim"))              # every claim was released
    FakeHttp.script = None
    assert cli.main(argv) == 0                                    # resumes; the 5 failures stay final
    calls = json.loads((out / "calls.json").read_text())
    assert calls["outcome_kinds"] == {"http_error": 5, "ok": 795}
    assert len(FakeHttp.instances[1].bodies) == 1 + 795
    # fix 2: HTTP 503 every time is re-sent MAX_RESENDS times, then the caller stops with NO outcome for that frame
    out2 = tmp_path / "calls-503"
    argv2 = [a if a != str(out) else str(out2) for a in argv]
    FakeHttp.script = lambda n, body: (200, _env(GOOD)) if n == 1 else (503, b'{"error": {"message": "busy"}}')
    assert cli.main(argv2) == 4
    assert not list((out2 / "cache").glob("*.json")) and not list((out2 / "cache").glob("*.claim"))
    (log,) = (out2 / "cache").glob("*.attempts.jsonl")
    attempts = [json.loads(ln) for ln in log.read_text().splitlines()]
    assert len(attempts) == C.MAX_RESENDS + 1 and all(a["http_status"] == 503 and a["retryable"] for a in attempts)
    FakeHttp.script = None
    assert cli.main(argv2) == 0                                   # the resume answers it; its 503s stay on record
    rec = json.loads(next((out2 / "cache").glob(f"{log.name.split('.')[0]}.json")).read_text())
    assert rec["kind"] == "ok" and rec["n_attempts"] == C.MAX_RESENDS + 2
    # a preflight that does not parse: stops before any scored call
    FakeHttp.script = lambda n, body: (200, _env("not json"))
    assert cli.main(["calls", "--mode", "run", "--frames", str(run), "--preflight-frames", str(pre),
                     "--out", str(tmp_path / "o2"), "--expect-prompt-sha256", C.PROMPT_SCHEMA_SHA256,
                     "--expect-request-sha256", C.REQUEST_SHA256]) == 5
    assert not list((tmp_path / "o2" / "cache").glob("*.json"))
    capsys.readouterr()


def test_claims_are_exclusive_and_published_outcomes_are_never_overwritten(tmp_path):
    """Fix 4: two overlapping resumes can never call the same frame twice."""
    cli = _load_cli()
    claim = tmp_path / "f__main.claim"
    assert cli.claim(claim, {"pid": 1}) is True and cli.claim(claim, {"pid": 2}) is False
    assert json.loads(claim.read_text())["pid"] == 1
    cli.release(claim)
    assert cli.claim(claim, {"pid": 3}) is True
    final = tmp_path / "f__main.json"
    cli.publish_once(final, {"kind": "ok"})
    with pytest.raises(FileExistsError):
        cli.publish_once(final, {"kind": "timeout"})
    assert json.loads(final.read_text()) == {"kind": "ok"}
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]   # no temporary file left behind


# =================================================================== CLI guards

def test_frames_cli_refuses_reserved_scored_and_partial_seeds(tmp_path):
    base = ["frames", "--deploy", "x", "--upstream", "x", "--cache-dir", str(tmp_path / "c")]
    cases = [
        (["--mode", "smoke", "--seeds", "900,901"], "smoke needs --seconds"),
        (["--mode", "smoke", "--seeds", "300", "--seconds", "14"], "throw-away seeds 900-901"),
        (["--mode", "smoke", "--seeds", "902", "--seconds", "14"], "throw-away seeds 900-901"),
        (["--mode", "smoke", "--seeds", "120", "--seconds", "14"], "reserved"),
        (["--mode", "run", "--seeds", "300-318"], "exactly seeds 300-319"),
        (["--mode", "run", "--seeds", "300-319", "--seconds", "14"], "exactly seeds 300-319"),
        (["--mode", "run", "--seeds", "0-19"], "reserved"),
    ]
    for extra, msg in cases:
        r = _run_cli(*base, *extra, "--out", tmp_path / "o", check=False)
        assert r.returncode != 0 and msg in (r.stdout + r.stderr), (extra, r.stdout + r.stderr)
        assert not (tmp_path / "o").exists()


def test_hashes_command_matches_the_launcher_and_the_code():
    r = _run_cli("hashes")
    h = json.loads(r.stdout)
    assert h["prompt_schema_sha256"] == C.PROMPT_SCHEMA_SHA256 and h["request_sha256"] == C.REQUEST_SHA256
    assert f"PROMPT_SHA256={C.PROMPT_SCHEMA_SHA256}" in CALLS_SH.read_text()
    assert C.REQUEST_SHA256 in CALLS_SH.read_text()
    assert "x-goog-api-key" in json.dumps(h["request_template"]) and FAKE_KEY not in r.stdout


def test_keycheck_command_never_prints_the_key(tmp_path):
    k = _key(tmp_path)
    r = _run_cli("keycheck", "--forbid", tmp_path / "out", env_extra={C.KEY_ENV: str(k)})
    assert "KEY FILE: OK" in r.stdout and FAKE_KEY not in r.stdout + r.stderr
    os.chmod(k, 0o644)
    r = _run_cli("keycheck", env_extra={C.KEY_ENV: str(k)}, check=False)
    assert r.returncode == 1 and "REFUSED" in r.stdout and FAKE_KEY not in r.stdout + r.stderr


# =================================================================== launchers

def _norm(s):
    return re.sub(r"\s+", " ", s).strip()


def _header(path):
    return _norm(" ".join(ln.lstrip("#").strip() for ln in path.read_text().splitlines() if ln.startswith("#")))


def test_launcher_headers_state_the_design_label_clauses_and_limits():
    for path in (FRAMES_SH, CALLS_SH):
        text, head = path.read_text(), _header(path)
        assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
        assert "#SBATCH --time=01:00:00" in text and "set -uo pipefail" in text and "|| true" not in text
        assert E.LABEL in text
        assert "ACCURACY PASS iff balanced accuracy ≥ 0.90 on all 4 booleans" in head
        assert "INCOMPLETE if any class has fewer than 20 frames" in head
        assert "IN-LOOP-ELIGIBLE is a separate verdict: p95 latency ≤ 1.0 s over 50 calls" in head
        assert "CLAUSES AS APPLIED" in head
    fh = _header(FRAMES_SH)
    assert _norm(E.DESIGN_TEXT) in fh
    assert "#SBATCH --gres=gpu:1" in FRAMES_SH.read_text() and "MUJOCO_EGL_DEVICE_ID" in FRAMES_SH.read_text()
    assert "SEED EVIDENCE" in fh and "900-901" in fh and "alpha 0" in fh and "reflection OFF" in fh
    assert "two gates" in fh and "21532217" in fh and "<= 2 levels and <= 100 differing pixels" in fh
    assert "dgxh" not in FRAMES_SH.read_text().splitlines()[3]           # the partition line
    ch = _header(CALLS_SH)
    assert "x-goog-api-key" in ch and "mode 600" in ch and "UNTESTED" in ch and "never printed" in ch
    assert "#SBATCH --partition=share" in CALLS_SH.read_text()


def test_launcher_seeds_and_sections():
    text = FRAMES_SH.read_text()
    smoke = text.split('if [ "$MODE" = smoke ]; then', 1)[1].split("\nfi\n", 1)[0]
    run = text.split('if [ "$MODE" = smoke ]; then', 1)[1].split("\nfi\n", 1)[1]
    assert re.findall(r"--seeds (\S+)", smoke) == ["900,901"] and "--seconds 14" in smoke
    assert re.findall(r"--seeds (\S+)", run) == ["300-319"] and "--seconds" not in run
    assert "*smoke*)" in smoke and "frames-smoke/$JOB" in smoke
    assert (run.index('if [ -z "$SMOKE_JOB" ]') < run.index('if [ "$SV" != SMOKE_PASS ]') < run.index("sha256sum -c")
            < run.index('if [ -e "$RUN" ]') < run.index("egl_preflight") < run.index("frames --mode run"))
    ctext = CALLS_SH.read_text()
    run = ctext.split('if [ "$MODE" = mock ]; then', 1)[1].split("\nfi\n", 1)[1]
    assert (run.index("sha256sum -c") < run.index("keycheck") < run.index("latency --mode run")
            < run.index("calls --mode run"))
    assert "--expect-prompt-sha256 \"$PROMPT_SHA256\"" in run


def _launch(path, args, extra, cwd=REPO):
    env = _clean_env()
    env.update(extra)
    return subprocess.run(["bash", str(path), *args], capture_output=True, text=True, env=env, cwd=cwd, timeout=300)


def test_launchers_refuse_without_mode_smoke_name_or_smoke_job(tmp_path):
    base = {"ER_OBS_DATA": str(tmp_path / "data"), "SLURM_JOB_ID": "pytest-refusal", "ER_OBS_CODE": str(REPO)}
    r = _launch(FRAMES_SH, [], base)
    assert r.returncode == 1 and "REFUSED (usage" in r.stdout
    r = _launch(FRAMES_SH, ["smoke"], {**base, "SLURM_JOB_NAME": "er-obs1-frames"})
    assert r.returncode == 1 and "job name must contain 'smoke'" in r.stdout
    r = _launch(FRAMES_SH, ["run"], base)
    assert r.returncode == 1 and "ER_OBS_FRAMES_SMOKE_JOB" in r.stdout
    r = _launch(FRAMES_SH, ["run"], {**base, "ER_OBS_FRAMES_SMOKE_JOB": "nosuch"})
    assert r.returncode == 1 and "reads MISSING, not SMOKE_PASS" in r.stdout
    r = _launch(CALLS_SH, [], base)
    assert r.returncode == 1 and "REFUSED (usage" in r.stdout
    r = _launch(CALLS_SH, ["mock"], {**base, "SLURM_JOB_NAME": "er-obs1-calls"})
    assert r.returncode == 1 and "job name must contain 'smoke'" in r.stdout
    r = _launch(CALLS_SH, ["mock"], {**base, "SLURM_JOB_NAME": "er-obs1-calls-smoke"})
    assert r.returncode == 1 and "ER_OBS_MOCK_FRAMES" in r.stdout
    for mode in ("run", "latency"):
        r = _launch(CALLS_SH, [mode], base)
        assert r.returncode == 1 and "ER_OBS_CALLS_SMOKE_JOB" in r.stdout
    assert not (tmp_path / "data").exists()


def test_frames_launcher_refusals_do_not_need_the_upstream_checkout(tmp_path):
    """In a bare checkout (no external/ symlink yet) the cheap refusals still come first."""
    bare = tmp_path / "bare"
    bare.mkdir()
    base = {"ER_OBS_DATA": str(tmp_path / "data"), "SLURM_JOB_ID": "pytest-bare", "ER_OBS_CODE": str(bare)}
    r = _launch(FRAMES_SH, ["smoke"], {**base, "SLURM_JOB_NAME": "er-obs1-frames"}, cwd=bare)
    assert r.returncode == 1 and "job name must contain 'smoke'" in r.stdout
    r = _launch(FRAMES_SH, ["run"], base, cwd=bare)
    assert r.returncode == 1 and "ER_OBS_FRAMES_SMOKE_JOB" in r.stdout
    r = _launch(FRAMES_SH, ["smoke"], {**base, "SLURM_JOB_NAME": "er-obs1-frames-smoke"}, cwd=bare)
    assert r.returncode == 1 and "deploy config missing" in r.stdout
    assert not (tmp_path / "data").exists()


STUB = r"""#!/bin/bash
# fake interpreter for the launcher tests: logs er_obs.py calls, fakes their outputs, runs -c/-m for real
if [ "$1" = "-c" ]; then
    case "$2" in *Renderer*) echo "EGL render preflight OK (stub)"; exit 0 ;; esac
    exec "$STUB_REAL_PY" "$@"
fi
if [ "$1" = "-m" ]; then exec "$STUB_REAL_PY" "$@"; fi
case "$1" in
  *er_obs.py)
    sub=$2
    case "$sub" in hashes|keycheck) exec "$STUB_REAL_PY" "$@" ;; esac
    shift
    echo "$*" >> "$STUB_LOG"
    out=""; prev=""
    for a in "$@"; do [ "$prev" = "--out" ] && out=$a; prev=$a; done
    case "$sub" in
      frames) mkdir -p "$out"; echo '{"check": "FRAMES_OK"}' > "$out/manifest.json" ;;
      calls) mkdir -p "$out"; echo '{}' > "$out/calls.json" ;;
      latency) mkdir -p "$out"; echo '{}' > "$out/latency.json" ;;
      verdict) echo '{"verdict": "STUB"}' > "$out" ;;
    esac
    exit 0 ;;
esac
echo "stub: unexpected $*" >&2; exit 9
"""

CODEFILES_RE = re.compile(r"CODEFILES=\((.*?)\)", re.S)


def _fake_tree(tmp_path, launcher):
    """A copy of the code files + launcher in a fake root with a fake deploy file, and a fake interpreter."""
    tree = tmp_path / "tree"
    files = CODEFILES_RE.search(launcher.read_text()).group(1).split()
    for rel in files + [f"slurm/repo20260923/{launcher.name}"]:
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, tree / rel)
    dep = tree / "external/Berkeley-Humanoid-Lite/logs/rsl_rl/humanoid/2026-08-18_20-57-50_arms-dr1.0-s0/exported"
    dep.mkdir(parents=True)
    (dep / "deploy.yaml").write_text("fake\n")
    stub = tmp_path / "stub_python"
    stub.write_text(STUB)
    stub.chmod(0o755)
    sha = subprocess.run(["sha256sum", *files], cwd=tree, capture_output=True, text=True, check=True).stdout
    lsha = subprocess.run(["sha256sum", str(tree / "slurm/repo20260923" / launcher.name)], capture_output=True,
                          text=True, check=True).stdout.split()[0]
    env = {"ER_OBS_CODE": str(tree), "ER_OBS_PY": str(stub), "ER_OBS_DATA": str(tmp_path / "data"),
           "STUB_REAL_PY": PY, "STUB_LOG": str(tmp_path / "stub.log"), "SLURM_JOB_ID": "pytest-fake",
           "PYTHONPATH": str(tree / "src")}
    return tree, env, sha, lsha


def test_frames_launcher_real_mode_command_line_on_a_fake_tree(tmp_path):
    tree, env, sha, lsha = _fake_tree(tmp_path, FRAMES_SH)
    sm = tmp_path / "data/frames-smoke/fake1"
    sm.mkdir(parents=True)
    (sm / "smoke_verdict.json").write_text(json.dumps({"verdict": "SMOKE_PASS", "step_status": {"pytest": 0}}))
    (sm / "code_sha256.txt").write_text(sha)
    (sm / "launcher_sha256.txt").write_text(lsha + "\n")
    launcher = tree / "slurm/repo20260923" / FRAMES_SH.name
    r = _launch(launcher, ["run"], {**env, "ER_OBS_FRAMES_SMOKE_JOB": "fake1"}, cwd=tree)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "BYTES-CHECK: PASS" in r.stdout and "ER-OBS-1 FRAMES RESULT: FRAMES_OK" in r.stdout
    (line,) = (tmp_path / "stub.log").read_text().splitlines()
    data = tmp_path / "data"
    assert line == (f"frames --mode run --deploy {tree}/external/Berkeley-Humanoid-Lite/logs/rsl_rl/humanoid/"
                    f"2026-08-18_20-57-50_arms-dr1.0-s0/exported/deploy.yaml --upstream "
                    f"{tree}/external/Berkeley-Humanoid-Lite --cache-dir {line.split('--cache-dir ')[1].split()[0]} "
                    f"--seeds 300-319 --out {data}/frames-run")
    # a second run refuses (outputs exist); a changed code file fails the bytes check
    r = _launch(launcher, ["run"], {**env, "ER_OBS_FRAMES_SMOKE_JOB": "fake1"}, cwd=tree)
    assert r.returncode == 1 and "never overwritten" in r.stdout
    with open(tree / "scripts/bench/er_obs.py", "a") as f:
        f.write("\n# edited\n")
    r = _launch(launcher, ["run"], {**env, "ER_OBS_FRAMES_SMOKE_JOB": "fake1"}, cwd=tree)
    assert r.returncode == 1 and "code differs" in r.stdout


def test_calls_launcher_real_modes_on_a_fake_tree_and_key_refusals(tmp_path):
    tree, env, sha, lsha = _fake_tree(tmp_path, CALLS_SH)
    data = tmp_path / "data"
    sm = data / "calls-mock-smoke/fake2"
    sm.mkdir(parents=True)
    (sm / "smoke_verdict.json").write_text(json.dumps({"verdict": "MOCK_SMOKE_PASS", "step_status": {"pytest": 0}}))
    (sm / "code_sha256.txt").write_text(sha)
    (sm / "launcher_sha256.txt").write_text(lsha + "\n")
    (data / "frames-run").mkdir(parents=True)
    (data / "frames-run/manifest.json").write_text(json.dumps({"check": "FRAMES_OK", "mode": "run",
                                                               "summary": {"classes_below_20": ["on_floor=true"]}}))
    pre = data / "frames-smoke/fake0/frames"
    pre.mkdir(parents=True)
    (pre / "manifest.json").write_text(json.dumps({"check": "FRAMES_OK", "mode": "smoke"}))
    keydir = tmp_path / "secret"
    keydir.mkdir()
    launcher = tree / "slurm/repo20260923" / CALLS_SH.name
    base = {**env, "ER_OBS_CALLS_SMOKE_JOB": "fake2", "ER_OBS_PREFLIGHT_FRAMES": str(pre)}
    good_early = _key(keydir, name="good-early")
    # fix 5: a run manifest with a class below 20 frames: refused before the key is checked
    r = _launch(launcher, ["run"], {**base, C.PAID_TIER_ENV: "1", C.KEY_ENV: str(good_early)}, cwd=tree)
    assert r.returncode == 1 and "classes_below_20 reads ['on_floor=true'], not []" in r.stdout
    (data / "frames-run/manifest.json").write_text(json.dumps({"check": "FRAMES_OK", "mode": "run",
                                                               "summary": {"classes_below_20": []}}))
    # fix 6: run and latency refuse without the paid-tier confirmation
    for mode in ("run", "latency"):
        r = _launch(launcher, [mode], {**base, C.KEY_ENV: str(good_early)}, cwd=tree)
        assert r.returncode == 1 and "ER_OBS_PAID_TIER_CONFIRMED=1" in r.stdout and "nothing was sent" in r.stdout
    # tier amendment (2026-10-05): a contradiction refuses; latency refuses on the free tier
    r = _launch(launcher, ["run"], {**base, C.PAID_TIER_ENV: "1", C.TIER_ENV: "free", C.KEY_ENV: str(good_early)},
                cwd=tree)
    assert r.returncode == 1 and "never both" in r.stdout and "nothing was sent" in r.stdout
    r = _launch(launcher, ["latency"], {**base, C.TIER_ENV: "free", C.KEY_ENV: str(good_early)}, cwd=tree)
    assert r.returncode == 1 and "rate limit" in r.stdout and "nothing was sent" in r.stdout
    base[C.PAID_TIER_ENV] = "1"
    # key file missing / mode 644 / inside the code root: refused, nothing logged as a call
    r = _launch(launcher, ["run"], {**base, C.KEY_ENV: str(keydir / "missing")}, cwd=tree)
    assert r.returncode == 1 and "KEY FILE: REFUSED" in r.stdout and "nothing was sent" in r.stdout
    loose = _key(keydir, 0o644, name="loose")
    r = _launch(launcher, ["latency"], {**base, C.KEY_ENV: str(loose)}, cwd=tree)
    assert r.returncode == 1 and "mode is 644" in r.stdout and FAKE_KEY not in r.stdout + r.stderr
    inside = _key(tree, name="key-in-tree")
    r = _launch(launcher, ["run"], {**base, C.KEY_ENV: str(inside)}, cwd=tree)
    assert r.returncode == 1 and "inside" in r.stdout
    assert not (tmp_path / "stub.log").exists()
    # a good key file: the real-mode command lines
    good = _key(keydir, name="good")
    r = _launch(launcher, ["latency"], {**base, C.KEY_ENV: str(good)}, cwd=tree)
    assert r.returncode == 0, r.stdout + r.stderr
    r = _launch(launcher, ["run"], {**base, C.KEY_ENV: str(good)}, cwd=tree)
    assert r.returncode == 0, r.stdout + r.stderr
    assert FAKE_KEY not in r.stdout + r.stderr
    lines = (tmp_path / "stub.log").read_text().splitlines()
    h, rh = C.PROMPT_SCHEMA_SHA256, C.REQUEST_SHA256
    assert lines == [
        f"latency --mode run --frames {data}/frames-run --preflight-frames {pre} --out {data}/latency-run "
        f"--expect-prompt-sha256 {h} --expect-request-sha256 {rh}",
        f"verdict latency --latency {data}/latency-run/latency.json --out {data}/latency-run/verdict_latency.json",
        f"calls --mode run --frames {data}/frames-run --preflight-frames {pre} --out {data}/calls-run "
        f"--expect-prompt-sha256 {h} --expect-request-sha256 {rh}",
        f"verdict accuracy --frames {data}/frames-run --calls {data}/calls-run/calls.json "
        f"--out {data}/calls-run/verdict_accuracy.json",
    ]
    # outputs exist now: both modes refuse to overwrite
    r = _launch(launcher, ["run"], {**base, C.KEY_ENV: str(good)}, cwd=tree)
    assert r.returncode == 1 and "never overwritten" in r.stdout
    r = _launch(launcher, ["latency"], {**base, C.KEY_ENV: str(good)}, cwd=tree)
    assert r.returncode == 1 and "never overwritten" in r.stdout


def test_tier_declaration():
    """Tier amendment (2026-10-05): paid via either variable, free only via ER_OBS_TIER, never both, nothing else."""
    assert C.declared_tier({}) is None
    assert C.declared_tier({C.PAID_TIER_ENV: "1"}) == "paid"
    assert C.declared_tier({C.TIER_ENV: "paid"}) == "paid"
    assert C.declared_tier({C.TIER_ENV: "paid", C.PAID_TIER_ENV: "1"}) == "paid"
    assert C.declared_tier({C.TIER_ENV: "free"}) == "free"
    assert C.declared_tier({C.TIER_ENV: "free", C.PAID_TIER_ENV: "1"}) is None      # a contradiction
    assert C.declared_tier({C.TIER_ENV: "FREE"}) is None and C.declared_tier({C.PAID_TIER_ENV: "yes"}) is None
    assert len(E.TIER_AMENDMENT) == 5 and any("latency" in t for t in E.TIER_AMENDMENT)


def test_free_tier_runs_the_calls_records_the_tier_and_refuses_latency(tmp_path, monkeypatch, capsys):
    cli, run, pre = _run_setup(tmp_path, monkeypatch)
    monkeypatch.delenv(C.PAID_TIER_ENV)
    monkeypatch.setenv(C.TIER_ENV, "free")
    hashes = ["--expect-prompt-sha256", C.PROMPT_SCHEMA_SHA256, "--expect-request-sha256", C.REQUEST_SHA256]
    out = tmp_path / "calls-run"
    rc = cli.main(["calls", "--mode", "run", "--frames", str(run), "--preflight-frames", str(pre), "--out", str(out),
                   *hashes])
    assert rc == 0, capsys.readouterr().out
    calls = json.loads((out / "calls.json").read_text())
    assert calls["tier"] == "free" and calls["tier_amendment"] == list(E.TIER_AMENDMENT)
    assert calls["prompt_schema_sha256"] == C.PROMPT_SCHEMA_SHA256 and calls["request_sha256"] == C.REQUEST_SHA256
    pre_files = list(out.glob("preflight-*.json"))
    assert pre_files and all(json.loads(f.read_text())["tier"] == "free" for f in pre_files)
    n = len(FakeHttp.instances)
    with pytest.raises(SystemExit, match="ER_OBS_TIER=free"):
        cli.main(["latency", "--mode", "run", "--frames", str(run), "--preflight-frames", str(pre),
                  "--out", str(tmp_path / "lat"), *hashes])
    assert len(FakeHttp.instances) == n and not (tmp_path / "lat").exists()          # nothing was sent

