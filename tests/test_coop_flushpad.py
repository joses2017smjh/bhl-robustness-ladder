"""C1 flush-pad wrist variant of the scripted cooperative lift (opt-in): the frozen rules and their boundaries, the
tilt clause, both stage verdicts, the smoke check, the CLI guard and verdict writer, the launcher header, the pad
geometry, and proof that the stock path is unchanged (model bytes, a short episode, and the source diff to 094797e).

No test runs a probe seed (120-124) or a scored seed (20-39): the only episodes here are 1 s long on declared tuning
seed 119."""

import difflib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bhl_robust.eval import scripted_carry as sc

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "external/Berkeley-Humanoid-Lite"
DEPLOY = REPO / "tests/fixtures/arms-dr1.0-s0/deploy.yaml"
LAUNCHER = REPO / "slurm/repo20260923/cpu_coop_flushpad.sbatch"
BASE_COMMIT = "094797e"          # the commit the C1 workstream started from (stock harness unchanged since fcd0c30)
DT = 0.04
N = 500                          # 20 s at the 0.04 s policy step
TUNING_SEED = 119                # a declared tuning seed: never 120-124 (probe) or 20-39 (scored)

needs_assets = pytest.mark.skipif(not (UPSTREAM / "source").is_dir(), reason="upstream assets missing")

PROBE_TEXT = (
    "Probe on exploration seeds 120-124 (state the grep evidence that they were never used). PROCEED iff the cube "
    "tilt stays <= 0.35 rad throughout lift-hold on >= 4/5 seeds, with LIFT_HOLD_RULE's robot-fall clauses unchanged. "
    "Otherwise STOP: NEGATIVE, the flush pad does not stop the roll.")
SCORED_TEXT = (
    "Scored, only after PROCEED (state the evidence that seeds 20-39 were never used): lift-hold on seeds 20-29 and "
    "lift-place on seeds 30-39, each with its UNCHANGED rule (LIFT_HOLD_RULE / LIFT_PLACE_RULE, pass_min 8) PLUS cube "
    "tilt <= 0.35 rad (hold: throughout the hold; place: at release and when seated), >= 8/10 per crew or pair as the "
    "existing rules define them. PASS iff every crew/pair meets it; otherwise NEGATIVE; INCOMPLETE if any episode is "
    "missing.")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _j(x):
    return json.loads(json.dumps(x))


def _script():
    spec = importlib.util.spec_from_file_location("coop_scripted_carry_c1",
                                                  REPO / "scripts/bench/coop_scripted_carry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------- synthetic score JSONs (no simulation)

def _series(value=0.1, n=N, spikes=None):
    """Tilt at every policy-step state: reset (0.0) + n states."""
    s = [0.0] + [float(value)] * n
    for i, v in (spikes or {}).items():
        s[i] = v
    return s


def _trace(series, n=N):
    """The harness trace: every 25 steps, cube_tilt_rad is the POST-step value = series[i + 1], 3 dp."""
    return [{"t": round(i * DT, 2), "cube_tilt_rad": [round(series[i + 1], 3)]} for i in range(0, n, 25)]


def _hold_row(pair=0, *, peak=0.2, up=(6.0, 20.0), max_tilt=0.2, floor=False, failed=None, n=N, series=None):
    t = np.arange(n) * DT
    lift = np.where((t >= up[0] - 1e-9) & (t < up[1] - 1e-9), peak, 0.0)
    row = sc.score_lift_hold(lift, np.zeros(n), DT, max_tilt=max_tilt, floor=floor, failed=failed,
                             rule=sc.LIFT_HOLD_RULE)
    row["pair"] = pair
    row["cube_tilt_series_rad"] = _series(n=n) if series is None else series
    return row


def _final(dz=0.0, contact=False):
    return {"dz_m": dz, "speed_mps": 0.0, "offset_xy_m": [0.01, -0.01], "cube_tilt_rad": 0.1,
            "robot_contact": contact, "normal_N": {"b": 0.0, "a": 0.0, "plinth": 4.9}}


def _place_row(pair=0, *, max_tilt=0.2, floor=False, contact=False, failed=None, n=N, series=None):
    t = np.arange(n) * DT
    lift = np.where((t >= 6.2) & (t < 10.0), 0.18, 0.0)
    row = sc.score_lift_place(lift, DT, max_tilt=max_tilt, floor=floor, failed=failed, final=_final(contact=contact),
                              rule=sc.LIFT_PLACE_RULE)
    row["pair"] = pair
    row["cube_tilt_series_rad"] = _series(n=n) if series is None else series
    return row


PLACE_TIMES = {k: round(v, 3) for k, v in sc.PlaceScript(sc.FlushPadParams(), sc.FlushPadPlaceParams()).t_end.items()}


def _sim_reset(detected=False, first_step=None):
    return {"detected": detected, "first_step": first_step, "evidence": None, "policy_steps": N}


def _episode(seed, rows, *, steps=N, failed=None, place=False, sim_reset=None):
    ep = {"seed": seed, "pairs": rows, "failed": failed, "steps": steps, "variant": sc.FLUSHPAD_VARIANT,
          "trace": _trace(rows[0]["cube_tilt_series_rad"], steps),
          "sim_reset": _sim_reset() if sim_reset is None else sim_reset}
    if place:
        ep.update({"protocol": "lift_place", "script_times_s": dict(PLACE_TIMES),
                   "place_params": sc.place_params_dict(sc.FlushPadPlaceParams())})
    else:
        ep["protocol"] = "lift_hold"
    return ep


FRAMES = {s: {"pos": [0.0, 0.0, -0.07], "quat": [0.96, -0.13, 0.25, -0.03], "size": [0.031, 0.043, 0.069]}
          for s in ("left", "right")}


def _model_check(crew, condim=4, tors=0.04, cone="elliptic", impratio=10.0, quat=None):
    return {"pads": [{"name": f"r{i}_arm_{s}_hand_pad", "condim": condim, "friction": [1.0, tors, 0.0001],
                      "quat": list(FRAMES[s]["quat"] if quat is None else quat), "size": list(FRAMES[s]["size"])}
                     for i in range(crew) for s in ("left", "right")],
            "cone": cone, "impratio": impratio, "noslip_iterations": 0}


def _payload(stage, crew, episodes, protocol=None):
    rule = (sc.flushpad_stage_rule("smoke", protocol, sorted({e["seed"] for e in episodes}))
            if stage == "smoke" else sc.flushpad_stage_rule(stage))
    pl = {"variant": sc.FLUSHPAD_VARIANT, "stage": stage, "protocol": rule["protocol"], "crew": crew,
          "pairs": crew // 2, "scored_run": stage != "smoke", "rule": rule,
          "params": sc.params_dict(sc.FlushPadParams()), "keyframes_left": sc.KEYFRAMES_LEFT,
          "model_check": _model_check(crew), "flush_pad_design": {"frames": FRAMES}, "policy_dt": DT,
          "episodes": episodes}
    if rule["protocol"] == "lift_place":
        pl["place_params"] = sc.place_params_dict(sc.FlushPadPlaceParams())
    return _j(pl)


def _probe(n_good=5, bad=None, seeds=(120, 121, 122, 123, 124)):
    """Probe payload: the first n_good seeds keep tilt 0.1; the rest exceed 0.35 rad (or use `bad` rows)."""
    eps = []
    for i, s in enumerate(seeds):
        row = _hold_row() if i < n_good else (bad() if bad else _hold_row(series=_series(1.0)))
        eps.append(_episode(s, [row]))
    return _payload("probe", 2, eps)


def _scored_group(stage, crew, n_good=10, fail=None):
    seeds = list(range(20, 30)) if stage == "hold" else list(range(30, 40))
    eps = []
    for i, s in enumerate(seeds):
        rows = []
        for k in range(crew // 2):
            good = i < n_good or (fail is not None and fail[0] != k)
            if stage == "hold":
                rows.append(_hold_row(k) if good else _hold_row(k, series=_series(0.9)))
            else:
                rows.append(_place_row(k) if good else _place_row(k, series=_series(0.9)))
        eps.append(_episode(s, rows, place=stage == "place"))
    return _payload(stage, crew, eps)


def _scored(**override):
    pls = {f"{st}_crew{c}": _scored_group(st, c) for st in ("hold", "place") for c in (2, 4)}
    pls.update(override)
    return pls


# --------------------------------------------------------------- frozen rules, params, labels

def test_rule_texts_are_the_predeclared_ones_verbatim():
    assert sc.FLUSHPAD_PROBE_RULE_TEXT == PROBE_TEXT
    assert sc.FLUSHPAD_SCORED_RULE_TEXT == SCORED_TEXT


def test_frozen_rules_extend_the_unchanged_base_rules():
    pr, hr, pl = sc.FLUSHPAD_PROBE_RULE, sc.FLUSHPAD_HOLD_RULE, sc.FLUSHPAD_PLACE_RULE
    for k, v in sc.LIFT_HOLD_RULE.items():
        if k not in ("seeds", "pass_min"):
            assert pr[k] == v, k
        if k != "seeds":
            assert hr[k] == v, k
    for k, v in sc.LIFT_PLACE_RULE.items():
        if k != "seeds":
            assert pl[k] == v, k
    assert pr["tilt_rad"] == 0.78 and pr["protocol"] == "lift_hold"           # robot-fall clause unchanged
    assert pr["seeds"] == [120, 121, 122, 123, 124] and pr["crews"] == [2] and pr["pass_min"] == 4
    assert pr["gated_checks"] == ["cube_tilt", "no_fall", "finite", "no_sim_reset", "full_episode"]
    assert hr["seeds"] == list(range(20, 30)) and pl["seeds"] == list(range(30, 40))
    assert hr["pass_min"] == pl["pass_min"] == 8 and hr["crews"] == pl["crews"] == [2, 4]
    assert pr["cube_tilt_max_rad"] == hr["cube_tilt_max_rad"] == pl["cube_tilt_max_rad"] == 0.35
    # the stock rules themselves are unchanged
    assert sc.LIFT_HOLD_RULE["seeds"] == list(range(10)) and sc.LIFT_PLACE_RULE["seeds"] == list(range(10, 20))
    assert sc.PROTOCOLS == ("carry", "lift_hold", "lift_place")
    # seed sets never overlap a used or reserved set
    used = set(range(20)) | set(sc.FLUSHPAD_SMOKE_SEEDS)
    for s in (pr["seeds"], hr["seeds"], pl["seeds"]):
        assert not used & set(s)
    assert not set(pr["seeds"]) & set(hr["seeds"] + pl["seeds"])
    assert set(sc.FLUSHPAD_SMOKE_SEEDS) == set(range(100, 120))


def test_params_are_opt_in_subclasses_with_the_frozen_values():
    stock, flush = sc.CarryParams(), sc.FlushPadParams()
    assert isinstance(flush, sc.CarryParams) and not sc.is_flushpad(stock) and sc.is_flushpad(flush)
    sd, fd = sc.params_dict(stock), sc.params_dict(flush)
    assert {k: fd[k] for k in sd} == sd                                       # every stock field and default
    assert {k: v for k, v in fd.items() if k not in sd} == {
        "pad_geometry": "flush", "pad_condim": 4, "pad_torsional_friction": 0.04,
        "solver_cone": "elliptic", "solver_impratio": 10.0}
    assert set(sd) == {                                                        # the stock key set (pinned elsewhere)
        "side_off", "cube_fwd", "cube_z", "plinth_half", "pair_pitch", "hand_pads", "pad_friction",
        "t_settle", "t_reach", "t_squeeze", "t_lift", "t_hold", "carry_speed", "carry_goal_m",
        "carry_gate_lift_m", "k_sync", "v_max", "v_min", "grasp_kp", "grasp_arm_mode",
        "outer_arm_mode", "reset_jitter", "xy_jitter"}
    q, q0 = sc.place_params_dict(sc.FlushPadPlaceParams()), sc.place_params_dict(sc.PlaceParams())
    assert q["lower_to"] is None and {k: v for k, v in q.items() if k != "lower_to"} == {
        k: v for k, v in q0.items() if k != "lower_to"}
    assert sc.PlaceParams().lower_to == (0.6, 1.1, 0.0, 1.0, 0.0)            # stock lowering unchanged
    script = sc.PlaceScript(sc.FlushPadParams(), sc.FlushPadPlaceParams())
    assert np.allclose(script.lower_target, sc.KEYFRAMES_LEFT["squeeze"])     # reverse keyframe: lift -> squeeze
    assert "reverse-keyframe" in sc.describe_lower_target(script.lower_target)
    assert (PLACE_TIMES["unload"], PLACE_TIMES["release"]) == (10.7, 11.7)


def test_labels_say_modified_end_effector_and_harness_change():
    assert sc.FLUSHPAD_LABEL.startswith(
        "LEARNED gait (frozen arms-dr1.0-s0) + SCRIPTED arms + ORACLE cube pose (scoring only)")
    assert "MODIFIED END-EFFECTOR" in sc.FLUSHPAD_LABEL and "not the stock robot" in sc.FLUSHPAD_NOTE
    assert "foot-floor" in sc.FLUSHPAD_HARNESS_NOTE and "robot-fall clauses are kept unchanged" in \
        sc.FLUSHPAD_HARNESS_NOTE
    head = sc._flushpad_head("probe")
    assert head["label"] == sc.FLUSHPAD_LABEL and head["end_effector"] == sc.FLUSHPAD_NOTE
    assert any("120-124" in s for s in head["seed_evidence"]) and any("20-39" in s for s in head["seed_evidence"])
    assert any("never lifts the cube" in s for s in head["clauses_as_applied"])
    assert any("auto-reset" in s and "no_sim_reset" in s and "written as null" in s
               for s in head["clauses_as_applied"])


# --------------------------------------------------------------- the tilt clause and its windows

@pytest.mark.parametrize("value,ok", [(0.35, True), (0.35 + 1e-12, False), (0.0, True), (1.571, False)])
def test_probe_tilt_clause_boundary(value, ok):
    ep = _episode(120, [_hold_row(series=_series(0.1, spikes={250: value}))])
    c = sc.flushpad_clauses(sc.FLUSHPAD_PROBE_RULE, ep, 0, DT)
    assert c["checks"]["cube_tilt"] is ok and c["success"] is ok


@pytest.mark.parametrize("i", [0, 1, 499, 500])
def test_probe_tilt_window_is_the_whole_episode(i):
    ep = _episode(120, [_hold_row(series=_series(0.1, spikes={i: 0.36}))])
    assert not sc.flushpad_clauses(sc.FLUSHPAD_PROBE_RULE, ep, 0, DT)["checks"]["cube_tilt"]


def test_probe_tilt_clause_fails_on_non_finite_and_unlogged():
    ep = _episode(120, [_hold_row(series=_series(0.1, spikes={10: None}))])
    assert not sc.flushpad_clauses(sc.FLUSHPAD_PROBE_RULE, ep, 0, DT)["checks"]["cube_tilt"]
    ep = _episode(120, [_hold_row(series=_series(0.1)[:-1])])                   # steps + 0 long
    c = sc.flushpad_clauses(sc.FLUSHPAD_PROBE_RULE, ep, 0, DT)
    assert not c["tilt_logged"] and not c["success"]


@pytest.mark.parametrize("max_tilt,ok", [(0.78, True), (0.7800001, False)])
def test_probe_keeps_the_robot_fall_clause_unchanged(max_tilt, ok):
    ep = _episode(120, [_hold_row(max_tilt=max_tilt)])
    c = sc.flushpad_clauses(sc.FLUSHPAD_PROBE_RULE, ep, 0, DT)
    assert c["checks"]["no_fall"] is ok and c["success"] is ok


def test_probe_does_not_gate_lift_hold_or_floor_disclosed():
    # a cube that is never lifted (and touches the floor) passes the probe's clauses: disclosed, not fixed
    ep = _episode(120, [_hold_row(peak=0.0, floor=True)])
    c = sc.flushpad_clauses(sc.FLUSHPAD_PROBE_RULE, ep, 0, DT)
    assert c["success"] and not ep["pairs"][0]["success"]


@pytest.mark.parametrize("i,ok", [(149, True), (150, False), (399, False), (400, True)])
def test_hold_tilt_window_is_the_scored_hold(i, ok):
    row = _hold_row(up=(6.0, 16.0), series=_series(0.1, spikes={i: 0.9}))
    assert (row["hold_start_s"], row["hold_end_s"]) == (6.0, 16.0)            # states 150..399
    c = sc.flushpad_clauses(sc.FLUSHPAD_HOLD_RULE, _episode(20, [row]), 0, DT)
    assert c["checks"]["cube_tilt_hold"] is ok and c["checks"]["rule"]


def test_hold_needs_the_unchanged_rule_too():
    row = _hold_row(floor=True)                                                # tilt fine, floor contact
    c = sc.flushpad_clauses(sc.FLUSHPAD_HOLD_RULE, _episode(20, [row]), 0, DT)
    assert c["checks"]["cube_tilt_hold"] and not c["checks"]["rule"] and not c["success"]
    row = _hold_row(peak=0.04)                                                 # never >= 5 cm: no hold window
    c = sc.flushpad_clauses(sc.FLUSHPAD_HOLD_RULE, _episode(20, [row]), 0, DT)
    assert not c["checks"]["cube_tilt_hold"] and not c["success"]


def test_release_window_states():
    assert sc.flushpad_release_window(PLACE_TIMES, DT) == (268, 293)


@pytest.mark.parametrize("i,ok", [(267, True), (268, False), (293, False), (294, True)])
def test_place_release_window_boundaries(i, ok):
    row = _place_row(series=_series(0.1, spikes={i: 0.9}))
    c = sc.flushpad_clauses(sc.FLUSHPAD_PLACE_RULE, _episode(30, [row], place=True), 0, DT)
    assert c["checks"]["cube_tilt_release"] is ok and c["checks"]["cube_tilt_seated"] and c["checks"]["rule"]


@pytest.mark.parametrize("value,ok", [(0.35, True), (0.3500001, False)])
def test_place_seated_tilt_is_the_final_state(value, ok):
    row = _place_row(series=_series(0.1, spikes={N: value}))
    c = sc.flushpad_clauses(sc.FLUSHPAD_PLACE_RULE, _episode(30, [row], place=True), 0, DT)
    assert c["checks"]["cube_tilt_seated"] is ok and c["success"] is ok


def test_place_needs_the_unchanged_rule_too():
    row = _place_row(contact=True)                                             # not released
    c = sc.flushpad_clauses(sc.FLUSHPAD_PLACE_RULE, _episode(30, [row], place=True), 0, DT)
    assert c["checks"]["cube_tilt_release"] and not c["checks"]["rule"] and not c["success"]


# --------------------------------------------------------------- numerical blow-ups: MuJoCo auto-reset, non-finite stop

def _clause_case(rule):
    if rule["clauses"] == "place":
        return _place_row, 30, True
    return _hold_row, (120 if rule["clauses"] == "probe" else 20), False


@pytest.mark.parametrize("rule", [sc.FLUSHPAD_PROBE_RULE, sc.FLUSHPAD_HOLD_RULE, sc.FLUSHPAD_PLACE_RULE],
                         ids=["probe", "hold", "place"])
def test_every_clause_set_fails_an_auto_reset_episode(rule):
    make, seed, place = _clause_case(rule)
    ok = sc.flushpad_clauses(rule, _episode(seed, [make()], place=place), 0, DT)
    assert ok["success"] and ok["checks"]["no_sim_reset"] and ok["sim_reset_logged"]
    bad = sc.flushpad_clauses(rule, _episode(seed, [make()], place=place, sim_reset=_sim_reset(True, 300)), 0, DT)
    assert not bad["success"] and bad["first_failed_check"] == "no_sim_reset" and bad["sim_reset_detected"]
    # only the new check fails: the stock row (its rule, its 'finite') cannot see an auto-reset
    assert [n for n, v in bad["checks"].items() if not v] == ["no_sim_reset"]


@pytest.mark.parametrize("record", [None, {}, {"detected": None}, {"detected": "no"}, "clear"])
def test_an_absent_or_malformed_auto_reset_record_is_missing_not_a_pass(record):
    pl = _probe(5)
    if record is None:
        del pl["episodes"][1]["sim_reset"]
    else:
        pl["episodes"][1]["sim_reset"] = record
    c = sc.flushpad_clauses(sc.FLUSHPAD_PROBE_RULE, pl["episodes"][1], 0, DT)
    assert not c["sim_reset_logged"] and not c["checks"]["no_sim_reset"] and not c["success"]
    v = sc.flushpad_probe_verdict(pl)
    assert v["verdict"] == "INCOMPLETE" and v["missing_seeds"] == [121]
    g = _scored_group("hold", 2)
    del g["episodes"][0]["sim_reset"]
    assert sc.flushpad_scored_verdict(_scored(hold_crew2=g))["verdict"] == "INCOMPLETE"


def test_probe_verdict_counts_auto_resets_as_failures():
    pl = _probe(5)
    pl["episodes"][4]["sim_reset"] = _sim_reset(True, 50)
    v = sc.flushpad_probe_verdict(pl)
    assert v["verdict"] == "PROCEED" and v["successes"] == 4 and v["missing_seeds"] == []
    pl["episodes"][0]["sim_reset"] = _sim_reset(True, 499)
    v = sc.flushpad_probe_verdict(pl)
    assert v["verdict"] == "NEGATIVE" and v["successes"] == 3
    assert [r["first_failed_check"] for r in v["per_seed"]] == ["no_sim_reset", None, None, None, "no_sim_reset"]


def test_scored_verdict_counts_auto_resets_as_failures():
    g = _scored_group("place", 4)
    for e in g["episodes"][:3]:
        e["sim_reset"] = _sim_reset(True, 100)
    v = sc.flushpad_scored_verdict(_scored(place_crew4=g))
    assert v["verdict"] == "NEGATIVE"
    assert {(x["crew"], x["pair"]): x["successes"] for x in v["groups"] if x["stage"] == "place"} == {
        (2, 0): 10, (4, 0): 7, (4, 1): 7}


def test_json_safe_nulls_non_finite_numbers_only():
    x = {"a": [1.0, float("nan"), np.float64("inf"), -np.inf, 2], "b": {"c": np.float32(0.5), "d": (1, 2)},
         "e": np.array([0.25, np.nan]), "f": True, "g": None, "h": "x", "i": np.int64(3), "j": np.bool_(False)}
    y, n = sc._json_safe(x)
    assert n == 4
    assert y == {"a": [1.0, None, None, None, 2], "b": {"c": 0.5, "d": [1, 2]}, "e": [0.25, None], "f": True,
                 "g": None, "h": "x", "i": 3, "j": False}
    assert type(y["i"]) is int and type(y["j"]) is bool and type(y["a"][4]) is int
    json.dumps(y, allow_nan=False)                                             # writable as the score JSON is


_BOX_XML = """<mujoco><option timestep="0.005"/><worldbody><geom name="floor" type="plane" size="2 2 0.1"/>
<body name="box" pos="0.3 0.2 0.5"><freejoint/><geom type="box" size="0.05 0.05 0.05" mass="1"/></body>
</worldbody></mujoco>"""


@pytest.mark.parametrize("field,value", [("qvel", 1e11), ("qvel", float("nan")), ("qpos", float("nan")),
                                         ("qvel", -2e10)])
def test_sim_reset_signal_catches_mujocos_silent_auto_reset(field, value, tmp_path, monkeypatch):
    import mujoco
    monkeypatch.chdir(tmp_path)                     # MuJoCo appends its warnings to ./MUJOCO_LOG.TXT: not the repo
    m = mujoco.MjModel.from_xml_string(_BOX_XML)
    assert not m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)    # auto-reset on, as in the harness
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    assert sc.sim_reset_signal(m, d, 0, 1) is None
    for _ in range(40):
        mujoco.mj_step(m, d)
    assert sc.sim_reset_signal(m, d, 40, 1) is None and sc.sim_reset_signal(m, d, 5, 8) is None   # 40 = 5 x 8
    getattr(d, field)[2] = value
    mujoco.mj_step(m, d)
    # MuJoCo reset the state by itself: the stock non-finite check sees nothing, and the box is back at qpos0
    assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
    assert d.qpos[2] > 0.49 and d.time == pytest.approx(0.005)
    sig = sc.sim_reset_signal(m, d, 41, 1)
    assert sig is not None and sig["time_mismatch"]
    bad = "mjWARN_BADQPOS" if field == "qpos" else "mjWARN_BADQVEL"
    assert sig["bad_number_warnings"][bad] >= 1                               # the counter survives the reset
    # a value below mjMAXVAL (1e10) is not a MuJoCo bad number: no reset, no signal
    d2 = mujoco.MjData(m)
    mujoco.mj_resetData(m, d2)
    d2.qvel[0] = 1e9
    mujoco.mj_step(m, d2)
    assert sc.sim_reset_signal(m, d2, 1, 1) is None


# --------------------------------------------------------------- stage 1: probe verdict

def test_probe_verdict_boundaries():
    assert sc.flushpad_probe_verdict(_probe(5))["verdict"] == "PROCEED"
    v = sc.flushpad_probe_verdict(_probe(4))
    assert v["verdict"] == "PROCEED" and v["successes"] == 4
    v = sc.flushpad_probe_verdict(_probe(3))
    assert v["verdict"] == "NEGATIVE" and v["successes"] == 3 and "does not stop the roll" in v["reading"]
    # a fall counts against the seed even with the cube flat
    v = sc.flushpad_probe_verdict(_probe(3, bad=lambda: _hold_row(max_tilt=0.9)))
    assert v["verdict"] == "NEGATIVE"
    assert v["rule_text"] == PROBE_TEXT and v["label"] == sc.FLUSHPAD_LABEL


def test_probe_verdict_incomplete_and_failed_episodes():
    pl = _probe(5)
    pl["episodes"] = pl["episodes"][:4]                                        # seed 124 missing
    v = sc.flushpad_probe_verdict(pl)
    assert v["verdict"] == "INCOMPLETE" and v["missing_seeds"] == [124]
    pl = _probe(5)
    pl["episodes"][2]["steps"] = 499                                           # short, no recorded failure
    assert sc.flushpad_probe_verdict(pl)["verdict"] == "INCOMPLETE"
    pl = _probe(5)
    del pl["episodes"][0]["pairs"][0]["cube_tilt_series_rad"]                  # no tilt log
    assert sc.flushpad_probe_verdict(pl)["verdict"] == "INCOMPLETE"
    assert sc.flushpad_probe_verdict(None)["verdict"] == "INCOMPLETE"
    # a non-finite stop is a scored failure, not a missing episode
    pl = _probe(5)
    row = _hold_row(n=300, failed="nonfinite_state", series=_series(0.1, n=300))
    pl["episodes"][4] = _j(_episode(124, [row], steps=300, failed="nonfinite_state"))
    v = sc.flushpad_probe_verdict(pl)
    assert v["verdict"] == "PROCEED" and v["successes"] == 4 and v["missing_seeds"] == []
    pl["episodes"][3] = _j(_episode(123, [row], steps=300, failed="nonfinite_state"))
    assert sc.flushpad_probe_verdict(pl)["verdict"] == "NEGATIVE"


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(stage="hold"),
    lambda p: p.update(scored_run=False),
    lambda p: p.update(crew=4),
    lambda p: p["rule"].update(pass_min=3),
    lambda p: p["rule"].update(cube_tilt_max_rad=0.5),
    lambda p: p["params"].update(pad_torsional_friction=0.02),
    lambda p: p["keyframes_left"].update(squeeze=[0.0, 1.0, 0.0, 0.5, 0.0]),
    lambda p: p.update(model_check=_model_check(2, condim=3)),
    lambda p: p.update(model_check=_model_check(2, cone="pyramidal")),
    lambda p: p.update(model_check=_model_check(2, impratio=1.0)),
    lambda p: p.update(model_check=_model_check(2, quat=[1.0, 0.0, 0.0, 0.0])),     # pads not the designed ones
    lambda p: p.update(variant="stock"),
])
def test_probe_verdict_invalid_json_never_proceeds(mutate):
    pl = _probe(5)
    mutate(pl)
    v = sc.flushpad_probe_verdict(pl)
    assert v["verdict"] == "INVALID"


# --------------------------------------------------------------- stage 2: scored verdict

def test_scored_verdict_pass_at_eight_of_ten_everywhere():
    assert sc.flushpad_scored_verdict(_scored())["verdict"] == "PASS"
    pls = {k: _scored_group(k.split("_")[0], int(k[-1]), n_good=8) for k in sc.FLUSHPAD_SCORED_KEYS}
    v = sc.flushpad_scored_verdict(pls)
    assert v["verdict"] == "PASS" and len(v["groups"]) == 6
    assert all(g["successes"] == 8 and g["episodes"] == 10 for g in v["groups"])
    assert v["rule_text"] == SCORED_TEXT


@pytest.mark.parametrize("key", sc.FLUSHPAD_SCORED_KEYS)
def test_scored_verdict_negative_at_seven_of_ten(key):
    st, crew = key.split("_")[0], int(key[-1])
    v = sc.flushpad_scored_verdict(_scored(**{key: _scored_group(st, crew, n_good=7)}))
    assert v["verdict"] == "NEGATIVE"


def test_scored_verdict_gates_each_pair_of_crew4():
    v = sc.flushpad_scored_verdict(_scored(hold_crew4=_scored_group("hold", 4, n_good=7, fail=(1,))))
    assert v["verdict"] == "NEGATIVE"
    g = {(x["stage"], x["crew"], x["pair"]): x for x in v["groups"]}
    assert g[("hold", 4, 0)]["pass"] and not g[("hold", 4, 1)]["pass"]


def test_scored_verdict_counts_base_rule_failures():
    pl = _scored_group("place", 2)
    for e in pl["episodes"][:3]:                                               # tilt fine, cube not released
        e["pairs"][0] = _j(_place_row(contact=True))
    v = sc.flushpad_scored_verdict(_scored(place_crew2=pl))
    assert v["verdict"] == "NEGATIVE"


def test_scored_verdict_incomplete_and_invalid():
    v = sc.flushpad_scored_verdict(_scored(hold_crew4=None))
    assert v["verdict"] == "INCOMPLETE"
    pl = _scored_group("place", 4)
    pl["episodes"] = pl["episodes"][:9]
    assert sc.flushpad_scored_verdict(_scored(place_crew4=pl))["verdict"] == "INCOMPLETE"
    pl = _scored_group("place", 2)
    pl["place_params"]["lower_to"] = [0.6, 1.1, 0.0, 1.0, 0.0]                # the stock (tuned) lowering
    assert sc.flushpad_scored_verdict(_scored(place_crew2=pl))["verdict"] == "INVALID"
    assert sc.flushpad_scored_verdict(_scored(hold_crew2=_scored_group("hold", 4)))["verdict"] == "INVALID"
    v = sc.flushpad_not_run_verdict({"verdict": "NEGATIVE"})
    assert v["verdict"] == "NOT_RUN" and "seeds 20-39 stay unused" in v["reading"]


# --------------------------------------------------------------- smoke check

def _smoke(protocol="lift_hold", crew=2, seed=TUNING_SEED):
    rows = [(_place_row(k) if protocol == "lift_place" else _hold_row(k)) for k in range(crew // 2)]
    ep = _episode(seed, rows, place=protocol == "lift_place")
    for k, r in enumerate(rows):
        for j, tr in enumerate(ep["trace"]):
            tr["cube_tilt_rad"] = [round(rr["cube_tilt_series_rad"][25 * j + 1], 3) for rr in rows]
    return _payload("smoke", crew, [ep], protocol=protocol)


def test_smoke_check_passes_a_good_smoke():
    assert sc.flushpad_smoke_check(_smoke())["verdict"] == "SMOKE_PASS"
    assert sc.flushpad_smoke_check(_smoke("lift_place", 4))["verdict"] == "SMOKE_PASS"


@pytest.mark.parametrize("mutate", [
    lambda p: p["episodes"][0].update(steps=499),
    lambda p: p["episodes"][0].update(failed="nonfinite_state"),
    lambda p: p["episodes"][0]["pairs"][0]["cube_tilt_series_rad"].pop(),
    lambda p: p["episodes"][0]["pairs"][0]["cube_tilt_series_rad"].__setitem__(7, None),
    lambda p: p["episodes"][0]["pairs"][0]["cube_tilt_series_rad"].__setitem__(26, 0.5),   # disagrees with trace
    lambda p: p["episodes"][0]["pairs"][0]["cube_tilt_series_rad"].__setitem__(0, 0.2),    # not flat at reset
    lambda p: p["episodes"][0].update(trace=[]),
    lambda p: p.update(model_check=_model_check(2, condim=3)),
    lambda p: p.update(scored_run=True),
    lambda p: p["rule"].update(episode_s=1.0),
    lambda p: p["params"].update(solver_impratio=1.0),
    lambda p: p["episodes"][0].pop("sim_reset"),                                    # auto-reset check not wired
    lambda p: p["episodes"][0].update(sim_reset=_sim_reset(True, 120)),             # a blown-up smoke episode
    lambda p: p["episodes"][0]["trace"][3]["cube_tilt_rad"].__setitem__(0, None),   # nulled trace value
])
def test_smoke_check_fails_on_pipeline_faults(mutate):
    pl = _smoke()
    mutate(pl)
    assert sc.flushpad_smoke_check(pl)["verdict"] == "SMOKE_FAIL"


@pytest.mark.parametrize("seed", [120, 124, 20, 39, 0])
def test_smoke_check_refuses_non_tuning_seeds(seed):
    assert sc.flushpad_smoke_check(_smoke(seed=seed))["verdict"] == "SMOKE_FAIL"


def test_smoke_outcome_is_never_gated():
    pl = _smoke()
    pl["episodes"][0]["pairs"][0] = _j(_hold_row(series=_series(1.5), max_tilt=1.2))
    pl["episodes"][0]["trace"] = _trace(pl["episodes"][0]["pairs"][0]["cube_tilt_series_rad"])
    assert sc.flushpad_smoke_check(pl)["verdict"] == "SMOKE_PASS"


# --------------------------------------------------------------- CLI guard and verdict writer

def _args(tmp_path, stage, seeds, protocol="lift_hold", crew=2, seconds=None, out="s.json"):
    import argparse
    return argparse.Namespace(flushpad_stage=stage, render_from=None, out=tmp_path / out, seeds=seeds,
                              protocol=protocol, crew=crew, seconds=seconds)


@pytest.mark.parametrize("stage,seeds,protocol,crew", [
    ("probe", "120-124", "lift_hold", 2), ("hold", "20-29", "lift_hold", 2), ("hold", "20-29", "lift_hold", 4),
    ("place", "30-39", "lift_place", 2), ("place", "30-39", "lift_place", 4),
    ("smoke", "119", "lift_hold", 2), ("smoke", "100-119", "lift_place", 4)])
def test_guard_accepts_exactly_the_declared_runs(tmp_path, stage, seeds, protocol, crew):
    rule, proto, s = _script().flushpad_guard(_args(tmp_path, stage, seeds, protocol, crew))
    assert proto == protocol and s == sorted(s) and rule["seeds"] == s
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("stage,seeds,protocol,crew,seconds", [
    ("probe", "119-123", "lift_hold", 2, None), ("probe", "120-123", "lift_hold", 2, None),
    ("probe", "120-124", "lift_hold", 4, None), ("probe", "120-124", "lift_place", 2, None),
    ("probe", "120-124", "lift_hold", 2, 5.0), ("hold", "20-28", "lift_hold", 2, None),
    ("hold", "20-29", "lift_place", 2, None), ("place", "30-39", "lift_hold", 2, None),
    ("place", "20-29", "lift_place", 2, None), ("smoke", "120", "lift_hold", 2, None),
    ("smoke", "124", "lift_hold", 2, None), ("smoke", "20", "lift_hold", 2, None),
    ("smoke", "39", "lift_place", 2, None), ("smoke", "119,120", "lift_hold", 2, None),
    ("smoke", "119", "carry", 2, None)])
def test_guard_refuses_everything_else(tmp_path, stage, seeds, protocol, crew, seconds):
    with pytest.raises(SystemExit, match="REFUSED"):
        _script().flushpad_guard(_args(tmp_path, stage, seeds, protocol, crew, seconds))
    assert list(tmp_path.iterdir()) == []


def test_guard_and_score_refuse_to_overwrite(tmp_path):
    mod = _script()
    out = tmp_path / "score_probe_crew2.json"
    out.write_text("{}")
    with pytest.raises(SystemExit, match="REFUSED"):
        mod.score_flushpad(_args(tmp_path, "probe", "120-124", out=out.name))
    assert out.read_text() == "{}"


def test_verdict_cli_writes_once_from_json(tmp_path, capsys):
    mod = _script()
    src = tmp_path / "score_probe_crew2.json"
    src.write_text(json.dumps(_probe(4)))
    out = tmp_path / "probe" / "verdict.json"
    assert mod.flushpad_verdict_cli(["probe", str(out), f"probe={src}"]) == 0
    v = json.loads(out.read_text())
    assert v["verdict"] == "PROCEED" and v["inputs"]["probe"]["sha256"] and v["rule_text"] == PROBE_TEXT
    assert "COOP-FLUSHPAD probe: PROCEED" in capsys.readouterr().out
    before = out.read_text()
    with pytest.raises(SystemExit, match="REFUSED"):
        mod.flushpad_verdict_cli(["probe", str(out), f"probe={src}"])
    assert out.read_text() == before
    # a missing input reads as missing; the scored stage reads NOT_RUN unless the probe proceeded
    out2 = tmp_path / "v2.json"
    mod.flushpad_verdict_cli(["probe", str(out2), f"probe={tmp_path / 'absent.json'}"])
    assert json.loads(out2.read_text())["verdict"] == "INCOMPLETE"
    out3 = tmp_path / "v3.json"
    mod.flushpad_verdict_cli(["scored_not_run", str(out3), f"probe_verdict={out2}"])
    v3 = json.loads(out3.read_text())
    assert v3["verdict"] == "NOT_RUN" and v3["probe_verdict"] == "INCOMPLETE"


def test_verdict_cli_scored_and_smoke(tmp_path):
    mod = _script()
    args = []
    for k, pl in _scored().items():
        (tmp_path / f"{k}.json").write_text(json.dumps(pl))
        args.append(f"{k}={tmp_path / (k + '.json')}")
    mod.flushpad_verdict_cli(["scored", str(tmp_path / "scored.json"), *args])
    assert json.loads((tmp_path / "scored.json").read_text())["verdict"] == "PASS"
    mod.flushpad_verdict_cli(["scored", str(tmp_path / "scored3.json"), *args[:3]])
    assert json.loads((tmp_path / "scored3.json").read_text())["verdict"] == "INCOMPLETE"
    (tmp_path / "sm.json").write_text(json.dumps(_smoke()))
    ok = tmp_path / "step_status.json"
    ok.write_text('{"pytest": 0, "smoke_hold_crew2": 0, "smoke_place_crew4": 0}\n')    # as the launcher writes it
    mod.flushpad_verdict_cli(["smoke", str(tmp_path / "smv.json"), f"step_status={ok}",
                              f"hold_crew2={tmp_path / 'sm.json'}"])
    sv = json.loads((tmp_path / "smv.json").read_text())
    assert sv["verdict"] == "SMOKE_PASS" and sv["unit_tests_passed"] is True and sv["step_status"]["pytest"] == 0
    assert set(sv["code_sha256"]) == {"scripted_carry.py", "coop_scripted_carry.py"}
    assert set(sv["inputs"]) == {"step_status", "hold_crew2"}
    mod.flushpad_verdict_cli(["smoke", str(tmp_path / "smv2.json"), f"step_status={ok}",
                              f"hold_crew2={tmp_path / 'sm.json'}", f"place_crew4={tmp_path / 'absent.json'}"])
    assert json.loads((tmp_path / "smv2.json").read_text())["verdict"] == "SMOKE_FAIL"
    # the same good episodes never read SMOKE_PASS without a step-status record or with failed unit tests
    mod.flushpad_verdict_cli(["smoke", str(tmp_path / "smv3.json"), f"hold_crew2={tmp_path / 'sm.json'}"])
    v3 = json.loads((tmp_path / "smv3.json").read_text())
    assert v3["verdict"] == "SMOKE_FAIL" and v3["unit_tests_passed"] is False
    bad = tmp_path / "step_status_bad.json"
    bad.write_text('{"pytest": 1, "smoke_hold_crew2": 0, "smoke_place_crew4": 0}\n')
    mod.flushpad_verdict_cli(["smoke", str(tmp_path / "smv4.json"), f"step_status={bad}",
                              f"hold_crew2={tmp_path / 'sm.json'}"])
    v4 = json.loads((tmp_path / "smv4.json").read_text())
    assert v4["verdict"] == "SMOKE_FAIL" and v4["unit_tests_passed"] is False
    assert any("pytest" in p for p in v4["problems"])


@pytest.mark.parametrize("status,ok", [
    ({"pytest": 0, "smoke_hold_crew2": 0, "smoke_place_crew4": 0}, True),
    ({"pytest": 1, "smoke_hold_crew2": 0, "smoke_place_crew4": 0}, False),
    ({"pytest": 5, "smoke_hold_crew2": 0, "smoke_place_crew4": 0}, False),          # pytest: no tests collected
    ({"smoke_hold_crew2": 0, "smoke_place_crew4": 0}, False),                       # unit tests not recorded
    ({"pytest": False, "smoke_hold_crew2": 0, "smoke_place_crew4": 0}, False),
    ({"pytest": "0", "smoke_hold_crew2": 0, "smoke_place_crew4": 0}, False),
    ({"pytest": 0, "smoke_hold_crew2": 0, "smoke_place_crew4": 1}, False),
    ({"pytest": 0, "smoke_hold_crew2": 0}, False),
    ({"pytest": 0, "smoke_hold_crew2": 0, "smoke_place_crew4": 0, "other": 2}, False),
    (None, False), ([0, 0, 0], False)])
def test_smoke_step_status_boundaries(status, ok):
    assert (sc.flushpad_smoke_step_problems(status) == []) is ok


# --------------------------------------------------------------- launcher

def _header_text():
    return _norm(" ".join(ln.lstrip("#").strip() for ln in LAUNCHER.read_text().splitlines() if ln.startswith("#")))


def test_launcher_header_states_the_frozen_rules_verbatim_and_the_labels():
    head = _header_text()
    assert _norm(PROBE_TEXT) in head and _norm(SCORED_TEXT) in head
    for s in ("LEARNED gait (frozen arms-dr1.0-s0) + SCRIPTED arms + ORACLE cube pose (scoring only)",
              "MODIFIED END-EFFECTOR -- NOT THE STOCK ROBOT", "HARNESS CHANGE: elliptic friction cone + impratio 10",
              "foot-floor contact", "the reverse keyframe, lift -> squeeze", "CLAUSES AS APPLIED", "SEED EVIDENCE",
              "never lifts the cube", "expected to read NEGATIVE", "no MuJoCo auto-reset, full 20 s",
              "check no_sim_reset", "written as null", "no tilt log or no auto-reset record is MISSING",
              "never reuse an old smoke id", "step_status.json"):
        assert _norm(s) in head, s


def test_launcher_runs_only_the_declared_seeds_and_reads_verdicts_from_json():
    text = LAUNCHER.read_text()
    assert "#SBATCH --time=01:00:00" in text                                    # a smoke never exceeds 1 h
    smoke = text.split('if [ "$MODE" = smoke ]; then', 1)[1].split("\nfi\n", 1)[0]
    assert re.findall(r"--seeds (\S+)", smoke) == ["119", "119"]
    run = text.split("\nfi\n", 1)[1]
    assert re.findall(r"--seeds (\S+)", run) == ["120-124", "20-29", "30-39"]
    assert 'if [ "$PV" = PROCEED ]; then' in run and "scored_not_run" in run
    assert "results are never overwritten" in run and 'if [ -e "$RUN" ]' in run
    assert "FLUSHPAD_SMOKE_JOB" in run and "sha256sum -c" in run                # bytes check against the smoke
    assert "*smoke*)" in smoke and "-smoke" in smoke                            # job name and output dir
    assert "|| true" not in text                                                # no swallowed exit status
    assert subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True).returncode == 0
    # the smoke records every step's exit status (unit tests first) and feeds it to its verdict JSON
    assert '"pytest": %d, "smoke_hold_crew2": %d, "smoke_place_crew4": %d' in smoke
    assert '"$t" "$s1" "$s2"' in smoke and 'step_status="$SM/step_status.json"' in smoke
    assert smoke.index("pytest -q") < smoke.index("step_status.json") < smoke.index("write_verdict smoke")
    # the run reads the recorded unit-test status from the smoke verdict JSON, and the bytes check covers the tests
    assert 'PT=$(pytest_status_of "$SM/smoke_verdict.json")' in run and 'if [ "$PT" != 0 ]; then' in run
    assert ("CODE=(src/bhl_robust/eval/scripted_carry.py scripts/bench/coop_scripted_carry.py "
            "tests/test_coop_flushpad.py)") in text


def _launch(launcher, args, extra, unset=()):
    import os
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_") and k not in unset}
    env.update(extra)
    return subprocess.run(["bash", str(launcher), *args], capture_output=True, text=True, env=env, cwd=REPO,
                          timeout=300)


def test_launcher_refuses_without_a_mode_a_smoke_name_or_a_passing_smoke(tmp_path):
    # Exercise the unchanged launcher body against isolated paths. Refusal
    # checks must neither import another checkout nor write cluster log dirs.
    logs = tmp_path / "logs"
    logs.mkdir()
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text("# present for launcher preflight; scoring must never run\n")
    launcher = tmp_path / LAUNCHER.name
    text = LAUNCHER.read_text()
    replacements = {
        "REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder\n": f"REPO={REPO}\n",
        "LOGS=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs\n": f"LOGS={logs}\n",
        "PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python\n": f"PY={sys.executable}\n",
        "DEPLOY=$U/logs/rsl_rl/humanoid/2026-08-18_20-57-50_arms-dr1.0-s0/exported/deploy.yaml\n":
            f"DEPLOY={deploy}\n",
    }
    for original, replacement in replacements.items():
        assert text.count(original) == 1
        text = text.replace(original, replacement)
    launcher.write_text(text)
    run = LAUNCHER.read_text().split("\nfi\n", 1)[1]
    # first the order that makes the dynamic checks below safe: the smoke and bytes checks come before the run
    # directory is created and before anything is scored
    assert (run.index('if [ -z "$SMOKE_JOB" ]') < run.index('if [ "$SV" != SMOKE_PASS ]')
            < run.index('if [ "$PT" != 0 ]') < run.index("sha256sum -c") < run.index('if [ -e "$RUN" ]')
            < run.index("mkdir -p") < run.index("score --flushpad-stage"))
    r = _launch(launcher, [], {})
    assert r.returncode == 1 and "REFUSED (usage" in r.stdout
    r = _launch(launcher, ["smoke"], {"SLURM_JOB_NAME": "coop-flushpad", "SLURM_JOB_ID": "pytest-refusal"})
    assert r.returncode == 1 and "job name must contain 'smoke'" in r.stdout
    r = _launch(launcher, ["run"], {"SLURM_JOB_ID": "pytest-refusal"}, unset=("FLUSHPAD_SMOKE_JOB",))
    assert r.returncode == 1 and "needs --export=ALL,FLUSHPAD_SMOKE_JOB" in r.stdout
    r = _launch(launcher, ["run"], {"SLURM_JOB_ID": "pytest-refusal", "FLUSHPAD_SMOKE_JOB": "pytest-no-such-smoke"})
    assert r.returncode == 1 and "reads MISSING, not SMOKE_PASS" in r.stdout
    assert not (logs / "coop-flushpad-pytest-refusal-smoke").exists()
    # a smoke verdict JSON that reads SMOKE_PASS but records failed (or no) unit tests is refused before anything
    # else; the fake smoke directory has no code_sha256.txt, so even a broken check could not reach the probe
    import os
    import shutil
    fake_id = f"pytest-fake-{os.getpid()}"
    fake = logs / f"coop-flushpad-{fake_id}-smoke"
    assert not fake.exists()
    try:
        fake.mkdir(parents=True)
        for status, shown in (({"pytest": 1, "smoke_hold_crew2": 0, "smoke_place_crew4": 0}, "exit 1"),
                              (None, "exit MISSING")):
            (fake / "smoke_verdict.json").write_text(json.dumps({"verdict": "SMOKE_PASS", "step_status": status}))
            r = _launch(launcher, ["run"], {"SLURM_JOB_ID": "pytest-refusal", "FLUSHPAD_SMOKE_JOB": fake_id})
            assert r.returncode == 1 and f"records its unit tests as {shown}, not 0" in r.stdout, r.stdout
            assert "BYTES-CHECK" not in r.stdout and "STAGE 1" not in r.stdout
    finally:
        shutil.rmtree(fake, ignore_errors=True)


# --------------------------------------------------------------- stock path unchanged: source diff to 094797e

def _git_show(path):
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{BASE_COMMIT}:{path}"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr                                        # fail loudly, never skip
    return r.stdout


def _top_names(lines):
    return {m.group(2) for ln in lines for m in [re.match(r"^(def|class) (\w+)", ln)] if m}


OPT_IN_LINE = re.compile(r"flushpad|flush_pad|\bflush\b(?!=)", re.IGNORECASE)    # not print(..., flush=True)


@pytest.mark.parametrize("path,end_marker,n_branch", [
    ("src/bhl_robust/eval/scripted_carry.py", None, 7),
    ("scripts/bench/coop_scripted_carry.py", "# ------------------------------------------------------------------ render", 3),
])
def test_source_diff_to_base_commit_is_opt_in_insertions_only(path, end_marker, n_branch):
    old = _git_show(path).splitlines()
    new = (REPO / path).read_text().splitlines()
    assert not any(OPT_IN_LINE.search(ln) for ln in old)       # dropping opt-in lines cannot drop a stock line
    starts = [i for i, ln in enumerate(new) if ln.startswith("# ---") and "(C1, opt-in)" in ln]
    assert len(starts) == 1                                     # one marked opt-in section
    s = starts[0]
    e = len(new) if end_marker is None else new.index(end_marker, s)
    section, rest = new[s:e], new[:s] + new[e:]
    if end_marker is None:                                      # appended at the end: drop its separating blanks
        while rest and not rest[-1].strip():
            rest.pop()
    branch = [ln for ln in rest if OPT_IN_LINE.search(ln)]
    assert len(branch) == n_branch, branch                      # the opt-in branch lines in stock functions
    assert [ln for ln in rest if not OPT_IN_LINE.search(ln)] == old   # every stock line unchanged, in order
    assert not _top_names(section) & _top_names(old)            # the section shadows no stock name
    ops = difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
    assert {op for op, *_ in ops} <= {"equal", "insert"}        # and the diff is insertions only


# --------------------------------------------------------------- geometry (kinematics; needs the MJCF)

def _off_axis_deg(v) -> float:
    return float(np.degrees(np.arccos(np.clip(np.max(np.abs(np.asarray(v, dtype=float))), -1.0, 1.0))))


@pytest.fixture(scope="module")
def robot_xml(tmp_path_factory):
    from bhl_robust.eval.mjcf_assets import prepare_mjcf
    return prepare_mjcf(UPSTREAM, tmp_path_factory.mktemp("c1_mjcf"), "humanoid").parent / "berkeley_humanoid_lite.xml"


@needs_assets
def test_flush_pad_faces_are_parallel_to_the_cube_faces_at_the_squeeze_pose(robot_xml):
    p = sc.FlushPadParams()
    des = sc.flush_pad_frames(robot_xml, p)
    stock_frames = sc.hand_pad_frames(robot_xml)
    fk, fk_stock = sc.PadFK(robot_xml, des["frames"]), sc.PadFK(robot_xml, stock_frames)
    face = p.side_off - sc.CUBE_HALF
    for side in ("left", "right"):
        d = des["design"][side]
        q = np.asarray(d["squeeze_pose_joints"], dtype=float)
        sgn = 1.0 if side == "left" else -1.0
        assert np.allclose(q[[0, 2, 3, 4]], sgn * np.asarray(sc.KEYFRAMES_LEFT["squeeze"])[[0, 2, 3, 4]])
        c, R, _, h = fk.pose(side, q)
        # robot frame axes are the cube's face normals (the cube is axis-aligned, the robots yawed 90 deg)
        assert np.allclose(R, np.asarray(d["pad_axes_at_squeeze_robot_frame"], dtype=float), atol=1e-9)
        assert max(_off_axis_deg(R[:, j]) for j in range(3)) < 1e-4
        assert abs(R[1, d["contact_axis"]]) == pytest.approx(1.0, abs=1e-12)  # contact face normal = pinch axis
        assert fk.lateral_extent(side, q) == pytest.approx(face, abs=1e-9)     # outer face on the cube face plane
        _, Rs, _, _ = fk_stock.pose(side, q)
        assert max(_off_axis_deg(Rs[:, j]) for j in range(3)) > 10.0           # the stock pad is clearly off-face
        assert des["frames"][side]["pos"] == stock_frames[side]["pos"]         # same box, same centre
        assert des["frames"][side]["size"] == stock_frames[side]["size"]
        assert np.allclose(h, stock_frames[side]["size"])
    assert des["design"]["left"]["stock_pad_contact_normal_off_pinch_axis_deg_at_squeeze_keyframe"] == \
        pytest.approx(26.2, abs=0.1)                                           # the investigation's 26 deg


def _posed_squeeze(model, slots, pairs, design):
    import mujoco
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(DEPLOY)
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    for s in slots:
        d.qpos[s.qpos_adr + 7:s.qpos_adr + 7 + len(s.ctrl)] = np.asarray(cfg.default_joint_positions, dtype=float)
    pr = pairs[0]
    a0, b0 = slots[pr.robot_a].qpos_adr + 7, slots[pr.robot_b].qpos_adr + 7
    d.qpos[a0 + 0:a0 + 5] = design["design"]["left"]["squeeze_pose_joints"]
    d.qpos[b0 + 5:b0 + 10] = design["design"]["right"]["squeeze_pose_joints"]
    mujoco.mj_forward(model, d)
    for i, s in enumerate(slots):                                              # feet on the floor, as reset does
        mine = [g for g in range(model.ngeom) if model.geom_contype[g] and (mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "").startswith(s.prefix)]
        d.qpos[s.qpos_adr + 2] -= sc.lowest_point(model, d, mine) - 0.002
    mujoco.mj_forward(model, d)
    return d


@needs_assets
def test_flush_pads_on_the_harness_model_are_flush_with_the_cube(tmp_path):
    import mujoco
    p = sc.FlushPadParams()
    design = sc.flushpad_design(UPSTREAM, tmp_path, p)
    for params, flush in ((p, True), (sc.CarryParams(), False)):
        model, slots, pairs = sc.build_carry(UPSTREAM, tmp_path, 1, params)
        d = _posed_squeeze(model, slots, pairs, design)
        Rc, cc = d.xmat[pairs[0].cube_body].reshape(3, 3), d.xpos[pairs[0].cube_body]
        for name in ("r1_arm_left_hand_pad", "r0_arm_right_hand_pad"):     # robot a's left, robot b's right hand
            g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            M = Rc.T @ d.geom_xmat[g].reshape(3, 3)                            # pad axes in the cube frame
            off = max(_off_axis_deg(M[:, j]) for j in range(3))
            if not flush:
                assert off > 10.0, name
                continue
            assert off < 1e-3, name
            rel = Rc.T @ (d.geom_xpos[g] - cc)
            ext = np.abs(M) @ model.geom_size[g]
            assert abs(rel[0]) - ext[0] == pytest.approx(sc.CUBE_HALF, abs=1e-6)   # pad face on the cube face
            assert abs(rel[1]) < sc.CUBE_HALF and abs(rel[2]) < sc.CUBE_HALF        # and over it


def _array_fields(m):
    out = {}
    for name in dir(m):
        if name.startswith("_"):
            continue
        try:
            v = getattr(m, name)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(v, np.ndarray):
            out[name] = v
    return out


@needs_assets
def test_flush_model_differs_from_stock_only_where_declared(tmp_path):
    import mujoco
    ms, _, _ = sc.build_carry(UPSTREAM, tmp_path, 1, sc.CarryParams())
    mf, _, _ = sc.build_carry(UPSTREAM, tmp_path, 1, sc.FlushPadParams())
    fs, ff = _array_fields(ms), _array_fields(mf)
    assert set(fs) == set(ff)
    differ = {k for k in fs if fs[k].shape != ff[k].shape or not np.array_equal(fs[k], ff[k], equal_nan=fs[k].dtype.kind == "f")}
    assert differ == {"geom_quat", "geom_condim", "geom_friction", "bvh_aabb"}  # bvh_aabb: derived from the pads
    pads = {g for g in range(ms.ngeom)
            if (mujoco.mj_id2name(ms, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("_hand_pad")}
    assert len(pads) == 4
    for k in ("geom_quat", "geom_condim", "geom_friction"):
        rows = {int(g) for g in np.where(~np.all((fs[k] == ff[k]).reshape(ms.ngeom, -1), axis=1))[0]}
        assert rows == pads, k
    assert np.array_equal(fs["geom_pos"], ff["geom_pos"]) and np.array_equal(fs["geom_size"], ff["geom_size"])
    opt = [n for n in dir(ms.opt) if not n.startswith("_") and not callable(getattr(ms.opt, n))]
    od = {n for n in opt if not np.array_equal(np.asarray(getattr(ms.opt, n)), np.asarray(getattr(mf.opt, n)))}
    assert od == {"cone", "impratio"}
    cf, cs = sc.flushpad_model_check(mf), sc.flushpad_model_check(ms)
    assert sc.flushpad_model_ok(cf, 2) and not sc.flushpad_model_ok(cs, 2)
    assert cf["cone"] == "elliptic" and cf["impratio"] == 10.0
    design = sc.flushpad_design(UPSTREAM, tmp_path, sc.FlushPadParams())
    assert sc.flushpad_design_matches_model(design, cf) and not sc.flushpad_design_matches_model(design, cs)


def _base_module(tmp_path):
    path = tmp_path / "scripted_carry_base.py"
    path.write_text(_git_show("src/bhl_robust/eval/scripted_carry.py"))
    spec = importlib.util.spec_from_file_location("scripted_carry_base_094797e", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mjb_sha(m):
    import hashlib
    import mujoco
    buf = np.zeros(mujoco.mj_sizeModel(m), dtype=np.uint8)
    mujoco.mj_saveModel(m, None, buf)
    return hashlib.sha256(buf.tobytes()).hexdigest()


@needs_assets
def test_stock_model_is_byte_identical_to_the_base_commit(tmp_path):
    base = _base_module(tmp_path)
    cache = tmp_path / "cache"                                                 # one cache: same asset paths
    m_old, _, _ = base.build_carry(UPSTREAM, cache, 1, base.CarryParams())
    h_old = _mjb_sha(m_old)
    del m_old
    m_new, _, _ = sc.build_carry(UPSTREAM, cache, 1, sc.CarryParams())
    assert _mjb_sha(m_new) == h_old


def _policy():
    from omegaconf import OmegaConf
    sys.path.insert(0, str(REPO / "scripts" / "bench"))
    from team_airlock import CpuPolicy
    cfg = OmegaConf.load(DEPLOY)
    cfg.policy_checkpoint_path = str(DEPLOY.parent / cfg.policy_checkpoint_path)
    return cfg, CpuPolicy(cfg.policy_checkpoint_path)


@needs_assets
def test_stock_episode_is_identical_to_the_base_commit(tmp_path):
    base = _base_module(tmp_path)
    cfg, policy = _policy()
    runs = {}
    for tag, mod in (("base", base), ("now", sc)):
        m, slots, pairs = mod.build_carry(UPSTREAM, tmp_path / "cache", 1, mod.CarryParams())
        ctrl, qpos = [], []

        def hook(**kw):
            ctrl.append(kw["runner"].d.ctrl.copy())
            qpos.append(kw["runner"].d.qpos.copy())
        ep = mod.run_episode(m, slots, pairs, cfg, policy, TUNING_SEED, mod.CarryParams(), frame_hook=hook,
                             rule=dict(mod.LIFT_HOLD_RULE, episode_s=1.0), protocol="lift_hold")
        runs[tag] = (np.array(ctrl), np.array(qpos), ep["steps"])
    assert runs["base"][2] == runs["now"][2] == 25
    assert np.array_equal(runs["base"][0], runs["now"][0])                    # same actions, step for step
    assert np.array_equal(runs["base"][1], runs["now"][1])


@needs_assets
def test_short_flush_episodes_log_the_tilt(tmp_path):
    cfg, policy = _policy()
    p = sc.FlushPadParams()
    model, slots, pairs = sc.build_carry(UPSTREAM, tmp_path, 1, p)
    for protocol in ("lift_hold", "lift_place"):
        rule = dict(sc.flushpad_stage_rule("smoke", protocol, [TUNING_SEED]), episode_s=1.0)
        ep = sc.run_flushpad_episode(model, slots, pairs, cfg, policy, TUNING_SEED, p, rule=rule)
        ser = ep["pairs"][0]["cube_tilt_series_rad"]
        assert ep["variant"] == sc.FLUSHPAD_VARIANT and ep["steps"] == 25 and len(ser) == 26
        assert ser[0] == 0.0 and all(v is not None and np.isfinite(v) for v in ser)
        assert ep["sim_reset"]["detected"] is False and ep["sim_reset"]["policy_steps"] == 25
        assert ep["nonfinite_values_nulled"] == 0
        json.dumps(ep, allow_nan=False)
        assert abs(ser[1] - ep["trace"][0]["cube_tilt_rad"][0]) <= 0.0005 + 1e-9   # trace = post-step value
        # a real episode's structure goes through every clause set its protocol is scored by
        kinds = ((sc.FLUSHPAD_PROBE_RULE, sc.FLUSHPAD_HOLD_RULE) if protocol == "lift_hold"
                 else (sc.FLUSHPAD_PLACE_RULE,))
        for r in (rule, *kinds):
            c = sc.flushpad_clauses(r, ep, 0, float(cfg.policy_dt))
            assert c["tilt_logged"] and set(c["checks"]) and c["first_failed_check"] in (None, *c["checks"])
        if protocol == "lift_place":
            assert ep["place_params"] == sc.place_params_dict(sc.FlushPadPlaceParams())
            assert ep["script_times_s"]["unload"] == 10.7 and ep["script_times_s"]["release"] == 11.7
    with pytest.raises(ValueError):
        sc.run_flushpad_episode(model, slots, pairs, cfg, policy, TUNING_SEED, sc.CarryParams(),
                                rule=sc.FLUSHPAD_PROBE_RULE)
    with pytest.raises(ValueError):
        sc.run_flushpad_episode(model, slots, pairs, cfg, policy, TUNING_SEED, p, rule=sc.LIFT_HOLD_RULE)
    with pytest.raises(ValueError, match="FROZEN"):
        sc.run_flushpad_episode(model, slots, pairs, cfg, policy, TUNING_SEED, p, rule=sc.FLUSHPAD_PLACE_RULE,
                                q=sc.PlaceParams())


def _inject(pairs, model, step, value):
    """frame_hook (chained after the tilt recorder): after policy step `step`, put `value` in the cube's x velocity."""
    def hook(**kw):
        if kw["step"] == step:
            kw["runner"].d.qvel[model.jnt_dofadr[model.body_jntadr[pairs[0].cube_body]]] = value
    return hook


@needs_assets
def test_real_episode_auto_reset_is_detected_and_fails_every_clause_set(tmp_path, monkeypatch):
    cfg, policy = _policy()
    p = sc.FlushPadParams()
    model, slots, pairs = sc.build_carry(UPSTREAM, tmp_path, 1, p)
    rule = dict(sc.flushpad_stage_rule("smoke", "lift_hold", [TUNING_SEED]), episode_s=1.0)
    monkeypatch.chdir(tmp_path)                     # MuJoCo appends its warnings to ./MUJOCO_LOG.TXT: not the repo
    # 1e11 is finite (the harness's own check passes it) but beyond mjMAXVAL: MuJoCo resets the world itself
    ep = sc.run_flushpad_episode(model, slots, pairs, cfg, policy, TUNING_SEED, p, rule=rule,
                                 frame_hook=_inject(pairs, model, 10, 1e11))
    assert ep["failed"] is None and ep["steps"] == 25 and ep["pairs"][0]["checks"]["finite"]   # stock: blind
    rec = ep["sim_reset"]
    assert rec["detected"] is True and rec["first_step"] == 11                # the policy step after the injection
    assert rec["evidence"]["time_mismatch"] and rec["evidence"]["bad_number_warnings"]["mjWARN_BADQVEL"] >= 1
    assert rec["evidence"]["sim_time_s"] < rec["evidence"]["expected_sim_time_s"] - 0.3
    for r in (rule, sc.FLUSHPAD_PROBE_RULE, sc.FLUSHPAD_HOLD_RULE):
        c = sc.flushpad_clauses(r, ep, 0, float(cfg.policy_dt))
        assert c["sim_reset_detected"] and not c["checks"]["no_sim_reset"] and not c["success"]
    json.dumps(ep, allow_nan=False)


@needs_assets
def test_real_non_finite_stop_is_written_and_scored_as_a_failure(tmp_path, monkeypatch):
    cfg, policy = _policy()
    dt = float(cfg.policy_dt)
    p = sc.FlushPadParams()
    model, slots, pairs = sc.build_carry(UPSTREAM, tmp_path, 1, p)
    rule = dict(sc.flushpad_stage_rule("smoke", "lift_place", [TUNING_SEED]), episode_s=1.0)
    monkeypatch.chdir(tmp_path)                     # MuJoCo appends its warnings to ./MUJOCO_LOG.TXT: not the repo
    ep = sc.run_flushpad_episode(model, slots, pairs, cfg, policy, TUNING_SEED, p, rule=rule,
                                 frame_hook=_inject(pairs, model, 10, float("nan")))
    assert ep["failed"] == "nonfinite_state" and ep["steps"] == 11              # the stock check stopped it
    row = ep["pairs"][0]
    assert ep["nonfinite_values_nulled"] >= 1 and row["final_speed_mps"] is None  # NaN in the stock row, nulled
    assert ep["sim_reset"]["detected"] is False and len(row["cube_tilt_series_rad"]) == 12
    json.dumps(ep, allow_nan=False)                                            # the score JSON can hold it now
    # scored as a failure of the place stage, not as a missing episode
    rows, missing = sc._flushpad_rows({"policy_dt": dt, "episodes": [ep]},
                                      dict(sc.FLUSHPAD_PLACE_RULE, seeds=[TUNING_SEED]), 0)
    assert missing == [] and len(rows) == 1 and not rows[0]["success"] and rows[0]["first_failed_check"] == "rule"
    assert not row["checks"]["finite"]
