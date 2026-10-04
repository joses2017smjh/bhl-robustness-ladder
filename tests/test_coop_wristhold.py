"""W wrist-orientation hold of the scripted cooperative lift (opt-in): the frozen rules and their boundaries, the hold's
kinematics on small stub models (it keeps the pinch-axis orientation when a joint can, saturates at the range when the
joint is off-axis, and never exceeds the PD cap), the kinematic design record, the hold on the real harness (only the
wrist targets change, within range and cap; the release fade), both stage verdicts, the smoke check and measurement,
the CLI guard and verdict writer, the launcher header and its real modes on a fake tree, and proof that the stock
harness and the flush-pad variant are unchanged (model bytes, short episodes, and the source diff to 3a67bc2).

No test runs a probe seed (125-129) or a scored seed (20-39): every episode here runs throw-away seed 140."""

import difflib
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

from bhl_robust.eval import scripted_carry as sc

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "external/Berkeley-Humanoid-Lite"
DEPLOY = UPSTREAM / "logs/rsl_rl/humanoid/2026-08-18_20-57-50_arms-dr1.0-s0/exported/deploy.yaml"
LAUNCHER = REPO / "slurm/repo20260923/cpu_coop_wristhold.sbatch"
BASE_COMMIT = "3a67bc2"          # HEAD when the W workstream started (stock harness + flush-pad variant)
DT = 0.04
N = 500                          # 20 s at the 0.04 s policy step
K_G = 113                        # the grasp state: first state at or after 4.5 s
SEED = 140                       # a throw-away seed: never 125-129 (probe) or 20-39 (scored)

needs_assets = pytest.mark.skipif(not (UPSTREAM / "source").is_dir(), reason="upstream assets missing")

PROBE_TEXT = (
    "Probe on exploration seeds 125-129 (state the grep evidence that they were never used). PROCEED iff the cube "
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
    spec = importlib.util.spec_from_file_location("coop_scripted_carry_w", REPO / "scripts/bench/coop_scripted_carry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------- synthetic score JSONs (no simulation)

def _series(value=0.1, n=N, spikes=None):
    s = [0.0] + [float(value)] * n
    for i, v in (spikes or {}).items():
        s[i] = v
    return s


def _hold_row(pair=0, *, peak=0.2, up=(6.0, 20.0), max_tilt=0.2, floor=False, failed=None, n=N, series=None):
    t = np.arange(n) * DT
    lift = np.where((t >= up[0] - 1e-9) & (t < up[1] - 1e-9), peak, 0.0)
    row = sc.score_lift_hold(lift, np.zeros(n), DT, max_tilt=max_tilt, floor=floor, failed=failed,
                             rule=sc.LIFT_HOLD_RULE)
    row["pair"] = pair
    row["cube_tilt_series_rad"] = _series(n=n) if series is None else series
    return row


def _final(contact=False):
    return {"dz_m": 0.0, "speed_mps": 0.0, "offset_xy_m": [0.01, -0.01], "cube_tilt_rad": 0.1,
            "robot_contact": contact, "normal_N": {"b": 0.0, "a": 0.0, "plinth": 4.9}}


def _place_row(pair=0, *, max_tilt=0.2, contact=False, failed=None, n=N, series=None):
    t = np.arange(n) * DT
    lift = np.where((t >= 6.2) & (t < 10.0), 0.18, 0.0)
    row = sc.score_lift_place(lift, DT, max_tilt=max_tilt, floor=False, failed=failed, final=_final(contact),
                              rule=sc.LIFT_PLACE_RULE)
    row["pair"] = pair
    row["cube_tilt_series_rad"] = _series(n=n) if series is None else series
    return row


PLACE_TIMES = {k: round(v, 3) for k, v in sc.PlaceScript(sc.WristHoldParams(), sc.FlushPadPlaceParams()).t_end.items()}


def _sim_reset(detected=False, first_step=None):
    return {"detected": detected, "first_step": first_step, "evidence": None, "policy_steps": N}


def _wrist_record(n_pairs, steps=N, servo=True, k_g=K_G, start=None, twist=0.3, target=0.5, ctrl=0.3):
    arms = []
    for k in range(n_pairs):
        for robot, side in ((2 * k, "right"), (2 * k + 1, "left")):
            tw = [None] * (steps + 1)
            if steps >= k_g:
                tw[k_g] = 0.0
                for i in range(k_g + 1, steps + 1):
                    tw[i] = twist
            log = list(range(k_g if start is None else start, steps)) if (servo and steps > k_g) else []
            arms.append({
                "robot": robot, "pair": k, "side": side, "wrist_joints": [f"arm_{side}_elbow_roll_joint"],
                "twist_series_rad": tw, "swing_series_rad": list(tw),
                "servo_log": {"step": log, "mode": ["hold"] * len(log), "target": [[target]] * len(log),
                              "ctrl_after_nm": [[ctrl]] * len(log)},
                "summary": {"wrist_range": [[-0.785398, 0.785398]], "cap_nm": [4.0], "twist_max_abs_rad": abs(twist),
                            "twist_at_lift_end_rad": twist, "twist_final_rad": twist, "swing_max_rad": 0.5,
                            "target_clamped_steps": 0, "pd_demand_above_cap_steps": 0}})
    return {"variant": sc.WRISTHOLD_VARIANT, "servo": servo, "installed": True, "installed_after_step": 0,
            "grasp_state": k_g, "ik": {"nonfinite_fallbacks": 0}, "policy_steps": steps, "arms": arms}


def _episode(seed, rows, *, steps=N, failed=None, place=False, sim_reset=None, wrist=None, servo=True):
    ep = {"seed": seed, "pairs": rows, "failed": failed, "steps": steps, "variant": sc.WRISTHOLD_VARIANT,
          "trace": [{"t": round(i * DT, 2), "cube_tilt_rad": [round(r["cube_tilt_series_rad"][i + 1], 3) for r in rows]}
                    for i in range(0, steps, 25)],
          "sim_reset": _sim_reset() if sim_reset is None else sim_reset,
          "wrist_hold": _wrist_record(len(rows), steps, servo) if wrist is None else wrist}
    if place:
        ep.update({"protocol": "lift_place", "script_times_s": dict(PLACE_TIMES),
                   "place_params": sc.place_params_dict(sc.FlushPadPlaceParams())})
    else:
        ep["protocol"] = "lift_hold"
    return ep


FRAMES = {s: {"pos": [0.0, 0.0, -0.07], "quat": [0.96, -0.13, 0.25, -0.03], "size": [0.031, 0.043, 0.069]}
          for s in ("left", "right")}


def _model_check(crew, condim=4, tors=0.04, cone="elliptic", impratio=10.0):
    return {"pads": [{"name": f"r{i}_arm_{s}_hand_pad", "condim": condim, "friction": [1.0, tors, 0.0001],
                      "quat": list(FRAMES[s]["quat"]), "size": list(FRAMES[s]["size"])}
                     for i in range(crew) for s in ("left", "right")],
            "cone": cone, "impratio": impratio, "noslip_iterations": 0}


def _payload(stage, crew, episodes, protocol=None):
    seeds = sorted({e["seed"] for e in episodes})
    rule = (sc.wristhold_stage_rule(stage, protocol, seeds) if stage in ("smoke", "smoke_nohold")
            else sc.wristhold_stage_rule(stage))
    nohold = stage == "smoke_nohold"
    pl = {"variant": sc.WRISTHOLD_VARIANT, "stage": stage, "protocol": rule["protocol"], "crew": crew,
          "pairs": crew // 2, "scored_run": stage in ("probe", "hold", "place"), "rule": rule,
          "params": sc.params_dict(sc.FlushPadParams() if nohold else sc.WristHoldParams()),
          "keyframes_left": sc.KEYFRAMES_LEFT, "model_check": _model_check(crew), "flush_pad_design": {"frames": FRAMES},
          "wrist_hold_design": {"chosen": ["elbow_roll"]}, "policy_dt": DT, "episodes": episodes}
    if rule["protocol"] == "lift_place":
        pl["place_params"] = sc.place_params_dict(sc.FlushPadPlaceParams())
    return _j(pl)


def _probe(n_good=5, bad=None, seeds=(125, 126, 127, 128, 129)):
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


def _smoke(protocol="lift_hold", crew=2, seed=SEED, nohold=False):
    rows = [(_place_row(k) if protocol == "lift_place" else _hold_row(k)) for k in range(crew // 2)]
    ep = _episode(seed, rows, place=protocol == "lift_place", servo=not nohold)
    return _payload("smoke_nohold" if nohold else "smoke", crew, [ep], protocol=protocol)


# --------------------------------------------------------------- frozen rules, params, labels

def test_rule_texts_are_the_predeclared_ones_verbatim():
    assert sc.WRISTHOLD_PROBE_RULE_TEXT == PROBE_TEXT
    assert sc.WRISTHOLD_SCORED_RULE_TEXT == SCORED_TEXT
    # the flush-pad probe rule verbatim except the seeds; the flush-pad scored rule verbatim
    assert sc.FLUSHPAD_PROBE_RULE_TEXT.replace("120-124", "125-129") == sc.WRISTHOLD_PROBE_RULE_TEXT
    assert sc.FLUSHPAD_SCORED_RULE_TEXT == sc.WRISTHOLD_SCORED_RULE_TEXT


def test_frozen_rules_are_the_flushpad_rules_with_the_w_seeds_only():
    for w, f, changed in ((sc.WRISTHOLD_PROBE_RULE, sc.FLUSHPAD_PROBE_RULE, {"variant", "seeds"}),
                          (sc.WRISTHOLD_HOLD_RULE, sc.FLUSHPAD_HOLD_RULE, {"variant"}),
                          (sc.WRISTHOLD_PLACE_RULE, sc.FLUSHPAD_PLACE_RULE, {"variant"})):
        assert set(w) == set(f)
        assert {k for k in w if w[k] != f[k]} == changed
        assert w["variant"] == sc.WRISTHOLD_VARIANT == "wristhold_v1"
    pr, hr, pl = sc.WRISTHOLD_PROBE_RULE, sc.WRISTHOLD_HOLD_RULE, sc.WRISTHOLD_PLACE_RULE
    assert pr["seeds"] == [125, 126, 127, 128, 129] and pr["crews"] == [2] and pr["pass_min"] == 4
    assert pr["tilt_rad"] == 0.78 and pr["cube_tilt_max_rad"] == 0.35 and pr["protocol"] == "lift_hold"
    assert pr["gated_checks"] == ["cube_tilt", "no_fall", "finite", "no_sim_reset", "full_episode"]
    assert hr["seeds"] == list(range(20, 30)) and pl["seeds"] == list(range(30, 40))
    assert hr["pass_min"] == pl["pass_min"] == 8 and hr["crews"] == pl["crews"] == [2, 4]
    # the base rules are unchanged
    assert sc.LIFT_HOLD_RULE["seeds"] == list(range(10)) and sc.LIFT_PLACE_RULE["seeds"] == list(range(10, 20))
    # seed sets: the probe seeds were never used; the smoke's are throw-away and never reserved
    used = set(range(20)) | set(range(100, 125))
    assert not used & set(pr["seeds"]) and not set(pr["seeds"]) & set(hr["seeds"] + pl["seeds"])
    forbidden = set(range(40)) | set(range(100, 130)) | set(range(300, 320))
    assert set(sc.WRISTHOLD_SMOKE_SEEDS) == set(range(140, 145)) and not forbidden & set(sc.WRISTHOLD_SMOKE_SEEDS)


def test_params_are_an_opt_in_subclass_of_the_flushpad_params():
    stock, flush, w = sc.CarryParams(), sc.FlushPadParams(), sc.WristHoldParams()
    assert isinstance(w, sc.FlushPadParams) and isinstance(w, sc.CarryParams)
    assert sc.is_flushpad(w) and sc.is_wristhold(w)                             # the flush-pad model, plus the hold
    assert not sc.is_wristhold(flush) and not sc.is_wristhold(stock)
    fd, wd = sc.params_dict(flush), sc.params_dict(w)
    assert {k: wd[k] for k in fd} == fd                                         # every flush-pad field and default
    assert {k: v for k, v in wd.items() if k not in fd} == {
        "wrist_hold": "pinch_axis_twist", "wrist_joints": ("elbow_roll",), "pinch_axis_world": (1.0, 0.0, 0.0),
        "wrist_ik_damping": 0.01, "wrist_ik_max_iter": 50, "wrist_ik_tol": 1e-10, "wrist_release_fade": "retract"}
    assert w.grasp_kp == 30.0 and (w.t_settle, w.t_reach, w.t_squeeze, w.t_lift) == (1.5, 1.5, 1.5, 2.5)
    assert sc.wristhold_grasp_state(DT) == K_G                                  # 4.52 s
    assert sc.FlushPadPlaceParams().lower_to is None                            # the frozen reverse-keyframe lowering


def test_labels_and_head():
    assert sc.WRISTHOLD_LABEL.startswith("LEARNED gait (frozen arms-dr1.0-s0) + SCRIPTED arms (with a kinematic "
                                         "wrist-orientation hold) + ORACLE cube pose (scoring only)")
    assert "MODIFIED END-EFFECTOR" in sc.WRISTHOLD_LABEL and "never the cube" in sc.WRISTHOLD_LABEL_DETAIL
    assert "elbow roll" in sc.WRISTHOLD_HOLD_NOTE and "4 Nm cap" in sc.WRISTHOLD_HOLD_NOTE
    head = sc._wristhold_head("probe")
    assert head["harness_change"] == sc.FLUSHPAD_HARNESS_NOTE and head["end_effector"] == sc.FLUSHPAD_NOTE
    ev = " ".join(head["seed_evidence"])
    assert "125-129" in ev and "20-39" in ev and "140-144" in ev and "NOT_RUN" in ev
    cl = " ".join(head["clauses_as_applied"])
    assert "seeds 125-129, PROCEED iff >= 4 of 5" in cl and "MISSING (INCOMPLETE)" in cl and "elbow roll" in cl
    assert "never a gate clause" in cl and "lands flat" in cl


# --------------------------------------------------------------- the hold's kinematics on small stub models

_ARM_XML = """<mujoco><compiler angle="radian"/><option timestep="0.0005" gravity="0 0 0"/>
<worldbody>
  <body name="upper" pos="0 0 1">
    <joint name="lift" type="hinge" axis="1 0 0" range="-2 2" damping="0.05"/>
    <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02" mass="0.3"/>
    <body name="fore" pos="0 0 -0.2">
      <joint name="wrist" type="hinge" axis="{axis}" range="{lo} {hi}" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0 0 0.15 0" size="0.02" mass="0.2"/>
      <body name="hand" pos="0 0.15 0"><geom type="box" size="0.03 0.04 0.06" mass="0.3"/></body>
    </body>
  </body>
</worldbody>
<actuator><motor joint="lift"/><motor joint="wrist"/></actuator></mujoco>"""
PINCH = (1.0, 0.0, 0.0)


def _arm(axis=(1.0, 0.0, 0.0), lo=-1.5, hi=1.5):
    import mujoco
    m = mujoco.MjModel.from_xml_string(_ARM_XML.format(axis=" ".join(f"{v:.12g}" for v in axis), lo=lo, hi=hi))
    ids = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ("lift", "wrist")}
    return m, mujoco.MjData(m), mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand"), ids


def _rot(axis, angle):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, dtype=float) / np.linalg.norm(axis), angle)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def test_twist_swing_decomposition():
    R0 = _rot((0.3, -0.2, 0.9), 0.7)
    for ang in (-1.2, -0.3, 0.0, 0.5, 1.4):
        tw, sw = sc.hand_twist_swing(R0, _rot(PINCH, ang) @ R0, PINCH)                 # pure twist about the axis
        assert tw == pytest.approx(ang, abs=1e-9) and sw == pytest.approx(0.0, abs=1e-6)
        tw, sw = sc.hand_twist_swing(R0, _rot((0.0, 0.6, 0.8), ang) @ R0, PINCH)       # pure swing (normal axis)
        assert tw == pytest.approx(0.0, abs=1e-9) and sw == pytest.approx(abs(ang), abs=1e-6)
    # swing then twist about the axis: the twist reads the twist
    R = _rot((0.0, 1.0, 0.0), 0.4) @ _rot(PINCH, 0.6) @ R0
    tw, sw = sc.hand_twist_swing(R0, R, PINCH)
    assert abs(tw) > 0.5 and sw > 0.3


def _twist_of(R_ref, R):
    return sc.hand_twist_swing(R_ref, R, PINCH)[0]


def test_twist_derivative_is_exact_including_large_swing():
    rng = np.random.default_rng(7)
    swings = []
    for _ in range(200):
        R_ref = _rot(rng.normal(size=3), rng.uniform(0.0, 3.0))
        R = _rot(rng.normal(size=3), rng.uniform(0.0, 2.6)) @ R_ref
        _, sw = sc.hand_twist_swing(R_ref, R, PINCH)
        if sw > 3.0:                                                            # near the half-turn singularity
            continue
        swings.append(sw)
        om = rng.normal(size=3)
        om /= np.linalg.norm(om)
        h = 1e-6
        fd = (_twist_of(R_ref, _rot(om, h) @ R) - _twist_of(R_ref, _rot(om, -h) @ R)) / (2 * h)
        assert sc.wrist_twist_derivative(R_ref, R, PINCH, [om])[0] == pytest.approx(fd, abs=1e-5)
    assert sum(s >= 1.5 for s in swings) >= 20                                  # large swings were covered
    # at a pure-twist state the exact derivative is the projected Jacobian, omega . pinch
    R_ref = _rot((0.2, 0.5, -0.3), 0.9)
    om = np.array([0.6, 0.0, 0.8])
    assert sc.wrist_twist_derivative(R_ref, _rot(PINCH, 0.7) @ R_ref, PINCH, [om])[0] == pytest.approx(0.6, abs=1e-12)


_SWING_XML = """<mujoco><compiler angle="radian"/><option gravity="0 0 0"/>
<worldbody>
  <body name="b0" pos="0 0 1">
    <joint name="yaw" type="hinge" axis="0 0 1"/>
    <joint name="pitch" type="hinge" axis="0 1 0"/>
    <joint name="lift" type="hinge" axis="1 0 0"/>
    <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02" mass="0.3"/>
    <body name="fore" pos="0 0 -0.2">
      <joint name="wrist" type="hinge" axis="0.1736481776669 0 0.984807753012208" range="-0.785398 0.785398"/>
      <body name="hand" pos="0 0.15 0"><geom type="box" size="0.03 0.04 0.06" mass="0.3"/></body>
    </body>
  </body>
</worldbody></mujoco>"""


def _old_projected_ik(m, d, hand, wj, R_ref):
    """The first (withdrawn) implementation: the projected Jacobian as the derivative, every step accepted."""
    import mujoco
    a, dof, lo, hi = np.array(PINCH), m.jnt_dofadr[wj], -0.785398, 0.785398
    q = float(d.qpos[m.jnt_qposadr[wj]])
    jacr = np.zeros((3, m.nv))
    for _ in range(50):
        d.qpos[m.jnt_qposadr[wj]] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_jacBody(m, d, None, jacr, hand)
        e, J = _twist_of(R_ref, d.xmat[hand]), float(a @ jacr[:, dof])
        qn = float(np.clip(q - J * e / (J * J + 1e-4), lo, hi))
        if abs(qn - q) < 1e-10:
            break
        q = qn
    d.qpos[m.jnt_qposadr[wj]] = q
    mujoco.mj_kinematics(m, d)
    return _twist_of(R_ref, d.xmat[hand])


def test_the_ik_never_returns_a_worse_twist_than_the_measured_state():
    import mujoco
    m = mujoco.MjModel.from_xml_string(_SWING_XML)
    d = mujoco.MjData(m)
    hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    jid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ("yaw", "pitch", "lift", "wrist")}
    mujoco.mj_kinematics(m, d)
    R_ref = d.xmat[hand].reshape(3, 3).copy()
    rng = np.random.default_rng(11)
    old_worse = 0
    for _ in range(300):
        for n, span in (("yaw", 1.4), ("pitch", 1.4), ("lift", 0.9), ("wrist", 0.78)):
            d.qpos[m.jnt_qposadr[jid[n]]] = rng.uniform(-span, span)
        mujoco.mj_kinematics(m, d)
        before = _twist_of(R_ref, d.xmat[hand])
        sol = sc.wrist_twist_ik(m, d, hand, [(jid["wrist"], -0.785398, 0.785398)], R_ref, PINCH,
                                damping=0.01, max_iter=50, tol=1e-10)
        assert sol["twist_before"] == pytest.approx(before, abs=1e-12)
        assert abs(sol["twist_after"]) <= abs(before) + 1e-12                   # never worse than measured
        assert -0.785398 - 1e-12 <= sol["q"][0] <= 0.785398 + 1e-12
        # the same measured state (wrist_twist_ik restored it) through the withdrawn algorithm
        old_worse += abs(_old_projected_ik(m, d, hand, jid["wrist"], R_ref)) > abs(before) + 1e-6
    assert old_worse >= 1                                                       # the withdrawn version could be worse


@pytest.mark.parametrize("q_meas", [0.7858, -0.7858, 0.7854])
def test_the_ik_target_is_inside_the_range_even_when_the_joint_sits_past_its_soft_limit(q_meas):
    import mujoco
    c, s = np.cos(np.radians(80.0)), np.sin(np.radians(80.0))
    m, d, hand, ids = _arm(axis=(c, 0.0, s), lo=-0.785398, hi=0.785398)
    mujoco.mj_kinematics(m, d)
    R_ref = d.xmat[hand].reshape(3, 3).copy()
    for lift in (0.6, -0.6, 0.0):                                               # push outward, inward, or nothing
        d.qpos[m.jnt_qposadr[ids["lift"]]] = lift
        d.qpos[m.jnt_qposadr[ids["wrist"]]] = q_meas
        mujoco.mj_kinematics(m, d)
        sol = _solve(m, d, hand, ids, R_ref)
        assert -0.785398 <= sol["q"][0] <= 0.785398
        assert d.qpos[m.jnt_qposadr[ids["wrist"]]] == q_meas                    # the measured state is restored


def _solve(m, d, hand, ids, R_ref, damping=0.01):
    return sc.wrist_twist_ik(m, d, hand, [(ids["wrist"], *m.jnt_range[ids["wrist"]])], R_ref, PINCH,
                             damping=damping, max_iter=50, tol=1e-10)


def test_ik_cancels_the_twist_with_an_aligned_wrist_and_restores_the_data():
    import mujoco
    m, d, hand, ids = _arm()
    mujoco.mj_kinematics(m, d)
    R_ref = d.xmat[hand].reshape(3, 3).copy()
    d.qpos[m.jnt_qposadr[ids["lift"]]] = 0.6                                    # the "lift" turns the hand 0.6 rad
    mujoco.mj_kinematics(m, d)
    q_before = d.qpos.copy()
    sol = _solve(m, d, hand, ids, R_ref)
    assert sol["twist_before"] == pytest.approx(0.6, abs=1e-9)
    assert sol["q"][0] == pytest.approx(-0.6, abs=1e-8) and abs(sol["twist_after"]) < 1e-9 and not sol["clamped"]
    assert np.array_equal(d.qpos, q_before)                                     # the caller's state is restored
    assert sc.hand_twist_swing(R_ref, d.xmat[hand], PINCH)[0] == pytest.approx(0.6, abs=1e-9)
    # the solution does not depend on the damping (a numerical safeguard only)
    assert _solve(m, d, hand, ids, R_ref, damping=0.2)["q"][0] == pytest.approx(sol["q"][0], abs=1e-8)
    # the derivative is exact: at this pure-twist state it is MuJoCo's rotational Jacobian column of the wrist
    # projected on the pinch axis
    assert sol["J"][0] == pytest.approx(1.0, abs=1e-12)


def test_ik_saturates_at_the_range_with_an_off_axis_wrist():
    import mujoco
    c, s = np.cos(np.radians(80.0)), np.sin(np.radians(80.0))                   # 80 deg off the pinch axis (BHL: ~80)
    m, d, hand, ids = _arm(axis=(c, 0.0, s), lo=-0.785398, hi=0.785398)
    mujoco.mj_kinematics(m, d)
    R_ref = d.xmat[hand].reshape(3, 3).copy()
    d.qpos[m.jnt_qposadr[ids["lift"]]] = 0.6
    mujoco.mj_kinematics(m, d)
    sol = _solve(m, d, hand, ids, R_ref)
    assert sol["J"][0] == pytest.approx(c, abs=1e-9)
    assert sol["clamped"] and sol["q"][0] == pytest.approx(-0.785398, abs=1e-9)   # held at the bound that helps
    assert 0.35 < sol["twist_after"] < sol["twist_before"] == pytest.approx(0.6, abs=1e-9)
    # the clamped solution is the best the range allows: the other bound and the middle are worse
    for q in (0.785398, 0.0, -0.4):
        d2 = mujoco.MjData(m)
        d2.qpos[:] = d.qpos
        d2.qpos[m.jnt_qposadr[ids["wrist"]]] = q
        mujoco.mj_kinematics(m, d2)
        assert abs(sc.hand_twist_swing(R_ref, d2.xmat[hand], PINCH)[0]) > abs(sol["twist_after"])


def _simulate_hold(axis, lo, hi, hold=True, load_nm=0.0, seconds=2.0):
    """The harness's control structure on the stub: every 0.04 s the scripted "lift" target moves (0 -> 0.6 rad over
    1 s, the script's smooth profile) and, with the hold, the wrist target is the IK solution from the measured state;
    80 physics substeps of PD kp 30, kd 2, torque clipped to 4 Nm (as CarryRunner.step). `load_nm` = an external torque
    on the wrist joint."""
    import mujoco
    m, d, hand, ids = _arm(axis, lo, hi)
    scratch = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    R_ref = d.xmat[hand].reshape(3, 3).copy()
    a = [m.jnt_qposadr[ids["lift"]], m.jnt_qposadr[ids["wrist"]]]
    v = [m.jnt_dofadr[ids["lift"]], m.jnt_dofadr[ids["wrist"]]]
    d.qfrc_applied[v[1]] = load_nm
    twist, torque, demand = [], [], []
    for k in range(int(round(seconds / 0.04))):
        tgt = np.array([0.6 * sc._smooth(k * 0.04 / 1.0), 0.0])
        if hold:
            scratch.qpos[:] = d.qpos
            sol = sc.wrist_twist_ik(m, scratch, hand, [(ids["wrist"], lo, hi)], R_ref, PINCH, damping=0.01,
                                    max_iter=50, tol=1e-10)
            tgt[1] = sol["q"][0]
        demand.append(30.0 * (tgt[1] - d.qpos[a[1]]) - 2.0 * d.qvel[v[1]])
        for _ in range(80):
            d.ctrl[:] = np.clip(30.0 * (tgt - d.qpos[a]) - 2.0 * d.qvel[v], -4.0, 4.0)
            torque.append(d.ctrl[1])
            mujoco.mj_step(m, d)
        scratch.qpos[:] = d.qpos
        mujoco.mj_kinematics(m, scratch)
        twist.append(sc.hand_twist_swing(R_ref, scratch.xmat[hand], PINCH)[0])
    return np.array(twist), np.array(torque), np.array(demand), d.qpos[a[1]]


def test_the_hold_keeps_the_pinch_axis_orientation_when_the_wrist_can():
    tw, torque, _, q_w = _simulate_hold((1.0, 0.0, 0.0), -1.5, 1.5)
    # held while the lift turns 0.6 rad: only the PD's tracking lag (kp 30, the lift at up to 0.94 rad/s) shows
    assert np.max(np.abs(tw)) < 0.15 and abs(tw[-1]) < 2e-3
    assert q_w == pytest.approx(-0.6, abs=5e-3)
    assert np.max(np.abs(torque)) <= 4.0
    tw0, _, _, _ = _simulate_hold((1.0, 0.0, 0.0), -1.5, 1.5, hold=False)
    assert tw0[-1] == pytest.approx(0.6, abs=0.01)                              # without it the hand turns 0.6 rad


def test_the_hold_saturates_at_the_range_when_the_wrist_is_off_axis():
    c, s = np.cos(np.radians(80.0)), np.sin(np.radians(80.0))
    tw, torque, _, q_w = _simulate_hold((c, 0.0, s), -0.785398, 0.785398)
    assert q_w == pytest.approx(-0.785398, abs=0.01)                            # at the joint limit
    assert 0.35 < tw[-1] < 0.6                                                  # only a fraction removed
    assert np.max(np.abs(torque)) <= 4.0


def test_the_hold_never_exceeds_the_cap_and_a_load_beyond_it_wins():
    tw, torque, demand, _ = _simulate_hold((1.0, 0.0, 0.0), -1.5, 1.5, load_nm=6.0)
    assert np.max(np.abs(torque)) == pytest.approx(4.0)                         # clipped at the cap, never above
    assert np.max(np.abs(demand)) > 4.0                                         # the hold asked for more
    assert np.max(np.abs(tw)) > 0.3                                             # and could not hold the twist


# --------------------------------------------------------------- the tilt clauses (the flush-pad clause code, W rules)

@pytest.mark.parametrize("value,ok", [(0.35, True), (0.35 + 1e-12, False), (0.0, True), (1.571, False)])
def test_probe_tilt_clause_boundary(value, ok):
    ep = _episode(125, [_hold_row(series=_series(0.1, spikes={250: value}))])
    c = sc.flushpad_clauses(sc.WRISTHOLD_PROBE_RULE, ep, 0, DT)
    assert c["checks"]["cube_tilt"] is ok and c["success"] is ok


@pytest.mark.parametrize("max_tilt,ok", [(0.78, True), (0.7800001, False)])
def test_probe_keeps_the_robot_fall_clause_unchanged(max_tilt, ok):
    c = sc.flushpad_clauses(sc.WRISTHOLD_PROBE_RULE, _episode(125, [_hold_row(max_tilt=max_tilt)]), 0, DT)
    assert c["checks"]["no_fall"] is ok and c["success"] is ok


@pytest.mark.parametrize("i,ok", [(149, True), (150, False), (399, False), (400, True)])
def test_hold_tilt_window_is_the_scored_hold(i, ok):
    row = _hold_row(up=(6.0, 16.0), series=_series(0.1, spikes={i: 0.9}))
    c = sc.flushpad_clauses(sc.WRISTHOLD_HOLD_RULE, _episode(20, [row]), 0, DT)
    assert c["checks"]["cube_tilt_hold"] is ok and c["checks"]["rule"]


@pytest.mark.parametrize("i,ok", [(267, True), (268, False), (293, False), (294, True)])
def test_place_release_window_boundaries(i, ok):
    row = _place_row(series=_series(0.1, spikes={i: 0.9}))
    c = sc.flushpad_clauses(sc.WRISTHOLD_PLACE_RULE, _episode(30, [row], place=True), 0, DT)
    assert c["checks"]["cube_tilt_release"] is ok and c["checks"]["cube_tilt_seated"] and c["checks"]["rule"]


# --------------------------------------------------------------- stage 1: probe verdict

def test_probe_verdict_boundaries():
    assert sc.wristhold_probe_verdict(_probe(5))["verdict"] == "PROCEED"
    v = sc.wristhold_probe_verdict(_probe(4))
    assert v["verdict"] == "PROCEED" and v["successes"] == 4 and v["rule_text"] == PROBE_TEXT
    v = sc.wristhold_probe_verdict(_probe(3))
    assert v["verdict"] == "NEGATIVE" and v["successes"] == 3 and "does not stop the roll" in v["reading"]
    assert sc.wristhold_probe_verdict(_probe(3, bad=lambda: _hold_row(max_tilt=0.9)))["verdict"] == "NEGATIVE"
    v = sc.wristhold_probe_verdict(_probe(4))
    assert v["label"] == sc.WRISTHOLD_LABEL and all(r["wrist_hold_diagnostic"] for r in v["per_seed"])


def test_probe_verdict_incomplete_and_failed_episodes():
    pl = _probe(5)
    pl["episodes"] = pl["episodes"][:4]
    v = sc.wristhold_probe_verdict(pl)
    assert v["verdict"] == "INCOMPLETE" and v["missing_seeds"] == [129]
    assert sc.wristhold_probe_verdict(None)["verdict"] == "INCOMPLETE"
    pl = _probe(5)
    pl["episodes"][2]["steps"] = 499
    assert sc.wristhold_probe_verdict(pl)["verdict"] == "INCOMPLETE"
    # a non-finite stop (even before the grasp state) is a scored failure, not a missing episode
    pl = _probe(5)
    row = _hold_row(n=100, failed="nonfinite_state", series=_series(0.1, n=100))
    pl["episodes"][4] = _j(_episode(129, [row], steps=100, failed="nonfinite_state"))
    v = sc.wristhold_probe_verdict(pl)
    assert v["verdict"] == "PROCEED" and v["successes"] == 4 and v["missing_seeds"] == []
    pl["episodes"][3] = _j(_episode(128, [row], steps=100, failed="nonfinite_state"))
    assert sc.wristhold_probe_verdict(pl)["verdict"] == "NEGATIVE"


@pytest.mark.parametrize("mutate", [
    lambda e: e.pop("wrist_hold"),                                                     # no hold record
    lambda e: e["wrist_hold"].update(servo=False),                                     # the hold did not run
    lambda e: e["wrist_hold"].update(installed=False),
    lambda e: e["wrist_hold"].update(installed_after_step=3),
    lambda e: e["wrist_hold"].update(grasp_state=112),
    lambda e: e["wrist_hold"].update(variant=sc.FLUSHPAD_VARIANT),
    lambda e: e["wrist_hold"]["arms"].pop(),                                           # a hand missing
    lambda e: e["wrist_hold"]["arms"][0]["twist_series_rad"].pop(),                    # not steps + 1 long
    lambda e: e["wrist_hold"]["arms"][1]["twist_series_rad"].__setitem__(K_G, 0.01),   # not 0 at the grasp state
    lambda e: e["wrist_hold"]["arms"][0]["servo_log"].update(step=list(range(K_G + 1, N))),  # engaged late
    lambda e: e["wrist_hold"]["arms"][0]["servo_log"].update(step=[]),                 # never engaged
])
def test_a_missing_or_malformed_hold_record_is_missing_not_a_pass(mutate):
    pl = _probe(5)
    mutate(pl["episodes"][1])
    v = sc.wristhold_probe_verdict(pl)
    assert v["verdict"] == "INCOMPLETE" and v["missing_seeds"] == [126]


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(variant=sc.FLUSHPAD_VARIANT),
    lambda p: p.update(stage="hold"),
    lambda p: p.update(scored_run=False),
    lambda p: p.update(crew=4),
    lambda p: p["rule"].update(seeds=[120, 121, 122, 123, 124]),
    lambda p: p["rule"].update(pass_min=3),
    lambda p: p["rule"].update(cube_tilt_max_rad=0.5),
    lambda p: p.update(params=sc.params_dict(sc.FlushPadParams())),                    # the flush-pad variant
    lambda p: p["params"].update(wrist_joints=["shoulder_yaw", "elbow_roll"]),
    lambda p: p["params"].update(grasp_kp=60.0),
    lambda p: p["keyframes_left"].update(lift=[-0.5, 1.1, 0.0, 1.2, 0.0]),
    lambda p: p.update(wrist_hold_design={"chosen": ["shoulder_yaw"]}),
    lambda p: p.update(model_check=_model_check(2, cone="pyramidal")),
])
def test_probe_verdict_invalid_json_never_proceeds(mutate):
    pl = _probe(5)
    mutate(pl)
    assert sc.wristhold_probe_verdict(pl)["verdict"] == "INVALID"
    assert sc.flushpad_probe_verdict(_probe(5))["verdict"] == "INVALID"         # never readable as a flush-pad probe


# --------------------------------------------------------------- stage 2: scored verdict

def test_scored_verdict_pass_at_eight_of_ten_everywhere():
    assert sc.wristhold_scored_verdict(_scored())["verdict"] == "PASS"
    pls = {k: _scored_group(k.split("_")[0], int(k[-1]), n_good=8) for k in sc.WRISTHOLD_SCORED_KEYS}
    v = sc.wristhold_scored_verdict(pls)
    assert v["verdict"] == "PASS" and len(v["groups"]) == 6 and v["rule_text"] == SCORED_TEXT
    assert all(g["successes"] == 8 and g["episodes"] == 10 for g in v["groups"])


@pytest.mark.parametrize("key", sc.WRISTHOLD_SCORED_KEYS)
def test_scored_verdict_negative_at_seven_of_ten(key):
    st, crew = key.split("_")[0], int(key[-1])
    assert sc.wristhold_scored_verdict(_scored(**{key: _scored_group(st, crew, n_good=7)}))["verdict"] == "NEGATIVE"


def test_scored_verdict_gates_each_pair_of_crew4():
    v = sc.wristhold_scored_verdict(_scored(hold_crew4=_scored_group("hold", 4, n_good=7, fail=(1,))))
    g = {(x["stage"], x["crew"], x["pair"]): x for x in v["groups"]}
    assert v["verdict"] == "NEGATIVE" and g[("hold", 4, 0)]["pass"] and not g[("hold", 4, 1)]["pass"]


def test_scored_verdict_incomplete_invalid_and_not_run():
    assert sc.wristhold_scored_verdict(_scored(hold_crew4=None))["verdict"] == "INCOMPLETE"
    pl = _scored_group("place", 4)
    pl["episodes"] = pl["episodes"][:9]
    assert sc.wristhold_scored_verdict(_scored(place_crew4=pl))["verdict"] == "INCOMPLETE"
    pl = _scored_group("hold", 2)
    pl["episodes"][0].pop("wrist_hold")
    assert sc.wristhold_scored_verdict(_scored(hold_crew2=pl))["verdict"] == "INCOMPLETE"
    pl = _scored_group("place", 2)
    pl["place_params"]["lower_to"] = [0.6, 1.1, 0.0, 1.0, 0.0]                  # the stock (tuned) lowering
    assert sc.wristhold_scored_verdict(_scored(place_crew2=pl))["verdict"] == "INVALID"
    assert sc.wristhold_scored_verdict(_scored(hold_crew2=_scored_group("hold", 4)))["verdict"] == "INVALID"
    v = sc.wristhold_not_run_verdict({"verdict": "NEGATIVE"})
    assert v["verdict"] == "NOT_RUN" and "seeds 20-39 stay unused" in v["reading"]
    assert sc.wristhold_not_run_verdict(None)["probe_verdict"] == "MISSING"


# --------------------------------------------------------------- smoke check and measurement

def test_smoke_check_passes_good_smokes():
    assert sc.wristhold_smoke_check(_smoke())["verdict"] == "SMOKE_PASS"
    assert sc.wristhold_smoke_check(_smoke("lift_place", 4))["verdict"] == "SMOKE_PASS"
    assert sc.wristhold_smoke_check(_smoke(nohold=True))["verdict"] == "SMOKE_PASS"


@pytest.mark.parametrize("mutate", [
    lambda p: p["episodes"][0].update(steps=499),
    lambda p: p["episodes"][0].update(failed="nonfinite_state"),
    lambda p: p["episodes"][0]["pairs"][0]["cube_tilt_series_rad"].pop(),
    lambda p: p["episodes"][0]["pairs"][0]["cube_tilt_series_rad"].__setitem__(26, 0.5),    # disagrees with trace
    lambda p: p["episodes"][0].pop("sim_reset"),
    lambda p: p["episodes"][0].update(sim_reset=_sim_reset(True, 120)),
    lambda p: p["episodes"][0].pop("wrist_hold"),
    lambda p: p["episodes"][0]["wrist_hold"].update(servo=False),
    lambda p: p["episodes"][0]["wrist_hold"]["ik"].update(nonfinite_fallbacks=1),
    lambda p: p["episodes"][0]["wrist_hold"]["arms"][0]["servo_log"]["target"].__setitem__(5, [0.79]),  # off range
    lambda p: p["episodes"][0]["wrist_hold"]["arms"][1]["servo_log"]["ctrl_after_nm"].__setitem__(9, [4.01]),
    lambda p: p.update(scored_run=True),
    lambda p: p.update(params=sc.params_dict(sc.FlushPadParams())),
    lambda p: p.update(model_check=_model_check(2, condim=3)),
    lambda p: p.update(wrist_hold_design={}),
    lambda p: p["rule"].update(episode_s=1.0),
])
def test_smoke_check_fails_on_pipeline_faults(mutate):
    pl = _smoke()
    mutate(pl)
    assert sc.wristhold_smoke_check(pl)["verdict"] == "SMOKE_FAIL"


@pytest.mark.parametrize("seed", [119, 124, 125, 129, 20, 39, 0, 300, 145])
def test_smoke_check_refuses_seeds_that_are_not_throw_away(seed):
    assert sc.wristhold_smoke_check(_smoke(seed=seed))["verdict"] == "SMOKE_FAIL"


def test_nohold_smoke_is_crew2_lift_hold_flushpad_with_no_servo():
    pl = _smoke(nohold=True)
    pl["episodes"][0]["wrist_hold"] = _j(_wrist_record(1, servo=True))
    assert sc.wristhold_smoke_check(pl)["verdict"] == "SMOKE_FAIL"
    pl = _smoke(nohold=True)
    pl["params"] = sc.params_dict(sc.WristHoldParams())
    assert sc.wristhold_smoke_check(pl)["verdict"] == "SMOKE_FAIL"
    pl = _smoke("lift_place", 4)
    pl["stage"] = "smoke_nohold"
    assert sc.wristhold_smoke_check(pl)["verdict"] == "SMOKE_FAIL"


def test_smoke_outcome_is_never_gated():
    pl = _smoke()
    row = _hold_row(series=_series(1.5), max_tilt=1.2)
    pl["episodes"][0] = _j(_episode(SEED, [row], wrist=_wrist_record(1, twist=1.4)))
    assert sc.wristhold_smoke_check(pl)["verdict"] == "SMOKE_PASS"


@pytest.mark.parametrize("status,ok", [
    ({"pytest": 0, "smoke_hold_crew2": 0, "smoke_place_crew4": 0, "smoke_nohold_crew2": 0}, True),
    ({"pytest": 1, "smoke_hold_crew2": 0, "smoke_place_crew4": 0, "smoke_nohold_crew2": 0}, False),
    ({"pytest": 0, "smoke_hold_crew2": 0, "smoke_place_crew4": 0}, False),
    ({"pytest": 0, "smoke_hold_crew2": 0, "smoke_place_crew4": 0, "smoke_nohold_crew2": 3}, False),
    ({"pytest": "0", "smoke_hold_crew2": 0, "smoke_place_crew4": 0, "smoke_nohold_crew2": 0}, False),
    ({"pytest": True, "smoke_hold_crew2": 0, "smoke_place_crew4": 0, "smoke_nohold_crew2": 0}, False),
    (None, False), ([0, 0, 0, 0], False)])
def test_smoke_step_status_boundaries(status, ok):
    assert (sc.wristhold_smoke_step_problems(status) == []) is ok


def test_smoke_measurement_compares_the_hold_with_the_no_hold_baseline():
    hold, nohold = _smoke(), _smoke(nohold=True)
    for a in hold["episodes"][0]["wrist_hold"]["arms"]:
        a["summary"].update(twist_at_lift_end_rad=0.2, twist_max_abs_rad=0.4)
    for a in nohold["episodes"][0]["wrist_hold"]["arms"]:
        a["summary"].update(twist_at_lift_end_rad=-0.8, twist_max_abs_rad=0.8)
    m = sc.wristhold_smoke_measurement(hold, nohold, _smoke("lift_place", 4))
    assert m["same_seed"] and m["seed_hold"] == SEED
    for side in ("left", "right"):
        assert m["hands"][side]["removed_fraction_at_lift_end"] == pytest.approx(0.75)
        assert m["hands"][side]["removed_fraction_of_max"] == pytest.approx(0.5)
    assert m["cube"]["hold"]["cube_tilt_max_rad"] == pytest.approx(0.1) and len(m["place_crew4"]["hands"]) == 4
    assert sc.wristhold_smoke_measurement(None, None)["hands"]["left"]["removed_fraction_at_lift_end"] is None


# --------------------------------------------------------------- CLI guard and verdict writer

def _args(tmp_path, stage, seeds, protocol="lift_hold", crew=2, out="s.json"):
    import argparse
    return argparse.Namespace(wristhold_stage=stage, out=tmp_path / out, seeds=seeds, protocol=protocol, crew=crew)


@pytest.mark.parametrize("stage,seeds,protocol,crew", [
    ("probe", "125-129", "lift_hold", 2), ("hold", "20-29", "lift_hold", 2), ("hold", "20-29", "lift_hold", 4),
    ("place", "30-39", "lift_place", 2), ("place", "30-39", "lift_place", 4), ("smoke", "140", "lift_hold", 2),
    ("smoke", "140-144", "lift_place", 4), ("smoke_nohold", "140", "lift_hold", 2)])
def test_guard_accepts_exactly_the_declared_runs(tmp_path, stage, seeds, protocol, crew):
    rule, proto, s = _script().wristhold_guard(_args(tmp_path, stage, seeds, protocol, crew))
    assert proto == protocol and rule["seeds"] == s and rule["variant"] == sc.WRISTHOLD_VARIANT
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("stage,seeds,protocol,crew", [
    ("probe", "120-124", "lift_hold", 2), ("probe", "125-128", "lift_hold", 2), ("probe", "124-129", "lift_hold", 2),
    ("probe", "125-129", "lift_hold", 4), ("probe", "125-129", "lift_place", 2), ("hold", "20-28", "lift_hold", 2),
    ("hold", "20-29", "lift_place", 2), ("place", "30-39", "lift_hold", 2), ("place", "20-29", "lift_place", 2),
    ("smoke", "119", "lift_hold", 2), ("smoke", "125", "lift_hold", 2), ("smoke", "20", "lift_hold", 2),
    ("smoke", "39", "lift_place", 2), ("smoke", "0", "lift_hold", 2), ("smoke", "300", "lift_hold", 2),
    ("smoke", "140,145", "lift_hold", 2), ("smoke", "140", "carry", 2), ("smoke_nohold", "140", "lift_place", 2),
    ("smoke_nohold", "140", "lift_hold", 4), ("smoke_nohold", "129", "lift_hold", 2), ("bogus", "140", "lift_hold", 2)])
def test_guard_refuses_everything_else(tmp_path, stage, seeds, protocol, crew):
    with pytest.raises(SystemExit, match="REFUSED"):
        _script().wristhold_guard(_args(tmp_path, stage, seeds, protocol, crew))
    assert list(tmp_path.iterdir()) == []


def test_score_refuses_to_overwrite_before_loading_anything(tmp_path):
    out = tmp_path / "score_probe_crew2.json"
    out.write_text("{}")
    with pytest.raises(SystemExit, match="REFUSED"):
        _script().score_wristhold(_args(tmp_path, "probe", "125-129", out=out.name))
    assert out.read_text() == "{}"


def test_verdict_cli_writes_once_from_json(tmp_path, capsys):
    mod = _script()
    src = tmp_path / "score_probe_crew2.json"
    src.write_text(json.dumps(_probe(4)))
    out = tmp_path / "probe" / "verdict.json"
    assert mod.wristhold_verdict_cli(["probe", str(out), f"probe={src}"]) == 0
    v = json.loads(out.read_text())
    assert v["verdict"] == "PROCEED" and v["inputs"]["probe"]["sha256"] and v["rule_text"] == PROBE_TEXT
    assert "COOP-WRISTHOLD probe: PROCEED" in capsys.readouterr().out
    before = out.read_text()
    with pytest.raises(SystemExit, match="REFUSED"):
        mod.wristhold_verdict_cli(["probe", str(out), f"probe={src}"])
    assert out.read_text() == before
    out2 = tmp_path / "v2.json"
    mod.wristhold_verdict_cli(["probe", str(out2), f"probe={tmp_path / 'absent.json'}"])
    assert json.loads(out2.read_text())["verdict"] == "INCOMPLETE"
    out3 = tmp_path / "v3.json"
    mod.wristhold_verdict_cli(["scored_not_run", str(out3), f"probe_verdict={out2}"])
    assert json.loads(out3.read_text())["verdict"] == "NOT_RUN"
    args = []
    for k, pl in _scored().items():
        (tmp_path / f"{k}.json").write_text(json.dumps(pl))
        args.append(f"{k}={tmp_path / (k + '.json')}")
    mod.wristhold_verdict_cli(["scored", str(tmp_path / "scored.json"), *args])
    assert json.loads((tmp_path / "scored.json").read_text())["verdict"] == "PASS"
    mod.wristhold_verdict_cli(["scored", str(tmp_path / "scored3.json"), *args[:3]])
    assert json.loads((tmp_path / "scored3.json").read_text())["verdict"] == "INCOMPLETE"


def test_verdict_cli_smoke_needs_every_step_and_all_three_episodes(tmp_path):
    mod = _script()
    files = {"hold_crew2": _smoke(), "place_crew4": _smoke("lift_place", 4), "nohold_crew2": _smoke(nohold=True)}
    for k, pl in files.items():
        (tmp_path / f"{k}.json").write_text(json.dumps(pl))
    good = tmp_path / "step_status.json"
    good.write_text('{"pytest": 0, "smoke_hold_crew2": 0, "smoke_place_crew4": 0, "smoke_nohold_crew2": 0}\n')
    inputs = [f"{k}={tmp_path / (k + '.json')}" for k in files]
    mod.wristhold_verdict_cli(["smoke", str(tmp_path / "v1.json"), f"step_status={good}", *inputs])
    v = json.loads((tmp_path / "v1.json").read_text())
    assert v["verdict"] == "SMOKE_PASS" and v["unit_tests_passed"] is True and v["problems"] == []
    assert v["measurement"]["same_seed"] and set(v["checks"]) == set(files)
    mod.wristhold_verdict_cli(["smoke", str(tmp_path / "v2.json"), f"step_status={good}", *inputs[:2]])
    assert json.loads((tmp_path / "v2.json").read_text())["verdict"] == "SMOKE_FAIL"           # no baseline
    bad = tmp_path / "bad.json"
    bad.write_text('{"pytest": 1, "smoke_hold_crew2": 0, "smoke_place_crew4": 0, "smoke_nohold_crew2": 0}\n')
    mod.wristhold_verdict_cli(["smoke", str(tmp_path / "v3.json"), f"step_status={bad}", *inputs])
    v3 = json.loads((tmp_path / "v3.json").read_text())
    assert v3["verdict"] == "SMOKE_FAIL" and v3["unit_tests_passed"] is False
    mod.wristhold_verdict_cli(["smoke", str(tmp_path / "v4.json"), *inputs])
    assert json.loads((tmp_path / "v4.json").read_text())["verdict"] == "SMOKE_FAIL"           # no step status


# --------------------------------------------------------------- launcher: header and structure

def _header_text():
    return _norm(" ".join(ln.lstrip("#").strip() for ln in LAUNCHER.read_text().splitlines() if ln.startswith("#")))


def test_launcher_header_states_the_frozen_rules_verbatim_and_the_labels():
    head = _header_text()
    assert _norm(PROBE_TEXT) in head and _norm(SCORED_TEXT) in head
    for s in ("LEARNED gait (frozen arms-dr1.0-s0) + SCRIPTED arms (with a kinematic wrist-orientation hold) + ORACLE "
              "cube pose (scoring only)", "MODIFIED END-EFFECTOR -- NOT THE STOCK ROBOT",
              "HARNESS CHANGE: elliptic friction cone + impratio 10", "foot-floor contact", "the reverse keyframe",
              "the wrist is the ELBOW ROLL", "state 113, t = 4.52 s", "grasping-arm PD kp 30 and kd 2, the 4 Nm arm cap",
              "CLAUSES AS APPLIED", "SEED EVIDENCE", "PROCEED iff >= 4 of seeds 125-129 pass", "never lifts the cube",
              "MISSING (INCOMPLETE)", "wrist-hold design record does not name the frozen wrist (elbow roll)",
              "KINEMATIC PREDICTION", "expected to read NEGATIVE", "DISCLOSED DEVELOPMENT RUNS",
              "made the twist worse", "exact derivative, a step accepted only if |twist| decreases",
              "never reuse an old smoke id", "step_status.json", "NO-HOLD baseline", "throw-away seed 140"):
        assert _norm(s) in head, s


def test_launcher_runs_only_the_declared_seeds_and_reads_verdicts_from_json():
    text = LAUNCHER.read_text()
    assert "#SBATCH --time=01:00:00" in text and "#SBATCH --job-name=coop-wristhold" in text
    smoke = text.split('if [ "$MODE" = smoke ]; then', 1)[1].split("\nfi\n", 1)[0]
    assert re.findall(r"--seeds (\S+)", smoke) == ["140", "140", "140"]
    assert re.findall(r"--wristhold-stage (\S+)", smoke) == ["smoke", "smoke", "smoke_nohold"]
    run = text.split("\nfi\n", 1)[1]
    assert re.findall(r"--seeds (\S+)", run) == ["125-129", "20-29", "30-39"]
    assert re.findall(r"--wristhold-stage (\S+)", run) == ["probe", "hold", "place"]
    assert 'if [ "$PV" = PROCEED ]; then' in run and "scored_not_run" in run
    assert "results are never overwritten" in run and 'if [ -e "$RUN" ]' in run
    assert "WRISTHOLD_SMOKE_JOB" in run and "sha256sum -c" in run
    assert "*smoke*)" in smoke and "-smoke" in smoke and "|| true" not in text
    assert subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True).returncode == 0
    assert '"$t" "$s1" "$s2" "$s3"' in smoke and 'step_status="$SM/step_status.json"' in smoke
    assert smoke.index("pytest -q") < smoke.index("step_status.json") < smoke.index("write_verdict smoke")
    assert ("CODE=(src/bhl_robust/eval/scripted_carry.py scripts/bench/coop_scripted_carry.py "
            "tests/test_coop_wristhold.py tests/test_coop_flushpad.py tests/test_scripted_carry.py)") in text
    assert (run.index('if [ -z "$SMOKE_JOB" ]') < run.index('if [ "$SV" != SMOKE_PASS ]')
            < run.index('if [ "$PT" != 0 ]') < run.index("sha256sum -c") < run.index('if [ -e "$RUN" ]')
            < run.index("mkdir -p") < run.index("score --wristhold-stage"))
    assert "m.wristhold_main(sys.argv[1:])" in text and "m.wristhold_verdict_cli(sys.argv[1:])" in text


# --------------------------------------------------------------- launcher: real modes on a fake tree (stubs)

_STUB = r'''"""Fake coop_scripted_carry for the launcher test: records its calls, writes stand-in JSONs, never simulates."""
import json, os, sys
from pathlib import Path


def _log(kind, argv):
    with open(os.environ["FAKE_CALLS"], "a") as f:
        f.write(json.dumps({"kind": kind, "argv": list(argv)}) + "\n")


def wristhold_main(argv):
    _log("score", argv)
    out = Path(argv[argv.index("--out") + 1])
    if out.exists():
        raise SystemExit("COOP-WRISTHOLD: REFUSED (exists)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stub": True, "argv": list(argv)}))
    print("stub score", " ".join(argv[argv.index("--wristhold-stage"):]))
    return int(os.environ.get("FAKE_EXIT_" + argv[argv.index("--wristhold-stage") + 1].upper(), "0"))


def wristhold_verdict_cli(argv):
    _log("verdict", argv)
    stage, out = argv[0], Path(argv[1])
    if out.exists():
        raise SystemExit("COOP-WRISTHOLD-VERDICT: REFUSED (exists)")
    default = {"probe": "NEGATIVE", "scored": "PASS", "scored_not_run": "NOT_RUN", "smoke": "SMOKE_PASS"}[stage]
    rec = {"verdict": os.environ.get("FAKE_VERDICT_" + stage.upper(), default)}
    if stage == "smoke":
        inputs = dict(a.split("=", 1) for a in argv[2:])
        st = json.loads(Path(inputs["step_status"]).read_text())
        rec["step_status"] = st
        if any(v != 0 for v in st.values()):
            rec["verdict"] = "SMOKE_FAIL"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec))
    return 0
'''

_CODE = ("src/bhl_robust/eval/scripted_carry.py", "scripts/bench/coop_scripted_carry.py", "tests/test_coop_wristhold.py",
         "tests/test_coop_flushpad.py", "tests/test_scripted_carry.py")


def _fake_tree(tmp_path):
    root, logs = tmp_path / "repo", tmp_path / "logs"
    for rel in _CODE[2:]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("def test_fake():\n    pass\n")
    (root / _CODE[0]).parent.mkdir(parents=True, exist_ok=True)
    (root / _CODE[0]).write_text("# fake module\n")
    (root / _CODE[1]).parent.mkdir(parents=True, exist_ok=True)
    (root / _CODE[1]).write_text(_STUB)
    dep = root / "external/Berkeley-Humanoid-Lite/logs/rsl_rl/humanoid/2026-08-18_20-57-50_arms-dr1.0-s0/exported"
    dep.mkdir(parents=True)
    (dep / "deploy.yaml").write_text("fake: true\n")
    logs.mkdir()
    text = LAUNCHER.read_text()
    real_repo = "REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder\n"
    real_logs = "LOGS=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs\n"
    assert text.count(real_repo) == 1 and text.count(real_logs) == 1
    text = text.replace(real_repo, f"REPO={root}\n").replace(real_logs, f"LOGS={logs}\n")
    launcher = root / "slurm/repo20260923/cpu_coop_wristhold.sbatch"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(text)
    return root, logs, launcher


def _run(launcher, mode, calls, **env_extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "FAKE_", "WRISTHOLD_"))}
    env.update({"FAKE_CALLS": str(calls), "SLURM_JOB_ID": "fakejob"}, **env_extra)
    r = subprocess.run(["bash", str(launcher), mode], capture_output=True, text=True, env=env, timeout=300)
    seen = [json.loads(ln) for ln in calls.read_text().splitlines()] if calls.exists() else []
    calls.unlink(missing_ok=True)
    return r, seen


def _fake_smoke_dir(root, logs, launcher, smoke_id, verdict="SMOKE_PASS", pytest_status=0):
    sm = logs / f"coop-wristhold-{smoke_id}-smoke"
    sm.mkdir()
    (sm / "smoke_verdict.json").write_text(json.dumps({"verdict": verdict, "step_status": {
        "pytest": pytest_status, "smoke_hold_crew2": 0, "smoke_place_crew4": 0, "smoke_nohold_crew2": 0}}))
    (sm / "code_sha256.txt").write_text(subprocess.run(["sha256sum", *_CODE], cwd=root, capture_output=True,
                                                       text=True, check=True).stdout)
    (sm / "launcher_sha256.txt").write_text(hashlib.sha256(launcher.read_bytes()).hexdigest() + "\n")
    return sm


def _score_calls(seen):
    out = []
    for c in seen:
        if c["kind"] == "score":
            a = c["argv"]
            out.append((a[a.index("--wristhold-stage") + 1], a[a.index("--protocol") + 1], a[a.index("--crew") + 1],
                        a[a.index("--seeds") + 1]))
    return out


def test_launcher_run_mode_on_a_fake_tree_negative_probe_stops(tmp_path):
    root, logs, launcher = _fake_tree(tmp_path)
    _fake_smoke_dir(root, logs, launcher, "s1")
    calls = tmp_path / "calls.jsonl"
    r, seen = _run(launcher, "run", calls, WRISTHOLD_SMOKE_JOB="s1", FAKE_VERDICT_PROBE="NEGATIVE")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "BYTES-CHECK: PASS" in r.stdout and "STAGE 2 NOT RUN" in r.stdout
    assert "COOP-WRISTHOLD RESULT: probe NEGATIVE | scored NOT_RUN" in r.stdout
    assert _score_calls(seen) == [("probe", "lift_hold", "2", "125-129")]
    assert [c["argv"][0] for c in seen if c["kind"] == "verdict"] == ["probe", "scored_not_run"]
    run_dir = root / "results/coop-wristhold-20261003"
    assert (run_dir / "probe/verdict.json").is_file() and (run_dir / "scored/verdict.json").is_file()
    # a second run refuses before anything (the run directory exists)
    r, seen = _run(launcher, "run", calls, WRISTHOLD_SMOKE_JOB="s1", FAKE_VERDICT_PROBE="PROCEED")
    assert r.returncode == 1 and "results are never overwritten" in r.stdout and seen == []


def test_launcher_run_mode_on_a_fake_tree_proceed_runs_the_scored_stage(tmp_path):
    root, logs, launcher = _fake_tree(tmp_path)
    _fake_smoke_dir(root, logs, launcher, "s2")
    calls = tmp_path / "calls.jsonl"
    r, seen = _run(launcher, "run", calls, WRISTHOLD_SMOKE_JOB="s2", FAKE_VERDICT_PROBE="PROCEED",
                   FAKE_VERDICT_SCORED="NEGATIVE")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _score_calls(seen) == [("probe", "lift_hold", "2", "125-129"), ("hold", "lift_hold", "2", "20-29"),
                                  ("hold", "lift_hold", "4", "20-29"), ("place", "lift_place", "2", "30-39"),
                                  ("place", "lift_place", "4", "30-39")]
    verdicts = [c["argv"] for c in seen if c["kind"] == "verdict"]
    assert [v[0] for v in verdicts] == ["probe", "scored"]
    assert sorted(a.split("=")[0] for a in verdicts[1][2:]) == ["hold_crew2", "hold_crew4", "place_crew2", "place_crew4"]
    assert "COOP-WRISTHOLD RESULT: probe PROCEED | scored NEGATIVE" in r.stdout


def test_launcher_run_mode_on_a_fake_tree_a_failed_step_exits_3_but_the_verdict_is_from_json(tmp_path):
    root, logs, launcher = _fake_tree(tmp_path)
    _fake_smoke_dir(root, logs, launcher, "s3")
    calls = tmp_path / "calls.jsonl"
    r, seen = _run(launcher, "run", calls, WRISTHOLD_SMOKE_JOB="s3", FAKE_EXIT_PROBE="2",
                   FAKE_VERDICT_PROBE="NEGATIVE")
    assert r.returncode == 3 and "COOP-WRISTHOLD RESULT: probe NEGATIVE | scored NOT_RUN" in r.stdout


@pytest.mark.parametrize("case", ["no_smoke_id", "smoke_fail", "pytest_1", "code_changed", "launcher_changed",
                                  "no_smoke_dir"])
def test_launcher_run_mode_on_a_fake_tree_refuses_before_anything(tmp_path, case):
    root, logs, launcher = _fake_tree(tmp_path)
    sm = _fake_smoke_dir(root, logs, launcher, "s4", verdict="SMOKE_FAIL" if case == "smoke_fail" else "SMOKE_PASS",
                         pytest_status=1 if case == "pytest_1" else 0)
    if case == "code_changed":
        (root / _CODE[0]).write_text("# fake module, edited after the smoke\n")
    if case == "launcher_changed":
        (sm / "launcher_sha256.txt").write_text("0" * 64 + "\n")
    env = {} if case == "no_smoke_id" else {"WRISTHOLD_SMOKE_JOB": "nosuch" if case == "no_smoke_dir" else "s4"}
    calls = tmp_path / "calls.jsonl"
    r, seen = _run(launcher, "run", calls, FAKE_VERDICT_PROBE="PROCEED", **env)
    assert r.returncode == 1 and "REFUSED" in r.stdout and seen == []
    assert "BYTES-CHECK" not in r.stdout and not (root / "results").exists()


def test_launcher_smoke_mode_on_a_fake_tree_feeds_a_run(tmp_path):
    root, logs, launcher = _fake_tree(tmp_path)
    calls = tmp_path / "calls.jsonl"
    r, seen = _run(launcher, "smoke", calls, SLURM_JOB_NAME="coop-wristhold", SLURM_JOB_ID="sm0")
    assert r.returncode == 1 and "job name must contain 'smoke'" in r.stdout and seen == []
    r, seen = _run(launcher, "smoke", calls, SLURM_JOB_NAME="coop-wristhold-smoke", SLURM_JOB_ID="sm1",
                   FAKE_EXIT_SMOKE_NOHOLD="1")
    assert r.returncode == 1 and "COOP-WRISTHOLD SMOKE: FAIL" in r.stdout
    r, seen = _run(launcher, "smoke", calls, SLURM_JOB_NAME="coop-wristhold-smoke", SLURM_JOB_ID="sm2")
    assert r.returncode == 0 and "COOP-WRISTHOLD SMOKE: PASS" in r.stdout, r.stdout + r.stderr
    assert _score_calls(seen) == [("smoke", "lift_hold", "2", "140"), ("smoke", "lift_place", "4", "140"),
                                  ("smoke_nohold", "lift_hold", "2", "140")]
    sm = logs / "coop-wristhold-sm2-smoke"
    assert json.loads((sm / "step_status.json").read_text()) == {
        "pytest": 0, "smoke_hold_crew2": 0, "smoke_place_crew4": 0, "smoke_nohold_crew2": 0}
    r, seen = _run(launcher, "smoke", calls, SLURM_JOB_NAME="coop-wristhold-smoke", SLURM_JOB_ID="sm2")
    assert r.returncode == 1 and "nothing is overwritten" in r.stdout                  # the smoke dir exists
    r, seen = _run(launcher, "run", calls, WRISTHOLD_SMOKE_JOB="sm2", FAKE_VERDICT_PROBE="NEGATIVE")
    assert r.returncode == 0 and "BYTES-CHECK: PASS" in r.stdout                       # the smoke's bytes feed the run
    r, seen = _run(launcher, "run", tmp_path / "c2.jsonl", WRISTHOLD_SMOKE_JOB="sm1")
    assert r.returncode == 1 and "reads SMOKE_FAIL, not SMOKE_PASS" in r.stdout and seen == []


# --------------------------------------------------------------- stock and flush-pad paths unchanged: diff to 3a67bc2

def _git_show(path):
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{BASE_COMMIT}:{path}"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr                                          # fail loudly, never skip
    return r.stdout


def _top_names(lines):
    return {m.group(2) for ln in lines for m in [re.match(r"^(def|class) (\w+)", ln)] if m}


@pytest.mark.parametrize("path", ["src/bhl_robust/eval/scripted_carry.py", "scripts/bench/coop_scripted_carry.py"])
def test_source_diff_to_base_commit_is_one_marked_insertion(path):
    old = _git_show(path).splitlines()
    new = (REPO / path).read_text().splitlines()
    starts = [i for i, ln in enumerate(new) if ln.startswith("# --- coop-wristhold ---")]
    ends = [i for i, ln in enumerate(new) if ln == "# --- end coop-wristhold ---"]
    assert len(starts) == 1 and len(ends) == 1 and starts[0] < ends[0]
    s, e = starts[0], ends[0]
    assert not any("coop-wristhold" in ln for ln in old)
    ops = difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
    assert {op for op, *_ in ops} <= {"equal", "insert"}                        # insertions only
    inserted = [j for op, i1, i2, j1, j2 in ops if op == "insert" for j in range(j1, j2)]
    assert set(range(s, e + 1)) <= set(inserted)                                # the whole section is new
    assert all(not new[j].strip() for j in inserted if not s <= j <= e)          # outside it: blank lines only
    assert len(inserted) - (e - s + 1) <= 2
    section = new[s:e + 1]
    assert not _top_names(section) & _top_names(old)                            # it shadows no stock name
    if path.startswith("scripts"):                                              # main() and its flags unchanged
        assert "--wristhold-stage" not in "\n".join(new[:s] + new[e + 1:])


def _base_module(tmp_path):
    path = tmp_path / "scripted_carry_base.py"
    path.write_text(_git_show("src/bhl_robust/eval/scripted_carry.py"))
    spec = importlib.util.spec_from_file_location("scripted_carry_base_3a67bc2", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mjb_sha(m):
    import mujoco
    buf = np.zeros(mujoco.mj_sizeModel(m), dtype=np.uint8)
    mujoco.mj_saveModel(m, None, buf)
    return hashlib.sha256(buf.tobytes()).hexdigest()


@needs_assets
def test_stock_and_flushpad_models_are_byte_identical_to_the_base_commit_and_w_uses_the_flushpad_model(tmp_path):
    base = _base_module(tmp_path)
    cache = tmp_path / "cache"
    for old_p, new_p in ((base.CarryParams(), sc.CarryParams()), (base.FlushPadParams(), sc.FlushPadParams())):
        m_old, _, _ = base.build_carry(UPSTREAM, cache, 1, old_p)
        m_new, _, _ = sc.build_carry(UPSTREAM, cache, 1, new_p)
        assert _mjb_sha(m_new) == _mjb_sha(m_old)
    m_f, _, _ = sc.build_carry(UPSTREAM, cache, 1, sc.FlushPadParams())
    m_w, _, _ = sc.build_carry(UPSTREAM, cache, 1, sc.WristHoldParams())
    assert _mjb_sha(m_w) == _mjb_sha(m_f)                                       # the hold changes no model byte


def _policy():
    from omegaconf import OmegaConf
    sys.path.insert(0, str(REPO / "scripts" / "bench"))
    from team_airlock import CpuPolicy
    cfg = OmegaConf.load(DEPLOY)
    return cfg, CpuPolicy(cfg.policy_checkpoint_path)


def _recorder(store):
    def hook(**kw):
        store.append((kw["runner"].d.ctrl.copy(), kw["runner"].d.qpos.copy()))
    return hook


@needs_assets
def test_stock_and_flushpad_episodes_are_identical_to_the_base_commit(tmp_path):
    base = _base_module(tmp_path)
    cfg, policy = _policy()
    for kind in ("stock", "flushpad"):
        runs = {}
        for tag, mod in (("base", base), ("now", sc)):
            p = mod.CarryParams() if kind == "stock" else mod.FlushPadParams()
            m, slots, pairs = mod.build_carry(UPSTREAM, tmp_path / "cache", 1, p)
            store = []
            if kind == "stock":
                ep = mod.run_episode(m, slots, pairs, cfg, policy, SEED, p, frame_hook=_recorder(store),
                                     rule=dict(mod.LIFT_HOLD_RULE, episode_s=1.0), protocol="lift_hold")
            else:
                ep = mod.run_flushpad_episode(m, slots, pairs, cfg, policy, SEED, p, frame_hook=_recorder(store),
                                              rule=dict(mod.flushpad_stage_rule("smoke", "lift_hold", [SEED]),
                                                        episode_s=1.0))
            runs[tag] = (np.array([c for c, _ in store]), np.array([q for _, q in store]), ep["steps"])
        assert runs["base"][2] == runs["now"][2] == 25
        assert np.array_equal(runs["base"][0], runs["now"][0]) and np.array_equal(runs["base"][1], runs["now"][1])


# --------------------------------------------------------------- the W plumbing and the hold on the real harness

@needs_assets
def test_wrist_hold_design_record(tmp_path):
    des = sc.wristhold_design(UPSTREAM, tmp_path, sc.WristHoldParams())
    assert des["chosen"] == ["elbow_roll"] and des["pinch_axis_world"] == [1.0, 0.0, 0.0]
    for side in ("left", "right"):
        sd = des["sides"][side]
        assert sd["chain"][-1].startswith(f"arm_{side}_hand_link (welded")
        for tag in ("squeeze_contact", "lift_contact"):
            pr = sd["projections"][tag]["axis_dot_pinch"]
            assert abs(pr["shoulder_pitch"]) == pytest.approx(1.0, abs=1e-6) and abs(pr["shoulder_roll"]) < 1e-6
            assert abs(pr["elbow_pitch"]) > 0.9 and abs(pr["elbow_roll"]) < 0.3 and abs(pr["shoulder_yaw"]) < 0.35
            assert sd["projections"][tag]["joints_left_order"][1] < 0.1             # the cube stops the roll near 0
        summ = sd["candidate_summary"]
        assert abs(summ["elbow_roll"]["twist_at_lift_contact_no_hold_rad"]) == pytest.approx(0.98, abs=0.02)
        assert all(abs(c["twist_at_lift_contact_hold_rad"]) > 0.35 for c in summ.values())
        assert summ["elbow_roll"]["pad_moved_max_m"] < 0.01 < 0.1 < summ["shoulder_yaw"]["pad_moved_max_m"]
    assert "probe expected NEGATIVE" in des["prediction"]
    assert sc.wristhold_design(UPSTREAM, tmp_path, sc.WristHoldParams()) == des    # deterministic


@needs_assets
@pytest.mark.parametrize("protocol,seconds", [("lift_hold", 8.0), ("lift_place", 13.0)])
def test_w_plumbing_with_the_hold_neutralised_is_the_flushpad_episode_step_for_step(tmp_path, monkeypatch, protocol,
                                                                                    seconds):
    # covers the grasp state (113), the lift and, for lift_place, the release and the fade window
    cfg, policy = _policy()
    monkeypatch.setattr(sc.WristHold, "_filter", lambda self, targets: targets)
    out = {}
    for tag in ("flush", "w"):
        p = sc.FlushPadParams() if tag == "flush" else sc.WristHoldParams()
        m, slots, pairs = sc.build_carry(UPSTREAM, tmp_path / "cache", 1, p)
        store = []
        if tag == "flush":
            ep = sc.run_flushpad_episode(m, slots, pairs, cfg, policy, SEED, p, frame_hook=_recorder(store),
                                         rule=dict(sc.flushpad_stage_rule("smoke", protocol, [SEED]), episode_s=seconds))
        else:
            ep = sc.run_wristhold_episode(m, slots, pairs, cfg, policy, SEED, p, frame_hook=_recorder(store),
                                          rule=dict(sc.wristhold_stage_rule("smoke", protocol, [SEED]),
                                                    episode_s=seconds))
            assert ep["wrist_hold"]["installed"] and ep["wrist_hold"]["installed_after_step"] == 0
        out[tag] = (np.array([c for c, _ in store]), np.array([q for _, q in store]), ep)
    assert out["flush"][0].shape[0] == int(round(seconds / DT))
    assert np.array_equal(out["flush"][0], out["w"][0]) and np.array_equal(out["flush"][1], out["w"][1])
    a, b = out["flush"][2]["pairs"][0], out["w"][2]["pairs"][0]
    assert a["cube_tilt_series_rad"] == b["cube_tilt_series_rad"] and a["lift_series_m"] == b["lift_series_m"]


@needs_assets
def test_nohold_baseline_is_the_flushpad_episode_step_for_step(tmp_path):
    cfg, policy = _policy()
    p = sc.FlushPadParams()
    m, slots, pairs = sc.build_carry(UPSTREAM, tmp_path / "cache", 1, p)
    sa, sb = [], []
    ea = sc.run_flushpad_episode(m, slots, pairs, cfg, policy, SEED, p, frame_hook=_recorder(sa),
                                 rule=dict(sc.flushpad_stage_rule("smoke", "lift_hold", [SEED]), episode_s=6.0))
    eb = sc.run_wristhold_episode(m, slots, pairs, cfg, policy, SEED, p, frame_hook=_recorder(sb),
                                  rule=dict(sc.wristhold_stage_rule("smoke_nohold", "lift_hold", [SEED]), episode_s=6.0))
    assert np.array_equal(np.array([c for c, _ in sa]), np.array([c for c, _ in sb]))
    assert np.array_equal(np.array([q for _, q in sa]), np.array([q for _, q in sb]))
    rec = eb["wrist_hold"]
    assert rec["servo"] is False and all(not a["servo_log"]["step"] for a in rec["arms"])
    assert all(abs(a["twist_series_rad"][K_G]) < 1e-9 and a["twist_series_rad"][K_G - 1] is None for a in rec["arms"])
    assert sc._wristhold_record_problem(eb, False, DT) is None and ea["steps"] == eb["steps"] == 150
    with pytest.raises(ValueError):
        sc.run_wristhold_episode(m, slots, pairs, cfg, policy, SEED, sc.WristHoldParams(),
                                 rule=sc.wristhold_stage_rule("smoke_nohold", "lift_hold", [SEED]))
    with pytest.raises(ValueError):
        sc.run_wristhold_episode(m, slots, pairs, cfg, policy, SEED, p,
                                 rule=sc.wristhold_stage_rule("smoke", "lift_hold", [SEED]))
    with pytest.raises(ValueError):
        sc.run_wristhold_episode(m, slots, pairs, cfg, policy, SEED, sc.WristHoldParams(),
                                 rule=sc.FLUSHPAD_PROBE_RULE)


@needs_assets
def test_the_hold_on_the_harness_only_replaces_wrist_targets_within_range_and_cap(tmp_path, monkeypatch):
    cfg, policy = _policy()
    p = sc.WristHoldParams()
    m, slots, pairs = sc.build_carry(UPSTREAM, tmp_path / "cache", 1, p)
    seen = []
    original = sc.WristHold._filter

    def spy(self, targets):
        held = original(self, targets)
        seen.append((self.next_step, [np.array(t, dtype=float) for t in targets],
                     [np.array(t, dtype=float) for t in held],
                     {a["robot"]: [j["idx"] for j in a["joints"]] for a in self.arms}))
        return held
    monkeypatch.setattr(sc.WristHold, "_filter", spy)
    ep = sc.run_wristhold_episode(m, slots, pairs, cfg, policy, SEED, p,
                                  rule=dict(sc.wristhold_stage_rule("smoke", "lift_hold", [SEED]), episode_s=6.5))
    assert ep["steps"] == 162 and ep["failed"] is None and ep["sim_reset"]["detected"] is False
    assert [k for k, *_ in seen] == list(range(1, 162))                          # wrapped from policy step 1 on
    for k, before, after, wrist in seen:
        for i, (b, a) in enumerate(zip(before, after)):
            other = np.ones(22, dtype=bool)
            other[wrist.get(i, [])] = False
            assert np.array_equal(b[other], a[other])                           # only the wrist entries may change
            if k < K_G:
                assert np.array_equal(b, a)                                     # nothing before the grasp step
    names = sc.actuator_joint_names(m, slots[0])
    assert {r: [names[x] for x in ix] for r, ix in seen[0][3].items()} == {
        0: ["arm_right_elbow_roll_joint"], 1: ["arm_left_elbow_roll_joint"]}
    rec = ep["wrist_hold"]
    assert sc._wristhold_record_problem(ep, True, DT) is None
    for a in rec["arms"]:
        lg, s = a["servo_log"], a["summary"]
        assert lg["step"][0] == K_G and lg["step"] == list(range(K_G, 162)) and set(lg["mode"]) == {"hold"}
        lo, hi = s["wrist_range"][0]
        assert all(lo - 1e-12 <= t[0] <= hi + 1e-12 for t in lg["target"])       # within the joint range
        assert all(abs(c[0]) <= 4.0 + 1e-9 for c in lg["ctrl_after_nm"])        # never above the 4 Nm cap
        assert s["cap_nm"] == [4.0] and abs(a["twist_series_rad"][K_G]) < 1e-9
        assert lg["target"][0][0] == pytest.approx(lg["q"][0][0], abs=1e-9)     # at grasp: hold where it is
    json.dumps(ep, allow_nan=False)


@needs_assets
def test_lift_place_hold_runs_through_the_release_then_fades_without_a_jump(tmp_path):
    cfg, policy = _policy()
    p = sc.WristHoldParams()
    m, slots, pairs = sc.build_carry(UPSTREAM, tmp_path / "cache", 1, p)
    ep = sc.run_wristhold_episode(m, slots, pairs, cfg, policy, SEED, p,
                                  rule=dict(sc.wristhold_stage_rule("smoke", "lift_place", [SEED]), episode_s=13.5))
    rec = ep["wrist_hold"]
    assert rec["window"] == {"from_state": K_G, "hold_until_t_s": pytest.approx(11.7),
                             "fade_until_t_s": pytest.approx(12.7)}
    for a in rec["arms"]:
        lg = a["servo_log"]
        hold = [k for k, md in zip(lg["step"], lg["mode"]) if md == "hold"]
        fade = [k for k, md in zip(lg["step"], lg["mode"]) if md == "fade"]
        assert hold == list(range(K_G, 293)) and fade == list(range(293, 318))   # 4.52-11.68 s, then 11.72-12.68 s
        tg = [t[0] for t in lg["target"]]
        steps = np.abs(np.diff(tg[len(hold) - 1:]))
        assert steps.max() < 0.1                                                # the fade never jumps the target
        assert abs(tg[-1]) < 0.01                                               # back to the script's 0 by 12.7 s
    assert sc._wristhold_record_problem(ep, True, DT) is None
