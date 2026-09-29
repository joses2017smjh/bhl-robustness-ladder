"""Scripted-arm cube carry: the rule, the verdict, the seed choice, the model."""

from pathlib import Path

import numpy as np
import pytest

from bhl_robust.eval import scripted_carry as sc

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "external/Berkeley-Humanoid-Lite"
DEPLOY = UPSTREAM / "logs/rsl_rl/humanoid/2026-08-18_20-57-50_arms-dr1.0-s0/exported/deploy.yaml"
DT = 0.04
N = int(sc.SUCCESS_RULE["episode_s"] / DT)


def _good_series():
    """Lifted to 12 cm at 8 s, carried 1.2 m by 20 s, held to the end."""
    t = np.arange(N) * DT
    lift = np.where(t < 6, 0.0, np.minimum(0.12, 0.06 * (t - 6)))
    horiz = np.clip(0.1 * (t - 8), 0.0, 1.2)
    return lift, horiz


def _score(lift, horiz, **kw):
    args = {"max_tilt": 0.2, "floor": False, "failed": None}
    args.update(kw)
    return sc.score_pair(lift, horiz, DT, **args)


def test_rule_numbers_are_the_predeclared_ones():
    r = sc.SUCCESS_RULE
    assert (r["episode_s"], r["lift_peak_m"], r["lift_hold_m"], r["lift_hold_s"]) == (30.0, 0.10, 0.05, 3.0)
    assert (r["carry_m"], r["tilt_rad"], r["pass_min"]) == (1.0, 0.78, 8)
    assert r["seeds"] == list(range(10))


def test_good_episode_succeeds_with_completion_time():
    lift, horiz = _good_series()
    row = _score(lift, horiz)
    assert row["success"] and row["first_failed_check"] is None
    # completion = first step >= 1.0 m while >= 5 cm up: 8 s + 10 s
    assert row["completion_s"] == pytest.approx(18.0, abs=DT)


@pytest.mark.parametrize("mutate,check", [
    (lambda l, h: (np.minimum(l, 0.09), h), "lift_peak"),
    (lambda l, h: (np.where(np.arange(N) % 70 == 0, 0.0, l), h), "lift_hold"),
    (lambda l, h: (l, np.minimum(h, 0.99)), "carry"),
])
def test_each_series_check_fails_on_its_own(mutate, check):
    lift, horiz = mutate(*_good_series())
    row = _score(lift, horiz)
    assert not row["success"] and not row["checks"][check]


def test_carry_only_counts_while_lifted():
    lift, horiz = _good_series()
    lift = np.where(np.arange(N) * DT > 15, 0.0, lift)     # dropped at 15 s, cube keeps sliding
    row = _score(lift, horiz)
    assert row["carry_m"] < 1.0 and not row["checks"]["carry"]


@pytest.mark.parametrize("kw,check", [
    ({"max_tilt": 0.79}, "no_fall"), ({"floor": True}, "no_floor_contact"),
    ({"failed": "nonfinite_state"}, "finite"),
])
def test_fall_floor_and_nonfinite_fail(kw, check):
    row = _score(*_good_series(), **kw)
    assert not row["success"] and not row["checks"][check]


def test_truncated_episode_fails():
    lift, horiz = _good_series()
    row = _score(lift[:-10], horiz[:-10])
    assert not row["checks"]["full_episode"] and not row["success"]


def _episodes(wins: list[bool], n_pairs=1, completion=None):
    eps = []
    for s, w in enumerate(wins):
        rows = [{"success": w, "first_failed_check": None if w else "carry", "lift_peak_m": 0.2,
                 "lift_hold_s": 5.0, "carry_m": 1.2 if w else 0.3,
                 "completion_s": (completion[s] if completion else 20.0) if w else None}
                for _ in range(n_pairs)]
        eps.append({"seed": s, "pairs": rows})
    return eps


def test_summary_gate_is_eight_of_ten():
    assert sc.summarize(_episodes([True] * 8 + [False] * 2), 1)["pass"]
    s = sc.summarize(_episodes([True] * 7 + [False] * 3), 1)
    assert not s["pass"] and s["complete"] and s["per_pair"][0]["successes"] == 7


def test_summary_incomplete_never_passes():
    s = sc.summarize(_episodes([True] * 9), 1)
    assert not s["complete"] and not s["pass"]


def test_crew4_gates_each_pair():
    eps = _episodes([True] * 10, n_pairs=2)
    for e in eps[:3]:
        e["pairs"][1]["success"] = False
        e["pairs"][1]["first_failed_check"] = "lift_peak"
    s = sc.summarize(eps, 2)
    assert s["per_pair"][0]["pass"] and not s["per_pair"][1]["pass"] and not s["pass"]


def test_median_seed_is_the_lower_median_by_completion():
    eps = _episodes([True, True, False, True, True], completion=[30, 10, None, 20, 25])
    # successful completions: 10 (s1), 20 (s3), 25 (s4), 30 (s0) -> lower median = 20 -> seed 3
    assert sc.median_seed(eps) == 3
    assert sc.median_seed(_episodes([False, False])) is None


def test_sync_speed_waits_and_catches_up():
    p = sc.CarryParams()
    assert sc.sync_speed(0.0, p) == pytest.approx(p.carry_speed)
    assert sc.sync_speed(0.05, p) > p.carry_speed              # lagging: faster
    assert sc.sync_speed(-0.05, p) == 0.0                      # ahead by 5 cm: wait
    assert sc.sync_speed(1.0, p) == pytest.approx(p.v_max)


def test_compose_replaces_only_the_grasping_arm():
    p = sc.CarryParams()
    pol = np.arange(22, dtype=float)
    q = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    left = sc.compose_arm_targets(pol, q, "left", p)
    assert np.allclose(left[:5], q) and np.allclose(left[5:], pol[5:])
    right = sc.compose_arm_targets(pol, q, "right", p)
    assert np.allclose(right[5:10], -q) and np.allclose(right[:5], pol[:5])
    assert np.allclose(right[10:], pol[10:])


def test_arm_script_is_continuous_and_ends_in_lift():
    p = sc.CarryParams()
    s = sc.ArmScript(p)
    ts = np.arange(0, 12, 0.04)
    qs = np.array([s(t) for t in ts])
    assert np.abs(np.diff(qs, axis=0)).max() < 0.05          # no jumps at the knots
    assert np.allclose(qs[0], sc.KEYFRAMES_LEFT["rest"])
    assert np.allclose(qs[-1], sc.KEYFRAMES_LEFT["lift"])
    assert s.phase(0.1) == "settle" and s.phase(s.t_carry + 0.1) == "carry"


# --------------------------------------------------------------- model checks

needs_assets = pytest.mark.skipif(not (UPSTREAM / "source").is_dir(), reason="upstream assets missing")


@pytest.fixture(scope="module")
def ik(tmp_path_factory):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(DEPLOY)
    leg = {n: v for n, v in zip(cfg.joints, cfg.default_joint_positions) if n.startswith("leg")}
    return sc.ArmIK(UPSTREAM, tmp_path_factory.mktemp("ik"), leg)


@needs_assets
def test_right_arm_is_the_mirror_of_the_left(ik):
    for name, q in sc.KEYFRAMES_LEFT.items():
        cl, _, _ = ik.pad_fk("left", q)
        cr, _, _ = ik.pad_fk("right", -np.asarray(q))
        assert cl[0] == pytest.approx(cr[0], abs=0.01) and cl[2] == pytest.approx(cr[2], abs=0.01), name
        assert cl[1] == pytest.approx(-cr[1], abs=0.01), name


@needs_assets
def test_keyframe_geometry_matches_the_design(ik):
    p = sc.CarryParams()
    face = p.side_off - sc.CUBE_HALF                            # lateral distance to the cube face
    # the hand starts clear of the face and the squeeze/lift targets sit inside it
    assert ik.pad_lateral_extent("left", sc.KEYFRAMES_LEFT["rest"]) < face
    assert ik.pad_lateral_extent("left", sc.KEYFRAMES_LEFT["reach"]) < face
    assert ik.pad_lateral_extent("left", sc.KEYFRAMES_LEFT["squeeze"]) > face + 0.1
    assert ik.pad_lateral_extent("left", sc.KEYFRAMES_LEFT["lift"]) > face + 0.1
    z_sq = ik.pad_fk("left", sc.KEYFRAMES_LEFT["squeeze"])[0][2]
    z_li = ik.pad_fk("left", sc.KEYFRAMES_LEFT["lift"])[0][2]
    assert z_li - z_sq > sc.SUCCESS_RULE["lift_peak_m"]      # the script asks for more than the rule


@needs_assets
def test_crew4_model_compiles_with_pads_and_deploy_order(tmp_path):
    from omegaconf import OmegaConf
    import mujoco
    cfg = OmegaConf.load(DEPLOY)
    p = sc.CarryParams()
    model, slots, pairs = sc.build_carry(UPSTREAM, tmp_path, 2, p)
    assert len(slots) == 4 and len(pairs) == 2
    for s in slots:
        assert sc.actuator_joint_names(model, s) == list(cfg.joints)
        for side in ("left", "right"):
            g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{s.prefix}arm_{side}_hand_pad")
            assert g >= 0 and model.geom_contype[g] and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
    # no welds, no equality constraints, no mocap: the cube is a free body
    assert model.neq == 0 and model.nmocap == 0
    for pr in pairs:
        j = model.body_jntadr[pr.cube_body]
        assert model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    # the pairs do not overlap: neighbouring robots of different pairs >= 0.5 m apart
    xs = sorted(float(model.body_pos[model.body_rootid[s.body_id]][0]) for s in slots)
    assert min(np.diff(xs)) > 0.5


# --------------------------------------------------------------- lift and hold

NL = int(sc.LIFT_HOLD_RULE["episode_s"] / DT)


def _hold_series(hold_s=12.0, peak=0.15):
    """Lifted to `peak` from 4.5 s to 7 s, then held for `hold_s`, then dropped to 0."""
    t = np.arange(NL) * DT
    ramp = np.clip((t - 4.5) / 2.5, 0.0, 1.0) * peak
    up_end = 7.0 + hold_s
    lift = np.where(t < up_end, ramp, 0.0)
    return lift, np.zeros(NL)


def _score_lh(lift, horiz=None, **kw):
    args = {"max_tilt": 0.2, "floor": False, "failed": None}
    args.update(kw)
    return sc.score_lift_hold(lift, np.zeros(len(lift)) if horiz is None else horiz, DT, **args)


def test_lift_hold_rule_numbers_are_the_predeclared_ones():
    r = sc.LIFT_HOLD_RULE
    assert r["protocol"] == "lift_hold"
    assert (r["episode_s"], r["lift_peak_m"], r["lift_hold_m"], r["lift_hold_s"]) == (20.0, 0.10, 0.05, 5.0)
    assert (r["tilt_rad"], r["floor_contact"], r["pass_min"]) == (0.78, False, 8)
    assert r["seeds"] == list(range(10)) and "carry_m" not in r
    assert sc.LIFT_HOLD_NOTE == "cooperative lift and hold — carry not achieved"
    assert sc.PROTOCOLS == ("carry", "lift_hold", "lift_place")     # lift_place added 2026-09-27


def test_carry_rule_and_params_are_unchanged():
    # the scored carry run (21434982) and its render gate depend on these exactly
    assert sc.SUCCESS_RULE == {"episode_s": 30.0, "lift_peak_m": 0.10, "lift_hold_m": 0.05,
                               "lift_hold_s": 3.0, "carry_m": 1.0, "tilt_rad": 0.78,
                               "floor_contact": False, "seeds": list(range(10)), "pass_min": 8}
    assert "protocol" not in sc.SUCCESS_RULE
    assert set(sc.params_dict(sc.CarryParams())) == {
        "side_off", "cube_fwd", "cube_z", "plinth_half", "pair_pitch", "hand_pads", "pad_friction",
        "t_settle", "t_reach", "t_squeeze", "t_lift", "t_hold", "carry_speed", "carry_goal_m",
        "carry_gate_lift_m", "k_sync", "v_max", "v_min", "grasp_kp", "grasp_arm_mode",
        "outer_arm_mode", "reset_jitter", "xy_jitter"}


def test_good_lift_hold_succeeds_and_ignores_drift():
    lift, _ = _hold_series()
    row = _score_lh(lift, horiz=np.full(NL, 3.0))          # drift is reported, never scored
    assert row["success"] and row["first_failed_check"] is None
    assert "carry" not in row["checks"] and row["horiz_max_m"] == 3.0
    assert row["lift_hold_s"] >= 12.0 and row["hold_start_s"] < row["hold_end_s"]


def test_hold_of_exactly_five_seconds_passes_and_just_under_fails():
    t = np.arange(NL) * DT
    n5 = int(round(5.0 / DT))
    for n, ok in ((n5, True), (n5 - 1, False)):
        lift = np.where((t >= 2.0) & (np.arange(NL) < int(round(2.0 / DT)) + n), 0.12, 0.0)
        row = _score_lh(lift)
        assert row["checks"]["lift_hold"] is ok, n
        assert row["success"] is ok


@pytest.mark.parametrize("mutate,check", [
    (lambda l: np.minimum(l, 0.099), "lift_peak"),
    (lambda l: np.where(np.arange(NL) % 120 == 0, 0.0, l), "lift_hold"),   # never 5 s contiguous
    (lambda l: l[:-5], "full_episode"),
])
def test_each_lift_hold_series_check_fails_on_its_own(mutate, check):
    lift = mutate(_hold_series()[0])
    row = _score_lh(lift)
    assert not row["success"] and not row["checks"][check]
    assert sum(not v for v in row["checks"].values()) == 1


@pytest.mark.parametrize("kw,check", [
    ({"max_tilt": 0.781}, "no_fall"), ({"floor": True}, "no_floor_contact"),
    ({"failed": "nonfinite_action"}, "finite"),
])
def test_lift_hold_fall_floor_and_nonfinite_fail(kw, check):
    row = _score_lh(_hold_series()[0], **kw)
    assert not row["success"] and row["first_failed_check"] == check


def test_lift_hold_longest_run_is_contiguous():
    t = np.arange(NL) * DT
    # two 3 s holds separated by a dip: 6 s total up, but never 5 s contiguous
    lift = np.where(((t >= 5) & (t < 8)) | ((t >= 8.2) & (t < 11.2)), 0.12, 0.0)
    row = _score_lh(lift)
    assert row["lift_hold_s"] == pytest.approx(3.0, abs=DT) and not row["checks"]["lift_hold"]


def _lh_episodes(wins, n_pairs=1, holds=None):
    eps = []
    for s, w in enumerate(wins):
        rows = [{"success": w, "first_failed_check": None if w else "lift_hold", "lift_peak_m": 0.18,
                 "lift_hold_s": (holds[s] if holds else 12.0) if w else 2.0,
                 "max_tilt_rad": 0.2, "cube_floor_contact": not w}
                for _ in range(n_pairs)]
        eps.append({"seed": s, "pairs": rows})
    return eps


def test_lift_hold_summary_gate_is_eight_of_ten_per_pair():
    assert sc.summarize_lift_hold(_lh_episodes([True] * 8 + [False] * 2), 1)["pass"]
    s = sc.summarize_lift_hold(_lh_episodes([True] * 7 + [False] * 3), 1)
    assert s["complete"] and not s["pass"] and s["per_pair"][0]["floor_contacts"] == 3
    assert not sc.summarize_lift_hold(_lh_episodes([True] * 9), 1)["complete"]
    eps = _lh_episodes([True] * 10, n_pairs=2)
    for e in eps[:3]:
        e["pairs"][1].update(success=False, first_failed_check="no_fall", max_tilt_rad=1.4)
    s = sc.summarize_lift_hold(eps, 2)
    assert s["per_pair"][0]["pass"] and not s["per_pair"][1]["pass"] and not s["pass"]
    assert s["per_pair"][1]["falls"] == 3


def test_median_seed_by_hold_is_the_lower_median():
    eps = _lh_episodes([True, True, False, True, True], holds=[12.9, 6.0, None, 9.0, 11.0])
    # successful holds: 6.0 (s1), 9.0 (s3), 11.0 (s4), 12.9 (s0) -> lower median = 9.0 -> seed 3
    assert sc.median_seed_by_hold(eps) == 3
    # an episode's hold is its weakest pair's
    eps = _lh_episodes([True, True, True], n_pairs=2, holds=[12.0, 12.0, 12.0])
    eps[0]["pairs"][1]["lift_hold_s"] = 5.5
    eps[2]["pairs"][0]["lift_hold_s"] = 7.0
    assert sc.median_seed_by_hold(eps) == 2
    assert sc.median_seed_by_hold(_lh_episodes([False, False])) is None


def _lh_payload(wins, scored=True, rule=None, n_pairs=1):
    return {"protocol": "lift_hold", "rule": dict(rule or sc.LIFT_HOLD_RULE), "scored_run": scored,
            "crew": 2 * n_pairs, "pairs": n_pairs, "label": sc.LABEL,
            "episodes": _lh_episodes(wins, n_pairs=n_pairs)}


def test_lift_hold_verdict_line_is_computed_from_the_json():
    line = sc.lift_hold_verdict_line(_lh_payload([True] * 8 + [False] * 2))
    assert line.startswith("COOP-LIFT-HOLD crew 2: PASS |") and "pair0=8/10" in line
    assert sc.LIFT_HOLD_NOTE in line and sc.LABEL in line
    assert sc.lift_hold_verdict_line(_lh_payload([True] * 7 + [False] * 3)).startswith(
        "COOP-LIFT-HOLD crew 2: NEGATIVE |")
    assert sc.lift_hold_verdict_line(_lh_payload([True] * 10, scored=False)).startswith(
        "COOP-LIFT-HOLD crew 2: INCOMPLETE |")
    assert sc.lift_hold_verdict_line(_lh_payload([True] * 9)).startswith("COOP-LIFT-HOLD crew 2: INCOMPLETE |")
    # a relaxed rule in the JSON is never scored
    assert sc.lift_hold_verdict_line(_lh_payload([True] * 10, rule=dict(sc.LIFT_HOLD_RULE, lift_hold_s=3.0))
                                     ).startswith("COOP-LIFT-HOLD crew 2: INVALID")
    carry_json = dict(_lh_payload([True] * 10), protocol=None)
    assert "INVALID" in sc.lift_hold_verdict_line(carry_json)
    # crew 4: each pair gated
    p4 = _lh_payload([True] * 10, n_pairs=2)
    for e in p4["episodes"][:3]:
        e["pairs"][0]["success"] = False
        e["pairs"][0]["first_failed_check"] = "lift_peak"
    assert sc.lift_hold_verdict_line(p4).startswith("COOP-LIFT-HOLD crew 4: NEGATIVE |")


def test_arm_script_lift_hold_only_relabels_the_phase():
    p = sc.CarryParams()
    carry, hold = sc.ArmScript(p), sc.ArmScript(p, hold_forever=True)
    for t in np.arange(0, 20, 0.04):
        assert np.allclose(carry(t), hold(t))
    t = hold.t_carry + 5.0
    assert carry.phase(t) == "carry" and hold.phase(t) == "hold"
    assert hold.phase(0.1) == "settle"


def test_run_episode_rejects_a_mismatched_rule():
    with pytest.raises(ValueError):
        sc.run_episode(None, [], [], None, None, 0, sc.CarryParams(), rule=sc.SUCCESS_RULE,
                       protocol="lift_hold")
    with pytest.raises(ValueError):
        sc.run_episode(None, [], [], None, None, 0, sc.CarryParams(), rule=sc.LIFT_HOLD_RULE)
    with pytest.raises(ValueError):
        sc.run_episode(None, [], [], None, None, 0, sc.CarryParams(), protocol="walk")


def test_gif_gate_deletes_an_over_budget_gif(tmp_path):
    gif, side = tmp_path / "x.gif", tmp_path / "x.json"
    gif.write_bytes(b"GIF89a")
    side.write_text("{}")
    assert sc.enforce_gif_budget({"within_budget": True}, gif, side) == {"kept": True, "deleted": []}
    assert gif.exists() and side.exists()
    out = sc.enforce_gif_budget({"within_budget": False}, gif, side)
    assert not out["kept"] and not gif.exists() and not side.exists()
    assert sc.enforce_gif_budget(None, gif)["kept"] is False       # nothing left to delete


def _script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("coop_scripted_carry",
                                                  REPO / "scripts/bench/coop_scripted_carry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lift_hold_score_refuses_to_overwrite(tmp_path):
    import argparse
    mod = _script()
    out = tmp_path / "score_lifthold_crew2.json"
    out.write_text("{}")
    args = argparse.Namespace(protocol="lift_hold", out=out, seeds="100", seconds=None, crew=2)
    with pytest.raises(SystemExit, match="REFUSED"):
        mod.score(args)
    assert out.read_text() == "{}"


def test_lift_hold_render_refuses_a_carry_json_and_skips_a_negative(tmp_path, capsys):
    import argparse
    import json
    mod = _script()
    carry = tmp_path / "score_crew2.json"
    carry.write_text(json.dumps({"crew": 2, "scored_run": True, "summary": {"verdict": "PASS", "median_seed": 0}}))
    args = argparse.Namespace(protocol="lift_hold", render_from=carry, pipeline_check_seed=None,
                              no_render=True, crew=2, out_dir=tmp_path, gif=None)
    with pytest.raises(SystemExit, match="protocol"):
        mod.render(args)
    neg = tmp_path / "score_lifthold_crew2.json"
    neg.write_text(json.dumps({"crew": 2, "protocol": "lift_hold", "scored_run": True,
                               "summary": {"verdict": "NEGATIVE", "median_seed": None}}))
    args.render_from = neg
    assert mod.render(args) == 0
    assert "COOP_LIFT_HOLD_RENDER=SKIPPED verdict=NEGATIVE" in capsys.readouterr().out
    assert sorted(p.name for p in tmp_path.iterdir()) == ["score_crew2.json", "score_lifthold_crew2.json"]


# ------------------------------------------------------------------ lift, hold and place

def _final(dz=0.0, speed=0.0, off=(0.01, -0.02), contact=False):
    return {"dz_m": dz, "speed_mps": speed, "offset_xy_m": list(off), "cube_tilt_rad": 1.57,
            "robot_contact": contact, "normal_N": {"b": 0.0, "a": 0.0, "plinth": 4.9}}


def _place_series(hold_s=4.0, peak=0.18):
    """Up at 6.2 s, >= 5 cm for `hold_s`, back on the plinth (lift 0) to the end of 20 s."""
    n = int(sc.LIFT_PLACE_RULE["episode_s"] / DT)
    t = np.arange(n) * DT
    return np.where((t >= 6.2) & (t < 6.2 + hold_s - 1e-9), peak, 0.0)


def _score_lp(lift=None, final=None, **kw):
    args = {"max_tilt": 0.2, "floor": False, "failed": None}
    args.update(kw)
    return sc.score_lift_place(_place_series() if lift is None else lift, DT,
                               final=_final() if final is None else final, **args)


def test_lift_place_rule_numbers_are_the_predeclared_ones():
    r = sc.LIFT_PLACE_RULE
    assert r["protocol"] == "lift_place" and r["episode_s"] == 20.0
    assert (r["lift_peak_m"], r["lift_hold_m"], r["lift_hold_s"]) == (0.10, 0.05, 3.0)
    assert (r["final_height_tol_m"], r["final_speed_mps"], r["plinth_half_m"]) == (0.03, 0.05, 0.09)
    assert r["released"] is True and r["tilt_rad"] == 0.78 and r["floor_contact"] is False
    assert r["seeds"] == list(range(10, 20)) and r["pass_min"] == 8
    # never scored on the seeds of the other protocols; footprint = the harness's plinth
    assert not set(r["seeds"]) & set(sc.SUCCESS_RULE["seeds"]) and not set(r["seeds"]) & set(sc.LIFT_HOLD_RULE["seeds"])
    assert r["plinth_half_m"] == sc.CarryParams().plinth_half
    assert "lift_place" in sc.PROTOCOLS


def test_good_lift_place_succeeds():
    row = _score_lp()
    assert row["success"] and row["first_failed_check"] is None
    assert row["lift_hold_s"] == pytest.approx(4.0, abs=DT) and row["hold_start_s"] == pytest.approx(6.2, abs=DT)


@pytest.mark.parametrize("kw,check", [
    ({"lift": _place_series(peak=0.09)}, "lift_peak"),
    ({"lift": _place_series(hold_s=2.96)}, "lift_hold"),
    ({"final": _final(dz=-0.031)}, "on_plinth_height"),
    ({"final": _final(dz=-0.19)}, "on_plinth_height"),      # on the floor
    ({"final": _final(dz=None)}, "on_plinth_height"),
    ({"final": _final(speed=0.05)}, "at_rest"),
    ({"final": _final(off=(0.091, 0.0))}, "on_plinth_footprint"),
    ({"final": _final(off=(0.0, -0.091))}, "on_plinth_footprint"),
    ({"final": _final(contact=True)}, "released"),
    ({"max_tilt": 0.79}, "no_fall"), ({"floor": True}, "no_floor_contact"),
    ({"failed": "nonfinite_state"}, "finite"),
    ({"lift": _place_series()[:-10]}, "full_episode"),
])
def test_each_lift_place_check_fails_on_its_own(kw, check):
    row = _score_lp(**kw)
    assert not row["success"] and not row["checks"][check]
    assert sum(not v for v in row["checks"].values()) == 1


def test_lift_place_edges_pass():
    assert _score_lp(lift=_place_series(hold_s=3.0))["success"]
    assert _score_lp(final=_final(dz=0.03, speed=0.049, off=(0.09, -0.09)))["success"]


def _lp_episodes(wins, n_pairs=1, seeds=None, holds=None):
    seeds = list(range(10, 10 + len(wins))) if seeds is None else seeds
    eps = []
    for i, (s, w) in enumerate(zip(seeds, wins)):
        row = _score_lp() if w else _score_lp(final=_final(dz=-0.19))
        if holds:
            row["lift_hold_s"] = holds[i]
        eps.append({"seed": s, "pairs": [dict(row, pair=k) for k in range(n_pairs)]})
    return eps


def _lp_payload(wins, scored=True, rule=None, n_pairs=1, seeds=None):
    return {"protocol": "lift_place", "rule": dict(rule or sc.LIFT_PLACE_RULE), "scored_run": scored,
            "crew": 2 * n_pairs, "pairs": n_pairs, "label": sc.LABEL,
            "episodes": _lp_episodes(wins, n_pairs=n_pairs, seeds=seeds)}


def test_lift_place_summary_gate_is_eight_of_ten_on_seeds_10_to_19():
    s = sc.summarize_lift_place(_lp_episodes([True] * 8 + [False] * 2), 1)
    assert s["complete"] and s["pass"] and s["per_pair"][0]["placed_on_plinth"] == 8
    assert not sc.summarize_lift_place(_lp_episodes([True] * 7 + [False] * 3), 1)["pass"]
    # seeds 0-9 or >= 100 never count toward the verdict
    s = sc.summarize_lift_place(_lp_episodes([True] * 10, seeds=list(range(10))), 1)
    assert not s["complete"] and not s["pass"] and s["per_pair"][0]["episodes"] == 0


def test_lift_place_verdict_line_is_computed_from_the_json():
    line = sc.lift_place_verdict_line(_lp_payload([True] * 8 + [False] * 2))
    assert line.startswith("COOP-LIFT-PLACE crew 2: PASS |") and "pair0=8/10" in line
    assert sc.LIFT_PLACE_NOTE in line and sc.LABEL in line
    assert sc.lift_place_verdict_line(_lp_payload([True] * 7 + [False] * 3)).startswith(
        "COOP-LIFT-PLACE crew 2: NEGATIVE |")
    assert sc.lift_place_verdict_line(_lp_payload([True] * 10, scored=False)).startswith(
        "COOP-LIFT-PLACE crew 2: INCOMPLETE |")
    assert sc.lift_place_verdict_line(_lp_payload([True] * 9)).startswith("COOP-LIFT-PLACE crew 2: INCOMPLETE |")
    for relaxed in (dict(sc.LIFT_PLACE_RULE, lift_hold_s=2.0), dict(sc.LIFT_PLACE_RULE, released=False),
                    dict(sc.LIFT_PLACE_RULE, seeds=list(range(10)))):
        assert sc.lift_place_verdict_line(_lp_payload([True] * 10, rule=relaxed)).startswith(
            "COOP-LIFT-PLACE crew 2: INVALID")
    assert "INVALID" in sc.lift_place_verdict_line(dict(_lp_payload([True] * 10), protocol="lift_hold"))
    p4 = _lp_payload([True] * 10, n_pairs=2)
    for e in p4["episodes"][:3]:
        e["pairs"][1]["success"] = False
        e["pairs"][1]["first_failed_check"] = "released"
    assert sc.lift_place_verdict_line(p4).startswith("COOP-LIFT-PLACE crew 4: NEGATIVE |")


def test_place_script_matches_the_arm_script_through_the_lift():
    p, q = sc.CarryParams(), sc.PlaceParams()
    arm, place = sc.ArmScript(p), sc.PlaceScript(p, q)
    for t in np.arange(0, place.t_lift_done, 0.04):
        assert np.allclose(arm(t), place(t)), t
        assert arm.phase(t) == place.phase(t)
    assert place.t_lift_done == pytest.approx(arm.t_lift_done)


def test_place_script_is_continuous_and_ends_at_rest():
    p = sc.CarryParams()
    for q in (sc.PlaceParams(), sc.PlaceParams(lower_to=None),
              sc.PlaceParams(t_via=0.5, lower_via=(-0.5, 1.1, 0, 0, 0), t_unload=0.5,
                             release_via=(0, 1.1, 0, 1.0, 0))):
        s = sc.PlaceScript(p, q)
        ts = np.arange(0, 20, 0.01)
        qs = np.array([s(t) for t in ts])
        assert np.abs(np.diff(qs, axis=0)).max() < 0.05
        assert np.allclose(s(19.9), sc.KEYFRAMES_LEFT["rest"]) and s.phase(19.9) == "stand"
        assert np.allclose(s(s.t_lower_start - 1e-6), sc.KEYFRAMES_LEFT["lift"])
        assert np.allclose(s(s.t_end["lower"] - 1e-6), s.lower_target, atol=1e-4)
        assert s.open_target[1] == sc.KEYFRAMES_LEFT["reach"][1]
    rev = sc.PlaceScript(p, sc.PlaceParams(lower_to=None))
    assert np.allclose(rev.lower_target, sc.KEYFRAMES_LEFT["squeeze"])          # pure reverse keyframe
    assert np.allclose(rev.open_target, sc.KEYFRAMES_LEFT["reach"])


def test_frozen_place_params_and_lowering_window():
    p, q = sc.CarryParams(), sc.PlaceParams()
    s = sc.PlaceScript(p, q)
    # lowering 9.2 -> 10.7 s: after the arm settles in the lift pose, done before the earliest
    # lift_hold grip slip seen on exploration seeds 100-119 (11.36 s)
    assert s.t_lower_start == pytest.approx(9.2) and s.t_end["lower"] == pytest.approx(10.7)
    assert s.t_end["lower"] < 11.36
    assert tuple(q.lower_to) == (0.6, 1.1, 0.0, 1.0, 0.0) and q.lower_via is None and q.release_via is None
    assert q.station_keep is False


def test_station_command_points_home_and_is_capped():
    q = sc.PlaceParams(station_keep=True)
    c = sc.station_command([0.1, 0.0], [0.0, 0.0], 0.0, q)
    assert c[0] < 0 and abs(c[1]) < 1e-12 and c[2] == 0.0
    c = sc.station_command([0.0, 1.0], [0.0, 0.0], np.pi / 2, q)      # facing +y: -y world = backward
    assert c[0] == pytest.approx(-q.v_station_max) and abs(c[1]) < 1e-9
    q2 = sc.PlaceParams(station_keep=True, v_station_min=0.1)
    assert not sc.station_command([0.01, 0.0], [0.0, 0.0], 0.0, q2).any()


def test_rules_and_protocols_do_not_cross():
    with pytest.raises(ValueError):
        sc.run_episode(None, [], [], None, None, 0, sc.CarryParams(), rule=sc.LIFT_PLACE_RULE)
    with pytest.raises(ValueError):
        sc.run_episode(None, [], [], None, None, 0, sc.CarryParams(), rule=sc.LIFT_PLACE_RULE,
                       protocol="lift_hold")
    with pytest.raises(ValueError):
        sc.run_place_episode(None, [], [], None, None, 0, sc.CarryParams(), rule=sc.LIFT_HOLD_RULE)
    with pytest.raises(ValueError):
        sc.run_episode(None, [], [], None, None, 0, sc.CarryParams(), rule=sc.LIFT_HOLD_RULE,
                       protocol="lift_place")


@pytest.mark.parametrize("seeds", ["0-9", "9", "5,100", "10-19,3"])
def test_lift_place_score_never_runs_seeds_0_to_9(tmp_path, seeds):
    import argparse
    mod = _script()
    args = argparse.Namespace(protocol="lift_place", out=tmp_path / "s.json", seeds=seeds, seconds=None, crew=2)
    with pytest.raises(SystemExit, match="seeds 0-9"):
        mod.score(args)
    assert not (tmp_path / "s.json").exists()


def test_lift_place_score_refuses_to_overwrite(tmp_path):
    import argparse
    mod = _script()
    out = tmp_path / "score_liftplace_crew2.json"
    out.write_text("{}")
    args = argparse.Namespace(protocol="lift_place", out=out, seeds="10-19", seconds=None, crew=2)
    with pytest.raises(SystemExit, match="REFUSED"):
        mod.score(args)
    assert out.read_text() == "{}"


def test_lift_place_render_refuses_other_json_and_skips_a_negative(tmp_path, capsys):
    import argparse
    import json
    mod = _script()
    hold = tmp_path / "score_lifthold_crew2.json"
    hold.write_text(json.dumps({"crew": 2, "protocol": "lift_hold", "scored_run": True,
                                "summary": {"verdict": "PASS", "median_seed": 0}}))
    args = argparse.Namespace(protocol="lift_place", render_from=hold, pipeline_check_seed=None,
                              no_render=True, crew=2, out_dir=tmp_path, gif=None, seconds=None)
    with pytest.raises(SystemExit, match="protocol"):
        mod.render(args)
    neg = tmp_path / "score_liftplace_crew2.json"
    neg.write_text(json.dumps(_lp_payload([True] * 7 + [False] * 3)))
    args.render_from = neg
    assert mod.render(args) == 0
    assert "COOP_LIFT_PLACE_RENDER=SKIPPED verdict=NEGATIVE" in capsys.readouterr().out
    # a JSON that CLAIMS PASS in its summary is still judged from its episodes
    lie = tmp_path / "score_liftplace_lie.json"
    lie.write_text(json.dumps(dict(_lp_payload([False] * 10), summary={"verdict": "PASS", "median_seed": 10})))
    args.render_from = lie
    assert mod.render(args) == 0
    assert "SKIPPED verdict=NEGATIVE" in capsys.readouterr().out
    args.pipeline_check_seed = 12
    args.render_from = neg
    with pytest.raises(SystemExit, match="scored seed"):
        mod.render(args)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "score_lifthold_crew2.json", "score_liftplace_crew2.json", "score_liftplace_lie.json"]


def test_lowering_target_description_matches_the_numbers():
    # the frozen target reverses the shoulder pitch only; roll AND elbow are the lift keyframe's
    d = sc.describe_lower_target(sc.PlaceScript(sc.CarryParams(), sc.PlaceParams()).lower_target)
    assert "pitch-only lowering" in d and "tuned target" in d
    assert "shoulder pitch -0.5 -> 0.6" in d
    assert "shoulder roll 1.1 held at the lift keyframe's value" in d
    assert "elbow pitch 1 held at the lift keyframe's value" in d
    assert "squeeze keyframe's elbow" not in d
    assert sc.KEYFRAMES_LEFT["lift"][3] == 1.0 and sc.KEYFRAMES_LEFT["squeeze"][3] == 0.5
    assert "pure reverse" in sc.describe_lower_target(sc.KEYFRAMES_LEFT["squeeze"])
    # labels never describe the frozen lowering as a plain keyframe reverse
    assert "reverse-keyframe" not in sc.LIFT_PLACE_LABEL_DETAIL
    assert "pitch-only lowering" in sc.LIFT_PLACE_LABEL_DETAIL
    src = (REPO / "scripts/bench/coop_scripted_carry.py").read_text()
    assert "squeeze keyframe's elbow" not in src and "sc.describe_lower_target(" in src


@pytest.mark.parametrize("seeds,seconds", [("10-14", None), ("15", None), ("10-19,100", None),
                                           ("10-19", 3.0), ("100,110", None)])
def test_lift_place_score_refuses_partial_use_of_reserved_seeds(tmp_path, seeds, seconds):
    import argparse
    mod = _script()
    args = argparse.Namespace(protocol="lift_place", out=tmp_path / "s.json", seeds=seeds, seconds=seconds, crew=2)
    if seeds == "100,110":
        # no reserved seed: allowed past the gate (stop at model load)
        mod.load = lambda a: (_ for _ in ()).throw(RuntimeError("stop"))
        with pytest.raises(RuntimeError, match="stop"):
            mod.score(args)
    else:
        with pytest.raises(SystemExit, match="reserved"):
            mod.score(args)
    assert not (tmp_path / "s.json").exists()


def test_lift_place_render_seed_is_recomputed_from_the_episodes(tmp_path):
    import argparse
    import json
    mod = _script()
    wins = [True] * 10
    pay = _lp_payload(wins)
    truth = sc.median_seed_by_hold(pay["episodes"])
    pay["summary"] = {"verdict": "PASS", "median_seed": 19 if truth != 19 else 10}   # tampered
    f = tmp_path / "score_liftplace_crew2.json"
    f.write_text(json.dumps(pay))
    args = argparse.Namespace(protocol="lift_place", render_from=f, pipeline_check_seed=None,
                              no_render=True, crew=2, out_dir=tmp_path, gif=None, seconds=None)
    mod.load = lambda a: (_ for _ in ()).throw(RuntimeError("stop"))
    with pytest.raises(RuntimeError, match="stop"):
        mod.render(args)
    assert args.seed_used == truth
