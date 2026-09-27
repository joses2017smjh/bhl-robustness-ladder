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
    assert sc.PROTOCOLS == ("carry", "lift_hold")


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
