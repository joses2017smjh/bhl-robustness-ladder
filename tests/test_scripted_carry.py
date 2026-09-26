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
