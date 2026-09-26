"""Tests for scripts/bench/ice_depth_freeze.py (the ice depth-freeze test).

Everything here runs without OpenGL: the analytic depth, the pooling order,
the loader parity, the predeclared verdict rules, and a short closed-loop
episode of the no-render arms with `mujoco.Renderer` booby-trapped.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

_spec = importlib.util.spec_from_file_location(
    "ice_depth_freeze", _REPO / "scripts" / "bench" / "ice_depth_freeze.py")
idf = importlib.util.module_from_spec(_spec)
sys.modules["ice_depth_freeze"] = idf      # dataclasses resolve their module by name
_spec.loader.exec_module(idf)

_RUNS = _REPO / "external" / "Berkeley-Humanoid-Lite" / "logs" / "rsl_rl" / "biped"
_DEPTH_S0 = _RUNS / "2026-09-16_14-41-36_ppo-ice-placed-depth-s0"
_BLIND_S0 = _RUNS / "2026-09-16_07-46-31_ppo-ice-placed-blind-s0"

# MuJoCo pose of the ego camera from mjcf_assets.EGO_CAM (xyaxes as columns).
_EGO_X = np.array([0.0, -1.0, 0.0])
_EGO_Y = np.array([0.342, 0.0, 0.940])
_EGO_Y = _EGO_Y / np.linalg.norm(_EGO_Y)
_EGO_Z = np.cross(_EGO_X, _EGO_Y)
_EGO_R = np.stack([_EGO_X, _EGO_Y, _EGO_Z], axis=1)


# --------------------------------------------------------------- analytic depth

def test_camera_straight_down_sees_its_height_everywhere():
    h = 0.42
    raw = idf.plane_depth_raw([0, 0, h], np.eye(3), 60.5)
    assert np.allclose(raw, h)
    obs = idf.pool_scale(raw)
    assert obs.shape == (256,)
    assert np.allclose(obs, h / 6.0, atol=1e-6)


def test_optical_axis_depth_matches_hand_computation():
    # res=1: the single pixel is the optical axis, pitched 20 deg down.
    raw = idf.plane_depth_raw([0.12, 0, 0.30], _EGO_R, 60.5, res=1)
    pitch = math.atan2(0.940, 0.342)          # angle of camera +y from horizontal
    down = math.pi / 2 - pitch                 # axis depression, ~20 deg
    assert raw[0, 0] == pytest.approx(0.30 / math.sin(down), rel=1e-6)


def test_every_hit_lies_on_the_floor_and_top_rows_are_sky():
    p = np.array([0.12, 0.0, 0.30])
    raw = idf.plane_depth_raw(p, _EGO_R, 60.5, no_hit=200.0)
    f = 32 / math.tan(math.radians(60.5) / 2)
    for r, c in [(63, 0), (63, 63), (40, 31), (45, 10)]:
        d_cam = np.array([(c + 0.5 - 32) / f, (32 - r - 0.5) / f, -1.0])
        pt = p + raw[r, c] * (_EGO_R @ d_cam)
        assert abs(pt[2]) < 1e-9
    # 20 deg down, 30.25 deg half-FOV: the top ~10 deg of the image is sky.
    assert np.all(raw[0] == 200.0)
    assert np.all(raw[-1] < 1.0)
    # depth decreases monotonically down the centre column once on the floor
    col = raw[:, 32]
    on = col < 200.0
    assert np.all(np.diff(col[on]) < 0)


def test_lower_camera_gives_nearer_floor():
    hi = idf.analytic_depth_obs([0.12, 0, 0.30], _EGO_R, 60.5)
    lo = idf.analytic_depth_obs([0.12, 0, 0.25], _EGO_R, 60.5)
    floor = hi < 1.0
    assert np.all(lo[floor] <= hi[floor]) and np.any(lo[floor] < hi[floor])


def test_pool_order_multirunner_vs_isaac():
    raw = np.full((64, 64), 5.0)
    raw[:2, :] = 200.0                         # half of each top pooled cell is sky
    mr = idf.pool_scale(raw, convention="multirunner").reshape(16, 16)
    isc = idf.pool_scale(raw, convention="isaac").reshape(16, 16)
    assert np.allclose(mr[0], 1.0)             # mean(5, 200) / 6 clipped
    assert np.allclose(isc[0], (5.0 + 6.0) / 2 / 6.0)
    assert np.allclose(mr[1:], 5.0 / 6.0) and np.allclose(isc[1:], 5.0 / 6.0)


def test_pool_scale_is_multirunner_depth_obs_arithmetic():
    """Feed the same raw image through MultiRunner._depth_obs with a fake renderer."""
    mujoco = pytest.importorskip("mujoco")
    from bhl_robust.eval.mjcf_assets import EGO_CAM_NAME
    from bhl_robust.eval.multi_robot import MultiRunner

    m = mujoco.MjModel.from_xml_string(
        f'<mujoco><worldbody><camera name="r0_{EGO_CAM_NAME}"/></worldbody></mujoco>')
    raw = np.random.default_rng(3).uniform(0.05, 250.0, (64, 64)).astype(np.float32)
    raw[5, 5] = np.nan
    raw[6, 6] = np.inf

    class FakeRenderer:
        def update_scene(self, d, camera):
            pass

        def render(self):
            return raw

    fake = types.SimpleNamespace(
        _depth_dim={0: 16}, _depth_pool={0: 4}, m=m, d=None,
        slots=[types.SimpleNamespace(prefix="r0_")], _depth_r=FakeRenderer(),
        DEPTH_CLIP=MultiRunner.DEPTH_CLIP)
    want = MultiRunner._depth_obs(fake, 0)
    got = idf.pool_scale(raw)
    assert np.allclose(got, want, atol=1e-6)


# --------------------------------------------------------------- policy loading

@pytest.mark.skipif(not (_DEPTH_S0 / "exported" / "policy.onnx").is_file(),
                    reason="exported depth ONNX not present")
def test_checkpoint_mlp_matches_exported_onnx():
    pytest.importorskip("onnxruntime")
    for run, width in ((_DEPTH_S0, 301), (_BLIND_S0, 45)):
        pol = idf.MlpPolicy(run / idf.CKPT_NAME)
        assert pol.in_dim == width and pol.out_dim == 12
        assert idf.onnx_parity(pol, run / "exported" / "policy.onnx") < idf.ONNX_PARITY_TOL


def test_parse_run():
    assert idf.parse_run("2026-09-24_16-48-43_ppo-ice-control-depth-s0") == ("control", "depth", 0)
    assert idf.parse_run("2026-09-16_07-49-05_ppo-ice-placed-blind-s1") == ("placed", "blind", 1)
    assert idf.parse_run("2026-09-16_23-52-13_ppo-ice-placed-visible-s0") is None


# --------------------------------------------------------------- verdict rules

def S(fall, disp, n=60):
    return {"n": n, "fall_rate": fall, "mean_disp_m": disp, "mean_lin_vel_err": 0.1,
            "mean_yaw_rate_err": 0.1}


def test_rules_every_branch():
    walk = S(0.0, 2.0)
    assert idf.classify(None, S(0, 2), walk)["reason"].startswith("arm A")
    assert idf.classify(S(0, 2), None, walk)["reason"].startswith("arm B")
    v = idf.classify(S(0.6, 1.0), S(0.0, 2.0), walk)
    assert v["outcome"] == "INCONCLUSIVE" and v["gate"] == "FAIL" and "sim2sim" in v["reason"]
    v = idf.classify(S(0.6, 1.0), S(0.0, 2.0), S(0.5, 1.0))
    assert v["outcome"] == "INCONCLUSIVE" and "harness" in v["reason"]
    v = idf.classify(S(0.6, 1.0), S(0.0, 2.0), None)
    assert v["outcome"] == "INCONCLUSIVE" and "missing" in v["reason"]
    # gate boundary: exactly 0.50 passes
    assert idf.classify(S(0.5, 2.0), S(0.5, 2.0), walk)["gate"] == "PASS"
    # (i) by fall delta (boundary 0.30 counts) and by displacement ratio
    assert idf.classify(S(0.1, 2.0), S(0.4, 2.0), walk)["outcome"] == "(i)"
    assert idf.classify(S(0.0, 2.0), S(0.0, 1.0), walk)["outcome"] == "(i)"
    # (ii) inside both tolerances (boundaries count)
    assert idf.classify(S(0.1, 2.0), S(0.2, 1.6), walk)["outcome"] == "(ii)"
    assert idf.classify(S(0.1, 2.0), S(0.0, 2.4), walk)["outcome"] == "(ii)"
    # (iii) between the bands
    assert idf.classify(S(0.0, 2.0), S(0.2, 2.0), walk)["outcome"] == "(iii)"
    assert idf.classify(S(0.0, 2.0), S(0.0, 1.5), walk)["outcome"] == "(iii)"
    # B better than A is not "live sensor"
    assert idf.classify(S(0.4, 1.0), S(0.0, 2.0), walk)["outcome"] == "(iii)"


def test_c_secondary_labels():
    walk = S(0.0, 2.0)
    assert idf.classify(S(0, 2), S(1, 0.3), walk, C=S(0.05, 1.9))["C_vs_A"].startswith("height/tilt")
    assert idf.classify(S(0, 2), S(1, 0.3), walk, C=S(0.5, 1.0))["C_vs_A"].startswith("MuJoCo render")
    # C is reported even when A fails the gate
    v = idf.classify(S(0.9, 0.2), S(1, 0.3), walk, C=S(0.9, 0.2))
    assert v["outcome"] == "INCONCLUSIVE" and "C_vs_A" in v


def _part(ring, kind, seed, arms, error=None):
    eps = {}
    for a, (n_fell, disp) in arms.items():
        eps[a] = [{"command": [0.3, 0, 0], "seed": i, "fell": i < n_fell, "distance_m": disp,
                   "lin_vel_err": 0.1, "yaw_rate_err": 0.1, "survival_s": 12.0}
                  for i in range(10)]
    p = {"run": f"x_ppo-ice-{ring}-{kind}-s{seed}", "ring": ring, "kind": kind,
         "train_seed": seed,
         "arms": {a: {"summary": idf.summarise_arm(e), "episodes": e} for a, e in eps.items()}}
    if error:
        p["error"] = error
    return p


def test_verdict_pairs_blind_by_ring_and_seed_and_pools_gate_passers_only():
    parts = [
        _part("placed", "depth", 0, {"A": (0, 2.0), "B": (8, 0.5), "C": (0, 2.0)}),
        _part("placed", "blind", 0, {"blind": (0, 2.0)}),
        _part("control", "depth", 1, {"A": (9, 0.2), "B": (9, 0.2)}),   # fails gate
        _part("control", "blind", 1, {"blind": (0, 2.0)}),
    ]
    v = idf.build_verdict(parts)
    rows = {r["run"]: r for r in v["per_checkpoint"]}
    r0 = rows["x_ppo-ice-placed-depth-s0"]
    assert r0["outcome"] == "(i)" and r0["blind_run"] == "x_ppo-ice-placed-blind-s0"
    assert r0["C_vs_A"].startswith("height/tilt")
    r1 = rows["x_ppo-ice-control-depth-s1"]
    assert r1["gate"] == "FAIL" and "sim2sim" in r1["reason"]
    assert v["pooled"]["n_checkpoints"] == 1
    assert v["pooled"]["arms"]["A"]["n"] == 10          # the failing checkpoint is excluded
    assert v["pooled"]["outcome"] == "(i)"


def test_verdict_without_arm_a_is_inconclusive_with_cpu_preview():
    parts = [_part("placed", "depth", 0, {"B": (5, 0.5), "C": (0, 2.0)}),
             _part("placed", "blind", 0, {"blind": (0, 2.0)})]
    v = idf.build_verdict(parts)
    assert v["per_checkpoint"][0]["outcome"] == "INCONCLUSIVE"
    assert v["pooled"]["outcome"] == "INCONCLUSIVE"
    assert v["cpu_preview_B_vs_C"] and "NOT the predeclared" in v["cpu_preview_B_vs_C"][0]["note"]


def test_parity_error_part_is_inconclusive():
    parts = [_part("placed", "depth", 0, {}, error="ONNX parity 1e-1 > 0.001")]
    v = idf.build_verdict(parts)
    assert v["per_checkpoint"][0]["outcome"] == "INCONCLUSIVE"


# --------------------------------------------------------------- closed loop, no GL

@pytest.mark.skipif(not (_DEPTH_S0 / idf.CKPT_NAME).is_file(), reason="checkpoint not present")
def test_no_render_arms_never_build_a_renderer(tmp_path, monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("berkeley_humanoid_lite_lowlevel")
    from bhl_robust.eval.harness import EvalConfig
    from bhl_robust.eval.multi_robot import MultiRunner

    def boom(*a, **k):
        raise AssertionError("arm B/C/blind must not construct a renderer")

    monkeypatch.setattr(mujoco, "Renderer", boom)
    sim = idf.Sim(tmp_path / "cache")
    pol = idf.MlpPolicy(_DEPTH_S0 / idf.CKPT_NAME)
    cfg = idf.make_cfg(idf.DEFAULT_TEMPLATE, _DEPTH_S0 / idf.CKPT_NAME, pol.in_dim)
    ecfg = EvalConfig(episode_s=1.2, seeds=(0,))
    seen = {}
    for arm in ("B", "Bsettle", "C"):
        ctrl = idf.make_controller(cfg, pol)
        runner = MultiRunner(sim.model, sim.slots, [cfg], [ctrl])
        # record what the policy is fed
        fed = []
        orig = ctrl.update

        def spy(obs, depth=None, _orig=orig, _fed=fed):
            _fed.append(None if depth is None else np.array(depth))
            return _orig(obs, depth)

        ctrl.update = spy
        e = idf.run_episode_arm(sim, runner, ctrl, arm, (0.3, 0.0, 0.0), 0, ecfg, "analytic")
        assert e.survival_s > 0
        assert all(f is not None and f.shape == (256,) for f in fed)
        seen[arm] = fed
    # B: one constant image equal to the analytic t=0 nominal
    B = seen["B"]
    assert all(np.array_equal(B[0], f) for f in B)
    assert np.allclose(B[0], sim.analytic(sim.nominal_state()))
    # C moves with the robot
    C = seen["C"]
    assert max(np.abs(f - C[0]).max() for f in C) > 1e-3
    # Bsettle: live until the settle step (t=25), constant afterwards
    Bs = seen["Bsettle"]
    settle = int(ecfg.settle_s / cfg.policy_dt)
    if len(Bs) > settle + 1:
        assert all(np.array_equal(Bs[settle], f) for f in Bs[settle:])


def test_parts_of_one_run_are_merged_and_duplicates_refused():
    b = _part("placed", "depth", 0, {"B": (9, 0.3)})
    c = _part("placed", "depth", 0, {"C": (0, 2.0)})
    v = idf.build_verdict([b, c, _part("placed", "blind", 0, {"blind": (0, 2.0)})])
    assert len(v["per_checkpoint"]) == 1
    assert set(v["per_checkpoint"][0]["arms"]) == {"B", "C"}
    with pytest.raises(ValueError):
        idf.build_verdict([b, _part("placed", "depth", 0, {"B": (0, 2.0)})])
