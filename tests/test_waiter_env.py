"""Waiter phase 2: the environment contract (docs/WAITER_INTERFACES.md) and the layout.

The layout tests are pure Python. The env tests build the MuJoCo scene with the selected phase 1c WBC (skipped when
the upstream checkout, the deploy file or MuJoCo is missing) and check shapes, clipping, info keys, determinism.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from bhl_robust.eval import waiter_scene as W

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "results/waiter-20261005/wbc1c/training/waiter-wbc1c-s7.deploy.yaml"
UPSTREAM = REPO / "external/Berkeley-Humanoid-Lite"


def test_layout_is_a_pure_function_of_the_seed():
    for s in (1000, 1234, 3999):
        assert W.sample_layout(s).as_dict() == W.sample_layout(s).as_dict()
    assert W.sample_layout(1000).as_dict() != W.sample_layout(1001).as_dict()


def test_layout_ranges():
    for s in range(1000, 1200):
        lay = W.sample_layout(s)
        assert lay.target_colour in W.COLOURS and lay.target_side in W.SIDES
        assert set(lay.tumblers) == set(W.COLOURS)
        ys = sorted(v[1] for v in lay.tumblers.values())
        assert ys[0] < 0 < ys[1]                                   # one tumbler on each side of the tray
        for x, y, yaw in lay.tumblers.values():
            slot = min(W.CUP_SLOTS, key=lambda c: abs(c["y"] - y))
            assert abs(x - slot["x"]) <= 0.01 + 1e-9 and abs(y - slot["y"]) <= 0.015 + 1e-9
            assert abs(math.degrees(yaw)) <= 6.0 + 1e-9
        tx, ty, tyaw = lay.tray
        assert abs(tx - W.TRAY_NOMINAL[0]) <= 0.008 + 1e-9 and abs(ty) <= 0.008 + 1e-9
        assert abs(math.degrees(tyaw)) <= 2.0 + 1e-9


def _boxes_apart(c1, h1, c2, h2, margin):
    """Axis-aligned 2-D boxes (centre, half sizes) separated by >= margin along x or y."""
    return (abs(c1[0] - c2[0]) - h1[0] - h2[0] >= margin) or (abs(c1[1] - c2[1]) - h1[1] - h2[1] >= margin)


def test_tumblers_clear_the_tray_pegs():
    """Footprints (2-D, world, yaw jitter <= 6 deg folded into a 6 mm pad): each tumbler (body box and handle posts)
    stays >= 5 mm from each tray peg's cap, over the layout's jitter range."""
    pad = 0.006
    for s in range(1000, 1500):
        lay = W.sample_layout(s)
        tx, ty, tyaw = lay.tray
        pegs = [(np.array([tx + W.EAR_DX + W.CAP_DX, ty + yy]), np.array(W.CAP_HALF[:2]) + pad)
                for yy in (W.ear_center_y(), -W.ear_center_y())]
        for x, y, yaw in lay.tumblers.values():
            parts = [(np.array([x, y]), np.array([W.CUP_R, W.CUP_R]))] + [
                (np.array([x, y + py]), np.array([W.HANDLE_T / 2, W.HANDLE_T / 2]) + pad)
                for py in (W.BAIL_HALF_SPAN, -W.BAIL_HALF_SPAN)]
            for pc, ph in pegs:
                for c, h in parts:
                    assert _boxes_apart(c, h, pc, ph, 0.005), (s, c, pc)


def test_carry_path_clears_the_pegs():
    """The tumbler is carried sideways at its pick x onto the tray: the carry band stays >= 1 cm behind the pegs."""
    for s in range(1000, 1500):
        lay = W.sample_layout(s)
        back = lay.tray[0] + W.EAR_DX + W.CAP_DX - W.CAP_HALF[0] - W.ear_center_y() * abs(math.sin(lay.tray[2]))
        for x, y, yaw in lay.tumblers.values():
            assert x + W.CUP_R + 0.01 < back, (s, x, lay.tray)


def test_templates_format_and_ids():
    from bhl_robust.eval.waiter_env import SLOTS, TEMPLATES, instruction_text
    assert len(TEMPLATES) == 6 and len(set(TEMPLATES.values())) == 6
    for t in TEMPLATES.values():
        for c in SLOTS["colour"]:
            for side in SLOTS["side"]:
                txt = t.format(colour=c, side=side)
                assert c in txt and side in txt
    lay = W.sample_layout(1000)
    txt, tid = instruction_text(lay)
    assert tid in TEMPLATES and lay.target_colour in txt and lay.target_side in txt


def _env(**kw):
    pytest.importorskip("mujoco")
    pytest.importorskip("onnxruntime")
    if not (DEPLOY.exists() and UPSTREAM.exists()):
        pytest.skip("selected WBC deploy or the upstream checkout not present")
    from omegaconf import OmegaConf
    if not Path(str(OmegaConf.load(DEPLOY).policy_checkpoint_path)).exists():
        pytest.skip("WBC policy file not present")
    from bhl_robust.eval.waiter_env import WaiterEnv
    return WaiterEnv(DEPLOY, UPSTREAM, Path(os.environ.get("TMPDIR", "/tmp")) / "waiter-env-test", render=False, **kw)


@pytest.fixture(scope="module")
def env():
    return _env()


def test_env_contract_shapes_and_info(env):
    from bhl_robust.eval.waiter_env import ACTION_DIM, STATE_DIM, IMAGE_HW
    obs = env.reset(1000)
    assert set(obs) == {"image", "state", "instruction"}
    assert obs["image"].shape == (IMAGE_HW, IMAGE_HW, 3) and obs["image"].dtype == np.uint8
    assert obs["state"].shape == (STATE_DIM,) and obs["state"].dtype == np.float32
    assert isinstance(obs["instruction"], str) and env.layout.target_colour in obs["instruction"]
    obs, r, done, info = env.step(np.zeros(ACTION_DIM, np.float32))
    assert set(info) >= {"success", "fell", "stage", "tumbler_on_tray", "tray_on_target", "t"}
    assert r == 0.0 and not done and abs(info["t"] - 0.08) < 1e-9
    o = env.oracle()
    for k in ("base_pos", "base_yaw", "tray", "tumbler_red", "tumbler_blue", "zone_left", "zone_right"):
        assert k in o


def test_env_clips_actions(env):
    env.reset(1001)
    a = np.zeros(15, np.float32)
    a[:3] = (1.0, -1.0, 2.0)
    a[3:13] = 9.0
    a[13:15] = (2.0, -1.0)
    env.step(a)
    app = env.applied_action()
    assert np.allclose(app[:3], (0.4, -0.3, 0.6))
    hi = np.r_[env.ids.arm_range["left"][:, 1], env.ids.arm_range["right"][:, 1]]
    assert np.allclose(app[3:13], hi)
    assert np.allclose(app[13:15], (1.0, 0.0))


def test_env_reset_is_deterministic(env):
    def run():
        env.reset(1002)
        for _ in range(5):
            env.step(np.zeros(15, np.float32))
        return env.d.qpos.copy()
    assert np.allclose(run(), run())


def test_env_settles_near_the_station(env):
    """The robot is spawned ahead of the origin and settles near it (stance hold); the pick area is placed after."""
    for s in (1000, 1003):
        env.reset(s)
        o = env.oracle()
        assert abs(o["base_pos"][0]) < 0.05 and abs(o["base_pos"][1]) < 0.05
        assert abs(math.degrees(o["base_yaw"])) < 8.0 and o["tilt"] < 0.3
        assert o["clock_held"]
        tp, _ = o["tray"]
        assert abs(tp[0] - env.layout.tray[0]) < 1e-6


def test_stance_lock_holds_releases_and_reengages(env):
    """Stance lock (2026-10-09, a declared simulation fixture): the episode starts with the base held at the station;
    a non-zero base command frees it at once (the WBC walks); standing still again re-engages it where it stands."""
    from bhl_robust.eval.waiter_env import LOCK_DELAY_S, STEP_DT
    env.reset(1004)
    o = env.oracle()
    assert o["base_locked"] and o["clock_held"]
    assert abs(o["base_pos"][0]) < 1e-3 and abs(o["base_pos"][1]) < 1e-3 and abs(o["base_yaw"]) < 1e-3
    a = np.zeros(15, np.float32)
    for _ in range(10):
        _, _, _, info = env.step(a)
    assert info["base_locked"] and np.allclose(env.oracle()["base_pos"][:2], o["base_pos"][:2], atol=2e-3)
    a[0] = 0.3
    _, _, _, info = env.step(a)
    assert info["base_locked"] and env.oracle()["clock_held"]          # releasing: the weld softens first
    from bhl_robust.eval.waiter_env import RELEASE_S
    for _ in range(int(round(RELEASE_S / STEP_DT))):
        _, _, _, info = env.step(a)
    assert not info["base_locked"]
    for _ in range(25):
        env.step(a)
    moved = env.oracle()["base_pos"][:2]
    assert np.linalg.norm(moved - o["base_pos"][:2]) > 0.05
    a[0] = 0.0
    for _ in range(int(round((0.8 + LOCK_DELAY_S + 1.0) / STEP_DT))):
        _, _, _, info = env.step(a)
    assert info["base_locked"]
    p_locked = env.oracle()["base_pos"][:2].copy()
    for _ in range(10):
        env.step(a)
    assert np.allclose(env.oracle()["base_pos"][:2], p_locked, atol=2e-3)


def test_free_base_option_has_no_lock():
    e = _env(stance_lock=False)
    e.reset(1000)
    o = e.oracle()
    assert not o["base_locked"] and abs(o["base_pos"][0]) < 0.05
    e.close()
