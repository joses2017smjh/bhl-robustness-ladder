"""NavGym v5 (env version 3; N2 of docs/SOLUTIONS_2026-10-01.md): the visitation memory, the coarse wide map, the
yaw-change penalty and the deployment speed brake; observation keys per env version; the v5 extractor, its ONNX inputs and
build_obs keyed by them; versions 1-2 and the v1/v2 policy networks unchanged; the predeclared verdicts at their
boundaries; the launchers' frozen headers and command lines."""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

import bhl_robust.navgym.env as nenv
from bhl_robust.eval.random_maze import generate
from bhl_robust.navgym.env import (DT, LIDAR_RAYS, LIDAR_SECTORS, MAP_RES, OBS_KEYS_V1, OBS_KEYS_V2, OBS_KEYS_V3, ROBOT_RADIUS,
                                   V3_COARSE_CROP, V3_VISIT_KEYS, V3_VISIT_TAU_S, V3_YAW_CHANGE_COST, V4_IDLE_COST, V_MAX, EgoMap,
                                   MazeNavEnv, VisitMap, brake_scale, build_obs, coarse_crop, coarse_grid, heldout_env)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "bench"))
SLURM = REPO / "slurm" / "repo20260923"


# ------------------------------------------------------------------ visitation memory
def test_visit_map_shares_the_ego_map_lattice_and_marks_the_footprint():
    bounds = generate(6, 6, 0, extra_openings=1).bounds()
    em, vm = EgoMap(bounds), VisitMap(bounds)
    assert (vm.x0, vm.y0, vm.nx, vm.ny, vm.res) == (em.x0, em.y0, em.nx, em.ny, em.res)
    assert vm.res == MAP_RES == 0.2 and vm.radius == ROBOT_RADIUS == 0.22 and vm.tau == V3_VISIT_TAU_S == 60.0 and vm.dt == DT == 0.04
    for x, y in ((2.13, 3.71), (0.0, 0.0), (4.3, 1.1)):
        v = VisitMap(bounds)
        v.step(x, y)
        I, J = np.meshgrid(np.arange(v.nx), np.arange(v.ny), indexing="ij")
        cx, cy = v.x0 + (I + 0.5) * v.res, v.y0 + (J + 0.5) * v.res
        inside = (cx - x) ** 2 + (cy - y) ** 2 <= ROBOT_RADIUS ** 2
        # exactly the cells whose centres lie within the footprint radius read 1; every other cell 0
        assert inside.sum() >= 2 and np.array_equal(v.r == 1.0, inside) and np.all(v.r[~inside] == 0.0)
        assert v.r.dtype == np.float32 and v.updates == 1


def test_visit_map_decays_by_exp_minus_dt_over_60s_every_step():
    vm = VisitMap((0.0, 8.0, 0.0, 8.0))
    vm.step(1.0, 1.0)
    first = vm.r == 1.0
    k = math.exp(-DT / 60.0)
    for n in range(1, 51):
        vm.step(6.0, 6.0)                              # far away: the first footprint only decays
        assert np.allclose(vm.r[first], k ** n, rtol=1e-5, atol=0.0)
    assert vm.updates == 51 and vm.r.max() == 1.0 and vm.r.min() == 0.0
    # decay first, then mark: the cells under the robot read exactly 1 every step (r stays in [0, 1])
    assert np.all((vm.r == 1.0) | (vm.r < k ** 50 + 1e-6) | (vm.r == 0.0))


def test_visitation_crop_samples_exactly_like_the_ego_map_crop():
    rng = np.random.default_rng(5)
    bounds = generate(6, 6, 0, extra_openings=1).bounds()
    em, vm = EgoMap(bounds), VisitMap(bounds)
    em.l[:] = np.where(rng.random(em.l.shape) < 0.3, 4.0, -4.0).astype(np.float32)
    vm.r[:] = (em.l > 0).astype(np.float32)          # the visitation grid holds exactly the occupied indicator
    for _ in range(60):
        x, y, yaw = rng.uniform(-2.5, 9.5), rng.uniform(-2.5, 9.5), rng.uniform(-math.pi, math.pi)
        assert np.array_equal(vm.crop(x, y, yaw), em.crop(x, y, yaw)[0])   # same cells, same rotation, 0 outside


def test_v3_env_steps_the_visitation_once_per_step_at_the_observed_pose():
    env, obs, info = heldout_env((5, 5), 2, version=3, gamma=1.0)
    assert info["version"] == 3 and env.visit.updates == 1                # the start pose is marked at reset
    ii, jj = env.visit.footprint(env.x, env.y)
    assert len(ii) >= 2 and np.all(env.visit.r[ii, jj] == 1.0)
    rng = np.random.default_rng(0)
    for t in range(1, 60):
        obs, r, te, tr, inf = env.step(rng.uniform(-1, 1, 2).astype(np.float32))
        assert env.visit.updates == t + 1
        ii, jj = env.visit.footprint(env.x, env.y)
        assert np.all(env.visit.r[ii, jj] == 1.0)
        assert np.array_equal(obs["map_visit"][:3], env.emap.crop(env.x, env.y, env.yaw))
        assert np.array_equal(obs["map_visit"][3], env.visit.crop(env.x, env.y, env.yaw))
        if te or tr:
            break


def test_runner_style_visitation_replays_the_gym_memory_exactly():
    """maze_explore.py --policy steps its own VisitMap(maze.bounds(), dt=policy_dt) once per control step, at the pose it
    builds the observation at, then calls build_obs with it: that replays the gym's memory and observation exactly."""
    env, obs, _ = heldout_env((5, 5), 6, version=3, gamma=1.0)
    runner = VisitMap(env.maze.bounds(), dt=0.04)
    rng = np.random.default_rng(2)
    for _ in range(80):
        runner.step(env.x, env.y)
        assert np.array_equal(runner.r, env.visit.r)
        feeds = build_obs(OBS_KEYS_V3, env.ranges, env.emap, env.x, env.y, env.yaw, env.goal_xy, env.prev_action, visit=runner)
        assert all(np.array_equal(feeds[k], obs[k]) for k in OBS_KEYS_V3)
        obs, r, te, tr, _ = env.step(rng.uniform(-0.2, 1.0, 2).astype(np.float32))
        if te or tr:
            break


# ------------------------------------------------------------------ coarse wide map
def test_coarse_grid_is_the_3x3_block_mean_of_the_four_fine_channels():
    bounds = generate(6, 6, 0, extra_openings=1).bounds()
    em, vm = EgoMap(bounds), VisitMap(bounds)
    nx, ny = em.nx, em.ny
    NX, NY = -(-nx // 3), -(-ny // 3)
    em.l[:] = -4.0                                    # every fine cell free
    em.l[3, 4] = 4.0                                  # one occupied cell in block (1, 1)
    em.l[6, 7] = 0.0                                  # one unknown cell in block (2, 2)
    vm.r[0, 0] = 0.9
    g = coarse_grid(em, vm)
    assert g.shape == (4, NX, NY) and g.dtype == np.float32
    assert np.allclose(g[:, 1, 1], [1 / 9, 8 / 9, 0, 0]) and np.allclose(g[:, 2, 2], [0, 8 / 9, 1 / 9, 0])
    assert np.allclose(g[:, 0, 0], [0, 1, 0, 0.1]) and np.allclose(g[:, 5, 5], [0, 1, 0, 0])
    # the last block row is padded past the grid edge: padding counts as unknown, unvisited
    rx, ry = nx - 3 * (NX - 1), ny - 3 * (NY - 1)
    assert np.allclose(g[:, NX - 1, 5], [0, rx / 3, 1 - rx / 3, 0]) and np.allclose(g[:, NX - 1, NY - 1], [0, rx * ry / 9, 1 - rx * ry / 9, 0])
    assert np.allclose(g[0] + g[1] + g[2], 1.0)        # occupied + free + unknown = 1 in every coarse cell


def _ego_reference(grid, x0, y0, res, x, y, yaw, size=24):
    """Independent reference: a world grid sampled egocentrically, EgoMap.crop's geometry written out cell by cell;
    outside the grid = unknown, unvisited."""
    out = np.zeros((4, size, size), np.float32)
    out[2] = 1.0
    c, s = math.cos(yaw), math.sin(yaw)
    for i in range(size):
        for j in range(size):
            u, v = (size // 2 - i) * res, (size // 2 - j) * res
            wx, wy = x + u * c - v * s, y + u * s + v * c
            I, J = math.floor((wx - x0) / res), math.floor((wy - y0) / res)
            if 0 <= I < grid.shape[1] and 0 <= J < grid.shape[2]:
                out[:, i, j] = grid[:, I, J]
    return out


def test_coarse_crop_is_the_coarse_grid_sampled_egocentrically_at_0_6_m():
    mz = generate(6, 6, 3, extra_openings=1)
    em, vm = EgoMap(mz.bounds()), VisitMap(mz.bounds())
    rng = np.random.default_rng(1)
    em.l[:] = rng.choice([-4.0, 0.0, 4.0], size=em.l.shape).astype(np.float32)
    vm.r[:] = rng.random(vm.r.shape).astype(np.float32)
    g = coarse_grid(em, vm)
    assert V3_COARSE_CROP == 24 and V3_COARSE_CROP * 3 * MAP_RES == pytest.approx(14.4)
    for _ in range(25):
        x, y, yaw = rng.uniform(-1, 8), rng.uniform(-1, 8), rng.uniform(-math.pi, math.pi)
        got = coarse_crop(em, vm, x, y, yaw)
        assert got.shape == (4, 24, 24) and np.array_equal(got, _ego_reference(g, em.x0, em.y0, 3 * MAP_RES, x, y, yaw))


def test_coarse_crop_frame_forward_is_up_left_is_left_outside_is_unknown():
    bounds = generate(6, 6, 0, extra_openings=1).bounds()
    em, vm = EgoMap(bounds), VisitMap(bounds)
    em.l[:] = -4.0
    I0, J0 = 4, 6                                     # the robot stands at the centre of coarse cell (4, 6)
    x, y = em.x0 + (I0 + 0.5) * 0.6, em.y0 + (J0 + 0.5) * 0.6
    em.l[3 * (I0 + 5):3 * (I0 + 6), 3 * J0:3 * (J0 + 1)] = 4.0   # coarse cell (9, 6): 3.0 m along world +x is occupied
    occ = lambda yaw: coarse_crop(em, vm, x, y, yaw)[0]          # noqa: E731
    assert occ(0.0)[7, 12] == 1.0 and occ(0.0).sum() == 1.0      # facing +x: 5 rows ahead (up)
    assert occ(math.pi / 2)[12, 17] == 1.0 and occ(math.pi / 2).sum() == 1.0   # facing +y: on the right
    assert occ(-math.pi / 2)[12, 7] == 1.0                       # facing -y: on the left
    assert occ(math.pi)[17, 12] == 1.0                           # facing -x: behind (down)
    out = coarse_crop(em, vm, em.x0 + 0.1, em.y0 + 0.1, 0.0)     # at the grid corner: behind and right lie outside
    assert np.all(out[:, 23, :] == np.array([0, 0, 1, 0], np.float32)[:, None])
    assert np.all(out[:, :, 23] == np.array([0, 0, 1, 0], np.float32)[:, None])


# ------------------------------------------------------------------ observation keys per env version
@pytest.mark.parametrize("version,keys", [(1, OBS_KEYS_V1), (2, OBS_KEYS_V2), (3, OBS_KEYS_V3)])
def test_observation_keys_and_shapes_per_env_version(version, keys):
    import navgym_train as nt
    env = MazeNavEnv(sizes=((4, 4),), version=version, gamma=1.0)
    obs, info = env.reset(seed=2)
    assert tuple(obs) == keys and env.obs_keys == keys and set(env.observation_space.spaces) == set(keys)
    assert env.observation_space.contains({k: np.asarray(v, np.float32) for k, v in obs.items()})
    shapes = {"lidar": (36,), "near": (36,), "map": (3, 24, 24), "map_visit": (4, 24, 24), "map_coarse": (4, 24, 24), "goal": (5,)}
    assert {k: v.shape for k, v in obs.items()} == {k: shapes[k] for k in keys}
    assert nt.onnx_keys(env.observation_space) == keys
    assert hasattr(env, "visit") == (version == 3)
    obs, r, te, tr, info = env.step(np.array([0.0, 0.1], np.float32))
    v3_info = {"yaw_change_cost", "brake_scale", "yaw_flips", "yaw_sat_steps", "braked_steps"}
    assert (v3_info <= set(info)) == (version == 3)
    if version == 2:
        assert set(info) == {"outcome", "maze_seed", "t"}            # the v2 info, unchanged
    assert not set(V3_VISIT_KEYS) & set(OBS_KEYS_V1 + OBS_KEYS_V2) and set(V3_VISIT_KEYS) <= set(OBS_KEYS_V3)


# ------------------------------------------------------------------ nothing else changes
def _descend(env):
    """Oracle controller (tests only): steer down the continuous geodesic (as tests/test_navgym.py)."""
    h = 0.05
    gx = env.geo(env.x + h, env.y) - env.geo(env.x - h, env.y)
    gy = env.geo(env.x, env.y + h) - env.geo(env.x, env.y - h)
    e = (math.atan2(-gy, -gx) - env.yaw + math.pi) % (2 * math.pi) - math.pi
    return np.array([1.0 if abs(e) < 0.35 else -1.0, max(-1.0, min(1.0, 2.0 * e))], np.float32)


def test_v3_shares_v2_episodes():
    for size, sd in (((5, 5), 3), ((6, 6), 7)):
        e2, o2, i2 = heldout_env(size, sd, version=2, gamma=1.0)
        e3, o3, i3 = heldout_env(size, sd, version=3, gamma=1.0)
        assert (e3.episode_seed, e3.dyn, e3.max_steps, e3.route_m, e3.x, e3.y, e3.yaw) == \
            (e2.episode_seed, e2.dyn, e2.max_steps, e2.route_m, e2.x, e2.y, e2.yaw)
        assert {k: v for k, v in i3.items() if k != "version"} == {k: v for k, v in i2.items() if k != "version"}
        assert (i2["version"], i3["version"]) == (2, 3)
        for k in ("lidar", "near", "goal"):
            assert np.array_equal(o3[k], o2[k])
        assert np.array_equal(o3["map_visit"][:3], o2["map"])


def test_v3_is_v2_plus_the_yaw_cost_when_the_brake_is_held_open(monkeypatch):
    """With the brake factor forced to 1, v3 replays v2's trajectory (dynamics, RNG draws, collisions, goal, time limit,
    stall price) step for step, and every v3 reward is the v2 reward minus exactly 0.005 * |a_yaw(t) - a_yaw(t-1)|."""
    monkeypatch.setattr(nenv, "brake_scale", lambda ranges: 1.0)
    for sd in (2, 5):
        e2 = MazeNavEnv(sizes=((4, 4),), version=2, gamma=1.0, idle_cost=V4_IDLE_COST)
        e3 = MazeNavEnv(sizes=((4, 4),), version=3, gamma=1.0, idle_cost=V4_IDLE_COST)
        e2.reset(seed=sd)
        e3.reset(seed=sd)
        prev, n = 0.0, 0
        while True:
            a = _descend(e2)
            assert np.array_equal(a, _descend(e3))
            o2, r2, te2, tr2, i2 = e2.step(a)
            o3, r3, te3, tr3, i3 = e3.step(a)
            n += 1
            assert (e3.x, e3.y, e3.yaw, te3, tr3, i3["outcome"], i3["idle_charged"]) == (e2.x, e2.y, e2.yaw, te2, tr2, i2["outcome"], i2["idle_charged"])
            assert r3 == pytest.approx(r2 - V3_YAW_CHANGE_COST * abs(float(a[1]) - prev), abs=1e-12)
            prev = float(a[1])
            if te2 or tr2:
                break
        assert i2["outcome"] == "goal" and n > 100


def test_yaw_change_penalty_is_0005_times_the_clipped_yaw_action_change():
    assert V3_YAW_CHANGE_COST == 0.005
    e2 = MazeNavEnv(sizes=((5, 5),), version=2, gamma=1.0, idle_cost=V4_IDLE_COST)
    e3 = MazeNavEnv(sizes=((5, 5),), version=3, gamma=1.0, idle_cost=V4_IDLE_COST)
    e2.reset(seed=9)
    e3.reset(seed=9)
    prev = 0.0                                         # a_yaw before the first step: the reset's previous action
    for w in (0.3, 0.3, -0.5, 1.7, -2.0, 0.0, 0.25):    # 1.7 and -2.0 clip to 1 and -1
        a = np.array([-1.0, w], np.float32)             # no forward command: nothing for the brake to scale
        _, r2, *_ = e2.step(a)
        _, r3, te, tr, info = e3.step(a)
        cur = float(np.clip(np.float32(w), -1.0, 1.0))
        assert info["yaw_change_cost"] == pytest.approx(0.005 * abs(cur - prev), abs=1e-12)
        assert r3 == pytest.approx(r2 - 0.005 * abs(cur - prev), abs=1e-12)
        prev = cur
    assert not (te or tr)


def test_v3_counts_yaw_flips_saturation_and_braked_steps_for_the_report():
    env = MazeNavEnv(sizes=((5, 5),), version=3, gamma=1.0)
    env.reset(seed=3)
    for w in (1.0, -1.0, 0.5, 0.0, -0.995, 0.3):
        _, _, _, _, info = env.step(np.array([-1.0, w], np.float32))
    # sign changes: (1, -1), (-1, 0.5), (-0.995, 0.3); 0 has no sign; |a_yaw| >= 0.99: 1, -1, -0.995; no forward command
    assert (info["yaw_flips"], info["yaw_sat_steps"], info["braked_steps"]) == (3, 3, 0)


# ------------------------------------------------------------------ deployment speed brake
def _scan_with(sectors_m):
    """A raw 108-ray scan whose 36 sector minima are `sectors_m` (every ray of a sector at its value)."""
    return np.repeat(np.asarray(sectors_m, float), LIDAR_RAYS // LIDAR_SECTORS)


def test_brake_factor_is_clip_clearance_minus_042_over_048_within_25_deg():
    far = np.full(36, 12.0)
    assert brake_scale(_scan_with(far)) == 1.0
    centres = np.linspace(-np.pi, np.pi, 108, endpoint=False).reshape(36, 3).mean(axis=1)
    sel = np.abs(centres) <= np.deg2rad(25.0)
    assert np.array_equal(nenv.V3_BRAKE_SECTORS, sel) and sel.sum() == 5
    for k in np.flatnonzero(sel):
        for clear, want in ((0.42, 0.0), (0.30, 0.0), (0.66, 0.5), (0.90, 1.0), (0.9 + 1e-9, 1.0), (0.42 + 0.048, 0.1)):
            s = far.copy()
            s[k] = clear
            assert brake_scale(_scan_with(s)) == pytest.approx(want, abs=1e-9)
    for k in np.flatnonzero(~sel):                    # only the sectors within +-25 deg of travel matter
        s = far.copy()
        s[k] = 0.1
        assert brake_scale(_scan_with(s)) == 1.0
    raw = _scan_with(far)
    raw[3 * int(np.flatnonzero(sel)[0]) + 1] = 0.54     # one short ray: its sector minimum sets the clearance
    assert brake_scale(raw) == pytest.approx((0.54 - 0.42) / 0.48, abs=1e-12)


def test_brake_factor_matches_team_sensors_brake_command():
    pytest.importorskip("mujoco")
    from bhl_robust.eval.team_sensors import brake_command
    rng = np.random.default_rng(3)
    imu = np.zeros(10)
    imu[2], imu[-1] = -1.0, 1.0                        # upright, still, valid
    depth = np.full((2, 8, 8), 6.0)                     # no stereo-depth tightening (the gym has no depth camera)
    for _ in range(300):
        raw = rng.uniform(0.2, 2.0, 108)
        out, info = brake_command(np.array([0.3, 0.0, 0.5]), lidar_m=raw.reshape(36, 3).min(axis=1), depth_m=depth, imu=imu, fresh=True)
        k = brake_scale(raw)
        assert k == pytest.approx(info["range_scale"], abs=1e-12)
        assert out[0] == pytest.approx(0.3 * k, abs=1e-12) and out[2] == 0.5     # forward speed scaled, yaw rate untouched


def test_brake_packet_refreshes_on_the_team_sensors_10hz_schedule(monkeypatch):
    env = MazeNavEnv(sizes=((5, 5),), version=3, gamma=1.0)
    env.reset(seed=1)
    real, calls = nenv.brake_scale, []
    monkeypatch.setattr(nenv, "brake_scale", lambda r: calls.append(env.t) or real(r))
    for _ in range(21):
        env.step(np.array([-1.0, 0.2], np.float32))
    assert calls == [0, 3, 5, 8, 10, 13, 15, 18, 20]
    nxt, want = 0.0, []                                  # TeamSensors.filter_commands' capture rule, written out
    for t in range(21):
        now = t * DT
        if now + 1e-9 >= nxt:
            while nxt <= now + 1e-9:
                nxt += 0.1
            want.append(t)
    assert calls == want


def test_the_brake_scales_the_forward_command_before_the_latency_queue():
    env = MazeNavEnv(sizes=((5, 5),), version=3, gamma=1.0, randomize_dynamics=False)
    env.reset(seed=1)
    env.yaw = float(env.yaw + env.angles[int(np.argmin(env.ranges))])     # face the nearest wall
    env._scan()
    k = brake_scale(env.ranges)
    assert 0.0 <= k < 1.0
    _, _, _, _, info = env.step(np.array([1.0, 0.3], np.float32))
    assert info["brake_scale"] == k and info["braked_steps"] == 1
    assert env.dyn.latency == 1 and env.queue[-1][0] == pytest.approx(V_MAX * k, abs=1e-12) and env.queue[-1][1] == pytest.approx(0.3, abs=1e-7)


# ------------------------------------------------------------------ v5 extractor, ONNX inputs, build_obs keyed by them
def _tiny_v5_model():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import navgym_train as nt
    venv = DummyVecEnv([lambda: MazeNavEnv(sizes=((3, 3),), version=3, gamma=1.0, idle_cost=V4_IDLE_COST)])
    pk = {"features_extractor_class": nt.NavExtractorV5, "features_extractor_kwargs": {"features_dim": 256},
          "net_arch": {"pi": [128, 128], "vf": [128, 128]}, "share_features_extractor": True, "log_std_init": -0.5}
    return nt, PPO("MultiInputPolicy", venv, n_steps=8, batch_size=8, seed=7, device="cpu", policy_kwargs=pk)


def test_v5_onnx_inputs_and_build_obs_keyed_by_them(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    nt, model = _tiny_v5_model()
    fe = model.policy.features_extractor
    assert type(fe).__name__ == "NavExtractorV5" and fe.cnn[0].in_channels == 4 and fe.cnn_coarse[0].in_channels == 4
    assert fe.vec_keys == ["lidar", "near", "goal"] and fe.out[0].in_features == 384
    info = nt.export_onnx(model, tmp_path / "actor.onnx")
    assert info["inputs"] == list(OBS_KEYS_V3) and info["max_abs_diff_vs_torch"] < 1e-5
    sess = ort.InferenceSession(str(tmp_path / "actor.onnx"))
    keys = tuple(i.name for i in sess.get_inputs())
    assert keys == OBS_KEYS_V3
    env, obs, _ = heldout_env((5, 5), 1, version=3, gamma=1.0)
    rng = np.random.default_rng(0)
    for _ in range(15):
        feeds = build_obs(keys, env.ranges, env.emap, env.x, env.y, env.yaw, env.goal_xy, env.prev_action, visit=env.visit)
        assert tuple(feeds) == keys and all(np.array_equal(feeds[k], obs[k]) for k in keys)
        onnx_a = sess.run(None, {k: np.asarray(v, np.float32)[None] for k, v in feeds.items()})[0][0]
        torch_a, _ = model.predict(obs, deterministic=True)
        assert np.allclose(np.clip(onnx_a, -1, 1), torch_a, atol=1e-5)
        obs, *_ = env.step(rng.uniform(-1, 1, 2).astype(np.float32))
    with pytest.raises(ValueError):                       # the v5 keys need the visitation memory
        build_obs(keys, env.ranges, env.emap, env.x, env.y, env.yaw, env.goal_xy, env.prev_action)
    v2 = build_obs(OBS_KEYS_V2, env.ranges, env.emap, env.x, env.y, env.yaw, env.goal_xy, env.prev_action)
    assert tuple(v2) == OBS_KEYS_V2 and np.array_equal(v2["map"], obs["map_visit"][:3])


# Recorded on 2026-10-01 at git HEAD 094797e, BEFORE the v5 edit (solutions-20260930/campaign-navgym-v5/gold/record_gold.py):
# PPO("MultiInputPolicy", seed=7) with NavExtractor on the v1 / v2 observation spaces -- (sum, abs-sum) of named initial
# parameters and the first 5 deterministic actions on MazeNavEnv(sizes=((4, 4),), version=v, gamma=1.0).reset(seed=3).
NET_GOLD = {
    1: {"features_extractor.cnn.0.weight": (-6.766318182460964, 93.80589790828526),
        "features_extractor.mlp.0.weight": (16.76953288902041, 521.6406949498687),
        "features_extractor.out.0.weight": (3.257783433087468, 4624.744443940346),
        "action_net.weight": (0.011341845849756282, 0.1858161642705909),
        "value_net.weight": (1.5021289555588737, 9.176232438883744),
        "log_std": (0.0, 0.0),
        "acts": [[0.0009740244131535292, 0.0062972791492938995], [0.0009148415992967784, 0.006391329690814018],
                 [0.0009179332409985363, 0.0063911848701536655], [0.0007738213171251118, 0.006269064266234636],
                 [0.0007764014881104231, 0.0062669687904417515]]},
    2: {"features_extractor.cnn.0.weight": (3.286366351414472, 96.44624925334938),
        "features_extractor.mlp.0.weight": (-17.92117141510971, 988.2976379463653),
        "features_extractor.out.0.weight": (-2.4319237163271055, 4624.2244674625945),
        "action_net.weight": (-0.014091089535611445, 0.1793809613964754),
        "value_net.weight": (0.5624440484389197, 9.006623099994613),
        "log_std": (-1.0, 1.0),
        "acts": [[-0.005151447840034962, -0.002591340336948633], [-0.005038367584347725, -0.002839841181412339],
                 [-0.005035554990172386, -0.0028385636396706104], [-0.0022865738719701767, -2.1996558643877506e-05],
                 [-0.0022817556746304035, -1.5681085642427206e-05]]},
}


@pytest.mark.parametrize("ver", [1, 2])
def test_v1_v2_policy_networks_are_unchanged(ver):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import navgym_train as nt
    venv = DummyVecEnv([lambda: MazeNavEnv(sizes=((3, 3),), version=ver, gamma=1.0)])
    P = nt.PPO_SETTINGS["v2" if ver == 2 else "v1"]
    pk = {"features_extractor_class": nt.NavExtractor, "features_extractor_kwargs": {"features_dim": 256},
          "net_arch": {"pi": [128, 128], "vf": [128, 128]}, "share_features_extractor": True}
    if P["log_std_init"] != 0.0:
        pk["log_std_init"] = P["log_std_init"]
    model = PPO("MultiInputPolicy", venv, n_steps=8, batch_size=8, seed=7, device="cpu", policy_kwargs=pk)
    sd = model.policy.state_dict()
    g = NET_GOLD[ver]
    for name, (s, a) in ((k, v) for k, v in g.items() if k != "acts"):
        # float32 last-bit differences of the orthogonal init (LAPACK QR) across CPUs: a signed sum of 65 536 weights moves by
        # ~1e-5, so it is compared at 1e-6 of the weights' absolute sum; any change of architecture or init moves it by O(1)
        assert float(sd[name].double().sum()) == pytest.approx(s, abs=1e-6 * max(1.0, a)), name
        assert float(sd[name].double().abs().sum()) == pytest.approx(a, rel=1e-6, abs=1e-9), name
    env = MazeNavEnv(sizes=((4, 4),), version=ver, gamma=1.0)
    obs, _ = env.reset(seed=3)
    for want in g["acts"]:
        act, _ = model.predict(obs, deterministic=True)
        assert np.allclose(act, want, rtol=1e-4, atol=1e-7)
        obs, *_ = env.step(act)


# ------------------------------------------------------------------ the reported yaw statistics
def test_yaw_stats_report_flips_per_second_and_saturation_share():
    import navgym_train as nt
    infos = {((5, 5), 0): {"t": 100, "yaw_flips": 10, "yaw_sat_steps": 50, "braked_steps": 20, "outcome": "goal"},
             ((5, 5), 1): {"t": 300, "yaw_flips": 2, "yaw_sat_steps": 150, "braked_steps": 0, "outcome": "time_out"},
             ((6, 6), 0): {"t": 250, "yaw_flips": 25, "yaw_sat_steps": 250, "braked_steps": 5, "outcome": "goal"}}
    y = nt.yaw_stats(infos)
    assert set(y) == {"5x5", "6x6"} and (y["5x5"]["episodes"], y["5x5"]["steps"]) == (2, 400)
    assert y["5x5"]["yaw_flips_per_s"] == pytest.approx(12 / (400 * 0.04)) and y["5x5"]["wz_saturation_share"] == pytest.approx(0.5)
    assert y["6x6"]["yaw_flips_per_s"] == pytest.approx(2.5) and y["6x6"]["wz_saturation_share"] == 1.0
    assert y["6x6"]["braked_share"] == pytest.approx(0.02)
    assert "NOT GATED" in nt.V5_YAW_NOTE


# ------------------------------------------------------------------ predeclared gym gate (FROZEN 2026-10-02)
def _v5_summary(s5=0.85, s6=0.75, c6=0.05, steps=30_000_000, n=48, base=61_000, **over):
    import navgym_train as nt
    ev = {"4x4": {"success": 1.0, "collision": 0.0, "time_out": 0.0, "n": n},
          "5x5": {"success": s5, "collision": 0.0, "time_out": 1 - s5, "n": n},
          "6x6": {"success": s6, "collision": c6, "time_out": 0.0, "n": n}}
    s = {"ppo": "v2", "env_version": 3, "std_ceiling": None, "total_steps": 30_000_000, "num_timesteps": steps,
         "reached_last_step": steps >= 30_000_000, "final_eval_fresh_48": ev, "final_eval_heldout_48": ev,
         "fresh_set": {"maze_seeds": f"{base} + k, k = 0..47", "dynamics": f"reset(seed={base} + k)"},
         "stall_price": dict(nt.V4_STALL_PRICE), "v5_config": json.loads(json.dumps(nt.V5_CONFIG))}
    s.update(over)
    return s


def test_v5_rule_is_v4s_clauses_on_the_never_used_61000_set():
    import navgym_train as nt
    same = ("5x5_success_min", "6x6_success_min", "6x6_collision_max", "n", "total_steps", "min_seeds_passing", "set")
    assert {k: nt.V5_RULE[k] for k in same} == {k: nt.V4_RULE[k] for k in same}
    assert (nt.V5_RULE["5x5_success_min"], nt.V5_RULE["6x6_success_min"], nt.V5_RULE["6x6_collision_max"]) == (0.80, 0.70, 0.15)
    assert nt.V5_RULE["seeds"] == (8, 9, 10) and nt.V5_RULE["fresh_base"] == 61_000 and nt.V5_RULE["min_seeds_passing"] == 2
    assert nt.V5_CONFIG["env_version"] == 3 and nt.V5_CONFIG["obs_keys"] == list(OBS_KEYS_V3)
    assert nt.V5_CONFIG == json.loads(json.dumps(nt.v5_config(MazeNavEnv(version=3).observation_space, "NavExtractorV5")))
    t = nt.V5_TRANSFER_RULE
    assert t["seeds"] == list(range(63_000, 63_012)) and (t["n"], t["min_goals"], t["max_falls"], t["time_limit_s"]) == (12, 10, 0, 180.0)
    assert t["maze"] == {"n": 6, "m": 6, "extra_openings": 1}


def test_v5_gym_gate_per_seed_boundaries():
    import navgym_train as nt
    J = lambda **kw: nt.judge_v5(_v5_summary(**kw))      # noqa: E731
    assert J()["meets"] and J(s5=0.80, s6=0.70, c6=0.15)["meets"]                 # every bound inclusive
    for kw in (dict(s5=0.79), dict(s6=0.69), dict(c6=0.16), dict(steps=29_999_999), dict(n=47), dict(base=40_000),
               dict(base=20_000), dict(ppo="v3"), dict(env_version=2), dict(std_ceiling=1.0), dict(stall_price=None),
               dict(stall_price={"lever": "idle_cost", "idle_cost": 0.004}), dict(total_steps=60_000_000, steps=60_000_000),
               dict(reached_last_step=False), dict(v5_config=None)):
        assert not J(**kw)["meets"], kw
    for path, val in ((("yaw_change_cost",), 0.004), (("visitation", "tau_s"), 30.0), (("coarse_map", "crop"), 16),
                      (("brake", "lo_m"), 0.40), (("extractor",), "NavExtractor"), (("obs_keys",), list(OBS_KEYS_V2))):
        cfg = json.loads(json.dumps(nt.V5_CONFIG))
        d = cfg
        for p in path[:-1]:
            d = d[p]
        d[path[-1]] = val
        assert not J(v5_config=cfg)["meets"], path
    s = _v5_summary()
    s["fresh_set"] = {"maze_seeds": "61000 + k, k = 0..47", "dynamics": "reset(seed=k)"}     # not the declared dynamics
    assert not nt.judge_v5(s)["meets"]
    assert nt.judge_v5(None) == {"meets": False, "checks": {"summary_exists": False}, "numbers": None}
    n = J()["numbers"]
    assert set(n["never_used_61000"]) == {"4x4", "5x5", "6x6"} and "heldout_10000_for_comparability" in n


def test_v5_arm_verdict_pending_pass_negative(tmp_path):
    import navgym_train as nt

    def write(sd, s):
        d = tmp_path / f"armV5-s{sd}"
        d.mkdir(exist_ok=True)
        (d / "summary.json").write_text(json.dumps(s))
    assert nt.verdict_v5(tmp_path)["verdict"] == "PENDING"
    assert nt.verdict_v5(tmp_path, final=True)["verdict"] == "NEGATIVE"
    write(8, _v5_summary())
    v = nt.verdict_v5(tmp_path)
    assert v["verdict"] == "PENDING" and v["seeds_passing"] == 1
    assert nt.verdict_v5(tmp_path, final=True)["verdict"] == "NEGATIVE"          # 1 of 3: missing runs are misses at the end
    write(9, _v5_summary(s5=0.79))
    assert nt.verdict_v5(tmp_path)["verdict"] == "PENDING"
    write(10, _v5_summary(s6=0.70, c6=0.15))
    v = nt.verdict_v5(tmp_path, final=True)
    assert v["verdict"] == "PASS" and v["seeds_passing"] == 2 and v["gate_passing_seeds"] == [8, 10] and v["final"] is True
    write(10, _v5_summary(c6=0.16))
    v = nt.verdict_v5(tmp_path, final=True)
    assert v["verdict"] == "NEGATIVE" and v["gate_passing_seeds"] == [8]
    assert v["rule"]["text"] == nt.V5_TEXT and "LEARNED" in v["label"] and "ORACLE" in v["label"]


def test_v5_verdict_file_is_written_once(tmp_path, capsys):
    import navgym_train as nt
    d = tmp_path / "armV5-s8"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps(_v5_summary()))
    assert nt.verdict_v5_main([str(tmp_path), "--final"]) == 0
    first = (tmp_path / "verdict_v5.json").read_text()
    assert json.loads(first)["verdict"] == "NEGATIVE" and "NAVGYM-V5 VERDICT: NEGATIVE" in capsys.readouterr().out
    for sd in (9, 10):
        dd = tmp_path / f"armV5-s{sd}"
        dd.mkdir()
        (dd / "summary.json").write_text(json.dumps(_v5_summary()))
    assert nt.verdict_v5_main([str(tmp_path), "--final"]) == 0
    assert (tmp_path / "verdict_v5.json").read_text() == first and "REFUSING" in capsys.readouterr().out


def test_v5_status_check_is_never_written_as_the_final_verdict_file(tmp_path, capsys):
    """Review 2026-10-02: a status check (no --final) must never fix a non-final verdict in <root>/verdict_v5.json, the
    file the final verdict job writes and the transfer launcher reads."""
    import navgym_train as nt
    assert nt.V5_FINAL_VERDICT_NAME == "verdict_v5.json"
    d = tmp_path / "armV5-s8"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps(_v5_summary()))
    final = tmp_path / "verdict_v5.json"
    assert nt.verdict_v5_main([str(tmp_path)]) == 0                                   # a status check: printed only
    out = capsys.readouterr().out
    assert "status only" in out and "NAVGYM-V5 VERDICT: PENDING" in out and "final=False" in out
    assert not final.exists() and sorted(p.name for p in tmp_path.iterdir()) == ["armV5-s8"]
    for target in (final, tmp_path / "elsewhere" / "verdict_v5.json"):                # the name is reserved for --final
        assert nt.verdict_v5_main([str(tmp_path), "--out", str(target)]) == 2
        assert not target.exists() and "NOT WRITTEN" in capsys.readouterr().out
    assert not (tmp_path / "elsewhere").exists()
    per_seed = d / "verdict_v5_at_end_of_s8.json"                                     # the train mode's per-seed status file
    assert nt.verdict_v5_main([str(tmp_path), "--out", str(per_seed)]) == 0
    assert json.loads(per_seed.read_text())["final"] is False and not final.exists()
    assert nt.verdict_v5_main([str(tmp_path), "--final"]) == 0                        # then the final verdict, once
    v = json.loads(final.read_text())
    assert v["final"] is True and v["verdict"] == "NEGATIVE" and "wrote" in capsys.readouterr().out


# ------------------------------------------------------------------ predeclared physics transfer (FROZEN 2026-10-02)
def test_transfer_runs_only_the_gate_passing_final_actors_after_a_final_pass():
    import navgym_train as nt
    per = {"8": {"meets": True}, "9": {"meets": False}, "10": {"meets": True}}
    assert nt.v5_gate_passing_actors({"verdict": "PASS", "final": True, "per_seed": per}) == ["armV5-s8", "armV5-s10"]
    for gv in ({"verdict": "PASS", "final": False, "per_seed": per}, {"verdict": "NEGATIVE", "final": True, "per_seed": per},
               {"verdict": "PENDING", "final": False, "per_seed": per}, None, {}):
        assert nt.v5_gate_passing_actors(gv) == []


def _episode(actor, sd, goal=True, outcome=None, capture=True, visit=True, n=6, m=6, extra=1, root="/r"):
    e = {"seed": sd, "success": goal, "clean_success": goal, "outcome": outcome or ("goal" if goal else "time_out"),
         "policy": f"{root}/{actor}/actor.onnx", "maze": {"n": n, "m": m, "extra_openings": extra}}
    if capture:
        e["policy_map_integration"] = {"mode": "capture_pose", "updates_at_capture_pose": 400, "updates_at_loop_top_pose": 0}
    if visit:
        e["policy_visitation"] = {"updates": 4500}
    return e


def _actor_run(actor, goals, falls=0, seeds=range(63_000, 63_012), root="/r"):
    rows = {}
    for k, sd in enumerate(seeds):
        rows[sd] = _episode(actor, sd, k < goals, outcome=("fall" if goals <= k < goals + falls else None), root=root)
    summ = {"n": len(rows), "success": goals, "falls": falls,
            "settings": {"n": 6, "m": 6, "extra_openings": 1, "time_limit": 180.0, "policy_capture_pose": True}}
    return rows, summ


def test_v5_transfer_rule_boundaries():
    import navgym_train as nt

    def J(spec):
        rows, summ = {}, {}
        for a, (g, f) in spec.items():
            rows[a], summ[a] = _actor_run(a, g, f)
        return nt.judge_v5_transfer(list(spec), rows, summ)
    assert J({"armV5-s8": (10, 0)})["verdict"] == "PASS"
    assert J({"armV5-s8": (9, 0)})["verdict"] == "NEGATIVE"
    assert J({"armV5-s8": (10, 1)})["verdict"] == "NEGATIVE"                       # 10 goals and 1 fall
    assert J({"armV5-s8": (12, 0), "armV5-s10": (9, 0)})["verdict"] == "NEGATIVE"  # EACH gate-passing actor
    v = J({"armV5-s8": (10, 0), "armV5-s10": (11, 0)})
    assert v["verdict"] == "PASS" and v["per_actor"]["armV5-s10"]["goals"] == 11 and v["meets"] == {"armV5-s8": True, "armV5-s10": True}
    # INCOMPLETE: no actor, a missing summary, not 12 episodes, the wrong seeds, not the declared run
    assert nt.judge_v5_transfer([], {}, {})["verdict"] == "INCOMPLETE"
    rows, summ = _actor_run("armV5-s8", 12)
    A = ["armV5-s8"]
    assert nt.judge_v5_transfer(A, {"armV5-s8": rows}, {"armV5-s8": None})["verdict"] == "INCOMPLETE"
    r11 = dict(list(rows.items())[:11])
    assert nt.judge_v5_transfer(A, {"armV5-s8": r11}, {"armV5-s8": {**summ, "n": 11, "success": 11}})["verdict"] == "INCOMPLETE"
    wrong, ws = _actor_run("armV5-s8", 12, seeds=range(50_000, 50_012))
    assert nt.judge_v5_transfer(A, {"armV5-s8": wrong}, {"armV5-s8": ws})["verdict"] == "INCOMPLETE"
    for bad in (dict(capture=False), dict(visit=False), dict(n=5, m=5), dict(extra=2)):
        rr = {sd: _episode("armV5-s8", sd, True, **bad) for sd in range(63_000, 63_012)}
        assert nt.judge_v5_transfer(A, {"armV5-s8": rr}, {"armV5-s8": summ})["verdict"] == "INCOMPLETE", bad
    other = {sd: _episode("armV5-s9", sd, True) for sd in range(63_000, 63_012)}      # another seed's actor.onnx
    assert nt.judge_v5_transfer(A, {"armV5-s8": other}, {"armV5-s8": summ})["verdict"] == "INCOMPLETE"
    for k, val in (("time_limit", 150.0), ("policy_capture_pose", None), ("extra_openings", 2)):
        bad = {**summ, "settings": {**summ["settings"], k: val}}
        assert nt.judge_v5_transfer(A, {"armV5-s8": rows}, {"armV5-s8": bad})["verdict"] == "INCOMPLETE", k
    assert nt.judge_v5_transfer(A, {"armV5-s8": rows}, {"armV5-s8": {**summ, "success": 11}})["verdict"] == "INCOMPLETE"
    # the A* reference is reported, not gated
    v = nt.judge_v5_transfer(A, {"armV5-s8": rows}, {"armV5-s8": summ}, astar_rows={sd: {"success": False} for sd in range(63_000, 63_012)})
    assert v["verdict"] == "PASS" and v["astar_reference"]["goals"] == 0 and "not gated" in v["astar_reference"]["label"]
    assert v["rule"] == nt.V5_TRANSFER_RULE["text"]


def test_v5_transfer_main_lists_actors_reads_the_files_and_writes_once(tmp_path, capsys):
    import navgym_train as nt
    gv = tmp_path / "verdict_v5.json"
    gv.write_text(json.dumps({"verdict": "PASS", "final": True,
                              "per_seed": {"8": {"meets": True}, "9": {"meets": False}, "10": {"meets": False}}}))
    out = tmp_path / "transfer"
    assert nt.verdict_v5_transfer_main([str(out), "--gym-verdict", str(gv), "--list-actors"]) == 0
    assert capsys.readouterr().out.split() == ["armV5-s8"]
    rows, summ = _actor_run("armV5-s8", 10, root=str(tmp_path))
    d = out / "armV5-s8"
    d.mkdir(parents=True)
    for sd, e in rows.items():
        (d / f"seed{sd}.json").write_text(json.dumps(e))
    (d / "summary.json").write_text(json.dumps(summ))
    capsys.readouterr()
    assert nt.verdict_v5_transfer_main([str(out), "--gym-verdict", str(gv)]) == 0       # no --final: a status check, printed only
    o = capsys.readouterr().out
    assert "status only" in o and "NAVGYM-V5-TRANSFER VERDICT: PASS" in o and not (out / "verdict.json").exists()
    assert nt.verdict_v5_transfer_main([str(out), "--gym-verdict", str(gv), "--final"]) == 0
    v = json.loads((out / "verdict.json").read_text())
    assert v["verdict"] == "PASS" and v["per_actor"]["armV5-s8"]["goals"] == 10 and v["gym_verdict"]["verdict"] == "PASS"
    first = (out / "verdict.json").read_text()
    (d / "seed63000.json").write_text(json.dumps(_episode("armV5-s8", 63000, False, root=str(tmp_path))))
    capsys.readouterr()
    nt.verdict_v5_transfer_main([str(out), "--gym-verdict", str(gv), "--final"])
    assert (out / "verdict.json").read_text() == first and "REFUSING" in capsys.readouterr().out


def test_v5_transfer_main_reads_only_a_final_gym_verdict(tmp_path, capsys):
    """Review 2026-10-02: a non-final (or missing) gym verdict lists, judges and writes nothing (exit 4), so the transfer
    launcher stops (set -e) instead of recording a permanent not_run.json."""
    import navgym_train as nt
    per = {"8": {"meets": True}, "9": {"meets": True}, "10": {"meets": False}}
    out = tmp_path / "transfer"
    for name, gv in (("pass-not-final", {"verdict": "PASS", "final": False, "per_seed": per}),
                     ("pending", {"verdict": "PENDING", "final": False, "per_seed": per}), ("missing", None)):
        p = tmp_path / name / "verdict_v5.json"
        if gv is not None:
            p.parent.mkdir()
            p.write_text(json.dumps(gv))
        for extra in (["--list-actors"], [], ["--final"]):
            capsys.readouterr()
            assert nt.verdict_v5_transfer_main([str(out), "--gym-verdict", str(p), *extra]) == 4, (name, extra)
            c = capsys.readouterr()
            assert "NOT A FINAL GYM VERDICT" in c.err and c.out == "", (name, extra)
    assert not out.exists()


# ------------------------------------------------------------------ launchers
V5_SPEC = (
    "v5 = the v4 configuration exactly (MazeNavEnv dynamics and rewards of env version 2, `--ppo v2`, the frozen v4 stall "
    "price, 16 envs, 30 M steps, the v4 curriculum and periodic evaluation; see slurm/repo20260923/cpu_navgym_v4.sbatch and "
    "scripts/bench/navgym_train.py V4_RULE) PLUS these changes ONLY (FROZEN), as a new env version (3):",
    "Visitation channel: a world-fixed grid at the ego map's 0.2 m resolution holding a recency value r in [0, 1]: every step, "
    "cells within the robot's footprint radius are set to 1, and all cells decay as r *= exp(-dt / 60 s). It is cropped and "
    "rotated exactly like the existing egocentric map crop and added to it as an extra channel.",
    "Coarse wide map: the same ego map (all existing channels) plus the visitation channel, downsampled to 0.6 m cells and "
    "cropped 24 x 24 (14.4 m) around the robot in the same egocentric frame, as a second map input.",
    "Yaw-change penalty: reward -= 0.005 * |a_yaw(t) - a_yaw(t-1)| (actions in [-1, 1]); chosen, not tuned.",
    "The deployment speed brake in the gym, as the physics stack applies it (team_sensors.brake_command: v *= clip((clearance "
    "- 0.42) / 0.48, 0, 1) within +-25 deg of travel, refreshed at 10 Hz); results/solutions-20260930/sysid/sysid_env.py has "
    "an implementation to port.",
    "Nothing else changes. Env versions 1 and 2 and the v3/v4 training configurations must reproduce exactly: the existing "
    "tests in tests/test_navgym.py (test_v1_is_reproduced_exactly etc.) must still pass, and the v2 gold test must pass.",
)
REPORTED = ("Reported, not gated: yaw flips per second and |wz| saturation share in the gym evaluation. Maze seeds 40000-40047 "
            "and 50000-50011 are scored seeds of earlier tests: never run them.")


def _header(path):
    """The launcher's comment header, '#' stripped and whitespace collapsed."""
    lines = path.read_text().splitlines()
    body = [ln for ln in lines if ln.startswith("#") and not ln.startswith(("#!", "#SBATCH"))]
    return " ".join(" ".join(ln.lstrip("#").split()) for ln in body)


@pytest.mark.parametrize("name", ["cpu_navgym_v5.sbatch", "cpu_navgym_v5_transfer.sbatch"])
def test_v5_launchers_carry_the_frozen_rules_verbatim(name):
    import navgym_train as nt
    collapse = lambda s: " ".join(s.split())             # noqa: E731
    h = _header(SLURM / name)
    assert collapse(nt.V5_TEXT) in h and collapse(nt.V5_TRANSFER_RULE["text"]) in h and collapse(REPORTED) in h
    if name == "cpu_navgym_v5.sbatch":
        for part in V5_SPEC:
            assert collapse(part) in h, part[:60]


def _trainer_cmd(text):
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if 'navgym_train.py --out "$OUT"' in ln)
    return " ".join((lines[i] + " " + lines[i + 1]).replace("\\", " ").split())


def test_v5_launcher_runs_the_v4_command_with_only_the_v5_changes():
    v4 = (SLURM / "cpu_navgym_v4.sbatch").read_text()
    v5 = (SLURM / "cpu_navgym_v5.sbatch").read_text()
    assert _trainer_cmd(v5) == _trainer_cmd(v4).replace("--env-version 2", "--env-version 3").replace(
        "--final-eval-fresh-base 40000", "--final-eval-fresh-base 61000")
    assert "#SBATCH --array=8-10" in v5 and "STEPS=30000000" in v5 and 'case "$SEED" in 8|9|10)' in v5
    assert 'if [ -e "$OUT" ]; then' in v5 and "REFUSING" in v5                    # never overwrite a run directory
    assert 'verdict-v5 "$ROOT" --seeds 8 9 10 --final --out "$ROOT/verdict_v5.json"' in v5
    smoke = next(ln for ln in v5.splitlines() if "--seed 99" in ln) + next(ln for ln in v5.splitlines() if "--final-eval-fresh-base 90000" in ln)
    assert "--total-steps 50000" in smoke and "61000" not in smoke
    assert re.search(r"--seed-start 9150 .*--time-limit 30 .*--policy-capture-pose", v5.replace("\\\n", " "))
    t = (SLURM / "cpu_navgym_v5_transfer.sbatch").read_text()
    assert ('COMMON=(--upstream "$U" --seeds 12 --seed-start 63000 --n 6 --m 6 --extra-openings 1 --time-limit 180 --no-overwrite)' in t)
    assert '--policy "$V5/$a/actor.onnx" --policy-capture-pose' in t and "verdict-v5-transfer" in t and "--list-actors" in t
    assert 'for f in "$OUT/verdict.json" "$OUT/not_run.json"; do' in t and "GV=$V5/verdict_v5.json" in t
    # review 2026-10-02: an unfinished run's files stop the launcher before the gym verdict is read or anything runs, and
    # verdict.json is written only by the --final call after every episode run has exited
    assert t.index('find "$OUT" -mindepth 1 -print -quit') < t.index('[ -f "$GV" ]') < t.index('"$PY"')
    assert t.rstrip().splitlines()[-2] == '"$PY" scripts/bench/navgym_train.py verdict-v5-transfer "$OUT" --gym-verdict "$GV" --final'
    assert t.index("\nwait\n") < t.index('--gym-verdict "$GV" --final')
    assert "range term only" in _header(SLURM / "cpu_navgym_v5.sbatch") and "stereo-depth" in _header(SLURM / "cpu_navgym_v5_transfer.sbatch")
    for text in (v5, t):          # the scored sets of earlier tests are never run
        assert not re.search(r"--seed-start (40000|50000|60000|62000)|fresh-base (10000|20000|40000)", text)


def _run_transfer_launcher(v5_root, out):
    import os
    import subprocess
    env = {**os.environ, "NAVGYM_V5_ROOT": str(v5_root), "NAVGYM_V5_TRANSFER_OUT": str(out)}
    return subprocess.run(["bash", str(SLURM / "cpu_navgym_v5_transfer.sbatch")], env=env, capture_output=True, text=True,
                          timeout=120)


def test_v5_transfer_launcher_refuses_before_running_anything(tmp_path):
    """Review 2026-10-02: the launcher refuses (exit 3) on an existing verdict.json / not_run.json and on the files of a
    transfer run that did not finish, before it reads the gym verdict or runs anything; with no gym verdict it exits 4.
    No Python runs on these paths. The gym verdict is absent here, so a launcher that failed to refuse would exit 4."""
    launcher = (SLURM / "cpu_navgym_v5_transfer.sbatch").read_text()
    repo = re.search(r"^REPO=(\S+)$", launcher, re.M).group(1)
    if not Path(repo).is_dir():
        pytest.skip(f"the launcher's REPO {repo} is not on this machine")
    v5 = tmp_path / "v5"                                                          # holds no verdict_v5.json
    v5.mkdir()
    leftovers = ("verdict.json", "not_run.json", "armV5-s8/seed9150.json", "armV5-s8/summary.json", "astar/seed9150.json",
                 "armV5-s8.log")
    for k, rel in enumerate(leftovers):
        out = tmp_path / f"out{k}"
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text("{}")
        before = sorted(p.relative_to(out).as_posix() for p in out.rglob("*"))
        r = _run_transfer_launcher(v5, out)
        assert r.returncode == 3 and "REFUSING" in r.stdout, (rel, r.returncode, r.stdout, r.stderr)
        assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*")) == before, rel      # nothing written
        if k >= 2:
            assert "did not finish" in r.stdout and "--final" in r.stdout and "Nothing was run" in r.stdout, rel
    empty = tmp_path / "empty-out"            # an empty output directory holds no run's files: it passes the refusal
    empty.mkdir()
    for out in (empty, tmp_path / "absent-out"):
        r = _run_transfer_launcher(v5, out)
        assert r.returncode == 4 and "NO GYM VERDICT" in r.stdout, (r.returncode, r.stdout, r.stderr)
    assert not (tmp_path / "absent-out").exists() and not any(empty.iterdir())
