import math
import numpy as np

from bhl_robust.navgym.env import (EgoMap, MazeNavEnv, cast_rays, geodesic_distance, geodesic_field, wall_boxes)
from bhl_robust.eval.random_maze import CELL, generate


def test_rays_hit_the_first_wall():
    boxes = np.array([[1.0, 1.1, -2.0, 2.0]])          # a wall slab at x = 1.0 .. 1.1
    d = cast_rays(boxes, (0.0, 0.0), np.array([0.0, math.pi / 2, math.pi]), 12.0)
    assert abs(d[0] - 1.0) < 1e-6 and d[1] == 12.0 and d[2] == 12.0


def test_env_obs_spaces_and_episode():
    env = MazeNavEnv(sizes=((3, 3),), randomize_dynamics=False)
    obs, info = env.reset(seed=3)
    assert env.observation_space.contains({k: np.asarray(v, dtype=np.float32) for k, v in obs.items()})
    assert obs["map"].shape == (3, 24, 24) and np.allclose(obs["map"].sum(axis=0), 1.0)
    # driving into the nearest wall ends the episode with the collision penalty
    env.yaw = float(env.yaw + env.angles[int(np.argmin(env.ranges))])
    env._scan()
    total, ended = 0.0, None
    for _ in range(200):
        obs, r, term, trunc, info = env.step(np.array([1.0, 0.0]))
        total += r
        if term or trunc:
            ended = info["outcome"]
            break
    assert ended == "collision" and total < 0


def test_geodesic_distance_decreases_toward_goal():
    mz = generate(4, 4, seed=0)
    field = geodesic_field(mz)
    sol = mz.solution()
    ds = [geodesic_distance(mz, field, *mz.centre(c)) for c in sol]
    assert all(a > b for a, b in zip(ds, ds[1:])) and ds[-1] < 1e-6


def test_egomap_marks_free_then_occupied_and_crops_forward_up():
    m = EgoMap((0, 4, 0, 4), margin=0.5)
    angles = np.array([0.0])
    for _ in range(4):
        m.update(0.5, 2.0, 0.0, angles, np.array([2.0]))
    crop = m.crop(0.5, 2.0, 0.0)
    occ, free, unk = crop
    # the wall 2 m ahead is in the upper half (rows < 12), the free ray below it
    assert occ[:12].sum() >= 1 and occ[12:].sum() == 0 and free[6:12, 12].sum() >= 4


# ------------------------------------------------------------------ NavGym v2
from pathlib import Path

import pytest

from bhl_robust.navgym.env import (OBS_KEYS_V1, OBS_KEYS_V2, ROBOT_RADIUS, V2_COLLISION, V2_MAX_STEPS_CAP, V2_STEP_COST, DT, V_MAX,
                                   box_clearance, build_obs, circle_in_wall, fine_geodesic, heldout_env, point_in_wall,
                                   route_time_limit)

REPO = Path(__file__).resolve().parents[1]


def test_round_footprint_clears_corners_the_square_clipped():
    boxes = np.array([[0.0, 1.0, 0.0, 1.0]])
    # 0.25 m from the corner along the diagonal: clear of a 0.22 m disc, inside v1's 0.22 m square
    x = y = 1.0 + 0.25 / math.sqrt(2)
    assert not circle_in_wall(boxes, x, y, ROBOT_RADIUS) and point_in_wall(boxes, x, y, ROBOT_RADIUS)
    assert abs(float(box_clearance(boxes, x, y)) - 0.25) < 1e-12
    # the square clipped out to the corner distance 0.22 * sqrt(2) = 0.311 m
    d = 0.30 / math.sqrt(2)
    assert point_in_wall(boxes, 1.0 + d, 1.0 + d, ROBOT_RADIUS) and not circle_in_wall(boxes, 1.0 + d, 1.0 + d, ROBOT_RADIUS)
    # facing a wall face both agree: 0.21 m collides, 0.23 m does not
    assert circle_in_wall(boxes, 1.21, 0.5, ROBOT_RADIUS) and point_in_wall(boxes, 1.21, 0.5, ROBOT_RADIUS)
    assert not circle_in_wall(boxes, 1.23, 0.5, ROBOT_RADIUS) and not point_in_wall(boxes, 1.23, 0.5, ROBOT_RADIUS)


# Recorded on 2026-09-26 from the v1 env at git HEAD 0a2c935 (before the v2 edit), same kwargs,
# seeds and action sequence: MazeNavEnv(version=1) -- the default -- must reproduce them exactly.
V1_GOLDEN = [
    (dict(sizes=((3, 3),), randomize_dynamics=False), 3, 400,
     dict(maze_seed=8115, t=351, outcome="collision", reward_sum=-2.7334952401095514, reward_last=-2.01, x=3.238052853455598,
          y=-0.034839756613299355, yaw=-0.27805736851448826, lidar_sum=1390.82437521033, map_sum=202176.0, goal_sum=649.7769218393369)),
    (dict(sizes=((4, 4), (5, 5)), randomize_dynamics=True), 7, 400,
     dict(maze_seed=6250, t=54, outcome="collision", reward_sum=-2.457062334817828, reward_last=-2.01, x=0.1417366558694708,
          y=-0.43688301632678034, yaw=-1.1988412173412348, lidar_sum=158.9484852720052, map_sum=31104.0, goal_sum=87.04779870482162)),
    (dict(sizes=((6, 6),), randomize_dynamics=True, seed_base=10_005, seed_span=1), 5, 1600,
     dict(maze_seed=10005, t=52, outcome="collision", reward_sum=-2.931486071732419, reward_last=-2.01, x=-0.43419953989046656,
          y=0.06079181392301855, yaw=2.998776482190901, lidar_sum=162.2652260493487, map_sum=29952.0, goal_sum=-1.7190331215970218)),
    (dict(sizes=((5, 5),), randomize_dynamics=True, max_steps=60), 11, 400,
     dict(maze_seed=1338, t=60, outcome="time_out", reward_sum=-0.23066102491179175, reward_last=-0.004870338413658573,
          x=0.41703241873275987, y=0.3160324196051014, yaw=0.4290759174954899, lidar_sum=181.33160679414868, map_sum=34560.0,
          goal_sum=125.5937615763396)),
]


@pytest.mark.parametrize("kwargs,seed,n_steps,gold", V1_GOLDEN)
def test_v1_is_reproduced_exactly(kwargs, seed, n_steps, gold):
    env = MazeNavEnv(**kwargs)
    assert env.version == 1 and env.max_steps == kwargs.get("max_steps", 1500)
    obs, info = env.reset(seed=seed)
    assert set(obs) == set(OBS_KEYS_V1) and set(info) == {"maze_seed", "size"}
    rng = np.random.default_rng(1234 + seed)
    rs, lid, mp, gl, out = [], 0.0, 0.0, 0.0, None
    for _ in range(n_steps):
        a = np.array([rng.uniform(-0.2, 1.0), rng.uniform(-1, 1)], np.float32)
        obs, r, term, trunc, info = env.step(a)
        rs.append(r); lid += float(obs["lidar"].astype(np.float64).sum()); mp += float(obs["map"].astype(np.float64).sum())
        gl += float(obs["goal"].astype(np.float64).sum())
        if term or trunc:
            out = info["outcome"]
            break
    got = dict(maze_seed=info["maze_seed"], t=env.t, outcome=out, reward_sum=float(np.sum(rs)), reward_last=float(rs[-1]),
               x=float(env.x), y=float(env.y), yaw=float(env.yaw), lidar_sum=lid, map_sum=mp, goal_sum=gl)
    for k, v in gold.items():
        if isinstance(v, float):
            assert got[k] == pytest.approx(v, rel=1e-12, abs=1e-12), k
        else:
            assert got[k] == v, k


def _doorway_segments(mz):
    """Every open internal edge as a (centre, centre) segment: each crosses one doorway."""
    segs = []
    for e in mz.open_edges:
        a, b = tuple(e)
        segs.append((np.array(mz.centre(a)), np.array(mz.centre(b))))
    return segs


def test_v2_potential_is_continuous_at_doorways_where_v1_jumps():
    worst_v1, worst_v2 = 0.0, 0.0
    for n, seed in ((4, 10_000), (5, 10_001), (6, 10_003), (6, 10_021)):
        mz = generate(n, n, seed, extra_openings=1)
        f = fine_geodesic(mz)
        fld = geodesic_field(mz)
        for pa, pb in _doorway_segments(mz):
            k = int(round(np.linalg.norm(pb - pa) / 0.02))
            pts = [pa + (pb - pa) * i / k for i in range(k + 1)]
            g2 = np.array([f(*p) for p in pts])
            g1 = np.array([geodesic_distance(mz, fld, *p) for p in pts])
            worst_v2 = max(worst_v2, float(np.abs(np.diff(g2)).max()))
            worst_v1 = max(worst_v1, float(np.abs(np.diff(g1)).max()))
    assert worst_v2 <= 0.1, worst_v2               # a 2 cm step changes the v2 potential by ~2 cm
    assert worst_v1 > 0.3                           # the v1 cell-hop potential jumps here (the fault being fixed)


def test_v2_potential_is_metric_and_zero_at_goal():
    mz = generate(3, 3, 1, extra_openings=1)
    f = fine_geodesic(mz)
    gx, gy = mz.centre(mz.goal)
    assert f(gx, gy) < 1e-6
    for dx, dy in ((0.4, 0.0), (0.3, 0.3), (0.0, -0.5)):         # open cell: geodesic = Euclidean within 3 %
        assert f(gx + dx, gy + dy) == pytest.approx(math.hypot(dx, dy), rel=0.03)
    assert fine_geodesic(mz) is f                                   # cached per maze


def test_v2_time_limit_scales_with_route_and_is_capped():
    assert route_time_limit(0.0) == 400
    assert route_time_limit(10.0) == math.ceil(3.0 * 10.0 / (V_MAX * DT)) + 400
    assert route_time_limit(100.0) == V2_MAX_STEPS_CAP == 4500
    lims = []
    for n in (3, 6):
        env = MazeNavEnv(sizes=((n, n),), version=2)
        _, info = env.reset(seed=4)
        assert info["version"] == 2 and env.max_steps == info["max_steps"] == route_time_limit(info["route_m"])
        assert info["route_m"] > (n - 1) * CELL * 0.9                 # at least ~ the Manhattan corner-to-corner lower bound
        lims.append(env.max_steps)
    assert lims[0] < lims[1]
    env = MazeNavEnv(sizes=((3, 3),), version=2, max_steps=77)
    env.reset(seed=0)
    assert env.max_steps == 77                                      # an explicit int is a flat limit
    assert MazeNavEnv().max_steps == 1500                           # v1 default untouched


def test_v2_rewards_crash_stall_and_shaping_gamma():
    env, obs, info = heldout_env((3, 3), 2, version=2, gamma=0.998)
    assert set(obs) == set(OBS_KEYS_V2) and env.observation_space.contains({k: np.asarray(v, np.float32) for k, v in obs.items()})
    # a stall step (no motion commanded, zero noise): reward = step cost + (1 - gamma) * geodesic
    env.dyn.v_noise = env.dyn.w_noise = 0.0
    env.dyn.drift = 0.0
    env.v = env.w = 0.0
    env.queue = [np.zeros(2) for _ in env.queue]
    g0 = env.potential
    _, r, term, trunc, _ = env.step(np.array([-1.0, 0.0], np.float32))
    assert not term and r == pytest.approx(V2_STEP_COST + (1 - 0.998) * g0, abs=1e-9)
    # drive into the nearest wall: the crash step pays exactly step cost + collision, no shaping
    env.yaw = float(env.yaw + env.angles[int(np.argmin(env.ranges))])
    env._scan()
    for _ in range(400):
        x0, y0 = env.x, env.y
        _, r, term, trunc, info = env.step(np.array([1.0, 0.0], np.float32))
        if term or trunc:
            break
    assert info["outcome"] == "collision" and r == pytest.approx(V2_STEP_COST + V2_COLLISION, abs=1e-12)
    assert (env.x, env.y) == (x0, y0)                              # the robot did not move into the wall
    assert float(box_clearance(env.boxes, env.x, env.y)) >= ROBOT_RADIUS


def test_v2_obs_come_from_the_shared_builder():
    env, obs, _ = heldout_env((4, 4), 0, version=2)
    for _ in range(20):
        obs, *_ = env.step(np.array([0.2, 0.3], np.float32))
    ref = build_obs(OBS_KEYS_V2, env.ranges, env.emap, env.x, env.y, env.yaw, env.goal_xy, env.prev_action)
    for k in OBS_KEYS_V2:
        assert np.array_equal(obs[k], ref[k]), k
    # near = sector minima clipped at 2 m / 2 m; lidar = the same at 12 m / 12 m
    assert np.allclose(obs["near"], np.minimum(obs["lidar"] * 12.0, 2.0) / 2.0, atol=1e-6)


def test_v1_onnx_actor_still_runs_through_the_runner_builder():
    """maze_explore.py --policy builds the feeds from the actor's input names with build_obs;
    the published v1 actor (lidar, map, goal) must still run."""
    ort = pytest.importorskip("onnxruntime")
    p = REPO / "results/navgym-20260924/run-0/actor.onnx"
    if not p.is_file():
        pytest.skip("v1 actor not present")
    sess = ort.InferenceSession(str(p))
    keys = tuple(i.name for i in sess.get_inputs())
    assert keys == OBS_KEYS_V1
    env = MazeNavEnv(sizes=((4, 4),))
    env.reset(seed=0)
    obs = build_obs(keys, env.ranges, env.emap, env.x, env.y, env.yaw, env.goal_xy, np.zeros(2, np.float32))
    old = env._obs()                                                # the v1 env's own observation
    for k in keys:
        assert np.array_equal(obs[k], old[k]), k
    act = sess.run(None, {k: np.asarray(v, np.float32)[None] for k, v in obs.items()})[0]
    assert act.shape == (1, 2) and np.isfinite(act).all()


def test_heldout_reset_reuses_an_env_for_the_same_episode():
    """The trainer's parallel evaluation pool reuses worker envs via heldout_reset; each must
    replay exactly the episode heldout_env builds (same maze, dynamics, start, limit, rollout)."""
    worker = MazeNavEnv(sizes=((3, 3),), version=2, gamma=0.998)
    worker.reset(seed=123)
    for size, sd in (((5, 5), 3), ((6, 6), 7)):
        ref, o_ref, _ = heldout_env(size, sd, version=2, gamma=0.998)
        o = worker.heldout_reset(size, sd)
        assert worker.episode_seed == ref.episode_seed and worker.dyn == ref.dyn and worker.max_steps == ref.max_steps
        rng = np.random.default_rng(sd)
        for _ in range(50):
            for k in OBS_KEYS_V2:
                assert np.array_equal(o[k], o_ref[k]), k
            a = rng.uniform(-1, 1, 2).astype(np.float32)
            o, r, te, tr, _ = worker.step(a)
            o_ref, r_ref, te_ref, tr_ref, _ = ref.step(a)
            assert r == r_ref and te == te_ref and tr == tr_ref
            if te or tr:
                break


def test_v2_progress_form_pays_nothing_for_stalling():
    """The v2 trainer uses shaping gamma 1.0: a stall step pays exactly the step cost."""
    env, obs, info = heldout_env((3, 3), 2, version=2, gamma=1.0)
    env.dyn.v_noise = env.dyn.w_noise = 0.0
    env.dyn.drift = 0.0
    env.v = env.w = 0.0
    env.queue = [np.zeros(2) for _ in env.queue]
    _, r, term, trunc, _ = env.step(np.array([-1.0, 0.0], np.float32))
    assert not term and r == pytest.approx(V2_STEP_COST, abs=1e-9)


# ------------------------------------------------------------------ NavGym v3 (2026-09-27)
import json
import sys

from bhl_robust.navgym.env import FRESH_BASE, HELDOUT_BASE, heldout_bases

sys.path.insert(0, str(REPO / "scripts" / "bench"))


def test_heldout_default_set_is_unchanged_and_fresh_set_is_separate():
    """Default heldout_env is the published set (maze 10 000 + k, dynamics reset(seed=k)); the v3
    fresh set is maze 20 000 + k with dynamics reset(seed=20 000 + k); a pool worker replays both."""
    assert heldout_bases() == (10_000, 0) and heldout_bases(FRESH_BASE) == (20_000, 20_000) and HELDOUT_BASE == 10_000
    for sd in (0, 5):
        env, o, info = heldout_env((5, 5), sd, version=2, gamma=1.0)
        old = MazeNavEnv(sizes=((5, 5),), randomize_dynamics=True, seed_base=10_000 + sd, seed_span=1, version=2, gamma=1.0)
        o_old, info_old = old.reset(seed=sd)
        assert info["maze_seed"] == 10_000 + sd and env.dyn == old.dyn and info == info_old
        for k in OBS_KEYS_V2:
            assert np.array_equal(o[k], o_old[k])
        fr, of, fi = heldout_env((5, 5), sd, version=2, gamma=1.0, maze_base=FRESH_BASE)
        assert fi["maze_seed"] == 20_000 + sd
        from bhl_robust.navgym.env import sample_dynamics
        rng = np.random.default_rng(20_000 + sd)
        rng.integers(1)                                   # the size draw of reset()
        rng.integers(1)                                   # the maze-seed draw (span 1)
        assert fr.dyn == sample_dynamics(rng) and fr.dyn != env.dyn
        worker = MazeNavEnv(sizes=((3, 3),), version=2, gamma=1.0)
        worker.reset(seed=99)
        ow = worker.heldout_reset((5, 5), sd, FRESH_BASE)
        assert worker.episode_seed == 20_000 + sd and worker.dyn == fr.dyn and worker.max_steps == fr.max_steps
        for k in OBS_KEYS_V2:
            assert np.array_equal(ow[k], of[k])


def _tiny_ppo(ppo_key):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv
    import navgym_train as nt
    venv = DummyVecEnv([lambda: MazeNavEnv(sizes=((3, 3),), version=2, gamma=1.0)])
    P = nt.PPO_SETTINGS[ppo_key]
    model = PPO("MultiInputPolicy", venv, n_steps=8, batch_size=8, device="cpu",
                policy_kwargs={"features_extractor_class": nt.NavExtractor, "log_std_init": P["log_std_init"]})
    model.set_logger(configure(None, [""]))
    return nt, model, P


def test_v3_is_v2_plus_a_std_ceiling_only():
    import navgym_train as nt
    v2, v3 = nt.PPO_SETTINGS["v2"], nt.PPO_SETTINGS["v3"]
    assert "std_ceiling" not in v2 and "std_ceiling" not in nt.PPO_SETTINGS["v1"]
    assert {k: v for k, v in v3.items() if k != "std_ceiling"} == v2 and v3["std_ceiling"] == 1.0


def test_std_clamp_ceiling_binds_in_v3_and_not_in_v2():
    import torch
    for key, expect in (("v2", [math.exp(2.0)]), ("v3", [1.0])):
        nt, model, P = _tiny_ppo(key)
        cb = nt.StdFloor(P["std_floor"], P.get("std_ceiling"))
        cb.init_callback(model)
        with torch.no_grad():
            model.policy.log_std.copy_(torch.tensor([2.0, -3.0]))
        cb._clamp()
        std = torch.exp(model.policy.log_std.detach()).tolist()
        # the floor lifts -3 to log 0.2 in both; only v3 caps +2 at log 1.0
        assert std[1] == pytest.approx(0.2, rel=1e-6)
        assert std[0] == pytest.approx(expect[0], rel=1e-6)
        assert cb.last_std_pre_clamp_per_dim == pytest.approx([math.exp(2.0), math.exp(-3.0)], rel=1e-6)
        assert cb.ceiling_clamps == (1 if key == "v3" else 0) and cb.clamps == 1


def _summ(s5, s6, c6, steps=30_000_000, n=48):
    ev = {"4x4": {"success": 1.0, "collision": 0.0, "time_out": 0.0, "n": n},
          "5x5": {"success": s5, "collision": 0.0, "time_out": 1 - s5, "n": n},
          "6x6": {"success": s6, "collision": c6, "time_out": 0.0, "n": n}}
    return {"ppo": "v3", "env_version": 2, "std_ceiling": 1.0, "total_steps": 30_000_000, "num_timesteps": steps,
            "reached_last_step": steps >= 30_000_000, "final_eval_fresh_48": ev, "final_eval_heldout_48": ev,
            "fresh_set": {"maze_seeds": "20000 + k, k = 0..47"}}


def test_v3_verdict_rule(tmp_path):
    import navgym_train as nt
    def write(sd, s):
        d = tmp_path / f"armV3-s{sd}"
        d.mkdir(exist_ok=True)
        (d / "summary.json").write_text(json.dumps(s))
    assert nt.judge_v3(_summ(0.80, 0.70, 0.15))["meets"]                 # every bound inclusive
    assert not nt.judge_v3(_summ(0.79, 0.90, 0.0))["meets"]
    assert not nt.judge_v3(_summ(0.90, 0.69, 0.0))["meets"]
    assert not nt.judge_v3(_summ(0.90, 0.90, 0.16))["meets"]
    assert not nt.judge_v3(_summ(0.90, 0.90, 0.0, steps=29_999_000))["meets"]   # did not reach the last step
    assert not nt.judge_v3(_summ(0.90, 0.90, 0.0, n=24))["meets"]
    assert not nt.judge_v3({**_summ(0.90, 0.90, 0.0), "ppo": "v2"})["meets"]
    write(2, _summ(0.85, 0.75, 0.05))
    assert nt.verdict_v3(tmp_path)["verdict"] == "PENDING"
    assert nt.verdict_v3(tmp_path, final=True)["verdict"] == "NEGATIVE"      # missing runs count as misses at the end
    write(3, _summ(0.70, 0.75, 0.05))
    assert nt.verdict_v3(tmp_path)["verdict"] == "PENDING"
    write(4, _summ(0.81, 0.71, 0.10))
    v = nt.verdict_v3(tmp_path, final=True)
    assert v["verdict"] == "PASS" and v["seeds_passing"] == 2
    write(4, _summ(0.81, 0.71, 0.20))
    assert nt.verdict_v3(tmp_path, final=True)["verdict"] == "NEGATIVE"


def test_diagnose_timeout_classes_follow_the_declared_order():
    import navgym_diagnose as nd
    assert nd.classify_timeout(0.1, 0.0, 1.0) == "stalled"                 # stalled wins over wall
    assert nd.classify_timeout(1.0, 0.1, 0.6) == "stuck_at_wall"
    assert nd.classify_timeout(1.0, 0.1, 0.1) == "looping"
    assert nd.classify_timeout(1.0, 0.6, 0.9) == "slow_but_progressing"
