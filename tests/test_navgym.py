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


# ------------------------------------------------------------------ 2026-09-28: v2 golden, v2ctl reading, v4 stall price
def _descend(env):
    """Oracle controller (tests only): steer down the continuous geodesic; reaches the goal on small mazes."""
    h = 0.05
    gx = env.geo(env.x + h, env.y) - env.geo(env.x - h, env.y)
    gy = env.geo(env.x, env.y + h) - env.geo(env.x, env.y - h)
    e = (math.atan2(-gy, -gx) - env.yaw + math.pi) % (2 * math.pi) - math.pi
    return np.array([1.0 if abs(e) < 0.35 else -1.0, max(-1.0, min(1.0, 2.0 * e))], np.float32)


# Recorded on 2026-09-28 from the v2 env at git HEAD 7c6ef05 (env.py sha256 c48ccd4e...), BEFORE the v4 stall-price
# edit: goal, crash, and stalled time-out episodes. MazeNavEnv(version=2) with no new keyword must reproduce them exactly.
V2_GOLDEN = [
    ("mz", dict(sizes=((3, 3),), version=2, gamma=1.0, randomize_dynamics=False), 0, "descend",
     dict(maze_seed=8506, t=421, outcome="goal", max_steps=1447, route_m=4.883171858868458, reward_sum=8.750247270277152,
          reward_last=5.011244706566285, x=2.797079898858059, y=2.5103371072756846, yaw=1.583930207436544, lidar_sum=1694.4702041689306,
          near_sum=9219.44812014699, map_sum=242496.0, goal_sum=744.3931784438901)),
    ("mz", dict(sizes=((4, 4),), version=2, gamma=1.0), 2, "descend",
     dict(maze_seed=8375, t=1020, outcome="goal", max_steps=3305, route_m=13.554978667474233, reward_sum=16.20856621338885,
          reward_last=5.008400665176615, x=4.197786272625983, y=3.908026584829612, yaw=1.572619374723427, lidar_sum=4128.594295883551,
          near_sum=22033.803452447057, map_sum=587520.0, goal_sum=1857.3569260610911)),
    ("mz", dict(sizes=((4, 4),), version=2, gamma=0.998), 5, "descend",
     dict(maze_seed=6707, t=514, outcome="goal", max_steps=1695, route_m=6.042969833513124, reward_sum=13.373006761652952,
          reward_last=5.011633654947392, x=4.052444860243608, y=3.9389648744830144, yaw=1.060831704313288, lidar_sum=2440.555840173736,
          near_sum=12511.52498409152, map_sum=296064.0, goal_sum=1020.2129253945313)),
    ("ho", dict(size=(5, 5), seed=3, gamma=0.998), None, "random",
     dict(maze_seed=10003, t=93, outcome="collision", max_steps=2647, route_m=10.483171857868472, reward_sum=-3.6834883439687562,
          reward_last=-5.002, x=-0.43801968080787884, y=-0.362683028686183, yaw=-2.409068470712409, lidar_sum=344.87228877842426,
          near_sum=1695.4588338062167, map_sum=53568.0, goal_sum=-20.911093686707318)),
    ("ho", dict(size=(6, 6), seed=4, gamma=1.0, max_steps=200), None, "stall",
     dict(maze_seed=10004, t=200, outcome="time_out", max_steps=200, route_m=12.849625289736696, reward_sum=-0.39068139687052095,
          reward_last=-0.0018572896830335566, x=0.013796420748577634, y=-0.04546934815832517, yaw=0.23882599197938026,
          lidar_sum=639.7048387341201, near_sum=3501.563729286194, map_sum=115200.0, goal_sum=270.9591631293297)),
]


def _run_golden(kind, kw, sd, pol, **extra):
    if kind == "mz":
        env = MazeNavEnv(**kw, **extra)
        obs, _ = env.reset(seed=sd)
    else:
        env, obs, _ = heldout_env(kw["size"], kw["seed"], version=2, gamma=kw["gamma"], max_steps=kw.get("max_steps"))
        for k, v in extra.items():
            setattr(env, k, v)
    rng = np.random.default_rng(77)
    rs, sums = [], {k: 0.0 for k in obs}
    while True:
        if pol == "descend":
            a = _descend(env)
        elif pol == "random":
            a = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)], np.float32)
        else:
            a = np.array([-1.0, 1.0 if env.t % 2 else -1.0], np.float32)
        obs, r, te, tr, st = env.step(a)
        rs.append(r)
        for k in obs:
            sums[k] += float(np.asarray(obs[k], np.float64).sum())
        if te or tr:
            break
    got = dict(maze_seed=int(st["maze_seed"]), t=int(env.t), outcome=st["outcome"], max_steps=int(env.max_steps), route_m=float(env.route_m),
               reward_sum=float(np.sum(rs)), reward_last=float(rs[-1]), x=float(env.x), y=float(env.y), yaw=float(env.yaw),
               **{f"{k}_sum": v for k, v in sums.items()})
    return env, got, rs


@pytest.mark.parametrize("kind,kw,sd,pol,gold", V2_GOLDEN)
def test_v2_is_reproduced_exactly(kind, kw, sd, pol, gold):
    _, got, _ = _run_golden(kind, kw, sd, pol)
    for k, v in gold.items():
        if isinstance(v, float):
            assert got[k] == pytest.approx(v, rel=1e-12, abs=1e-12), k
        else:
            assert got[k] == v, k


def _v2ctl_summ(s5, c6, ppo="v2", ceiling=None, steps=30_000_000, base=40_000, s6=0.8):
    ev = {"4x4": {"success": 1.0, "collision": 0.0, "time_out": 0.0, "n": 48},
          "5x5": {"success": s5, "collision": 0.0, "time_out": 1 - s5, "n": 48},
          "6x6": {"success": s6, "collision": c6, "time_out": 0.0, "n": 48}}
    return {"ppo": ppo, "env_version": 2, "std_ceiling": ceiling, "total_steps": 30_000_000, "num_timesteps": steps,
            "reached_last_step": steps >= 30_000_000, "final_eval_heldout_48": ev, "final_eval_fresh_48": ev,
            "fresh_set": {"maze_seeds": f"{base} + k, k = 0..47"}}


def test_v2ctl_determinism_check_cuts_at_the_first_clamp():
    import navgym_train as nt
    twin = [(4096 * i, float(i)) for i in range(1, 11)]
    clamps = [(4096 * i, 0.0 if i < 6 else float(i - 5)) for i in range(1, 11)]       # first logged clamp at 6 * 4096
    same_before = [(s, v) if s < 6 * 4096 else (s, v + 3.0) for s, v in twin]            # differs only from the clamp on
    r = nt.determinism_check(same_before, twin, clamps)
    assert r["result"] == "single-factor comparison" and r["points_compared"] == 5 and r["first_ceiling_clamp_step"] == 6 * 4096
    off = [(s, v + (1e-3 if s == 3 * 4096 else 0.0)) for s, v in twin]
    r = nt.determinism_check(off, twin, clamps)
    assert r["result"] == "seed-matched replicate" and r["first_divergence_step"] == 3 * 4096
    missing = [p for p in twin if p[0] != 2 * 4096]
    assert nt.determinism_check(missing, twin, clamps)["first_divergence_step"] == 2 * 4096
    assert nt.determinism_check(None, twin, clamps)["result"] == "no result"


def test_v2ctl_reading_rule(tmp_path):
    import navgym_train as nt
    own, v3 = tmp_path / "v2ctl", tmp_path / "v3"

    def write(root, d, s):
        (root / d).mkdir(parents=True, exist_ok=True)
        (root / d / "summary.json").write_text(json.dumps(s))
    # v3 twins: 6x6 collision 0.25 / 0.25 / 0.25 on the 10000-set
    for sd in (2, 3, 4):
        write(v3, f"armV3-s{sd}", _v2ctl_summ(0.5, 12 / 48, ppo="v3", ceiling=1.0, base=20_000))
    R = lambda: nt.reading_v2ctl(own, v3, None, tb=False)   # noqa: E731
    assert R()["b"]["reading"].startswith("no result") and R()["c"]["reading"].startswith("no result")
    write(own, "armV2C-s2", _v2ctl_summ(14 / 48, 7 / 48))       # 0.292 < 0.30; collision lower by 5/48 = 0.104
    write(own, "armV2C-s3", _v2ctl_summ(0.9, 7 / 48))
    r = R()
    assert r["b"]["reading"] == "the collapse is not caused by the ceiling"
    assert r["c"]["reading"] == "the ceiling raised collisions" and r["c"]["n_lower_by_0.10"] == 2
    write(own, "armV2C-s2", _v2ctl_summ(29 / 48, 8 / 48))       # 0.604 >= 0.60; lower by 4/48 = 0.083 (neither)
    write(own, "armV2C-s3", _v2ctl_summ(0.9, 10 / 48))          # within 2/48
    write(own, "armV2C-s4", _v2ctl_summ(0.9, 14 / 48))          # within 2/48 (higher)
    r = R()
    assert r["b"]["reading"] == "the ceiling caused or deepened it"
    assert r["c"]["reading"] == "the ceiling did not raise collisions" and r["c"]["n_within_0.05"] == 2
    write(own, "armV2C-s2", _v2ctl_summ(0.5, 9 / 48))           # 0.30 <= 0.5 < 0.60; lower by 3/48 (neither)
    write(own, "armV2C-s4", _v2ctl_summ(0.9, 6 / 48))           # lower by 6/48
    r = R()
    assert r["b"]["reading"] == "inconclusive" and r["c"]["reading"] == "inconclusive"
    # a v3-configured or unfinished run is not a valid v2ctl seed
    write(own, "armV2C-s2", _v2ctl_summ(0.1, 0.0, ppo="v3", ceiling=1.0))
    write(own, "armV2C-s3", _v2ctl_summ(0.9, 0.0, steps=29_000_000))
    r = R()
    assert r["b"]["reading"].startswith("no result") and not r["per_seed"]["3"]["valid"]
    write(own, "armV2C-s4", _v2ctl_summ(0.9, 0.0, base=20_000))                 # evaluated on the spent 20000-set
    assert not R()["per_seed"]["4"]["valid"]


from bhl_robust.navgym.env import V2_COLLISION, V2_GOAL, V4_IDLE_A0_MAX, V4_IDLE_COST, V4_IDLE_FAR_M


def test_v4_flags_default_off_and_v2_golden_with_explicit_zero():
    """New keywords default off: MazeNavEnv / make_env / the trainer's PPO settings are unchanged, and passing
    idle_cost=0.0 explicitly reproduces the pre-edit v2 golden bit for bit."""
    import navgym_train as nt
    assert MazeNavEnv().idle_cost == 0.0 and MazeNavEnv(version=2).idle_cost == 0.0
    assert set(nt.PPO_SETTINGS) == {"v1", "v2", "v3"}
    env = nt.make_env(0, 0, ((3, 3),), version=2, gamma=1.0)()
    assert env.idle_cost == 0.0
    ap = nt.build_parser()
    a = ap.parse_args(["--out", "x"])
    assert a.idle_cost is None and a.env_version == 1 and a.ppo == "v1" and a.final_eval_fresh_base is None
    for kind, kw, sd, pol, gold in V2_GOLDEN:
        if kind != "mz":
            continue
        _, got, _ = _run_golden(kind, kw, sd, pol, idle_cost=0.0)
        for k, v in gold.items():
            assert got[k] == (pytest.approx(v, rel=1e-12, abs=1e-12) if isinstance(v, float) else v), k
    with pytest.raises(ValueError):
        MazeNavEnv(version=1, idle_cost=0.008)                       # v1 is never priced
    with pytest.raises(ValueError):
        MazeNavEnv(version=2, idle_cost=-0.008)


def test_v4_stall_cost_arithmetic():
    gamma = 0.998
    # an endless idle stall costs what a crash costs, at the learner's gamma (truncations are bootstrapped)
    assert (V2_STEP_COST - V4_IDLE_COST) / (1 - gamma) == pytest.approx(V2_COLLISION, abs=1e-9)
    assert V2_STEP_COST / (1 - gamma) == pytest.approx(-1.0, abs=1e-9)          # v2: the cheap stall being priced
    t = np.arange(20_000)
    assert float(np.sum((V2_STEP_COST - V4_IDLE_COST) * gamma ** t)) == pytest.approx(-5.0, abs=1e-6)
    # V4_IDLE_A0_MAX is "forward command ~0": v_cmd = (a + 1) / 2 * V_MAX <= 5 % of V_MAX
    assert (V4_IDLE_A0_MAX + 1.0) * 0.5 * V_MAX == pytest.approx(0.05 * V_MAX)

    def fresh(cost):
        env, _, _ = heldout_env((3, 3), 2, version=2, gamma=1.0)
        env.idle_cost = cost
        env.dyn.v_noise = env.dyn.w_noise = 0.0
        env.dyn.drift = 0.0
        env.v = env.w = 0.0
        env.queue = [np.zeros(2) for _ in env.queue]
        return env
    # a stall step far from the goal: step cost + idle cost, charged and flagged
    env = fresh(V4_IDLE_COST)
    assert math.hypot(env.goal_xy[0] - env.x, env.goal_xy[1] - env.y) > V4_IDLE_FAR_M
    _, r, te, tr, info = env.step(np.array([-1.0, 0.7], np.float32))
    assert not (te or tr) and r == pytest.approx(V2_STEP_COST - V4_IDLE_COST, abs=1e-12) and info["idle_charged"] is True
    # below the -0.9 threshold is idle, above it is not (float32(-0.9) itself rounds to just above -0.9 in double
    # precision, so the boundary value is deliberately not tested; the pilot compared exactly as the env does)
    for a0, charged in ((-1.0, True), (-0.95, True), (-0.85, False), (1.0, False)):
        e1, e0 = fresh(V4_IDLE_COST), fresh(0.0)
        _, r1, *_, i1 = e1.step(np.array([a0, 0.0], np.float32))
        _, r0, *_, i0 = e0.step(np.array([a0, 0.0], np.float32))
        assert i1["idle_charged"] is charged and "idle_charged" not in i0
        assert r1 == pytest.approx(r0 - (V4_IDLE_COST if charged else 0.0), abs=1e-12)
    # within V4_IDLE_FAR_M of the goal a stall is not charged
    env = fresh(V4_IDLE_COST)
    env.x, env.y = env.goal_xy[0] + 0.38, env.goal_xy[1]
    env.potential = env.geo(env.x, env.y)
    _, r, te, tr, info = env.step(np.array([-1.0, 0.0], np.float32))
    assert info["idle_charged"] is False and r == pytest.approx(V2_STEP_COST, abs=1e-12)


def test_v4_idle_cost_changes_only_idle_steps_of_a_goal_episode():
    """The oracle reaches the goal: with the idle cost every reward equals the unpriced one minus exactly
    V4_IDLE_COST on the steps it charges, the trajectory is identical, and the goal step is never charged."""
    kind, kw, sd, pol, _ = V2_GOLDEN[1]
    e0, g0, r0 = _run_golden(kind, kw, sd, pol)
    e1, g1, r1 = _run_golden(kind, kw, sd, pol, idle_cost=V4_IDLE_COST)
    assert g0["outcome"] == g1["outcome"] == "goal" and g0["t"] == g1["t"] and (g0["x"], g0["y"]) == (g1["x"], g1["y"])
    d = np.asarray(r1) - np.asarray(r0)
    charged = np.isclose(d, -V4_IDLE_COST, atol=1e-12)
    assert np.all(charged | (d == 0.0)) and charged.sum() > 0 and not charged[-1]
    assert r0[-1] > V2_GOAL - 0.1


def test_v4_verdict_rule(tmp_path):
    import navgym_train as nt

    def summ(s5, s6, c6, steps=30_000_000, n=48, price="frozen", ppo="v2", base=40_000):
        ev = {"4x4": {"success": 1.0, "collision": 0.0, "time_out": 0.0, "n": n},
              "5x5": {"success": s5, "collision": 0.0, "time_out": 1 - s5, "n": n},
              "6x6": {"success": s6, "collision": c6, "time_out": 0.0, "n": n}}
        sp = dict(nt.V4_STALL_PRICE) if price == "frozen" else price
        return {"ppo": ppo, "env_version": 2, "std_ceiling": None, "total_steps": 30_000_000, "num_timesteps": steps,
                "reached_last_step": steps >= 30_000_000, "final_eval_fresh_48": ev, "final_eval_heldout_48": ev,
                "fresh_set": {"maze_seeds": f"{base} + k, k = 0..47"}, "stall_price": sp}

    def write(sd, s):
        d = tmp_path / f"armV4-s{sd}"
        d.mkdir(exist_ok=True)
        (d / "summary.json").write_text(json.dumps(s))
    assert nt.judge_v4(summ(0.80, 0.70, 0.15))["meets"]                             # every bound inclusive
    assert not nt.judge_v4(summ(0.79, 0.90, 0.0))["meets"]
    assert not nt.judge_v4(summ(0.90, 0.69, 0.0))["meets"]
    assert not nt.judge_v4(summ(0.90, 0.90, 0.16))["meets"]
    assert not nt.judge_v4(summ(0.90, 0.90, 0.0, steps=29_999_000))["meets"]
    assert not nt.judge_v4(summ(0.90, 0.90, 0.0, n=24))["meets"]
    assert not nt.judge_v4(summ(0.90, 0.90, 0.0, price=None))["meets"]               # a plain v2 run is not v4
    assert not nt.judge_v4(summ(0.90, 0.90, 0.0, price={**nt.V4_STALL_PRICE, "idle_cost": 0.004}))["meets"]
    assert not nt.judge_v4(summ(0.90, 0.90, 0.0, ppo="v3"))["meets"]
    assert not nt.judge_v4(summ(0.90, 0.90, 0.0, base=20_000))["meets"]             # the spent set
    write(5, summ(0.85, 0.75, 0.05))
    assert nt.verdict_v4(tmp_path)["verdict"] == "PENDING"
    assert nt.verdict_v4(tmp_path, final=True)["verdict"] == "NEGATIVE"
    write(6, summ(0.70, 0.75, 0.05))
    write(7, summ(0.81, 0.71, 0.10))
    v = nt.verdict_v4(tmp_path, final=True)
    assert v["verdict"] == "PASS" and v["seeds_passing"] == 2
    write(7, summ(0.81, 0.71, 0.20))
    assert nt.verdict_v4(tmp_path, final=True)["verdict"] == "NEGATIVE"
