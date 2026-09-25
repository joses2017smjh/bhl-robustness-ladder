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
