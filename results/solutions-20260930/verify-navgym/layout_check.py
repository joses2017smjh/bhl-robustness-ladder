"""verify-navgym: does the gym replay build the SAME maze layouts as the physics transfer?
Builds mazes only (no policy is run, nothing is scored). Compares
sha256(rm.world_xml(maze, textured=False))[:16] -- the exact expression of maze_explore.py:891 --
for (a) rm.generate(6, 6, seed, extra_openings=1) and (b) the maze MazeNavEnv builds with the
constructor arguments and reset seeds that gym_replay.py used, against the 'maze.layout_sha256'
field of the physics per-seed JSONs."""
import hashlib, json, sys
import numpy as np
from bhl_robust.eval import random_maze as rm
from bhl_robust.navgym import env as ng

RES = "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/results/navgym-v4-transfer-20260930"
h = lambda mz: hashlib.sha256(rm.world_xml(mz, textured=False).encode()).hexdigest()[:16]
out = []
for k in range(12):
    seed = 50000 + k
    phys = {a: json.load(open(f"{RES}/{a}/seed{seed}.json"))["maze"] for a in ("armV4-s5", "armV4-s6", "astar")}
    ph = {a: v["layout_sha256"] for a, v in phys.items()}
    gen = rm.generate(6, 6, seed, extra_openings=1)
    gym_hashes = []
    for r in range(5):   # (a) randomized-dynamics construction of gym_replay.py
        env = ng.MazeNavEnv(sizes=((6, 6),), randomize_dynamics=True, seed_base=seed, seed_span=1, version=2)
        env.reset(seed=seed * 10 + r)
        gym_hashes.append((env.episode_seed, env.maze.n, env.maze.m, env.maze.extra_openings, h(env.maze)))
    env = ng.MazeNavEnv(sizes=((6, 6),), randomize_dynamics=False, seed_base=seed, seed_span=1, version=2, max_steps=4500)
    env.reset(seed=seed)     # (b) nominal construction
    gym_hashes.append((env.episode_seed, env.maze.n, env.maze.m, env.maze.extra_openings, h(env.maze)))
    # also: wall boxes the gym ray-casts vs the boxes world_xml writes for MuJoCo (same source: maze.wall_segments())
    boxes = ng.wall_boxes(env.maze)
    row = {"seed": seed, "physics": ph, "generate_6x6_eo1": h(gen),
           "gym_envs": gym_hashes, "all_equal": len({h(gen), *ph.values(), *[g[4] for g in gym_hashes]}) == 1,
           "n_boxes": int(len(boxes)), "physics_walls": phys["armV4-s5"]["walls"],
           "cells_on_route_gen": len(gen.solution()), "cells_on_route_phys": phys["armV4-s5"]["cells_on_shortest_route"],
           "route_m_gym": round(float(env.route_m), 3), "max_steps_gym_default": ng.route_time_limit(env.route_m)}
    out.append(row)
    print(seed, row["all_equal"], h(gen), ph["armV4-s5"], [g[4] for g in gym_hashes][0], "eps", {g[0] for g in gym_hashes},
          "eo", {g[3] for g in gym_hashes}, "boxes", row["n_boxes"], row["physics_walls"], "route", row["cells_on_route_gen"], row["cells_on_route_phys"],
          "route_m", row["route_m_gym"], "limit", row["max_steps_gym_default"], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("ALL EQUAL:", all(r["all_equal"] for r in out))
