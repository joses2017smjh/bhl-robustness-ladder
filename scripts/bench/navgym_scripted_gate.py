"""Acceptance gate E0 for NavGym v2: can a known-good controller solve the gym?

The scripted controller that reached the goal on 24/24 physics mazes
(results/maze-explore-20260924: lidar log-odds OccupancyGrid -> A* with unknown
space free -> TurnWalkController, pose and goal oracle) is driven *inside* the
gym, on the proxy dynamics randomized exactly as in the trainer's held-out
evaluation (maze seed 10 000 + k, dynamics from reset(seed=k), k = 0..n-1) and
under the gym's own collision test and time limit. It sees only what the physics
runner sees: pose, goal, and the 108 raw lidar ranges, fed to its map at 10 Hz
(the physics packet rate) and replanned every 0.4 s.

If the gym is a fair proxy, this controller must pass it. Rule (predeclared in
the task, 2026-09-26): PASS iff >= 22/24 goals on 5x5 AND >= 20/24 on 6x6 AND
<= 1 collision per size. v1 scored 1/24 and 0/24 under the same driver.

Inflation 0.45 m (physics used 0.30 m): the gym's proxy has per-episode lag up
to 0.45 s, latency up to 2 steps and drift up to 0.06 rad/s, and the gym's
footprint is a 0.22 m disc around the body centre; at 0.30 m the driver's path
sits 0.25-0.30 m from walls and the proxy's tracking error eats that margin
(diagnosis h23_inf0.30_circle: 17/24 collisions on 5x5). 0.45 m still leaves
a 0.42 m wide centre band in the 1.32 m corridors.

Usage:
  python scripts/bench/navgym_scripted_gate.py --out gate.json [--n 24] [--version 2] [--workers 2]
Prints one line "NAVGYM E0 GATE: PASS|FAIL ..." computed from the JSON it writes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

RULE = {"5x5": {"min_goals": 22, "max_collisions": 1}, "6x6": {"min_goals": 20, "max_collisions": 1}}


def run_episode(job) -> dict:
    size, k, version, inflate_m, max_steps = job
    from bhl_robust.eval import random_maze as rm
    from bhl_robust.navgym.env import DT, LIDAR_RANGE, V_MAX, W_MAX, box_clearance, heldout_env

    env, _, info = heldout_env((size, size), k, version=version, max_steps=max_steps)
    grid = rm.OccupancyGrid(env.maze.bounds(), res=0.10, margin=0.6)
    planner = rm.Planner(grid, inflate_m=inflate_m)
    ctrl = rm.TurnWalkController(cruise=0.30, turn_rate=0.6)
    goal = env.goal_xy
    plan, wp, last_plan, last_pkt = None, 0, -1e9, -1
    outcome, min_clear, states = None, 9.0, {}
    t0 = time.time()
    while True:
        now = env.t * DT
        xy = (env.x, env.y)
        pkt = int(now / 0.1)                              # 10 Hz lidar packets, as on the physics robot
        if pkt != last_pkt:
            last_pkt = pkt
            grid.update(env.x, env.y, env.yaw, env.angles, env.ranges, LIDAR_RANGE)
        if plan is None or now - last_plan >= 0.4:
            p = planner.plan(xy, goal)
            last_plan = now
            if p:
                plan, wp = p, (1 if len(p) > 1 else 0)
        if plan:
            while wp < len(plan) - 1 and math.hypot(plan[wp][0] - xy[0], plan[wp][1] - xy[1]) < ctrl.waypoint_radius:
                wp += 1
            cmd, state, _ = ctrl.command(xy, env.yaw, plan[wp], wp == len(plan) - 1)
        else:
            cmd, state = np.array([0.0, 0.0, ctrl.turn_rate]), "search"
        states[state] = states.get(state, 0) + 1
        # (vx, wz) command -> the gym's normalized action
        a = np.array([cmd[0] / V_MAX * 2.0 - 1.0, cmd[2] / W_MAX], dtype=np.float32)
        _, _, term, trunc, step_info = env.step(a)
        min_clear = min(min_clear, float(box_clearance(env.boxes, env.x, env.y)))
        if term or trunc:
            outcome = step_info["outcome"]
            break
    d = env.dyn
    return {"size": f"{size}x{size}", "k": k, "maze_seed": int(env.episode_seed), "outcome": outcome, "steps": int(env.t),
            "max_steps": int(env.max_steps), "steps_frac_of_limit": round(env.t / env.max_steps, 4),
            "route_m": round(float(info.get("route_m", float("nan"))), 3), "min_centre_clearance_m": round(min_clear, 4),
            "states": states, "replans": planner.replans, "wall_s": round(time.time() - t0, 1),
            "dyn": {"w_gain": round(d.w_gain, 3), "v_gain": round(d.v_gain, 3), "tau": round(d.tau, 3), "drift": round(d.drift, 4),
                    "latency": d.latency, "v_noise": round(d.v_noise, 3), "w_noise": round(d.w_noise, 3)}}


def verdict(episodes: list, sizes) -> dict:
    per = {}
    ok = True
    for s in sizes:
        key = f"{s}x{s}"
        R = [e for e in episodes if e["size"] == key]
        goals = sum(e["outcome"] == "goal" for e in R)
        coll = sum(e["outcome"] == "collision" for e in R)
        tout = sum(e["outcome"] == "time_out" for e in R)
        fr = [e["steps_frac_of_limit"] for e in R if e["outcome"] == "goal"]
        per[key] = {"n": len(R), "goals": goals, "collisions": coll, "time_outs": tout,
                    "goal_steps_frac_of_limit_max": max(fr) if fr else None,
                    "goal_steps_frac_of_limit_median": float(np.median(fr)) if fr else None,
                    "min_centre_clearance_m": min(e["min_centre_clearance_m"] for e in R) if R else None,
                    "failures": [(e["k"], e["outcome"], e["steps"], e["max_steps"]) for e in R if e["outcome"] != "goal"]}
        rule = RULE.get(key)
        if rule is not None:
            passed = len(R) == 24 and goals >= rule["min_goals"] and coll <= rule["max_collisions"]
            per[key]["rule"] = rule
            per[key]["pass"] = passed
            ok &= passed
        else:
            per[key]["pass"] = None
    ok &= all(f"{s}x{s}" in RULE for s in sizes) and set(RULE) <= {f"{s}x{s}" for s in sizes}
    return {"pass": bool(ok), "per_size": per}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=24, help="held-out episodes per size (the rule needs 24)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[5, 6])
    ap.add_argument("--version", type=int, default=2)
    ap.add_argument("--inflate", type=float, default=0.45)
    ap.add_argument("--max-steps", type=int, default=None, help="flat limit override (default: the version's own)")
    ap.add_argument("--only", type=int, nargs="*", default=None, help="run only these k (diagnosis; the verdict needs all 24)")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    ks = args.only if args.only else list(range(args.n))
    jobs = [(s, k, args.version, args.inflate, args.max_steps) for s in args.sizes for k in ks]
    t0 = time.time()
    if args.workers > 1:
        with Pool(args.workers) as p:
            eps = p.map(run_episode, jobs, chunksize=1)
    else:
        eps = [run_episode(j) for j in jobs]
    v = verdict(eps, args.sizes)
    out = {"gate": "navgym_E0_scripted", "env_version": args.version, "inflate_m": args.inflate, "max_steps_override": args.max_steps,
           "driver": "OccupancyGrid(0.10 m, 10 Hz) + Planner(A*, unknown free, replan 0.4 s) + TurnWalkController(cruise 0.30, turn 0.6)",
           "seeds": "held-out: maze seed 10000+k, reset(seed=k), randomized dynamics", "rule": RULE, "wall_s": round(time.time() - t0, 1),
           **v, "episodes": eps}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    # the verdict line is recomputed from the file just written
    rd = json.loads(args.out.read_text())
    parts = [f"{k}: goals {p['goals']}/{p['n']} collisions {p['collisions']} time-outs {p['time_outs']} "
             f"(max steps/limit on goals {p['goal_steps_frac_of_limit_max']})" for k, p in rd["per_size"].items()]
    print(f"NAVGYM E0 GATE: {'PASS' if rd['pass'] else 'FAIL'} | v{rd['env_version']} inflate {rd['inflate_m']} | " + " | ".join(parts), flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.exit(main())
