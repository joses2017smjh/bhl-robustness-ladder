"""Randomized-maze navigation with the robot's own lidar building the map.

A new maze per seed (`bhl_robust.eval.random_maze.generate`), the biped's
frozen learned gait (`dr-default-s0`, the checkpoint that turns in place), a
log-odds occupancy map built from the 108-ray lidar at 10 Hz, an A* planner on
that map that treats unknown space as free and replans as walls appear, and a
turn-then-walk command generator (no sideways walking). The lidar/depth speed
brake of `team_sensors` stays on as the last safety layer.

What is oracle here, and said on every frame: localization (the map frame is
the simulator's true pose) and the goal coordinate. What is not: the maze --
the planner sees only what the lidar has mapped.

Per seed it writes `<out-dir>/seed<k>.json`; `--seeds N` produces
`<out-dir>/summary.json`. Predeclared success: goal reached within
--time-limit with no fall; wall-contact steps are reported separately and a
"clean" success has none. `--render` records the three-panel clip (overhead
with the trajectory as breadcrumbs, the depth pair, the lidar scan and the
occupancy map with the planned path) for one seed; `--no-render` runs the
same pipeline with a blank main view for a login-node check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]

import mujoco                                                      # noqa: E402
from omegaconf import OmegaConf                                    # noqa: E402
from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController  # noqa: E402
from team_airlock import ContactRunner, CpuPolicy                  # noqa: E402
from maze_record import FrameSink                                  # noqa: E402
from bhl_robust.eval import panels                                 # noqa: E402
from bhl_robust.eval import random_maze as rm                      # noqa: E402
from bhl_robust.eval.multi_robot import _WORLDS, build_multi       # noqa: E402
from bhl_robust.eval.team_sensors import (DEPTH_RANGE, LIDAR_RANGE, TeamSensors)  # noqa: E402

# The biped's rig: the Isaac maze policies' mounts (sensors_rig.py), body frame.
LIDAR_MOUNT_BIPED = (0.0, 0.0, 0.34)
STEREO_CENTER_BIPED = (0.12, 0.0, 0.30)
GAIT_DEFAULT = "logs/rsl_rl/biped/2026-08-17_09-54-10_dr-default-s0/exported/deploy.yaml"


def yaw_of(q):
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ recorder
class ExploreRecorder:
    """Top view with breadcrumbs + depth pair + lidar + occupancy map, per step."""

    def __init__(self, args, model, slot, maze: rm.Maze, policy_dt: float, out_mp4: Path, png_dir: Path):
        self.args, self.model, self.slot, self.maze = args, model, slot, maze
        self.render = not args.no_render
        self.w, self.h = args.width, args.height
        self.side_w = 320
        self.fps = 1.0 / policy_dt
        self.sink = FrameSink(out_mp4, png_dir, self.fps)
        self.frames = 0
        self.last_frame = None
        self.crumbs: list[tuple[float, float, float]] = []      # (x, y, t)
        xmin, xmax, ymin, ymax = maze.bounds()
        self.centre = ((xmin + xmax) / 2, (ymin + ymax) / 2)
        if self.render:
            self.top = mujoco.Renderer(model, height=self.h, width=self.w, max_geom=20000)
            self.cam = mujoco.MjvCamera()
            self.cam.lookat[:] = (self.centre[0], self.centre[1], 0.0)
            extent = max(xmax - xmin, ymax - ymin) + 0.6
            self.cam.distance = args.distance if args.distance else extent * 1.25
            self.cam.azimuth, self.cam.elevation = args.azimuth, args.elevation

    def _draw_crumbs(self, t_now: float):
        scn = self.top.scene
        for (x, y, t) in self.crumbs:
            if scn.ngeom >= scn.maxgeom:
                break
            g = scn.geoms[scn.ngeom]
            f = 0.3 + 0.7 * (t / max(t_now, 1e-6))
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.045, 0, 0]),
                                np.array([x, y, 0.05]), np.eye(3).ravel(), np.array([1.0, 0.55 * f, 0.1, 0.95], dtype=np.float32))
            scn.ngeom += 1

    def map_panel(self, grid: rm.OccupancyGrid, blocked, xy, yaw, plan, size=(320, 200)):
        from PIL import Image, ImageDraw
        w, h = size
        p = grid.prob()
        img = np.full((grid.ny, grid.nx, 3), 128, np.uint8)       # unknown grey
        img[(p < 0.35).T] = (235, 235, 235)                         # free
        img[(p > 0.65).T] = (35, 40, 50)                            # occupied
        if blocked is not None:
            band = blocked.T & ~(p > 0.65).T
            img[band] = (90, 96, 110)                               # inflation band
        pil = Image.fromarray(img[::-1])                            # +y up
        title_h = 22
        scale = min((w - 8) / grid.nx, (h - title_h - 6) / grid.ny)
        pil = pil.resize((max(1, int(grid.nx * scale)), max(1, int(grid.ny * scale))), Image.NEAREST)
        out = Image.new("RGB", (w, h), panels.PANEL_BG)
        d = ImageDraw.Draw(out)
        d.rectangle((0, 0, w, title_h), fill=(40, 44, 52))
        d.text((6, 4), "lidar-built map (oracle pose) + plan", font=panels.load_font(13), fill=panels.TEXT)
        ox, oy = 4, title_h + 3
        out.paste(pil, (ox, oy))

        def px(x, y):
            i = (x - grid.x0) / grid.res * scale
            j = (grid.ny - (y - grid.y0) / grid.res) * scale
            return ox + i, oy + j
        gx, gy = self.maze.centre(self.maze.goal)
        d.ellipse((*(np.array(px(gx, gy)) - 5), *(np.array(px(gx, gy)) + 5)), fill=(40, 200, 90))
        if len(self.crumbs) > 1:
            d.line([px(x, y) for x, y, _ in self.crumbs], fill=(255, 140, 30), width=2)
        if plan:
            d.line([px(*xy)] + [px(x, y) for x, y in plan], fill=(80, 200, 255), width=2)
        cx, cy = px(*xy)
        tip = (cx + 9 * math.cos(yaw), cy - 9 * math.sin(yaw))
        left = (cx + 6 * math.cos(yaw + 2.5), cy - 6 * math.sin(yaw + 2.5))
        right = (cx + 6 * math.cos(yaw - 2.5), cy - 6 * math.sin(yaw - 2.5))
        d.polygon([tip, left, right], fill=panels.ROBOT)
        return out

    def __call__(self, *, step, now, runner, sensors, xy, yaw, state, err, plan, grid, blocked, known, replans):
        if self.args.stride > 1 and step % self.args.stride:
            return
        if not self.crumbs or math.hypot(xy[0] - self.crumbs[-1][0], xy[1] - self.crumbs[-1][1]) > 0.2:
            self.crumbs.append((float(xy[0]), float(xy[1]), now))
        if self.render:
            self.top.update_scene(runner.d, camera=self.cam)
            self._draw_crumbs(now)
            top_rgb = self.top.render()
        else:
            top_rgb = np.full((self.h, self.w, 3), 40, dtype=np.uint8)
        pkt = sensors.latest[0]
        stale = not (pkt and pkt.get("extero_fresh"))
        lidar = np.asarray(pkt["lidar_sector_m"]) if pkt else None
        depth = np.asarray(pkt["paired_idealized_depth_m"]) if pkt else None
        brake = pkt.get("brake") if pkt else None
        dp = panels.depth_pair_panel(depth, DEPTH_RANGE, (self.side_w, 170), "stereo rig: ray depth 8x8, L / R (no RGB)",
                                     stale=stale, subtitle="10 Hz packets; feeds the speed brake only")
        lp = panels.lidar_panel(lidar, LIDAR_RANGE, (self.side_w, 170), "lidar: 36 sector minima of 108 rays",
                                window_m=4.0, stale=stale, brake=brake, subtitle="forward is up; raw rays build the map")
        mp = self.map_panel(grid, blocked, xy, yaw, plan, size=(self.side_w, max(160, self.h - 340)))
        header = (f"random maze seed {self.maze.seed} ({self.maze.n}x{self.maze.m}, unknown to the planner) | t = {now:5.1f} s"
                  f" | {state} | mapped {100 * known:3.0f} % | replans {replans} | 1x")
        footer = ("gait: learned PPO (biped dr-default), frozen | map: lidar log-odds | pose: oracle | plan: A* on the map,"
                  " unknown = free | turn in place, then walk forward; never sideways")
        frame = panels.compose_frame(top_rgb, [dp, lp, mp], header, footer, side_w=self.side_w)
        self.last_frame = frame
        self.sink.add(np.ascontiguousarray(frame))
        self.frames += 1

    def hold(self, seconds: float, banner: str, colour=(20, 110, 60)):
        if self.last_frame is None:
            return
        from PIL import Image, ImageDraw
        img = Image.fromarray(self.last_frame.copy())
        draw = ImageDraw.Draw(img)
        font = panels.load_font(30)
        tw = draw.textlength(banner, font=font)
        x, y = (img.size[0] - self.side_w - tw) / 2, 60
        draw.rectangle((x - 14, y - 8, x + tw + 14, y + 40), fill=colour)
        draw.text((x, y), banner, font=font, fill=(255, 255, 255))
        arr = np.asarray(img)
        for _ in range(int(seconds * self.fps / max(1, self.args.stride))):
            self.sink.add(np.ascontiguousarray(arr))
            self.frames += 1

    def close(self):
        if self.render:
            self.top.close()
        return self.sink.close()


# ------------------------------------------------------------------- episode
def run_seed(args, cfg, policy, seed: int, recorder_factory=None) -> dict:
    maze = rm.generate(args.n, args.m, seed, extra_openings=args.extra_openings)
    _WORLDS["random_maze"] = rm.world_xml(maze, textured=not args.plain_world)
    cache = Path(args.cache_dir) / f"maze{seed}"
    cache.mkdir(parents=True, exist_ok=True)
    model, slots = build_multi(Path(args.upstream), cache, 1, ["explorer"], variant="biped", world="random_maze")
    controller = RlController(cfg)
    controller.policy = policy
    runner = ContactRunner(model, slots, [cfg], [controller])
    rng = np.random.default_rng(seed)
    runner.reset(rng)
    slot = slots[0]
    sx, sy = maze.centre(maze.start)
    runner.d.qpos[slot.qpos_adr:slot.qpos_adr + 2] = (sx + rng.normal(0, 0.03), sy + rng.normal(0, 0.03))
    yaw0 = rng.uniform(-math.pi, math.pi) if args.random_heading else 0.0
    runner.d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7] = (math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2))
    mujoco.mj_forward(model, runner.d)
    owners = np.array([0 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "").startswith(slot.prefix)
                       else -1 for g in range(model.ngeom)])
    runner.configure_contacts(owners)
    sensors = TeamSensors(model, slots, owners, mode=args.sensor_mode, seed=seed, dropout_probability=args.dropout_probability,
                          lidar_mount=LIDAR_MOUNT_BIPED, stereo_center=STEREO_CENTER_BIPED)
    grid = rm.OccupancyGrid(maze.bounds(), res=args.map_res, margin=0.6)
    planner = rm.Planner(grid, inflate_m=args.inflate)
    ctrl = rm.TurnWalkController(cruise=args.cruise, turn_rate=args.turn_rate)
    goal = maze.centre(maze.goal)
    dt = float(cfg.policy_dt)
    rec = recorder_factory(model, slot, maze, dt) if recorder_factory else None

    plan, wp_index, last_stamp, last_plan_t = None, 0, None, -1e9
    outcome, t_done, wall_steps, path_m = None, None, 0, 0.0
    trace = []
    prior_xy = runner.d.xpos[slot.body_id, :2].copy()
    steps = int(args.time_limit / dt)
    t_wall0 = time.time()
    for step in range(steps):
        now = step * dt
        xy = runner.d.xpos[slot.body_id, :2].copy()
        q = runner.d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7]
        yaw = yaw_of(q)
        if not np.isfinite(runner.d.qpos).all():
            outcome = "nonfinite_state"
            break
        # map from the newest lidar packet (10 Hz)
        pkt = sensors.packets[0]
        if pkt is not None and pkt["stamp_s"] != last_stamp:
            last_stamp = pkt["stamp_s"]
            grid.update(xy[0], xy[1], yaw, sensors.angles, np.asarray(pkt["lidar_raw_m"]), LIDAR_RANGE)
        # plan on a timer, or when there is no plan yet
        if plan is None or now - last_plan_t >= args.replan_s:
            new_plan = planner.plan(xy, goal)
            last_plan_t = now
            if new_plan:
                plan, wp_index = new_plan, 1 if len(new_plan) > 1 else 0
        if plan:
            while wp_index < len(plan) - 1 and math.hypot(plan[wp_index][0] - xy[0], plan[wp_index][1] - xy[1]) < ctrl.waypoint_radius:
                wp_index += 1
            wp = plan[wp_index]
            is_goal = wp_index == len(plan) - 1
            raw, state, err = ctrl.command(xy, yaw, wp, is_goal)
        else:
            raw, state, err = np.array([0.0, 0.0, ctrl.turn_rate]), "search", 0.0
        if now < args.settle_s:
            raw, state = np.zeros(3), "settle"
        command = sensors.filter_commands(runner.d, [raw], now)[0]
        obs = runner.observe(0, command)
        target = controller.update(obs)
        if not np.isfinite(target).all():
            outcome = "nonfinite_action"
            break
        runner.step([target])
        wall_steps += int(runner.hit_wall)
        after = runner.d.xpos[slot.body_id, :2].copy()
        path_m += float(np.linalg.norm(after - prior_xy))
        prior_xy = after
        known = grid.known_fraction() if step % 25 == 0 else (trace[-1]["known"] if trace else 0.0)
        if step % 5 == 0:
            trace.append({"t": round(now, 2), "xy": [round(float(after[0]), 3), round(float(after[1]), 3)], "yaw": round(yaw, 3),
                          "state": state, "cmd": [round(float(c), 3) for c in command], "known": round(known, 3),
                          "plan_len": len(plan) if plan else 0})
        if rec is not None:
            rec(step=step, now=now, runner=runner, sensors=sensors, xy=after, yaw=yaw, state=state, err=err, plan=plan[wp_index:] if plan else None,
                grid=grid, blocked=planner.last_blocked, known=known, replans=planner.replans)
        if runner.tilt(0) >= 0.78:
            outcome = "fall"
            t_done = now
            break
        if math.hypot(after[0] - goal[0], after[1] - goal[1]) < ctrl.goal_radius:
            outcome = "goal"
            t_done = now + dt
            break
    if outcome is None:
        outcome = "time_out"
    success = outcome == "goal"
    result = {
        "seed": seed, "maze": {"n": maze.n, "m": maze.m, "extra_openings": maze.extra_openings, "cells_on_shortest_route": len(maze.solution() or []),
                               "walls": len(maze.wall_segments()), "layout_sha256": hashlib.sha256(rm.world_xml(maze, textured=False).encode()).hexdigest()[:16]},
        "outcome": outcome, "success": success, "clean_success": success and wall_steps == 0,
        "completion_s": round(t_done, 2) if success else None, "elapsed_s": round(now, 2), "wall_contact_steps": wall_steps,
        "path_length_m": round(path_m, 2), "turns": ctrl.turns, "replans": planner.replans, "map_updates": grid.updates,
        "mapped_fraction_end": round(grid.known_fraction(), 3), "sensor_mode": args.sensor_mode, "sensor_stats": sensors.stats,
        "initial_heading_rad": round(yaw0, 3), "controller": {"cruise": args.cruise, "turn_rate": args.turn_rate, "inflate_m": args.inflate,
                                                              "replan_s": args.replan_s, "map_res": args.map_res},
        "control": "frozen_learned_biped_gait+lidar_occupancy_map+astar_unknown_free+turn_then_walk; pose and goal are oracle",
        "wall_seconds": round(time.time() - t_wall0, 1), "trace": trace,
    }
    if rec is not None:
        banner = f"GOAL REACHED  {t_done:.1f} s" if success else f"{outcome.upper().replace('_', ' ')}  {now:.1f} s"
        rec.hold(1.5, banner, colour=(20, 110, 60) if success else (150, 40, 40))
        sink = rec.close()
        result["clip"] = {**sink, "frames": rec.frames, "video_fps": rec.fps / max(1, args.stride), "playback_speed": 1.0,
                          "render": {"no_render": args.no_render, "main": [args.width, args.height], "camera": {"distance": rec.cam.distance if rec.render else None,
                                     "azimuth": args.azimuth, "elevation": args.elevation}, "textured_world": not args.plain_world}}
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--gait", type=Path, default=None, help="deploy.yaml of the gait (default: biped dr-default-s0)")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--extra-openings", type=int, default=2)
    ap.add_argument("--time-limit", type=float, default=150.0)
    ap.add_argument("--settle-s", type=float, default=1.0)
    ap.add_argument("--cruise", type=float, default=0.30)
    ap.add_argument("--turn-rate", type=float, default=0.6)
    ap.add_argument("--inflate", type=float, default=0.30)
    ap.add_argument("--replan-s", type=float, default=0.4)
    ap.add_argument("--map-res", type=float, default=0.10)
    ap.add_argument("--random-heading", action="store_true", help="random initial heading (default: facing +x)")
    ap.add_argument("--sensor-mode", choices=("record", "reactive", "reactive_dropout"), default="reactive")
    ap.add_argument("--dropout-probability", type=float, default=0.35)
    ap.add_argument("--out-dir", type=Path, default=REPO / "results/maze-explore-20260924")
    ap.add_argument("--render", action="store_true", help="record the three-panel clip for each seed run")
    ap.add_argument("--no-render", action="store_true", help="with --render: blank main view, no OpenGL")
    ap.add_argument("--frames-root", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "maze-explore-frames"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--distance", type=float, default=0.0, help="camera distance (0 = fit the maze)")
    ap.add_argument("--azimuth", type=float, default=90.0)
    ap.add_argument("--elevation", type=float, default=-66.0)
    ap.add_argument("--stride", type=int, default=2, help="record every Nth policy step")
    ap.add_argument("--plain-world", action="store_true")
    ap.add_argument("--gif", type=Path, default=None, help="with --render and one seed: write this GIF + sidecar")
    ap.add_argument("--gif-speed", type=float, default=4.0)
    ap.add_argument("--gif-fps", type=int, default=8)
    ap.add_argument("--gif-width", type=int, default=860)
    ap.add_argument("--tag", default="", help="suffix for per-seed files")
    args = ap.parse_args()

    gait = args.gait or (Path(args.upstream) / GAIT_DEFAULT)
    cfg = OmegaConf.load(gait)
    if cfg.num_actions != 12:
        raise SystemExit("this mission uses the 12-DoF biped gait (the humanoid gait does not turn in place)")
    policy = CpuPolicy(cfg.policy_checkpoint_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for k in range(args.seeds):
        seed = args.seed_start + k
        factory = None
        if args.render:
            def factory(model, slot, maze, dt, seed=seed):
                return ExploreRecorder(args, model, slot, maze, dt, args.out_dir / f"seed{seed}{args.tag}.mp4",
                                       Path(args.frames_root) / f"seed{seed}{args.tag}")
        res = run_seed(args, cfg, policy, seed, factory)
        res["gait"] = {"deploy": str(gait), "checkpoint": str(cfg.policy_checkpoint_path),
                       "checkpoint_sha256": sha256(Path(cfg.policy_checkpoint_path)) if Path(cfg.policy_checkpoint_path).is_file() else None}
        (args.out_dir / f"seed{seed}{args.tag}.json").write_text(json.dumps(res, indent=2) + "\n")
        results.append(res)
        print(json.dumps({k_: res[k_] for k_ in ("seed", "outcome", "completion_s", "wall_contact_steps", "path_length_m", "turns", "replans",
                                                 "mapped_fraction_end", "wall_seconds")} | {"route_cells": res["maze"]["cells_on_shortest_route"]}), flush=True)
    n = len(results)
    succ = [r for r in results if r["success"]]
    clean = [r for r in succ if r["clean_success"]]
    times = sorted(r["completion_s"] for r in succ)
    summary = {"seeds": [r["seed"] for r in results], "n": n, "success": len(succ), "clean_success": len(clean),
               "falls": sum(r["outcome"] == "fall" for r in results), "time_outs": sum(r["outcome"] == "time_out" for r in results),
               "median_completion_s": (times[len(times) // 2] if times else None), "completion_s": times,
               "wall_contact_steps": [r["wall_contact_steps"] for r in results],
               "median_seed_by_time": (sorted(succ, key=lambda r: r["completion_s"])[len(succ) // 2]["seed"] if succ else None),
               "median_clean_seed_by_time": (sorted(clean, key=lambda r: r["completion_s"])[len(clean) // 2]["seed"] if clean else None),
               "settings": {k_: getattr(args, k_) for k_ in ("n", "m", "extra_openings", "time_limit", "cruise", "turn_rate", "inflate", "replan_s",
                                                             "map_res", "sensor_mode", "random_heading")},
               "gait": results[0]["gait"] if results else None}
    (args.out_dir / f"summary{args.tag}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("MAZE-EXPLORE SUMMARY " + json.dumps({k_: summary[k_] for k_ in ("n", "success", "clean_success", "falls", "time_outs", "median_completion_s",
                                                                          "median_clean_seed_by_time")}), flush=True)
    if args.gif and args.render and n == 1 and results[0].get("clip", {}).get("mp4"):
        r = results[0]
        g = panels.write_gif(Path(r["clip"]["mp4"]), args.gif, fps=args.gif_fps, speed=args.gif_speed, width=args.gif_width)
        side = {"output": os.path.relpath(args.gif, REPO), "output_sha256": sha256(args.gif), "output_mb": g["mb"], "gif": g,
                "source_clip": os.path.relpath(r["clip"]["mp4"], REPO), "source_sha256": sha256(Path(r["clip"]["mp4"])),
                "evidence": os.path.relpath(args.out_dir / f"seed{r['seed']}{args.tag}.json", REPO),
                "episode": {k_: r[k_] for k_ in ("seed", "outcome", "success", "clean_success", "completion_s", "wall_contact_steps", "path_length_m",
                                                  "turns", "replans", "mapped_fraction_end")}, "maze": r["maze"],
                "playback_speed": args.gif_speed, "gait": r["gait"],
                "scope": ("MuJoCo 3.3.5 | frozen learned biped gait (dr-default-s0) | lidar log-odds map | A* on the map with unknown = free | "
                          "turn in place then walk forward (no sideways) | oracle pose and goal coordinate; the maze itself is unknown to the planner | "
                          "one seed = one maze; the multi-seed table is the evidence, this clip the illustration")}
        args.gif.with_suffix(".json").write_text(json.dumps(side, indent=2) + "\n")
        print(f"GIF {side['output']} {g['mb']} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
