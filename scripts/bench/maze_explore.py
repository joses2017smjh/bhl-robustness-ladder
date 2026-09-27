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

Sensor-fusion stress options (all off by default; at default flags the run is
the published one, step for step):

* `--pose-yaw-deg`, `--pose-bias-m`, `--pose-noise-m`: localization error in
  the SF-02 conventions of `inspection_maze.py` (seeded
  `default_rng(10_000 + seed)`, bias direction drawn first, white noise drawn
  per policy step). The error enters the ESTIMATED pose, which is what the
  lidar map, the A* planner and the turn-then-walk controller use. The judge
  (goal radius, fall, wall contact, path length) keeps the TRUE pose.
* `--imu-source estimated`: the SF-03b attitude filter in the loop, by import
  of `inspection_maze.EstimatedAttitude` (Mahony at `--imu-rate-hz` from the
  physics substep hook, `--imu-delay-ms`, stationary-alignment gate). The
  estimate replaces the quaternion and gyro of the raw observation vector
  (`obs[0:4]`, `obs[4:7]`), which `RlController.update` parses the same way for
  the 12-DoF biped as for the 22-DoF humanoid. The navigation heading stays the
  oracle yaw; only the gait sees the filter.

Goal-check honesty: the controller stops when its estimated position is within
the goal radius, the judge needs the true position there. Each seed records
both (`goal_check`), and `outcome_class` refines a non-reached episode into
`arrived_not_judged` (the controller claimed the goal, the judge did not
agree), `stuck` (true position stayed within 0.2 m for the final 20 s) or
`time_out`. Neither counts as reached; `outcome` and `success` are unchanged.
`maze_explore.py sf-verdict <root>` applies the predeclared sweep rule
(`sf_sweep_verdict`) to the per-seed JSONs of `slurm/repo20260923/cpu_maze_sf.sbatch`.
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


# ------------------------------------------------- sensor-fusion stress (SF)
STUCK_WINDOW_S = 20.0      # predeclared: "stuck" = no true displacement > STUCK_RADIUS_M in the final 20 s
STUCK_RADIUS_M = 0.2


class PoseError:
    """Localization error on the estimated pose, SF-02 conventions.

    Same generator and draw order as `inspection_maze.episode`: one
    `default_rng(10_000 + seed)`, the bias direction drawn first (only when a
    bias is set), then one 2-D normal per policy step (only when noise is
    set); the heading error is a constant added to the yaw. With all three at
    zero `apply` returns its inputs untouched and consumes no random numbers.
    """

    def __init__(self, seed: int, bias_m: float = 0.0, noise_m: float = 0.0, yaw_deg: float = 0.0):
        self.bias_m, self.noise_m, self.yaw_deg = float(bias_m), float(noise_m), float(yaw_deg)
        self.rng = np.random.default_rng(10_000 + seed)
        self.bias = self.bias_m * (lambda v: v / np.linalg.norm(v))(self.rng.normal(size=2)) if self.bias_m else np.zeros(2)
        self.yaw_rad = float(np.deg2rad(self.yaw_deg))
        self.active = bool(self.bias_m or self.noise_m or self.yaw_deg)

    def apply(self, xy, yaw):
        """(true xy, true yaw) -> (estimated xy, estimated yaw)."""
        if not self.active:
            return xy, yaw
        xy_est = xy + self.bias + (self.rng.normal(size=2) * self.noise_m if self.noise_m else 0.0)
        return xy_est, yaw + self.yaw_rad

    def summary(self):
        return {"pose_bias_m": self.bias_m, "pose_noise_m": self.noise_m, "pose_yaw_deg": self.yaw_deg,
                "bias_vector_m": [round(float(v), 4) for v in self.bias], "rng": "default_rng(10_000 + seed), SF-02",
                "applied_to": "map + planner + controller (estimated pose); judge uses the true pose"}


class StallTracker:
    """Time since the true position last left a disc of `radius` around an anchor.

    The anchor moves to the current position whenever the robot is more than
    `radius` from it, so `current_s` is the length of the ongoing stall and
    `longest_s` the longest one in the episode.
    """

    def __init__(self, xy, t: float, radius: float = STUCK_RADIUS_M):
        self.anchor = np.array(xy, dtype=float)
        self.t0 = float(t)
        self.radius = radius
        self.current_s = 0.0
        self.longest_s = 0.0

    def update(self, xy, t: float):
        if math.hypot(float(xy[0]) - self.anchor[0], float(xy[1]) - self.anchor[1]) > self.radius:
            self.anchor = np.array(xy, dtype=float)
            self.t0 = float(t)
        self.current_s = float(t) - self.t0
        self.longest_s = max(self.longest_s, self.current_s)


def classify_outcome(outcome: str, controller_arrived: bool, final_stall_s: float, stuck_s: float = STUCK_WINDOW_S) -> str:
    """Refine the judge's outcome. Precedence: the judge's own verdicts (goal,
    fall, non-finite) stand; a non-reached episode in which the controller
    claimed the goal at least once is `arrived_not_judged`; otherwise one
    whose true position stayed within STUCK_RADIUS_M for the final `stuck_s`
    is `stuck`; otherwise `time_out`. Only `goal` counts as reached."""
    if outcome != "time_out":
        return outcome
    if controller_arrived:
        return "arrived_not_judged"
    if final_stall_s >= stuck_s:
        return "stuck"
    return "time_out"


# The predeclared sweep (slurm/repo20260923/cpu_maze_sf.sbatch, hard 6x6, seeds 0-11). Tags are the output
# sub-directories. Rule: primary metric = reached with no fall, out of 12; a level PASSES with >= 10/12 reached
# and 0 falls. Heading tolerance = the largest heading error that passes, valid only if the 0 deg control is 12/12.
# IMU budget = the largest delay that passes, valid only if the 0 ms estimated control is >= 11/12. Bias and noise
# levels pass or fail on their own. Brake-off twins are reported, not gated. Nothing is re-tuned after the sweep.
SF_SEEDS = 12
SF_PASS_REACHED = 10
SF_SWEEP = {
    "heading": {"control": "yaw0", "control_min": 12, "unit": "deg",
                "levels": [(1.0, "yaw1"), (3.0, "yaw3"), (10.0, "yaw10")]},
    "imu": {"control": "imu-est0", "control_min": 11, "unit": "ms",
            "levels": [(20.0, "imu-est20"), (30.0, "imu-est30"), (40.0, "imu-est40"), (60.0, "imu-est60")]},
    "bias": {"unit": "m", "levels": [(0.05, "bias0.05"), (0.15, "bias0.15"), (0.30, "bias0.30")]},
    "noise": {"unit": "m", "levels": [(0.10, "noise0.10")]},
    "exploratory": ["yaw3-brakeoff", "yaw10-brakeoff"],
}


def sf_counts(rows) -> dict:
    """Per-configuration counts from per-seed result dicts (never from exit codes)."""
    classes = [r.get("outcome_class", r["outcome"]) for r in rows]
    return {"n": len(rows), "seeds": sorted(r["seed"] for r in rows),
            "reached": sum(bool(r["success"]) for r in rows),
            "falls": sum(r["outcome"] == "fall" for r in rows),
            "clean": sum(bool(r.get("clean_success")) for r in rows),
            "arrived_not_judged": classes.count("arrived_not_judged"), "stuck": classes.count("stuck"),
            "time_out": classes.count("time_out"),
            "nonfinite": sum(c.startswith("nonfinite") for c in classes)}


def _level_pass(c, n_seeds):
    return c is not None and c["n"] == n_seeds and c["reached"] >= SF_PASS_REACHED and c["falls"] == 0


def sf_sweep_verdict(rows_by_tag: dict, n_seeds: int = SF_SEEDS) -> dict:
    """Apply the predeclared rule to {tag: [per-seed result dict]}."""
    counts = {tag: sf_counts(rows) for tag, rows in rows_by_tag.items()}
    out = {"rule": {"primary": f"reached with no fall out of {n_seeds}", "level_pass": f">= {SF_PASS_REACHED}/{n_seeds} reached and 0 falls",
                    "heading_valid": "0 deg control 12/12", "imu_valid": "0 ms estimated control >= 11/12",
                    "stuck_and_arrived_not_judged": "reported separately, never counted as reached"},
           "counts": counts}
    for name in ("heading", "imu"):
        spec = SF_SWEEP[name]
        ctrl = counts.get(spec["control"])
        complete = ctrl is not None and ctrl["n"] == n_seeds and all(counts.get(t, {}).get("n") == n_seeds for _, t in spec["levels"])
        valid = complete and ctrl["reached"] >= spec["control_min"]
        passes = {t: _level_pass(counts.get(t), n_seeds) for _, t in spec["levels"]}
        passing = [lv for lv, t in spec["levels"] if passes[t]]
        largest = max(passing) if passing else None
        # a smaller level failing while a larger one passes is flagged, not smoothed over
        non_monotone = any(not passes[t] and largest is not None and lv < largest for lv, t in spec["levels"])
        verdict = ("INCOMPLETE" if not complete else "INVALID_CONTROL" if not valid
                   else "MEASURED" if largest is not None else "BELOW_SMALLEST_LEVEL")
        out[name] = {"verdict": verdict, "control": spec["control"], "control_reached": ctrl["reached"] if ctrl else None,
                     "control_min": spec["control_min"], "level_pass": passes,
                     ("tolerance_" if name == "heading" else "budget_") + spec["unit"]: largest if valid else None,
                     "non_monotone": non_monotone}
    for name in ("bias", "noise"):
        spec = SF_SWEEP[name]
        out[name] = {t: ("INCOMPLETE" if counts.get(t, {}).get("n") != n_seeds else "PASS" if _level_pass(counts[t], n_seeds) else "FAIL")
                     for _, t in spec["levels"]}
    out["exploratory"] = {t: counts.get(t) for t in SF_SWEEP["exploratory"]}
    return out


def load_sweep(root: Path) -> dict:
    """{tag: [per-seed dict]} from <root>/<tag>/seed<k>.json (render sidecars excluded)."""
    import re
    rows = {}
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        files = [f for f in d.iterdir() if re.fullmatch(r"seed\d+\.json", f.name)]
        if files:
            rows[d.name] = [json.loads(f.read_text()) for f in sorted(files)]
    return rows


def _pose_error_on(args) -> bool:
    return bool(getattr(args, "pose_yaw_deg", 0.0) or getattr(args, "pose_bias_m", 0.0) or getattr(args, "pose_noise_m", 0.0))


# ------------------------------------------------------------------ recorder
class ExploreRecorder:
    """Top view with breadcrumbs + depth pair + lidar + occupancy map, per step."""

    def __init__(self, args, model, slot, maze: rm.Maze, policy_dt: float, out_mp4: Path, png_dir: Path):
        self.args, self.model, self.slot, self.maze = args, model, slot, maze
        self.render = not args.no_render
        self.w, self.h = args.width, args.height
        self.side_w = 320
        self.fps = 1.0 / policy_dt
        # the clip holds every `stride`-th policy step, so its frame rate is the policy rate over the stride: real time
        self.sink = FrameSink(out_mp4, png_dir, self.fps / max(1, args.stride))
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
        d.text((6, 4), "lidar-built map (" + ("ESTIMATED pose" if _pose_error_on(self.args) else "oracle pose") + ") + plan",
               font=panels.load_font(13), fill=panels.TEXT)
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
        footer = ("gait: learned PPO (biped dr-default), frozen | map: lidar log-odds | "
                  + ("pose: oracle + injected error | " if _pose_error_on(self.args) else "pose: oracle | ")
                  + ("attitude: filter in the loop | " if getattr(self.args, "imu_source", "truth") == "estimated" else "")
                  + ("commands: LEARNED NavGym policy (PPO, trained in the gym, deployed here) | no sideways command" if self.args.policy is not None
                     else "plan: A* on the map, unknown = free | turn in place, then walk forward; never sideways"))
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
    learned = None
    if args.policy is not None:
        import onnxruntime as ort
        from bhl_robust.navgym.env import EgoMap, LIDAR_RANGE as NAV_RANGE, V_MAX, W_MAX, build_obs
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(args.policy), so)
        # the actor's own input names decide the observation: v1 (lidar, map, goal), v2 (lidar, near, map, goal)
        keys = tuple(i.name for i in sess.get_inputs())
        learned = {"sess": sess, "keys": keys, "emap": EgoMap(maze.bounds()), "prev": np.zeros(2, np.float32),
                   "scale": (V_MAX, W_MAX), "build": build_obs, "range": NAV_RANGE}
    goal = maze.centre(maze.goal)
    dt = float(cfg.policy_dt)
    rec = recorder_factory(model, slot, maze, dt) if recorder_factory else None
    # SF-02: error on the estimated pose (map + planner + controller); the judge below keeps the true pose
    pose = PoseError(seed, args.pose_bias_m, args.pose_noise_m, args.pose_yaw_deg)
    imu = None
    if args.imu_source == "estimated":
        # SF-03b: the attitude filter in the loop, the inspection-maze class itself (not a copy)
        from inspection_maze import EstimatedAttitude
        imu = EstimatedAttitude(args, model, slot, seed)
        imu.bind_sensor_rate(runner, float(cfg.physics_dt))

    plan, wp_index, last_stamp, last_plan_t = None, 0, None, -1e9
    outcome, t_done, wall_steps, path_m = None, None, 0, 0.0
    trace = []
    prior_xy = runner.d.xpos[slot.body_id, :2].copy()
    stall = StallTracker(prior_xy, 0.0)
    arrived_first, arrived_steps, arrived_last, arrived_true_d, arrived_est_d = None, 0, False, None, None
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
        xy_e, yaw_e = pose.apply(xy, yaw)       # the robot's belief; identical objects when no error is set
        # map from the newest lidar packet (10 Hz)
        pkt = sensors.packets[0]
        if pkt is not None and pkt["stamp_s"] != last_stamp:
            last_stamp = pkt["stamp_s"]
            grid.update(xy_e[0], xy_e[1], yaw_e, sensors.angles, np.asarray(pkt["lidar_raw_m"]), LIDAR_RANGE)
            if learned is not None:
                learned["emap"].update(xy_e[0], xy_e[1], yaw_e, sensors.angles, np.asarray(pkt["lidar_raw_m"]))
        # plan on a timer, or when there is no plan yet
        if plan is None or now - last_plan_t >= args.replan_s:
            new_plan = planner.plan(xy_e, goal)
            last_plan_t = now
            if new_plan:
                plan, wp_index = new_plan, 1 if len(new_plan) > 1 else 0
        if learned is not None:
            # the gym's observation, built by the gym's own builder from the packet's raw rays and the ego map
            raw_m = np.asarray(pkt["lidar_raw_m"]) if pkt is not None else np.full(108, learned["range"])
            obs_ = learned["build"](learned["keys"], raw_m, learned["emap"], xy_e[0], xy_e[1], yaw_e, goal, learned["prev"])
            feeds = {k: np.asarray(v, dtype=np.float32)[None] for k, v in obs_.items()}
            act = np.clip(learned["sess"].run(None, feeds)[0][0], -1.0, 1.0).astype(np.float32)
            learned["prev"] = act
            v_max, w_max = learned["scale"]
            raw = np.array([(act[0] + 1.0) * 0.5 * v_max, 0.0, act[1] * w_max])
            state, err = "policy", 0.0
        elif plan:
            while wp_index < len(plan) - 1 and math.hypot(plan[wp_index][0] - xy_e[0], plan[wp_index][1] - xy_e[1]) < ctrl.waypoint_radius:
                wp_index += 1
            wp = plan[wp_index]
            is_goal = wp_index == len(plan) - 1
            raw, state, err = ctrl.command(xy_e, yaw_e, wp, is_goal)
        else:
            raw, state, err = np.array([0.0, 0.0, ctrl.turn_rate]), "search", 0.0
        if now < args.settle_s:
            raw, state = np.zeros(3), "settle"
        # goal-check honesty: the controller's own "arrived" (estimated pose), recorded beside the judge's verdict
        arrived_last = learned is None and state == "arrived" and is_goal
        if arrived_last:
            arrived_steps += 1
            if arrived_first is None:
                arrived_first = now
                arrived_true_d = math.hypot(xy[0] - goal[0], xy[1] - goal[1])
                arrived_est_d = math.hypot(xy_e[0] - goal[0], xy_e[1] - goal[1])
        command = sensors.filter_commands(runner.d, [raw], now)[0]
        obs = runner.observe(0, command)
        if imu is not None:
            obs = imu.apply(obs, runner.d, dt)
        target = controller.update(obs)
        if not np.isfinite(target).all():
            outcome = "nonfinite_action"
            break
        runner.step([target])
        wall_steps += int(runner.hit_wall)
        after = runner.d.xpos[slot.body_id, :2].copy()
        path_m += float(np.linalg.norm(after - prior_xy))
        prior_xy = after
        stall.update(after, now + dt)
        known = grid.known_fraction() if step % 25 == 0 else (trace[-1]["known"] if trace else 0.0)
        if step % 5 == 0:
            trace.append({"t": round(now, 2), "xy": [round(float(after[0]), 3), round(float(after[1]), 3)], "yaw": round(yaw, 3),
                          "state": state, "cmd": [round(float(c), 3) for c in command], "known": round(known, 3),
                          "plan_len": len(plan) if plan else 0})
            if pose.active:
                trace[-1]["xy_est"] = [round(float(xy_e[0]), 3), round(float(xy_e[1]), 3)]
                trace[-1]["yaw_est"] = round(float(yaw_e), 3)
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
    final_xy = runner.d.xpos[slot.body_id, :2]
    outcome_class = classify_outcome(outcome, arrived_first is not None, stall.current_s)
    oracle_pose = not pose.active
    result = {
        "seed": seed, "maze": {"n": maze.n, "m": maze.m, "extra_openings": maze.extra_openings, "cells_on_shortest_route": len(maze.solution() or []),
                               "walls": len(maze.wall_segments()), "layout_sha256": hashlib.sha256(rm.world_xml(maze, textured=False).encode()).hexdigest()[:16]},
        "outcome": outcome, "success": success, "clean_success": success and wall_steps == 0,
        "completion_s": round(t_done, 2) if success else None, "elapsed_s": round(now, 2), "wall_contact_steps": wall_steps,
        "path_length_m": round(path_m, 2), "turns": ctrl.turns, "replans": planner.replans, "map_updates": grid.updates,
        "mapped_fraction_end": round(grid.known_fraction(), 3), "sensor_mode": args.sensor_mode, "sensor_stats": sensors.stats,
        "initial_heading_rad": round(yaw0, 3), "controller": {"cruise": args.cruise, "turn_rate": args.turn_rate, "inflate_m": args.inflate,
                                                              "replan_s": args.replan_s, "map_res": args.map_res},
        "control": ("frozen_learned_biped_gait+lidar_occupancy_map+LEARNED_navgym_policy(onnx); " if learned is not None
                    else "frozen_learned_biped_gait+lidar_occupancy_map+astar_unknown_free+turn_then_walk; ")
                   + ("pose and goal are oracle" if oracle_pose else "goal is oracle; pose = oracle + injected error (pose_error)")
                   + ("" if imu is None else "; gait attitude = filter in the loop (imu)"),
        "policy": str(args.policy) if args.policy is not None else None,
        "wall_seconds": round(time.time() - t_wall0, 1), "trace": trace,
        "outcome_class": outcome_class,
        "goal_check": {
            "judge": {"rule": "TRUE position within goal_radius (ends the episode)", "goal_radius_m": ctrl.goal_radius, "reached": success,
                      "final_true_dist_m": round(math.hypot(final_xy[0] - goal[0], final_xy[1] - goal[1]), 3)},
            "controller": {"rule": "ESTIMATED position within goal_radius of the plan's last waypoint (the exact goal); commands zero",
                           "arrived_ever": arrived_first is not None, "first_arrival_s": round(arrived_first, 2) if arrived_first is not None else None,
                           "true_dist_at_first_arrival_m": round(arrived_true_d, 3) if arrived_true_d is not None else None,
                           "est_dist_at_first_arrival_m": round(arrived_est_d, 3) if arrived_est_d is not None else None,
                           "arrived_steps": arrived_steps, "arrived_at_end": bool(arrived_last)},
            "note": ("with no pose error the judge fires on the post-step true pose one step before the controller would see it, "
                     "so arrived_ever is expected false on a reached episode; the learned --policy has no arrival state")},
        "stall": {"final_s": round(stall.current_s, 2), "longest_s": round(stall.longest_s, 2), "window_s": STUCK_WINDOW_S,
                  "radius_m": STUCK_RADIUS_M, "position": "true"},
        "pose_error": pose.summary(),
        "imu": imu.summary() if imu is not None else {"source": "truth"},
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
    ap.add_argument("--policy", type=Path, default=None,
                    help="NavGym actor (.onnx from navgym_train.py): replaces the A* planner + turn-then-walk with the learned "
                         "policy; same lidar sectors, the same 0.2 m egocentric map built from the same raw rays, oracle pose + goal")
    sf = ap.add_argument_group("sensor-fusion stress (all off by default; SF-02 / SF-03b conventions of inspection_maze.py)")
    sf.add_argument("--pose-yaw-deg", type=float, default=0.0, help="constant heading error of the estimated pose")
    sf.add_argument("--pose-bias-m", type=float, default=0.0, help="constant position offset magnitude, random direction per seed")
    sf.add_argument("--pose-noise-m", type=float, default=0.0, help="white position noise std per policy step")
    sf.add_argument("--imu-source", choices=("truth", "estimated"), default="truth",
                    help="'estimated' replaces the oracle quaternion/gyro in the gait observation with inspection_maze.EstimatedAttitude")
    sf.add_argument("--imu-filter", choices=("mahony", "madgwick"), default="mahony")
    sf.add_argument("--imu-init", choices=("accel", "truth"), default="accel")
    sf.add_argument("--imu-kp", type=float, default=1.0)
    sf.add_argument("--imu-ki", type=float, default=0.1)
    sf.add_argument("--imu-beta", type=float, default=0.1)
    sf.add_argument("--imu-gyro-std", type=float, default=0.0, help="rad/s white noise (stress setting, not a calibration)")
    sf.add_argument("--imu-accel-std", type=float, default=0.0, help="m/s^2 white noise")
    sf.add_argument("--imu-gyro-bias", type=float, default=0.0, help="rad/s constant bias magnitude, random direction per seed")
    sf.add_argument("--imu-gyro-bias-walk", type=float, default=0.0, help="rad/s/sqrt(s)")
    sf.add_argument("--imu-delay-steps", type=int, default=0, help="policy steps of IMU delay (policy-rate mode, --imu-rate-hz 0, only)")
    sf.add_argument("--imu-rate-hz", type=float, default=200.0, help="filter rate from the physics substep hook (0 = policy rate)")
    sf.add_argument("--imu-delay-ms", type=float, default=0.0, help="IMU delivery delay in ms (sensor-rate mode)")
    args = ap.parse_args()
    if min(args.pose_bias_m, args.pose_noise_m) < 0:
        ap.error("--pose-bias-m and --pose-noise-m are magnitudes (>= 0)")
    imu_opts = ("imu_gyro_std", "imu_accel_std", "imu_gyro_bias", "imu_gyro_bias_walk", "imu_delay_steps", "imu_delay_ms")
    if args.imu_source == "truth" and any(getattr(args, k) for k in imu_opts):
        ap.error("IMU corruption options need --imu-source estimated (they would be silently ignored)")
    if args.imu_source == "estimated" and args.imu_rate_hz <= 0 and args.imu_delay_ms:
        ap.error("--imu-delay-ms needs the sensor-rate filter (--imu-rate-hz > 0)")
    if args.imu_source == "estimated" and args.imu_rate_hz > 0 and args.imu_delay_steps:
        ap.error("--imu-delay-steps is the policy-rate delay; with --imu-rate-hz > 0 use --imu-delay-ms")

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
        print(json.dumps({k_: res[k_] for k_ in ("seed", "outcome", "outcome_class", "completion_s", "wall_contact_steps", "path_length_m", "turns", "replans",
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
    classes = [r["outcome_class"] for r in results]
    summary["outcome_classes"] = {c: classes.count(c) for c in sorted(set(classes))}
    summary["reached_no_fall"] = len(succ)      # a fall ends the episode, so reached == reached with no fall
    summary["arrived_not_judged"] = classes.count("arrived_not_judged")
    summary["stuck"] = classes.count("stuck")
    summary["controller_arrived_ever"] = sum(r["goal_check"]["controller"]["arrived_ever"] for r in results)
    summary["stress"] = {"pose_error": {k_: getattr(args, k_) for k_ in ("pose_yaw_deg", "pose_bias_m", "pose_noise_m")},
                         "imu": ({k_: getattr(args, k_) for k_ in ("imu_source", "imu_filter", "imu_init", "imu_kp", "imu_ki", "imu_rate_hz",
                                                                   "imu_delay_ms", "imu_delay_steps") + imu_opts[:4]}
                                 if args.imu_source == "estimated" else {"imu_source": "truth"})}
    (args.out_dir / f"summary{args.tag}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("MAZE-EXPLORE SUMMARY " + json.dumps({k_: summary[k_] for k_ in ("n", "success", "clean_success", "falls", "time_outs", "median_completion_s",
                                                                          "median_clean_seed_by_time", "arrived_not_judged", "stuck")}), flush=True)
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


def verdict_main(argv) -> int:
    """`maze_explore.py sf-verdict <root> [--out FILE]`: the predeclared sweep rule over per-seed JSONs."""
    ap = argparse.ArgumentParser(prog="maze_explore.py sf-verdict")
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/verdict.json")
    a = ap.parse_args(argv)
    v = sf_sweep_verdict(load_sweep(a.root))
    out = a.out or a.root / "verdict.json"
    out.write_text(json.dumps(v, indent=2) + "\n")
    print("MAZE-SF VERDICT " + json.dumps({k_: v[k_] for k_ in ("heading", "imu", "bias", "noise")}), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sf-verdict":
        sys.exit(verdict_main(sys.argv[2:]))
    sys.exit(main())
