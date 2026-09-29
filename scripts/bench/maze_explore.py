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

`--variant humanoid` (opt-in, 2026-09-28; the default `biped` path is unchanged
step for step): the 22-DoF humanoid (`berkeley_humanoid_lite.xml`) driven by
the one QUALIFIED turning checkpoint, arms-turn-turnboth-s0 (LEARNED gait, one
checkpoint of 12 seeds; it marches in place at zero command and turns while
stepping). Same SCRIPTED lidar map + A* + turn-then-walk, ORACLE pose and goal,
same judge (goal radius 0.30 m, first entry ends the episode). Differences, all
gated on the flag: the team_sensors deck mounts (lidar 0.72 m / stereo 0.70 m
above the base frame, which sits at the feet: world z ~0.66 / 0.64 m, under the
1.1 m walls), each scan mapped from the lidar's own pose (`scan_rays_from_sensor`:
the deck lidar rides ~0.2 m ahead of the base, which smeared walls into a
corridor on pilot seed 106), the run_eval fall rule (tilt > 0.78 rad OR root
sink > 0.25 m, the rule behind the 7/60 push qualification), the settings
frozen from the seed >= 100 pilot (`HUMANOID_FROZEN`), diagnostics (wall
contacts per controller state, planner failures, floor returns), and every
frame/footer/JSON label.
`maze_explore.py humanoid-verdict <root>` applies the predeclared rule of
`slurm/repo20260923/cpu_maze_humanoid.sbatch` (`humanoid_maze_verdict`).
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

# ------------------------------------------------- 22-DoF humanoid (--variant humanoid)
# The only QUALIFIED turning checkpoint (cpu_turn_qualify.sbatch, 2026-09-27: turn 10/10, walk 3/3,
# push 7/60). ONE checkpoint (1 of 12 turning-arm seeds), not a recipe; every output says so.
HUMANOID_RUN = "arms-turn-turnboth-s0"
HUMANOID_GAIT_GLOB = f"logs/rsl_rl/humanoid/*_{HUMANOID_RUN}/exported/deploy.yaml"
# Fall = the run_eval / harness rule that produced the 7/60 push rate (copied, not imported: harness pulls
# in torch). tests/test_maze_humanoid.py asserts both equal bhl_robust.eval.harness.TILT_LIMIT_RAD / MAX_SINK_M.
HUMANOID_TILT_LIMIT_RAD = 0.78
HUMANOID_MAX_SINK_M = 0.25
# Measured 2026-09-28 (standing 3 s at zero command, MuJoCo, collision-geom AABB corners in the base frame):
# humanoid planar radius 0.309 m (arms at +-0.307 m), biped 0.204 m; maze corridor clear width
# CELL - WALL_T = 1.32 m. Recorded in every humanoid JSON beside the planner inflation.
HUMANOID_RADIUS_M = 0.309
BIPED_RADIUS_M = 0.204
# Frozen from the pilot on maze seeds >= 100 only (hard 6x6 and base 5x5; every pilot seed is listed in
# the launcher header). Planner inflation must be a multiple of --map-res (0.10): Planner rounds it to cells.
HUMANOID_FROZEN = {"cruise": 0.30, "turn_rate": 0.6, "inflate": 0.50, "settle_s": 1.0,
                   "turn_enter": None, "turn_exit": None, "wz_walk": None, "waypoint_radius": None}
# The predeclared scored block (never run by anything before 2026-09-28; checked by grep over results/).
HUMANOID_SCORED = {"n": 6, "m": 6, "extra_openings": 1, "time_limit": 180.0, "seeds": list(range(12, 24))}
HUMANOID_PASS_REACHED = 10
HUMANOID_MAZE_RULE = ("PASS iff >= 10 of the 12 humanoid episodes (hard 6x6, 1 extra opening, maze seeds 12-23, "
                      "180 s each, frozen settings) reach the goal AND there are 0 falls in the 12; else FAIL; "
                      "INCOMPLETE if any humanoid seed JSON or the humanoid summary is missing or does not match "
                      "the declared gait/maze/settings. Clean (no wall contact) counts and median completion times "
                      "are reported beside the biped A* reference on the same seeds, not gated.")


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


def _arm_counts(rows) -> dict:
    """sf_counts plus completion times; the median is summary.json's convention (times[n // 2] of the successes)."""
    c = sf_counts(rows)
    times = sorted(r["completion_s"] for r in rows if r["success"])
    c["median_completion_s"] = times[len(times) // 2] if times else None
    c["completion_s"] = times
    c["wall_contact_steps"] = {r["seed"]: r["wall_contact_steps"] for r in sorted(rows, key=lambda r: r["seed"])}
    return c


BIPED_REFERENCE_SETTINGS = {"cruise": 0.30, "turn_rate": 0.6, "inflate": 0.30}   # the published hard-6x6 table's


def _arm_problems(name, rows, summary, humanoid, declared, frozen):
    p = []
    got = sorted(r.get("seed") for r in rows)
    if got != sorted(declared["seeds"]):
        p.append(f"{name}: seeds {got} != declared {sorted(declared['seeds'])}")
    want_maze = (declared["n"], declared["m"], declared["extra_openings"])
    bad_maze = sorted(r.get("seed") for r in rows if tuple(r.get("maze", {}).get(k) for k in ("n", "m", "extra_openings")) != want_maze)
    if bad_maze:
        p.append(f"{name}: seeds {bad_maze} are not the declared {want_maze} maze")
    if summary is None:
        p.append(f"{name}: no summary.json")
    else:
        s = summary.get("settings", {})
        bad = {k: (s.get(k), declared[k]) for k in ("n", "m", "extra_openings", "time_limit") if s.get(k) != declared[k]}
        want = frozen if humanoid else BIPED_REFERENCE_SETTINGS
        bad.update({k: (s.get(k), v) for k, v in want.items() if s.get(k) != v})
        if bad:
            p.append(f"{name}: settings differ from the declared ones (got, want): {bad}")
    for r in rows:
        deploy = str(r.get("gait", {}).get("deploy", ""))
        if humanoid and (r.get("variant") != "humanoid" or f"_{HUMANOID_RUN}/" not in deploy):
            p.append(f"{name}: seed {r.get('seed')} is not the humanoid {HUMANOID_RUN} gait ({deploy})")
        if not humanoid and (r.get("variant", "biped") != "biped" or "/biped/" not in deploy):
            p.append(f"{name}: seed {r.get('seed')} is not the biped gait ({deploy})")
    return p


def humanoid_maze_verdict(h_rows, h_summary, b_rows, b_summary, declared=None, frozen=None) -> dict:
    """The predeclared rule of cpu_maze_humanoid.sbatch (HUMANOID_MAZE_RULE), from per-seed JSONs only.
    PASS / FAIL read the humanoid arm alone; the biped A* reference is reported beside it, never gated."""
    declared = HUMANOID_SCORED if declared is None else declared
    frozen = HUMANOID_FROZEN if frozen is None else frozen
    n = len(declared["seeds"])
    hp = _arm_problems("humanoid", h_rows, h_summary, True, declared, frozen)
    bp = _arm_problems("biped", b_rows, b_summary, False, declared, frozen)
    hc = _arm_counts(h_rows) if h_rows else None
    bc = _arm_counts(b_rows) if b_rows else None
    if hp:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS" if hc["reached"] >= HUMANOID_PASS_REACHED and hc["falls"] == 0 else "FAIL"

    def line(c):
        if c is None:
            return "no episodes"
        return (f"{c['reached']}/{c['n']} reached, {c['falls']} falls, clean {c['clean']}/{c['n']}, "
                f"median {c['median_completion_s']} s (upper median of the successes), stuck {c['stuck']}, time-out {c['time_out']}")
    detail = (f"humanoid ({HUMANOID_RUN}, LEARNED gait, one checkpoint): {line(hc)} (need >= {HUMANOID_PASS_REACHED}/{n} and 0 falls)"
              f" | biped A* reference (dr-default-s0): {line(bc)}" + ("" if not bp else " [reference INCOMPLETE]"))
    return {"verdict": verdict, "detail": detail, "rule": HUMANOID_MAZE_RULE, "declared": declared, "frozen_humanoid_settings": frozen,
            "humanoid": hc, "biped_reference": bc, "problems": hp,
            "reference_status": "COMPLETE" if not bp else "INCOMPLETE: " + "; ".join(bp),
            "labels": {"humanoid": humanoid_labels(HUMANOID_RUN),
                       "biped_reference": {"gait": "LEARNED biped gait dr-default-s0 (frozen, 12-DoF)",
                                           "planner": "SCRIPTED A* on the lidar log-odds map + turn-then-walk",
                                           "pose": "ORACLE (simulator pose)", "goal": "ORACLE goal coordinate"}},
            "median_convention": "times[n // 2] of the sorted successful completion times (summary.json's convention)",
            "comparability": ("same maze seeds, judge (0.30 m goal radius, first entry ends the episode), 180 s limit, lidar, "
                              "planner and controller code; differences: the humanoid maps each scan from the lidar's own pose "
                              "(0.12 m forward, 0.72 m up; ~0.21-0.23 m ahead of the base while walking) with floor returns "
                              "clearing only, the biped from the base xy as published (lidar over the base); planner inflation "
                              "0.50 m (humanoid, radius 0.309 m) vs 0.30 m (biped, radius 0.204 m); fall rule tilt > 0.78 rad "
                              "or sink > 0.25 m (humanoid, run_eval's) vs tilt >= 0.78 rad (biped, published)"),
            "not_scored": ["stopping at the goal: the judge ends the episode on first entry into the 0.30 m radius",
                           "localization: oracle pose; goal: oracle coordinate"]}


def load_arm(d: Path):
    """(per-seed rows, summary or None) from <d>/seed<k>.json and <d>/summary.json."""
    import re
    d = Path(d)
    if not d.is_dir():
        return [], None
    rows = [json.loads(f.read_text()) for f in sorted(d.iterdir()) if re.fullmatch(r"seed\d+\.json", f.name)]
    s = d / "summary.json"
    return rows, (json.loads(s.read_text()) if s.is_file() else None)


def _pose_error_on(args) -> bool:
    return bool(getattr(args, "pose_yaw_deg", 0.0) or getattr(args, "pose_bias_m", 0.0) or getattr(args, "pose_noise_m", 0.0))


# ------------------------------------------------------------ humanoid helpers
def is_humanoid(args) -> bool:
    return getattr(args, "variant", "biped") == "humanoid"


def gait_run_name(deploy) -> str:
    """'.../<YYYY-MM-DD_hh-mm-ss>_<run>/exported/deploy.yaml' -> '<run>'."""
    import re
    return re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_", "", Path(deploy).parent.parent.name)


def humanoid_labels(run: str, pose_error: bool = False, imu_filter: bool = False) -> dict:
    """The honest labels every humanoid output carries: what is learned, scripted and oracle."""
    gait = f"LEARNED gait {run} (one checkpoint, frozen; 22-DoF humanoid)"
    if run == HUMANOID_RUN:
        gait += ("; QUALIFIED 2026-09-27 as a single checkpoint (turn 10/10, walk 3/3, push 7/60), "
                 "1 of 12 turning-arm seeds, not a recipe; marches in place at zero command, turns while stepping")
    return {"gait": gait + ("; attitude = filter in the loop" if imu_filter else ""),
            "planner": "SCRIPTED A* on the lidar log-odds map (unknown = free) + turn-then-walk (never sideways)",
            "pose": "ORACLE + injected error (the estimated pose)" if pose_error else "ORACLE (simulator pose)",
            "goal": "ORACLE goal coordinate"}


def humanoid_footer(run: str, pose_error: bool = False, imu_filter: bool = False) -> str:
    return (f"LEARNED gait {run} (one checkpoint) | SCRIPTED A* planner + turn-then-walk | "
            + ("ORACLE goal, pose = oracle + injected error" if pose_error else "ORACLE pose and goal")
            + (" | attitude: filter in the loop" if imu_filter else ""))


def humanoid_fell(tilt: float, sink: float) -> bool:
    """run_eval's fall rule (harness.TILT_LIMIT_RAD / MAX_SINK_M): tilt past 0.78 rad or the root 0.25 m low."""
    return tilt > HUMANOID_TILT_LIMIT_RAD or sink > HUMANOID_MAX_SINK_M


FLOOR_HIT_Z_M = 0.02      # a lidar return whose 3-D end point is this close to the floor plane is the floor, not a wall


def scan_rays_from_sensor(base_pos, base_rot, mount, dirs_body, ranges, max_range, xy_est=None, yaw_err=0.0):
    """Map rays of one lidar scan integrated from the SENSOR's pose (humanoid path).

    The biped path integrates every scan from the base xy with the body yaw; its lidar sits over the base
    (mount x = 0, 0.34 m up, body tilt ~0.03 rad), so the error is ~1 cm. The humanoid's deck lidar is
    0.12 m forward and 0.72 m up, and the marching gait pitches the body ~0.14 rad, so its origin is up to
    ~0.2 m ahead of the base: integrated from the base, walls it faces are drawn ~0.1-0.2 m too close
    (pilot seed 106: a 1.32 m corridor closed to A*). Here each ray is rotated by the full body rotation
    at capture, starts at base + R @ mount, is projected onto the floor plane (reach = r * horizontal
    fraction), and a return that ends on the floor (z < FLOOR_HIT_Z_M, the pitched rays of a 0.66 m
    lidar reach it at ~4-5 m) clears the cells it crossed but marks nothing occupied.
    With pose error (SF-02) the estimate replaces the base xy and adds `yaw_err` to every azimuth; the
    mount offset is rotated by the same error. Returns (origin_xy, azimuths, reaches, hits)."""
    base_pos, base_rot = np.asarray(base_pos, float), np.asarray(base_rot, float).reshape(3, 3)
    origin = base_pos + base_rot @ np.asarray(mount, float)
    d = np.asarray(dirs_body, float) @ base_rot.T
    r = np.minimum(np.asarray(ranges, float), max_range)
    hit = r < max_range - 1e-3
    floor = hit & (origin[2] + r * d[:, 2] < FLOOR_HIT_Z_M)
    offset = origin[:2] - base_pos[:2]
    if xy_est is not None:
        c, s = math.cos(yaw_err), math.sin(yaw_err)
        offset = np.array([c * offset[0] - s * offset[1], s * offset[0] + c * offset[1]])
        origin_xy = np.asarray(xy_est, float) + offset
    else:
        origin_xy = base_pos[:2] + offset
    return origin_xy, np.arctan2(d[:, 1], d[:, 0]) + yaw_err, r * np.hypot(d[:, 0], d[:, 1]), hit & ~floor


def integrate_scan(grid, origin_xy, azimuths, reaches, hits):
    """OccupancyGrid.update's log-odds model with a world azimuth and an explicit hit flag per ray: free
    cells along the reach, occupied at its end only for a wall hit. With azimuths = yaw + angles,
    reaches = min(r, max_range) and hits = r < max_range - 1e-3 it is OccupancyGrid.update, cell for
    cell (tests/test_maze_humanoid.py)."""
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    half = grid.res * 0.5
    for a, reach, hit in zip(azimuths, reaches, hits):
        reach = float(reach)
        ca, sa = math.cos(a), math.sin(a)
        n = int(reach / half)
        for k in range(n):
            dd = k * half
            if dd >= reach - half:
                break
            i, j = grid.to_cell(ox + dd * ca, oy + dd * sa)
            if grid.inside(i, j):
                grid.l[i, j] = max(grid.L_MIN, grid.l[i, j] + grid.L_FREE * 0.5)
        if hit:
            i, j = grid.to_cell(ox + reach * ca, oy + reach * sa)
            if grid.inside(i, j):
                grid.l[i, j] = min(grid.L_MAX, grid.l[i, j] + grid.L_OCC)
    grid.updates += 1


CONTROLLER_KNOBS = ("turn_enter", "turn_exit", "wz_walk", "waypoint_radius")


def controller_overrides(args) -> dict:
    """TurnWalkController keyword overrides that were set (None = the constructor default). goal_radius is
    the judge's 0.30 m and is deliberately not a knob."""
    return {k: getattr(args, k) for k in CONTROLLER_KNOBS if getattr(args, k, None) is not None}


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
        # optional display-only extras (--imu-panel / --rig-panels), attached by run_seed
        self.imu, self.rig, self.flow = None, None, None
        self.strip_h = 190
        self.eff_side_w = self.side_w
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
        if self.args.policy is None:
            title = "lidar-built map (" + ("ESTIMATED pose" if _pose_error_on(self.args) else "oracle pose") + ") + plan"
        else:
            title = "lidar map (" + ("EST. pose" if _pose_error_on(self.args) else "oracle pose") + "); no planner"
        d.text((6, 4), title, font=panels.load_font(13), fill=panels.TEXT)
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
        self._runner_d = runner.d
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
        extras = self.imu is not None or self.rig is not None
        total_h = self.h + (self.strip_h if self.imu is not None else 0)
        mp = self.map_panel(grid, blocked, xy, yaw, plan, size=(self.side_w, max(160, total_h - 340)))
        if self.args.policy is None:
            header = (f"random maze seed {self.maze.seed} ({self.maze.n}x{self.maze.m}, unknown to the planner) | t = {now:5.1f} s"
                      f" | {state} | mapped {100 * known:3.0f} % | replans {replans} | 1x")
        else:
            header = (f"random maze seed {self.maze.seed} ({self.maze.n}x{self.maze.m}, walls seen only through the lidar) | t = {now:5.1f} s"
                      f" | LEARNED NavGym policy | mapped {100 * known:3.0f} % | 1x")
        if is_humanoid(self.args):
            header = "22-DoF humanoid | " + header
            footer = humanoid_footer(self.args.gait_run, _pose_error_on(self.args),
                                     getattr(self.args, "imu_source", "truth") == "estimated")
        else:
            footer = ("gait: learned PPO (biped dr-default), frozen | map: lidar log-odds | "
                      + ("pose: oracle + injected error | " if _pose_error_on(self.args) else "pose: oracle | ")
                      + ("attitude: filter in the loop | " if getattr(self.args, "imu_source", "truth") == "estimated" else "")
                      + ("commands: LEARNED NavGym policy (PPO in the gym, deployed on the physics robot) | goal: oracle | no sideways"
                         if self.args.policy is not None
                         else "plan: A* on the map, unknown = free | turn in place, then walk forward; never sideways"))
        if extras:
            frame = self._compose_extras(top_rgb, [dp, lp, mp], header, footer, total_h)
        else:
            frame = panels.compose_frame(top_rgb, [dp, lp, mp], header, footer, side_w=self.side_w)
        self.last_frame = frame
        self.sink.add(np.ascontiguousarray(frame))
        self.frames += 1

    def _compose_extras(self, top_rgb, col_a, header, footer, total_h):
        """Main view (+ IMU strip) | column A (ray depth, lidar, map) | column B (RGB, depth, flow, IMU readout)."""
        from PIL import Image
        main = np.asarray(top_rgb)[..., :3]
        if self.imu is not None:
            strip = panels.imu_strip(self.imu.hist, self.imu.latest, (main.shape[1], self.strip_h),
                                     f"IMU: simulated IM10A @ {self.imu.rate_hz:.0f} Hz, datasheet noise "
                                     "(gyro 0.07 deg/s rms + 1 deg/s bias; accel 1 mg rms + 40 mg bias) | display only")
            main = np.vstack([main, np.asarray(strip)])
        col_b = []
        hb = total_h
        if self.rig is not None:
            rl, rr, dl, dr = self.rig.render(self._runner_d)
            flow = self.flow.update(rl, self.args.stride * self._dt) if self.flow is not None else None
            h_rgb = h_dep = 170
            h_flow = 250
            col_b.append(panels.stereo_rgb_panel(rl, rr, (self.side_w, h_rgb), "stereo rig RGB, L / R (render)",
                                                 "display only: the brake reads no RGB"))
            col_b.append(panels.depth_pair_panel(np.stack([dl, dr]), DEPTH_RANGE, (self.side_w, h_dep),
                                                 f"rendered depth {dl.shape[1]}x{dl.shape[0]}, L / R",
                                                 subtitle="display only; the brake reads the 8x8 above"))
            st = self.flow.last_stats if self.flow is not None else None
            col_b.append(panels.image_panel(flow, (self.side_w, h_flow),
                                            "optical flow, left eye (Farneback)" + (
                                                f" | median {st['median_px_per_s']:.0f} px/s" if st else "")))
            hb -= h_rgb + h_dep + h_flow
        if self.imu is not None:
            col_b.append(panels.imu_readout_panel(
                self.imu.latest, (self.side_w, max(120, hb)), "IMU: 6-axis filter vs truth; mag + baro",
                ["filter: Mahony on the noisy gyro + accel;", "heading from gyro only (drifts, as a",
                 "6-axis unit does). Magnetometer (field", "approx.) and barometer are simulated", "and NOT used: mag off indoors,",
                 "baro too coarse for a flat maze."]))
        elif col_b:
            col_b.append(Image.new("RGB", (self.side_w, max(1, hb)), panels.PANEL_BG))
        col_a_img = Image.new("RGB", (self.side_w, total_h), panels.PANEL_BG)
        y = 0
        for p in col_a:
            col_a_img.paste(p, (0, y))
            y += p.size[1]
        cols = Image.new("RGB", (2 * self.side_w, total_h), panels.PANEL_BG)
        cols.paste(col_a_img, (0, 0))
        y = 0
        for p in col_b:
            cols.paste(p, (self.side_w, y))
            y += p.size[1]
        self.eff_side_w = 2 * self.side_w
        return panels.compose_frame(main, [cols], header, footer, side_w=2 * self.side_w)

    def hold(self, seconds: float, banner: str, colour=(20, 110, 60)):
        if self.last_frame is None:
            return
        from PIL import Image, ImageDraw
        img = Image.fromarray(self.last_frame.copy())
        draw = ImageDraw.Draw(img)
        font = panels.load_font(30)
        tw = draw.textlength(banner, font=font)
        x, y = (img.size[0] - self.eff_side_w - tw) / 2, 60
        draw.rectangle((x - 14, y - 8, x + tw + 14, y + 40), fill=colour)
        draw.text((x, y), banner, font=font, fill=(255, 255, 255))
        arr = np.asarray(img)
        for _ in range(int(seconds * self.fps / max(1, self.args.stride))):
            self.sink.add(np.ascontiguousarray(arr))
            self.frames += 1

    def close(self):
        # EGL teardown can raise on some nodes (dgx2) after every frame is written; never lose the clip to it
        for closer in ((self.top.close,) if self.render else ()) + ((self.rig.close,) if self.rig is not None else ()):
            try:
                closer()
            except Exception as exc:                      # noqa: BLE001
                print(f"[render] renderer close failed ({exc!r}); frames are already written", flush=True)
        return self.sink.close()


# ------------------------------------------------------------------- episode
def run_seed(args, cfg, policy, seed: int, recorder_factory=None) -> dict:
    maze = rm.generate(args.n, args.m, seed, extra_openings=args.extra_openings)
    _WORLDS["random_maze"] = rm.world_xml(maze, textured=not args.plain_world)
    cache = Path(args.cache_dir) / f"maze{seed}"
    cache.mkdir(parents=True, exist_ok=True)
    humanoid = is_humanoid(args)
    model, slots = build_multi(Path(args.upstream), cache, 1, ["explorer"], variant="humanoid" if humanoid else "biped",
                               world="random_maze")
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
    if humanoid:
        # team_sensors' own defaults are the full humanoid's deck mounts (lidar +0.72 m, stereo +0.70 m, 0.12 m forward)
        sensors = TeamSensors(model, slots, owners, mode=args.sensor_mode, seed=seed, dropout_probability=args.dropout_probability)
    else:
        sensors = TeamSensors(model, slots, owners, mode=args.sensor_mode, seed=seed, dropout_probability=args.dropout_probability,
                              lidar_mount=LIDAR_MOUNT_BIPED, stereo_center=STEREO_CENTER_BIPED)
    grid = rm.OccupancyGrid(maze.bounds(), res=args.map_res, margin=0.6)
    planner = rm.Planner(grid, inflate_m=args.inflate)
    ctrl = rm.TurnWalkController(cruise=args.cruise, turn_rate=args.turn_rate, **controller_overrides(args))
    if humanoid:
        # judge-side fall bookkeeping (run_eval's rule) and where the wall contacts happen
        spawn_z = float(runner.d.qpos[slot.qpos_adr + 2])
        rot0, pos0 = runner.d.xmat[slot.body_id].reshape(3, 3), runner.d.xpos[slot.body_id]
        mounts = {"lidar_body_m": [round(float(v), 3) for v in sensors.lidar_mount],
                  "stereo_centre_body_m": [round(float(v), 3) for v in sensors.stereo_center],
                  "lidar_world_z_at_spawn_m": round(float((pos0 + rot0 @ sensors.lidar_mount)[2]), 3),
                  "stereo_world_z_at_spawn_m": round(float((pos0 + rot0 @ sensors.stereo_center)[2]), 3),
                  "wall_height_m": rm.WALL_H}
        hum = {"max_tilt": 0.0, "max_sink": -1e9, "wall_by_state": {}, "steps_by_state": {}, "floor_returns": 0,
               "origin_offset_max": 0.0, "planner_failures": 0}
        cap, cap_stamp = None, None
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
    if rec is not None and getattr(args, "imu_panel", False):
        # display only: reads the IMU site on its own random stream, chained after any bound hook
        from bhl_robust.eval.imu_sim import SimIM10A
        rec.imu = SimIM10A(model, slot, seed, float(cfg.physics_dt), rate_hz=args.imu_panel_rate,
                           kp=args.imu_kp, ki=args.imu_ki)
        rec.imu.attach(runner)
    if rec is not None and getattr(args, "rig_panels", False) and rec.render:
        from bhl_robust.eval.rig_views import FlowView, RigCameras
        rw, rh = (int(v) for v in args.rig_res.lower().split("x"))
        rec.rig = RigCameras(model, slot, sensors.stereo_center, width=rw, height=rh)
        rec.flow = FlowView()
    if rec is not None:
        rec._dt = dt

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
            if humanoid:
                # from the sensor's own pose at capture (scan_rays_from_sensor); the biped line below is unchanged
                o_xy, az, reach, hits = scan_rays_from_sensor(cap[0], cap[1], sensors.lidar_mount, sensors.lidar_dirs,
                                                              pkt["lidar_raw_m"], LIDAR_RANGE, cap[2] if pose.active else None, cap[3])
                integrate_scan(grid, o_xy, az, reach, hits)
                hum["floor_returns"] += int(np.sum((np.asarray(pkt["lidar_raw_m"]) < LIDAR_RANGE - 1e-3) & ~hits))
                hum["origin_offset_max"] = max(hum["origin_offset_max"], float(np.hypot(*(o_xy - np.asarray(cap[2] if pose.active else cap[0][:2])))))
            else:
                grid.update(xy_e[0], xy_e[1], yaw_e, sensors.angles, np.asarray(pkt["lidar_raw_m"]), LIDAR_RANGE)
            if learned is not None:
                learned["emap"].update(xy_e[0], xy_e[1], yaw_e, sensors.angles, np.asarray(pkt["lidar_raw_m"]))
        # plan on a timer, or when there is no plan yet (never with --policy: the learned actor does not use a plan)
        if learned is None and (plan is None or now - last_plan_t >= args.replan_s):
            new_plan = planner.plan(xy_e, goal)
            last_plan_t = now
            if new_plan:
                plan, wp_index = new_plan, 1 if len(new_plan) > 1 else 0
            elif humanoid:
                hum["planner_failures"] += 1        # diagnostic only: the previous plan is kept, as on the biped path
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
        if humanoid and sensors.packets[0] is not None and sensors.packets[0]["stamp_s"] != cap_stamp:
            # the pose the new packet was captured at (runner.d is unchanged since the loop top)
            cap_stamp = sensors.packets[0]["stamp_s"]
            cap = (runner.d.xpos[slot.body_id].copy(), runner.d.xmat[slot.body_id].copy(), np.array(xy_e, dtype=float), yaw_e - yaw)
        obs = runner.observe(0, command)
        if imu is not None:
            obs = imu.apply(obs, runner.d, dt)
        target = controller.update(obs)
        if not np.isfinite(target).all():
            outcome = "nonfinite_action"
            break
        runner.step([target])
        wall_steps += int(runner.hit_wall)
        if humanoid:
            hum["steps_by_state"][state] = hum["steps_by_state"].get(state, 0) + 1
            if runner.hit_wall:
                hum["wall_by_state"][state] = hum["wall_by_state"].get(state, 0) + 1
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
        if humanoid:
            tilt_now, sink_now = runner.tilt(0), spawn_z - float(runner.d.qpos[slot.qpos_adr + 2])
            hum["max_tilt"], hum["max_sink"] = max(hum["max_tilt"], tilt_now), max(hum["max_sink"], sink_now)
            fell_now = humanoid_fell(tilt_now, sink_now)
        else:
            fell_now = runner.tilt(0) >= 0.78
        if fell_now:
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
        "control": ("frozen_learned_biped_gait+lidar_egomap+LEARNED_navgym_policy(onnx, PPO in the gym; lidar at 10 Hz packets here, "
                    "every 0.04 s step in the gym; team_sensors speed brake still filters its commands); " if learned is not None
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
    if humanoid:
        labels = humanoid_labels(args.gait_run, pose.active, imu is not None)
        result["control"] = (f"frozen LEARNED humanoid gait {args.gait_run} (one checkpoint, 22-DoF) + SCRIPTED lidar_occupancy_map"
                             "+astar_unknown_free+turn_then_walk; "
                             + ("ORACLE pose and goal" if oracle_pose else "ORACLE goal; pose = oracle + injected error (pose_error)")
                             + ("" if imu is None else "; gait attitude = filter in the loop (imu)"))
        result["labels"] = labels
        result["variant"] = "humanoid"
        result["time_limit_s"] = args.time_limit
        result["settle_s"] = args.settle_s
        result["controller"].update({"turn_enter": ctrl.turn_enter, "turn_exit": ctrl.turn_exit, "wz_walk": ctrl.wz_walk,
                                     "k_yaw": ctrl.k_yaw, "waypoint_radius": ctrl.waypoint_radius, "goal_radius": ctrl.goal_radius,
                                     "overrides": controller_overrides(args)})
        result["robot"] = {"model": "berkeley_humanoid_lite.xml (22-DoF)", "sensor_mounts": mounts,
                           "planar_radius_m": HUMANOID_RADIUS_M, "biped_planar_radius_m": BIPED_RADIUS_M,
                           "planner_inflate_m": args.inflate, "planner_inflate_cells": planner.r_cells,
                           "corridor_clear_m": round(rm.CELL - rm.WALL_T, 3),
                           "nominal_free_channel_m": round(rm.CELL - rm.WALL_T - 2 * args.inflate, 3),
                           "note": "free channel = corridor - 2 x inflation, before the 0.10 m grid discretization (about 0.1 m less)"}
        result["fall_rule"] = {"rule": f"tilt > {HUMANOID_TILT_LIMIT_RAD} rad OR root sink > {HUMANOID_MAX_SINK_M} m below spawn "
                                       "(run_eval / harness, the rule behind the 7/60 push qualification)",
                               "max_tilt_rad": round(hum["max_tilt"], 3), "max_sink_m": round(hum["max_sink"], 3)}
        result["map_integration"] = {"rule": "each scan from the lidar's own pose at capture (base + R @ mount, full body rotation), "
                                             "rays projected on the floor plane; floor returns clear cells, mark nothing",
                                     "floor_returns": hum["floor_returns"], "max_origin_offset_from_base_m": round(hum["origin_offset_max"], 3)}
        result["planner_failures"] = {"count": hum["planner_failures"], "of_replans": planner.replans,
                                      "note": "A* found no path on the inflated map; the previous plan was kept (same as the biped path)"}
        result["wall_contact_steps_by_state"] = hum["wall_by_state"]
        result["steps_by_state"] = hum["steps_by_state"]
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
    ap.add_argument("--imu-panel", action="store_true",
                    help="with --render: add a simulated IM10A-like 10-axis IMU (datasheet noise, 100 Hz) as a trace strip "
                         "under the main view and a readout panel; display only, never in the control loop")
    ap.add_argument("--imu-panel-rate", type=float, default=100.0, help="simulated IMU output rate, Hz (IM10A: 0.2-200)")
    ap.add_argument("--rig-panels", action="store_true",
                    help="with --render: add a second column with RGB and rendered depth per eye from the stereo mount and "
                         "left-eye optical flow; display only (the brake still reads the 8x8 ray depth)")
    ap.add_argument("--rig-res", default="160x120", help="per-eye render size for --rig-panels, WxH")
    ap.add_argument("--policy", type=Path, default=None,
                    help="NavGym actor (.onnx from navgym_train.py): replaces the A* planner + turn-then-walk with the learned "
                         "policy; same lidar sectors, the same 0.2 m egocentric map built from the same raw rays, oracle pose + goal")
    ap.add_argument("--no-overwrite", action="store_true",
                    help="refuse to run if any per-seed JSON or the summary this run would write already exists")
    hm = ap.add_argument_group("22-DoF humanoid (opt-in; the default is the biped path, unchanged)")
    hm.add_argument("--variant", choices=("biped", "humanoid"), default="biped",
                    help="'humanoid': berkeley_humanoid_lite.xml + the QUALIFIED turning checkpoint arms-turn-turnboth-s0 "
                         "(default --gait for this variant), team_sensors deck mounts, run_eval's fall rule, humanoid labels")
    for knob, what in (("turn_enter", "heading error (rad) that starts a turn in place (default 0.40)"),
                       ("turn_exit", "heading error (rad) that ends it (default 0.15)"),
                       ("wz_walk", "yaw-rate cap while walking (rad/s, default 0.4)"),
                       ("waypoint_radius", "intermediate-waypoint acceptance radius (m, default 0.30); the goal radius "
                                           "is the judge's 0.30 m and is not a knob")):
        hm.add_argument("--" + knob.replace("_", "-"), type=float, default=None, help=what)
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

    if is_humanoid(args):
        if args.policy is not None:
            ap.error("--policy (the NavGym actor, scaled for the biped) is not supported with --variant humanoid")
        if args.gait is None:
            import glob
            hits = sorted(glob.glob(str(Path(args.upstream) / HUMANOID_GAIT_GLOB)))
            if len(hits) != 1:
                raise SystemExit(f"expected exactly one exported {HUMANOID_RUN} deploy.yaml, found {len(hits)}: {hits}")
            args.gait = Path(hits[0])
        gait = args.gait
        cfg = OmegaConf.load(gait)
        if cfg.num_actions != 22 or cfg.num_joints != 22 or cfg.num_observations != 75:
            raise SystemExit("--variant humanoid needs the 22-DoF, 75-observation humanoid gait")
        args.gait_run = gait_run_name(gait)
    else:
        gait = args.gait or (Path(args.upstream) / GAIT_DEFAULT)
        cfg = OmegaConf.load(gait)
        if cfg.num_actions != 12:
            raise SystemExit("the default (biped) variant needs the 12-DoF biped gait; a 22-DoF humanoid gait "
                             "runs with --variant humanoid")
    if args.no_overwrite:
        targets = [args.out_dir / f"seed{args.seed_start + k}{args.tag}.json" for k in range(args.seeds)] + [args.out_dir / f"summary{args.tag}.json"]
        present = [str(p) for p in targets if p.exists()]
        if present:
            raise SystemExit(f"--no-overwrite: refusing to run, outputs already exist: {present}")
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
        if is_humanoid(args):
            res["gait"].update({"variant": "humanoid", "run": args.gait_run, "label": res["labels"]["gait"]})
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
    if is_humanoid(args):
        summary["variant"] = "humanoid"
        summary["labels"] = humanoid_labels(args.gait_run, _pose_error_on(args), args.imu_source == "estimated")
        summary["settings"].update({"settle_s": args.settle_s, **{k_: getattr(args, k_) for k_ in CONTROLLER_KNOBS}})
        summary["frozen_humanoid_settings"] = HUMANOID_FROZEN
        summary["settings_match_frozen"] = all(summary["settings"][k_] == v for k_, v in HUMANOID_FROZEN.items())
        summary["humanoid_diagnostics"] = {
            "max_tilt_rad": [r["fall_rule"]["max_tilt_rad"] for r in results],
            "max_sink_m": [r["fall_rule"]["max_sink_m"] for r in results],
            "wall_contact_steps_by_state": [r["wall_contact_steps_by_state"] for r in results],
            "planner_failures": [r["planner_failures"]["count"] for r in results],
            "floor_returns": [r["map_integration"]["floor_returns"] for r in results],
            "max_origin_offset_from_base_m": [r["map_integration"]["max_origin_offset_from_base_m"] for r in results]}
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
                          "one seed = one maze; the multi-seed table is the evidence, this clip the illustration") if args.policy is None else
                         ("MuJoCo 3.3.5 | frozen learned biped gait (dr-default-s0) | (vx, wz) commands from the LEARNED NavGym policy "
                          f"({os.path.relpath(args.policy, REPO)}; PPO trained in the 2-D NavGym proxy, deployed here unchanged, deterministic mean) | "
                          "its observation: 36 lidar sector minima + a 0.2 m egocentric log-odds map built from the robot's own 108 lidar rays | "
                          "oracle pose and goal coordinate; the maze walls are seen only through the lidar | no planner, no scripted controller; "
                          "the team_sensors speed brake still filters the commands | one seed = one maze; the multi-seed transfer table is the evidence, "
                          "this clip the illustration")}
        if args.policy is not None:
            side["policy"] = {"onnx": os.path.relpath(args.policy, REPO), "onnx_sha256": sha256(Path(args.policy)),
                              "label": "LEARNED (PPO in NavGym, deployed on the physics robot); oracle pose + goal"}
        if is_humanoid(args):
            side["labels"] = r["labels"]
            side["scope"] = (f"MuJoCo {mujoco.__version__} | 22-DoF humanoid | LEARNED gait {args.gait_run} (ONE checkpoint, frozen) | "
                             "SCRIPTED lidar log-odds map + A* (unknown = free) + turn-then-walk (no sideways) | ORACLE pose and goal "
                             "coordinate; the maze itself is unknown to the planner | one seed = one maze; the multi-seed table is the "
                             "evidence, this clip the illustration")
        if getattr(args, "imu_panel", False):
            from bhl_robust.eval.imu_sim import IM10A_DATASHEET
            side["imu_panel"] = {"label": "SIMULATED IM10A-like 10-axis IMU, display only (never in the control loop)",
                                 "rate_hz": args.imu_panel_rate, "noise": IM10A_DATASHEET,
                                 "filter": f"Mahony kp={args.imu_kp} ki={args.imu_ki}, 6-axis, own random stream",
                                 "not_used": "magnetometer (approx. local field) and barometer are simulated and not used"}
        if getattr(args, "rig_panels", False):
            side["rig_panels"] = {"label": "display only: the brake reads the 8x8 ray-cast depth, the map the lidar",
                                  "rgb_depth": f"MuJoCo renders at the stereo mount, {args.rig_res} per eye, 20 deg down, 60.5 deg vfov",
                                  "optical_flow": "OpenCV Farneback on consecutive left-eye frames (HSV: hue = direction)"}
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


def humanoid_verdict_main(argv) -> int:
    """`maze_explore.py humanoid-verdict <root>`: HUMANOID_MAZE_RULE over <root>/humanoid-hard-6x6 and
    <root>/biped-hard-6x6. An existing verdict JSON is never overwritten: it is re-printed."""
    ap = argparse.ArgumentParser(prog="maze_explore.py humanoid-verdict")
    ap.add_argument("root", type=Path)
    ap.add_argument("--humanoid-dir", default="humanoid-hard-6x6")
    ap.add_argument("--biped-dir", default="biped-hard-6x6")
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/verdict.json")
    a = ap.parse_args(argv)
    out = a.out or a.root / "verdict.json"
    if out.exists():
        v = json.loads(out.read_text())
        print(f"verdict already recorded, not overwritten: {out}")
    else:
        v = humanoid_maze_verdict(*load_arm(a.root / a.humanoid_dir), *load_arm(a.root / a.biped_dir))
        if v["verdict"] != "INCOMPLETE":           # an INCOMPLETE reading is printed, not recorded
            out.write_text(json.dumps(v, indent=2) + "\n")
            print(f"wrote {out}")
        else:
            print("INCOMPLETE problems: " + "; ".join(v["problems"]))
    print(f"MAZE-HUMANOID VERDICT: {v['verdict']} -- {v['detail']}", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sf-verdict":
        sys.exit(verdict_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "humanoid-verdict":
        sys.exit(humanoid_verdict_main(sys.argv[2:]))
    sys.exit(main())
