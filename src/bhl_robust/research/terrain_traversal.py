"""Sensor-informed traversal with a frozen gait and independent outcome scoring.

Only raw 3D ray returns and simulated IMU orientation enter terrain inference.
The controller never receives terrain names, height-field samples, evaluator
ground height, robot translation, or goal-crossing labels. Simulator translation
is used by the evaluator and by the ray simulator, as for any simulated sensor.
This is an instantaneous-scan, gravity-aligned geometric baseline, not SLAM.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from bhl_robust.research.sensor_geometry import TerrainConfig, elevation_map, terrain_metrics


TERRAINS = ("flat", "small_steps", "ramp")
ARMS = ("baseline", "lidar", "lidar50")


@dataclass(frozen=True)
class TraversalSettings:
    seconds: float = 40.
    settle_seconds: float = 1.
    goal_distance_m: float = 5.
    corridor_half_width_m: float = .75
    desired_speed_m_s: float = .30
    caution_speed_m_s: float = .20
    lidar_hz: float = 5.
    lidar_rings: int = 16
    lidar_columns: int = 64
    lidar_min_elevation_deg: float = -75.
    lidar_max_elevation_deg: float = -12.
    lidar_half_azimuth_deg: float = 60.
    lidar_max_range_m: float = 5.
    # Body origin is near the feet in the shipped BHL MJCF.
    lidar_mount_x_m: float = .10
    lidar_mount_z_m: float = .38
    map_resolution_m: float = .10
    lookahead_min_m: float = .20
    lookahead_max_m: float = .70
    footprint_half_width_m: float = .18
    minimum_known_fraction: float = .55
    caution_slope_deg: float = 2.
    caution_roughness_m: float = .003
    caution_step_m: float = .010
    stop_slope_deg: float = 8.
    stop_roughness_m: float = .015
    stop_step_m: float = .025
    fall_tilt_rad: float = .78
    fall_local_sink_m: float = .25

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                if name not in ("lidar_min_elevation_deg", "lidar_max_elevation_deg"):
                    raise ValueError(f"invalid setting {name}")
        if not (self.seconds > self.settle_seconds and
                0 < self.minimum_known_fraction <= 1 and
                self.lookahead_min_m < self.lookahead_max_m and
                -90 < self.lidar_min_elevation_deg < self.lidar_max_elevation_deg < 0):
            raise ValueError("invalid traversal timing, support or sensor bounds")
        if self.lidar_rings < 3 or self.lidar_columns < 3:
            raise ValueError("3D lidar requires multiple elevation and azimuth rays")

    def map_config(self) -> TerrainConfig:
        return TerrainConfig(x_min_m=.10, x_max_m=1.60, y_min_m=-.50,
                             y_max_m=.50, resolution_m=self.map_resolution_m,
                             min_points=1, slope_hazard_deg=self.stop_slope_deg,
                             roughness_hazard_m=self.stop_roughness_m,
                             step_hazard_m=self.stop_step_m)


def quaternion_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-3:
        raise ValueError("finite unit IMU quaternion required")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def gravity_aligned_points(points_l: np.ndarray, imu_quaternion_wxyz: np.ndarray,
                           settings: TraversalSettings) -> np.ndarray:
    """Keep local yaw, remove roll/pitch using IMU attitude; no translation pose."""
    points_l = np.asarray(points_l, dtype=float)
    if points_l.ndim != 2 or points_l.shape[1] != 3:
        raise ValueError("raw 3D lidar points must have shape (N,3)")
    rotation = quaternion_rotation(imu_quaternion_wxyz)
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    r_w_heading = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    mount = np.array([settings.lidar_mount_x_m, 0, settings.lidar_mount_z_m])
    return (points_l + mount) @ (r_w_heading.T @ rotation).T


def infer_terrain(points_l: np.ndarray, imu_quaternion_wxyz: np.ndarray,
                  settings: TraversalSettings = TraversalSettings()) -> dict:
    settings.validate()
    return elevation_map(gravity_aligned_points(points_l, imu_quaternion_wxyz, settings),
                         settings.map_config(), lidar_dimension=3)


def terrain_command(terrain: dict, settings: TraversalSettings = TraversalSettings()) -> tuple[np.ndarray, dict]:
    """Fixed support/risk governor. Unknown cells never count as traversable."""
    settings.validate()
    x, y = np.meshgrid(terrain["x_m"], terrain["y_m"])
    corridor = ((x >= settings.lookahead_min_m) & (x <= settings.lookahead_max_m)
                & (np.abs(y) <= settings.footprint_half_width_m))
    known = np.asarray(terrain["known"], dtype=bool) & corridor
    denominator = int(corridor.sum())
    fraction = float(known.sum()/denominator) if denominator else 0.
    diagnostic = {"corridor_cells": denominator, "known_corridor_cells": int(known.sum()),
                  "known_fraction": fraction, "reason": "nominal", "speed_m_s": 0.}
    if fraction < settings.minimum_known_fraction:
        diagnostic["reason"] = "unknown_support_stop"
        return np.zeros(3), diagnostic
    maxima = {}
    for name in ("slope_deg", "roughness_m", "neighbor_step_m"):
        values = np.asarray(terrain[name], dtype=float)[known]
        if not len(values) or not np.isfinite(values).all():
            diagnostic["reason"] = "invalid_map_stop"
            return np.zeros(3), diagnostic
        maxima[name] = float(values.max())
    diagnostic.update(maxima)
    if (maxima["slope_deg"] >= settings.stop_slope_deg or
            maxima["roughness_m"] >= settings.stop_roughness_m or
            maxima["neighbor_step_m"] >= settings.stop_step_m):
        diagnostic["reason"] = "observed_hazard_stop"
        return np.zeros(3), diagnostic
    caution = (maxima["slope_deg"] >= settings.caution_slope_deg or
               maxima["roughness_m"] >= settings.caution_roughness_m or
               maxima["neighbor_step_m"] >= settings.caution_step_m)
    speed = settings.caution_speed_m_s if caution else settings.desired_speed_m_s
    diagnostic.update(reason="observed_caution" if caution else "nominal", speed_m_s=speed)
    return np.array([speed, 0., 0.]), diagnostic


def traversal_outcome(progress_m: float | None, goal_crossed: bool, fell: bool,
                      collision: bool, nonfinite: bool, elapsed_s: float,
                      settings: TraversalSettings = TraversalSettings()) -> dict:
    """A stopped upright robot is a timeout, never a successful traversal."""
    finite = progress_m is not None and math.isfinite(progress_m)
    complete_horizon = math.isfinite(elapsed_s) and elapsed_s >= settings.seconds - 1e-6
    success = bool(goal_crossed and finite and complete_horizon and
                   not fell and not collision and not nonfinite)
    return {"success": success, "goal_crossed": bool(goal_crossed), "fell": bool(fell),
            "collision": bool(collision), "nonfinite": bool(nonfinite),
            "full_matched_horizon": bool(complete_horizon),
            "forward_progress_m": float(progress_m) if finite else None,
            "elapsed_s": float(elapsed_s),
            "failure": ("nonfinite" if nonfinite or not finite else "fall" if fell else
                        "collision" if collision else "timeout" if not goal_crossed else
                        "incomplete_horizon" if not complete_horizon else None)}


def qualified_actor(rows: list[dict], expected_episodes: int = 6) -> bool:
    if len(rows) != expected_episodes:
        return False
    identities = {(r.get("terrain"), r.get("group")) for r in rows}
    if (len(identities) != expected_episodes or {r.get("terrain") for r in rows} != set(TERRAINS)
            or len({r.get("group") for r in rows}) != 2):
        return False
    for row in rows:
        try:
            rescored = traversal_outcome(row["forward_progress_m"], row["goal_crossed"],
                row["fell"], row["collision"], row["nonfinite"], row["elapsed_s"])
        except (KeyError, TypeError, ValueError):
            return False
        if row.get("success") is not True or not rescored["success"]:
            return False
    return True


def prepare_course(flat_scene: Path, output: Path, terrain: str, group: int,
                   settings: TraversalSettings = TraversalSettings()) -> tuple[Path, dict]:
    """Physical five-metre fixtures, independent of perception inputs.

    Steps have 1 cm rises and 36 cm treads. Ramps have 3 degree grades.
    Group changes onset by +/-2.5 cm and friction by +/-0.02 about 0.8;
    neither is supplied to the terrain controller.
    """
    from PIL import Image
    settings.validate()
    if terrain not in TERRAINS:
        raise ValueError(f"unknown terrain {terrain}")
    output.mkdir(parents=True, exist_ok=False)
    root = ET.parse(flat_scene).getroot()
    for element in root.iter():
        if "file" in element.attrib:
            p = Path(element.attrib["file"])
            if not p.is_absolute():
                element.set("file", str((flat_scene.parent/p).resolve()))
    world = root.find("worldbody")
    asset = root.find("asset")
    if world is None:
        raise ValueError("scene needs worldbody")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    floors = [g for g in world.findall("geom") if g.get("name") == "floor"]
    if len(floors) != 1:
        raise ValueError("exactly one original floor required")
    world.remove(floors[0])
    rng = np.random.default_rng(group)
    onset = .8 + rng.uniform(-.025, .025)
    friction = .8 + rng.uniform(-.02, .02)
    contact = dict(group="0", friction=f"{friction:.9f} .005 .0001", priority="1",
                   solref=".006 1", solimp=".9 .95 .001", rgba=".42 .52 .38 1")
    ET.SubElement(world, "geom", name="terrain_base", type="box", size="3.5 1.0 .05",
                  pos="2.5 0 -.05", **contact)
    metadata = {"terrain": terrain, "group": group, "course_length_m": settings.goal_distance_m,
                "onset_m": float(onset), "friction": float(friction),
                "group_physical_variation": "onset +/-2.5 cm; friction .8 +/- .02",
                "simulation_scope": "rigid MuJoCo geometric/contact fixtures"}
    if terrain == "small_steps":
        for k, level in enumerate((1, 2, 3, 2, 1)):
            height = .01 * level
            ET.SubElement(world, "geom", name=f"terrain_step{k}", type="box",
                          size=f".18 1.0 {height/2:.6f}",
                          pos=f"{onset+.36*k+.18:.9f} 0 {height/2:.6f}", **contact)
        metadata.update(step_rise_m=.01, max_height_m=.03, tread_m=.36)
    elif terrain == "ramp":
        # Quantized PNG is the actual physics height field, not an inference input.
        xs = np.linspace(-1., 6., 513)
        height = np.tan(np.deg2rad(3.)) * (np.clip(xs-onset, 0, 1.8)-np.clip(xs-onset-2.15, 0, 1.8))
        height = np.maximum(height, 0)
        scale = float(height.max())
        pixels = np.tile(np.rint(height/scale*255).astype(np.uint8), (65, 1))
        heightfile = output / "ramp_height.png"
        Image.fromarray(pixels).save(heightfile)
        ET.SubElement(asset, "hfield", name="terrain_hfield", file=str(heightfile),
                      size=f"3.5 1.0 {scale:.12f} .05")
        world.find("geom[@name='terrain_base']").set("pos", "2.5 0 -.052")
        ET.SubElement(world, "geom", name="terrain_ramp", type="hfield", hfield="terrain_hfield",
                      pos="2.5 0 0", **contact)
        metadata.update(grade_deg=3., height_quantization_m=scale/255, max_height_m=scale)
    for side in (-1, 1):
        ET.SubElement(world, "geom", name=f"course_wall_{side}", type="box",
                      size="3.5 .025 .3", pos=f"2.5 {side*(settings.corridor_half_width_m+.025)} .3",
                      group="0", rgba=".5 .5 .5 1")
    scene = output / "course.xml"
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(scene, encoding="unicode")
    return scene, metadata


def make_environment(cfg, flat_scene: Path, scene: Path):
    """Reuse the shipped gait observation/PD path without overwriting the ramp."""
    import mujoco
    from bhl_robust.eval.harness import HeadlessMujocoEnv

    class CourseEnvironment(HeadlessMujocoEnv):
        def ground_z(self, x=None, y=None):
            if x is None:
                x, y = self.mj_data.qpos[:2]
            geom = np.zeros(1, dtype=np.int32)
            distance = mujoco.mj_ray(self.mj_model, self.mj_data, np.array([x, y, 3.]),
                                     np.array([0., 0., -1.]), np.array([1, 0, 0, 0, 0, 0], np.uint8),
                                     1, -1, geom)
            if distance < 0:
                return 0.
            return 3. - float(distance)

        def step(self, target_joint_pos):
            targets = np.zeros(self.num_joints, dtype=np.float32)
            targets[self.action_indices] = target_joint_pos
            self.hit_wall = False
            for _ in range(self.substeps):
                torque = self.joint_kp*(targets-self._joint_pos())-self.joint_kd*self._joint_vel()
                self.mj_data.ctrl[:] = np.clip(torque, -self.effort_limits, self.effort_limits)
                mujoco.mj_step(self.mj_model, self.mj_data)
                for contact in self.mj_data.contact:
                    g1, g2 = int(contact.geom1), int(contact.geom2)
                    if ((g1 in self.wall_ids and self.mj_model.geom_bodyid[g2] != 0) or
                            (g2 in self.wall_ids and self.mj_model.geom_bodyid[g1] != 0)):
                        self.hit_wall = True

    env = CourseEnvironment(cfg, flat_scene, terrain_difficulty=0.)
    env.mj_model = mujoco.MjModel.from_xml_path(str(scene))
    env.mj_model.opt.timestep = cfg.physics_dt
    env.mj_data = mujoco.MjData(env.mj_model)
    env.terrain = lambda x, y: env.ground_z(x, y)
    env.wall_ids = {g for g in range(env.mj_model.ngeom)
                    if (mujoco.mj_id2name(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("course_wall_")}
    mujoco.mj_forward(env.mj_model, env.mj_data)
    return env


def capture_lidar(env, settings: TraversalSettings = TraversalSettings()) -> np.ndarray:
    """Raw simultaneous 3D rays. Robot geoms excluded, no ideal terrain lookup."""
    import mujoco
    rotation = quaternion_rotation(env._base_quat())
    origin = np.asarray(env.mj_data.qpos[:3], dtype=float) + rotation @ np.array(
        [settings.lidar_mount_x_m, 0., settings.lidar_mount_z_m])
    azimuth = np.deg2rad(np.linspace(-settings.lidar_half_azimuth_deg,
                                    settings.lidar_half_azimuth_deg, settings.lidar_columns))
    elevation = np.deg2rad(np.linspace(settings.lidar_min_elevation_deg,
                                     settings.lidar_max_elevation_deg, settings.lidar_rings))
    points = []
    geom = np.zeros(1, dtype=np.int32)
    for el in elevation:
        for az in azimuth:
            direction = np.array([math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])
            distance = mujoco.mj_ray(env.mj_model, env.mj_data, origin, rotation @ direction,
                                     np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), 1, -1, geom)
            if 0 < distance <= settings.lidar_max_range_m:
                points.append(direction * distance)
    return np.asarray(points, dtype=np.float32).reshape(-1, 3)


def evaluate_terrain_map(env, inferred: dict, settings: TraversalSettings) -> tuple[dict, np.ndarray, np.ndarray]:
    """Evaluator-only dense vertical rays, acquired after controller inference.

    Ground truth is defined in the same gravity-aligned local coordinates as
    inference but does not reuse any lidar return. Hazards are geometric map
    hazards, not semantic labels or a physical traversability certificate.
    """
    rotation = quaternion_rotation(env._base_quat())
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    xx, yy = np.meshgrid(inferred["x_m"], inferred["y_m"])
    base = np.asarray(env.mj_data.qpos[:3], dtype=float)
    truth = np.empty(xx.shape)
    for row, col in np.ndindex(xx.shape):
        x_world = base[0]+c*xx[row, col]-s*yy[row, col]
        y_world = base[1]+s*xx[row, col]+c*yy[row, col]
        truth[row, col] = env.ground_z(x_world, y_world)-base[2]
    dense = np.column_stack((xx.ravel(), yy.ravel(), truth.ravel()))
    reference = elevation_map(dense, settings.map_config(), lidar_dimension=3)
    metrics = terrain_metrics(inferred, truth, reference["hazard"])
    metrics["truth_source"] = "independent dense vertical scene rays, after inference; geometric hazard thresholds"
    return metrics, truth, reference["hazard"]


def evaluate_episode(env, controller, arm: str, seed: int, settings: TraversalSettings,
                     scan_output: Path, *, sensor_seed: int) -> dict:
    """Run the real learned controller and save every raw scan and every command."""
    from bhl_robust.eval.harness import EvalConfig
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    settings.validate()
    env.reset(np.random.default_rng(seed), EvalConfig())
    controller.prev_actions[:] = 0
    controller.policy_observations[:] = 0
    dt = float(env.cfg.policy_dt)
    sensor_interval = int(round(1/settings.lidar_hz/dt))
    if sensor_interval < 1 or not math.isclose(sensor_interval*dt, 1/settings.lidar_hz, abs_tol=1e-8):
        raise ValueError("sensor period must be an integral number of gait updates")
    start = env.base_xy.copy()
    previous_xy = start.copy()
    goal_crossed = False
    crossed_at_s = None
    fell = collision = nonfinite = False
    path = 0.
    traces, scans, scan_times, imu, indices, shapes, maps = [], [], [], [], [], [], []
    map_metrics, truth_heights, truth_hazards = [], [], []
    last_command = np.zeros(3)
    diagnostic = {"reason": "not_sampled", "speed_m_s": 0.}
    scan_rng = np.random.default_rng(sensor_seed)
    for step in range(int(round(settings.seconds/dt))):
        now = step*dt
        if step % sensor_interval == 0:
            raw = capture_lidar(env, settings)
            keep = np.sort(scan_rng.permutation(len(raw))[:len(raw)//2]) if arm == "lidar50" else np.arange(len(raw))
            attitude = env._base_quat().copy()
            terrain = infer_terrain(raw[keep], attitude, settings)
            last_command, diagnostic = terrain_command(terrain, settings)
            # Evaluator input is deliberately introduced only after both
            # terrain inference and the command have already been computed.
            scored_map, truth_height, truth_hazard = evaluate_terrain_map(env, terrain, settings)
            map_metrics.append({"time_s": now, **scored_map})
            truth_heights.append(truth_height.astype(np.float32))
            truth_hazards.append(truth_hazard)
            scans.append(raw)
            scan_times.append(now)
            imu.append(attitude)
            indices.append(keep)
            shapes.append(len(raw))
            maps.append(np.stack([terrain[k] for k in ("height_m", "slope_deg", "roughness_m", "neighbor_step_m")]).astype(np.float32))
        if now < settings.settle_seconds:
            command = np.zeros(3)
            reason = "settle"
        elif goal_crossed:
            command = np.zeros(3)
            reason = "evaluator_goal_stop"
        elif arm == "baseline":
            command = np.array([settings.desired_speed_m_s, 0., 0.])
            reason = "blind_reference"
        else:
            command = last_command.copy()
            reason = diagnostic["reason"]
        target = controller.update(env.robot_observations(tuple(command)))
        if not np.isfinite(target).all():
            nonfinite = True
            break
        env.step(target)
        if not np.isfinite(env.mj_data.qpos).all() or not np.isfinite(env.mj_data.qvel).all():
            nonfinite = True
            break
        xy = env.base_xy.copy()
        path += float(np.linalg.norm(xy-previous_xy))
        previous_xy = xy
        progress = float(xy[0]-start[0])
        tilt, sink = env.tilt_rad, env.sink_m
        fell = tilt >= settings.fall_tilt_rad or sink > settings.fall_local_sink_m
        collision = collision or env.hit_wall
        if progress >= settings.goal_distance_m and abs(float(xy[1]-start[1])) <= settings.corridor_half_width_m:
            goal_crossed = True
            if crossed_at_s is None:
                crossed_at_s = (step+1)*dt
        traces.append({"time_s": (step+1)*dt, "xy_evaluator_m": xy.tolist(),
                       "progress_evaluator_m": progress, "tilt_evaluator_rad": tilt,
                       "local_sink_evaluator_m": sink, "command_m_s_rad_s": command.tolist(),
                       "command_reason": reason, "map_diagnostic": diagnostic,
                       "wall_contact": bool(env.hit_wall), "goal_crossed": goal_crossed})
        if fell or collision:
            break
    elapsed = (step+1)*dt
    progress = float(env.base_xy[0]-start[0]) if not nonfinite else None
    offsets = np.concatenate(([0], np.cumsum(shapes))).astype(np.int64)
    kept_lengths = np.asarray([len(k) for k in indices], dtype=np.int64)
    kept_offsets = np.concatenate(([0], np.cumsum(kept_lengths))).astype(np.int64)
    np.savez_compressed(scan_output, raw_points_l_m=np.concatenate(scans) if scans else np.empty((0, 3), np.float32),
                        raw_offsets=offsets, timestamp_s=np.asarray(scan_times),
                        imu_quaternion_wxyz=np.asarray(imu),
                        retained_indices=np.concatenate(indices) if indices else np.empty(0, np.int64),
                        retained_offsets=kept_offsets, inferred_maps=np.asarray(maps),
                        evaluator_truth_height_m=np.asarray(truth_heights),
                        evaluator_truth_hazard=np.asarray(truth_hazards),
                        map_x_m=settings.map_config().axes()[0], map_y_m=settings.map_config().axes()[1])
    result = traversal_outcome(progress, goal_crossed, fell, collision, nonfinite, elapsed, settings)
    result.update(seed=seed, sensor_seed=sensor_seed, arm=arm, path_m=path,
                  first_goal_crossing_s=crossed_at_s, raw_scan_count=len(scans),
                  raw_point_count=int(sum(shapes)), trace=traces, terrain_map_metrics=map_metrics,
                  pre_goal_unknown_stop_steps=sum(r["command_reason"] == "unknown_support_stop" for r in traces),
                  pre_goal_observed_hazard_stop_steps=sum(r["command_reason"] == "observed_hazard_stop" for r in traces),
                  inference_scope="current raw 3D scan plus simulated IMU attitude; no translational pose/map/terrain oracle",
                  evaluator_scope="simulator pose and independent downward terrain rays for progress/local sink only",
                  goal_stop_scope="evaluator stops command after goal crossing, then full-horizon survival is still required")
    return result
