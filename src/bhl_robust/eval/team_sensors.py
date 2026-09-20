"""Actual MuJoCo ray/IMU inputs for an optional team-navigation brake.

Depth is paired *idealized ray depth*, not RGB stereo matching. There is no
detector model here and no synthetic SSD output. Mounts/noise are experiment
choices; they are not calibration for an unidentified USB AHRS module.
"""

from __future__ import annotations

import mujoco
import numpy as np

from bhl_robust.sensor_io import SensorTiming, imu_features


MODES = ("off", "record", "reactive", "reactive_dropout")
LIDAR_RAYS = 108
LIDAR_SECTORS = 36
LIDAR_RANGE = 12.0
DEPTH_RANGE = 6.0
DEPTH_SIDE = 8
EXTERO_PERIOD = .1
# Full humanoid asset: base body origin is at the feet, torso is near z=.71.
# These are full-body mounts, not the biped's .34 m lidar mount.
LIDAR_MOUNT = np.array([.12, 0.0, .72])
STEREO_BASELINE = .06
STEREO_CENTER = np.array([.12, 0.0, .70])


def ray_pattern():
    """Directions are normalized, body-attached, with image +v downward."""
    angles = np.linspace(-np.pi, np.pi, LIDAR_RAYS, endpoint=False)
    lidar = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(LIDAR_RAYS)])
    coord = ((np.arange(DEPTH_SIDE) + .5) / DEPTH_SIDE * 2 - 1) * np.tan(np.deg2rad(30.25))
    u, v = np.meshgrid(coord, coord)
    optical = np.column_stack([np.ones(u.size), -u.ravel(), -v.ravel()])
    optical /= np.linalg.norm(optical, axis=1, keepdims=True)
    pitch = np.deg2rad(20)
    rotation = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0],
                         [-np.sin(pitch), 0, np.cos(pitch)]])
    return angles, lidar, optical @ rotation.T, optical[:, 0]


def brake_command(command, *, lidar_m, depth_m, imu, fresh):
    """Consume directional ranges and IMU; do not read simulator object IDs.

    Missing/stale exteroception stops translation. Otherwise the nearest hit
    within 25 degrees of travel sets a speed cap. Forward paired depth can
    tighten that cap. IMU tilt/angular speed adds a conservative speed limit.
    The frozen low-level policy keeps receiving its normal proprioception.
    """
    output = np.asarray(command, dtype=float).copy()
    info = {"range_scale": 1.0, "imu_scale": 1.0, "stale_stop": False}
    if not fresh or not bool(imu[-1]):
        output[:2] = 0.0
        info.update(range_scale=0.0, stale_stop=True)
        return output, info
    if np.linalg.norm(output[:2]) < 1e-8:
        return output, info
    heading = np.arctan2(output[1], output[0])
    angles = np.linspace(-np.pi, np.pi, LIDAR_RAYS, endpoint=False)
    centers = angles.reshape(LIDAR_SECTORS, -1).mean(axis=1)
    difference = np.arctan2(np.sin(centers-heading), np.cos(centers-heading))
    selected = np.abs(difference) <= np.deg2rad(25)
    clearance = float(np.min(np.asarray(lidar_m)[selected]))
    if abs(heading) < np.deg2rad(25):
        # Upper central pixels keep the flat floor out of the close-stop cone.
        # The input is optical-axis depth, so no ground-truth geom labels enter.
        clearance = min(clearance, float(np.min(np.asarray(depth_m)[:, 1:4, 2:6])))
    scale = float(np.clip((clearance - .42) / .48, 0.0, 1.0))
    tilt = float(np.arccos(np.clip(-imu[2], -1, 1)))
    imu_scale = .5 if tilt > .35 or np.linalg.norm(imu[3:6]) > 2.5 else 1.0
    output[:2] *= min(scale, imu_scale)
    info.update(range_scale=scale, imu_scale=imu_scale)
    return output, info


class TeamSensors:
    """Sample each robot's own sensors without changing the physical model."""

    def __init__(self, model, slots, owners, *, mode, seed, dropout_probability=.35):
        if mode not in MODES or mode == "off":
            raise ValueError("sensor sampler requires a non-off sensor mode")
        if not 0 <= dropout_probability <= 1:
            raise ValueError("dropout probability must be in [0,1]")
        self.model, self.slots, self.mode = model, slots, mode
        self.rng = np.random.default_rng(seed + 170009)
        self.dropout = dropout_probability if mode == "reactive_dropout" else 0.0
        self.timing = SensorTiming(max_age_s=.15)
        self.angles, self.lidar_dirs, self.depth_dirs, self.depth_cos = ray_pattern()
        self.own_geoms = [np.flatnonzero(owners == i) for i in range(len(slots))]
        self.accel_adrs = []
        for slot in slots:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, slot.prefix + "imu_acc")
            if sid < 0:
                raise ValueError(f"missing actual accelerometer for {slot.prefix}")
            self.accel_adrs.append(int(model.sensor_adr[sid]))
        self.next_capture = np.zeros(len(slots))
        self.packets = [None] * len(slots)
        self.latest = [None] * len(slots)
        self.stats = {"received_extero_packets": 0, "dropped_extero_packets": 0,
                      "braked_robot_steps": 0, "stale_stop_robot_steps": 0,
                      "range_limited_robot_steps": 0, "imu_limited_robot_steps": 0,
                      "sampled_robot_steps": 0}

    def _rays(self, data, origin, directions, maximum):
        n = len(directions)
        ids = np.empty(n, dtype=np.int32)
        distance = np.empty(n, dtype=np.float64)
        # Visual triangle meshes and the current robot are excluded; actual
        # collision geometry of teammates and the world remains observable.
        groups = np.array([1, 1, 0, 1, 1, 0], dtype=np.uint8)
        mujoco.mj_multiRay(self.model, data, np.asarray(origin, dtype=np.float64),
                          np.asarray(directions, dtype=np.float64).ravel(), groups,
                          1, -1, ids, distance, n, maximum)
        return np.where(distance >= 0, np.minimum(distance, maximum), maximum)

    def _capture(self, data, i, now):
        slot = self.slots[i]
        rotation = data.xmat[slot.body_id].reshape(3, 3)
        position = data.xpos[slot.body_id]
        geoms = self.own_geoms[i]
        previous = self.model.geom_group[geoms].copy()
        try:
            # geom_group affects ray/render filtering only, never contact masks.
            self.model.geom_group[geoms] = 5
            raw = self._rays(data, position + rotation @ LIDAR_MOUNT,
                             self.lidar_dirs @ rotation.T, LIDAR_RANGE)
            depths = []
            for side in (+1, -1):
                mount = STEREO_CENTER + np.array([0, side * STEREO_BASELINE / 2, 0])
                ray_range = self._rays(data, position + rotation @ mount,
                                       self.depth_dirs @ rotation.T, DEPTH_RANGE)
                depths.append((ray_range * self.depth_cos).reshape(DEPTH_SIDE, DEPTH_SIDE))
        finally:
            self.model.geom_group[geoms] = previous
        return {"stamp_s": now, "lidar_m": raw.reshape(LIDAR_SECTORS, -1).min(axis=1),
                "paired_depth_m": np.array(depths)}

    def filter_commands(self, data, commands, now):
        outputs = []
        for i, (slot, command) in enumerate(zip(self.slots, commands)):
            if now + 1e-9 >= self.next_capture[i]:
                while self.next_capture[i] <= now + 1e-9:
                    self.next_capture[i] += EXTERO_PERIOD
                if self.rng.random() < self.dropout:
                    self.stats["dropped_extero_packets"] += 1
                else:
                    self.packets[i] = self._capture(data, i, now)
                    self.stats["received_extero_packets"] += 1
            q = data.sensordata[slot.quat_adr:slot.quat_adr + 4]
            gyro = data.sensordata[slot.gyro_adr:slot.gyro_adr + 3]
            acc = self.accel_adrs[i]
            imu = imu_features(q[[1, 2, 3, 0]], gyro, data.sensordata[acc:acc + 3],
                               stamp_s=now, now_s=now, timing=self.timing)
            packet = self.packets[i]
            fresh = packet is not None and self.timing.fresh(packet["stamp_s"], now)
            lidar = packet["lidar_m"] if packet is not None else np.zeros(LIDAR_SECTORS)
            depth = packet["paired_depth_m"] if packet is not None else np.zeros((2, DEPTH_SIDE, DEPTH_SIDE))
            filtered, info = brake_command(command, lidar_m=lidar, depth_m=depth, imu=imu, fresh=fresh)
            consumed = self.mode != "record"
            self.stats["sampled_robot_steps"] += 1
            self.stats["braked_robot_steps"] += int(consumed and not np.allclose(filtered, command))
            self.stats["stale_stop_robot_steps"] += int(consumed and info["stale_stop"])
            self.stats["range_limited_robot_steps"] += int(consumed and not info["stale_stop"]
                                                         and info["range_scale"] < 1)
            self.stats["imu_limited_robot_steps"] += int(consumed and info["imu_scale"] < 1)
            outputs.append(filtered if consumed else command)
            self.latest[i] = {"imu_stamp_s": now, "imu_gravity_gyro_specific_force_valid": imu.tolist(),
                              "extero_stamp_s": packet["stamp_s"] if packet is not None else None,
                              "extero_fresh": bool(fresh), "lidar_sector_m": lidar.tolist(),
                              "paired_idealized_depth_m": depth.tolist(),
                              "brake": info, "consumed_for_braking": consumed,
                              "ssd_status": "not_configured_no_rgb_or_weights"}
        return outputs

    @staticmethod
    def metadata():
        return {"lidar": {"rays": LIDAR_RAYS, "sectors": LIDAR_SECTORS, "max_range_m": LIDAR_RANGE,
                           "body_mount_m": LIDAR_MOUNT.tolist(), "nominal_hz": 1 / EXTERO_PERIOD},
                "stereo": {"kind": "paired_idealized_ray_depth_not_rgb_matching", "side": DEPTH_SIDE,
                            "baseline_m": STEREO_BASELINE, "body_center_m": STEREO_CENTER.tolist(),
                            "pitch_down_deg": 20, "vfov_deg": 60.5, "max_range_m": DEPTH_RANGE},
                "imu": {"kind": "mujoco_site_quaternion_gyro_accelerometer", "frame": "body_to_ENU_xyzw",
                        "gravity_unitless": True, "gyro_units": "rad/s", "specific_force_units": "m/s^2",
                        "hardware_calibrated": False},
                "localization": "oracle_simulator_pose_and_map",
                "ssd": "not_configured_no_rgb_or_weights"}
