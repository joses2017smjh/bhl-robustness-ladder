"""Masked, timestamped ray packets. No geometry IDs or global poses leave here."""
import numpy as np
import mujoco
from bhl_robust.eval.team_sensors import TeamSensors
from bhl_robust.sensor_io import SensorTiming

ARMS = ("blind", "lidar", "stereo", "both")
FAILURES = ("normal", "lidar_missing", "stereo_missing", "both_missing",
            "lidar_stale", "stereo_stale", "noisy", "occluded", "intermittent")
PROPRIO = 61
LIDAR = 36
DEPTH = 128
FRAME = PROPRIO + 2*LIDAR+1 + 2*DEPTH+1
HISTORY = 4


def pack(values, stamp, now, maximum):
    values = np.asarray(values, dtype=float).ravel()
    fresh = stamp is not None and SensorTiming().fresh(stamp, now)
    valid = np.isfinite(values) & (values > 0) & (values <= maximum) & fresh
    age = 1. if stamp is None else np.clip((now-stamp)/.15, 0, 1)
    # Invalid has zero value AND zero validity, never a valid free-space return.
    return np.r_[np.where(valid, values/maximum, 0), valid.astype(float), age].astype(np.float32)


class MissionSensors(TeamSensors):
    def __init__(self, model, slots, owners, seed, arm, failure="normal"):
        super().__init__(model, slots, owners, mode="record", seed=seed)
        if arm not in ARMS or failure not in FAILURES:
            raise ValueError("unknown sensor condition")
        self.arm, self.failure = arm, failure
        self.packet = None
        # Separate RNG from spawn/layout/action RNG preserves matched scenarios.
        self.noise_rng = np.random.default_rng(seed+901001)

    def _rays(self, data, origin, directions, maximum):
        n = len(directions)
        ids = np.empty(n, np.int32)
        distance = np.empty(n, np.float64)
        mujoco.mj_multiRay(self.model, data, np.asarray(origin, np.float64),
                          np.asarray(directions, np.float64).ravel(),
                          np.array([1, 1, 0, 1, 1, 0], np.uint8),
                          1, -1, ids, distance, n, maximum)
        return np.where((distance > 0) & (distance <= maximum), distance, np.nan)

    def capture(self, data, now):
        if now+1e-9 < self.next_capture[0]:
            return
        while self.next_capture[0] <= now+1e-9:
            self.next_capture[0] += .1
        if self.failure == "intermittent" and self.noise_rng.random() < .35:
            return
        self.packet = self._capture(data, 0, now)
        # Keep separate modality timestamps; stale one must not invalidate the other.
        for key in ("lidar_m", "paired_depth_m"):
            values = self.packet[key]
            noise = .04 if self.failure == "noisy" else .003
            self.packet[key] = values + self.noise_rng.normal(0, noise, values.shape)
            if self.failure == "occluded":
                self.packet[key].reshape(-1)[:values.size//3] = np.nan

    def features(self, now):
        outputs = []
        for name, key, count, maximum in (("lidar", "lidar_m", LIDAR, 12.),
                                          ("stereo", "paired_depth_m", DEPTH, 6.)):
            available = self.arm in (name, "both") and self.failure not in (name+"_missing", "both_missing")
            stamp = self.packet["stamp_s"] if self.packet is not None and available else None
            values = self.packet[key] if stamp is not None else np.full(count, np.nan)
            if stamp is not None and self.failure == name+"_stale":
                stamp -= .3
            outputs.append(pack(values, stamp, now, maximum))
        return outputs

    @staticmethod
    def provenance():
        metadata = TeamSensors.metadata()
        metadata["localization"] = "none_in_actor_or_critic"
        metadata["invalid"] = "zero value with explicit zero mask; no hit is invalid"
        metadata["stereo"]["calibration_kind"] = "analytic ray mounts; no image correspondence"
        return metadata
