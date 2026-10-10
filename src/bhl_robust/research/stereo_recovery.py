"""Conservative native-map continuity using independently estimated LiDAR–IMU poses.

No simulator pose, goal, map geometry or evaluator interface is accepted here.
This is a hybrid registration method, not stereo-only relocalization. A new map
must fit a calibration window and pass a later, disjoint validation window.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import numpy as np
from .pose_metrics import transform


@dataclass(frozen=True)
class RecoveryConfig:
    calibration_samples: int = 3
    validation_samples: int = 3
    minimum_sample_spacing_s: float = .15
    maximum_pair_skew_s: float = .01
    maximum_translation_residual_m: float = .06
    maximum_rotation_residual_rad: float = .10
    watchdog_translation_m: float = .15
    watchdog_rotation_rad: float = .20
    minimum_effective_lidar_points: int = 30

    def __post_init__(self):
        for name in ("calibration_samples", "validation_samples", "minimum_effective_lidar_points"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("positive integer recovery count required")
        for name, value in asdict(self).items():
            if name not in ("calibration_samples", "validation_samples", "minimum_effective_lidar_points"):
                if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                    raise ValueError("positive finite recovery threshold required")
        if self.calibration_samples < 3 or self.validation_samples < 3:
            raise ValueError("at least three calibration and three later validation samples required")


def pose_residual(reference, estimate):
    relative = np.linalg.inv(transform(reference)) @ transform(estimate)
    translation = float(np.linalg.norm(relative[:3, 3]))
    angle = float(np.arccos(np.clip((np.trace(relative[:3, :3])-1)/2, -1, 1)))
    return translation, angle


def mean_transform(poses):
    """Unit-scale SE(3) mean; no similarity-scale correction is permitted."""
    matrices = np.asarray([transform(pose) for pose in poses])
    u, _, vt = np.linalg.svd(matrices[:, :3, :3].sum(axis=0))
    rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, matrices[:, :3, 3].mean(axis=0)
    return transform(result)


class VerifiedMapRecovery:
    """Stop on coordinate changes; resume only after later sensor corroboration.

    The first synchronized pair anchors LIO to the initial stereo coordinate
    frame. Later alignment uses three distinct calibration observations, freezes
    that transform, and tests three strictly later observations. These windows
    are disjoint in time; they are not claimed statistically independent.
    LIO poses only register/check stereo; they are never returned as a fallback.
    """
    def __init__(self, config=None):
        self.config = config or RecoveryConfig()
        self.original_map = self.active_map = self.pending_map = None
        self.t_original_lio = None
        self.accepted_transform = np.eye(4)
        self.candidate = None
        self.calibration = []
        self.validation = []
        self.previous_stamp = -math.inf
        self.last_sample = -math.inf
        self.reference_reset_latched = False
        self.recoveries = 0
        self.events = []

    def _result(self, stamp, state, pose=None, **details):
        result = {"timestamp_s": float(stamp), "state": state, "tracked": pose is not None,
                  "T_W_I": None if pose is None else transform(pose).tolist(),
                  "map_reset_id": 0 if pose is not None else 1,
                  "original_map_id": self.original_map, "active_map_id": self.active_map,
                  "pending_map_id": self.pending_map, "recoveries": self.recoveries,
                  "calibration_samples": len(self.calibration), "validation_samples": len(self.validation),
                  "hybrid_reference": "native_FAST_LIO2", "ground_truth_inputs": [], **details}
        self.events.append(result)
        return result

    def _clear_pending(self, map_id=None):
        self.pending_map = map_id
        self.candidate = None
        self.calibration = []
        self.validation = []
        self.last_sample = -math.inf

    def update(self, stereo, reference):
        stamp = stereo.get("timestamp_s")
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or stamp <= self.previous_stamp:
            raise ValueError("strictly increasing finite original stereo timestamps required")
        self.previous_stamp = float(stamp)
        if reference.get("map_reset_id") != 0:
            self.reference_reset_latched = True
        if self.reference_reset_latched:
            return self._result(stamp, "REFERENCE_RESET_STOP")
        reference_stamp = reference.get("timestamp_s")
        if (reference.get("tracked") is not True or not isinstance(reference_stamp, (float, int))
                or isinstance(reference_stamp, bool) or not math.isfinite(reference_stamp)
                or reference_stamp > stamp+1e-9
                or stamp-reference_stamp > self.config.maximum_pair_skew_s
                or reference.get("effective_points", 0) < self.config.minimum_effective_lidar_points):
            self._clear_pending()
            return self._result(stamp, "REFERENCE_UNHEALTHY_STOP")
        if stereo.get("tracked") is not True or type(stereo.get("map_id")) is not int or stereo["map_id"] < 0:
            self._clear_pending()
            return self._result(stamp, "STEREO_UNTRACKED_STOP")
        pose = transform(stereo["T_W_I"])
        reference_pose = transform(reference["T_W_I"])
        map_id = stereo["map_id"]
        if self.original_map is None:
            self.original_map = self.active_map = map_id
            self.t_original_lio = pose @ np.linalg.inv(reference_pose)
            return self._result(stamp, "INITIAL_MAP_ACCEPTED", pose)
        expected = self.t_original_lio @ reference_pose
        if map_id == self.active_map and self.pending_map is None:
            corrected = self.accepted_transform @ pose
            distance, angle = pose_residual(expected, corrected)
            if distance <= self.config.watchdog_translation_m and angle <= self.config.watchdog_rotation_rad:
                return self._result(stamp, "VERIFIED_CONTINUOUS_MAP", corrected,
                                    translation_residual_m=distance, rotation_residual_rad=angle)
            self._clear_pending(map_id)
            return self._result(stamp, "CONTINUITY_WATCHDOG_STOP", translation_residual_m=distance, rotation_residual_rad=angle)
        if self.pending_map != map_id:
            self._clear_pending(map_id)
        if stamp-self.last_sample < self.config.minimum_sample_spacing_s-1e-10:
            return self._result(stamp, "RECOVERY_WAIT_DISTINCT_OBSERVATION")
        self.last_sample = stamp
        proposal = expected @ np.linalg.inv(pose)
        if self.candidate is None:
            self.calibration.append({"timestamp_s": stamp, "transform": proposal.tolist()})
            if len(self.calibration) == self.config.calibration_samples:
                candidate = mean_transform([row["transform"] for row in self.calibration])
                residuals = [pose_residual(candidate, row["transform"]) for row in self.calibration]
                if any(d > self.config.maximum_translation_residual_m or a > self.config.maximum_rotation_residual_rad for d, a in residuals):
                    self._clear_pending(map_id)
                    return self._result(stamp, "RECOVERY_INCONSISTENT_CALIBRATION_STOP")
                self.candidate = candidate
            return self._result(stamp, "RECOVERY_CALIBRATION_STOP")
        distance, angle = pose_residual(expected, self.candidate @ pose)
        if distance > self.config.maximum_translation_residual_m or angle > self.config.maximum_rotation_residual_rad:
            self._clear_pending(map_id)
            return self._result(stamp, "RECOVERY_VALIDATION_REJECTED_STOP", translation_residual_m=distance, rotation_residual_rad=angle)
        self.validation.append({"timestamp_s": stamp, "translation_residual_m": distance, "rotation_residual_rad": angle})
        if len(self.validation) < self.config.validation_samples:
            return self._result(stamp, "RECOVERY_VALIDATION_STOP", translation_residual_m=distance, rotation_residual_rad=angle)
        self.accepted_transform = self.candidate.copy()
        self.active_map = map_id
        self.recoveries += 1
        evidence = {"calibration_timestamps_s": [r["timestamp_s"] for r in self.calibration],
                    "validation_timestamps_s": [r["timestamp_s"] for r in self.validation],
                    "T_original_native_map": self.accepted_transform.tolist()}
        self._clear_pending()
        return self._result(stamp, "VERIFIED_MAP_RECOVERY_RESUME", self.accepted_transform @ pose,
                            recovery_evidence=evidence, translation_residual_m=distance, rotation_residual_rad=angle)

    def receipt(self):
        return {"schema": "bhl-verified-hybrid-map-recovery-v1", "config": asdict(self.config),
                "method": "stereo_pose_registered_and_validated_with_native_LIO", "events": self.events,
                "recoveries": self.recoveries, "ground_truth_inputs": [], "lio_pose_fallback": False,
                "reference_health_is_calibrated_failure_probability": False}
