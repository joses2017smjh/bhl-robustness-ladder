"""Atlas-aware stereo coordinate epochs and a fail-closed resume-after-relocalization gate.

Inputs are native ORB responses (tracking state, the legacy tracked-map-point
``map_id`` and the wrapper's read-only Atlas snapshot), estimated poses in the
fixed known-start registration, and the controller's own applied commands. No
simulator, evaluator, map geometry or goal information is accepted here.

Background (processing-schedule v1, bounded-latest stereo, 12 episodes): every
episode invalidated coordinates with ``native_map_identity_unavailable`` because
the legacy wrapper derived map identity only from tracked map points, which are
absent during RECENTLY_LOST. ORB-SLAM3 then (a) relocalized in the same map in
6/12, (b) reset the active map while reusing id 0 in 1/12, and (c) created a new
Atlas map in 5/12. Only (a) preserves the registered coordinates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .native_orb import validate_atlas_telemetry
from .pose_metrics import transform

SYSTEM_NOT_READY, NO_IMAGES_YET, NOT_INITIALIZED, TRACKING_OK, RECENTLY_LOST, LOST, OK_KLT = -1, 0, 1, 2, 3, 4, 5
REFERENCE_FIELDS = ("current_map_id", "current_map_init_kf_id", "current_map_big_change_idx", "atlas_map_count",
                    "current_map_imu_initialized", "current_map_inertial_ba1", "current_map_inertial_ba2")
INERTIAL_FIELDS = ("current_map_imu_initialized", "current_map_inertial_ba1", "current_map_inertial_ba2")
FAIL_CLOSED_REASONS = (
    "atlas_telemetry_missing", "atlas_snapshot_inconsistent", "atlas_map_id_changed", "atlas_active_map_reset",
    "atlas_big_change", "atlas_map_count_changed", "atlas_inertial_stage_changed",
    "native_returned_to_initialization", "map_point_identity_conflict")


def _stamp(value, previous):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= previous:
        raise ValueError("unique ordered finite native timestamps required")
    return float(value)


class AtlasCoordinateEpochTracker:
    """Causal coordinate epochs from the Atlas map incarnation, failing closed.

    An epoch starts at the first tracked frame on an initialized Atlas map
    (keyframes > 0, consistent snapshot, no map-point conflict). Its reference
    is the current map id, the map's init keyframe id (which changes when an
    active-map reset reuses the id), the map's big-change index, the Atlas map
    count and the inertial refinement stage. Later frames invalidate the epoch
    on any change of that reference, the wrapper's reset flag, an emptied
    keyframe set, ``System::MapChanged()``, a return to NO_IMAGES_YET or
    NOT_INITIALIZED, a tracked map point in another map, an inconsistent
    snapshot or missing telemetry. A RECENTLY_LOST/LOST interval in the same
    incarnation without any such signal is NOT an invalidation. Observe each
    unique native camera response exactly once.
    """

    def __init__(self):
        self.previous_stamp = -math.inf
        self.reference = None
        self.epoch = -1
        self.events = []

    @property
    def pending(self):
        return self.reference is None

    def observe(self, frame):
        stamp = _stamp(frame.get("timestamp_s"), self.previous_stamp)
        state = frame.get("tracking_state")
        if type(state) is not int or state not in (-1, 0, 1, 2, 3, 4, 5):
            raise ValueError("known native tracking state required")
        legacy = frame.get("map_id")
        if legacy is not None and (type(legacy) is not int or legacy < 0):
            raise ValueError("legacy native map id must be a nonnegative integer or null")
        self.previous_stamp = stamp
        atlas = frame.get("atlas")
        atlas = None if atlas is None else validate_atlas_telemetry(atlas)
        conflict = atlas is not None and legacy is not None and legacy != atlas["current_map_id"]
        reasons = []
        if not self.pending:
            reference = self.reference
            if atlas is None:
                reasons.append("atlas_telemetry_missing")
            else:
                if not atlas["snapshot_consistent"]:
                    reasons.append("atlas_snapshot_inconsistent")
                if atlas["current_map_id"] != reference["current_map_id"]:
                    reasons.append("atlas_map_id_changed")
                elif (atlas["active_map_reset"] or atlas["current_map_keyframes"] == 0
                      or atlas["current_map_init_kf_id"] != reference["current_map_init_kf_id"]):
                    reasons.append("atlas_active_map_reset")
                if (atlas["current_map_big_change_idx"] != reference["current_map_big_change_idx"]
                        or atlas["atlas_last_big_change_idx"] != reference["current_map_big_change_idx"]
                        or atlas["system_map_changed"]):
                    reasons.append("atlas_big_change")
                if atlas["atlas_map_count"] != reference["atlas_map_count"]:
                    reasons.append("atlas_map_count_changed")
                if any(atlas[name] != reference[name] for name in INERTIAL_FIELDS):
                    reasons.append("atlas_inertial_stage_changed")
                if conflict:
                    reasons.append("map_point_identity_conflict")
            if state in (NO_IMAGES_YET, NOT_INITIALIZED):
                reasons.append("native_returned_to_initialization")
        if reasons:
            self.events.append({"timestamp_s": stamp, "previous_epoch": self.epoch, "reasons": reasons,
                                "reference": dict(self.reference),
                                "observed": None if atlas is None else {k: atlas[k] for k in REFERENCE_FIELDS}})
            self.reference = None
        if (self.pending and atlas is not None and state == TRACKING_OK and atlas["current_map_keyframes"] > 0
                and atlas["snapshot_consistent"] and not conflict):
            self.epoch += 1
            self.reference = {name: atlas[name] for name in REFERENCE_FIELDS}
        return {"epoch_id": None if self.pending else self.epoch,
                "continuity_invalidated": bool(reasons), "invalidation_reasons": reasons,
                "coordinate_invalidations": len(self.events),
                "atlas_map_id": None if atlas is None else atlas["current_map_id"],
                "atlas_init_kf_id": None if atlas is None else atlas["current_map_init_kf_id"]}

    def receipt(self):
        return {"schema": "bhl-atlas-coordinate-epochs-v1", "epochs_observed": self.epoch + 1,
                "coordinate_invalidations": len(self.events), "events": self.events,
                "uses_read_only_atlas_telemetry": True, "fail_closed_reasons": list(FAIL_CLOSED_REASONS),
                "transient_loss_in_same_map_incarnation_is_not_invalidation": True,
                "incarnation": "Atlas current map id + init keyframe id + big-change index + map count + inertial stage"}


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def planar_pose(pose):
    """World-frame x, y and yaw about world z of a registered body pose."""
    value = transform(pose)
    return float(value[0, 3]), float(value[1, 3]), math.atan2(value[1, 0], value[0, 0])


def tilt_change(first, second):
    """Angle between the two body z axes in the registered world frame."""
    a, b = transform(first)[:3, 2], transform(second)[:3, 2]
    return float(math.acos(max(-1., min(1., float(np.dot(a, b))))))


def integrate_commands(commands, start_s, end_s, *, step_s, start=(0., 0., 0.)):
    """Dead-reckon the controller's applied body-frame commands over [start_s, end_s).

    ``commands`` holds strictly increasing ``(decision_time_s, vx, vy, yaw_rate)``
    rows; each command acts for ``step_s`` from its decision time, matching the
    gait's policy period. Integration is first order (translate along the
    current heading, then rotate), exactly as in the bound calibration.
    """
    for value in (start_s, end_s, step_s):
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("finite integration interval and step required")
    if step_s <= 0 or end_s < start_s:
        raise ValueError("positive step and ordered interval required")
    x, y, yaw = (float(v) for v in start)
    path = rotation = 0.
    previous = -math.inf
    for row in commands:
        decision, vx, vy, wz = (float(v) for v in row)
        if not all(math.isfinite(v) for v in (decision, vx, vy, wz)) or decision <= previous:
            raise ValueError("finite strictly increasing applied command log required")
        previous = decision
        duration = min(decision + step_s, end_s) - max(decision, start_s)
        if duration <= 0:
            continue
        x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * duration
        y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * duration
        yaw += wz * duration
        path += math.hypot(vx, vy) * duration
        rotation += abs(wz) * duration
    return {"x": x, "y": y, "yaw": yaw, "commanded_path_m": path, "commanded_rotation_rad": rotation}


@dataclass(frozen=True)
class ResumeBound:
    """Predeclared consistency bound between a relocalized pose and dead reckoning.

    Calibrated before this campaign on the processing-schedule v1 bounded
    episodes (seeds 750000/750001, evaluator truth versus the applied commands,
    61,944 windows of 0.04-5 s): this joint bound was exceeded by truth in
    0.29% of all windows, 0.008% of windows up to 3 s and 0% of zero-command
    windows. Body height and tilt varied by at most 2.2 cm and 6.7 degrees.
    """
    base_translation_m: float = .15
    translation_per_commanded_path: float = .60
    base_yaw_rad: float = math.radians(10.)
    yaw_per_commanded_rotation: float = .40
    max_vertical_m: float = .10
    max_tilt_change_rad: float = math.radians(10.)
    confirmation_frames: int = 2

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "confirmation_frames":
                if type(value) is not int or value < 1:
                    raise ValueError("positive integer confirmation count required")
            elif isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("finite nonnegative bound required")


class VerifiedRelocalizationGate:
    """Resume after a loss only on verified same-map relocalization; else stop for good.

    States: WAITING_FOR_REGISTRATION, TRACKING, LOSS_STOP, CONFIRMING and
    PERMANENT_STOP. Before the first accepted pose nothing is registered and
    epoch changes are harmless. Afterwards any coordinate invalidation reported
    by :class:`AtlasCoordinateEpochTracker` (map id change, active-map reset,
    big change, ...) or an epoch mismatch stops permanently, as before. After a
    tracking loss, each tracked frame is compared with the last accepted pose
    dead-reckoned through the commands actually applied since that pose's
    capture time. ``confirmation_frames`` consecutive consistent tracked frames
    resume navigation; one inconsistent frame stops permanently. An untracked
    frame during confirmation restarts the confirmation without stopping.
    """

    def __init__(self, bound=None, *, step_s):
        if isinstance(step_s, bool) or not math.isfinite(step_s) or step_s <= 0:
            raise ValueError("positive finite policy step required")
        self.bound = bound or ResumeBound()
        self.step_s = float(step_s)
        self.state = "WAITING_FOR_REGISTRATION"
        self.epoch = None
        self.reference = None
        self.loss_start = None
        self.confirmations = []
        self.previous_stamp = -math.inf
        self.recoveries = []
        self.losses = 0
        self.stop = None

    def residual(self, timestamp_s, body_pose, commands):
        t0, pose0 = self.reference
        motion = integrate_commands(commands, t0, timestamp_s, step_s=self.step_s, start=planar_pose(pose0))
        x, y, yaw = planar_pose(body_pose)
        translation = math.hypot(x - motion["x"], y - motion["y"])
        yaw_error = abs(wrap_angle(yaw - motion["yaw"]))
        vertical = abs(float(transform(body_pose)[2, 3] - transform(pose0)[2, 3]))
        tilt = tilt_change(pose0, body_pose)
        b = self.bound
        allowed_translation = b.base_translation_m + b.translation_per_commanded_path * motion["commanded_path_m"]
        allowed_yaw = b.base_yaw_rad + b.yaw_per_commanded_rotation * motion["commanded_rotation_rad"]
        return {"timestamp_s": float(timestamp_s), "reference_timestamp_s": t0,
                "translation_residual_m": translation, "allowed_translation_m": allowed_translation,
                "yaw_residual_rad": yaw_error, "allowed_yaw_rad": allowed_yaw,
                "vertical_change_m": vertical, "tilt_change_rad": tilt,
                "commanded_path_m": motion["commanded_path_m"], "commanded_rotation_rad": motion["commanded_rotation_rad"],
                "within_bound": bool(translation <= allowed_translation and yaw_error <= allowed_yaw
                                     and vertical <= b.max_vertical_m and tilt <= b.max_tilt_change_rad)}

    def _result(self, accepted, reason, **details):
        return {"accepted": bool(accepted), "gate_state": self.state, "gate_reason": reason,
                "map_reset_id": int(self.state == "PERMANENT_STOP"), **details}

    def _permanent_stop(self, stamp, reason, **details):
        self.state = "PERMANENT_STOP"
        self.stop = {"timestamp_s": stamp, "reason": reason, "loss_start_s": self.loss_start, **details}
        return self._result(False, reason, **details)

    def observe(self, *, timestamp_s, tracked, body_pose, epoch, commands):
        stamp = _stamp(timestamp_s, self.previous_stamp)
        self.previous_stamp = stamp
        if type(tracked) is not bool:
            raise ValueError("explicit native tracking flag required")
        if tracked and body_pose is None:
            raise ValueError("tracked native frames require a pose")
        if self.state == "PERMANENT_STOP":
            return self._result(False, "PERMANENT_STOP")
        if self.state == "WAITING_FOR_REGISTRATION":
            if tracked and epoch["epoch_id"] is not None and not epoch["continuity_invalidated"]:
                self.epoch, self.reference, self.state = epoch["epoch_id"], (stamp, transform(body_pose)), "TRACKING"
                return self._result(True, "REGISTERED_INITIAL_EPOCH")
            return self._result(False, "NOT_YET_REGISTERED")
        if epoch["continuity_invalidated"]:
            return self._permanent_stop(stamp, "COORDINATE_INVALIDATION", invalidation_reasons=list(epoch["invalidation_reasons"]))
        if epoch["epoch_id"] != self.epoch:
            return self._permanent_stop(stamp, "COORDINATE_EPOCH_MISMATCH")
        if not tracked:
            if self.state == "TRACKING":
                self.losses += 1
                self.loss_start = stamp
            self.state, self.confirmations = "LOSS_STOP", []
            return self._result(False, "TRACKING_LOSS_STOP")
        if self.state == "TRACKING":
            self.reference = (stamp, transform(body_pose))
            return self._result(True, "TRACKING")
        check = self.residual(stamp, body_pose, commands)
        if not check["within_bound"]:
            return self._permanent_stop(stamp, "RELOCALIZATION_INCONSISTENT_WITH_COMMANDED_MOTION", residual=check)
        self.confirmations.append(check)
        if len(self.confirmations) < self.bound.confirmation_frames:
            self.state = "CONFIRMING"
            return self._result(False, "CONFIRMING_RELOCALIZATION", residual=check)
        self.recoveries.append({"loss_start_s": self.loss_start, "resume_s": stamp,
                                "duration_s": stamp - self.loss_start, "checks": list(self.confirmations)})
        self.state, self.reference, self.confirmations, self.loss_start = "TRACKING", (stamp, transform(body_pose)), [], None
        return self._result(True, "VERIFIED_RELOCALIZATION_RESUME", residual=check)

    def receipt(self):
        return {"schema": "bhl-verified-relocalization-gate-v1", "bound": asdict(self.bound), "policy_step_s": self.step_s,
                "final_state": self.state, "losses": self.losses, "verified_resumes": len(self.recoveries),
                "recoveries": self.recoveries, "permanent_stop": self.stop,
                "inputs": "native ORB state, Atlas epoch, registered estimated pose, controller applied commands; no evaluator state"}


def classify_losses(packets, horizon_s):
    """Recovery outcomes of each tracking loss, from unique native camera packets.

    Each packet needs ``timestamp_s``, ``tracked`` (accepted for navigation),
    ``native_ORB_tracking_state`` and ``atlas_epoch`` (tracker output). A loss
    starts at the first unaccepted packet after an accepted one and ends at the
    next accepted packet (navigation resume). Raw native recovery is the first
    later packet whose native state is OK, whatever the policy accepted.
    Mechanisms: ``same_map_relocalization`` (native OK again with no coordinate
    invalidation), ``active_map_reset``, ``new_atlas_map``, ``other_invalidation``
    or ``no_native_recovery``. Unresumed losses are right-censored at the horizon.
    """
    losses, current, seen = [], None, False
    for packet in packets:
        stamp = float(packet["timestamp_s"])
        epoch = packet.get("atlas_epoch") or {}
        if current is not None:
            reasons = epoch.get("invalidation_reasons") or []
            current["invalidation_reasons"].extend(r for r in reasons if r not in current["invalidation_reasons"])
            if current["native_recovery_s"] is None and packet.get("native_ORB_tracking_state") == TRACKING_OK:
                current["native_recovery_s"] = stamp
            if packet["tracked"]:
                current.update(resume_s=stamp, navigation_recovery_s=stamp - current["loss_start_s"])
                losses.append(current)
                current = None
            continue
        if packet["tracked"]:
            seen = True
        elif seen:
            current = {"loss_start_s": stamp, "resume_s": None, "navigation_recovery_s": None,
                       "native_recovery_s": None, "invalidation_reasons": list(epoch.get("invalidation_reasons") or [])}
            if packet.get("native_ORB_tracking_state") == TRACKING_OK:
                current["native_recovery_s"] = stamp
    if current is not None:
        current["right_censored_s"] = horizon_s - current["loss_start_s"]
        losses.append(current)
    for loss in losses:
        reasons = loss["invalidation_reasons"]
        loss["native_relocalization_s"] = (None if loss["native_recovery_s"] is None
                                           else loss["native_recovery_s"] - loss["loss_start_s"])
        loss["mechanism"] = ("new_atlas_map" if "atlas_map_id_changed" in reasons else
                             "active_map_reset" if "atlas_active_map_reset" in reasons else
                             "other_invalidation" if reasons else
                             "same_map_relocalization" if loss["native_recovery_s"] is not None else
                             "no_native_recovery")
        loss.setdefault("right_censored_s", None)
    completed = [l["navigation_recovery_s"] for l in losses if l["navigation_recovery_s"] is not None]
    native = [l["native_relocalization_s"] for l in losses if l["native_relocalization_s"] is not None]
    return {"losses": losses, "tracking_loss_events": len(losses),
            "navigation_recoveries": len(completed), "completed_navigation_recovery_s": completed,
            "right_censored_navigation_loss_s": [l["right_censored_s"] for l in losses if l["right_censored_s"] is not None],
            "native_relocalization_s": native,
            "loss_mechanisms": {m: sum(l["mechanism"] == m for l in losses) for m in
                                ("same_map_relocalization", "active_map_reset", "new_atlas_map",
                                 "other_invalidation", "no_native_recovery")},
            "scope": "acquisition timestamps of unique native camera frames; delivery latency excluded"}
