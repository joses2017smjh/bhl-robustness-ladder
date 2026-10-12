"""Pure timing and accounting for a bounded latest joined-sensor queue.

This is a discrete-event scheduling model, not a hardware concurrency claim.
Simulator rendering/raycasting and evidence archival must be measured separately
by the caller. Required selected-frame preprocessing and native IPC are charged
once by ``service_timing``. Native thresholds and command gates are not inputs.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np


def _seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite nonnegative seconds")
    return float(value)


def _identity(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("sensor identity must be a nonempty string or finite number")
    if isinstance(value, str) and not value:
        raise ValueError("sensor identity must be nonempty")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("sensor identity must be finite")
    return value


class LatestJoinedQueue:
    """One waiting joined event; selection never replaces an active request.

    Payloads are opaque and retained by reference. The caller owns their
    immutability and retains the complete IMU stream outside this waiting slot;
    replacing a slot must not delete IMU needed by a later selected LiDAR scan.
    ``selected`` counts dispatches, independently of response delivery. The
    caller's causal response state owns its one active request and horizon.
    """

    def __init__(self):
        self.pending = None
        self.produced = self.selected = self.superseded = 0
        self.last_event_s = -math.inf
        self.last_pop_s = -math.inf
        self.streams = {name: {"last_id": None, "last_capture_s": -math.inf,
            "identities": {}, "selected_unique": 0, "superseded_before_selection": 0,
            "dispatches": 0, "reused_dispatches": 0} for name in ("camera", "lidar")}

    def offer(self, payload, *, event_s, camera_id, camera_capture_s, lidar_id, lidar_capture_s):
        event_s = _seconds(event_s, "event_s")
        if event_s <= self.last_event_s:
            raise ValueError("joined event times must be strictly increasing")
        values = {"camera": (_identity(camera_id), _seconds(camera_capture_s, "camera_capture_s")),
                  "lidar": (_identity(lidar_id), _seconds(lidar_capture_s, "lidar_capture_s"))}
        # Validate the entire offer before changing any accounting state.
        for name, (identity, stamp) in values.items():
            state = self.streams[name]
            if stamp > event_s + 1e-9:
                raise ValueError("future sensor capture cannot enter a joined event")
            if identity in state["identities"]:
                if identity != state["last_id"] or stamp != state["last_capture_s"]:
                    raise ValueError("sensor identities cannot be reused with altered or regressing capture times")
            elif stamp <= state["last_capture_s"]:
                raise ValueError("new sensor identities require increasing capture times")
        if self.pending is not None:
            self.superseded += 1
        for name, (identity, stamp) in values.items():
            state = self.streams[name]
            if self.pending is not None:
                old = self.pending["queue"][name + "_id"]
                if old != identity and state["identities"][old] == "pending":
                    state["identities"][old] = "superseded"
                    state["superseded_before_selection"] += 1
            if identity not in state["identities"]:
                state["identities"][identity] = "pending"
            state["last_id"], state["last_capture_s"] = identity, stamp
        self.produced += 1
        self.last_event_s = event_s
        self.pending = {"payload": payload, "queue": {"event_index": self.produced - 1,
            "event_s": event_s, "enqueue_s": event_s,
            "camera_id": camera_id, "camera_capture_s": values["camera"][1],
            "lidar_id": lidar_id, "lidar_capture_s": values["lidar"][1]}}

    def pop(self, now_s):
        now_s = _seconds(now_s, "now_s")
        if now_s < self.last_pop_s:
            raise ValueError("selection clock cannot regress")
        self.last_pop_s = now_s
        if self.pending is None or self.pending["queue"]["enqueue_s"] > now_s:
            return None
        result, self.pending = self.pending, None
        result["queue"].update(selected_s=now_s, queue_residence_s=now_s - result["queue"]["enqueue_s"])
        self.selected += 1
        for name, state in self.streams.items():
            identity = result["queue"][name + "_id"]
            state["dispatches"] += 1
            if state["identities"][identity] == "selected":
                state["reused_dispatches"] += 1
            else:
                state["identities"][identity] = "selected"
                state["selected_unique"] += 1
        return result

    def receipt(self):
        streams = {}
        for name, state in self.streams.items():
            pending = sum(status == "pending" for status in state["identities"].values())
            produced = len(state["identities"])
            conserved = produced == state["selected_unique"] + state["superseded_before_selection"] + pending
            streams[name] = {"produced_unique": produced, "selected_unique": state["selected_unique"],
                "superseded_before_selection": state["superseded_before_selection"],
                "pending_unselected": pending, "dispatches": state["dispatches"],
                "reused_dispatches": state["reused_dispatches"], "conservation_pass": conserved}
        pending = int(self.pending is not None)
        return {"schema": "bhl-bounded-latest-joined-queue-v1", "maximum_waiting_events": 1,
            "produced": self.produced, "selected": self.selected, "superseded": self.superseded,
            "pending": pending, "conservation_pass": self.produced == self.selected + self.superseded + pending,
            "streams": streams, "imu_policy": "caller retains every intervening IMU sample outside the replaceable slot",
            "selected_scope": "dispatched requests, not necessarily delivered responses"}


def service_timing(*, request_s, capture_s, preprocessing_s, native_client_s, policy_dt):
    """Charge selected preprocessing then native service exactly once.

    ``request_s`` is the actual selection time, so prior capture/queue waiting
    is already elapsed and must not be added again to ``service_s``. Capture is
    the original estimator sensor timestamp, not a joined-event timestamp.
    Quantization matches ``DelayedNativeState``: at least one policy step.
    """
    request_s = _seconds(request_s, "request_s")
    capture_s = _seconds(capture_s, "capture_s")
    preprocessing_s = _seconds(preprocessing_s, "preprocessing_s")
    native_client_s = _seconds(native_client_s, "native_client_s")
    policy_dt = _seconds(policy_dt, "policy_dt")
    if policy_dt <= 0 or capture_s > request_s + 1e-9:
        raise ValueError("positive policy period and causal original capture required")
    service = preprocessing_s + native_client_s
    if not math.isfinite(service):
        raise ValueError("total service must remain finite")
    ratio = service / policy_dt
    if not math.isfinite(ratio):
        raise ValueError("service/period ratio must remain finite")
    ticks = max(1, int(math.ceil(ratio)))
    ready = request_s + preprocessing_s
    end = ready + native_client_s
    delivery = request_s + ticks * policy_dt
    if not all(math.isfinite(value) for value in (ready, end, delivery)):
        raise ValueError("service timestamps must remain finite")
    return {"capture_s": capture_s, "request_s": request_s,
        "preprocessing_s": preprocessing_s, "ready_s": ready, "estimator_start_s": ready,
        "native_client_s": native_client_s, "estimator_end_s": end,
        "service_s": service, "delivery_ticks": ticks, "delivery_s": delivery,
        "capture_age_at_request_s": max(0., request_s - capture_s),
        "capture_age_at_delivery_s": delivery - capture_s,
        "delivery_quantization_s": max(0., delivery - end),
        "accounting": "selected preprocessing plus native client once; prior waiting excluded from service; simulator generation and archival reported separately"}


def decoded_frame_identity(left, right):
    """Hash decoded uint8 pixels, including dtype/shape, independent of PNG bytes."""
    result = {"schema": "bhl-decoded-stereo-identity-v1"}
    for name, value in (("left", left), ("right", right)):
        pixels = np.asarray(value)
        if (pixels.dtype != np.uint8 or pixels.ndim not in (2, 3) or not pixels.size
                or (pixels.ndim == 3 and pixels.shape[2] not in (1, 3, 4))):
            raise ValueError("decoded pixels must be nonempty uint8 mono/RGB/RGBA arrays")
        header = json.dumps({"shape": list(pixels.shape), "dtype": "uint8"}, sort_keys=True).encode()
        digest = hashlib.sha256(header + b"\0" + np.ascontiguousarray(pixels).tobytes()).hexdigest()
        result[name] = {"shape": list(pixels.shape), "dtype": "uint8", "sha256": digest}
    return result


def compare_replay_inputs(first, second):
    """Validate identical ordered decoded input rows for a paired timing profile.

    Rows contain ``timestamp_s`` and ``decoded_identity`` from the helper above.
    This validates the offered sequence, not identical post-drop selections or
    native poses, and makes no closed-loop or accuracy claim.
    """
    normalized = []
    for rows in (first, second):
        current, previous = [], -math.inf
        for row in rows:
            stamp = _seconds(row["timestamp_s"], "timestamp_s")
            if stamp <= previous:
                raise ValueError("profile inputs require strictly ordered timestamps")
            identity = row["decoded_identity"]
            if not isinstance(identity, dict) or identity.get("schema") != "bhl-decoded-stereo-identity-v1":
                raise ValueError("decoded frame identity receipt required")
            for name in ("left", "right"):
                side = identity.get(name, {})
                if not isinstance(side, dict):
                    raise ValueError("valid decoded shape/dtype/SHA256 required")
                shape, digest = side.get("shape"), side.get("sha256")
                if (side.get("dtype") != "uint8" or not isinstance(shape, list) or len(shape) not in (2, 3)
                        or any(type(n) is not int or n < 1 for n in shape)
                        or (len(shape) == 3 and shape[2] not in (1, 3, 4))
                        or not isinstance(digest, str) or len(digest) != 64
                        or any(c not in "0123456789abcdef" for c in digest)):
                    raise ValueError("valid decoded shape/dtype/SHA256 required")
            current.append({"timestamp_s": stamp, "decoded_identity": identity})
            previous = stamp
        if not current:
            raise ValueError("nonempty paired input sequences required")
        normalized.append(current)
    if normalized[0] != normalized[1]:
        raise ValueError("paired profiles must offer identical decoded inputs and timestamps")
    encoded = json.dumps(normalized[0], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"status": "PASS", "frames": len(normalized[0]), "ordered_inputs_sha256": hashlib.sha256(encoded).hexdigest(),
        "scope": "identical offered decoded frames and timestamps; selected subsets may differ"}


def replay_transport_queue(rows, service_seconds, policy_dt=.04, *, cumulative=False):
    """Replay ordered frame arrivals using measured per-frame transport costs.

    Native service is explicitly zero: this checks transport queue behavior,
    not native inference speed or navigation. ``cumulative=True`` models the
    legacy charge of every callback since the previous request, including
    superseded frames. Otherwise only selected-frame transport is charged.
    There is one waiting frame and one active response. Delivery occupies its
    policy step, preserving the campaign's no-submission-on-arrival rule.
    The bounded horizon is one policy step after the last offered capture;
    unfinished responses/queued frames remain visible, never silently drained.
    """
    rows, costs = list(rows), list(service_seconds)
    dt = _seconds(policy_dt, "policy_dt")
    if dt <= 0 or type(cumulative) is not bool or not rows or len(rows) != len(costs):
        raise ValueError("nonempty matched inputs, positive policy period and boolean cumulative mode required")
    stamps = [_seconds(row["timestamp_s"] if isinstance(row, dict) else row, "timestamp_s") for row in rows]
    costs = [_seconds(value, "transport service") for value in costs]
    if not math.isfinite(sum(costs)):
        raise ValueError("total profiled transport service must remain finite")
    if any(a >= b for a, b in zip(stamps, stamps[1:])):
        raise ValueError("transport replay must preserve strictly ordered input timestamps")
    ratios = [stamps[0] / dt, stamps[-1] / dt]
    if not all(math.isfinite(value) for value in ratios):
        raise ValueError("transport replay timestamps exceed finite policy clock")
    first_step = int(math.floor(ratios[0] + 1e-9))
    last_step = int(math.ceil(ratios[1] - 1e-9)) + 1
    if last_step - first_step > 100000:
        raise ValueError("transport replay exceeds bounded policy-step horizon")
    offered, superseded, services, deliveries = [], [], [], []
    cursor, waiting, active, available = 0, None, None, None
    cumulative_cost = 0.
    ages = []
    for step in range(first_step, last_step + 1):
        now = step * dt
        while cursor < len(stamps) and stamps[cursor] <= now + 1e-9:
            if waiting is not None:
                superseded.append(waiting)
            waiting = cursor
            offered.append(cursor)
            cumulative_cost += costs[cursor]
            cursor += 1
        arrived = active is not None and step >= active["arrival_step"]
        if arrived:
            deliveries.append(active["input_index"])
            available = stamps[active["input_index"]]
            active = None
        if available is not None:
            ages.append(now - available)
        if active is not None or arrived or waiting is None:
            continue
        index, waiting = waiting, None
        service = cumulative_cost if cumulative else costs[index]
        cumulative_cost = 0.
        timing = service_timing(request_s=now, capture_s=stamps[index],
            preprocessing_s=service, native_client_s=0., policy_dt=dt)
        active = {"input_index": index, "arrival_step": step + timing["delivery_ticks"]}
        services.append(dict(timing, input_index=index, queue_residence_s=max(0., now - stamps[index])))
    selected = [row["input_index"] for row in services]
    delivered = set(deliveries)
    for row in services:
        row["delivered_within_horizon"] = row["input_index"] in delivered
    pending = [] if waiting is None else [waiting]
    in_flight = [] if active is None else [active["input_index"]]
    return {"schema": "bhl-recorded-transport-queue-profile-v1", "status": "PASS",
        "mode": "cumulative_callbacks" if cumulative else "selected_transport_only",
        "native_service_seconds": 0., "policy_dt": dt,
        "start_s": first_step * dt, "horizon_s": last_step * dt,
        "offered_input_indices": offered, "selected_input_indices": selected,
        "superseded_input_indices": superseded, "pending_input_indices": pending,
        "delivered_input_indices": deliveries, "in_flight_input_indices": in_flight,
        "produced": len(offered), "selected": len(selected), "superseded": len(superseded),
        "pending": len(pending), "delivered": len(deliveries), "in_flight": len(in_flight),
        "conservation_pass": len(offered) == len(selected) + len(superseded) + len(pending)
            and len(selected) == len(deliveries) + len(in_flight) and len(offered) == len(rows),
        "profiled_service_total_s": float(sum(costs)),
        "charged_service_total_s": float(sum(row["service_s"] for row in services)),
        "unbilled_cumulative_s": cumulative_cost if cumulative else None,
        "unselected_profiled_service_s": None if cumulative else float(sum(costs[i] for i in superseded + pending)),
        "services": services,
        "available_response_decisions": len(ages),
        "stale_available_response_decisions": sum(age > .35 for age in ages),
        "available_capture_age_p95_s": float(np.quantile(ages, .95)) if ages else None,
        "scope": "transport-only recorded-input queue model; no native service, generated sensor cost, pose estimate, navigation score or hardware concurrency claim"}
