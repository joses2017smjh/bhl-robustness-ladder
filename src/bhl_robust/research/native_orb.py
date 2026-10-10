"""Strict IPC for an actual headless ORB-SLAM3 stereo executable.

Only images, calibration, vocabulary and original sensor times cross the native
boundary. Lost frames retain null poses; this client never fabricates estimates.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time
import uuid

from .pose_metrics import transform

UPSTREAM_COMMIT = "4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4"
TRACKING_STATES = {-1: "SYSTEM_NOT_READY", 0: "NO_IMAGES_YET", 1: "NOT_INITIALIZED",
                   2: "OK", 3: "RECENTLY_LOST", 4: "LOST", 5: "OK_KLT"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_rectified_stereo_settings(path, *, fx, fy, cx, cy, width, height,
                                    fps, baseline_m):
    """Use pinned upstream's legacy parser for already rectified pinhole images.

    The 4452a3c Settings v1 Rectified printer dereferences an uninitialized
    originalCalib2_. The supported legacy parser avoids that diagnostic fault;
    Camera.bf carries the same metric calibration as Stereo.b times fx.
    Native tracking code, image geometry and feature parameters are unchanged.
    """
    values = (fx, fy, cx, cy, baseline_m)
    if any(isinstance(x, bool) or not math.isfinite(x) for x in values):
        raise ValueError("Stereo calibration values must be finite")
    if min(fx, fy, baseline_m) <= 0 or any(type(x) is not int or x <= 0 for x in (width, height, fps)):
        raise ValueError("Stereo focal lengths, baseline and integer image timing must be positive")
    rows = ["%YAML:1.0", 'Camera.type: "PinHole"',
            f"Camera.fx: {float(fx)}", f"Camera.fy: {float(fy)}",
            f"Camera.cx: {float(cx)}", f"Camera.cy: {float(cy)}"]
    rows += [f"Camera.{key}: 0.0" for key in ("k1", "k2", "p1", "p2")]
    rows += [f"Camera.width: {width}", f"Camera.height: {height}", f"Camera.fps: {fps}",
             "Camera.RGB: 1", f"Camera.bf: {float(fx * baseline_m)}", "ThDepth: 40.0",
             # Audit-only annotations; legacy tracking consumes Camera.bf.
             f"Stereo.b: {float(baseline_m)}", "Stereo.ThDepth: 40.0",
             "Stereo.T_c1_c2: !!opencv-matrix", "  rows: 4", "  cols: 4", "  dt: f",
             f"  data: [1,0,0,{float(baseline_m)},0,1,0,0,0,0,1,0,0,0,0,1]",
             "ORBextractor.nFeatures: 1200", "ORBextractor.scaleFactor: 1.2",
             "ORBextractor.nLevels: 8", "ORBextractor.iniThFAST: 20", "ORBextractor.minThFAST: 7",
             "Viewer.KeyFrameSize: 0.05", "Viewer.KeyFrameLineWidth: 1.0", "Viewer.GraphLineWidth: 0.9",
             "Viewer.PointSize: 2.0", "Viewer.CameraSize: 0.08", "Viewer.CameraLineWidth: 3.0",
             "Viewer.ViewpointX: 0.0", "Viewer.ViewpointY: -0.7", "Viewer.ViewpointZ: -1.8",
             "Viewer.ViewpointF: 500.0", "Viewer.imageViewScale: 1.0"]
    Path(path).write_text("\n".join(rows) + "\n")


def validate_response(value, timestamp_s):
    if not isinstance(value, dict) or value.get("schema") != "bhl-orb-native-frame-v1":
        raise ValueError("Invalid native ORB frame schema")
    stamp = value.get("timestamp_s")
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
        raise ValueError("Native response requires a finite sensor timestamp")
    if isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float)) or not math.isfinite(timestamp_s):
        raise ValueError("Expected sensor timestamp must be finite")
    if not math.isclose(float(stamp), timestamp_s, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Native response belongs to a different sensor time")
    state = value.get("tracking_state")
    if type(state) is not int or state not in TRACKING_STATES:
        raise ValueError("Unknown native tracking state")
    tracked = value.get("tracked")
    if type(tracked) is not bool or tracked != (state == 2):
        raise ValueError("Only actual ORB OK frames are tracked")
    pose = value.get("T_W_C")
    if tracked:
        transform(pose)
    elif pose is not None:
        raise ValueError("Untracked native frames must keep null poses")
    duration = value.get("compute_seconds")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
        raise ValueError("Invalid native compute duration")
    map_id = value.get("map_id")
    if map_id is not None and (type(map_id) is not int or map_id < 0):
        raise ValueError("Native map_id must be a nonnegative integer or null")
    diagnostic = value.get("diagnostics")
    if diagnostic is not None:
        if not isinstance(diagnostic, dict) or diagnostic.get("schema") != "bhl-orb-frame-diagnostics-v1":
            raise ValueError("Invalid native ORB diagnostics schema")
        for name in ("features_total", "features_left", "features_right", "positive_stereo_depth_matches"):
            if type(diagnostic.get(name)) is not int or diagnostic[name] < 0:
                raise ValueError("Native diagnostic counts must be nonnegative integers")
        if diagnostic["positive_stereo_depth_matches"] > diagnostic["features_total"]:
            raise ValueError("Native positive stereo matches exceed extracted features")
        if diagnostic.get("initialization_feature_threshold_exclusive") != 500 or diagnostic.get("read_only_native_frame") is not True:
            raise ValueError("Read-only pinned N>500 initialization diagnostic required")
        expected_reason = ("NOT_IN_INITIALIZATION" if state != 1 else
                           "FEATURE_COUNT_NOT_ABOVE_500" if diagnostic["features_total"] <= 500 else
                           "OTHER_NATIVE_INITIALIZATION_CONDITION")
        if diagnostic.get("initialization_reason") != expected_reason:
            raise ValueError("Native initialization reason contradicts state or feature count")
    return dict(value)


def _image_path(value):
    raw = str(value)
    if any(c in raw for c in ("\t", "\n", "\r", "\x00")):
        raise ValueError("Image paths cannot contain IPC control characters")
    path = Path(value).absolute()
    if "evaluator" in path.parts or "ground_truth" in path.parts:
        raise ValueError("Evaluator inputs cannot enter the native estimator")
    if any(ancestor.is_symlink() for ancestor in (path, *path.parents)):
        raise ValueError("Original image paths cannot follow symbolic links")
    if not path.is_file() or path.suffix.lower() != ".png":
        raise ValueError("An original regular PNG image is required")
    return path


class NativeOrbClient:
    """Synchronous original-image IPC; caller provides the pinned runtime libs.

    Example, inside the Ubuntu SIF with runtime/lib on LD_LIBRARY_PATH::

        with NativeOrbClient(runtime / 'bin/orb_native', runtime / 'ORBvoc.txt',
                             settings, episode_dir) as estimator:
            frame = estimator.track(sensor_time_s, left_png, right_png)

    ``frame['T_W_C']`` maps left optical camera coordinates into native map
    coordinates, and is null until/whenever tracking is unavailable. The native
    map origin is arbitrary; a navigation caller must register it without truth
    and reject coordinate changes when map_id changes. Call sites should record
    all responses, including tracking failures, rather than filtering them out.
    """

    def __init__(self, executable, vocabulary, settings, working_directory,
                 *, startup_timeout_s=120.0, frame_timeout_s=30.0, environment=None,
                 sensor_mode="stereo"):
        if sensor_mode not in ("stereo", "stereo_inertial"):
            raise ValueError("Native ORB sensor_mode must be stereo or stereo_inertial")
        self.sensor_mode = sensor_mode
        self._previous_imu = -math.inf
        self.executable = Path(executable).resolve()
        self.vocabulary = Path(vocabulary).resolve()
        self.settings = Path(settings).resolve()
        self.directory = Path(working_directory).resolve()
        for path in (self.executable, self.vocabulary, self.settings):
            if not path.is_file():
                raise ValueError(f"Missing native runtime input: {path}")
        for timeout in (startup_timeout_s, frame_timeout_s):
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("Native IPC timeouts must be finite and positive")
        self.startup_timeout_s, self.frame_timeout_s = startup_timeout_s, frame_timeout_s
        self.environment = environment
        self.process = None
        self._stream_fd = None
        self._log = None
        self._buffer = bytearray()
        self._previous = -math.inf
        self.responses = []
        self.ready = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fifo = self.directory / f"orb-responses-{uuid.uuid4().hex}.fifo"
        os.mkfifo(self.fifo, 0o600)
        self._stream_fd = os.open(self.fifo, os.O_RDWR | os.O_NONBLOCK)
        self.log_path = self.directory / "native-orb.log"
        if self.log_path.exists():
            self.close()
            raise ValueError("Native ORB log must be new to preserve each run")
        self._log = self.log_path.open("wb")
        try:
            self.process = subprocess.Popen(
                [str(self.executable), str(self.vocabulary), str(self.settings), str(self.fifo)]
                + (["--stereo-inertial"] if self.sensor_mode == "stereo_inertial" else []),
                stdin=subprocess.PIPE, stdout=self._log, stderr=subprocess.STDOUT,
                cwd=self.directory, env=self.environment, start_new_session=True,
            )
            ready = self._read_json(self.startup_timeout_s)
            if ready.get("schema") != "bhl-orb-native-ready-v1" or ready.get("upstream_commit") != UPSTREAM_COMMIT:
                raise ValueError("Unexpected native runtime readiness/pin")
            if self.sensor_mode == "stereo_inertial" and ready.get("sensor_mode") != self.sensor_mode:
                raise ValueError("Native runtime did not enable stereo-inertial mode")
            self.ready = ready
            return self
        except BaseException:
            self.close()
            raise

    def _read_json(self, timeout):
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self._stream_fd, selectors.EVENT_READ)
            while True:
                if b"\n" in self._buffer:
                    line, _, remainder = self._buffer.partition(b"\n")
                    self._buffer = bytearray(remainder)
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("Native IPC response must be an object")
                    return value
                if len(self._buffer) > 65536:
                    raise ValueError("Native IPC response exceeds bounded size")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Native ORB IPC timed out; consult native-orb.log")
                if selector.select(min(remaining, .1)):
                    self._buffer.extend(os.read(self._stream_fd, 65536))
                elif self.process.poll() is not None:
                    raise RuntimeError(f"Native ORB exited {self.process.returncode}; consult native-orb.log")

    def track(self, timestamp_s, left, right, *, imu_samples=None):
        if isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float)) or not math.isfinite(timestamp_s) or timestamp_s <= self._previous:
            raise ValueError("Original sensor timestamps must be finite and strictly increasing")
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Native ORB process is not running")
        left_path, right_path = _image_path(left), _image_path(right)
        request = f"{timestamp_s:.17g}\t{left_path}\t{right_path}"
        batch = None
        if self.sensor_mode == "stereo_inertial":
            from .native_orb_inertial import validate_imu_batch
            batch = validate_imu_batch(imu_samples, timestamp_s, previous_imu_s=self._previous_imu,
                                       previous_frame_s=self._previous)
            imu_path = self.directory / f"imu-{len(self.responses):06d}.tsv"
            with imu_path.open("x") as stream:
                stream.write("".join("\t".join(format(x, ".17g") for x in row) + "\n" for row in batch))
            request += f"\t{imu_path}"
        elif imu_samples is not None:
            raise ValueError("Stereo-only mode must not silently discard supplied IMU")
        request = (request + "\n").encode()
        self.process.stdin.write(request)
        self.process.stdin.flush()
        value = validate_response(self._read_json(self.frame_timeout_s), float(timestamp_s))
        if batch is not None:
            from .native_orb_inertial import validate_inertial_response
            validate_inertial_response(value, batch)
            self._previous_imu = batch[-1][0]
        self._previous = float(timestamp_s)
        self.responses.append(value)
        return value

    def close(self):
        if self.process is not None:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=3)
        if self._stream_fd is not None:
            os.close(self._stream_fd)
            self._stream_fd = None
        if self._log is not None:
            self._log.close()
            self._log = None
        if hasattr(self, "fifo") and self.fifo.exists():
            self.fifo.unlink()

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        if exc_type is None and self.process.returncode != 0:
            raise RuntimeError(f"Native ORB exited {self.process.returncode}; consult native-orb.log")
        return False


def tracking_summary(frames):
    """Count all native statuses and map changes without creating pose samples."""
    counts = {name: 0 for name in TRACKING_STATES.values()}
    maps, resets, previous, last_stamp = [], 0, None, -math.inf
    durations = []
    for raw in frames:
        if not isinstance(raw, dict):
            raise ValueError("Native responses must be objects")
        frame = validate_response(raw, raw.get("timestamp_s"))
        if frame["timestamp_s"] <= last_stamp:
            raise ValueError("Native response times must remain increasing")
        counts[TRACKING_STATES[frame["tracking_state"]]] += 1
        durations.append(frame["compute_seconds"])
        if frame["map_id"] is not None:
            if previous is not None and frame["map_id"] != previous:
                resets += 1
            previous = frame["map_id"]
            if previous not in maps:
                maps.append(previous)
        last_stamp = frame["timestamp_s"]
    return {"frames": len(frames), "tracked_frames": counts["OK"], "tracking_states": counts,
            "observed_map_ids": maps, "observed_map_changes": resets,
            "all_frames_tracked": bool(frames) and counts["OK"] == len(frames),
            "metric_scoring_ready": counts["OK"] >= 2 and resets == 0,
            "native_compute_seconds_total": sum(durations),
            "pose_source": "native_ORB_SLAM3_TrackStereo_returned_SE3_inverse",
            "no_truth_inputs": True}
