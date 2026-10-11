"""Strict headless FAST-LIO2 transport and native-output validation.

This module packages recorded raw sensors; it does not estimate trajectories.
No ground-truth input or pose initializer is part of the binary transport.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import select
import struct
import subprocess

import numpy as np

from bhl_robust.research.pose_metrics import transform


POINT = struct.Struct("<5fH")
IMU = struct.Struct("<7d")
UPSTREAM_COMMIT = "7cc4175de6f8ba2edf34bab02a42195b141027e9"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def write_transport(packets, t_imu_lidar, destination, *, surface_voxel_m=.15,
                    map_voxel_m=.15, blind_m=.1, point_filter_every=1):
    """Preserve timed raw returns and causal high-rate IMU in a typed stream.

    FAST's native IMU deskewer expects each point offset in milliseconds in
    PointXYZINormal.curvature; only the C++ transport performs that conversion.
    A scan receives exactly IMU measurements after the previous scan tail and
    through its own original tail. Later IMU samples are never used early.
    """
    extrinsic = transform(t_imu_lidar)
    if not all(math.isfinite(x) and x > 0 for x in (surface_voxel_m, map_voxel_m)):
        raise ValueError("positive finite voxel sizes required")
    if not math.isfinite(blind_m) or blind_m < 0 or type(point_filter_every) is not int or point_filter_every < 1:
        raise ValueError("invalid raw-return filtering parameters")
    scans, imus = [], []
    for packet in packets:
        if packet.get("kind") == "pointcloud2": scans.append(packet)
        elif packet.get("kind") == "imu": imus.append(packet)
        else: raise ValueError("unsupported inference packet kind")
    if not scans or not imus: raise ValueError("both original lidar scans and IMU required")
    imu_times = np.asarray([p["capture_time_s"] for p in imus], dtype=float)
    scan_starts = np.asarray([p["capture_time_s"] for p in scans], dtype=float)
    if (not np.isfinite(imu_times).all() or np.any(np.diff(imu_times) <= 0)
            or not np.isfinite(scan_starts).all() or np.any(np.diff(scan_starts) <= 0)):
        raise ValueError("sensor times must be finite and strictly increasing")
    prepared, next_imu, previous_tail = [], 0, -math.inf
    for scan in scans:
        points = np.asarray(scan["points"], dtype=float)
        if points.ndim != 2 or points.shape[1] != 6 or not 1 < len(points) <= 100000 or not np.isfinite(points).all():
            raise ValueError("finite native XYZ/intensity/time/ring scan required")
        if (np.any(points[:, 4] < 0) or points[-1, 4] <= 0 or np.any(np.diff(points[:, 4]) < 0)
                or np.any(points[:, 5] < 0) or np.any(points[:, 5] > 127)
                or np.any(points[:, 5] != np.floor(points[:, 5]))):
            raise ValueError("sorted nonzero original point offsets and integer rings required")
        beginning = float(scan["capture_time_s"])
        tail = beginning + float(points[-1, 4])
        if beginning <= previous_tail or imu_times[0] > beginning or imu_times[-1] < tail:
            raise ValueError("nonoverlapping scans and IMU spanning every tail required")
        end_imu = int(np.searchsorted(imu_times, tail, side="right"))
        samples = imus[next_imu:end_imu]
        if not samples or len(samples) > 10000:
            raise ValueError("each scan requires bounded causal IMU measurements")
        next_imu = end_imu
        previous_tail = tail
        for sample in samples:
            values = np.asarray([sample["capture_time_s"], *sample["gyro_rad_s"], *sample["specific_force_m_s2"]], dtype=float)
            if values.shape != (7,) or not np.isfinite(values).all(): raise ValueError("finite IMU SI values required")
        prepared.append((beginning, tail, points, samples))
    path = Path(destination)
    if path.exists(): raise ValueError("native transport must be new")
    with path.open("xb") as stream:
        stream.write(b"BHLFAST1")
        stream.write(struct.pack("<12d", *extrinsic[:3, :3].reshape(-1), *extrinsic[:3, 3]))
        stream.write(struct.pack("<3dII", surface_voxel_m, map_voxel_m, blind_m, point_filter_every, len(prepared)))
        for beginning, tail, points, samples in prepared:
            stream.write(struct.pack("<ddII", beginning, tail, len(points), len(samples)))
            for x, y, z, intensity, offset, ring in points:
                stream.write(POINT.pack(x, y, z, intensity, offset, int(ring)))
            for sample in samples:
                stream.write(IMU.pack(sample["capture_time_s"], *sample["gyro_rad_s"], *sample["specific_force_m_s2"]))
    return {"schema": "bhl-native-lio-transport-v1", "scan_count": len(prepared),
            "raw_point_count": sum(len(row[2]) for row in prepared),
            "imu_count": sum(len(row[3]) for row in prepared),
            "first_scan_begin_s": prepared[0][0], "last_scan_end_s": prepared[-1][1],
            "expected_timestamps_s": [row[1] for row in prepared], "T_imu_lidar": extrinsic.tolist(),
            "config": {"surface_voxel_m": surface_voxel_m, "map_voxel_m": map_voxel_m,
                       "blind_m": blind_m, "point_filter_every": point_filter_every,
                       "feature_extraction": False, "extrinsic_estimation": False,
                       "imu_time_offset_s": 0.0, "maximum_iterations": 4,
                       "gyro_covariance": .1, "acceleration_covariance": .1,
                       "gyro_bias_covariance": .0001, "acceleration_bias_covariance": .0001},
            "ground_truth_inputs": [], "transport_sha256": sha256(path), "transport_bytes": path.stat().st_size}


def validate_native_output(path, transport):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    expected = transport["expected_timestamps_s"]
    if len(rows) != len(expected): raise ValueError("native estimator omitted original scans")
    for index, (row, stamp) in enumerate(zip(rows, expected)):
        if row.get("frame_index") != index or not math.isclose(row.get("timestamp_s", math.nan), stamp, rel_tol=0, abs_tol=1e-9):
            raise ValueError("native output timestamp/index mismatch")
        if type(row.get("tracked")) is not bool or not isinstance(row.get("state"), str):
            raise ValueError("native tracking state must be explicit")
        if (not isinstance(row.get("compute_seconds"), (int, float))
                or not math.isfinite(row["compute_seconds"]) or row["compute_seconds"] < 0
                or row.get("map_reset_id") != 0):
            raise ValueError("invalid native timing/reset provenance")
        if row["tracked"]:
            transform(row.get("T_W_I"))
            if row.get("effective_points", 0) < 1 or row["state"] != "TRACKED":
                raise ValueError("tracked state requires native measurement matches")
        elif row.get("T_W_I") is not None:
            raise ValueError("untracked native output must not invent a pose")
    return rows


class NativeLioClient:
    """Persistent actual native estimator, with one causal request per scan.

    ``scan`` contains ``points_xyz_m``, absolute ``point_time_s``, integer
    ``ring_index`` and ``frame_timestamp_s`` equal to its last return time.
    ``imu_samples`` contains ``timestamp_s``, ``gyro_rad_s`` and
    ``specific_force_m_s2``. The caller must provide every new IMU sample,
    including samples between scan tails. No truth pose or motion seed exists.
    """

    def __init__(self, runtime_dir, t_imu_lidar, work_dir, *, max_frames=1000,
                 surface_voxel_m=.15, map_voxel_m=.15, blind_m=.1,
                 point_filter_every=1, timeout_s=30):
        self.runtime_dir = Path(runtime_dir).resolve()
        self.work_dir = Path(work_dir).resolve()
        if self.work_dir.exists(): raise ValueError("native client directory must be new")
        if type(max_frames) is not int or not 1 <= max_frames <= 100000:
            raise ValueError("bounded positive max_frames required")
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= 60:
            raise ValueError("native per-scan timeout must be in (0,60] seconds")
        if (not all(math.isfinite(x) and x > 0 for x in (surface_voxel_m, map_voxel_m))
                or not math.isfinite(blind_m) or blind_m < 0
                or type(point_filter_every) is not int or point_filter_every < 1):
            raise ValueError("invalid native client configuration")
        extrinsic = transform(t_imu_lidar)
        self.runtime = json.loads((self.runtime_dir / "runtime.json").read_text())
        binary = self.runtime_dir / "fastlio_headless"
        if (self.runtime.get("status") != "PASS" or self.runtime.get("upstream_commit") != UPSTREAM_COMMIT
                or sha256(binary) != self.runtime.get("binary_sha256")):
            raise ValueError("checksum-verified pinned native runtime required")
        self.work_dir.mkdir(parents=True)
        self.input_path, self.output_path = self.work_dir / "requests.fifo", self.work_dir / "responses.fifo"
        os.mkfifo(self.input_path, mode=0o600); os.mkfifo(self.output_path, mode=0o600)
        self.log = (self.work_dir / "native.log").open("w")
        environment = dict(os.environ, LD_LIBRARY_PATH=str(self.runtime_dir / "lib"), OMP_NUM_THREADS="1")
        self.process = subprocess.Popen([str(binary), str(self.input_path), str(self.output_path)],
                                        stdout=self.log, stderr=subprocess.STDOUT, env=environment)
        self.writer = None; self.reader = None
        self.index = 0; self.max_frames = max_frames; self.previous_tail = -math.inf
        self.previous_imu = -math.inf; self.timeout_s = timeout_s; self.closed = False
        self.receipt = {"schema": "bhl-native-lio-stream-v1", "upstream_commit": UPSTREAM_COMMIT,
                        "runtime_binary_sha256": self.runtime["binary_sha256"],
                        "runtime_receipt_sha256": sha256(self.runtime_dir / "runtime.json"),
                        "T_imu_lidar": extrinsic.tolist(), "maximum_frames": max_frames,
                        "config": {"surface_voxel_m": surface_voxel_m, "map_voxel_m": map_voxel_m,
                                   "blind_m": blind_m, "point_filter_every": point_filter_every},
                        "ground_truth_inputs": [], "clock_semantics": "absolute original sensor times; native pose at scan tail"}
        try:
            # FIFO open handshakes follow the native constructor order.
            # O_NONBLOCK permits a bounded startup failure instead of hanging.
            import time
            deadline = time.monotonic() + timeout_s
            while True:
                try:
                    fd = os.open(self.input_path, os.O_WRONLY | os.O_NONBLOCK)
                    os.set_blocking(fd, True)
                    self.writer = os.fdopen(fd, "wb", buffering=0)
                    break
                except OSError as error:
                    import errno
                    if error.errno != errno.ENXIO: raise
                    if self.process.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("native FIFO process failed to open inputs") from error
                    time.sleep(.01)
            fd = os.open(self.output_path, os.O_RDONLY | os.O_NONBLOCK)
            self.reader = os.fdopen(fd, "rb", buffering=0)
            header = b"BHLFAST1" + struct.pack("<12d", *extrinsic[:3, :3].reshape(-1), *extrinsic[:3, 3])
            header += struct.pack("<3dII", surface_voxel_m, map_voxel_m, blind_m, point_filter_every, max_frames)
            self.writer.write(header)
            self.requests = (self.work_dir / "input.bin").open("xb")
            self.requests.write(header)
            self.responses = (self.work_dir / "native_frames.jsonl").open("x")
        except Exception:
            self.close()
            raise

    def track(self, scan, imu_samples):
        if self.closed or self.index >= self.max_frames: raise ValueError("native stream is closed or exhausted")
        xyz = np.asarray(scan["points_xyz_m"], dtype=float)
        stamps = np.asarray(scan["point_time_s"], dtype=float)
        rings = np.asarray(scan["ring_index"])
        if (xyz.ndim != 2 or xyz.shape[1] != 3 or not 1 < len(xyz) <= 100000
                or stamps.shape != (len(xyz),) or rings.shape != (len(xyz),)
                or not np.isfinite(xyz).all() or not np.isfinite(stamps).all()
                or not np.isfinite(rings).all() or np.any(rings < 0) or np.any(rings > 127)
                or np.any(rings != np.floor(rings))):
            raise ValueError("native scan requires finite original timed 3D returns/rings")
        order = np.argsort(stamps, kind="stable")
        beginning, tail = float(stamps.min()), float(stamps.max())
        if (tail <= beginning or beginning <= self.previous_tail
                or not math.isclose(float(scan["frame_timestamp_s"]), tail, rel_tol=0, abs_tol=1e-8)):
            raise ValueError("native scan timestamp must be its original tail, after the previous scan")
        imu_t = np.asarray(imu_samples["timestamp_s"], dtype=float)
        gyro = np.asarray(imu_samples["gyro_rad_s"], dtype=float)
        accel = np.asarray(imu_samples["specific_force_m_s2"], dtype=float)
        if (imu_t.ndim != 1 or not 1 <= len(imu_t) <= 10000 or gyro.shape != (len(imu_t), 3)
                or accel.shape != gyro.shape or not all(np.isfinite(a).all() for a in (imu_t, gyro, accel))
                or np.any(np.diff(imu_t) <= 0) or imu_t[0] <= self.previous_imu
                or imu_t[-1] > tail + 1e-9 or (self.index == 0 and imu_t[0] > beginning)):
            raise ValueError("new finite causal SI-unit IMU samples required through scan tail")
        if max(float(np.max(np.diff(imu_t))) if len(imu_t) > 1 else 0,
               float(imu_t[0] - self.previous_imu) if self.index else 0,
               tail - float(imu_t[-1])) > .02 + 1e-8:
            raise ValueError("IMU coverage gaps exceed 20 ms")
        # Intensity is absent only for explicitly ideal ray simulation scans.
        if "intensity" in scan:
            intensity = np.asarray(scan["intensity"], dtype=float)
            if intensity.shape != stamps.shape or not np.isfinite(intensity).all(): raise ValueError("invalid intensity")
        elif scan.get("origin") == "simulation": intensity = np.zeros(len(xyz))
        else: raise ValueError("only explicit simulation permits unmeasured zero intensity")
        payload = bytearray(struct.pack("<ddII", beginning, tail, len(xyz), len(imu_t)))
        for i in order: payload.extend(POINT.pack(*xyz[i], intensity[i], float(stamps[i] - beginning), int(rings[i])))
        for t, g, a in zip(imu_t, gyro, accel): payload.extend(IMU.pack(float(t), *g, *a))
        remaining_payload = memoryview(payload)
        while remaining_payload:
            written = self.writer.write(remaining_payload)
            if not written: raise RuntimeError("native request FIFO closed")
            remaining_payload = remaining_payload[written:]
        self.requests.write(payload); self.requests.flush()
        # Read a bounded complete line; C++ flushes once per original scan.
        import time
        deadline = time.monotonic() + self.timeout_s
        data = bytearray()
        while not data.endswith(b"\n"):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.reader], [], [], max(0, remaining))[0]:
                raise TimeoutError("native FAST-LIO2 scan output exceeded timeout")
            chunk = os.read(self.reader.fileno(), 1)
            if not chunk:
                if self.process.poll() is not None: raise RuntimeError("native FAST-LIO2 process terminated before scan output")
                time.sleep(.001)
                continue
            data.extend(chunk)
            if len(data) > 32768: raise ValueError("native frame output exceeds bounded record")
        row = json.loads(data)
        if (row.get("frame_index") != self.index or type(row.get("tracked")) is not bool
                or not math.isclose(row.get("timestamp_s", math.nan), tail, rel_tol=0, abs_tol=1e-9)
                or row.get("map_reset_id") != 0 or not isinstance(row.get("state"), str)
                or not math.isfinite(row.get("compute_seconds", math.nan)) or row["compute_seconds"] < 0):
            raise ValueError("invalid actual native estimator output")
        if row["tracked"]:
            transform(row.get("T_W_I"))
            if row.get("effective_points", 0) < 1 or row["state"] != "TRACKED": raise ValueError("native tracking lacks map matches")
        elif row.get("T_W_I") is not None: raise ValueError("native untracked pose must remain null")
        self.responses.write(json.dumps(row, allow_nan=False) + "\n"); self.responses.flush()
        self.previous_tail, self.previous_imu = tail, float(imu_t[-1])
        self.index += 1
        return row

    def close(self):
        if getattr(self, "closed", False): return
        self.closed = True
        if self.writer: self.writer.close()
        process = getattr(self, "process", None)
        if process:
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
        if self.reader: self.reader.close()
        for name in ("requests", "responses", "log"):
            stream = getattr(self, name, None)
            if stream: stream.close()
        if hasattr(self, "receipt"):
            receipt = dict(self.receipt, frames=self.index, native_returncode=process.returncode if process else None,
                           status="PASS" if process and process.returncode == 0 and not getattr(self, "failure", None) else "INCOMPLETE",
                           failure=getattr(self, "failure", None))
            for name in ("input.bin", "native_frames.jsonl"):
                path = self.work_dir / name
                if path.exists(): receipt[name + "_sha256"] = sha256(path)
            (self.work_dir / "stream_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None: self.failure = str(exc_value)
        self.close()
