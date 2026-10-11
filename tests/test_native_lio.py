"""The transport must preserve clocks/extrinsics and never fabricate poses."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from bhl_robust.research.native_lio import IMU, POINT, UPSTREAM_COMMIT, NativeLioClient, sha256, validate_native_output, write_transport


def packets():
    rows = []
    for t in np.arange(0, .201, .005):
        rows.append({"kind": "imu", "capture_time_s": float(t), "gyro_rad_s": [0, 0, .1], "specific_force_m_s2": [0, 0, 9.81]})
    for t in (0., .1):
        rows.append({"kind": "pointcloud2", "capture_time_s": t,
                     "points": [[1, 0, 0, 0, 0, 0], [2, .1, 0, 0, .05, 1]]})
    return sorted(rows, key=lambda r: r["capture_time_s"])


class NativeLioTransportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "transport.bin"

    def test_original_extrinsics_times_si_units_and_causal_imu(self):
        extrinsic = np.eye(4)
        extrinsic[:3, 3] = [.2, -.1, .7]
        receipt = write_transport(packets(), extrinsic, self.path)
        raw = self.path.read_bytes()
        self.assertEqual(raw[:8], b"BHLFAST1")
        calibration = struct.unpack_from("<12d", raw, 8)
        np.testing.assert_allclose(calibration[:9], np.eye(3).reshape(-1))
        np.testing.assert_allclose(calibration[9:], extrinsic[:3, 3])
        position = 8 + 12 * 8 + struct.calcsize("<3dII")
        consumed = []
        for beginning, tail in ((0., .05), (.1, .15)):
            got_beginning, got_tail, points, imus = struct.unpack_from("<ddII", raw, position)
            self.assertAlmostEqual(got_beginning, beginning)
            self.assertAlmostEqual(got_tail, tail)
            position += struct.calcsize("<ddII")
            scan = [POINT.unpack_from(raw, position + i * POINT.size) for i in range(points)]
            self.assertAlmostEqual(scan[-1][4], .05)
            self.assertEqual(scan[-1][5], 1)
            position += points * POINT.size
            samples = [IMU.unpack_from(raw, position + i * IMU.size) for i in range(imus)]
            self.assertTrue(all(sample[0] <= tail + 1e-12 for sample in samples))
            self.assertEqual(samples[0][1:4], (0., 0., .1))
            self.assertAlmostEqual(samples[0][6], 9.81)
            consumed.extend(sample[0] for sample in samples)
            position += imus * IMU.size
        self.assertEqual(position, len(raw))
        self.assertEqual(len(set(consumed)), len(consumed))
        self.assertEqual(receipt["scan_count"], 2)
        self.assertEqual(receipt["ground_truth_inputs"], [])

    def test_overlapping_scans_rejected(self):
        data = packets()
        [row for row in data if row["kind"] == "pointcloud2"][1]["capture_time_s"] = .04
        with self.assertRaisesRegex(ValueError, "nonoverlapping"):
            write_transport(data, np.eye(4), self.path)

    def test_missing_original_time_rejected(self):
        data = packets()
        [row for row in data if row["kind"] == "pointcloud2"][0]["points"][-1][4] = 0
        with self.assertRaisesRegex(ValueError, "original point offsets"):
            write_transport(data, np.eye(4), self.path)

    def test_fractional_ring_rejected(self):
        data = packets()
        [row for row in data if row["kind"] == "pointcloud2"][0]["points"][-1][5] = .5
        with self.assertRaisesRegex(ValueError, "integer rings"):
            write_transport(data, np.eye(4), self.path)

    def test_future_imu_cannot_cover_missing_scan_tail(self):
        data = [p for p in packets() if not (p["kind"] == "imu" and p["capture_time_s"] > .12)]
        with self.assertRaisesRegex(ValueError, "spanning every tail"):
            write_transport(data, np.eye(4), self.path)

    def test_scaled_extrinsic_rejected(self):
        extrinsic = np.eye(4); extrinsic[0, 0] = 2
        with self.assertRaisesRegex(ValueError, "SO\(3\)"):
            write_transport(packets(), extrinsic, self.path)

    def test_existing_transport_is_never_overwritten(self):
        self.path.write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "must be new"):
            write_transport(packets(), np.eye(4), self.path)
        self.assertEqual(self.path.read_bytes(), b"keep")


class NativeOutputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "native.jsonl"
        self.receipt = {"expected_timestamps_s": [.05, .15]}
        self.rows = [{"frame_index": 0, "timestamp_s": .05, "tracked": False, "state": "INITIALIZING",
                      "compute_seconds": .01, "map_reset_id": 0, "T_W_I": None, "effective_points": 0},
                     {"frame_index": 1, "timestamp_s": .15, "tracked": True, "state": "TRACKED",
                      "compute_seconds": .02, "map_reset_id": 0, "T_W_I": np.eye(4).tolist(), "effective_points": 10}]

    def validate(self):
        self.path.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        return validate_native_output(self.path, self.receipt)

    def test_explicit_initialization_null_pose_preserved(self):
        result = self.validate()
        self.assertIsNone(result[0]["T_W_I"])
        self.assertFalse(result[0]["tracked"])

    def test_omitted_native_frame_rejected(self):
        self.rows.pop(0)
        with self.assertRaisesRegex(ValueError, "omitted original scans"): self.validate()

    def test_identity_placeholder_cannot_be_native_estimate(self):
        self.rows[0]["T_W_I"] = np.eye(4).tolist()
        with self.assertRaisesRegex(ValueError, "must not invent a pose"): self.validate()

    def test_tracked_without_map_matches_rejected(self):
        self.rows[1]["effective_points"] = 0
        with self.assertRaisesRegex(ValueError, "native measurement matches"): self.validate()

    def test_injected_nonrigid_pose_rejected(self):
        self.rows[1]["T_W_I"][0][0] = 1.5
        with self.assertRaisesRegex(ValueError, "SO\(3\)"): self.validate()

    def test_bad_timestamp_or_nan_timing_rejected(self):
        self.rows[1]["timestamp_s"] = .16
        with self.assertRaisesRegex(ValueError, "timestamp/index mismatch"): self.validate()
        self.rows[1]["timestamp_s"] = .15
        self.rows[1]["compute_seconds"] = float("nan")
        with self.assertRaisesRegex(ValueError, "timing/reset"): self.validate()


class NativeFifoTransportTests(unittest.TestCase):
    """IPC contract only: the child fixture emits no estimated poses.

    Native estimator execution is checked separately by the allocated build
    and real replay smoke. This fixture must not count as native science.
    """
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        runtime = self.root / "runtime"; runtime.mkdir(); (runtime / "lib").mkdir()
        binary = runtime / "fastlio_headless"
        import sys
        binary.write_text("#!" + sys.executable + "\n" + '''
import json,struct,sys
with open(sys.argv[1],'rb') as source, open(sys.argv[2],'w') as output:
 header=source.read(136)
 assert header[:8]==b'BHLFAST1'
 count=struct.unpack_from('<I',header,132)[0]
 for index in range(count):
  part=source.read(24)
  if not part: break
  beginning,tail,points,imus=struct.unpack('<ddII',part)
  assert len(source.read(points*22+imus*56))==points*22+imus*56
  output.write(json.dumps({'frame_index':index,'timestamp_s':tail,'scan_begin_s':beginning,'tracked':False,'state':'INITIALIZING','compute_seconds':.001,'map_reset_id':0,'effective_points':0,'T_W_I':None})+'\\n')
  output.flush()
''')
        binary.chmod(0o700)
        (runtime / "runtime.json").write_text(json.dumps({"status": "PASS", "upstream_commit": UPSTREAM_COMMIT, "binary_sha256": sha256(binary)}))
        self.runtime = runtime
        self.scan = {"points_xyz_m": [[1,0,0],[2,.1,0]], "point_time_s": [0,.05],
                     "ring_index": [0,1], "frame_timestamp_s": .05, "origin": "simulation"}
        self.imu = {"timestamp_s": np.arange(0,.051,.005), "gyro_rad_s": np.zeros((11,3)),
                    "specific_force_m_s2": np.tile([0,0,9.81],(11,1))}

    def test_causal_roundtrip_and_early_clean_eof(self):
        with NativeLioClient(self.runtime, np.eye(4), self.root / "client", max_frames=3, timeout_s=3) as client:
            result = client.track(self.scan, self.imu)
            self.assertFalse(result["tracked"])
            self.assertIsNone(result["T_W_I"])
        receipt = json.loads((self.root / "client/stream_receipt.json").read_text())
        self.assertEqual(receipt["frames"], 1)
        self.assertEqual(receipt["native_returncode"], 0)
        self.assertEqual(receipt["ground_truth_inputs"], [])

    def test_future_imu_is_rejected_before_native_request(self):
        with self.assertRaisesRegex(ValueError, "causal"):
            with NativeLioClient(self.runtime, np.eye(4), self.root / "client", timeout_s=3) as client:
                self.imu["timestamp_s"][-1] = .06
                client.track(self.scan, self.imu)
        self.assertEqual(client.index, 0)

    def test_missing_physical_intensity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "explicit simulation"):
            with NativeLioClient(self.runtime, np.eye(4), self.root / "client", timeout_s=3) as client:
                self.scan["origin"] = "physical"
                client.track(self.scan, self.imu)


if __name__ == "__main__": unittest.main()
