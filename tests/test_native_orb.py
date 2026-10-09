"""Native IPC contract tests. Fixture processes are protocol tests, not SLAM results."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from bhl_robust.research.native_orb import (
    NativeOrbClient, UPSTREAM_COMMIT, tracking_summary, validate_response,
)


def frame(timestamp=1.0, state=2, map_id=7):
    return {"schema": "bhl-orb-native-frame-v1", "timestamp_s": timestamp,
            "tracking_state": state, "tracked": state == 2, "map_id": map_id,
            "compute_seconds": .01, "T_W_C": np.eye(4).tolist() if state == 2 else None}


class NativeProtocolTests(unittest.TestCase):
    def test_accepts_metric_pose_and_keeps_lost_pose_null(self):
        self.assertTrue(validate_response(frame(), 1.)["tracked"])
        self.assertIsNone(validate_response(frame(state=3), 1.)["T_W_C"])

    def test_rejects_stale_time_and_wrong_tracking_state(self):
        with self.assertRaisesRegex(ValueError, "different sensor time"):
            validate_response(frame(), 2.)
        for state in (-2, 6, True):
            value = frame(); value["tracking_state"] = state
            with self.assertRaises(ValueError): validate_response(value, 1.)

    def test_only_ok_is_usable_and_lost_cannot_carry_pose(self):
        value = frame(state=3); value["tracked"] = True
        with self.assertRaisesRegex(ValueError, "Only actual ORB OK"):
            validate_response(value, 1.)
        value = frame(state=4); value["T_W_C"] = np.eye(4).tolist()
        with self.assertRaisesRegex(ValueError, "null poses"):
            validate_response(value, 1.)

    def test_rejects_scaled_nonfinite_pose_and_bad_duration(self):
        for scale in (2., float("nan")):
            value = frame(); value["T_W_C"][0][0] = scale
            with self.assertRaises(ValueError): validate_response(value, 1.)
        for duration in (-1., True, float("nan")):
            value = frame(); value["compute_seconds"] = duration
            with self.assertRaises(ValueError): validate_response(value, 1.)

    def test_map_ids_are_native_integers_or_explicit_unknown(self):
        validate_response(frame(map_id=None), 1.)
        for value in (-1, True, "7"):
            with self.assertRaises(ValueError): validate_response(frame(map_id=value), 1.)

    def test_summary_records_initialization_loss_and_map_changes(self):
        summary = tracking_summary([frame(0., 1, None), frame(1.), frame(2., map_id=8), frame(3., 4, None)])
        self.assertEqual(summary["tracked_frames"], 2)
        self.assertEqual(summary["tracking_states"]["NOT_INITIALIZED"], 1)
        self.assertEqual(summary["tracking_states"]["LOST"], 1)
        self.assertEqual(summary["observed_map_changes"], 1)
        self.assertFalse(summary["metric_scoring_ready"])
        with self.assertRaises(ValueError): tracking_summary([frame(1.), frame(1.)])


class NativeProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.vocab = self.root / "vocab.txt"; self.vocab.write_text("protocol-test-fixture")
        self.settings = self.root / "settings.yaml"; self.settings.write_text("protocol-test-fixture")
        self.left = self.root / "left.png"; self.left.write_bytes(b"fixture-image")
        self.right = self.root / "right.png"; self.right.write_bytes(b"fixture-image")

    def tearDown(self):
        self.temp.cleanup()

    def executable(self, body):
        path = self.root / "fixture-native"
        path.write_text(f"#!{sys.executable}\n" + body)
        path.chmod(0o700)
        return path

    def fixture(self, *, wrong_pin=False, wrong_time=False):
        pin = "wrong-pin" if wrong_pin else UPSTREAM_COMMIT
        return self.executable(f'''import json, sys
with open(sys.argv[3], 'w') as out:
 out.write(json.dumps({{"schema":"bhl-orb-native-ready-v1","upstream_commit":{pin!r}}})+'\\n'); out.flush()
 for request in sys.stdin:
  stamp, left, right = request.rstrip('\\n').split('\\t')
  value={frame()!r}
  value['timestamp_s']=float(stamp)+{1. if wrong_time else 0.}
  out.write(json.dumps(value)+'\\n'); out.flush()
''')

    def client(self, exe, **kwargs):
        return NativeOrbClient(exe, self.vocab, self.settings, self.root / "episode", **kwargs)

    def test_streams_original_timestamps_and_rejects_duplicate_requests(self):
        with self.client(self.fixture()) as client:
            result = client.track(123.456789, self.left, self.right)
            self.assertEqual(result["timestamp_s"], 123.456789)
            with self.assertRaises(ValueError): client.track(123.456789, self.left, self.right)
            self.assertEqual(len(client.responses), 1)
        self.assertEqual(client.process.returncode, 0)
        self.assertFalse(client.fifo.exists())

    def test_refuses_wrong_upstream_and_mismatched_reply(self):
        with self.assertRaisesRegex(ValueError, "readiness/pin"):
            with self.client(self.fixture(wrong_pin=True)): pass
        # Preserve the first failure log; a separate episode owns the next process.
        other = NativeOrbClient(self.fixture(wrong_time=True), self.vocab, self.settings, self.root / "episode2")
        with other as client:
            with self.assertRaisesRegex(ValueError, "different sensor time"):
                client.track(1., self.left, self.right)

    def test_native_exit_is_visible_and_cleanup_removes_fifo(self):
        client = self.client(self.executable("import sys\nsys.exit(17)\n"))
        with self.assertRaisesRegex(RuntimeError, "exited 17"):
            with client: pass
        self.assertFalse(client.fifo.exists())

    def test_rejects_missing_and_evaluator_images_before_native_submission(self):
        forbidden = self.root / "evaluator"; forbidden.mkdir()
        truth = forbidden / "depth.png"; truth.write_bytes(b"not-inference")
        with self.client(self.fixture()) as client:
            for left in (truth, self.root / "missing.png"):
                with self.assertRaises(ValueError): client.track(1., left, self.right)
            self.assertEqual(client.responses, [])

    def test_rejects_image_parent_aliases_to_evaluator_namespace(self):
        evaluator = self.root / "evaluator"; evaluator.mkdir()
        (evaluator / "truth.png").write_bytes(b"not-inference")
        alias = self.root / "inference-alias"; alias.symlink_to(evaluator, target_is_directory=True)
        with self.client(self.fixture()) as client:
            with self.assertRaisesRegex(ValueError, "symbolic links"):
                client.track(1., alias / "truth.png", self.right)

    def test_no_ready_response_has_a_bounded_timeout(self):
        executable = self.executable("import time\ntime.sleep(2)\n")
        with self.assertRaises(TimeoutError):
            with self.client(executable, startup_timeout_s=.05): pass


if __name__ == "__main__":
    unittest.main()
