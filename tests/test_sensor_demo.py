"""A demo may show only traceable outputs with validated latency evidence."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).parents[1] / "scripts/bench/sensor_demo.py"
SPEC = importlib.util.spec_from_file_location("sensor_demo_test", PATH)
DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEMO)


def test_relocated_prediction_digest_and_tamper_detection(tmp_path):
    path = tmp_path / "stereo/predictions/depth.npy"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"fixture bytes")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {"/old/container/output/stereo/predictions/depth.npy": digest}
    assert DEMO.verified_artifact(tmp_path, "stereo/predictions/depth.npy", provenance) == path
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        DEMO.verified_artifact(tmp_path, "stereo/predictions/depth.npy", provenance)


def test_conflicting_digests_rejected():
    provenance = {"/a/stereo/x": "0" * 64, "/b/stereo/x": "1" * 64}
    with pytest.raises(ValueError, match="ambiguous"):
        DEMO.provenance_hash(provenance, "stereo/x")


@pytest.mark.parametrize("relative", ["../outside", "/absolute", "stereo/../evaluator/truth.npy"])
def test_path_escapes_rejected(relative):
    with pytest.raises(ValueError, match="unsafe"):
        DEMO.provenance_hash({}, relative)


def test_latency_must_match_samples_and_scope():
    timing = {"samples_ms": [10., 20., 30.], "timed_iterations": 3, "p95_ms": 29.,
              "scope": "warm_cache_PNG_decode_preprocess_inference_depth_validity"}
    assert DEMO.verified_p95(timing) == pytest.approx(29.)
    with pytest.raises(ValueError, match="disagrees"):
        DEMO.verified_p95({**timing, "p95_ms": 1.})
    with pytest.raises(ValueError, match="scope"):
        DEMO.verified_p95({**timing, "scope": "camera-to-control"})


def test_missing_pilot_writes_no_demo(tmp_path):
    output = tmp_path / "demo"
    assert DEMO.main(["--pilot-output", str(tmp_path / "missing"), "--out", str(output)]) == 2
    assert not output.exists()
