"""Native transport/configuration contracts, not fabricated SLAM measurements."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from bhl_robust.research.native_orb import NativeOrbClient, UPSTREAM_COMMIT
from bhl_robust.research.native_orb_inertial import (
    ImuSettings, partition_imu, validate_imu_batch, validate_inertial_response,
    write_stereo_inertial_settings,
)


def samples(start=0., count=5):
    times = start + np.arange(count) * .005
    return np.column_stack((times, np.tile([.1, .2, .3, 1., 2., 9.81], (count, 1))))


def test_partition_causal_original_measurements_once_with_explicit_tail():
    original = samples(count=30)
    batches, receipt = partition_imu([.02, .04, .07], original)
    consumed = np.concatenate(batches)
    np.testing.assert_array_equal(consumed, original[:15])
    assert receipt["consumed_imu_samples"] == 15
    assert receipt["unconsumed_tail_samples"] == 15
    assert all(batch[-1][0] <= stamp for batch, stamp in zip(batches, [.02, .04, .07]))


@pytest.mark.parametrize("mutation", ["future", "repeated", "unordered", "gap", "nonfinite", "wrong_shape", "boolean"])
def test_refuse_corrupt_or_noncausal_imu(mutation):
    value = samples()
    if mutation == "future": value[-1, 0] = .03
    elif mutation == "repeated": value[-1, 0] = value[-2, 0]
    elif mutation == "unordered": value = value[::-1]
    elif mutation == "gap": value[-1, 0] = .2
    elif mutation == "nonfinite": value[1, 2] = np.nan
    elif mutation == "wrong_shape": value = value[:, :6]
    elif mutation == "boolean": value = np.ones((5, 7), dtype=bool)
    with pytest.raises(ValueError): validate_imu_batch(value, .2 if mutation == "gap" else .02)


def test_refuse_reused_samples_and_uncovered_camera_interval():
    with pytest.raises(ValueError, match="unused"):
        validate_imu_batch(samples(), .02, previous_imu_s=0.)
    with pytest.raises(ValueError, match="gaps"):
        validate_imu_batch(samples(.1), .12, previous_imu_s=.01, previous_frame_s=.02)
    with pytest.raises(ValueError, match="gaps"):
        validate_imu_batch(samples(), .1)


def test_refuse_missing_recording_coverage():
    with pytest.raises(ValueError, match="span"):
        partition_imu([.02, .2], samples())
    with pytest.raises(ValueError): partition_imu([.02, .02], samples())
    with pytest.raises(ValueError): partition_imu([.02, .04], samples()[::-1])


def test_native_configuration_uses_camera_to_imu_transform_and_real_densities(tmp_path):
    import cv2
    matrix = np.eye(4); matrix[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]; matrix[:3, 3] = [.1, .2, .3]
    path = tmp_path / "native.yaml"
    write_stereo_inertial_settings(path, t_imu_camera=matrix, fx=400., fy=400., cx=319.5, cy=239.5,
                                  width=640, height=480, fps=5, baseline_m=.12)
    reader = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        np.testing.assert_allclose(reader.getNode("Tbc").mat(), matrix, atol=1e-7)
        assert reader.getNode("IMU.Frequency").real() == 200
        assert reader.getNode("IMU.NoiseGyro").isReal()
        assert reader.getNode("IMU.NoiseGyro").real() == .001
        assert reader.getNode("InsertKFsWhenLost").real() == 0
        assert reader.getNode("ORBextractor.nFeatures").real() == 1200
    finally: reader.release()


@pytest.mark.parametrize("config", [ImuSettings(frequency_hz=True), ImuSettings(frequency_hz=20),
                                     ImuSettings(noise_acc=0), ImuSettings(gyro_walk=float("nan"))])
def test_invalid_imu_config_refused(config):
    with pytest.raises(ValueError): config.validate()


def response(batch, initialized=False):
    return {"schema": "bhl-orb-native-frame-v1", "timestamp_s": batch[-1][0], "tracking_state": 1,
            "tracked": False, "map_id": 0 if initialized else None, "T_W_C": None, "compute_seconds": .01,
            "inertial": {"schema": "bhl-orb-inertial-frame-v1", "imu_samples": len(batch),
                         "first_imu_timestamp_s": batch[0][0], "last_imu_timestamp_s": batch[-1][0],
                         "map_imu_initialized": initialized}}


def test_response_requires_actual_native_count_time_and_initialization_flag():
    batch = samples().tolist()
    assert validate_inertial_response(response(batch), batch)["map_imu_initialized"] is False
    for key, value in (("imu_samples", 2), ("first_imu_timestamp_s", -1.),
                       ("last_imu_timestamp_s", float("nan")), ("map_imu_initialized", 1)):
        output = response(batch); output["inertial"][key] = value
        with pytest.raises(ValueError): validate_inertial_response(output, batch)
    output = response(batch, True); output["map_id"] = None
    with pytest.raises(ValueError): validate_inertial_response(output, batch)
    with pytest.raises(ValueError): validate_inertial_response({}, batch)


def setup_process(tmp_path, *, mode_ready="stereo_inertial", acknowledge_count=True):
    for name in ("vocab.txt", "settings.yaml", "left.png", "right.png"):
        (tmp_path / name).write_bytes(b"protocol fixture; not native sensor data")
    script = tmp_path / "fixture_native"
    script.write_text(f'''#!{sys.executable}
import json, sys
assert sys.argv[-1] == '--stereo-inertial'
with open(sys.argv[3], 'w') as out:
 out.write(json.dumps({{"schema":"bhl-orb-native-ready-v1","upstream_commit":{UPSTREAM_COMMIT!r},"sensor_mode":{mode_ready!r}}})+'\\n'); out.flush()
 for request in sys.stdin:
  stamp, left, right, imu_path = request.rstrip('\\n').split('\\t')
  samples = [[float(x) for x in line.split()] for line in open(imu_path)]
  row = {{"schema":"bhl-orb-native-frame-v1","timestamp_s":float(stamp),"tracking_state":1,"tracked":False,"map_id":None,"T_W_C":None,"compute_seconds":.01,
         "inertial":{{"schema":"bhl-orb-inertial-frame-v1","imu_samples":len(samples) if {acknowledge_count!r} else 0,"first_imu_timestamp_s":samples[0][0],"last_imu_timestamp_s":samples[-1][0],"map_imu_initialized":False}}}}
  out.write(json.dumps(row)+'\\n'); out.flush()
''')
    script.chmod(0o700)
    return NativeOrbClient(script, tmp_path / "vocab.txt", tmp_path / "settings.yaml", tmp_path / "episode",
                           sensor_mode="stereo_inertial")


def test_process_transmits_original_imu_values_and_does_not_claim_initialization(tmp_path):
    with setup_process(tmp_path) as client:
        value = client.track(.02, tmp_path / "left.png", tmp_path / "right.png", imu_samples=samples())
        assert value["tracked"] is False
        assert value["inertial"]["map_imu_initialized"] is False
        np.testing.assert_array_equal(np.loadtxt(tmp_path / "episode/imu-000000.tsv"), samples())
        with pytest.raises(ValueError):
            client.track(.04, tmp_path / "left.png", tmp_path / "right.png", imu_samples=samples())


def test_process_rejects_silent_stereo_fallback(tmp_path):
    with pytest.raises(ValueError, match="did not enable"):
        with setup_process(tmp_path, mode_ready="stereo"): pass


def test_process_rejects_unconsumed_imu_claim(tmp_path):
    with setup_process(tmp_path, acknowledge_count=False) as client:
        with pytest.raises(ValueError, match="count differs"):
            client.track(.02, tmp_path / "left.png", tmp_path / "right.png", imu_samples=samples())
