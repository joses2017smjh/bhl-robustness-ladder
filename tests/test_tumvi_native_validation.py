import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/bench/tumvi_native_validation.py"
SPEC = importlib.util.spec_from_file_location("tumvi_validation", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
ORIGINAL_PATH = sys.path.copy()
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path[:] = ORIGINAL_PATH


def synthetic_calibration():
    rotation = np.array([[np.cos(.04), 0, np.sin(.04)], [0, 1, 0],
                         [-np.sin(.04), 0, np.cos(.04)]])
    left = np.eye(4)
    left[:3, 3] = [.02, .03, .04]
    relative = np.eye(4)
    relative[:3, :3] = rotation
    relative[:3, 3] = [-.1, -.002, .001]
    common = dict(camera_model="pinhole", distortion_model="equidistant",
                  intrinsics=[191., 192., 254., 256.], resolution=[512, 512],
                  distortion_coeffs=[.003, .001, -.002, .0002])
    return {"cam0": {**common, "T_cam_imu": left.tolist()},
            "cam1": {**common, "T_cam_imu": (relative @ left).tolist(),
                     "T_cn_cnm1": relative.tolist()}}


def test_original_nanosecond_prefix_before_float_conversion():
    origin = 1520530287234599995
    stamps = [origin + i * 50_000_000 for i in range(1300)]
    first, selected = MODULE.selected_timestamps(stamps, stamps + [origin - 1])
    assert first == origin and len(selected) == 1200
    assert MODULE.relative_seconds([origin, origin + 1, origin + 50_000_000], origin).tolist() == [0., 1e-9, .05]


def test_selection_uses_exact_pairs_and_exclusive_end():
    origin, selected = MODULE.selected_timestamps([5, 15, 60_000_000_005], [5, 25, 60_000_000_005])
    assert origin == 5 and selected == [5]
    with pytest.raises(ValueError, match="No exact"):
        MODULE.selected_timestamps([1], [2])


def test_fixed_conversion_without_contrast_tuning():
    image = np.zeros((512, 512), dtype=np.uint16)
    image[0, :5] = [0, 255, 256, 32768, 65535]
    assert MODULE.fixed_uint16_to_uint8(image)[0, :5].tolist() == [0, 0, 1, 128, 255]
    with pytest.raises(ValueError):
        MODULE.fixed_uint16_to_uint8(image.astype(np.uint8))


def test_rectified_geometry_and_camera_imu_rotation():
    cv2 = pytest.importorskip("cv2")
    calibration = synthetic_calibration()
    _, stereo, t_i_c, receipt = MODULE.rectification(calibration)
    relative = np.array(calibration["cam1"]["T_cn_cnm1"])
    points = np.array([[-.4, .1, 2.], [.4, -.3, 3.], [.2, .4, 4.]])
    right = (relative[:3, :3] @ points.T).T + relative[:3, 3]
    projected = []
    for i, values in enumerate((points, right)):
        camera = calibration[f"cam{i}"]
        k = MODULE.camera_matrix(camera)
        d = np.array(camera["distortion_coeffs"])
        distorted, _ = cv2.fisheye.projectPoints(values.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, d)
        suffix = "left" if i == 0 else "right"
        projected.append(cv2.fisheye.undistortPoints(distorted, k, d,
            R=np.array(receipt[f"R_{suffix}_rectification"]), P=np.array(receipt[f"P_{suffix}"])).reshape(-1, 2))
    np.testing.assert_allclose(projected[0][:, 1], projected[1][:, 1], atol=1e-7)
    assert np.all(projected[0][:, 0] > projected[1][:, 0]) and stereo["baseline_m"] > .1
    expected = np.eye(4)
    expected[:3, :3] = np.array(receipt["R_left_rectification"]).T
    np.testing.assert_allclose(np.array(calibration["cam0"]["T_cam_imu"]) @ t_i_c, expected, atol=1e-10)


def test_inconsistent_independent_extrinsics_rejected():
    pytest.importorskip("cv2")
    calibration = synthetic_calibration()
    calibration["cam1"]["T_cn_cnm1"][0][3] += .01
    with pytest.raises(ValueError, match="disagree"):
        MODULE.rectification(calibration)


def test_imu_causal_clip_preserves_measurements(tmp_path):
    origin = 1520530287234599995
    file = tmp_path / "imu.csv"
    file.write_text("#timestamp,gyro,acc\n" + "\n".join(
        f"{origin + i * 5_000_000},1,2,3,4,5,6" for i in range(-5, 12)))
    samples, stamps = MODULE.read_imu(file, origin, origin + 50_000_000)
    assert stamps[0] == origin - 20_000_000 and stamps[-1] == origin + 50_000_000
    batches, receipt = MODULE.partition_imu([0., .05], samples)
    assert receipt["consumed_imu_samples"] == 15
    assert batches[0][-1][0] == 0 and batches[1][0][0] == .005
    np.testing.assert_equal(samples[:, 1:], np.tile([1, 2, 3, 4, 5, 6], (15, 1)))
