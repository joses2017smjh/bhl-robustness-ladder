import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/bench/tumvi_native_evaluate.py"
SPEC = importlib.util.spec_from_file_location("tumvi_evaluate", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
ORIGINAL_PATH = sys.path.copy()
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path[:] = ORIGINAL_PATH


def test_ground_truth_quaternion_order_and_sensor_conversion():
    q = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
    pose = MODULE.truth_pose([1, 2, 3], q)
    np.testing.assert_allclose(pose[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    t_i_c = MODULE.truth_pose([.1, .2, .3], [1, 0, 0, 0])
    t_w_c = pose @ t_i_c
    np.testing.assert_allclose(t_w_c @ np.linalg.inv(t_i_c), pose, atol=1e-12)
    with pytest.raises(ValueError, match="unit norm"):
        MODULE.truth_pose([0, 0, 0], [2, 0, 0, 0])


def test_independent_imu_frame_reference_and_integer_clock(tmp_path):
    (tmp_path / "dso").mkdir()
    (tmp_path / "mav0/mocap0").mkdir(parents=True)
    origin = 1520530308199447626
    text = f"# calibrated IMU truth\n{origin},1,2,3,1,0,0,0\n{origin + 1},1,2,3,1,0,0,0\n"
    for path in (tmp_path / "dso/gt_imu.csv", tmp_path / "mav0/mocap0/data.csv"):
        path.write_text(text)
    truth = MODULE.load_truth(tmp_path, origin)
    assert truth.timestamp_s.tolist() == [0., 1e-9]
    (tmp_path / "mav0/mocap0/data.csv").write_text(text.replace(",1,2,3,", ",2,2,3,"))
    with pytest.raises(ValueError, match="must match"):
        MODULE.load_truth(tmp_path, origin)


def test_reused_map_id_after_native_reset_splits_epoch():
    def frame(i, map_id, initialized):
        return {"timestamp_s": i * .05, "map_id": map_id, "tracked": map_id is not None,
                "inertial": {"map_imu_initialized": initialized}}
    rows = [frame(0, None, False), frame(1, 0, False), frame(2, 0, True),
            frame(3, None, False), frame(4, None, False), frame(5, 0, False), frame(6, 0, True)]
    epochs, events = MODULE.map_epochs(rows)
    assert [e["map_id"] for e in epochs] == [0, 0]
    assert len(events) == 1 and events[0]["frame_index"] == 3
    assert len(MODULE.accepted_frames(epochs[0], "stereo_inertial")) == 1
    assert len(MODULE.accepted_frames(epochs[1], "stereo_inertial")) == 1


def test_inertial_reset_without_id_change_splits_epoch():
    rows = [{"timestamp_s": i * .05, "map_id": 0, "tracked": True,
             "inertial": {"map_imu_initialized": initialized}}
            for i, initialized in enumerate([False, True, False, False, True])]
    epochs, events = MODULE.map_epochs(rows)
    assert len(epochs) == 2 and len(events) == 1
    assert events[0]["reasons"] == ["native_imu_initialization_lost"]
