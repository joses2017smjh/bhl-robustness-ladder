import importlib.util
from pathlib import Path
import numpy as np
from bhl_robust.mission.sensors import HISTORY, FRAME, PROPRIO, LIDAR

spec = importlib.util.spec_from_file_location('mission7_diagnostic', Path(__file__).resolve().parents[1]/'scripts/mission7_diagnostic.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def test_oracle_body_command_and_braking():
    action = diag.oracle_action(np.zeros(2), np.pi/2, np.array([1., 0.]), .28)
    command = np.tanh(action[:3])*[.4, .35, .4]
    np.testing.assert_allclose(command[:2], [0, -.28], atol=1e-7)
    assert command[2] < 0
    stopped = diag.oracle_action(np.zeros(2), 0, np.array([.02, .01]), .28)
    assert not stopped.any()


def test_counterfactual_masks_all_history_without_changing_proprioception():
    obs = np.ones(HISTORY*FRAME, dtype=np.float32)
    masked = diag.mask_modality(obs, 'lidar_missing').reshape(HISTORY, FRAME)
    np.testing.assert_array_equal(masked[:, :PROPRIO], 1)
    np.testing.assert_array_equal(masked[:, PROPRIO:PROPRIO+2*LIDAR], 0)
    np.testing.assert_array_equal(masked[:, PROPRIO+2*LIDAR:], 1)
    np.testing.assert_array_equal(obs, 1)
