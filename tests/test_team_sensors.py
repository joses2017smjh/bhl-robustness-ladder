"""Sensor transforms, missing-data behavior, and actual MuJoCo geometry rays."""

from types import SimpleNamespace

import mujoco
import numpy as np

from bhl_robust.eval.team_sensors import TeamSensors, brake_command, ray_pattern


def test_body_attached_ray_axes_and_down_pitch():
    angles, lidar, depth, cos = ray_pattern()
    assert np.allclose(np.linalg.norm(lidar, axis=1), 1)
    assert np.allclose(np.linalg.norm(depth, axis=1), 1)
    assert np.allclose(lidar[np.argmin(abs(angles))], [1, 0, 0])
    assert depth[:, 2].mean() < 0
    assert depth[0, 2] > depth[-1, 2]  # top row sees above bottom row
    # A rolled body must tilt the scan; there is no hidden yaw-only transform.
    roll = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    assert abs((lidar @ roll.T)[:, 2]).max() > .99
    assert np.all((cos > 0) & (cos <= 1))


def test_directional_brake_consumes_lidar_and_paired_depth():
    imu = np.array([0, 0, -1, 0, 0, 0, 0, 0, 9.81, 1])
    lidar, depth = np.full(36, 12.), np.full((2, 8, 8), 6.)
    full, info = brake_command([.4, 0, 0], lidar_m=lidar, depth_m=depth, imu=imu, fresh=True)
    assert np.allclose(full, [.4, 0, 0])
    lidar[18] = .3
    stopped, _ = brake_command([.4, 0, 0], lidar_m=lidar, depth_m=depth, imu=imu, fresh=True)
    assert stopped[0] == 0
    # A rear obstacle does not brake forward travel.
    lidar[:] = 12
    lidar[0] = .2
    free, _ = brake_command([.4, 0, 0], lidar_m=lidar, depth_m=depth, imu=imu, fresh=True)
    assert free[0] == .4
    depth[0, 2, 3] = .3
    stopped, _ = brake_command([.4, 0, 0], lidar_m=lidar, depth_m=depth, imu=imu, fresh=True)
    assert stopped[0] == 0


def test_stale_data_stops_translation_and_imu_limits_speed():
    imu = np.array([0, 0, -1, 0, 0, 0, 0, 0, 9.81, 1])
    kw = dict(lidar_m=np.full(36, 12.), depth_m=np.full((2, 8, 8), 6.), imu=imu)
    stopped, info = brake_command([.4, .1, .2], fresh=False, **kw)
    assert np.allclose(stopped, [0, 0, .2])
    assert info["stale_stop"]
    imu[3] = 3
    limited, info = brake_command([.4, 0, 0], fresh=True, **kw)
    assert limited[0] == .2
    assert info["imu_scale"] == .5


def test_real_ray_distance_and_own_robot_mask_restored():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="wall" type="box" pos="2 0 .72" size=".1 2 1"/>
      <body name="r0_base"><geom type="sphere" size=".3" pos=".2 0 .72"/>
        <site name="r0_imu" pos="0 0 .7"/></body>
      </worldbody><sensor><accelerometer name="r0_imu_acc" site="r0_imu"/>
      </sensor></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    slot = SimpleNamespace(prefix="r0_", body_id=1)
    sensors = TeamSensors(model, [slot], np.array([-1, 0]), mode="record", seed=0)
    old = model.geom_group.copy()
    packet = sensors._capture(data, 0, 0)
    assert np.array_equal(model.geom_group, old)
    # Wall front x1.9, ray origin x.12. Own sphere must not occlude it.
    assert abs(packet["lidar_m"][18] - 1.78) < .02
    assert packet["paired_depth_m"].shape == (2, 8, 8)
