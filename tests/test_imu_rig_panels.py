"""Display-only IMU (simulated IM10A) and stereo-rig panels for the maze clip."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "bench"))

from bhl_robust.eval import imu_sim, panels  # noqa: E402


def test_datasheet_numbers_are_the_published_upper_bounds():
    p = imu_sim.IM10A_DATASHEET
    assert (p["gyro_noise_dps"], p["gyro_bias_dps"], p["accel_noise_mg"], p["accel_bias_mg"]) == (0.07, 1.0, 1.0, 40.0)
    assert p["baro_noise_pa"] == 0.5 and p["max_rate_hz"] == 200.0 and "Hiwonder" in p["source"]


def test_rpy_and_pressure_helpers():
    assert np.allclose(imu_sim.rpy_deg((1, 0, 0, 0)), 0)
    q = (math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4))               # 90 deg yaw
    assert np.allclose(imu_sim.rpy_deg(q), (0, 0, 90))
    assert imu_sim.wrap_deg(190) == -170
    for h in (0.0, 0.5, 10.0):
        assert abs(imu_sim._altitude_m(imu_sim._pressure_pa(h)) - h) < 1e-6
    assert abs(imu_sim._pressure_pa(0.0) - imu_sim._pressure_pa(1.0) - 12.0) < 0.5   # ~12 Pa per metre


def _fake_model_data():
    """A one-body MuJoCo model with the slot's IMU sensor names."""
    import mujoco
    # welded to the world (no joint): the accelerometer reads +g, as a robot standing still does
    xml = """<mujoco><worldbody><body name="r_base" pos="0 0 0.7">
      <geom type="box" size=".1 .1 .1" mass="1"/><site name="r_imu" pos="0 0 0"/></body></worldbody>
      <sensor><framequat name="r_imu_quat" objtype="site" objname="r_imu"/>
      <gyro name="r_imu_gyro" site="r_imu"/><accelerometer name="r_imu_acc" site="r_imu"/></sensor></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)          # noqa: E731
    slot = SimpleNamespace(prefix="r_", quat_adr=int(m.sensor_adr[sid("r_imu_quat")]),
                           gyro_adr=int(m.sensor_adr[sid("r_imu_gyro")]))
    return mujoco, m, d, slot


def test_sim_imu_noise_matches_the_datasheet_and_uses_its_own_stream():
    mujoco, m, d, slot = _fake_model_data()
    m.opt.gravity[:] = (0, 0, -9.81)
    mujoco.mj_forward(m, d)
    assert abs(d.sensordata[slot.gyro_adr + 5] - 9.81) < 1e-6          # accel z = +g at rest
    rng_state = np.random.get_state()
    imu = imu_sim.SimIM10A(m, slot, seed=1, physics_dt=0.005, rate_hz=100.0)
    assert imu.every == 2 and abs(imu.rate_hz - 100.0) < 1e-9
    g, a = [], []
    for k in range(3000):                                           # held still: sensors constant
        s = imu.sample(d, k * imu.dt)
        g.append(s["gyro_dps"]); a.append(s["accel"])
    g, a = np.array(g), np.array(a)
    assert np.allclose(g.std(axis=0), 0.07, rtol=0.1)              # deg/s rms
    assert np.allclose(a.std(axis=0), 1e-3 * imu_sim.G, rtol=0.1)  # 1 mg rms
    assert abs(np.linalg.norm(g.mean(axis=0)) - 1.0) < 0.05          # 1 deg/s bias magnitude
    assert abs(np.linalg.norm(imu.accel_bias) - 0.040 * imu_sim.G) < 1e-9
    assert np.array_equal(np.random.get_state()[1], rng_state[1])  # global numpy stream untouched
    last = imu.latest
    assert abs(last["rpy_err"][0]) < 3 and abs(last["rpy_err"][1]) < 3   # filter tracks tilt at rest
    assert abs(np.linalg.norm(last["mag_ut"]) - np.linalg.norm(imu_sim.FIELD_ENU_UT)) < 0.01
    assert abs(last["baro_dalt_m"]) < 0.3
    assert len(imu.hist) == imu.hist.maxlen == 400
    # same seed -> same stream; a different seed -> a different bias
    assert np.array_equal(imu_sim.SimIM10A(m, slot, 1, 0.005).gyro_bias, imu.gyro_bias)
    assert not np.array_equal(imu_sim.SimIM10A(m, slot, 2, 0.005).gyro_bias, imu.gyro_bias)
    with pytest.raises(ValueError):
        imu_sim.SimIM10A(m, slot, 1, 0.005, rate_hz=400.0)


def test_attach_chains_an_existing_hook_and_samples_at_rate():
    mujoco, m, d, slot = _fake_model_data()
    calls = []
    runner = SimpleNamespace(substep_hook=lambda data: calls.append(1))
    imu = imu_sim.SimIM10A(m, slot, seed=0, physics_dt=0.005, rate_hz=50.0)
    imu.attach(runner)
    for _ in range(40):
        runner.substep_hook(d)
    assert len(calls) == 40 and len(imu.hist) == 10                  # previous hook every substep; IMU every 4th


def test_imu_panels_draw():
    hist = [(t * 0.01, np.array([10.0, -5.0, 30.0]) * math.sin(t / 10), np.array([0.3, -0.1, 9.8])) for t in range(400)]
    latest = {"t": 3.99, "gyro_dps": hist[-1][1], "accel": hist[-1][2], "mag_ut": imu_sim.FIELD_ENU_UT,
              "mag_heading_deg": 12.0, "baro_pa": 101300.0, "baro_dalt_m": 0.01,
              "rpy_est": np.array([1.0, -2.0, 45.0]), "rpy_true": np.array([1.2, -1.8, 44.0]),
              "rpy_err": np.array([-0.2, -0.2, 1.0])}
    strip = panels.imu_strip(hist, latest, (1280, 190), "imu")
    assert strip.size == (1280, 190)
    a = np.asarray(strip).astype(int)
    assert (np.abs(a - np.array(panels.AXIS_COLOURS[2])).sum(-1) < 30).any()   # the z trace is drawn
    ro = panels.imu_readout_panel(latest, (320, 220), "readout", ["note"])
    assert ro.size == (320, 220)
    assert panels.imu_strip([], None, (400, 100), "x").size == (400, 100)
    rgb = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    assert panels.stereo_rgb_panel(rgb, rgb, (320, 150), "rgb", "sub").size == (320, 150)


def test_recorder_composes_the_extra_columns_without_opengl(tmp_path):
    """The layout path the GPU render takes, with synthetic rig images (no GL on the login node)."""
    pytest.importorskip("cv2")
    import maze_explore as me
    from bhl_robust.eval.rig_views import FlowView

    args = SimpleNamespace(no_render=True, width=640, height=360, stride=2, policy=None, gait_run=None,
                           imu_source="truth", pose_bias_m=0.0, pose_noise_m=0.0, pose_yaw_deg=0.0, variant="biped")
    rec = me.ExploreRecorder.__new__(me.ExploreRecorder)
    rec.args, rec.side_w, rec.h, rec.w, rec.strip_h, rec.eff_side_w = args, 320, 360, 640, 190, 320
    rec._dt = 0.04
    rec._runner_d = None
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)

    class FakeRig:
        def render(self, d):
            return img, np.roll(img, 3, axis=1), np.full((120, 160), 2.0), np.full((120, 160), 3.0)

    rec.rig, rec.flow = FakeRig(), FlowView()
    rec.flow.update(np.roll(img, -3, axis=1), 0.08)                  # a previous frame, so flow exists
    hist = [(k * 0.01, np.zeros(3), np.array([0, 0, 9.8])) for k in range(50)]
    latest = {"t": 0.49, "gyro_dps": np.zeros(3), "accel": np.array([0, 0, 9.8]), "mag_ut": imu_sim.FIELD_ENU_UT,
              "mag_heading_deg": 0.0, "baro_pa": 101325.0, "baro_dalt_m": 0.0, "rpy_est": np.zeros(3),
              "rpy_true": np.zeros(3), "rpy_err": np.zeros(3)}
    rec.imu = SimpleNamespace(hist=hist, latest=latest, rate_hz=100.0)
    col_a = [panels.image_panel(None, (320, 170), "a"), panels.image_panel(None, (320, 170), "b"),
             panels.image_panel(None, (320, 360 + 190 - 340), "map")]
    frame = rec._compose_extras(np.zeros((360, 640, 3), np.uint8), col_a, "header", "footer", 360 + 190)
    assert frame.shape[1] == 640 + 2 * 320
    assert frame.shape[0] == 360 + 190 + 2 * 34
    assert rec.eff_side_w == 640
    from PIL import Image
    Image.fromarray(frame).save(tmp_path / "layout.png")
