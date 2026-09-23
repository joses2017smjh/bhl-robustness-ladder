import torch
import pytest

from bhl_robust.robust_sensing import RobustSensingState


def test_dropout_bias_and_delay_semantics():
    g = torch.Generator().manual_seed(0)
    s = RobustSensingState(2000, "cpu", p_lidar_off=0.2, p_stereo_off=0.2, gyro_bias_std=0.02, max_delay_steps=1, generator=g)
    s.resample(torch.arange(2000))
    assert abs(1 - s.lidar_on.float().mean().item() - 0.2) < 0.04
    assert abs(1 - s.stereo_on.float().mean().item() - 0.2) < 0.04
    assert abs(s.gyro_bias.std().item() - 0.02) < 0.004
    assert set(s.delay.unique().tolist()) == {0, 1} and 0.4 < s.delay.float().mean().item() < 0.6
    # Zeroing only the dropped rows.
    lidar = torch.ones(2000, 36)
    out = s.lidar(lidar)
    assert torch.equal(out.sum(1) > 0, s.lidar_on)
    # Delay: delayed envs see the previous step's (biased) value, others the current one.
    a1 = torch.randn(2000, 3); a2 = torch.randn(2000, 3)
    s.angular_velocity(a1)
    got = s.angular_velocity(a2)
    delayed = s.delay > 0
    assert torch.allclose(got[delayed], (a1 + s.gyro_bias)[delayed])
    assert torch.allclose(got[~delayed], (a2 + s.gyro_bias)[~delayed])


def test_eval_overrides_and_validation():
    s = RobustSensingState(8, "cpu", p_lidar_off=1.0, max_delay_steps=2)
    s.force_lidar_on = True; s.force_delay = 2
    s.resample([0, 1, 2, 3, 4, 5, 6, 7])
    assert s.lidar_on.all() and (s.delay == 2).all()
    with pytest.raises(ValueError):
        RobustSensingState(4, "cpu", p_lidar_off=1.5)
    with pytest.raises(ValueError):
        RobustSensingState(4, "cpu", max_delay_steps=-1)
