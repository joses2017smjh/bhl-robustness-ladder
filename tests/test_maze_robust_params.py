"""SF-04 single-ingredient arms: each parameter set perturbs only its own
ingredient, and every arm survives the evaluation probe's forcing code.
Pure torch (bhl_robust.robust_sensing); no Isaac import."""
import torch
import pytest

from bhl_robust.robust_sensing import SF04_ARM_PARAMS, SF04_PARAM_NAMES, RobustSensingState, sf04_params

N = 2000
ARMS = ("BothRobust", "BothDelay", "BothDrop", "BothBias")


def make(arm, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = RobustSensingState(N, "cpu", generator=g, **SF04_ARM_PARAMS[arm])
    s.resample(torch.arange(N))
    return s


def test_param_table_shape_and_types():
    assert set(SF04_ARM_PARAMS) == set(ARMS)
    for arm, p in SF04_ARM_PARAMS.items():
        assert tuple(p) == SF04_PARAM_NAMES, arm
        for name in SF04_PARAM_NAMES[:-1]:
            assert type(p[name]) is float, (arm, name)   # Hydra round-trip isinstance check
        assert type(p["max_delay_steps"]) is int, arm
    d, r, b = SF04_ARM_PARAMS["BothDelay"], SF04_ARM_PARAMS["BothDrop"], SF04_ARM_PARAMS["BothBias"]
    assert d["max_delay_steps"] == 1 and d["p_lidar_off"] == d["p_stereo_off"] == 0.0 \
        and d["gyro_bias_std"] == d["gravity_bias_std"] == 0.0
    assert r["p_lidar_off"] == r["p_stereo_off"] == 0.2 and r["max_delay_steps"] == 0 \
        and r["gyro_bias_std"] == r["gravity_bias_std"] == 0.0
    assert b["gyro_bias_std"] == b["gravity_bias_std"] == 0.02 and b["max_delay_steps"] == 0 \
        and b["p_lidar_off"] == b["p_stereo_off"] == 0.0
    full = SF04_ARM_PARAMS["BothRobust"]
    assert full["p_lidar_off"] == 0.2 and full["gyro_bias_std"] == 0.02 and full["max_delay_steps"] == 1


def test_sf04_params_reads_cfg_objects_dicts_and_defaults():
    class Cfg:
        p_lidar_off = 0.0; p_stereo_off = 0.0; gyro_bias_std = 0.0; gravity_bias_std = 0.0; max_delay_steps = 1
    assert sf04_params(Cfg()) == SF04_ARM_PARAMS["BothDelay"]
    assert sf04_params(dict(SF04_ARM_PARAMS["BothDrop"])) == SF04_ARM_PARAMS["BothDrop"]
    assert sf04_params(None) == SF04_ARM_PARAMS["BothRobust"]
    # Partial source: missing entries fall back to BothRobust; ints/floats normalised.
    got = sf04_params({"max_delay_steps": 0.0, "p_lidar_off": 0})
    assert got["max_delay_steps"] == 0 and type(got["max_delay_steps"]) is int
    assert got["p_lidar_off"] == 0.0 and type(got["p_lidar_off"]) is float
    assert got["gyro_bias_std"] == 0.02


def test_delay_only_never_zeroes_sensors_or_biases():
    s = make("BothDelay")
    assert s.lidar_on.all() and s.stereo_on.all()
    assert torch.equal(s.gyro_bias, torch.zeros(N, 3)) and torch.equal(s.gravity_bias, torch.zeros(N, 3))
    assert set(s.delay.unique().tolist()) == {0, 1} and 0.4 < s.delay.float().mean().item() < 0.6
    lidar, stereo = torch.rand(N, 36), torch.rand(N, 32)
    assert torch.equal(s.lidar(lidar), lidar) and torch.equal(s.stereo(stereo), stereo)
    a1, a2 = torch.randn(N, 3), torch.randn(N, 3)
    s.angular_velocity(a1)
    got = s.angular_velocity(a2)
    delayed = s.delay > 0
    assert delayed.any() and (~delayed).any()
    assert torch.equal(got[delayed], a1[delayed]) and torch.equal(got[~delayed], a2[~delayed])  # unbiased
    g1, g2 = torch.randn(N, 3), torch.randn(N, 3)
    s.gravity(g1)
    got = s.gravity(g2)
    assert torch.equal(got[delayed], g1[delayed]) and torch.equal(got[~delayed], g2[~delayed])


def test_dropout_only_never_delays_or_biases():
    s = make("BothDrop")
    assert (s.delay == 0).all()
    assert torch.equal(s.gyro_bias, torch.zeros(N, 3)) and torch.equal(s.gravity_bias, torch.zeros(N, 3))
    assert abs(1 - s.lidar_on.float().mean().item() - 0.2) < 0.04
    assert abs(1 - s.stereo_on.float().mean().item() - 0.2) < 0.04
    lidar, stereo = torch.rand(N, 36) + 0.1, torch.rand(N, 32) + 0.1
    assert torch.equal(s.lidar(lidar).sum(1) > 0, s.lidar_on)
    assert torch.equal(s.stereo(stereo).sum(1) > 0, s.stereo_on)
    a1, a2 = torch.randn(N, 3), torch.randn(N, 3)
    s.angular_velocity(a1)
    assert torch.equal(s.angular_velocity(a2), a2)        # current, unbiased, for every env
    g1, g2 = torch.randn(N, 3), torch.randn(N, 3)
    s.gravity(g1)
    assert torch.equal(s.gravity(g2), g2)


def test_bias_only_never_delays_or_zeroes_sensors():
    s = make("BothBias")
    assert (s.delay == 0).all() and s.lidar_on.all() and s.stereo_on.all()
    assert abs(s.gyro_bias.std().item() - 0.02) < 0.004
    assert abs(s.gravity_bias.std().item() - 0.02) < 0.004
    lidar, stereo = torch.rand(N, 36), torch.rand(N, 32)
    assert torch.equal(s.lidar(lidar), lidar) and torch.equal(s.stereo(stereo), stereo)
    a1, a2 = torch.randn(N, 3), torch.randn(N, 3)
    s.angular_velocity(a1)
    assert torch.allclose(s.angular_velocity(a2), a2 + s.gyro_bias)   # biased, not delayed
    g1, g2 = torch.randn(N, 3), torch.randn(N, 3)
    s.gravity(g1)
    assert torch.allclose(s.gravity(g2), g2 + s.gravity_bias)
    # Bias is constant within the episode.
    assert torch.allclose(s.angular_velocity(a2), a2 + s.gyro_bias)


def test_both_robust_has_all_three():
    s = make("BothRobust")
    assert abs(1 - s.lidar_on.float().mean().item() - 0.2) < 0.04
    assert abs(1 - s.stereo_on.float().mean().item() - 0.2) < 0.04
    assert abs(s.gyro_bias.std().item() - 0.02) < 0.004
    assert set(s.delay.unique().tolist()) == {0, 1}


@pytest.mark.parametrize("arm", ARMS)
def test_probe_forcing_turns_every_arm_into_identity(arm):
    # Mirrors scripts/bench/maze_recovery_probe.py under --settings.
    s = make(arm)
    s.force_lidar_on, s.force_stereo_on, s.force_delay = True, True, 0
    s.gyro_bias_std = s.gravity_bias_std = 0.0
    s.resample(torch.arange(N))
    assert s.lidar_on.all() and s.stereo_on.all() and (s.delay == 0).all()
    assert torch.equal(s.gyro_bias, torch.zeros(N, 3)) and torch.equal(s.gravity_bias, torch.zeros(N, 3))
    lidar, stereo = torch.rand(N, 36), torch.rand(N, 32)
    assert torch.equal(s.lidar(lidar), lidar) and torch.equal(s.stereo(stereo), stereo)
    a1, a2 = torch.randn(N, 3), torch.randn(N, 3)
    s.angular_velocity(a1)
    assert torch.equal(s.angular_velocity(a2), a2)
    g1, g2 = torch.randn(N, 3), torch.randn(N, 3)
    s.gravity(g1)
    assert torch.equal(s.gravity(g2), g2)


def test_partial_resample_touches_only_reset_rows():
    s = make("BothRobust")
    before = s.delay.clone(), s.lidar_on.clone(), s.gyro_bias.clone()
    ids = torch.arange(0, N, 2)
    s.resample(ids)
    keep = torch.ones(N, dtype=torch.bool); keep[ids] = False
    assert torch.equal(s.delay[keep], before[0][keep]) and torch.equal(s.lidar_on[keep], before[1][keep])
    assert torch.equal(s.gyro_bias[keep], before[2][keep])
