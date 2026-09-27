"""SF-04 bias setting of scripts/bench/maze_recovery_probe.py, without Isaac.

The probe boots Kit (AppLauncher) at import, so the pure helpers, the
_ImuDelay hook and _apply_setting are pulled out of its source with `ast` and
executed against torch / math only. What is checked: the bias vectors
(norm, determinism, independence of gyro and gravity), the settings-key
parser, that the observation hook adds the bias to exactly the IMU columns,
that every pre-existing path of the hook (no bias: zero_terms, delay 0/1/2) is
unchanged, that settings without a bias key produce the same `applied` row as
before, and that the SF-04 bias launcher's settings JSON parses.
"""
import ast
import json
import math
import re
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "scripts" / "bench" / "maze_recovery_probe.py"
LAUNCHER = REPO / "slurm" / "repo20260923" / "gpu_sf04_bias_probe.sbatch"
WANT = {"IMU_BIAS_KEYS", "IMU_BIAS_SEED_OFFSET", "IMU_BIAS_MODEL",
        "_parse_imu_bias", "_imu_bias_vectors", "_ImuDelay", "_apply_setting"}


def _load():
    tree = ast.parse(PROBE.read_text())
    body = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in WANT:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in WANT for t in node.targets):
            body.append(node)
    found = {getattr(n, "name", None) or n.targets[0].id for n in body}
    assert found == WANT, f"probe is missing {WANT - found}"
    ns = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(PROBE), "exec"), ns)
    return types.SimpleNamespace(**{k: ns[k] for k in WANT})


P = _load()


# ------------------------------------------------------------------ vectors
@pytest.mark.parametrize("gyro,grav", [(0.02, 0.02), (0.05, 0.05), (0.10, 0.10), (0.05, 0.0)])
def test_bias_vector_norm_is_the_magnitude(gyro, grav):
    g, v = P._imu_bias_vectors(32, gyro, grav, 100)
    assert g.shape == (32, 3) and v.shape == (32, 3)
    assert g.dtype == torch.float32 and g.device.type == "cpu"
    assert torch.allclose(g.norm(dim=1), torch.full((32,), gyro), atol=1e-6)
    assert torch.allclose(v.norm(dim=1), torch.full((32,), grav), atol=1e-6)


def test_zero_magnitude_is_exact_zero():
    g, v = P._imu_bias_vectors(32, 0.0, 0.0, 100)
    assert torch.count_nonzero(g) == 0 and torch.count_nonzero(v) == 0


def test_deterministic_and_seed_dependent():
    a = P._imu_bias_vectors(32, 0.05, 0.05, 100)
    b = P._imu_bias_vectors(32, 0.05, 0.05, 100)
    c = P._imu_bias_vectors(32, 0.05, 0.05, 101)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.allclose(a[0], c[0])


def test_same_directions_at_every_magnitude():
    """Paired comparison: only the length changes between bias levels."""
    unit = [P._imu_bias_vectors(32, m, m, 100) for m in (0.02, 0.05, 0.10)]
    for g, v in unit[1:]:
        assert torch.allclose(g / g.norm(dim=1, keepdim=True), unit[0][0] / 0.02, atol=1e-5)
        assert torch.allclose(v / v.norm(dim=1, keepdim=True), unit[0][1] / 0.02, atol=1e-5)
    # directions do not depend on the other term's magnitude either
    g_only, _ = P._imu_bias_vectors(32, 0.05, 0.0, 100)
    assert torch.equal(g_only, P._imu_bias_vectors(32, 0.05, 0.05, 100)[0])


def test_gyro_and_gravity_directions_are_independent_and_spread():
    g, v = P._imu_bias_vectors(4096, 1.0, 1.0, 7)
    assert not torch.allclose(g, v)
    assert abs(float((g * v).sum(dim=1).mean())) < 0.05     # independent unit vectors: E[cos] = 0
    assert float(g.mean(dim=0).norm()) < 0.05               # uniform on the sphere: mean ~ 0
    assert torch.allclose((g ** 2).mean(dim=0), torch.full((3,), 1 / 3), atol=0.03)


# ------------------------------------------------------------------- parser
def test_parse_defaults_and_values():
    assert P._parse_imu_bias({}) == (0.0, 0.0)
    assert P._parse_imu_bias({"gyro_bias": None, "gravity_bias": None}) == (0.0, 0.0)
    assert P._parse_imu_bias({"gyro_bias": 0.05, "gravity_bias": 0.1}) == (0.05, 0.1)
    assert P._parse_imu_bias({"gyro_bias": 0}) == (0.0, 0.0)


@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf"), "0.05", True, [0.05]])
def test_parse_rejects(bad):
    with pytest.raises(ValueError):
        P._parse_imu_bias({"gyro_bias": bad})
    with pytest.raises(ValueError):
        P._parse_imu_bias({"gravity_bias": bad})


# ------------------------------------------------------------ the obs hook
N, D = 5, 20
SLICES = {"base_ang_vel": slice(3, 6), "projected_gravity": slice(6, 9)}


def _obs_seq(steps=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(N, D, generator=g) for _ in range(steps)]


def _reference(seq, steps, zero=()):
    """Hand-written FIFO: IMU columns from `steps` calls ago (the first value
    repeats until the queue fills), zero_slices zeroed before the delay."""
    out, hist = [], []
    for x in seq:
        x = x.clone()
        for s in zero:
            x[:, s] = 0.0
        hist.append(x.clone())
        src = hist[max(0, len(hist) - 1 - steps)] if steps > 0 else x
        y = x.clone()
        for s in SLICES.values():
            y[:, s] = src[:, s]
        out.append(y)
    return out


@pytest.mark.parametrize("steps", [0, 1, 2])
@pytest.mark.parametrize("zero", [(), (slice(12, 20),)])
@pytest.mark.parametrize("as_dict", [False, True])
def test_hook_without_bias_is_unchanged(steps, zero, as_dict):
    seq = _obs_seq()
    for bias in (None, {}):
        hook = P._ImuDelay(SLICES, steps, zero, bias=bias)
        legacy = P._ImuDelay(SLICES, steps, zero)          # the pre-existing call signature
        for x, want in zip(seq, _reference(seq, steps, zero)):
            got = hook({"policy": x.clone()})["policy"] if as_dict else hook(x.clone())
            old = legacy({"policy": x.clone()})["policy"] if as_dict else legacy(x.clone())
            assert torch.equal(got, want) and torch.equal(old, want)


@pytest.mark.parametrize("steps", [0, 1, 2])
@pytest.mark.parametrize("as_dict", [False, True])
def test_hook_adds_bias_to_imu_columns_only(steps, as_dict):
    seq = _obs_seq()
    gv, pv = P._imu_bias_vectors(N, 0.05, 0.10, 100)
    hook = P._ImuDelay(SLICES, steps, (), bias={"base_ang_vel": gv, "projected_gravity": pv})
    for x, want in zip(seq, _reference(seq, steps)):
        inp = x.clone()
        got = hook({"policy": inp})["policy"] if as_dict else hook(inp)
        assert torch.equal(inp, x), "the hook must not modify its input in place"
        want = want.clone()
        want[:, 3:6] += gv
        want[:, 6:9] += pv
        assert torch.allclose(got, want, atol=1e-7)
        assert torch.equal(got[:, :3], x[:, :3]) and torch.equal(got[:, 9:], x[:, 9:])


def test_zero_bias_is_a_numerical_no_op():
    seq = _obs_seq()
    gv, pv = P._imu_bias_vectors(N, 0.0, 0.0, 100)
    hook = P._ImuDelay(SLICES, 0, (), bias={"base_ang_vel": gv, "projected_gravity": pv})
    for x in seq:
        assert torch.equal(hook(x.clone()), x)


def test_zero_terms_override_bias():
    """An outage (zero_terms) wins over the bias, as a dead sensor reads zero."""
    x = _obs_seq(1)[0]
    gv, pv = P._imu_bias_vectors(N, 0.05, 0.05, 100)
    hook = P._ImuDelay(SLICES, 0, (SLICES["base_ang_vel"],), bias={"base_ang_vel": gv, "projected_gravity": pv})
    y = hook(x.clone())
    assert torch.count_nonzero(y[:, 3:6]) == 0
    assert torch.allclose(y[:, 6:9], x[:, 6:9] + pv)


def test_hook_rejects_bad_bias():
    with pytest.raises(RuntimeError):
        P._ImuDelay({"base_ang_vel": slice(3, 6)}, 0, (), bias={"projected_gravity": torch.zeros(N, 3)})
    with pytest.raises(RuntimeError):
        P._ImuDelay(SLICES, 0, (), bias={"base_ang_vel": torch.zeros(N, 2)})


# ----------------------------------------------------------- _apply_setting
def _fake_env(num_envs=4):
    cfgs = {t: types.SimpleNamespace(noise=types.SimpleNamespace(std=s))
            for t, s in (("base_ang_vel", 0.05), ("projected_gravity", 0.02))}
    om = types.SimpleNamespace(get_term_cfg=lambda group, term: cfgs[term],
                               active_terms={"policy": list(cfgs)})
    cmd = types.SimpleNamespace()
    cm = types.SimpleNamespace(get_term=lambda name: cmd)
    return types.SimpleNamespace(observation_manager=om, command_manager=cm, device="cpu", num_envs=num_envs)


OLD_KEYS = {"gyro_std", "gravity_std", "imu_delay_steps", "pose_bias_m", "pose_noise_m", "pose_yaw_deg"}


@pytest.mark.parametrize("setting", [
    {"name": "baseline", "seed": 100},
    {"name": "delay1", "imu_delay_steps": 1, "seed": 100},
    {"name": "gyro0.10", "gyro_std": 0.10, "seed": 100},
    {"name": "lidar_off", "zero_terms": ["lidar"], "seed": 100},
    {"name": "loc", "pose_bias_m": 0.2, "pose_noise_m": 0.05, "pose_yaw_deg": 5, "seed": 100},
])
def test_existing_settings_produce_the_same_applied_row(setting):
    applied = P._apply_setting(_fake_env(), dict(setting))
    assert set(applied) == OLD_KEYS


def test_bias_setting_is_recorded():
    applied = P._apply_setting(_fake_env(), {"name": "bias0.05", "gyro_bias": 0.05, "gravity_bias": 0.05, "seed": 100})
    assert applied["gyro_bias"] == 0.05 and applied["gravity_bias"] == 0.05
    assert applied["imu_bias_seed"] == 100 and "policy-side" in applied["imu_bias_model"]
    assert OLD_KEYS <= set(applied)
    zero = P._apply_setting(_fake_env(), {"name": "bias0", "gyro_bias": 0.0, "gravity_bias": 0.0, "seed": 100})
    assert zero["gyro_bias"] == zero["gravity_bias"] == 0.0


def test_bad_bias_setting_raises_in_apply_setting():
    """Surfaces as SETTING-ERROR in the probe (the loop catches _apply_setting)."""
    with pytest.raises(ValueError):
        P._apply_setting(_fake_env(), {"name": "bad", "gyro_bias": -1, "seed": 100})


# ------------------------------------------------------------ the launcher
def test_launcher_settings_json():
    text = LAUNCHER.read_text()
    m = re.search(r"^SETTINGS='(\[.*\])'$", text, flags=re.M)
    assert m, "SETTINGS='[...]' line not found in the launcher"
    items = json.loads(m.group(1))
    assert [s["name"] for s in items] == ["bias0", "bias0.02", "bias0.05", "bias0.10"]
    for s, level in zip(items, (0.0, 0.02, 0.05, 0.10)):
        assert P._parse_imu_bias(s) == (level, level)
        assert set(s) == {"name", "gyro_bias", "gravity_bias"}   # nothing else perturbed
