"""Waiter phase 3 lerobot glue (scripts/waiter/vla/smolvla_io.py), CPU only, no model weights loaded.

Needs venv-lehome-train (lerobot 0.4.3) and the local HF cache (smolvla_base config + SmolVLM2 tokenizer); skipped
elsewhere. Run:
  HF_HOME=/nfs/hpc/share/sanchej7/Humanoid_Lite/.cache/huggingface HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/nfs/hpc/share/sanchej7/Humanoid_Lite/envs-waiter/pytest-overlay \
  /nfs/hpc/share/sanchej7/Humanoid_Lite/venv-lehome-train/bin/python -m pytest -q tests/test_waiter_vla_lerobot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")
torch = pytest.importorskip("torch")

VLA = Path(__file__).resolve().parents[1] / "scripts" / "waiter" / "vla"
sys.path.insert(0, str(VLA))

import smolvla_io as io  # noqa: E402
import waiter_vla_data as wd  # noqa: E402


@pytest.fixture(scope="module")
def base_dir():
    try:
        return io.resolve_base(io.BASE_REPO)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"smolvla_base not in the local HF cache: {exc}")


@pytest.fixture(scope="module")
def stats():
    rng = np.random.default_rng(0)
    state = rng.standard_normal((400, 30)) * np.linspace(0.1, 2, 30) + np.arange(30)
    state[:, 24:27] = [0.0, 0.0, -1.0]  # constant gravity -> floored std
    action = rng.standard_normal((400, 15)) * 0.2
    action[:, 13:15] = rng.integers(0, 2, size=(400, 2))
    return {wd.STATE_KEY: wd.compute_norm_stats([state.astype(np.float32)]),
            wd.ACTION_KEY: wd.compute_norm_stats([action.astype(np.float32)])}


def test_config_targets_waiter_features(base_dir):
    cfg = io.make_config(base_dir, chunk_size=50, n_action_steps=10, device="cpu")
    assert cfg.load_vlm_weights is False  # everything must come from smolvla_base's safetensors
    assert list(cfg.image_features) == [wd.IMAGE_KEY]
    assert cfg.robot_state_feature.shape == (30,) and cfg.action_feature.shape == (15,)
    assert cfg.max_state_dim >= 30 and cfg.max_action_dim >= 15
    assert cfg.chunk_size == 50 and cfg.n_action_steps == 10
    assert tuple(cfg.resize_imgs_with_padding) == (512, 512)
    assert cfg.freeze_vision_encoder and cfg.train_expert_only
    full = io.make_config(base_dir, chunk_size=50, n_action_steps=10, device="cpu", train_mode="full")
    assert not full.freeze_vision_encoder and not full.train_expert_only
    with pytest.raises(ValueError):
        io.make_config(base_dir, chunk_size=10, n_action_steps=11, device="cpu")


def test_processors_normalise_like_numpy_and_survive_checkpoint_roundtrip(base_dir, stats, tmp_path):
    cfg = io.make_config(base_dir, chunk_size=8, n_action_steps=4, device="cpu")
    pre, post = io.make_processors(cfg, stats)
    rng = np.random.default_rng(1)
    images = rng.integers(0, 256, size=(3, 224, 224, 3), dtype=np.uint8)
    states = rng.standard_normal((3, 30)).astype(np.float32)
    states[:, 24:27] = [0.0, 0.01, -1.0]
    actions = rng.standard_normal((3, 8, 15)).astype(np.float32) * 0.2
    tasks = ["pick up the red tumbler, put it on the tray and bring the tray to the left table"] * 3
    out = pre(io.to_batch(images, states, tasks, "cpu", actions=actions))
    np.testing.assert_allclose(out[wd.STATE_KEY].numpy(), wd.normalize(states, stats[wd.STATE_KEY]), atol=1e-4)
    np.testing.assert_allclose(out[wd.ACTION_KEY].numpy(), wd.normalize(actions, stats[wd.ACTION_KEY]), atol=1e-4)
    assert abs(out[wd.STATE_KEY][0, 25].item() - 1.0) < 1e-4  # 0.01 / floored std 0.01
    img = out[wd.IMAGE_KEY]
    assert img.shape == (3, 3, 224, 224) and img.dtype == torch.float32 and 0 <= img.min() and img.max() <= 1
    np.testing.assert_allclose(img[0, :, 5, 7].numpy() * 255, images[0, 5, 7], atol=1e-3)
    tokens = out["observation.language.tokens"]
    assert tokens.shape == (3, cfg.tokenizer_max_length)
    # post-processing unnormalises whole (B, chunk, 15) chunks
    norm = torch.as_tensor(wd.normalize(actions, stats[wd.ACTION_KEY]), dtype=torch.float32)
    np.testing.assert_allclose(post(norm).numpy(), actions, atol=1e-4)
    # the statistics travel with the checkpoint
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    pre2, post2 = io.load_processors(cfg, tmp_path)
    out2 = pre2(io.to_batch(images, states, tasks, "cpu", actions=actions))
    for key in (wd.STATE_KEY, wd.ACTION_KEY, wd.IMAGE_KEY, "observation.language.tokens"):
        assert torch.equal(out[key], out2[key]), key
    np.testing.assert_allclose(post2(norm).numpy(), actions, atol=1e-4)


def test_instruction_fits_tokenizer_without_truncation(base_dir):
    from transformers import AutoTokenizer

    cfg = io.make_config(base_dir, chunk_size=50, n_action_steps=10, device="cpu")
    tok = AutoTokenizer.from_pretrained(cfg.vlm_model_name)
    longest = "take the yellow tumbler by its handle, set it on the tray and deliver the tray to the right table\n"
    assert len(tok(longest)["input_ids"]) < cfg.tokenizer_max_length


def test_chunk_loss_masks_episode_end():
    class FakePolicy:
        def forward(self, batch, noise=None, time=None):
            losses = torch.ones(2, 4, 32)
            losses[0, 2:] = 100.0  # padded steps of sample 0 carry a huge loss
            losses[:, :, 15:] = 3.0  # padded action dims
            return losses.mean(), {"losses_after_forward": losses}

    is_pad = torch.tensor([[False, False, True, True], [False, False, False, False]])
    loss, per_sample = io.chunk_loss(FakePolicy(), {}, is_pad)
    # backprop loss: over all 32 dims, valid steps only -> (15*1 + 17*3)/32 for every valid step
    assert loss.item() == pytest.approx((15 + 17 * 3) / 32)
    # per-sample: real 15 dims only, valid steps only
    assert per_sample.tolist() == pytest.approx([1.0, 1.0])


def test_to_batch_rejects_wrong_image_layout():
    with pytest.raises(ValueError):
        io.to_batch(np.zeros((1, 3, 224, 224), np.uint8), np.zeros((1, 30), np.float32), ["x"], "cpu")
    with pytest.raises(ValueError):
        io.to_batch(np.zeros((1, 224, 224, 3), np.float32), np.zeros((1, 30), np.float32), ["x"], "cpu")


def test_fp32_master_weights_survive_save_and_load(tmp_path):
    """A bf16-constructed module trained with fp32 master weights must reload in fp32, bit-exact."""
    from safetensors.torch import save_model

    def make():
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
        m[0].to(torch.bfloat16)  # like the action expert at construction
        return m

    trained = make()
    for p in trained.parameters():
        p.requires_grad_(True)
    assert io.cast_trainable_fp32(trained) == 2
    with torch.no_grad():
        trained[0].weight.add_(1e-5)  # an update bf16 could not represent
    save_model(trained, str(tmp_path / "model.safetensors"))
    fresh = make()
    assert io.match_file_dtypes(fresh, tmp_path / "model.safetensors") == 2
    assert fresh[0].weight.dtype == torch.float32
    assert torch.equal(fresh[0].weight, trained[0].weight)
    assert torch.equal(fresh[1].weight, trained[1].weight)
    report = io.dtype_report(fresh)
    assert set(report) == {"trainable_float32"}
