#!/usr/bin/env python3
"""Frozen SmolVLA with an explicit initial flow noise: ``chunk = policy_with_noise(obs, noise)``.

DSRL (Wagenmaker et al., CoRL 2025, arXiv 2506.15799) keeps the flow/diffusion policy frozen and lets a
small SAC agent choose the INITIAL NOISE of the denoiser. In lerobot 0.4.3 SmolVLA samples that noise in
``VLAFlowMatching.sample_actions`` (``lerobot/policies/smolvla/modeling_smolvla.py``):
``noise = sample_noise((B, chunk_size, max_action_dim))`` -> 10 Euler steps of ``denoise_step`` from
t=1 to t=0 over a KV cache of the prefix (images + language + state). Noise shape = ``(chunk_size=50,
max_action_dim=32)`` per sample; the action is the first ``action_dim`` (15 for the Waiter) columns.

This module splits that call in two, with no change to lerobot code:
  ``encode(obs_list) -> SmolVLAContext``  prefix pass (vision + VLM layers, KV cache) + pooled features
  ``denoise(ctx, noise) -> (B, chunk, action_dim)``  the 10-step Euler integration from the given noise
The prefix pass does not depend on the noise, so the SAC observation features come from the same pass
for free (see ``feature_mode``) and the KV cache can be reused for several noises.

``NoiseMap`` maps a low-dimensional latent (the SAC action) to the full noise tensor.

Self-check (GPU): ``python smolvla_noise.py --check --untuned-from <smolvla_base dir> --out check.json``
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

WAITER_STATE_DIM = 30
WAITER_ACTION_DIM = 15
WAITER_IMAGE_KEY = "observation.images.head_rgb"


# ------------------------------------------------------------------------------------------ noise map
class NoiseMap:
    """Latent ``w`` (B, latent_dim) -> initial noise (B, chunk_size, noise_dim).

    ``noise[b, t, d] = sum_j basis[t, j] * w[b, j, d]`` for ``d < active_dims``; inactive dims are 0.
    Modes:
      ``repeat`` (k=1): one noise row repeated over the whole chunk. This is what the official pi0 DSRL
                 code does (``lasgroup/swissai-dsrl`` examples/train_utils_sim.py: the SAC action is a
                 (1, 32) row, repeated over the 50-step chunk). latent_dim = active_dims.
      ``dct``    (k>=1): the k lowest-frequency orthonormal cosines over the chunk, scaled by 1/sqrt(k)
                 so a standard-normal latent gives unit average marginal variance. k=1 == repeat.
      ``full``:  every entry is free (k = chunk_size, identity basis): latent_dim = chunk * active_dims.
    """

    def __init__(self, chunk_size: int, noise_dim: int, mode: str = "repeat", k: int = 1,
                 active_dims: int | None = None):
        self.chunk_size, self.noise_dim = int(chunk_size), int(noise_dim)
        self.active_dims = int(active_dims or noise_dim)
        if not 1 <= self.active_dims <= self.noise_dim:
            raise ValueError("active_dims out of range")
        T = self.chunk_size
        if mode == "repeat":
            basis = np.ones((T, 1))
        elif mode == "dct":
            if not 1 <= k <= T:
                raise ValueError("dct k out of range")
            t = np.arange(T) + 0.5
            cols = [np.ones(T)] + [math.sqrt(2.0) * np.cos(math.pi * j * t / T) for j in range(1, k)]
            basis = np.stack(cols, 1) / math.sqrt(k)
        elif mode == "full":
            basis = np.eye(T)
        else:
            raise ValueError(f"unknown noise mode {mode!r}")
        self.mode, self.k = mode, basis.shape[1]
        self.basis = torch.as_tensor(basis, dtype=torch.float32)

    @property
    def latent_dim(self) -> int:
        return self.k * self.active_dims

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        w = torch.as_tensor(w, dtype=torch.float32)
        B = w.shape[0]
        basis = self.basis.to(w.device)
        z = torch.einsum("tk,bkd->btd", basis, w.reshape(B, self.k, self.active_dims))
        if self.active_dims == self.noise_dim:
            return z
        out = torch.zeros(B, self.chunk_size, self.noise_dim, device=w.device, dtype=w.dtype)
        out[:, :, : self.active_dims] = z
        return out

    def describe(self) -> dict:
        return {"mode": self.mode, "k": self.k, "active_dims": self.active_dims,
                "latent_dim": self.latent_dim, "noise_shape": [self.chunk_size, self.noise_dim]}


# ------------------------------------------------------------------------------------------ SmolVLA
@dataclass
class SmolVLAContext:
    prefix_pad_masks: torch.Tensor
    past_key_values: dict
    features: torch.Tensor          # (B, feature_dim) float32, no grad
    bsize: int


class SmolVLANoisePolicy:
    """Frozen SmolVLA: ``encode`` then ``denoise``; deterministic given the noise.

    feature_mode (SAC observation encoder; no gradient into the VLA):
      ``pooled``: [mean image-token output, mean language-token output, state-token output] of the
                  prefix pass (3 x VLM hidden = 2880 for SmolVLM2-500M) + the normalised 30-D state.
      ``mean``:   mean over all valid prefix tokens (960) + normalised state.
      ``state``:  normalised state only (ablation).
    """

    def __init__(self, policy, preprocessor, postprocessor, device="cuda", image_key: str | None = None,
                 feature_mode: str = "pooled"):
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
        self._tok_key, self._mask_key, self._state_key = OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE
        self.policy = policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)
        self.pre, self.post = preprocessor, postprocessor
        self.device = torch.device(device)
        cfg = policy.config
        self.config = cfg
        if cfg.rtc_config is not None and getattr(cfg.rtc_config, "enabled", False):
            raise ValueError("RTC-enabled configs are not supported by the noise wrapper")
        self.chunk_size, self.noise_dim = int(cfg.chunk_size), int(cfg.max_action_dim)
        self.action_dim = int(cfg.action_feature.shape[0])
        self.state_dim = int(cfg.robot_state_feature.shape[0])
        self.n_action_steps = int(cfg.n_action_steps)
        self.num_steps = int(cfg.num_steps)
        keys = list(cfg.image_features)
        if image_key is None:
            if len(keys) != 1 and WAITER_IMAGE_KEY not in keys:
                raise ValueError(f"cannot pick the head image key from {keys}; pass image_key")
            image_key = WAITER_IMAGE_KEY if WAITER_IMAGE_KEY in keys else keys[0]
        self.image_key = image_key
        if feature_mode not in ("pooled", "mean", "state"):
            raise ValueError(feature_mode)
        self.feature_mode = feature_mode
        hidden = int(self.policy.model.vlm_with_expert.config.text_config.hidden_size)
        self.feature_dim = {"pooled": 3 * hidden, "mean": hidden, "state": 0}[feature_mode] + self.state_dim

    # ---- construction
    @classmethod
    def from_pretrained(cls, path, device="cuda", **kw):
        """A fine-tuned SmolVLA directory (config.json, model.safetensors, policy_*processor*). A phase 3 checkpoint
        (written by scripts/waiter/vla/train_smolvla.py: it carries waiter_vla.json, or `path` is its best/latest
        .json pointer) is loaded by phase 3's own loader (dtype-exact weights, its processors), so the frozen base
        policy here is exactly the one phase 3's rollouts evaluated."""
        from pathlib import Path as _P
        pth = _P(str(path))
        if pth.suffix == ".json" or (pth / "waiter_vla.json").is_file():
            import sys as _sys
            _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "vla"))
            from smolvla_io import load_finetuned
            policy, pre, post, _cfg, _meta = load_finetuned(pth, device=str(device))
            return cls(policy, pre, post, device=device, **kw)
        import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401  (registers the config)
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        cfg = PreTrainedConfig.from_pretrained(str(path))
        cfg.load_vlm_weights = False   # all weights come from model.safetensors; no VLM download
        cfg.device = str(device)
        policy = SmolVLAPolicy.from_pretrained(str(path), config=cfg).to(device)
        pre, post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": str(device)}})
        return cls(policy, pre, post, device=device, **kw)

    @classmethod
    def untuned_waiter(cls, base_path, device="cuda", **kw):
        """UNTUNED plumbing model: smolvla_base weights with the Waiter I/O (30-D state, one 224x224 head
        image, 15-D action) and IDENTITY normalisation stats. Its actions are meaningless for the task;
        it exists to test the wrapper and to time the real network."""
        import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
        cfg = PreTrainedConfig.from_pretrained(str(base_path))
        cfg.input_features = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(WAITER_STATE_DIM,)),
            WAITER_IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        }
        cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(WAITER_ACTION_DIM,))}
        cfg.load_vlm_weights = False
        cfg.device = str(device)
        policy = SmolVLAPolicy.from_pretrained(str(base_path), config=cfg).to(device)
        stats = {"observation.state": {"mean": torch.zeros(WAITER_STATE_DIM), "std": torch.ones(WAITER_STATE_DIM)},
                 "action": {"mean": torch.zeros(WAITER_ACTION_DIM), "std": torch.ones(WAITER_ACTION_DIM)}}
        pre, post = make_smolvla_pre_post_processors(cfg, dataset_stats=stats)
        return cls(policy, pre, post, device=device, **kw)

    # ---- interface shared with the stub policy
    @property
    def noise_shape(self) -> tuple[int, int]:
        return (self.chunk_size, self.noise_dim)

    def describe(self) -> dict:
        return {"kind": "smolvla", "chunk_size": self.chunk_size, "noise_shape": list(self.noise_shape),
                "action_dim": self.action_dim, "state_dim": self.state_dim, "num_steps": self.num_steps,
                "n_action_steps": self.n_action_steps, "image_key": self.image_key,
                "feature_mode": self.feature_mode, "feature_dim": self.feature_dim}

    def _batch(self, obs_list):
        img = np.stack([np.asarray(o["image"], np.uint8) for o in obs_list])
        state = np.stack([np.asarray(o["state"], np.float32) for o in obs_list])
        if img.shape[1:] != (224, 224, 3) or state.shape[1] != self.state_dim:
            raise ValueError(f"bad obs shapes {img.shape} {state.shape}")
        batch = {self.image_key: torch.from_numpy(img).permute(0, 3, 1, 2).float().div_(255.0),
                 "observation.state": torch.from_numpy(state),
                 "task": [str(o["instruction"]) for o in obs_list]}
        return self.policy._prepare_batch(self.pre(batch))

    @torch.no_grad()
    def encode(self, obs_list) -> SmolVLAContext:
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        batch = self._batch(obs_list)
        p, m = self.policy, self.policy.model
        images, img_masks = p.prepare_images(batch)
        state = p.prepare_state(batch)
        lang_tokens, lang_masks = batch[self._tok_key], batch[self._mask_key]
        embs, pad_masks, att_masks = m.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        att_2d = make_att_2d_masks(pad_masks, att_masks)
        pos = torch.cumsum(pad_masks, dim=1) - 1
        outs, past_kv = m.vlm_with_expert.forward(
            attention_mask=att_2d, position_ids=pos, past_key_values=None,
            inputs_embeds=[embs, None], use_cache=m.config.use_cache, fill_kv_cache=True)
        feats = self._pool(outs[0], pad_masks, att_masks, lang_tokens.shape[1],
                           batch[self._state_key][:, : self.state_dim].float())
        return SmolVLAContext(pad_masks, past_kv, feats, len(obs_list))

    def _pool(self, out, pad_masks, att_masks, n_lang, state_norm):
        if self.feature_mode == "state":
            return state_norm.float()
        out = out.float()
        valid = pad_masks.float()
        if self.feature_mode == "mean":
            pooled = (out * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1.0)
            return torch.cat([pooled, state_norm], 1)
        # embed_prefix marks exactly the state token(s) with att_mask 1 (images/language are 0)
        state_idx = int(torch.nonzero(att_masks[0]).max())
        lang0 = state_idx - n_lang

        def mmean(a, b):
            w = valid[:, a:b, None]
            return (out[:, a:b] * w).sum(1) / w.sum(1).clamp_min(1.0)

        return torch.cat([mmean(0, lang0), mmean(lang0, state_idx), out[:, state_idx], state_norm], 1)

    @torch.no_grad()
    def denoise(self, ctx: SmolVLAContext, noise: torch.Tensor) -> torch.Tensor:
        """Euler integration of the frozen flow from ``noise`` (B, chunk, 32); returns the UNNORMALISED
        action chunk (B, chunk, action_dim) as a float32 CPU tensor. Mirrors ``sample_actions``."""
        m = self.policy.model
        x_t = torch.as_tensor(noise, dtype=torch.float32, device=self.device)
        if tuple(x_t.shape) != (ctx.bsize, *self.noise_shape):
            raise ValueError(f"noise shape {tuple(x_t.shape)} != {(ctx.bsize, *self.noise_shape)}")
        dt = -1.0 / self.num_steps
        for step in range(self.num_steps):
            t = torch.tensor(1.0 + step * dt, dtype=torch.float32, device=self.device).expand(ctx.bsize)
            v_t = m.denoise_step(x_t=x_t, prefix_pad_masks=ctx.prefix_pad_masks,
                                 past_key_values=ctx.past_key_values, timestep=t)
            x_t = x_t + dt * v_t
        actions = x_t[:, :, : self.action_dim]
        if self.config.adapt_to_pi_aloha:
            actions = self.policy._pi_aloha_encode_actions(actions)
        return self.post(actions).float().cpu()

    def __call__(self, obs_list, noise):
        """policy_with_noise: one obs (dict) or a list of obs -> chunk (B, chunk, action_dim)."""
        if isinstance(obs_list, dict):
            obs_list = [obs_list]
        return self.denoise(self.encode(obs_list), noise)


def policy_with_noise(policy, obs, noise):
    return policy(obs, noise)


# ------------------------------------------------------------------------------------------ self-check
def _sync(dev):
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize()


def run_check(policy, device, out_path=None, batch_sizes=(1, 4, 8, 16, 32), reps=5):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from stub_env import StubWaiterEnv  # toy observations: a real 224x224 image + 30-D state + instruction

    env = StubWaiterEnv()
    obs = [env.reset(1000 + i) for i in range(4)]   # dev-range ids, TOY observations
    T, D = policy.noise_shape
    g = torch.Generator().manual_seed(0)
    n1 = torch.randn(4, T, D, generator=g)
    n2 = torch.randn(4, T, D, generator=g)
    res = {"policy": policy.describe(), "device": str(device),
           "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    c1a = policy(obs, n1)
    c1b = policy(obs, n1)               # fresh prefix pass, same noise
    ctx = policy.encode(obs)
    c1c = policy.denoise(ctx, n1)       # reused context
    c1d = policy.denoise(ctx, n1)
    c2 = policy.denoise(ctx, n2)
    res["same_noise_max_abs_diff_fresh_encode"] = float((c1a - c1b).abs().max())
    res["same_noise_max_abs_diff_reused_ctx"] = float((c1c - c1d).abs().max())
    res["diff_noise_max_abs_diff"] = float((c1a - c2).abs().max())
    res["diff_noise_mean_abs_diff"] = float((c1a - c2).abs().mean())
    res["chunk_shape"] = list(c1a.shape)
    res["chunk_abs_mean"] = float(c1a.abs().mean())
    # Reference: lerobot's own sampling path with the same noise must give the same chunk.
    try:
        batch = policy._batch(obs)
        ref = policy.post(policy.policy.predict_action_chunk(batch, noise=n1.to(device))).float().cpu()
        res["vs_lerobot_predict_action_chunk_max_abs_diff"] = float((ref - c1a).abs().max())
    except Exception as exc:  # recorded, not hidden
        res["vs_lerobot_predict_action_chunk_error"] = repr(exc)
    # Noise sensitivity in the repeat latent (the DSRL action space): spread of chunks over latents.
    nm = NoiseMap(T, D, "repeat")
    lat = torch.randn(16, nm.latent_dim, generator=g)
    ctx16 = policy.encode([obs[0]] * 16)
    ch = policy.denoise(ctx16, nm(lat))
    res["repeat_latent_chunk_std_over_16_latents"] = float(ch.std(0).mean())
    res["feature_dim"] = int(ctx.features.shape[1])
    res["feature_abs_mean"] = float(ctx.features.abs().mean())
    timing = {}
    for B in batch_sizes:
        ob = [obs[i % 4] for i in range(B)]
        nz = torch.randn(B, T, D, generator=g)
        policy.denoise(policy.encode(ob), nz)  # warm-up
        _sync(device)
        te, td = [], []
        for _ in range(reps):
            t0 = time.perf_counter(); c = policy.encode(ob); _sync(device)
            t1 = time.perf_counter(); policy.denoise(c, nz); _sync(device)
            t2 = time.perf_counter(); te.append(t1 - t0); td.append(t2 - t1)
        timing[str(B)] = {"encode_ms": 1e3 * float(np.median(te)), "denoise_ms": 1e3 * float(np.median(td))}
    res["timing_median_ms"] = timing
    if torch.cuda.is_available():
        res["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
    tol = 1e-3
    res["PASS_same_noise_same_chunk"] = (res["same_noise_max_abs_diff_fresh_encode"] < tol
                                         and res["same_noise_max_abs_diff_reused_ctx"] < tol)
    res["PASS_diff_noise_diff_chunk"] = res["diff_noise_max_abs_diff"] > 100 * max(
        res["same_noise_max_abs_diff_fresh_encode"], 1e-6)
    print(json.dumps(res, indent=2))
    if out_path:
        Path(out_path).write_text(json.dumps(res, indent=2) + "\n")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--policy-path", help="fine-tuned SmolVLA pretrained_model directory")
    g.add_argument("--untuned-from", help="smolvla_base directory -> UNTUNED Waiter-I/O plumbing model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--feature-mode", default="pooled")
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.manual_seed(0)
    if a.policy_path:
        pol = SmolVLANoisePolicy.from_pretrained(a.policy_path, device=a.device, feature_mode=a.feature_mode)
    else:
        pol = SmolVLANoisePolicy.untuned_waiter(a.untuned_from, device=a.device, feature_mode=a.feature_mode)
    if a.check:
        r = run_check(pol, a.device, a.out)
        return 0 if (r["PASS_same_noise_same_chunk"] and r["PASS_diff_noise_diff_chunk"]) else 1
    print(json.dumps(pol.describe(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
