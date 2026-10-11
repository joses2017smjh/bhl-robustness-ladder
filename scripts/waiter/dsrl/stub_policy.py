"""STUB frozen base policy with the SmolVLANoisePolicy interface (TOY; pairs with stub_env.StubWaiterEnv).

Same interface as ``smolvla_noise.SmolVLANoisePolicy``: ``noise_shape == (50, 32)``, ``encode(obs_list)
-> ctx`` (``ctx.features`` = SAC observation features), ``denoise(ctx, noise) -> (B, 50, 15)``, and
``policy(obs, noise)``. Deterministic given the noise.

Behaviour (TOY): from the image it estimates both marker positions (colour centroids), aims the hand at
their MIDPOINT (it ignores the instruction), and adds an offset that is a fixed nonlinear function of
the first noise row: ``offset = 0.5 * tanh(W2 tanh(W1 n0))`` (frozen random weights, std ~0.25 m per
axis under n0 ~ N(0, I)). So under Gaussian noise the base sometimes lands on the instructed marker;
a specific, observation-dependent noise direction makes it succeed every time. Other action entries
depend weakly on the other noise rows (so different noise -> different chunks) but do not matter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from stub_env import IMAGE_HW, STEP_DT  # noqa: F401

CHUNK, NOISE_DIM, ACTION_DIM, STATE_DIM = 50, 32, 15, 30


def _px_to_xy(row, col):
    x = col / (IMAGE_HW - 1) * 2.0 - 1.0
    y = 1.0 - row / (IMAGE_HW - 1) * 2.0
    return x, y


def _centroid(mask):
    rr, cc = np.nonzero(mask)
    if rr.size == 0:
        return 0.0, 0.0
    return _px_to_xy(rr.mean(), cc.mean())


@dataclass
class StubContext:
    features: torch.Tensor
    mid: torch.Tensor
    bsize: int


class StubFlowPolicy:
    toy = True

    def __init__(self, device="cpu", n_action_steps: int = 10, weight_seed: int = 12345,
                 offset_scale: float = 0.5, pre_std: float = 0.6):
        self.device = torch.device(device)
        self.chunk_size, self.noise_dim, self.action_dim = CHUNK, NOISE_DIM, ACTION_DIM
        self.state_dim, self.n_action_steps, self.num_steps = STATE_DIM, int(n_action_steps), 1
        g = torch.Generator().manual_seed(weight_seed)
        self.W1 = torch.randn(16, NOISE_DIM, generator=g) / NOISE_DIM ** 0.5
        W2 = torch.randn(2, 16, generator=g)
        probe = torch.tanh(torch.randn(4096, NOISE_DIM, generator=g) @ self.W1.T) @ W2.T
        self.W2 = W2 * (pre_std / probe.std(0, keepdim=True).T)    # per-output pre-activation std
        self.W1, self.W2 = self.W1.to(self.device), self.W2.to(self.device)
        self.offset_scale = float(offset_scale)
        self.feature_dim = 8 + 192

    @property
    def noise_shape(self):
        return (self.chunk_size, self.noise_dim)

    def describe(self):
        return {"kind": "stub", "toy": True, "chunk_size": self.chunk_size, "noise_shape": list(self.noise_shape),
                "action_dim": self.action_dim, "n_action_steps": self.n_action_steps,
                "feature_dim": self.feature_dim}

    def offset(self, n0: torch.Tensor) -> torch.Tensor:
        return self.offset_scale * torch.tanh(torch.tanh(n0 @ self.W1.T) @ self.W2.T)

    def _features(self, o):
        img = np.asarray(o["image"], np.uint8)
        r, g, b = (img[..., i].astype(np.int16) for i in range(3))
        red = _centroid((r > 150) & (g < 100) & (b < 100))
        green = _centroid((g > 130) & (r < 100) & (b < 100))
        hand = np.asarray(o["state"], np.float32)[3:5]
        text = str(o["instruction"])
        onehot = [float("red" in text), float("green" in text)]
        coarse = img.reshape(8, IMAGE_HW // 8, 8, IMAGE_HW // 8, 3).mean((1, 3)).reshape(-1) / 255.0
        f = np.concatenate([red, green, hand, onehot, coarse]).astype(np.float32)
        return f, 0.5 * (np.asarray(red) + np.asarray(green))

    @torch.no_grad()
    def encode(self, obs_list) -> StubContext:
        if isinstance(obs_list, dict):
            obs_list = [obs_list]
        fs, mids = zip(*(self._features(o) for o in obs_list))
        return StubContext(torch.as_tensor(np.stack(fs), device=self.device),
                           torch.as_tensor(np.stack(mids), dtype=torch.float32, device=self.device),
                           len(obs_list))

    @torch.no_grad()
    def denoise(self, ctx: StubContext, noise) -> torch.Tensor:
        n = torch.as_tensor(noise, dtype=torch.float32, device=self.device)
        if tuple(n.shape) != (ctx.bsize, CHUNK, NOISE_DIM):
            raise ValueError(f"noise shape {tuple(n.shape)}")
        a = torch.zeros(ctx.bsize, CHUNK, ACTION_DIM, device=self.device)
        a[:, :, 0:3] = 0.02 * n[:, :, 12:15]
        a[:, :, 3:5] = (ctx.mid + self.offset(n[:, 0, :]))[:, None, :]
        a[:, :, 5:13] = 0.05 * n[:, :, 2:10]
        a[:, :, 13:15] = 0.15 + 0.05 * n[:, :, 10:12]
        return a.cpu()

    def __call__(self, obs_list, noise):
        return self.denoise(self.encode(obs_list), noise)
