"""SAC in the latent-noise space (DSRL-SAC; Wagenmaker et al., CoRL 2025, arXiv 2506.15799).

The SAC action is a latent ``u in [-1, 1]^d`` (tanh-squashed Gaussian). The flow noise is
``w = action_magnitude * u`` (the paper's box ``W = [-b_W, b_W]^d``; b_W in [1, 3] for online RL), then
``NoiseMap`` expands ``w`` to the (chunk, 32) initial noise. Entropy and the target entropy are measured
in the normalised ``u`` space (as in the official SB3-based code, which rescales to [-1, 1]).

One SAC transition = one action chunk: (obs at the decision, latent, reward summed over the executed
steps, per-transition discount ``gamma ** n_executed``, terminal flag, obs at the next decision).

Observation encoders:
  ``vec``: a feature vector (default: SmolVLA's own prefix-pass features, no gradient into the VLA,
           stored in the replay buffer as float16).
  ``cnn``: a small conv encoder on a 64x64 image + a vector (state, instruction hash), trained by the
           critic / actor losses (as the official pi0 DSRL code does), with random-shift augmentation.
Pure torch + numpy; no lerobot.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


@dataclass
class SACConfig:
    vec_dim: int
    act_dim: int
    img_hw: int = 0                 # >0 enables the CNN encoder on (img_hw, img_hw, 3) uint8 images
    action_magnitude: float = 1.5   # b_W: noise = b_W * u, u in [-1, 1]^d
    hidden: int = 1024
    layers: int = 3
    layer_norm: bool = True
    activation: str = "tanh"        # official DSRL (SB3) uses Tanh + post-linear LayerNorm
    n_critics: int = 2
    critic_reduction: str = "min"   # "min" or "mean" over the critic ensemble in the backup
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 1.0
    target_entropy: float | None = 0.0   # None -> -act_dim ("auto")
    tau: float = 0.005
    batch_size: int = 256
    actor_delay: int = 1
    backup_entropy: bool = True
    cnn_latent: int = 64
    grad_clip: float = 0.0
    extra: dict = field(default_factory=dict)


def _act(name):
    return {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}[name]


def mlp(in_dim, out_dim, hidden, layers, layer_norm=True, activation="tanh"):
    mods, d = [], in_dim
    for _ in range(layers):
        mods.append(nn.Linear(d, hidden))
        if layer_norm:
            mods.append(nn.LayerNorm(hidden))
        mods.append(_act(activation)())
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class ConvEncoder(nn.Module):
    """Small DrQ / pi0-DSRL-style encoder: 4 conv layers (32 ch, strides 2,1,1,1) -> linear -> LN -> tanh."""

    def __init__(self, hw: int, latent: int):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2), nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1), nn.ReLU())
        with torch.no_grad():
            n = self.convs(torch.zeros(1, 3, hw, hw)).numel()
        self.head = nn.Sequential(nn.Linear(n, latent), nn.LayerNorm(latent), nn.Tanh())

    def forward(self, img_u8):          # (B, H, W, 3) uint8
        x = img_u8.permute(0, 3, 1, 2).float() / 255.0 - 0.5
        return self.head(self.convs(x).flatten(1))


class ObsEncoder(nn.Module):
    """vec (LayerNorm) [+ conv features]."""

    def __init__(self, cfg: SACConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.vec_dim) if cfg.vec_dim > 0 else None
        self.conv = ConvEncoder(cfg.img_hw, cfg.cnn_latent) if cfg.img_hw > 0 else None
        self.out_dim = cfg.vec_dim + (cfg.cnn_latent if self.conv is not None else 0)

    def forward(self, obs):
        parts = []
        if self.norm is not None:
            parts.append(self.norm(obs["vec"].float()))
        if self.conv is not None:
            parts.append(self.conv(obs["img"]))
        return torch.cat(parts, -1)


class Actor(nn.Module):
    def __init__(self, cfg: SACConfig):
        super().__init__()
        self.enc = ObsEncoder(cfg)
        self.net = mlp(self.enc.out_dim, 2 * cfg.act_dim, cfg.hidden, cfg.layers, cfg.layer_norm, cfg.activation)

    def forward(self, obs, deterministic=False, with_logp=True):
        mu, log_std = self.net(self.enc(obs)).chunk(2, -1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        if deterministic:
            return torch.tanh(mu), None
        std = log_std.exp()
        x = mu + std * torch.randn_like(mu)
        u = torch.tanh(x)
        if not with_logp:
            return u, None
        logp = (-0.5 * ((x - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - (2 * (np.log(2.0) - x - F.softplus(-2 * x))).sum(-1)   # tanh correction (stable)
        return u, logp


class CriticEnsemble(nn.Module):
    def __init__(self, cfg: SACConfig):
        super().__init__()
        self.enc = ObsEncoder(cfg)
        self.qs = nn.ModuleList(
            mlp(self.enc.out_dim + cfg.act_dim, 1, cfg.hidden, cfg.layers, cfg.layer_norm, cfg.activation)
            for _ in range(cfg.n_critics))

    def forward(self, obs, u, detach_encoder=False):
        h = self.enc(obs)
        if detach_encoder:
            h = h.detach()
        x = torch.cat([h, u], -1)
        return torch.stack([q(x).squeeze(-1) for q in self.qs], 0)   # (n_critics, B)


def random_shift(img_u8, pad=4):
    """DrQ random shift on (B, H, W, 3) uint8 -> same shape uint8."""
    B, H, W, _ = img_u8.shape
    dev = img_u8.device
    x = F.pad(img_u8.permute(0, 3, 1, 2).float(), (pad, pad, pad, pad), mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (B,), device=dev)
    oy = torch.randint(0, 2 * pad + 1, (B,), device=dev)
    rows = oy[:, None] + torch.arange(H, device=dev)[None]          # (B, H)
    cols = ox[:, None] + torch.arange(W, device=dev)[None]          # (B, W)
    x = x[torch.arange(B, device=dev)[:, None, None, None], torch.arange(3, device=dev)[None, :, None, None],
          rows[:, None, :, None], cols[:, None, None, :]]
    return x.permute(0, 2, 3, 1).to(torch.uint8)


class ReplayBuffer:
    """Ring buffer of chunk-level transitions. vec obs stored float16, images uint8."""

    def __init__(self, capacity: int, vec_dim: int, act_dim: int, img_hw: int = 0, seed: int = 0):
        self.capacity, self.size, self.ptr = int(capacity), 0, 0
        self.vec_dim, self.act_dim, self.img_hw = vec_dim, act_dim, img_hw
        self.rng = np.random.default_rng(seed)
        self.data = {"vec": np.zeros((capacity, vec_dim), np.float16),
                     "next_vec": np.zeros((capacity, vec_dim), np.float16),
                     "act": np.zeros((capacity, act_dim), np.float32),
                     "rew": np.zeros(capacity, np.float32), "disc": np.zeros(capacity, np.float32),
                     "term": np.zeros(capacity, np.float32)}
        if img_hw:
            self.data["img"] = np.zeros((capacity, img_hw, img_hw, 3), np.uint8)
            self.data["next_img"] = np.zeros((capacity, img_hw, img_hw, 3), np.uint8)

    def add(self, obs, act, rew, disc, term, next_obs):
        i = self.ptr
        self.data["vec"][i] = obs["vec"]
        self.data["next_vec"][i] = next_obs["vec"]
        if self.img_hw:
            self.data["img"][i] = obs["img"]
            self.data["next_img"][i] = next_obs["img"]
        self.data["act"][i], self.data["rew"][i] = act, rew
        self.data["disc"][i], self.data["term"][i] = disc, float(term)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n, device):
        idx = self.rng.integers(0, self.size, n)
        t = {k: torch.as_tensor(v[idx], device=device) for k, v in self.data.items()}
        obs = {"vec": t["vec"].float()}
        nobs = {"vec": t["next_vec"].float()}
        if self.img_hw:
            obs["img"], nobs["img"] = t["img"], t["next_img"]
        return obs, t["act"], t["rew"], t["disc"], t["term"], nobs

    def save(self, path):
        np.savez(path, size=self.size, ptr=self.ptr,
                 **{k: v[: self.size] if self.size < self.capacity else v for k, v in self.data.items()})

    def load(self, path):
        with np.load(path) as z:
            self.size, self.ptr = int(z["size"]), int(z["ptr"])
            for k in self.data:
                self.data[k][: len(z[k])] = z[k]


class NoiseSAC:
    def __init__(self, cfg: SACConfig, device="cpu", seed: int = 0):
        self.cfg, self.device = cfg, torch.device(device)
        torch.manual_seed(seed)
        self.actor = Actor(cfg).to(self.device)
        self.critic = CriticEnsemble(cfg).to(self.device)
        self.critic_targ = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = torch.tensor(float(np.log(cfg.init_alpha)), device=self.device, requires_grad=True)
        self.target_entropy = float(-cfg.act_dim if cfg.target_entropy is None else cfg.target_entropy)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def _to_obs(self, obs):
        out = {"vec": torch.as_tensor(np.asarray(obs["vec"], np.float32), device=self.device)}
        if self.cfg.img_hw:
            out["img"] = torch.as_tensor(np.asarray(obs["img"], np.uint8), device=self.device)
        return out

    @torch.no_grad()
    def act(self, obs, deterministic=False) -> np.ndarray:
        """obs: {"vec": (B, vec_dim)[, "img": (B, hw, hw, 3)]} -> normalised latent u (B, act_dim) in [-1, 1]."""
        u, _ = self.actor(self._to_obs(obs), deterministic=deterministic, with_logp=False)
        return u.cpu().numpy()

    def _reduce(self, qs):
        return qs.min(0).values if self.cfg.critic_reduction == "min" else qs.mean(0)

    def update(self, buffer: ReplayBuffer) -> dict:
        cfg = self.cfg
        obs, act, rew, disc, term, nobs = buffer.sample(cfg.batch_size, self.device)
        if cfg.img_hw:
            obs["img"], nobs["img"] = random_shift(obs["img"]), random_shift(nobs["img"])
        with torch.no_grad():
            nu, nlogp = self.actor(nobs)
            nq = self._reduce(self.critic_targ(nobs, nu))
            if cfg.backup_entropy:
                nq = nq - self.alpha * nlogp
            target = rew + disc * (1.0 - term) * nq
        qs = self.critic(obs, act)
        critic_loss = ((qs - target[None]) ** 2).mean(1).sum()
        self.opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
        self.opt_critic.step()
        info = {"critic_loss": float(critic_loss), "q_mean": float(qs.mean()), "target_mean": float(target.mean())}
        self.updates += 1
        if self.updates % cfg.actor_delay == 0:
            u, logp = self.actor(obs)
            q = self._reduce(self.critic(obs, u, detach_encoder=True))
            actor_loss = (self.alpha * logp - q).mean()
            self.opt_actor.zero_grad(set_to_none=True)
            actor_loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
            self.opt_actor.step()
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.opt_alpha.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.opt_alpha.step()
            info.update(actor_loss=float(actor_loss), entropy=float(-logp.mean()), alpha=float(self.alpha))
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_targ.parameters()):
                pt.lerp_(p, cfg.tau)
        return info

    # ---- checkpointing
    def state_dict(self):
        return {"cfg": asdict(self.cfg), "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "critic_targ": self.critic_targ.state_dict(), "log_alpha": self.log_alpha.detach().cpu(),
                "opt_actor": self.opt_actor.state_dict(), "opt_critic": self.opt_critic.state_dict(),
                "opt_alpha": self.opt_alpha.state_dict(), "updates": self.updates}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_targ.load_state_dict(sd["critic_targ"])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"].to(self.device))
        self.opt_actor.load_state_dict(sd["opt_actor"])
        self.opt_critic.load_state_dict(sd["opt_critic"])
        self.opt_alpha.load_state_dict(sd["opt_alpha"])
        self.updates = int(sd["updates"])

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, device="cpu"):
        sd = torch.load(path, map_location=device, weights_only=False)
        agent = cls(SACConfig(**sd["cfg"]), device=device)
        agent.load_state_dict(sd)
        return agent


def config_json(cfg: SACConfig) -> str:
    return json.dumps(asdict(cfg), indent=2)


__all__ = ["SACConfig", "NoiseSAC", "ReplayBuffer", "random_shift", "config_json", "Path"]
