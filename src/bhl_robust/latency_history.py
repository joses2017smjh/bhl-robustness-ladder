"""Causal, episode-isolated IMU packets for the H3 information ablation.

Only the six gyro/projected-gravity columns are delayed. Commands, joint
states and previous actions retain their current values. Both arms expose
four packets to an identical actor: history contains past delivered packets;
feedforward repeats the current delivered packet. Timestamps here count
policy ticks, never wall-clock or device latency.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path

import torch

BASE_ACTOR_WIDTH = 45
ACTOR_WIDTH = 63
CRITIC_WIDTH = 48
IMU_WIDTH = 6
HISTORY_PACKETS = 4
POLICY_DT = 0.04


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CausalImuHistory:
    """One update per environment/tick, with selective reset and no future data.

    At reset, unavailable samples repeat the first captured packet. This
    startup padding is shared by both arms. Source ticks expose its effective
    age rather than claiming unavailable pre-reset samples were captured.
    """

    def __init__(self, num_envs: int, device="cpu", *, mode="history",
                 max_delay_steps=1, generator=None, noise=True):
        if mode not in ("history", "feedforward"):
            raise ValueError("mode must be history or feedforward")
        if isinstance(max_delay_steps, bool) or int(max_delay_steps) != max_delay_steps or max_delay_steps < 0:
            raise ValueError("max_delay_steps must be a non-negative integer")
        if int(num_envs) < 1:
            raise ValueError("num_envs must be positive")
        self.n = int(num_envs)
        self.device = torch.device(device)
        self.mode = mode
        self.max_delay_steps = int(max_delay_steps)
        self.generator = generator
        self.noise = bool(noise)
        self.delay = torch.zeros(self.n, dtype=torch.long, device=device)
        self.raw = torch.zeros(self.n, self.max_delay_steps + 1, IMU_WIDTH, device=device)
        self.raw_ticks = torch.full(self.raw.shape[:2], -1, dtype=torch.long, device=device)
        self.delivered = torch.zeros(self.n, HISTORY_PACKETS, IMU_WIDTH, device=device)
        self.delivered_ticks = torch.full(self.delivered.shape[:2], -1, dtype=torch.long, device=device)
        self.last_tick = torch.full((self.n,), -1, dtype=torch.long, device=device)
        self.initialized = torch.zeros(self.n, dtype=torch.bool, device=device)
        self.force_delay = None

    def reset(self, env_ids=None):
        ids = torch.arange(self.n, device=self.device) if env_ids is None else torch.as_tensor(
            env_ids, dtype=torch.long, device=self.device).reshape(-1)
        if ids.numel() == 0:
            return
        if (ids < 0).any() or (ids >= self.n).any():
            raise ValueError("reset environment id out of range")
        self.delay[ids] = torch.randint(self.max_delay_steps + 1, (ids.numel(),),
                                       generator=self.generator, device=self.device)
        if self.force_delay is not None:
            delay = int(self.force_delay)
            if delay != self.force_delay or not 0 <= delay <= self.max_delay_steps:
                raise ValueError("forced delay outside the allocated queue")
            self.delay[ids] = delay
        self.raw[ids] = 0
        self.raw_ticks[ids] = -1
        self.delivered[ids] = 0
        self.delivered_ticks[ids] = -1
        self.last_tick[ids] = -1
        self.initialized[ids] = False

    def update(self, clean_packet: torch.Tensor, tick: int) -> torch.Tensor:
        if clean_packet.shape != (self.n, IMU_WIDTH):
            raise ValueError(f"IMU shape {tuple(clean_packet.shape)} != {(self.n, IMU_WIDTH)}")
        if clean_packet.device != self.device or not torch.isfinite(clean_packet).all():
            raise ValueError("IMU packet must be finite and on the queue device")
        if isinstance(tick, bool) or int(tick) != tick or tick < 0:
            raise ValueError("tick must be a non-negative integer")
        tick = int(tick)
        if ((self.last_tick > tick) & self.initialized).any():
            raise ValueError("time moved backwards without an episode reset")
        if ((self.last_tick < tick - 1) & self.initialized).any():
            raise ValueError("policy tick skipped: queue cannot invent a missing captured packet")
        advancing = self.last_tick != tick
        if advancing.any():
            packet = clean_packet.to(dtype=self.raw.dtype)
            if self.noise:
                scales = packet.new_tensor([.3, .3, .3, .05, .05, .05])
                packet = packet + (2 * torch.rand(packet.shape, device=self.device,
                                                 generator=self.generator) - 1) * scales
            ids = advancing.nonzero(as_tuple=False).flatten()
            fresh = ids[~self.initialized[ids]]
            old = ids[self.initialized[ids]]
            if old.numel():
                self.raw[old, 1:] = self.raw[old, :-1].clone()
                self.raw_ticks[old, 1:] = self.raw_ticks[old, :-1].clone()
                self.delivered[old, :-1] = self.delivered[old, 1:].clone()
                self.delivered_ticks[old, :-1] = self.delivered_ticks[old, 1:].clone()
            if fresh.numel():
                self.raw[fresh] = packet[fresh, None, :]
                self.raw_ticks[fresh] = tick
            self.raw[ids, 0] = packet[ids]
            self.raw_ticks[ids, 0] = tick
            selected = self.raw[ids, self.delay[ids]]
            selected_ticks = self.raw_ticks[ids, self.delay[ids]]
            if fresh.numel():
                self.delivered[fresh] = self.raw[fresh, self.delay[fresh]][:, None, :]
                self.delivered_ticks[fresh] = self.raw_ticks[fresh, self.delay[fresh]][:, None]
            self.delivered[ids, -1] = selected
            self.delivered_ticks[ids, -1] = selected_ticks
            self.last_tick[ids] = tick
            self.initialized[ids] = True
        if self.mode == "feedforward":
            return self.delivered[:, -1:, :].expand(-1, HISTORY_PACKETS, -1).clone()
        return self.delivered.clone()

    def source_ticks(self):
        if self.mode == "feedforward":
            return self.delivered_ticks[:, -1:].expand(-1, HISTORY_PACKETS).clone()
        return self.delivered_ticks.clone()


def expanded_teacher_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copy every teacher tensor and add zero weights for the eighteen past slots."""
    if "actor.0.weight" not in state or tuple(state["actor.0.weight"].shape) != (256, BASE_ACTOR_WIDTH):
        raise ValueError("teacher must have a 256 by 45 first actor layer")
    if "critic.0.weight" not in state or tuple(state["critic.0.weight"].shape) != (256, CRITIC_WIDTH):
        raise ValueError("teacher must have the clean 48-column critic")
    result = {key: value.detach().clone() for key, value in state.items()}
    first = state["actor.0.weight"]
    result["actor.0.weight"] = torch.cat([first, first.new_zeros(256, ACTOR_WIDTH - BASE_ACTOR_WIDTH)], dim=1)
    return result


def actor_from_state(state: Mapping[str, torch.Tensor], *, width=ACTOR_WIDTH):
    """The frozen ELU [256,128,128] deterministic actor, without Isaac imports."""
    actor = torch.nn.Sequential(torch.nn.Linear(width, 256), torch.nn.ELU(),
                                torch.nn.Linear(256, 128), torch.nn.ELU(),
                                torch.nn.Linear(128, 128), torch.nn.ELU(),
                                torch.nn.Linear(128, 12))
    actor.load_state_dict({key.removeprefix("actor."): value for key, value in state.items()
                           if key.startswith("actor.")}, strict=True)
    return actor.eval()
