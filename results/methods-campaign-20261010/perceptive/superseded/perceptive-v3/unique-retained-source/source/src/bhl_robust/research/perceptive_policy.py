"""Trainable sparse-terrain students with explicit missing data and causal memory.

These are project ablations inspired by perceptive teacher/student locomotion,
query-based reconstruction and gated memory; they are not paper reproductions.
The student API accepts proprioception and measured local points only. Teacher
terrain and true velocity have separate training-only arguments.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

PROPRIO = 45
JOINTS = 12
RAYS = 128
HISTORY = 4
GRID_X = 11
GRID_Y = 7
CELLS = GRID_X * GRID_Y
TEACHER_WIDTH = PROPRIO + CELLS
CRITIC_WIDTH = TEACHER_WIDTH + 3
ARMS = ("dense", "query", "query_memory", "query_memory_trajectory")


def terrain_queries(device="cpu"):
    x, y = torch.meshgrid(torch.linspace(-.5, .5, GRID_X, device=device),
                          torch.linspace(-.3, .3, GRID_Y, device=device), indexing="ij")
    return torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)


def privileged_height(base_origin_z, hit_z, *, nominal_base_height=.30):
    """Height relative to the BASE, independent of sensor pose conventions.

    Isaac's RayCaster reports the tracked body pose while applying cfg.offset
    to ray origins separately. Accepting explicit base z avoids counting that
    offset twice. Invalid truth is rejected rather than used as a label.
    """
    if not torch.isfinite(base_origin_z).all() or not torch.isfinite(hit_z).all():
        raise ValueError("privileged ground-height query missed terrain")
    return (base_origin_z[..., None] - hit_z - nominal_base_height).clamp(-1., 1.)


class CausalPointHistory:
    """Episode-isolated sensor packets; unavailable past packets stay unknown."""
    def __init__(self, count, *, rays=RAYS, history=HISTORY, device="cpu", delay_steps=0):
        if count < 1 or rays < 1 or history < 1 or not 0 <= delay_steps <= 8:
            raise ValueError("invalid point-history dimensions/delay")
        self.count, self.rays, self.history = count, rays, history
        self.delay_steps, self.device = delay_steps, torch.device(device)
        self.raw = torch.zeros(count, history + 8, rays, 4, device=device)
        self.stamps = torch.full((count, history + 8), -1, dtype=torch.long, device=device)
        self.last = torch.full((count,), -1, dtype=torch.long, device=device)

    def reset(self, ids=None):
        ids = torch.arange(self.count, device=self.device) if ids is None else ids
        self.raw[ids] = 0
        self.stamps[ids] = -1
        self.last[ids] = -1

    def update(self, points, valid, tick):
        if points.shape != (self.count, self.rays, 3) or valid.shape != points.shape[:-1]:
            raise ValueError("wrong measured point dimensions")
        if tick < 0 or ((self.last > tick) & (self.last >= 0)).any():
            raise ValueError("point sensor time moved backwards")
        if ((self.last >= 0) & (self.last < tick-1)).any():
            raise ValueError("missing sensor tick must be explicitly recorded")
        advance = self.last != tick
        if advance.any():
            good = valid.bool() & torch.isfinite(points).all(-1)
            packet = torch.cat((torch.where(good[..., None], points, 0.), good[..., None].float()), -1)
            self.raw[advance, 1:] = self.raw[advance, :-1].clone()
            self.stamps[advance, 1:] = self.stamps[advance, :-1].clone()
            self.raw[advance, 0] = packet[advance]
            self.stamps[advance, 0] = tick
            self.last[advance] = tick
        start = int(self.delay_steps)
        data = self.raw[:, start:start+self.history].clone()
        stamps = self.stamps[:, start:start+self.history]
        age = (tick-stamps).float() * .04
        age = torch.where(stamps >= 0, age, torch.ones_like(age))
        return torch.cat((data, age[..., None, None].expand(-1, -1, self.rays, 1)), -1)


def trunk(width, output):
    return nn.Sequential(nn.Linear(width, 256), nn.ELU(), nn.Linear(256, 128),
                         nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, output))


class TerrainTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = trunk(TEACHER_WIDTH, JOINTS)
        self.critic = trunk(CRITIC_WIDTH, 1)
        self.log_std = nn.Parameter(torch.full((JOINTS,), math.log(.3)))

    def warm_start(self, state):
        """Copy frozen gait, adding zero terrain weights and a fresh optimizer."""
        for prefix, module, expected in (("actor", self.actor, PROPRIO), ("critic", self.critic, 48)):
            source = {k[len(prefix)+1:]: v for k, v in state.items() if k.startswith(prefix+".")}
            if source["0.weight"].shape != (256, expected):
                raise ValueError("warm-start gait architecture differs")
            first = module[0].weight.detach().new_zeros(module[0].weight.shape)
            if prefix == "actor":
                first[:, :PROPRIO] = source["0.weight"]
            else:
                first[:, :PROPRIO] = source["0.weight"][:, :PROPRIO]
                first[:, -3:] = source["0.weight"][:, -3:]
            source["0.weight"] = first
            module.load_state_dict(source, strict=True)
        if "std" in state:
            with torch.no_grad():
                self.log_std.copy_(state["std"].clamp(.05, 1.).log())

    def forward(self, proprio, teacher_height):
        return self.actor(torch.cat((proprio, teacher_height), -1))

    def value(self, critic):
        return self.critic(critic).squeeze(-1)


class QueryReconstructor(nn.Module):
    def __init__(self, temporal=False):
        super().__init__()
        self.temporal = temporal
        self.register_buffer("locations", terrain_queries())
        self.point_encoder = nn.Sequential(nn.Linear(5, 64), nn.ELU(), nn.Linear(64, 64))
        self.query_encoder = nn.Linear(10 + PROPRIO, 64)
        self.attention = nn.MultiheadAttention(64, 4, batch_first=True)
        self.null_token = nn.Parameter(torch.zeros(1, 1, 64))
        self.memory_gate = nn.Sequential(nn.Linear(5 + 6, 32), nn.ELU(), nn.Linear(32, 1), nn.Sigmoid())
        self.decoder = nn.Sequential(nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, proprio, history):
        packets = history if self.temporal else history[:, :1]
        points = packets.reshape(proprio.shape[0], -1, 5)
        valid = (points[..., 3] > .5) & torch.isfinite(points).all(-1)
        clean = torch.where(valid[..., None], points, 0.)
        tokens = self.point_encoder(clean)
        if self.temporal:
            imu = proprio[:, None, 3:9].expand(-1, points.shape[1], -1)
            tokens = tokens * self.memory_gate(torch.cat((clean, imu), -1))
        # One explicit null token keeps entirely absent point clouds finite.
        tokens = torch.cat((tokens, self.null_token.expand(proprio.shape[0], -1, -1)), 1)
        padding = torch.cat((~valid, torch.zeros(proprio.shape[0], 1, dtype=torch.bool, device=proprio.device)), 1)
        xy = self.locations
        fourier = torch.cat((xy, torch.sin(2*math.pi*xy), torch.cos(2*math.pi*xy),
                             torch.sin(4*math.pi*xy), torch.cos(4*math.pi*xy)), -1)
        queries = self.query_encoder(torch.cat((fourier[None].expand(proprio.shape[0], -1, -1),
                                                proprio[:, None].expand(-1, CELLS, -1)), -1))
        encoded, _ = self.attention(queries, tokens, tokens, key_padding_mask=padding, need_weights=False)
        return self.decoder(encoded + queries).squeeze(-1)


class TerrainStudent(nn.Module):
    def __init__(self, arm="dense"):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("unknown perceptive student arm")
        self.arm = arm
        self.reconstructor = (nn.Sequential(nn.Linear(RAYS*5+PROPRIO, 256), nn.ELU(),
                                            nn.Linear(256, CELLS), nn.Tanh())
                              if arm == "dense" else QueryReconstructor("memory" in arm))
        self.actor = trunk(TEACHER_WIDTH, JOINTS)
        self.critic = trunk(CRITIC_WIDTH, 1)
        self.log_std = nn.Parameter(torch.full((JOINTS,), math.log(.15)))

    def from_teacher(self, teacher):
        self.actor.load_state_dict(teacher.actor.state_dict())
        self.critic.load_state_dict(teacher.critic.state_dict())

    def reconstruct(self, proprio, points):
        if points.shape[1:] != (HISTORY, RAYS, 5):
            raise ValueError("student requires causal measured point history")
        if self.arm == "dense":
            return self.reconstructor(torch.cat((proprio, points[:, 0].reshape(proprio.shape[0], -1)), -1))
        return self.reconstructor(proprio, points)

    def forward(self, proprio, points):
        return self.actor(torch.cat((proprio, self.reconstruct(proprio, points)), -1))

    def value(self, critic):
        return self.critic(critic).squeeze(-1)


def export_student(student, path):
    """Export only the measured-input actor and check independent input batches."""
    import copy
    class InferenceActor(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            del self.model.critic
            del self.model.log_std

        def forward(self, proprio, points):
            return self.model(proprio, points)

    actor = InferenceActor(copy.deepcopy(student).cpu().eval())
    generator = torch.Generator().manual_seed(730901)
    p = torch.randn(2, PROPRIO, generator=generator)
    h = torch.randn(2, HISTORY, RAYS, 5, generator=generator)
    h[..., 3] = 1.
    # Tracing a fixed observation contract; check separate sizes/missing packets
    # explicitly because MHA's native fast path changes the traced graph.
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            traced = torch.jit.trace(actor, (p, h), check_trace=False)
            errors = []
            for n, missing in ((1, True), (7, False)):
                p2 = torch.randn(n, PROPRIO, generator=generator)
                h2 = torch.randn(n, HISTORY, RAYS, 5, generator=generator)
                h2[..., 3] = 0. if missing else 1.
                expected, actual = actor(p2, h2), traced(p2, h2)
                errors.append(float((expected-actual).abs().max()))
                if not torch.isfinite(actual).all() or errors[-1] > 1e-5:
                    raise ValueError("student deployment parity failed")
            torch.jit.save(traced, str(path))
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    return dict(schema="bhl-perceptive-actor-v1", arm=student.arm,
        inputs={"proprio": ["batch", PROPRIO], "points": ["batch", HISTORY, RAYS, 5]},
        output=["batch", JOINTS], maximum_parity_error=max(errors),
        policy_dt_s=.04, points_layout="body-relative gravity-aligned xyz, valid, age_seconds",
        scope="simulation inference export; no physical robot validation; training critic excluded")


def edge_reconstruction_loss(predicted, target, edge_weight=1.):
    """Height error plus adjacent-height differences; labels are training-only."""
    p, t = predicted.reshape(-1, GRID_X, GRID_Y), target.reshape(-1, GRID_X, GRID_Y)
    loss = (p-t).abs().mean()
    for axis in (1, 2):
        loss = loss + edge_weight * (p.diff(dim=axis)-t.diff(dim=axis)).abs().mean()
    return loss


def trajectory_penalty(student_next, teacher_next, terminated, coefficient=.1):
    """Actual next-state disagreement penalty, propagated backwards by GAE.

    Auto-reset next states belong to a different episode and receive no penalty.
    This is an explicit project ablation inspired by trajectory distillation.
    """
    if student_next.shape != teacher_next.shape or terminated.shape != student_next.shape[:-1]:
        raise ValueError("next-state penalty shape mismatch")
    error = (student_next-teacher_next).square().mean(-1).clamp(max=10.)
    return -coefficient * error * (~terminated.bool()).float()


def generalized_advantage(rewards, values, dones, last_value, gamma=.99, lam=.95):
    if rewards.shape != values.shape or dones.shape != values.shape:
        raise ValueError("rollout shapes differ")
    advantages = torch.zeros_like(rewards)
    previous = torch.zeros_like(last_value)
    for i in reversed(range(len(rewards))):
        next_value = last_value if i == len(rewards)-1 else values[i+1]
        live = (~dones[i].bool()).float()
        delta = rewards[i] + gamma * next_value * live - values[i]
        previous = delta + gamma * lam * live * previous
        advantages[i] = previous
    return advantages, advantages + values
