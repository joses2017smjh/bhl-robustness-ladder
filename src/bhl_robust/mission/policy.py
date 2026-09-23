"""Validity-aware modality encoders compatible with the installed RSL-RL PPO."""
import torch
from torch import nn
from rsl_rl.modules import ActorCritic
from bhl_robust.mission.sensors import PROPRIO, LIDAR, DEPTH, FRAME, HISTORY


class FusionNet(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.lidar = nn.Sequential(nn.Linear(2*LIDAR+1, 32), nn.ELU(), nn.Linear(32, 32), nn.ELU())
        self.stereo = nn.Sequential(nn.Linear(2*DEPTH+1, 32), nn.ELU(), nn.Linear(32, 32), nn.ELU())
        self.gate = nn.Sequential(nn.Linear(68, 32), nn.ELU(), nn.Linear(32, 2))
        self.head = nn.Sequential(nn.Linear(HISTORY*(PROPRIO+66), 128), nn.ELU(),
                                  nn.Linear(128, 64), nn.ELU(), nn.Linear(64, outputs))

    def encode(self, obs):
        frames = obs.reshape(-1, HISTORY, FRAME)
        p = frames[..., :PROPRIO]
        l = frames[..., PROPRIO:PROPRIO+2*LIDAR+1]
        s = frames[..., PROPRIO+2*LIDAR+1:]
        # Mask again at the network boundary: invalid values cannot influence gates.
        lm, sm = l[..., LIDAR:2*LIDAR], s[..., DEPTH:2*DEPTH]
        l = torch.cat([l[..., :LIDAR]*lm, lm, l[..., -1:]], -1)
        s = torch.cat([s[..., :DEPTH]*sm, sm, s[..., -1:]], -1)
        le, se = self.lidar(l), self.stereo(s)
        reliability = torch.stack([lm.mean(-1), sm.mean(-1)], -1)
        ages = torch.stack([l[..., -1], s[..., -1]], -1)
        weights = self.gate(torch.cat([le, se, reliability, ages], -1)).softmax(-1)*reliability
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        fused = torch.cat([p, le*weights[..., :1], se*weights[..., 1:], weights], -1)
        return fused.flatten(1), weights

    def forward(self, obs):
        return self.head(self.encode(obs)[0])


class MissionPolicy(ActorCritic):
    def __init__(self, obs):
        super().__init__(obs, {"policy": ["policy"], "critic": ["policy"]}, 5,
                         actor_hidden_dims=[16], critic_hidden_dims=[16],
                         noise_std_type="log", init_noise_std=.5)
        self.actor = FusionNet(5)
        self.critic = FusionNet(1)
