"""Isolated diagnostic instrumentation; production rewards and gates stay intact."""
from dataclasses import asdict
import numpy as np
import mujoco
import torch
from torch import nn
from rsl_rl.modules import ActorCritic
from bhl_robust.mission.env import MissionEnv
from bhl_robust.mission.state import MissionState
from bhl_robust.mission.policy import FusionNet, MissionPolicy
from bhl_robust.mission.sensors import FRAME, HISTORY

TERMS = ('time', 'collision', 'failure', 'correct_button', 'wrong_button',
         'door_crossing', 'acquisition', 'success')


class AuditedState(MissionState):
    def __post_init__(self):
        self.reward_totals = dict.fromkeys(TERMS, 0.)

    def __init__(self, stage):
        super().__init__(stage)
        self.reward_totals = dict.fromkeys(TERMS, 0.)

    def acquire(self):
        reward = super().acquire()
        self.reward_totals['acquisition'] += reward
        return reward

    def update(self, **kwargs):
        prior = (self.last_s, self.collisions, sum(self.open), sum(self.crossed), self.wrong_buttons,
                 self.failure, self.completed_s)
        reward = super().update(**kwargs)
        if self.last_s != prior[0]:
            values = dict(time=-.01*kwargs['dt'], collision=-.2*(self.collisions-prior[1]),
                          correct_button=2.*(sum(self.open)-prior[2]),
                          door_crossing=2.*(sum(self.crossed)-prior[3]),
                          wrong_button=-float(self.wrong_buttons-prior[4]),
                          failure=-5.*int(prior[5] is None and self.failure is not None),
                          success=10.*int(prior[6] is None and self.completed_s is not None))
            if not np.isclose(sum(values.values()), reward, atol=1e-8):
                raise AssertionError('reward instrumentation diverged from production state machine')
            for key, value in values.items():
                self.reward_totals[key] += value
        return reward


class StudyEnv(MissionEnv):
    def __init__(self, *args, approach_distance=.65, **kwargs):
        super().__init__(*args, **kwargs)
        if not .36 < approach_distance <= .65:
            raise ValueError('diagnostic approach spawn must start outside success radius')
        self.approach_distance = approach_distance

    def reset(self, layout_index=None):
        super().reset(layout_index)
        if self.stage == 'approach' and self.approach_distance != .65:
            direction = (self.layout.xy(self.layout.route[-2])-self.layout.xy(self.layout.route[-1]))/self.layout.cell_m
            s = self.slot
            self.runner.d.qpos[s.qpos_adr:s.qpos_adr+2] += direction*(self.approach_distance-.65)
            mujoco.mj_forward(self.model, self.runner.d)
            self.sensors.next_capture[0] = 0
            self.sensors.capture(self.runner.d, 0.)
        self.state = AuditedState(self.stage)
        self.reward_samples = []
        self.action_samples = []
        self.history[:] = self.observe_frame()
        return self.history.ravel().copy()

    def step(self, action):
        before = self.state.reward_totals.copy()
        obs, reward, done, info = super().step(action)
        terms = {k: self.state.reward_totals[k]-before[k] for k in TERMS}
        if not np.isclose(sum(terms.values()), reward, atol=1e-7):
            raise AssertionError('per-decision reward audit mismatch')
        self.reward_samples.append(terms)
        self.action_samples.append(np.tanh(action).tolist())
        info['reward_terms'] = terms
        return obs, reward, done, info

    def oracle_vector(self):
        # An explicit extra observation group, used only by privileged PPO B1.
        r, s = self.runner, self.slot
        delta = self.layout.xy(self.layout.route[-1])-r.d.xpos[s.body_id, :2]
        rot = r.d.xmat[s.body_id].reshape(3, 3)
        local = rot.T @ np.r_[delta, 0.]
        return np.r_[np.clip(local[:2]/2., -1., 1.), min(np.linalg.norm(delta)/2., 1.)].astype(np.float32)

    def metrics(self):
        row = super().metrics()
        row.update(reward_terms=self.state.reward_totals.copy(),
                   return_total=sum(self.state.reward_totals.values()),
                   action_mean_abs=float(np.mean(np.abs(self.action_samples))) if self.action_samples else 0.,
                   high_level_steps=len(self.action_samples), approach_spawn_m=self.approach_distance)
        return row


class OracleFusionNet(FusionNet):
    def __init__(self, outputs):
        super().__init__(outputs)
        self.head = nn.Sequential(nn.Linear(HISTORY*(61+66)+3, 128), nn.ELU(),
                                  nn.Linear(128, 64), nn.ELU(), nn.Linear(64, outputs))

    def forward(self, obs):
        base, oracle = obs[:, :HISTORY*FRAME], obs[:, HISTORY*FRAME:]
        return self.head(torch.cat([self.encode(base)[0], oracle], -1))


class OraclePolicy(ActorCritic):
    def __init__(self, obs):
        super().__init__(obs, {'policy': ['policy', 'oracle'], 'critic': ['policy', 'oracle']}, 5,
                         actor_hidden_dims=[16], critic_hidden_dims=[16], noise_std_type='log', init_noise_std=.5)
        self.actor, self.critic = OracleFusionNet(5), OracleFusionNet(1)


def study_matrix():
    base = dict(mode='train', arm='both', seed=0, updates=200, horizon=64, lr=3e-4,
                entropy=.01, noise=.5, schedule='adaptive', gamma=.995, lam=.95,
                epochs=4, minibatches=4, normalization=False, reward_scale=1.,
                approach_distance=.65, oracle=False, wall_hours=2)
    rows = [{**base, 'name': f'A{i+1}', 'study': 'stage isolation'+(' / C1 current PPO' if i == 0 else ''), 'stage': stage}
            for i, stage in enumerate(('approach','branches','navigation','doors','transport'))]
    rows += [{**base, 'name': 'B1', 'study': 'privileged goal PPO', 'stage': 'approach', 'oracle': True},
             {**base, 'name': 'C2', 'study': 'exploration', 'stage': 'approach', 'entropy': .02, 'noise': .8},
             {**base, 'name': 'C3', 'study': 'conservative optimizer', 'stage': 'approach', 'lr': 1e-4, 'schedule': 'fixed'},
             {**base, 'name': 'C4', 'study': 'easier approach / sensor Both', 'stage': 'approach', 'approach_distance': .40}]
    for name, arm in zip(('S1','S2','S3'), ('blind','lidar','stereo')):
        rows.append({**base, 'name': name, 'study': 'matched sensor pilot', 'stage': 'approach',
                     'arm': arm, 'approach_distance': .40, 'requires': 'C4'})
    rows += [dict(name='D1', mode='audits', study='reward + observation + layout audit', seed=0, wall_hours=2),
             dict(name='D2', mode='interactions', study='door + transport feasibility', seed=0, wall_hours=2)]
    return rows
