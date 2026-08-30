"""Limb agents over the existing locomotion env: MAPPO/IPPO against PPO.

The design constraint that shapes everything here: **the only thing that may
differ from the single-agent baseline is the action factorisation.** Same
physics, same rewards, same terminations, same observation content. If anything
else moves, a Tier 1 row stops being "does factoring the policy by limb help"
and becomes "does this other env behave differently", which is not a question
anyone asked.

So this wraps `ManagerBasedRLEnv` rather than reimplementing it as a Direct env.
Writing a Direct MARL locomotion env from scratch would mean reproducing the
reward manager, the curriculum, the events and the terminations -- hundreds of
lines whose only job is to be identical to code that already exists, and every
one of them a chance to be accidentally different. The wrapper cannot drift,
because there is nothing to drift from.

Agents are cooperative and share one team reward. That is deliberate and worth
stating, because it is the assumption most likely to be wrong: a per-limb reward
would need a credit assignment this task has no way to compute, and inventing
one would make the comparison against PPO meaningless.
"""

from __future__ import annotations

from typing import Any

import torch

from bhl_robust.limb_partition import PARTITIONS, reassemble, validate


class LimbMarlEnv:
    """Multi-agent view of a single-agent locomotion env, split by limb.

    Exposes the PettingZoo-flavoured surface skrl's multi-agent wrapper expects:
    `possible_agents`, `observation_spaces`, `action_spaces`, `state()`, and a
    `step` taking `{agent: action}`.
    """

    def __init__(self, env, partition: str = "limb4", share_obs: bool = True):
        self.env = env
        self.kind = partition
        self.partition = PARTITIONS[partition]
        validate(self.partition)
        self.share_obs = share_obs
        self.possible_agents = list(self.partition.keys())
        self.agents = list(self.possible_agents)

        self._n_act = {a: len(i) for a, i in self.partition.items()}
        obs, _ = self.env.reset()
        self._obs_dim = int(self._policy(obs).shape[-1])

    # ------------------------------------------------------------- plumbing

    @staticmethod
    def _policy(obs) -> torch.Tensor:
        return obs["policy"] if isinstance(obs, dict) else obs

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def num_envs(self) -> int:
        return self.env.unwrapped.num_envs

    @property
    def device(self):
        return self.env.unwrapped.device

    @property
    def max_episode_length(self) -> int:
        return self.env.unwrapped.max_episode_length

    @property
    def num_agents(self) -> int:
        """Live agent count, which skrl's trainer reads every step.

        Distinct from `len(possible_agents)` in a general PettingZoo env, where
        agents can drop out mid-episode. Here every limb is attached for the
        whole episode, so the two are always equal -- but the trainer asks for
        this name and got an AttributeError, which is how all three Tier 1 MARL
        rows died in a minute while the PPO control ran for five hours.
        """
        return len(self.agents)

    @property
    def num_actions(self) -> dict[str, int]:
        return dict(self._n_act)

    @property
    def num_observations(self) -> dict[str, int]:
        return {a: self._obs_dim for a in self.possible_agents}

    @property
    def num_states(self) -> int:
        """State handed to a centralised critic.

        The full policy observation, which is the same information the
        single-agent baseline's privileged critic already receives. Giving MAPPO
        *more* than that would confound the algorithm with the information.
        """
        return self._obs_dim

    # ------------------------------------------------------------ interface

    def _fan_out(self, obs) -> dict[str, torch.Tensor]:
        """One observation to every agent.

        Shared rather than per-limb-sliced. A limb that sees only its own joints
        cannot know what the others are doing, and the resulting failure would be
        a story about partial observability rather than about factorisation.
        Slicing is a separate experiment, and a later one.
        """
        p = self._policy(obs)
        return {a: p for a in self.possible_agents}

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._last = self._policy(obs)
        self.agents = list(self.possible_agents)
        return self._fan_out(obs), info

    def step(self, actions: dict[str, torch.Tensor]):
        joined = reassemble(actions, self.partition)
        obs, rew, term, trunc, info = self.env.step(joined)
        self._last = self._policy(obs)
        o = self._fan_out(obs)
        # One team reward and one shared done, copied per agent -- each as
        # (num_envs, 1), not (num_envs,).
        #
        # The single-agent env returns a flat (num_envs,) vector. skrl's
        # multi-agent memory stores one column per agent and broadcasts a bare
        # (4096,) against a (4096, 1) slot, which produces
        #   output with shape [4096, 1] doesn't match the broadcast shape
        #   [4096, 4096]
        # and killed every Tier 1 MARL row in under a minute while the PPO
        # control ran for five hours. The trailing axis is the whole fix.
        def col(x):
            return x.reshape(-1, 1)

        r = {a: col(rew) for a in self.possible_agents}
        t = {a: col(term) for a in self.possible_agents}
        u = {a: col(trunc) for a in self.possible_agents}
        return o, r, t, u, info

    def state(self) -> torch.Tensor:
        return self._last

    def render(self, *a, **kw):
        return self.env.render(*a, **kw)

    def close(self):
        return self.env.close()


#: The arm-deviation penalty, under the names upstream actually uses.
#:
#: The work order calls it `joint_deviation_arms` and so does this repo's own
#: `arms_env_cfg` docstring, but no such term exists: upstream splits it into
#: `joint_deviation_shoulder` and `joint_deviation_elbow`. G-B4 reported
#: "TERM NOT FOUND" against the wrong name and would have trained every Tier 1
#: row with the penalty still on -- an ablation that silently does nothing is
#: worse than one that fails, because the table would have carried a column
#: claiming it happened.
ARM_DEVIATION_TERMS = ("joint_deviation_shoulder", "joint_deviation_elbow",
                       "joint_deviation_arms", "joint_deviation_arm")


def ablate_arm_deviation(cfg) -> list[str]:
    """Turn off the arm-deviation penalties. Returns the names actually cleared.

    They penalise the arms for leaving their default pose. With one agent owning
    each arm that is a per-agent penalty for moving at all, and section 5 already
    found this family of term fighting a squat-and-pinch. Returning the names
    lets a caller fail loudly rather than assume the ablation happened.
    """
    rewards = getattr(cfg, "rewards", None)
    if rewards is None:
        return []
    cleared = []
    for name in ARM_DEVIATION_TERMS:
        if getattr(rewards, name, None) is not None:
            setattr(rewards, name, None)
            cleared.append(name)
    return cleared
