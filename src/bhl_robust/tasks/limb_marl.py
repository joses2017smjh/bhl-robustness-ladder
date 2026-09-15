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

from bhl_robust.limb_partition import partition_for, reassemble_n, validate


def loggable(log, device) -> dict:
    """Isaac Lab's `extras["log"]`, with every scalar made a 0-d tensor.

    skrl's trainer writes an info entry only if it is a one-element tensor.
    Isaac Lab hands the reward sums over as tensors but calls `.item()` on the
    curriculum and termination entries, so `Curriculum/terrain_levels` -- the
    work order's primary metric -- arrived as a float and was dropped, even
    after `environment_info="log"` (`21302171_4` logged 28 Info tags, none of
    them the terrain level). Converting here keeps the env untouched.
    """
    if not isinstance(log, dict):
        return log
    for k, v in list(log.items()):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            log[k] = torch.tensor(float(v), device=device)
    return log


class LimbMarlEnv:
    """Multi-agent view of a single-agent locomotion env, split by limb.

    Exposes the PettingZoo-flavoured surface skrl's multi-agent wrapper expects:
    `possible_agents`, `observation_spaces`, `action_spaces`, `state()`, and a
    `step` taking `{agent: action}`.
    """

    def __init__(self, env, partition: str = "limb4", share_obs: bool = True,
                 critic: str = "privileged"):
        self.env = env
        self.kind = partition
        # Size the partition to the asset in use: 22 DoF with welded hands, 24
        # with grippers. Assuming one broke the gate against the other.
        self.n_dof = int(env.unwrapped.action_space.shape[-1])
        self.partition = partition_for(partition, self.n_dof)
        self.share_obs = share_obs
        self.possible_agents = list(self.partition.keys())
        self.agents = list(self.possible_agents)

        self._n_act = {a: len(i) for a, i in self.partition.items()}
        self.joint_names = self._action_joint_names()
        self._check_semantics()
        if critic not in ("privileged", "policy"):
            raise ValueError(f"critic must be 'privileged' or 'policy', got {critic!r}")
        self.critic = critic
        obs, _ = self.env.reset()
        self._obs_dim = int(self._policy(obs).shape[-1])
        self._state_dim = int(self._state_of(obs).shape[-1])

    # ------------------------------------------------------------ semantics

    #: The joint-name prefix every joint an agent owns must carry.
    _PREFIX = {"arm_left": "arm_left", "arm_right": "arm_right",
               "leg_left": "leg_left", "leg_right": "leg_right",
               "arms": "arm_", "legs": "leg_"}

    def _action_joint_names(self) -> list[str] | None:
        """The joint order the env's action vector is actually in.

        Read from the constructed action term rather than assumed from
        `limb_partition`'s constants, so a task whose action term lists its
        joints in another order is caught here instead of training a robot
        whose "left leg" agent drives a right hip.
        """
        try:
            mgr = self.env.unwrapped.action_manager
            names = []
            for term in mgr.active_terms:
                names.extend(mgr.get_term(term)._joint_names)
            return names if len(names) == self.n_dof else None
        except Exception:                                        # noqa: BLE001
            return None

    def _check_semantics(self) -> None:
        if self.joint_names is None:
            raise RuntimeError("could not read the action term's joint names; "
                               "refusing to assume the partition is semantic")
        for agent, idx in self.partition.items():
            prefix = self._PREFIX.get(agent)
            if prefix is None:      # limb1's "whole" owns everything by definition
                continue
            wrong = [self.joint_names[i] for i in idx if not self.joint_names[i].startswith(prefix)]
            if wrong:
                raise RuntimeError(f"agent {agent} would drive {wrong}: the env's action "
                                   f"order is not the layout {self.kind} was built for")

    # ------------------------------------------------------------- plumbing

    @staticmethod
    def _policy(obs) -> torch.Tensor:
        return obs["policy"] if isinstance(obs, dict) else obs

    def _state_of(self, obs) -> torch.Tensor:
        """What a centralised critic reads.

        `privileged` is the env's `critic` group -- the policy terms uncorrupted,
        plus base linear velocity -- which is exactly what rsl-rl's PPO critic
        reads. The first version handed MAPPO the noisy *policy* observation,
        which is also what every IPPO critic already sees (agents share the full
        observation), so MAPPO and IPPO were the same algorithm, and both critics
        lacked the velocity a velocity-tracking value needs. On stairs every such
        row plateaued at 0.53 tracking reward against PPO's 1.52 and never left
        terrain level 0 (21328607/8). `policy` keeps that behaviour for replaying
        the first block.
        """
        if self.critic == "privileged" and isinstance(obs, dict) and "critic" in obs:
            return obs["critic"]
        return self._policy(obs)

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
        """Width of the state handed to a centralised critic (see `_state_of`).

        This used to claim the policy observation "is the same information the
        single-agent baseline's privileged critic already receives". It is not:
        rsl-rl's critic reads the `critic` group, which adds base linear velocity
        and drops the noise. Matching that is what makes PPO a valid control.
        """
        return self._state_dim

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
        self._last = self._state_of(obs)
        self.agents = list(self.possible_agents)
        return self._fan_out(obs), info

    def step(self, actions: dict[str, torch.Tensor]):
        joined = reassemble_n(actions, self.partition, self.n_dof)
        obs, rew, term, trunc, info = self.env.step(joined)
        if isinstance(info, dict) and "log" in info:
            loggable(info["log"], self.device)
        self._last = self._state_of(obs)
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
