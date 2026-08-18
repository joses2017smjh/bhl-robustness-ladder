"""Registers the overlay gym tasks.

Imported by `scripts/train.py` *after* SimulationApp exists. Registration uses
direct class references, matching upstream's pattern.
"""

import gymnasium as gym

from . import push_env_cfg, terrain_env_cfg
from berkeley_humanoid_lite.tasks.locomotion.velocity.config.biped import agents

_PPO_CFG = agents.rsl_rl_ppo_cfg.BerkeleyHumanoidLiteBipedPPORunnerCfg

gym.register(
    id="Velocity-BHL-Biped-Push-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": push_env_cfg.BipedPushEnvCfg,
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
)

gym.register(
    id="Velocity-BHL-Biped-PushCurriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": push_env_cfg.BipedPushCurriculumCfg,
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
)

gym.register(
    id="Velocity-BHL-Biped-Bumpy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": terrain_env_cfg.BipedBumpyEnvCfg,
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
)

gym.register(
    id="Velocity-BHL-Biped-Smooth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": terrain_env_cfg.BipedSmoothEnvCfg,
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
)

gym.register(
    id="Velocity-BHL-Biped-PushAdaptive-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": push_env_cfg.BipedPushAdaptiveCfg,
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
)
