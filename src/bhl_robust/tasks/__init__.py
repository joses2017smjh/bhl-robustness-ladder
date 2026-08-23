"""Registers the overlay gym tasks.

Imported by `scripts/train.py` *after* SimulationApp exists. Registration uses
direct class references, matching upstream's pattern.
"""

import gymnasium as gym

from . import coop_crew_env_cfg  # noqa: F401
from . import (push_env_cfg, terrain_env_cfg, arms_env_cfg, collision_env_cfg,
               coop_lift_env_cfg, coop_depth_env_cfg, depth_env_cfg,
               scan_env_cfg)
from berkeley_humanoid_lite.tasks.locomotion.velocity.config.biped import agents
from berkeley_humanoid_lite.tasks.locomotion.velocity.config.humanoid import agents as arm_agents

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

# --- 22-DoF (arms) counterparts ------------------------------------------
_ARM_PPO_CFG = arm_agents.rsl_rl_ppo_cfg.BerkeleyHumanoidLitePPORunnerCfg

gym.register(
    id="Velocity-BHL-Arms-PushAdaptive-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": arms_env_cfg.HumanoidPushAdaptiveCfg,
        "rsl_rl_cfg_entry_point": _ARM_PPO_CFG,
    },
)

gym.register(
    id="Velocity-BHL-Arms-Bumpy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": arms_env_cfg.HumanoidBumpyEnvCfg,
        "rsl_rl_cfg_entry_point": _ARM_PPO_CFG,
    },
)

gym.register(
    id="Velocity-BHL-Biped-ConvexCollision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": collision_env_cfg.BipedConvexCollisionCfg,
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
)

# --- dual-humanoid cooperative lift --------------------------------------
_COOP_PPO = coop_lift_env_cfg.CoopLiftPPORunnerCfg

gym.register(
    id="CoopLift-BHL-Cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": coop_lift_env_cfg.CoopLiftCubeCfg,
        "rsl_rl_cfg_entry_point": _COOP_PPO,
    },
)
gym.register(
    id="CoopLift-BHL-Ladder-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": coop_lift_env_cfg.CoopLiftLadderCfg,
        "rsl_rl_cfg_entry_point": _COOP_PPO,
    },
)
gym.register(
    id="CoopLift-BHL-Ball-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": coop_lift_env_cfg.CoopLiftBallCfg,
        "rsl_rl_cfg_entry_point": _COOP_PPO,
    },
)
# Vision inside the lift loop. `MultiMeshRayCasterCamera` tracks the payload's
# transform, which the static-mesh `RayCasterCamera` of §6 cannot do — so this
# is depth of a scene whose interesting object moves, without the RTX renderer.
gym.register(
    id="CoopLift-BHL-Cube-Depth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": coop_depth_env_cfg.CoopLiftDepthCfg,
        "rsl_rl_cfg_entry_point": _COOP_PPO,
    },
)

# --- depth-conditioned locomotion -----------------------------------------
gym.register(
    id="Velocity-BHL-Biped-Depth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": depth_env_cfg.BipedDepthEnvCfg,
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
)

gym.register(
    id="Velocity-BHL-Biped-Scan-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": scan_env_cfg.BipedScanEnvCfg,
        # PPO on the privileged group; distillation on the blind one. Same env.
        "rsl_rl_cfg_entry_point": scan_env_cfg.ScanTeacherPPORunnerCfg,
        "rsl_rl_distillation_cfg_entry_point": scan_env_cfg.ScanStudentDistillCfg,
    },
)

gym.register(
    id="Velocity-BHL-Biped-FlatFill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": terrain_env_cfg.BipedFlatFillEnvCfg,
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
)

# --- real crews: N robots, ONE payload ------------------------------------
# The pair task replicated across independent crates is not a crew; these are.
# Registered blind and sighted at each size, so "does vision help a lift" and
# "does a bigger crew help a lift" are separable rather than confounded.
for _n in (3, 4):
    for _vision in (False, True):
        _sfx = "-Depth" if _vision else ""
        gym.register(
            id=f"CoopLift-BHL-Cube-Crew{_n}{_sfx}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": coop_crew_env_cfg.make_crew_cfg(
                    _n, "cube", vision=_vision),
                "rsl_rl_cfg_entry_point": _COOP_PPO,
            },
        )
