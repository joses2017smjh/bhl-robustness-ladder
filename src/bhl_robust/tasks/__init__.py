"""Registers the overlay gym tasks.

Imported by `scripts/train.py` *after* SimulationApp exists. Registration uses
direct class references, matching upstream's pattern.
"""

import gymnasium as gym

# Must run before anything imports an upstream config: on Isaac Lab 3.x those
# configs import names the 3.x API removed, and the overlays inherit the
# failure. No-op on 2.x.
from bhl_robust import compat as _compat
_compat.apply()

from . import coop_crew_generated as crew  # noqa: F401
from . import (push_env_cfg, terrain_env_cfg, arms_env_cfg, collision_env_cfg,
               coop_lift_env_cfg, coop_depth_env_cfg, coop_hard_env_cfg,
               depth_env_cfg, rgb_env_cfg, scan_env_cfg, task_v2_env_cfg)
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
#
# The config classes are generated source (scripts/gen_crew_cfg.py), not classes
# assembled at import with type(). The dynamic version passed parse_env_cfg and
# then lost every generated term to Hydra's to_dict/from_dict round trip,
# because that path carries declared dataclass fields only.
for _n, _cls in ((3, crew.Crew3Cfg), (4, crew.Crew4Cfg)):
    gym.register(
        id=f"CoopLift-BHL-Cube-Crew{_n}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": _cls, "rsl_rl_cfg_entry_point": _COOP_PPO},
    )
for _n, _cls in ((3, crew.Crew3DepthCfg), (4, crew.Crew4DepthCfg)):
    gym.register(
        id=f"CoopLift-BHL-Cube-Crew{_n}-Depth-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": _cls, "rsl_rl_cfg_entry_point": _COOP_PPO},
    )

# --- harder payloads, and the only fair test of vision --------------------
# Randomised mass/friction; and an "occluded" pair that withholds the exact
# object pose so depth has something to contribute instead of duplicating a
# quantity the policy was already handed.
for _id, _cls in (
    ("CoopLift-BHL-Cube-Random-v0", coop_hard_env_cfg.CoopLiftRandomCfg),
    ("CoopLift-BHL-Cube-Occluded-v0", coop_hard_env_cfg.CoopLiftOccludedCfg),
    ("CoopLift-BHL-Cube-Occluded-Depth-v0", coop_hard_env_cfg.CoopLiftOccludedDepthCfg),
):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": _cls, "rsl_rl_cfg_entry_point": _COOP_PPO},
    )

# --- rendered colour, Isaac Sim 6.0 only ----------------------------------
# Registers on either stack because TiledCameraCfg exists in both, but only
# *runs* on 6.0: on 5.1 the RTX renderer segfaults before the first frame, which
# is the entire reason section 6 uses ray-casting.
gym.register(
    id="Velocity-BHL-Biped-Rgb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rgb_env_cfg.BipedRgbEnvCfg,
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
)

# --- terrain family: material terrains and geometry terrains ---------------
# `slippery` shares its geometry with `bumpy` exactly and differs only in
# contact friction, so it is the terrain on which ray-cast depth must show no
# gain. That is the point of it: a negative control for the depth claim.
gym.register(
    id="Velocity-BHL-Biped-Slippery-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": terrain_env_cfg.BipedSlipperyEnvCfg,
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
)

# Stairs, blind and with ray-cast depth. The geometry terrain, paired with
# `slippery` above: depth must help here and must not help there.
gym.register(
    id="Velocity-BHL-Biped-Stairs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": terrain_env_cfg.BipedStairsEnvCfg,
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
)
gym.register(
    id="Velocity-BHL-Biped-Stairs-Depth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": depth_env_cfg.BipedStairsDepthEnvCfg,
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
)
gym.register(
    id="Velocity-BHL-Biped-Slippery-Depth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": depth_env_cfg.BipedSlipperyDepthEnvCfg,
        "rsl_rl_cfg_entry_point": _PPO_CFG,
    },
)


# ---------------------------------------------------------------- v2 tasks
# Three tasks with a terminal success state, each in blind / depth / rgb.
# All nine are v60-only: the sighted arms need an RTX renderer that 5.1
# segfaults in, and the blind arm runs on the same stack or it is not a
# control for them.
for _task, _variants in (
    ("CubeToShelf", task_v2_env_cfg.CUBE_VARIANTS),
    ("BallToNet", task_v2_env_cfg.BALL_VARIANTS),
    ("PlankToWall", task_v2_env_cfg.PLANK_VARIANTS),
    # The solo control decides whether the paired ball number is a cooperation
    # result or one robot doing the job with an audience.
    ("BallToNetSolo", task_v2_env_cfg.BALL_SOLO_VARIANTS),
    # The same three tasks on the 24-DoF gripper asset. Separate ids, not a
    # flag, so the welded-hand arms stay runnable as their control.
    ("CubeToShelfGrip", task_v2_env_cfg.CUBE_GRIPPER_VARIANTS),
    ("BallToNetGrip", task_v2_env_cfg.BALL_GRIPPER_VARIANTS),
    ("PlankToWallGrip", task_v2_env_cfg.PLANK_GRIPPER_VARIANTS),
):
    for _vis, _cls in _variants.items():
        gym.register(
            id=f"TaskV2-BHL-{_task}-{_vis.capitalize()}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": _cls,
                "rsl_rl_cfg_entry_point": task_v2_env_cfg._V2_RUNNER,
            },
        )

# ------------------------------------------------------------------ B3: ice
# Patchy friction on flat ground. The blind/depth pair is the negative control
# for the depth claim; the visible arm separates "depth helps without seeing"
# from "a camera helps once the patch can be seen".
for _id, _cfg in (
    ("Velocity-BHL-Biped-Ice-v0", terrain_env_cfg.BipedIceEnvCfg),
    ("Velocity-BHL-Biped-Ice-Depth-v0", depth_env_cfg.BipedIceDepthEnvCfg),
    ("Velocity-BHL-Biped-IceVisible-v0", terrain_env_cfg.BipedIceVisibleEnvCfg),
):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": _cfg, "rsl_rl_cfg_entry_point": _PPO_CFG},
    )
