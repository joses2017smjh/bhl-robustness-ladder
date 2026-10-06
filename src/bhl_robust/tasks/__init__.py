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
               depth_env_cfg, rgb_env_cfg, scan_env_cfg, task_v2_env_cfg,
               cloth_sort_env_cfg)
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

# Turning-gait arms (2026-09-24): the shipped humanoid gait ignores yaw-rate
# commands in MuJoCo; each arm changes one suspect in the reward set (see
# arms_env_cfg.py). Same DR (s = 1.0) as arms-dr1.0 through the launcher.
for _id, _cfg in (
    ("Velocity-BHL-Arms-TurnHip-v0", arms_env_cfg.HumanoidTurnHipCfg),
    ("Velocity-BHL-Arms-TurnTrack-v0", arms_env_cfg.HumanoidTurnTrackCfg),
    ("Velocity-BHL-Arms-TurnBoth-v0", arms_env_cfg.HumanoidTurnBothCfg),
    # 2026-09-26: TurnBoth rewards + a command mix that trains sustained pure
    # turns (turn_command.TurnMixVelocityCommand); judged by turn_test --protocol v2.
    ("Velocity-BHL-Arms-TurnCmd-v0", arms_env_cfg.HumanoidTurnCmdCfg),
    # 2026-09-27: TurnBoth rewards + pure-turn / rest-then-turn / direct / upstream command mix
    # (turn_command.TurnRestMixVelocityCommand); trained as a FINE-TUNE of arms-turn-turnboth-s0
    # by slurm/repo20260923/gpu_turngait_v4.sbatch; judged by turn_test --protocol v2.
    ("Velocity-BHL-Arms-TurnRest-v0", arms_env_cfg.HumanoidTurnRestCfg),
    # 2026-09-27: TurnRest + interval push with the adaptive push curriculum (TurnRest
    # failed its push regression); fine-tuned by gpu_turngait_v5.sbatch.
    ("Velocity-BHL-Arms-TurnRestPush-v0", arms_env_cfg.HumanoidTurnRestPushCfg),
    # 2026-10-01 R1 / R2 (docs/SOLUTIONS_2026-10-01.md section 2): TurnBoth + feet_gait contact schedule at
    # every command + swing height + fixed pushes, from scratch (gpu_turngait_r12.sbatch); clock in actor / critic only.
    ("Velocity-BHL-Arms-TurnGaitClock-v0", arms_env_cfg.HumanoidTurnGaitClockCfg),
    ("Velocity-BHL-Arms-TurnGaitCritic-v0", arms_env_cfg.HumanoidTurnGaitCriticCfg),
):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": _cfg, "rsl_rl_cfg_entry_point": _ARM_PPO_CFG},
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
    # Standing-height cube: a different, easier task (see its docstring).
    ("CubeToShelfStand", task_v2_env_cfg.CUBE_STAND_VARIANTS),
    # Standing-height cube, v2: upright-gated shaping, priced fall, placement
    # worth more than hovering. A different, easier task than CubeToShelf.
    ("CubeToShelfStand2", task_v2_env_cfg.CUBE_STAND2_VARIANTS),
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

# Side-deck standing cube (CubeToShelfStand3): a different, easier task than
# CubeToShelf and CubeToShelfStand2, with its OWN runner (log std, entropy 0.001).
# Separate loop: the tuple loop above hardcodes _V2_RUNNER.
for _vis, _cls in task_v2_env_cfg.CUBE_STAND3_VARIANTS.items():
    gym.register(
        id=f"TaskV2-BHL-CubeToShelfStand3-{_vis.capitalize()}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": _cls,
            "rsl_rl_cfg_entry_point": task_v2_env_cfg._STAND3_RUNNER,
        },
    )
# Roll-proof lift cube (CubeToShelfStand4, 2026-10-01): a different task from Stand3
# and CubeToShelf, Stand3's runner. Blind only (the id its launchers train).
gym.register(
    id="TaskV2-BHL-CubeToShelfStand4-Blind-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": task_v2_env_cfg.CUBE_STAND4_VARIANTS["blind"],
        "rsl_rl_cfg_entry_point": task_v2_env_cfg._STAND3_RUNNER,
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

# ---------------------------------------------------------------- B5: the maze
# Corridor navigation with a 2D lidar and a global-shutter stereo pair, on the
# locomotion rung rather than the manipulation one -- locomotion is the half of
# this project that works, and a new sensor rung belongs there.
#
# Four arms so each sensor can be credited only with what it could have seen:
# the floor obstacles sit under the lidar plane, the corners are outside the
# cameras' cone until the turn is made, and the blind arm is the control without
# which a lidar number is a fact about the maze.
from bhl_robust.tasks import maze_env_cfg  # noqa: E402

for _id, _cfg in (
    ("Velocity-BHL-Maze-Blind-v0", maze_env_cfg.MazeBlindEnvCfg),
    ("Velocity-BHL-Maze-Lidar-v0", maze_env_cfg.MazeLidarEnvCfg),
    ("Velocity-BHL-Maze-Stereo-v0", maze_env_cfg.MazeStereoEnvCfg),
    ("Velocity-BHL-Maze-Both-v0", maze_env_cfg.MazeBothEnvCfg),
    # Pooling sweep: does coarser stereo recover, or do cameras simply not help?
    ("Velocity-BHL-Maze-StereoP8-v0", maze_env_cfg.MazeStereoP8EnvCfg),
    ("Velocity-BHL-Maze-StereoP16-v0", maze_env_cfg.MazeStereoP16EnvCfg),
    ("Velocity-BHL-Maze-BothP16-v0", maze_env_cfg.MazeBothP16EnvCfg),
):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": _cfg, "rsl_rl_cfg_entry_point": _PPO_CFG},
    )

# Versioned repair: leave the legacy maze IDs/rewards replayable as recorded.
from .maze_recovery_env_cfg import RECOVERY_CONFIGS  # noqa: E402
for _id, _cfg in RECOVERY_CONFIGS.items():
    gym.register(id=_id, entry_point="isaaclab.envs:ManagerBasedRLEnv",
                 disable_env_checker=True,
                 kwargs={"env_cfg_entry_point": _cfg, "rsl_rl_cfg_entry_point": _PPO_CFG})

# ------------------------------------------------ SF-04 teacher-student distillation
# The Full-stage BothRobust env publishing a clean `teacher` group beside the
# degraded `policy` group, so the working Both Full s0 MLP can be distilled into
# a recurrent student (src/bhl_robust/tasks/maze_distill.py). Three entry
# points: the env, the ordinary PPO cfg (unused, keeps the id uniform) and the
# distillation cfg that scripts/train_distill.py and the probe's
# --runner distillation read.
#
# Guarded like Tier 3: the module needs isaaclab_rl's rsl-rl >= 4 model cfgs
# (RslRlRNNModelCfg, handle_deprecated_rsl_rl_cfg), which the v51 stack does
# not ship, and every v51 job imports this registry.
try:
    from bhl_robust.tasks import maze_distill  # noqa: E402

    gym.register(
        id="Velocity-BHL-MazeRecovery-Full-BothRobustDistill-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": maze_distill.MazeBothRobustDistillEnvCfg,
            "rsl_rl_cfg_entry_point": _PPO_CFG,
            "rsl_rl_distillation_cfg_entry_point": maze_distill.MazeStudentDistillCfg,
        },
    )
except Exception as _exc:  # noqa: BLE001
    import sys as _sys
    print(f"[bhl_robust.tasks] SF-04 distillation id NOT registered: {_exc!r}", file=_sys.stderr, flush=True)

# ------------------------------------------------------------------ Tier 3
# The 22-DoF robot on stairs with depth and the arm-deviation penalty ablated in
# the task itself, so the rsl-rl PPO rows and the skrl MAPPO rows cannot differ
# on it. Ice joins once B3's patches are where the robots are.
#
# Guarded: this registry is imported by every job in the shared tree, including
# ones already queued on other experiments, and a new module that fails to
# import must not take them down. The failure is printed, the id is simply
# absent, and the Tier 3 gate refuses to pass without it.
try:
    from bhl_robust.tasks import arms_terrain_env_cfg  # noqa: E402

    gym.register(
        id="Velocity-BHL-Arms-Stairs-Depth-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": arms_terrain_env_cfg.HumanoidStairsDepthEnvCfg,
            "rsl_rl_cfg_entry_point": _ARM_PPO_CFG,
        },
    )
except Exception as _exc:  # noqa: BLE001
    import sys as _sys
    print(f"[bhl_robust.tasks] Tier 3 ids NOT registered: {_exc!r}", file=_sys.stderr, flush=True)

# ---------------------------------------------------------------- cloth-sort
# Hierarchical rigid-to-deformable ladder. See docs/CLOTH_SORT.md.
# Rigid ids are the training path. Deformable ids construct a Newton
# MeshRectangle (default 8×8) and are cost-gated before any job. ActiveCloth
# is Mode B: one live cloth plus four rigid proxies, not five live cloths.
_CLOTH_PPO = cloth_sort_env_cfg.ClothSortPPORunnerCfg
for _id, _cfg in (
    ("ClothSort-BHL-Rigid-Oracle-v0", cloth_sort_env_cfg.ClothSortRigidEnvCfg),
    ("ClothSort-BHL-RigidFixedBase-Oracle-v0", cloth_sort_env_cfg.ClothSortRigidFixedBaseEnvCfg),
    ("ClothSort-BHL-RigidBalance-Oracle-v0", cloth_sort_env_cfg.ClothSortRigidBalanceEnvCfg),
    ("ClothSort-BHL-RigidResidual-Oracle-v0", cloth_sort_env_cfg.ClothSortRigidResidualEnvCfg),
    ("ClothSort-BHL-RigidFive-Oracle-v0", cloth_sort_env_cfg.ClothSortRigidFiveEnvCfg),
    ("ClothSort-BHL-Deformable-Oracle-v0", cloth_sort_env_cfg.ClothSortDeformableEnvCfg),
    ("ClothSort-BHL-DeformableFixedBase-Oracle-v0", cloth_sort_env_cfg.ClothSortDeformableFixedBaseEnvCfg),
    ("ClothSort-BHL-ActiveCloth-Oracle-v0", cloth_sort_env_cfg.ClothSortActiveDeformableEnvCfg),
):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": _cfg, "rsl_rl_cfg_entry_point": _CLOTH_PPO},
    )

# ------------------------------------------------- B3 control: ice, no ice
# The placed-ice cfgs with every ice_* patch at the ground's own friction.
# Everything else -- patch placement, reset, terrain, curriculum, cameras --
# is inherited unchanged, so a depth advantage that survives here is not
# about friction (LOC-11, docs/REPO_TASKS.md).
from bhl_robust.tasks import ice_control_env_cfg as _ice_control  # noqa: E402
for _id, _cfg in (
    ("Velocity-BHL-Biped-IceControl-v0", _ice_control.BipedIceControlEnvCfg),
    ("Velocity-BHL-Biped-IceControl-Depth-v0", _ice_control.BipedIceControlDepthEnvCfg),
):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": _cfg, "rsl_rl_cfg_entry_point": _PPO_CFG},
    )

# --- m7-platecross ---
# Mission 7 learned crossing (2026-10-02; SLURM_JOBS.md 'User approval recorded 2026-10-02 14:15', item (B)):
# Velocity-BHL-Arms-TurnGaitClock-v0 (R1) unchanged except the terrain, flat ground scattered with Mission 7's
# own plates (platecross_env_cfg.py / platecross_terrain.py); fine-tuned from arms-turngait-clock-s2 by
# slurm/repo20260923/gpu_platecross.sbatch. Guarded like Tier 3: a failure is printed and only this id is absent.
try:
    from bhl_robust.tasks import platecross_env_cfg as _platecross  # noqa: E402

    gym.register(
        id="Velocity-BHL-Arms-PlateCross-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": _platecross.HumanoidPlateCrossCfg, "rsl_rl_cfg_entry_point": _ARM_PPO_CFG},
    )
except Exception as _exc:  # noqa: BLE001
    import sys as _sys
    print(f"[bhl_robust.tasks] m7-platecross id NOT registered: {_exc!r}", file=_sys.stderr, flush=True)
# --- end m7-platecross ---

# --- turning-hold ---
# Turning follow-up R1H (2026-10-02; SLURM_JOBS.md '(C') revised design'): Velocity-BHL-Arms-TurnGaitClock-v0 (R1)
# + 30 % explicit-command envs, half of them at wz = 0 (turn_command.TurnHoldMixVelocityCommand) + the heading_hold
# reward (gait_clock_mdp.heading_hold); a NEW task trained from scratch by slurm/repo20260923/gpu_turngait_hold.sbatch,
# R1's runner. Guarded like m7-platecross: a failure is printed and only this id is absent.
try:
    gym.register(
        id="Velocity-BHL-Arms-TurnGaitClockHold-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": arms_env_cfg.HumanoidTurnGaitClockHoldCfg, "rsl_rl_cfg_entry_point": _ARM_PPO_CFG},
    )
except Exception as _exc:  # noqa: BLE001
    import sys as _sys
    print(f"[bhl_robust.tasks] turning-hold id NOT registered: {_exc!r}", file=_sys.stderr, flush=True)
# --- end turning-hold ---

# --- stand5 ---
# CubeToShelfStand5 (2026-10-03; SLURM_JOBS.md 'User approval recorded 2026-10-03 09:55', item (S)): the Stand4
# task plus two changes -- the actor and the critic observe the cube's z axis in each robot's root frame, and the
# lift curriculum promotes on the roll-proof lift condition (task_v2_env_cfg.CubeToShelfStand5Cfg, stand5_mdp.py).
# Stand3's runner, as Stand4. Blind only (the id its launchers train). Guarded like m7-platecross: a failure is
# printed and only this id is absent.
try:
    gym.register(
        id="TaskV2-BHL-CubeToShelfStand5-Blind-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": task_v2_env_cfg.CUBE_STAND5_VARIANTS["blind"],
            "rsl_rl_cfg_entry_point": task_v2_env_cfg._STAND3_RUNNER,
        },
    )
except Exception as _exc:  # noqa: BLE001
    import sys as _sys
    print(f"[bhl_robust.tasks] stand5 id NOT registered: {_exc!r}", file=_sys.stderr, flush=True)
# --- end stand5 ---

# --- m7-platecross2 ---
# Mission 7 learned crossing F1 "PlateCross v2" (2026-10-04; SLURM_JOBS.md 'User approval recorded 2026-10-03 09:55',
# item (F), part F1): Velocity-BHL-Arms-PlateCross-v0 with half of the tiles flat (platecross_env_cfg.py,
# HumanoidPlateCross2Cfg); fine-tuned from arms-turngait-clock-s2 by slurm/repo20260923/gpu_platecross2.sbatch.
# Guarded: a failure is printed and only this id is absent.
try:
    from bhl_robust.tasks import platecross_env_cfg as _platecross2  # noqa: E402

    gym.register(
        id=_platecross2.TASK_ID_V2,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": _platecross2.HumanoidPlateCross2Cfg, "rsl_rl_cfg_entry_point": _ARM_PPO_CFG},
    )
except Exception as _exc:  # noqa: BLE001
    import sys as _sys
    print(f"[bhl_robust.tasks] m7-platecross2 id NOT registered: {_exc!r}", file=_sys.stderr, flush=True)
# --- end m7-platecross2 ---

# --- waiter-wbc ---
# Waiter program phase 1 (2026-10-05; docs/WAITER_PROGRAM.md, SLURM_JOBS.md 'Predeclared now ... the Waiter
# program'): the 24-DoF gripper whole-body controller, R1's recipe with the legs as the policy and the arms and
# grippers following an upper-body command (waiter_env_cfg.HumanoidWaiterWbcCfg); trained from scratch by
# slurm/repo20260923/gpu_waiter_wbc.sbatch. Guarded: a failure is printed and only this id is absent.
try:
    from bhl_robust.tasks import waiter_env_cfg as _waiter  # noqa: E402

    gym.register(
        id=_waiter.TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": _waiter.HumanoidWaiterWbcCfg, "rsl_rl_cfg_entry_point": _ARM_PPO_CFG},
    )
except Exception as _exc:  # noqa: BLE001
    import sys as _sys
    print(f"[bhl_robust.tasks] waiter-wbc id NOT registered: {_exc!r}", file=_sys.stderr, flush=True)
# --- end waiter-wbc ---
