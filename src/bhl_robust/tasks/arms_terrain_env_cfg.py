"""Tier 3: the 22-DoF robot on stairs, with ray-cast depth and no arm-deviation penalty.

The work order's third tier asks whether the arms matter where perturbations
are: {PPO, MAPPO} x {ice, stairs} on the arms robot at N=4, depth on, with
`joint_deviation_arms` weighted 0. Ice is out -- B3's patches were never under
the robots (`scripts/bench/ice_placement_probe.py`) -- so this is stairs alone
until they are.

Built exactly as the biped depth rung is built on top of `BipedBumpyEnvCfg`:
upstream's observation groups with the same pooled depth term appended to policy
and critic, the same camera on the base, the same stairs menu. The humanoid and
biped observation configs differ only in their joint lists, so the depth term
lands in the same place and is 256 wide in both.

**The arm-deviation ablation lives in the task, not in a trainer flag.** The
first block ablated it only on the skrl path (`--ablate-arm-deviation`), and its
rsl-rl PPO control silently kept the penalty at every seed. Baking the ablation
in here means PPO and MAPPO rows cannot differ on it, and `__post_init__`
refuses to build if either term it should clear is not there to clear -- an
ablation that finds nothing is the failure the work order warned about.
Upstream names the terms `joint_deviation_shoulder` and `joint_deviation_elbow`;
there is no `joint_deviation_arms`.
"""

from __future__ import annotations

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

import berkeley_humanoid_lite.tasks.locomotion.velocity.mdp as mdp
from berkeley_humanoid_lite.tasks.locomotion.velocity.config.humanoid.env_cfg import (
    ObservationsCfg as HumanoidObservationsCfg,
)

from bhl_robust.tasks.arms_env_cfg import HumanoidBumpyEnvCfg
from bhl_robust.tasks.depth_env_cfg import depth_obs, make_depth_camera_cfg
from bhl_robust.tasks.limb_marl import ablate_arm_deviation

#: What must be cleared for the ablation to count.
ARM_DEVIATION_REQUIRED = {"joint_deviation_shoulder", "joint_deviation_elbow"}


@configclass
class HumanoidDepthObservationsCfg(HumanoidObservationsCfg):
    """Upstream's humanoid groups with the depth rung's term appended to both."""

    @configclass
    class PolicyCfg(HumanoidObservationsCfg.PolicyCfg):
        depth = ObsTerm(
            func=depth_obs,
            params={"sensor_cfg": SceneEntityCfg("depth_cam"), "pool": 4},
            noise=GaussianNoiseCfg(mean=0.0, std=0.0033),
        )

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class HumanoidDepthNoArmDevEnvCfg(HumanoidBumpyEnvCfg):
    """22 DoF on the bumpy menu, forward depth camera, arm deviation ablated."""

    observations: HumanoidDepthObservationsCfg = HumanoidDepthObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.depth_cam = make_depth_camera_cfg(res=64)
        cleared = set(ablate_arm_deviation(self))
        if not ARM_DEVIATION_REQUIRED <= cleared:
            raise RuntimeError(
                f"arm-deviation ablation cleared {sorted(cleared)}, needed "
                f"{sorted(ARM_DEVIATION_REQUIRED)}: refusing to build a Tier 3 task "
                f"with the penalty still on")


@configclass
class HumanoidStairsDepthEnvCfg(HumanoidDepthNoArmDevEnvCfg):
    """Tier 3 stairs: same risers as the biped stairs rung (`STAIRS_TERRAINS_CFG`)."""

    def __post_init__(self):
        super().__post_init__()
        from bhl_robust.terrains.stairs import STAIRS_TERRAINS_CFG
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG
