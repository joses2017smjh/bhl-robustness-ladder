"""The robot with its grippers modelled, as an opt-in overlay.

The shipped asset welds both hands (`docs/GRIPPER.md`). This adds the two DoF
the hardware actually has, without touching `external/` and without changing the
asset every existing result was produced against.

**Opt-in, not a replacement.** `HUMANOID_LITE_CFG` keeps its 22 joints, so every
published number stays reproducible and every 194-wide checkpoint still loads.
`HUMANOID_LITE_GRIPPER_CFG` is a separate 24-joint articulation, and a task
chooses one. Swapping the default would silently invalidate the whole
manipulation section rather than superseding it, and a before/after on the same
tasks is the only way to show what the gripper bought.

Gripper joints are **appended** to the joint order rather than interleaved, so
indices 0..21 keep their meaning. That is what lets the MuJoCo replay load an
old checkpoint against the first 22 joints of a 24-joint model.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite import (
    HUMANOID_LITE_ARM_JOINTS,
    HUMANOID_LITE_CFG,
    HUMANOID_LITE_LEG_JOINTS,
)
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

#: Where `scripts/add_gripper.py` and the USD conversion put their output.
GRIPPER_ASSET_DIR = os.environ.get(
    "BHL_GRIPPER_ASSET_DIR",
    "/nfs/hpc/share/sanchej7/Humanoid_Lite/assets/gripper",
)
#: The converter nests its output one directory deep and writes `.usda`, not
#: the `.usd` filename it is handed -- so the obvious path is wrong and
#: `get_gripper_cfg()` would have raised FileNotFoundError on a conversion that
#: had in fact succeeded. Resolved by search rather than by assumption.
def _find_gripper_usd() -> str:
    root = os.path.join(GRIPPER_ASSET_DIR, "usd")
    for ext in (".usda", ".usd"):
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.startswith("berkeley_humanoid_lite_gripper") and f.endswith(ext):
                    return os.path.join(dirpath, f)
    return os.path.join(root, "berkeley_humanoid_lite_gripper.usd")


GRIPPER_USD = _find_gripper_usd()

GRIPPER_JOINTS = ["arm_left_gripper_joint", "arm_right_gripper_joint"]

#: Appended, so 0..21 are unchanged.
HUMANOID_LITE_GRIPPER_JOINT_ORDER = (
    HUMANOID_LITE_ARM_JOINTS + HUMANOID_LITE_LEG_JOINTS + GRIPPER_JOINTS
)

#: The driver documents 0.2 open / 0.85 closed on a raw scale it reaches from a
#: [0, 1] target. In joint space that is 0 rad open to 1.2 rad closed, and the
#: action is scaled onto that range so a policy's [-1, 1] output maps the way the
#: hardware's does.
GRIPPER_OPEN_RAD = 0.0
GRIPPER_CLOSED_RAD = 1.20

#: Smaller motor than the arms, which are 4 Nm. Deliberately not generous: a
#: gripper that can apply arbitrary force would make the grasp trivially easy
#: and the result meaningless.
GRIPPER_EFFORT = 2.0
GRIPPER_STIFFNESS = 20.0
GRIPPER_DAMPING = 1.0


def _gripper_actuator() -> ImplicitActuatorCfg:
    return ImplicitActuatorCfg(
        joint_names_expr=["arm_.*_gripper_joint"],
        effort_limit=GRIPPER_EFFORT,
        velocity_limit=6.0,
        stiffness=GRIPPER_STIFFNESS,
        damping=GRIPPER_DAMPING,
    )


def make_gripper_cfg(usd_path: str | None = None) -> ArticulationCfg:
    """`HUMANOID_LITE_CFG` with the two hand DoF restored."""
    cfg = HUMANOID_LITE_CFG.copy()
    cfg.spawn = cfg.spawn.replace(usd_path=usd_path or GRIPPER_USD)
    # Start open. A robot that spawns with its hands shut has to learn to let go
    # before it can learn to grasp, which is a detour with no research content.
    joint_pos = dict(cfg.init_state.joint_pos)
    for j in GRIPPER_JOINTS:
        joint_pos[j] = GRIPPER_OPEN_RAD
    cfg.init_state = cfg.init_state.replace(joint_pos=joint_pos)
    actuators = dict(cfg.actuators)
    actuators["grippers"] = _gripper_actuator()
    cfg.actuators = actuators
    return cfg


HUMANOID_LITE_GRIPPER_CFG = None
"""Built lazily by `get_gripper_cfg()`; constructing it at import time would
require the USD to exist, and this module is imported by things that only want
the joint names."""


def get_gripper_cfg() -> ArticulationCfg:
    global HUMANOID_LITE_GRIPPER_CFG
    if HUMANOID_LITE_GRIPPER_CFG is None:
        if not os.path.isfile(GRIPPER_USD):
            raise FileNotFoundError(
                f"gripper USD not found at {GRIPPER_USD}. Run "
                "scripts/add_gripper.py and the USD conversion "
                "(slurm/inner/_gripper_usd.sh) first."
            )
        HUMANOID_LITE_GRIPPER_CFG = make_gripper_cfg()
    return HUMANOID_LITE_GRIPPER_CFG
