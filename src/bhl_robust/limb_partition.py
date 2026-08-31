"""How the 22 DoF split into limb agents, and the guarantee that it is a split.

Kept free of Isaac imports on purpose -- `tasks/__init__.py` registers gym ids
and pulls in isaaclab, which aborts unless SimulationApp ran first, and G-B4
needs to check this arithmetic without booting a simulator. Same reasoning as
`reach_band`.

The joint order is the one the observation and action vectors were built in:
`HUMANOID_LITE_JOINTS = ARM_JOINTS + LEG_JOINTS`, with every coop observation
term passing `preserve_order=True`. So these are index slices into a fixed
layout, not a remapping, and `reassemble` is exactly the inverse of `split`.

The 22 joints here are the **arms** tasks (`Velocity-BHL-Arms-*`), not the
biped ones. The biped overlays actuate 12 leg joints only, and pointing a
four-limb partition at one of those fails with `Invalid action shape, expected:
12, received: 22` -- which is how G-B4 found it, on its first online run.

That inverse property is the whole point. If the four agents' actions do not
reassemble into precisely the vector a single-agent policy would have emitted,
then MAPPO is not being compared against PPO -- it is being compared against a
different robot, and every row in the Tier 1 table would be measuring the
partition rather than the algorithm.
"""

from __future__ import annotations

ARM_JOINTS = [
    "arm_left_shoulder_pitch_joint", "arm_left_shoulder_roll_joint",
    "arm_left_shoulder_yaw_joint", "arm_left_elbow_pitch_joint",
    "arm_left_elbow_roll_joint",
    "arm_right_shoulder_pitch_joint", "arm_right_shoulder_roll_joint",
    "arm_right_shoulder_yaw_joint", "arm_right_elbow_pitch_joint",
    "arm_right_elbow_roll_joint",
]
LEG_JOINTS = [
    "leg_left_hip_roll_joint", "leg_left_hip_yaw_joint",
    "leg_left_hip_pitch_joint", "leg_left_knee_pitch_joint",
    "leg_left_ankle_pitch_joint", "leg_left_ankle_roll_joint",
    "leg_right_hip_roll_joint", "leg_right_hip_yaw_joint",
    "leg_right_hip_pitch_joint", "leg_right_knee_pitch_joint",
    "leg_right_ankle_pitch_joint", "leg_right_ankle_roll_joint",
]

#: One per hand. Absent from the shipped asset, which welds both hands shut --
#: see `docs/GRIPPER.md`. Appended rather than interleaved so the first 22
#: indices keep their meaning and a 22-DoF checkpoint still maps onto the same
#: joints it was trained on.
GRIPPER_JOINTS = ["arm_left_gripper_joint", "arm_right_gripper_joint"]

#: The layout the welded-hand results were produced against. Kept because every
#: published manipulation number in this repo is indexed by it, and a checkpoint
#: cannot be replayed against a joint list it never saw.
JOINTS_22 = ARM_JOINTS + LEG_JOINTS

JOINTS = ARM_JOINTS + LEG_JOINTS + GRIPPER_JOINTS
N_JOINTS = len(JOINTS)          # 24


def _idx(prefix: str) -> list[int]:
    return [i for i, j in enumerate(JOINTS) if j.startswith(prefix)]


#: One agent per limb. 6 / 6 / 6 / 6, summing to 24 -- each arm agent owns its
#: own gripper, which is the only assignment that makes sense: a limb that can
#: reach for something and not close on it is not an agent for that limb.
LIMB4 = {
    "arm_left": _idx("arm_left"),
    "arm_right": _idx("arm_right"),
    "leg_left": _idx("leg_left"),
    "leg_right": _idx("leg_right"),
}

#: The ablation: upper body against lower body. This is the credit-assignment
#: boundary section 5 already characterised -- the lift lives entirely in the
#: arms while the legs do the standing -- so a 2-way split tests whether that
#: separation is what a limb factorisation is actually buying.
LIMB2 = {
    "arms": _idx("arm_"),
    "legs": _idx("leg_"),
}

PARTITIONS = {"limb4": LIMB4, "limb2": LIMB2}


def validate(partition: dict[str, list[int]], n_dof: int | None = None) -> None:
    """Raise unless the partition covers 0..n_dof-1 exactly once.

    `n_dof` defaults to the partition's own span rather than to `N_JOINTS`.
    Both layouts are live -- 22 with welded hands, 24 with grippers -- so
    validating everything against the larger one reports the 22-DoF partition
    as missing joints 22 and 23, which it is supposed to be missing.
    """
    seen: list[int] = []
    for idx in partition.values():
        seen.extend(idx)
    if n_dof is None:
        n_dof = len(seen)
    if sorted(seen) != list(range(n_dof)):
        missing = sorted(set(range(n_dof)) - set(seen))
        dupes = sorted({i for i in seen if seen.count(i) > 1})
        raise ValueError(
            f"partition is not a partition of {N_JOINTS} joints: "
            f"missing={missing} duplicated={dupes}"
        )


def split(action, partition: dict[str, list[int]]) -> dict:
    """One (N, 22) action tensor -> {agent: (N, k)}."""
    return {name: action[..., idx] for name, idx in partition.items()}


def reassemble(actions: dict, partition: dict[str, list[int]]):
    """{agent: (N, k)} -> one (N, 22) tensor, in the canonical joint order.

    Scatters rather than concatenates. Concatenating the agents' slices in dict
    order only happens to be right when the partition is contiguous and ordered,
    which `limb4` is and `limb2` is not -- `arms` and `legs` interleave in
    neither, but a future partition (say, one agent per actuator group) could
    easily be non-contiguous and would then silently permute the robot's joints.
    """
    any_a = next(iter(actions.values()))
    out = any_a.new_zeros((*any_a.shape[:-1], N_JOINTS))
    for name, idx in partition.items():
        out[..., idx] = actions[name]
    return out


def agent_names(kind: str) -> list[str]:
    return list(PARTITIONS[kind].keys())


def action_widths(kind: str) -> dict[str, int]:
    return {k: len(v) for k, v in PARTITIONS[kind].items()}

def _idx_in(names: list[str], prefix: str) -> list[int]:
    return [i for i, j in enumerate(names) if j.startswith(prefix)]


def partition_for(kind: str, n_dof: int) -> dict[str, list[int]]:
    """The named partition, sized to the asset actually in use.

    Both joint layouts are live: the welded-hand asset is 22 DoF and the gripper
    asset is 24, and a task chooses one. A wrapper that assumed 24 asked a
    22-DoF env for 24 actions and got `Invalid action shape, expected: 22,
    received: 24` -- so the partition is derived from the env's own action
    width rather than from a module constant.
    """
    if n_dof == N_JOINTS:
        names = JOINTS
    elif n_dof == len(JOINTS_22):
        names = JOINTS_22
    else:
        raise ValueError(
            f"no joint layout with {n_dof} DoF; known are "
            f"{len(JOINTS_22)} (welded hands) and {N_JOINTS} (grippers)"
        )
    if kind == "limb4":
        part = {
            "arm_left": _idx_in(names, "arm_left"),
            "arm_right": _idx_in(names, "arm_right"),
            "leg_left": _idx_in(names, "leg_left"),
            "leg_right": _idx_in(names, "leg_right"),
        }
    elif kind == "limb2":
        part = {"arms": _idx_in(names, "arm_"), "legs": _idx_in(names, "leg_")}
    else:
        raise ValueError(f"unknown partition {kind!r}")
    seen = sorted(i for v in part.values() for i in v)
    if seen != list(range(n_dof)):
        raise ValueError(f"{kind} does not partition {n_dof} DoF")
    return part


def reassemble_n(actions: dict, partition: dict[str, list[int]], n_dof: int):
    """`reassemble` for an explicit DoF count."""
    any_a = next(iter(actions.values()))
    out = any_a.new_zeros((*any_a.shape[:-1], n_dof))
    for name, idx in partition.items():
        out[..., idx] = actions[name]
    return out
