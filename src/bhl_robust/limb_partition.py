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
JOINTS = ARM_JOINTS + LEG_JOINTS
N_JOINTS = len(JOINTS)          # 22


def _idx(prefix: str) -> list[int]:
    return [i for i, j in enumerate(JOINTS) if j.startswith(prefix)]


#: One agent per limb. 5 / 5 / 6 / 6, summing to 22.
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


def validate(partition: dict[str, list[int]]) -> None:
    """Raise unless the partition covers 0..21 exactly once."""
    seen: list[int] = []
    for idx in partition.values():
        seen.extend(idx)
    if sorted(seen) != list(range(N_JOINTS)):
        missing = sorted(set(range(N_JOINTS)) - set(seen))
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
