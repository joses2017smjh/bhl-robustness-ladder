"""Oracle observation layout for cloth-sort.

Privileged simulator state is labeled as such. The first milestone uses this
oracle so manipulation physics can be validated without a perception stack.
A later sensor rung replaces ``garment_xy`` with a perception estimate; the
rest of the vector stays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bhl_robust.cloth.garments import SEMANTIC_CLASSES
from bhl_robust.limb_partition import JOINTS_22

N_JOINTS = len(JOINTS_22)
N_BASKETS = 3
N_CLASSES = len(SEMANTIC_CLASSES)


@dataclass(frozen=True)
class ObservationSpec:
    """Widths of each named block, in order."""

    n_garments: int
    privileged: bool = True

    @property
    def proprio_size(self) -> int:
        # joint_pos + joint_vel + base quat + base ang vel + two hand xyz
        return N_JOINTS + N_JOINTS + 4 + 3 + 6

    @property
    def task_size(self) -> int:
        n = self.n_garments
        # xy + yaw + class one-hot + rel-to-correct-basket + selected + sorted
        return n * 2 + n + n * N_CLASSES + n * 2 + n + n + N_BASKETS * 2 + 1

    @property
    def size(self) -> int:
        return self.proprio_size + self.task_size

    @property
    def labels(self) -> tuple[str, ...]:
        return (
            "joint_pos",
            "joint_vel",
            "base_quat",
            "base_ang_vel",
            "hand_pos",
            "garment_xy",          # privileged in oracle mode
            "garment_yaw",
            "garment_class",
            "rel_garment_basket",
            "selected_garment",
            "sorted_mask",
            "basket_xy",
            "progress",
        )


def pack_oracle(
    *,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    base_quat: np.ndarray,
    base_ang_vel: np.ndarray,
    hand_pos: np.ndarray,
    garment_xy: np.ndarray,
    garment_yaw: np.ndarray,
    class_index: np.ndarray,
    rel_xy: np.ndarray,
    selected: int,
    sorted_mask: np.ndarray,
    basket_xy: np.ndarray,
    progress: float,
    spec: ObservationSpec,
) -> np.ndarray:
    """Concatenate the oracle vector. All inputs are 1-D except the per-garment
    arrays, which are ``(n_garments, ...)``.
    """
    n = spec.n_garments
    class_oh = np.zeros((n, N_CLASSES), dtype=float)
    class_oh[np.arange(n), np.asarray(class_index, dtype=int)] = 1.0
    selected_oh = np.zeros(n, dtype=float)
    selected_oh[int(np.clip(selected, 0, n - 1))] = 1.0
    parts = [
        np.asarray(joint_pos, dtype=float).reshape(N_JOINTS),
        np.asarray(joint_vel, dtype=float).reshape(N_JOINTS),
        np.asarray(base_quat, dtype=float).reshape(4),
        np.asarray(base_ang_vel, dtype=float).reshape(3),
        np.asarray(hand_pos, dtype=float).reshape(6),
        np.asarray(garment_xy, dtype=float).reshape(n, 2).ravel(),
        np.asarray(garment_yaw, dtype=float).reshape(n),
        class_oh.ravel(),
        np.asarray(rel_xy, dtype=float).reshape(n, 2).ravel(),
        selected_oh,
        np.asarray(sorted_mask, dtype=float).reshape(n),
        np.asarray(basket_xy, dtype=float).reshape(N_BASKETS, 2).ravel(),
        np.array([progress], dtype=float),
    ]
    obs = np.concatenate(parts)
    if obs.size != spec.size:
        raise ValueError(f"packed obs has {obs.size} entries, spec says {spec.size}")
    return obs
