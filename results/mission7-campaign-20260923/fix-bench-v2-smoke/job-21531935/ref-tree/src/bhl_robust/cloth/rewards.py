"""Interpretable reward components for the high-level sweep policy.

Each term is computed separately so a TensorBoard (or CSV) log can show which
one is carrying the return. Shaping stays simple: progress, success, wrong
basket, efficiency, safety, time.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class RewardBreakdown:
    progress: float = 0.0
    success: float = 0.0
    wrong_basket: float = 0.0
    sweep_efficiency: float = 0.0
    fall: float = 0.0
    invalid: float = 0.0
    time: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.progress + self.success + self.wrong_basket
            + self.sweep_efficiency + self.fall + self.invalid + self.time
        )

    def as_dict(self) -> dict[str, float]:
        out = {f.name: float(getattr(self, f.name)) for f in fields(self)}
        out["total"] = float(self.total)
        return out


@dataclass(frozen=True)
class RewardWeights:
    progress: float = 1.0
    success: float = 5.0
    wrong_basket: float = -2.0
    sweep_efficiency: float = -0.15
    fall: float = -4.0
    invalid: float = -0.5
    time: float = -0.05


def compute_reward(
    *,
    prev_dist: float,
    curr_dist: float,
    newly_correct: bool,
    newly_wrong: bool,
    sweep_distance: float,
    min_useful_distance: float,
    fell: bool,
    invalid: bool,
    weights: RewardWeights | None = None,
) -> RewardBreakdown:
    """One macro-step of reward.

    ``prev_dist`` / ``curr_dist`` are planar distances from the active garment
    to its correct basket. Progress is the reduction, so moving the wrong way
    costs.
    """
    w = weights or RewardWeights()
    extra = max(sweep_distance - min_useful_distance, 0.0)
    return RewardBreakdown(
        progress=w.progress * (prev_dist - curr_dist),
        success=w.success if newly_correct else 0.0,
        wrong_basket=w.wrong_basket if newly_wrong else 0.0,
        sweep_efficiency=w.sweep_efficiency * extra,
        fall=w.fall if fell else 0.0,
        invalid=w.invalid if invalid else 0.0,
        time=w.time,
    )
