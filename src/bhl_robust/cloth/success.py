"""Geometric sorting success. Not a visual judgement.

Rigid proxies are scored on centre-of-mass. Deformables are scored on the
fraction of vertices inside the correct basket. A garment in the wrong basket
is a distinct outcome from "not yet sorted" — those two must not be collapsed
into a single failure bit or the wrong-basket penalty cannot fire.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bhl_robust.cloth.garments import GarmentSpec
from bhl_robust.cloth.layout import (
    DEFORMABLE_SUCCESS_FRACTION,
    RIGID_SUCCESS_FRACTION,
    all_basket_aabbs,
    basket_aabb,
)


@dataclass(frozen=True)
class SortOutcome:
    """Per-garment result of one geometric query."""

    correct: bool
    wrong: bool
    basket_id: str | None
    fraction_in_correct: float
    fraction_in_wrong: float


def classify_points(
    points: np.ndarray,
    spec: GarmentSpec,
    *,
    threshold: float | None = None,
    deformable: bool = False,
) -> SortOutcome:
    """Assign a garment's points to correct / wrong / neither.

    ``threshold`` defaults to the rigid or deformable constant matching
    ``deformable``. Passing it explicitly is what makes the threshold
    configurable without a rewrite.
    """
    if threshold is None:
        threshold = DEFORMABLE_SUCCESS_FRACTION if deformable else RIGID_SUCCESS_FRACTION
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    correct_box = basket_aabb(spec.target_basket)
    in_correct = np.all((pts >= correct_box.low) & (pts <= correct_box.high), axis=1)
    frac_correct = float(in_correct.mean())

    frac_wrong = 0.0
    wrong_id: str | None = None
    for bid, box in all_basket_aabbs().items():
        if bid == spec.target_basket:
            continue
        inside = np.all((pts >= box.low) & (pts <= box.high), axis=1)
        frac = float(inside.mean())
        if frac > frac_wrong:
            frac_wrong = frac
            wrong_id = bid

    correct = frac_correct >= threshold
    wrong = (not correct) and frac_wrong >= threshold
    return SortOutcome(
        correct=correct,
        wrong=wrong,
        basket_id=spec.target_basket if correct else (wrong_id if wrong else None),
        fraction_in_correct=frac_correct,
        fraction_in_wrong=frac_wrong,
    )


def rigid_outcome(com: np.ndarray, spec: GarmentSpec,
                  threshold: float = RIGID_SUCCESS_FRACTION) -> SortOutcome:
    """Centre-of-mass test used by Stage 1."""
    return classify_points(com, spec, threshold=threshold, deformable=False)


def deformable_outcome(vertices: np.ndarray, spec: GarmentSpec,
                       threshold: float = DEFORMABLE_SUCCESS_FRACTION) -> SortOutcome:
    """Vertex-fraction test used by Stage 2 / Stage 3."""
    return classify_points(vertices, spec, threshold=threshold, deformable=True)
