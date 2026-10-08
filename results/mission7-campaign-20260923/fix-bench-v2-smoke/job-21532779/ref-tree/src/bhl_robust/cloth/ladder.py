"""C0–C5 experiment ladder.

Cheap simulation for learning, expensive simulation for validation. Nothing
in this table is a result; it is the progression. Measured cells get numbers
from ``scripts/cloth/`` and are written into ``docs/CLOTH_SORT.md``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rung:
    id: str
    physics: str
    policy: str
    n_garments: int
    n_baskets: int
    target_num_envs: tuple[int, int]
    purpose: str
    deformable: bool
    notes: str = ""


LADDER: tuple[Rung, ...] = (
    Rung(
        "C0", "rigid_proxy", "scripted", 1, 1, (1, 64),
        "Verify task geometry and the sweep controller.",
        False,
        "Must pass before any training. Failure is a controller/layout bug.",
    ),
    Rung(
        "C1", "rigid_proxy", "learned", 1, 1, (1024, 2048),
        "Train a high-level sweep strategy at rigid-body throughput.",
        False,
        "Action is 5-D sweep parameters, not 22-DoF joint commands.",
    ),
    Rung(
        "C2", "deformable_lowres", "scripted", 1, 1, (8, 32),
        "Does the sweep primitive physically transfer onto cloth?",
        True,
        "If this fails, the physics/controller interaction is the problem, not RL.",
    ),
    Rung(
        "C3", "deformable_lowres", "learned_zero_shot", 1, 1, (8, 32),
        "Zero-shot rigid-to-deformable transfer of the C1 policy.",
        True,
        "transfer_gap = C1_success - C3_success. Do not fine-tune if this is already good.",
    ),
    Rung(
        "C4", "deformable_lowres", "adapted", 1, 1, (8, 32),
        "BC / residual / small RL fine-tune, only if C3 transfer is poor.",
        True,
        "Cost-gated. No 8,000-iteration deformable job without a CostReport.",
    ),
    Rung(
        "C5", "five_garment_eval", "best", 5, 3, (1, 8),
        "Evaluate complete sorting. Evaluation and rendering, not RL.",
        True,
        "Mode A: five active deformables. Mode B: one active, others frozen/rigid. "
        "Mode B must not be reported as five simultaneous cloth objects.",
    ),
)

LADDER_BY_ID = {r.id: r for r in LADDER}

ACTIVE_GARMENT_MODE = "B"
ACTIVE_GARMENT_WARNING = (
    "Stage 3 Mode B evaluates one active deformable at a time. "
    "That is not five simultaneously active cloth objects."
)
