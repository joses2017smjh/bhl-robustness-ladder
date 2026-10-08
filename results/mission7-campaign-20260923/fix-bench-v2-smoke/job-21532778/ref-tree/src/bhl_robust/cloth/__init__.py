"""Cloth-sort experiment: rigid-proxy training, deformable validation.

Isaac-free on purpose. ``bhl_robust.tasks`` registers gym ids and pulls in
isaaclab, which aborts unless SimulationApp has started. Everything a login
node can check — garments, layout, sweep planning, success, rewards, the
kinematic C0/C1 env, the cost gate — lives here.

Isaac configs live in ``bhl_robust.tasks.cloth_sort_env_cfg``.
"""

from bhl_robust.cloth.cost import CostReport, guard_or_raise, report_cost
from bhl_robust.cloth.garments import BASKET_IDS, GARMENTS, GarmentSpec
from bhl_robust.cloth.ladder import ACTIVE_GARMENT_WARNING, LADDER
from bhl_robust.cloth.metrics import RunMetrics, transfer_gap
from bhl_robust.cloth.robot import EXPECTED_ARMS, EXPECTED_JOINTS, assert_articulation
from bhl_robust.cloth.sweep import ACTION_DIM, ACTION_NAMES

__all__ = [
    "ACTION_DIM",
    "ACTION_NAMES",
    "ACTIVE_GARMENT_WARNING",
    "BASKET_IDS",
    "CostReport",
    "EXPECTED_ARMS",
    "EXPECTED_JOINTS",
    "GARMENTS",
    "GarmentSpec",
    "LADDER",
    "RunMetrics",
    "assert_articulation",
    "guard_or_raise",
    "report_cost",
    "transfer_gap",
]
