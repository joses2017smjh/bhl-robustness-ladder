"""Domain-randomisation ranges for Stage 1 rigid-proxy training.

Ranges live here rather than as literals in the env so a sweep can change
them without editing task logic. The point is to stop a rigid-proxy policy
from exploiting one exact cuboid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DomainRandomization:
    mass_scale: tuple[float, float] = (0.70, 1.40)
    size_scale: tuple[float, float] = (0.85, 1.20)
    friction: tuple[float, float] = (0.35, 0.90)
    restitution: tuple[float, float] = (0.00, 0.05)
    spawn_xy_jitter: float = 0.06
    yaw_range: tuple[float, float] = (-0.60, 0.60)
    basket_xy_jitter: float = 0.03
    contact_offset_jitter: float = 0.03
    surface_friction: tuple[float, float] = (0.40, 0.85)

    def sample(self, rng: np.random.Generator) -> dict[str, float]:
        def u(lo_hi: tuple[float, float]) -> float:
            return float(rng.uniform(lo_hi[0], lo_hi[1]))

        return {
            "mass_scale": u(self.mass_scale),
            "size_scale": u(self.size_scale),
            "friction": u(self.friction),
            "restitution": u(self.restitution),
            "spawn_dx": float(rng.uniform(-self.spawn_xy_jitter, self.spawn_xy_jitter)),
            "spawn_dy": float(rng.uniform(-self.spawn_xy_jitter, self.spawn_xy_jitter)),
            "yaw": u(self.yaw_range),
            "basket_dx": float(rng.uniform(-self.basket_xy_jitter, self.basket_xy_jitter)),
            "basket_dy": float(rng.uniform(-self.basket_xy_jitter, self.basket_xy_jitter)),
            "contact_offset": float(rng.uniform(
                -self.contact_offset_jitter, self.contact_offset_jitter
            )),
            "surface_friction": u(self.surface_friction),
        }


#: Off: every garment keeps its catalog mass/size/friction. Used by C0 so the
#: scripted baseline is scored on the geometry, not on a random draw.
IDENTITY = DomainRandomization(
    mass_scale=(1.0, 1.0),
    size_scale=(1.0, 1.0),
    friction=(0.60, 0.60),
    restitution=(0.0, 0.0),
    spawn_xy_jitter=0.0,
    yaw_range=(0.0, 0.0),
    basket_xy_jitter=0.0,
    contact_offset_jitter=0.0,
    surface_friction=(0.60, 0.60),
)
