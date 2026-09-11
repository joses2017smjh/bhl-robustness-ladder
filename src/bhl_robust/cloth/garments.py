"""The five-garment set and the three baskets they sort into.

Identities match the original design in ``docs/CLOTH_SORT.md``: two socks, two
shirts, one jacket. Three baskets, one per semantic class, so at least one
basket takes two items and a policy cannot succeed by memorising a 1-1 layout.

These are *specifications*, not assets. Rigid-proxy, kinematic, and deformable
stages all read the same catalog; only the physics representation changes.
"""

from __future__ import annotations

from dataclasses import dataclass

BASKET_SOCKS = "socks"
BASKET_SHIRTS = "shirts"
BASKET_JACKETS = "jackets"

BASKET_IDS: tuple[str, ...] = (BASKET_SOCKS, BASKET_SHIRTS, BASKET_JACKETS)

CLASS_TO_BASKET = {
    "sock": BASKET_SOCKS,
    "shirt": BASKET_SHIRTS,
    "jacket": BASKET_JACKETS,
}

SEMANTIC_CLASSES: tuple[str, ...] = ("sock", "shirt", "jacket")


@dataclass(frozen=True)
class GarmentSpec:
    """One garment identity, shared across rigid / deformable / kinematic."""

    name: str
    semantic_class: str
    target_basket: str
    #: Axis-aligned size of the rigid proxy (metres). Thin in z: a flattened
    #: rectangle, not a cube pretending to be cloth.
    proxy_size: tuple[float, float, float]
    mass: float
    rgb: tuple[float, float, float]
    friction: float
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.semantic_class not in CLASS_TO_BASKET:
            raise ValueError(f"unknown class {self.semantic_class!r}")
        if self.target_basket != CLASS_TO_BASKET[self.semantic_class]:
            raise ValueError(
                f"{self.name}: class {self.semantic_class} maps to "
                f"{CLASS_TO_BASKET[self.semantic_class]}, not {self.target_basket}"
            )


#: Representative set. Sizes stay well under the 0.355 m hand span so a sweep
#: contact is a push, not a pinch the morphology cannot perform.
GARMENTS: tuple[GarmentSpec, ...] = (
    GarmentSpec("sock_a", "sock", BASKET_SOCKS, (0.18, 0.10, 0.020), 0.04,
                (0.20, 0.40, 0.75), 0.60),
    GarmentSpec("sock_b", "sock", BASKET_SOCKS, (0.18, 0.10, 0.020), 0.04,
                (0.30, 0.50, 0.85), 0.60),
    GarmentSpec("shirt_a", "shirt", BASKET_SHIRTS, (0.28, 0.22, 0.020), 0.12,
                (0.75, 0.22, 0.18), 0.55),
    GarmentSpec("shirt_b", "shirt", BASKET_SHIRTS, (0.28, 0.22, 0.020), 0.12,
                (0.85, 0.40, 0.16), 0.55),
    GarmentSpec("jacket", "jacket", BASKET_JACKETS, (0.34, 0.26, 0.025), 0.22,
                (0.16, 0.16, 0.20), 0.50),
)

GARMENT_BY_NAME = {g.name: g for g in GARMENTS}


def class_index(semantic_class: str) -> int:
    return SEMANTIC_CLASSES.index(semantic_class)


def basket_index(basket_id: str) -> int:
    return BASKET_IDS.index(basket_id)


def one_garment(name: str = "shirt_a") -> tuple[GarmentSpec, ...]:
    """Stage 1 / Stage 2 default: a single garment, one target."""
    return (GARMENT_BY_NAME[name],)


def sibling_index(spec: GarmentSpec, catalog: tuple[GarmentSpec, ...] | None = None) -> tuple[int, int]:
    """(index among same-basket items, how many share that basket)."""
    catalog = catalog if catalog is not None else GARMENTS
    sibs = [g for g in catalog if g.target_basket == spec.target_basket]
    return sibs.index(spec), len(sibs)
