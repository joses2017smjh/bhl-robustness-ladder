"""Static scene geometry for the redesigned tasks: plinths, shelf, net, wall.

All of it is `AssetBaseCfg` with a static collider rather than `RigidObjectCfg`.
Nothing here is meant to move, and a static body costs the solver nothing per
step, which matters at 4,096 envs where the furniture would otherwise be four
thousand extra dynamic bodies.

Heights come from `bhl_robust.reach_band`, which is measured rather than chosen:
a payload centre at `GRASP_Z = 0.30 m` needs a 15.5 cm squat to reach and is
unreachable from the 41 cm collapse the old policies learned. The plinth is
whatever height puts the payload centre there.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg

from bhl_robust.reach_band import GRASP_Z

#: Furniture is visually distinct from the payload so a clip reads without a key.
FURNITURE_RGB = (0.32, 0.33, 0.36)
TARGET_RGB = (0.18, 0.55, 0.35)

_STATIC = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=True,
    kinematic_enabled=True,
)


def _box(prim: str, size: tuple[float, float, float],
         pos: tuple[float, float, float], rgb=FURNITURE_RGB,
         rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)) -> AssetBaseCfg:
    """One static, collidable box."""
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=_STATIC,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=PreviewSurfaceCfg(diffuse_color=rgb),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9, dynamic_friction=0.8, restitution=0.0),
        ),
    )


def plinth(payload_half_height: float, top: float = 0.34,
           prim: str = "plinth") -> AssetBaseCfg:
    """Pedestal that stands a payload's centre at `GRASP_Z`.

    `top` is the footprint, wide enough that the payload rests stably but not so
    wide that the robots have to reach across it -- the pinch happens on the
    payload's sides, so the plinth must not be broader than the payload.
    """
    h = GRASP_Z - payload_half_height
    return _box(prim, (top, top, h), (0.0, 0.0, h / 2.0))


def shelf(slot: float, deck_z: float, x: float,
          wall_t: float = 0.04, depth: float = 0.40) -> list[AssetBaseCfg]:
    """A shelf with one slot, built from a deck, two cheeks and a back.

    `slot` is the clear width and height of the opening. The cube task sets it
    6 cm larger than the cube, which is a placement problem rather than a
    throwing one -- the payload cannot be flung at a hole it barely fits.
    """
    c = slot / 2.0 + wall_t / 2.0
    return [
        _box("shelf_deck", (depth, slot + 2 * wall_t, wall_t),
             (x, 0.0, deck_z - wall_t / 2.0)),
        _box("shelf_left", (depth, wall_t, slot), (x, +c, deck_z + slot / 2.0)),
        _box("shelf_right", (depth, wall_t, slot), (x, -c, deck_z + slot / 2.0)),
        _box("shelf_back", (wall_t, slot + 2 * wall_t, slot + wall_t),
             (x + depth / 2.0, 0.0, deck_z + slot / 2.0)),
        _box("shelf_top", (depth, slot + 2 * wall_t, wall_t),
             (x, 0.0, deck_z + slot + wall_t / 2.0)),
    ]


def net(mouth: float, rim_z: float, x: float,
        wall_t: float = 0.03, depth: float = 0.45) -> list[AssetBaseCfg]:
    """An open-topped box on a stand: a target a ball can only enter from above.

    Four sides and a floor, no lid. Entry is checked as a volume test rather
    than by contact, so the net does not need to be a cloth -- what is being
    measured is whether the pair can put a ball in a place, not whether the
    simulator can model netting.
    """
    h = depth
    c = mouth / 2.0 + wall_t / 2.0
    base = rim_z - h
    return [
        _box("net_floor", (mouth, mouth, wall_t), (x, 0.0, base), TARGET_RGB),
        _box("net_xp", (wall_t, mouth, h), (x + c, 0.0, base + h / 2.0), TARGET_RGB),
        _box("net_xn", (wall_t, mouth, h), (x - c, 0.0, base + h / 2.0), TARGET_RGB),
        _box("net_yp", (mouth, wall_t, h), (x, +c, base + h / 2.0), TARGET_RGB),
        _box("net_yn", (mouth, wall_t, h), (x, -c, base + h / 2.0), TARGET_RGB),
        # Post, so the net is not floating.
        _box("net_post", (0.08, 0.08, base), (x, 0.0, base / 2.0)),
    ]


def wall(x: float, width: float = 2.4, height: float = 1.6,
         thickness: float = 0.10) -> AssetBaseCfg:
    """A flat vertical surface to lean a plank against."""
    return _box("wall", (thickness, width, height), (x, 0.0, height / 2.0))
