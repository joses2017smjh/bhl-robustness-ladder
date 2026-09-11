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
         rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
         collider_only: bool = False) -> AssetBaseCfg:
    """One static, collidable box.

    ``collider_only`` spawns plain collision geometry: no ``rigid_props`` and
    no ``physics_material``. This is exactly the shape Isaac Lab's own Newton
    cloth example uses for its static ``cube``
    (``lift_franka_soft/franka_cloth_env_cfg.py``), and it is the only form
    Newton does not register as a physics body -- ``AssetBaseCfg`` otherwise
    dies with ``FrameView prim ... is a Newton physics body`` (21233868 and
    21233960; 21125073 before them). Dropping ``rigid_props`` alone was not
    enough: 21233960 still failed with the material bound.

    **Fidelity note.** A collider-only box has no explicit friction, so it
    takes the simulator default rather than 0.9/0.8. Contact friction combines
    the two surfaces and the garment keeps its own per-garment (and
    randomized) material, so the garment side still sets the behaviour. This
    is applied to the rigid scene as well as the deformable one, so C0/C1 and
    C2/C3 are not comparing different table surfaces.

    Default stays False so the maze, plinth, shelf and net keep the exact
    physics their published numbers were measured on.
    """
    spawn_kwargs = dict(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=PreviewSurfaceCfg(diffuse_color=rgb),
    )
    if not collider_only:
        spawn_kwargs["rigid_props"] = _STATIC
        spawn_kwargs["physics_material"] = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.9, dynamic_friction=0.8, restitution=0.0)
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
        spawn=sim_utils.CuboidCfg(**spawn_kwargs),
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

# ------------------------------------------------------------------ the maze

#: Corridor width. The robot's hand-to-hand span is 0.355 m and it oscillates
#: several centimetres a step, so a corridor has to clear the body with room for
#: a gait that was never trained to walk a line.
CORRIDOR_W = 0.90
WALL_H = 0.60
WALL_T = 0.05

ARROW_RGB = (0.90, 0.35, 0.10)
OBSTACLE_RGB = (0.20, 0.20, 0.24)
BUTTON_RGB = (0.15, 0.70, 0.35)


def corridor_walls(segments, prefix: str = "maze"):
    """Static walls from (x, y, length, axis) segments.

    Walls are 0.60 m tall: above the 0.34 m lidar plane so the scanner sees
    them, and low enough that the stereo pair, pitched 20 degrees down at
    0.30 m, still sees over them at range. Both sensors have to be useful or
    the task cannot separate them.
    """
    out = []
    for i, (x, y, length, axis) in enumerate(segments):
        size = (length, WALL_T, WALL_H) if axis == "x" else (WALL_T, length, WALL_H)
        out.append(_box(f"{prefix}_wall{i}", size, (x, y, WALL_H / 2)))
    return out


def arrow_marker(prim: str, pos: tuple[float, float, float], heading: str):
    """A junction sign: a flat plate whose long axis points the way to go.

    Deliberately geometric rather than a texture. This project cannot render
    RGB on the 5.1 stack, and a marker that only a colour camera can read would
    make the task unrunnable there. An oriented plate is legible to the stereo
    pair as depth structure and to nothing else in the observation, so vision
    still has to earn it.
    """
    if heading not in ("left", "right"):
        raise ValueError(f"heading must be 'left' or 'right', got {heading!r}")
    long_axis = (0.30, 0.04, 0.10)
    # Offset to the side it indicates: the sign of the y offset is the answer,
    # and it is readable as depth without needing colour.
    dy = 0.12 if heading == "left" else -0.12
    return _box(prim, long_axis, (pos[0], pos[1] + dy, pos[2]), rgb=ARROW_RGB)


def small_obstacle(prim: str, pos: tuple[float, float, float], size: float = 0.10):
    """A floor obstacle below the stereo pair's down-pitched view.

    0.10 m is above the foot but under the 0.34 m lidar plane, so this is the
    hazard the lidar sees and the cameras mostly do not -- the counterpart of
    the flush friction patch in the ice rung, where depth beat blind on
    something it could not see.
    """
    return _box(prim, (size, size, size), (pos[0], pos[1], size / 2), rgb=OBSTACLE_RGB)


def button(prim: str, pos: tuple[float, float, float]):
    """A wall-mounted plate to press. Non-prehensile, one contact point.

    Chosen because this morphology can do it: no fingers are required to push a
    plate, which is the same reasoning that made the cloth task a sweeping task
    rather than a picking one.
    """
    return _box(prim, (0.06, 0.16, 0.16), pos, rgb=BUTTON_RGB)


# ---------------------------------------------------------- cloth-sort scene
# Static table and open baskets. Geometry constants live in
# ``bhl_robust.cloth.layout`` so the kinematic env and this scene cannot drift
# apart; these helpers only spawn the colliders.

BASKET_RGB = {
    "socks": (0.20, 0.40, 0.75),
    "shirts": (0.75, 0.25, 0.18),
    "jackets": (0.20, 0.20, 0.22),
}


def cloth_table(prim: str = "table"):
    """Table whose top is at ``GRASP_Z``. Thin, wide enough for five garments.

    Collider-only in the rigid scene as well as the deformable one. Making it
    a rigid body under PhysX and a plain collider under Newton would put a
    physics difference between C0/C1 and C2/C3 and charge it to the cloth.
    """
    from bhl_robust.cloth import layout as L
    hx, hy = L.TABLE_SIZE_XY[0] / 2.0, L.TABLE_SIZE_XY[1] / 2.0
    h = L.TABLE_THICKNESS
    cx, cy = L.TABLE_CENTER_XY
    return _box(prim, (2 * hx, 2 * hy, h), (cx, cy, L.TABLE_TOP_Z - h / 2.0),
                collider_only=True)


def sorting_basket(basket_id: str, prim: str | None = None):
    """Open-topped box under the table's front edge. Same idea as ``net``."""
    from bhl_robust.cloth import layout as L
    prim = prim or f"basket_{basket_id}"
    inner = L.BASKET_INNER
    wall = L.BASKET_WALL
    x, y = L.BASKET_X, L.BASKET_Y[basket_id]
    rgb = BASKET_RGB.get(basket_id, TARGET_RGB)
    hx, hy, hz = inner[0] / 2.0, inner[1] / 2.0, inner[2]
    t = wall
    floor_z = t / 2.0
    mid_z = hz / 2.0
    return [
        _box(f"{prim}_floor", (inner[0] + 2 * t, inner[1] + 2 * t, t),
             (x, y, floor_z), rgb, collider_only=True),
        _box(f"{prim}_xp", (t, inner[1], hz), (x + hx + t / 2.0, y, mid_z), rgb,
             collider_only=True),
        _box(f"{prim}_xn", (t, inner[1], hz), (x - hx - t / 2.0, y, mid_z), rgb,
             collider_only=True),
        _box(f"{prim}_yp", (inner[0], t, hz), (x, y + hy + t / 2.0, mid_z), rgb,
             collider_only=True),
        _box(f"{prim}_yn", (inner[0], t, hz), (x, y - hy - t / 2.0, mid_z), rgb,
             collider_only=True),
    ]

