"""Isaac-free geometry for one B5 maze terrain tile."""

from __future__ import annotations

import numpy as np
import trimesh


CORRIDOR_W = 0.90
WALL_H = 0.60
WALL_T = 0.05

# Positions are relative to the terrain origin.
MAZE_WALLS = (
    (0.0, +CORRIDOR_W / 2, 6.0, "x"),
    (0.0, -CORRIDOR_W / 2, 6.0, "x"),
    (3.0, +1.8, 2.6, "y"),
    (-3.0, -1.8, 2.6, "y"),
)
JUNCTIONS = (((1.5, 0.0, 0.45), "left"), ((-1.5, 0.0, 0.45), "right"))
OBSTACLES = ((0.6, 0.15), (-0.4, -0.20), (2.2, 0.05))
BUTTON_AT = (3.0, 0.0, 0.55)

# Clip-only PreviewSurface colours. Collision lives in the fused ground mesh;
# these tuples are visual. Kept here so login-node tests can check them without
# Isaac. Matches furniture.py's maze palette (arrow / button) with a wall
# colour that reads against a warm floor.
WALL_RGB = (0.22, 0.40, 0.62)
ARROW_RGB = (0.90, 0.35, 0.10)
OBSTACLE_RGB = (0.72, 0.28, 0.18)
BUTTON_RGB = (0.15, 0.70, 0.35)
FLOOR_RGB = (0.55, 0.50, 0.44)

# Tile-relative route the robot is commanded along. Origin is the terrain
# origin (env spawn). The first node is the signed +x junction; the last is
# the button plate. A wall-follower that goes -x, or that turns into the
# +y stub at x=+3, leaves this path and is a dead end.
PATH = ((1.5, 0.0), BUTTON_AT[:2])
#: Forward cruise used as the body-x velocity command while heading-tracking.
CRUISE_SPEED = 0.40
WAYPOINT_RADIUS = 0.35
BUTTON_RADIUS = 0.40
#: |y| past this has left the 0.90 m corridor (half-width 0.45 plus a 10 cm
#: contact margin). Spawn jitter is ±0.15 m and must stay inside.
CORRIDOR_Y_LIMIT = CORRIDOR_W / 2 + 0.10
#: Walking -x of this has taken the wrong T-junction.
WRONG_WAY_X = -2.2


def _box(
    size: tuple[float, float, float],
    centre: tuple[float, float, float],
) -> trimesh.Trimesh:
    return trimesh.creation.box(
        extents=size,
        transform=trimesh.transformations.translation_matrix(centre),
    )


def maze_meshes(
    size: tuple[float, float],
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Build floor and hazards in the tile frame, whose origin is its centre."""
    ox, oy = 0.5 * size[0], 0.5 * size[1]
    # Two triangles are enough for a flat square and keep this helper free of
    # Isaac imports so login-node tests can inspect the actual geometry.
    floor = trimesh.Trimesh(
        vertices=np.array(((0, 0, 0), (size[0], 0, 0), (size[0], size[1], 0), (0, size[1], 0))),
        faces=np.array(((0, 1, 2), (0, 2, 3))),
        process=False,
    )
    meshes: list[trimesh.Trimesh] = [floor]

    for x, y, length, axis in MAZE_WALLS:
        box_size = (length, WALL_T, WALL_H) if axis == "x" else (WALL_T, length, WALL_H)
        meshes.append(_box(box_size, (ox + x, oy + y, 0.5 * WALL_H)))

    for (x, y, z), heading in JUNCTIONS:
        dy = 0.12 if heading == "left" else -0.12
        meshes.append(_box((0.30, 0.04, 0.10), (ox + x, oy + y + dy, z)))

    for x, y in OBSTACLES:
        obstacle_size = 0.10
        meshes.append(
            _box(
                (obstacle_size, obstacle_size, obstacle_size),
                (ox + x, oy + y, 0.5 * obstacle_size),
            )
        )

    meshes.append(
        _box(
            (0.06, 0.16, 0.16),
            (ox + BUTTON_AT[0], oy + BUTTON_AT[1], BUTTON_AT[2]),
        )
    )
    return meshes, np.array((ox, oy, 0.0))


def visual_overlays() -> tuple[tuple[str, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]], ...]:
    """Tile-relative coloured cuboids for camera-sensor clips.

    Each item is ``(name, size, centre_xyz, rgb)`` in the same frame as
    ``PATH`` (origin = terrain / env origin). They are visual only: the fused
    ground mesh keeps collision and the ray sensors. Overlays are a couple of
    centimetres thicker than the collision walls so they cover the gray mesh
    instead of z-fighting with it.
    """
    extra = 0.02
    items: list[tuple[str, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = []
    for i, (x, y, length, axis) in enumerate(MAZE_WALLS):
        size = (length, WALL_T + extra, WALL_H) if axis == "x" else (WALL_T + extra, length, WALL_H)
        items.append((f"wall{i}", size, (x, y, 0.5 * WALL_H), WALL_RGB))
    for i, ((x, y, z), heading) in enumerate(JUNCTIONS):
        dy = 0.12 if heading == "left" else -0.12
        items.append((f"arrow{i}", (0.30, 0.04, 0.10), (x, y + dy, z), ARROW_RGB))
    for i, (x, y) in enumerate(OBSTACLES):
        items.append((f"obs{i}", (0.10, 0.10, 0.10), (x, y, 0.05), OBSTACLE_RGB))
    items.append(("button", (0.06, 0.16, 0.16), BUTTON_AT, BUTTON_RGB))
    return tuple(items)


def dead_end_xy(xy: np.ndarray) -> np.ndarray:
    """True where tile-relative ``xy`` (..., 2) has left the commanded route."""
    xy = np.asarray(xy, dtype=float)
    x, y = xy[..., 0], xy[..., 1]
    return (np.abs(y) > CORRIDOR_Y_LIMIT) | (x < WRONG_WAY_X)


def button_xy_reached(xy: np.ndarray, radius: float = BUTTON_RADIUS) -> np.ndarray:
    """True where tile-relative ``xy`` is on the button plate."""
    xy = np.asarray(xy, dtype=float)
    goal = np.asarray(BUTTON_AT[:2], dtype=float)
    return np.linalg.norm(xy - goal, axis=-1) <= radius


def junction_indicated_heading(label: str) -> float:
    """World yaw the geometric arrow at a junction encodes, travelling +x.

    ``left`` is +y (positive yaw); ``right`` is −y. The plate's y-offset in
    ``maze_meshes`` is the same sign, so a stereo pair that sees the plate
    offset can read the heading the command already tracks.
    """
    if label == "left":
        return float(np.pi / 2.0)
    if label == "right":
        return float(-np.pi / 2.0)
    raise ValueError(f"junction heading must be 'left' or 'right', got {label!r}")
