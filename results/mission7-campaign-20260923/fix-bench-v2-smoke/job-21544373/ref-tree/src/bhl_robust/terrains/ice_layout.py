"""Where B3's ice patches sit, in the env frame.

Kept free of Isaac Lab so the packing invariant can be tested on the login
node. ``ice.py`` turns these offsets into kinematic bodies.
"""

from __future__ import annotations

#: Must match ``BUMPY_TERRAINS_CFG.size[0]``. A patch that leaves this square
#: sits on a neighbouring tile, and GPU collision filtering lets a robot touch
#: only its own env's prims -- the same fault that put the original patches a
#: median 72 m away (``21328532``).
TILE_SIZE = 8.0

#: Patch top face sits at exactly z = 0, flush with the ground plane.
PATCH_THICKNESS = 0.02
PATCH_INSET = PATCH_THICKNESS / 2.0

PATCH_SIZE = 1.2
N_PATCHES = 6
#: Keep every face inside the tile, not just the centre.
TILE_MARGIN = 0.15


def ice_local_offsets(
    n: int = N_PATCHES,
    size: float = PATCH_SIZE,
    tile_size: float = TILE_SIZE,
    margin: float = TILE_MARGIN,
) -> tuple[tuple[float, float], ...]:
    """Patch centres in the env frame, packed inside one terrain tile.

    Laid along +x on alternating y, so a straight-line velocity command cannot
    skip them by drifting sideways. The original layout used 2 m spacing from
    x = 2 to x = 12, which does not fit in an 8 m tile.
    """
    if n < 1:
        raise ValueError(f"need at least one patch, got {n}")
    half = size / 2.0
    usable = tile_size / 2.0 - half - margin
    if usable <= 0:
        raise ValueError(
            f"patch size {size} does not fit in a {tile_size} m tile "
            f"with margin {margin}"
        )
    if n == 1:
        xs = [0.0]
    else:
        xs = [-usable + 2.0 * usable * i / (n - 1) for i in range(n)]
    y_amp = min(size * 0.6, usable)
    return tuple((float(x), float(y_amp if i % 2 else -y_amp)) for i, x in enumerate(xs))


def patch_fits_tile(
    pos: tuple[float, float],
    size: float = PATCH_SIZE,
    tile_size: float = TILE_SIZE,
    margin: float = TILE_MARGIN,
) -> bool:
    half = size / 2.0
    limit = tile_size / 2.0 - margin
    return abs(pos[0]) + half <= limit + 1e-9 and abs(pos[1]) + half <= limit + 1e-9
