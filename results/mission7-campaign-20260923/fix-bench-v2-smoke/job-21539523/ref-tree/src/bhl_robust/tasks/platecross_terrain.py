"""Mission 7's plates as a training terrain (workstream m7-platecross, 2026-10-02).

Geometry for `Velocity-BHL-Arms-PlateCross-v0` (`platecross_env_cfg.py`): flat ground scattered with
Mission 7's own plates, baked into the static terrain collision mesh that Isaac Lab's TerrainGenerator
imports at /World/ground/terrain/mesh (a triangle-mesh collider with no rigid-body API, i.e. a PhysX
static rigid body the feet stand and step on, with the ground's own physics material).

ISAAC-FREE on purpose (numpy at import; trimesh inside `plates_terrain`, scipy inside `find_plates`):
the unit tests load this file by path, and the scene probe (`scripts/bench/platecross_select.py
probe-scene`) measures the mesh Isaac actually imported with the same `find_plates`.

The plates, exactly as Mission 7 builds them (src/bhl_robust/mission/layout.py, world_xml(), lines
152-153):
    shape = 'type="cylinder" size=".24 .015"' if correct else 'type="box" size=".24 .24 .015"'
    parts.append(f'<geom name="plate_{i}_{side}" {shape} pos="{x} {y} .015" ' ...)
MuJoCo sizes are half-sizes, so the round (correct) plate is a disc of radius 0.24 m and the square
(wrong) plate a 0.48 x 0.48 m box, both 0.03 m tall (half-height 0.015 m) resting on the floor
(centre z = 0.015) and axis-aligned (no euler / quat attribute). layout.plate() (lines 65-69) puts a
door's two plates 0.84 m apart; the field below does not copy door positions, only the plates.

The field (DECLARED 2026-10-02, before any training; chosen, not tuned):
  * a square lattice, pitch 1.2 m centre to centre in x and y (0.72 m of floor between neighbouring
    plates along the lattice axes);
  * tiles of 8.4 m = 7 x 1.2 m, so the lattice is centred on the tile and keeps the same 1.2 m pitch
    across tile seams;
  * shapes in a checkerboard inside each tile: round where (i + j) is even, square where odd (i, j =
    lattice offsets from the tile centre). The 7 x 7 tile is odd, so along a tile seam two
    neighbouring plates have the same shape;
  * the tile-centre lattice point is EMPTY: robots spawn there (upstream reset_base: +/-0.5 m in x
    and y, any yaw), and the nearest plate edge is 0.96 m from the centre;
  * per tile 24 round + 24 square = 48 plates, 0.68 plates/m^2 and 14.0 % of the floor. A Mission 7
    door cell (1.5-1.7 m) holds 2 plates: 0.69-0.89 plates/m^2, 14-18 % of its floor; the 1.2 m pitch
    is that density at the 1.7 m cell end;
  * the round plate is a 64-gon prism with its vertices ON r = 0.24 m (apothem 0.2397 m);
  * 10 x 10 tiles (84 m x 84 m) inside Isaac Lab's flat 20 m border (the repo's terrain convention,
    bumpy / maze / stairs); every tile is identical and the generator seed is fixed, so all three
    training seeds see the same field.
"""

from __future__ import annotations

import math
import sys

import numpy as np

# ------------------------------------------------------------------------ Mission 7's plates (exact)
ROUND_RADIUS_M = 0.24          # layout.world_xml: cylinder size=".24 .015" (radius, half-height)
SQUARE_SIDE_M = 0.48           # layout.world_xml: box size=".24 .24 .015" (half-extents)
PLATE_HEIGHT_M = 0.03          # 2 x 0.015; pos z = .015 -> resting on the floor
# The literal MuJoCo attributes, for the test that parses world_xml against these numbers.
MJCF_ROUND_SIZE = (0.24, 0.015)
MJCF_SQUARE_SIZE = (0.24, 0.24, 0.015)
MJCF_PLATE_Z = 0.015

# ------------------------------------------------------------------------ the declared field (frozen)
PITCH_M = 1.2                  # lattice pitch, centre to centre, x and y
LATTICE_N = 7                  # lattice points per tile side (odd: the tile centre is a lattice point)
TILE_M = 8.4                   # LATTICE_N x PITCH_M
DISC_SECTIONS = 64             # round plate polygon; vertices on r = ROUND_RADIUS_M
CLEAR_CENTRE = True            # the tile-centre lattice point holds no plate (the spawn point)
NUM_ROWS = 10
NUM_COLS = 10
BORDER_M = 20.0
TERRAIN_SEED = 0
PLATES_PER_TILE = {"round": 24, "square": 24}

# Read from upstream, not changed: the humanoid reset_base pose_range is +/-0.5 m in x and y.
SPAWN_HALF_RANGE_M = 0.5

# The one line the terrain function prints (once per process); the launcher's log guard greps it.
MARKER_PREFIX = "[platecross_terrain]"

_ANNOUNCED = False


def _lattice_n(length_m: float, pitch_m: float) -> int:
    n = length_m / pitch_m
    k = int(round(n))
    if abs(n - k) > 1e-9 or k < 1 or k % 2 == 0:
        raise ValueError(f"tile side {length_m} m is not an odd multiple of the {pitch_m} m pitch ({n:.6f})")
    return k


def plate_layout(size=(TILE_M, TILE_M), pitch_m: float = PITCH_M, clear_centre: bool = CLEAR_CENTRE):
    """[(shape, x, y)] relative to the tile centre: 'round' where (i + j) is even, 'square' where odd."""
    nx, ny = _lattice_n(float(size[0]), pitch_m), _lattice_n(float(size[1]), pitch_m)
    hx, hy = (nx - 1) // 2, (ny - 1) // 2
    out = []
    for i in range(-hx, hx + 1):
        for j in range(-hy, hy + 1):
            if clear_centre and i == 0 and j == 0:
                continue
            out.append(("round" if (i + j) % 2 == 0 else "square", i * pitch_m, j * pitch_m))
    return out


def disc_arrays(cx: float, cy: float, radius: float = ROUND_RADIUS_M, height: float = PLATE_HEIGHT_M,
                sections: int = DISC_SECTIONS):
    """A closed `sections`-gon prism from z = 0 to `height`, vertices on the circle, outward normals."""
    a = 2.0 * np.pi * np.arange(sections) / sections
    ring = np.column_stack([cx + radius * np.cos(a), cy + radius * np.sin(a)])
    v = np.vstack([[cx, cy, 0.0], [cx, cy, height],
                   np.column_stack([ring, np.zeros(sections)]),
                   np.column_stack([ring, np.full(sections, height)])])
    b0, t0 = 2, 2 + sections
    k = np.arange(sections)
    k1 = (k + 1) % sections
    f = np.vstack([np.column_stack([np.zeros(sections, np.int64), b0 + k1, b0 + k]),   # bottom, -z
                   np.column_stack([np.ones(sections, np.int64), t0 + k, t0 + k1]),    # top, +z
                   np.column_stack([b0 + k, b0 + k1, t0 + k1]),                        # side, outward
                   np.column_stack([b0 + k, t0 + k1, t0 + k])])
    return v, f.astype(np.int64)


def box_arrays(cx: float, cy: float, side: float = SQUARE_SIDE_M, height: float = PLATE_HEIGHT_M):
    """An axis-aligned side x side x height box resting on z = 0, outward normals."""
    h = 0.5 * side
    x0, x1, y0, y1 = cx - h, cx + h, cy - h, cy + h
    v = np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0],
                  [x0, y0, height], [x1, y0, height], [x1, y1, height], [x0, y1, height]])
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int64)
    return v, f


def tile_arrays(size=(TILE_M, TILE_M), radius: float = ROUND_RADIUS_M, side: float = SQUARE_SIDE_M,
                height: float = PLATE_HEIGHT_M, pitch_m: float = PITCH_M, sections: int = DISC_SECTIONS,
                clear_centre: bool = CLEAR_CENTRE):
    """(vertices, faces) of one tile in Isaac Lab's sub-terrain frame (corner at the origin, x in [0, sx]):
    a two-triangle floor at z = 0 plus every plate of `plate_layout`."""
    sx, sy = float(size[0]), float(size[1])
    verts = [np.array([[0.0, 0.0, 0.0], [sx, 0.0, 0.0], [sx, sy, 0.0], [0.0, sy, 0.0]])]
    faces = [np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)]
    n = 4
    for shape, x, y in plate_layout((sx, sy), pitch_m, clear_centre):
        if shape == "round":
            v, f = disc_arrays(0.5 * sx + x, 0.5 * sy + y, radius, height, sections)
        else:
            v, f = box_arrays(0.5 * sx + x, 0.5 * sy + y, side, height)
        verts.append(v)
        faces.append(f + n)
        n += len(v)
    return np.vstack(verts), np.vstack(faces)


def marker_line(size, radius, side, height, pitch_m, sections, clear_centre) -> str:
    counts = {"round": 0, "square": 0}
    for shape, _, _ in plate_layout(size, pitch_m, clear_centre):
        counts[shape] += 1
    return (f"{MARKER_PREFIX} plates tile {float(size[0]):g} x {float(size[1]):g} m: {counts['round']} round "
            f"(r {radius:g} m, {sections}-gon) + {counts['square']} square ({side:g} m), height {height:g} m, "
            f"pitch {pitch_m:g} m, {'centre clear' if clear_centre else 'centre plated'}")


def plates_terrain(difficulty: float, cfg):
    """Isaac Lab sub-terrain function (`SubTerrainBaseCfg.function`): one identical plate tile.

    `difficulty` is unused (every tile is the same field). Reads the declared numbers from `cfg`
    (platecross_env_cfg.PlatesTerrainCfg, so they are recorded in the run's params/env.yaml) and
    `cfg.size`, which the TerrainGenerator sets to its tile size. Returns ([mesh], origin), origin =
    the tile centre on the floor, in the tile frame the generator then translates."""
    import trimesh

    global _ANNOUNCED
    del difficulty
    size = (float(cfg.size[0]), float(cfg.size[1]))
    args = (size, float(cfg.round_radius_m), float(cfg.square_side_m), float(cfg.height_m), float(cfg.pitch_m),
            int(cfg.disc_sections), bool(cfg.clear_centre))
    v, f = tile_arrays(size, radius=args[1], side=args[2], height=args[3], pitch_m=args[4], sections=args[5],
                       clear_centre=args[6])
    if not _ANNOUNCED:
        print(marker_line(*args), flush=True)
        _ANNOUNCED = True
    return [trimesh.Trimesh(vertices=v, faces=f, process=False)], np.array([0.5 * size[0], 0.5 * size[1], 0.0])


# ------------------------------------------------------------------------ the scene, as built

def tile_centres(num_rows: int = NUM_ROWS, num_cols: int = NUM_COLS, size=(TILE_M, TILE_M)) -> np.ndarray:
    """(num_rows * num_cols, 2) tile centres in Isaac Lab's world frame: TerrainGenerator puts tile (r, c)
    at ((r + 0.5) * sx, (c + 0.5) * sy) and then shifts the whole terrain by -(rows * sx, cols * sy) / 2."""
    sx, sy = float(size[0]), float(size[1])
    r, c = np.meshgrid(np.arange(num_rows), np.arange(num_cols), indexing="ij")
    return np.column_stack([((r + 0.5) * sx - 0.5 * num_rows * sx).ravel(),
                            ((c + 0.5) * sy - 0.5 * num_cols * sy).ravel()])


def expected_world_plates(num_rows: int = NUM_ROWS, num_cols: int = NUM_COLS, size=(TILE_M, TILE_M),
                          pitch_m: float = PITCH_M, clear_centre: bool = CLEAR_CENTRE):
    """[(shape, x, y)] of every plate the declared field puts in Isaac's world frame."""
    out = []
    for cx, cy in tile_centres(num_rows, num_cols, size):
        out.extend((s, cx + x, cy + y) for s, x, y in plate_layout(size, pitch_m, clear_centre))
    return out


def find_plates(vertices, faces, min_height: float = 0.5 * PLATE_HEIGHT_M, max_height: float = 2.0 * PLATE_HEIGHT_M,
                max_extent: float = 1.0, tol: float = 5e-4):
    """Plates in a triangle mesh: the connected components standing min_height..max_height off z = 0 with an
    xy extent <= max_extent (floors and the generator's 1 m deep border are not). Per plate: centre (xy
    bounds midpoint), extents, height, base z, max radial vertex distance from the centre, vertex count,
    and the shape it measures as ('round': extents 2 r and every rim vertex at r; 'square': extents = side
    and corners at side / sqrt 2; else 'other')."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    n = len(v)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]]])
    g = coo_matrix((np.ones(len(e), dtype=np.int8), (e[:, 0], e[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    used = np.zeros(n, dtype=bool)
    used[f.ravel()] = True
    idx = np.flatnonzero(used)
    order = idx[np.argsort(lab[idx], kind="stable")]
    labs = lab[order]
    starts = np.flatnonzero(np.r_[True, labs[1:] != labs[:-1]])
    pts = v[order]
    lo = np.minimum.reduceat(pts, starts, axis=0)
    hi = np.maximum.reduceat(pts, starts, axis=0)
    counts = np.diff(np.r_[starts, len(order)])
    ext = hi - lo
    keep = ((ext[:, 2] >= min_height) & (ext[:, 2] <= max_height) & (lo[:, 2] > -tol)
            & (ext[:, 0] <= max_extent) & (ext[:, 1] <= max_extent))
    centre = 0.5 * (lo[:, :2] + hi[:, :2])
    rad = np.hypot(pts[:, 0] - np.repeat(centre[:, 0], counts), pts[:, 1] - np.repeat(centre[:, 1], counts))
    rmax = np.maximum.reduceat(rad, starts)
    out = []
    for k in np.flatnonzero(keep):
        p = {"x": float(centre[k, 0]), "y": float(centre[k, 1]), "ex": float(ext[k, 0]), "ey": float(ext[k, 1]),
             "height": float(ext[k, 2]), "z0": float(lo[k, 2]), "rmax": float(rmax[k]), "n_vertices": int(counts[k])}
        p["shape"] = classify(p, tol)
        out.append(p)
    return out


def classify(p: dict, tol: float = 5e-4) -> str:
    near = lambda a, b: abs(a - b) <= tol  # noqa: E731
    if near(p["ex"], 2 * ROUND_RADIUS_M) and near(p["ey"], 2 * ROUND_RADIUS_M) and near(p["rmax"], ROUND_RADIUS_M):
        return "round"
    if (near(p["ex"], SQUARE_SIDE_M) and near(p["ey"], SQUARE_SIDE_M)
            and near(p["rmax"], SQUARE_SIDE_M / math.sqrt(2.0))):
        return "square"
    return "other"


def check_scene_plates(found: list, expected: list, tol: float = 5e-4, match_tol: float = 2e-3):
    """Compare the plates measured in a scene mesh with the declared field. Returns (summary, problems);
    problems == [] iff every declared plate is there, at its place, with its shape, 0.03 m tall on the
    floor, and nothing else plate-like is."""
    from scipy.spatial import cKDTree

    problems = []
    by = {"round": [], "square": [], "other": []}
    for p in found:
        by[p["shape"]].append(p)
    want = {"round": sum(s == "round" for s, _, _ in expected), "square": sum(s == "square" for s, _, _ in expected)}
    summary = {"found": {k: len(v) for k, v in by.items()}, "expected": want}
    for k in ("round", "square"):
        if len(by[k]) != want[k]:
            problems.append(f"{len(by[k])} {k} plates in the scene, declared {want[k]}")
    if by["other"]:
        problems.append(f"{len(by['other'])} plate-like components of another shape (first {by['other'][0]})")
    for key, target in (("height", PLATE_HEIGHT_M), ("z0", 0.0)):
        vals = [p[key] for p in found]
        if vals:
            worst = max(abs(x - target) for x in vals)
            summary[f"max_abs_{key}_error_m"] = worst
            if worst > tol:
                problems.append(f"plate {key} off by up to {worst:.5f} m (declared {target})")
    for k, (target_e, target_r) in (("round", (2 * ROUND_RADIUS_M, ROUND_RADIUS_M)),
                                    ("square", (SQUARE_SIDE_M, SQUARE_SIDE_M / math.sqrt(2.0)))):
        if by[k]:
            summary[k] = {"extent_m": [min(min(p["ex"], p["ey"]) for p in by[k]), max(max(p["ex"], p["ey"]) for p in by[k])],
                          "rmax_m": [min(p["rmax"] for p in by[k]), max(p["rmax"] for p in by[k])],
                          "height_m": [min(p["height"] for p in by[k]), max(p["height"] for p in by[k])],
                          "declared": {"extent_m": target_e, "rmax_m": target_r, "height_m": PLATE_HEIGHT_M}}
    if expected and found:
        tree = cKDTree(np.array([[p["x"], p["y"]] for p in found]))
        d, j = tree.query(np.array([[x, y] for _, x, y in expected]))
        missing = int(np.sum(d > match_tol))
        wrong = int(sum(d[i] <= match_tol and found[j[i]]["shape"] != expected[i][0] for i in range(len(expected))))
        matched = set(int(j[i]) for i in range(len(expected)) if d[i] <= match_tol)
        extra = len(found) - len(matched)
        summary.update({"max_position_error_m": float(d.max()), "missing": missing, "wrong_shape": wrong, "extra": extra})
        if missing:
            problems.append(f"{missing} declared plates have no scene plate within {match_tol} m")
        if wrong:
            problems.append(f"{wrong} scene plates have the other shape than declared at their lattice point")
        if extra:
            problems.append(f"{extra} scene plates are not at a declared lattice point")
    elif expected:
        problems.append("no plates found in the scene mesh")
    return summary, problems


def spawn_clearance(origins_xy, plates: list) -> float:
    """Smallest Chebyshev (max-norm) distance from any spawn origin to any plate footprint edge (the
    footprints' bounding squares); a robot spawned within +/-SPAWN_HALF_RANGE_M of its origin starts clear
    of every plate iff this exceeds SPAWN_HALF_RANGE_M plus its foot reach."""
    from scipy.spatial import cKDTree

    o = np.asarray(origins_xy, dtype=np.float64).reshape(-1, 2)
    if not len(plates) or not len(o):
        return float("inf")
    c = np.array([[p["x"], p["y"]] for p in plates])
    half = np.array([0.5 * max(p["ex"], p["ey"]) for p in plates])
    tree = cKDTree(c)
    best = float("inf")
    for row in np.unique(np.round(o, 6), axis=0):
        for k in tree.query_ball_point(row, r=4.0):
            best = min(best, float(np.max(np.abs(c[k] - row)) - half[k]))
    return best


if __name__ == "__main__":  # pragma: no cover - a quick look, not a gate
    v, f = tile_arrays()
    print(marker_line((TILE_M, TILE_M), ROUND_RADIUS_M, SQUARE_SIDE_M, PLATE_HEIGHT_M, PITCH_M, DISC_SECTIONS,
                      CLEAR_CENTRE))
    print(f"tile: {len(v)} vertices, {len(f)} triangles", file=sys.stderr)
