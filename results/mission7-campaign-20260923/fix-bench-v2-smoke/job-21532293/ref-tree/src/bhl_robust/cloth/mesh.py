"""Low-resolution cloth plane meshes.

Stage 2 is not allowed to start at 961 vertices. These generators write a
triangle grid at 8×8 / 10×10 / 12×12 / 16×16 so a throughput bench can name
the cheapest mesh that still bends, drags, and slides.
"""

from __future__ import annotations

from pathlib import Path

RESOLUTIONS: tuple[tuple[str, int], ...] = (
    ("very_low", 8),
    ("low", 10),
    ("medium_low", 12),
    ("medium", 16),
)


def grid_counts(n: int) -> tuple[int, int]:
    """``n`` is vertices along one side. A 8×8 grid is 64 vertices, 98 tris."""
    if n < 2:
        raise ValueError("need at least a 2×2 vertex grid")
    n_verts = n * n
    n_tris = 2 * (n - 1) * (n - 1)
    return n_verts, n_tris


def plane_vertices(n: int, size_xy: tuple[float, float] = (0.28, 0.22)) -> list[tuple[float, float, float]]:
    sx, sy = size_xy
    verts: list[tuple[float, float, float]] = []
    for j in range(n):
        v = 0.0 if n == 1 else j / (n - 1)
        y = (v - 0.5) * sy
        for i in range(n):
            u = 0.0 if n == 1 else i / (n - 1)
            x = (u - 0.5) * sx
            verts.append((x, y, 0.0))
    return verts


def plane_faces(n: int) -> list[tuple[int, int, int]]:
    faces: list[tuple[int, int, int]] = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = a + 1
            c = a + n
            d = c + 1
            faces.append((a + 1, c + 1, b + 1))  # OBJ is 1-indexed
            faces.append((b + 1, c + 1, d + 1))
    return faces


def write_obj(path: Path, n: int, size_xy: tuple[float, float] = (0.28, 0.22)) -> dict[str, int | str]:
    """Write a triangle plane. Returns vertex/face counts for the bench table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    verts = plane_vertices(n, size_xy)
    faces = plane_faces(n)
    lines = [
        f"# cloth proxy {n}x{n} vertices={len(verts)} faces={len(faces)}",
        f"# size_xy={size_xy}",
    ]
    for x, y, z in verts:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    for a, b, c in faces:
        lines.append(f"f {a} {b} {c}")
    path.write_text("\n".join(lines) + "\n")
    n_verts, n_tris = grid_counts(n)
    return {
        "path": str(path),
        "side": n,
        "vertices": n_verts,
        "triangles": n_tris,
    }


def default_mesh_dir(repo_root: Path | None = None) -> Path:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "assets" / "cloth"


def isaac_grid_counts(resolution: int) -> tuple[int, int]:
    """Vertices and triangles for an Isaac Lab ``MeshRectangleCfg``.

    ``resolution=(n, n)`` counts **cells**, not vertices, so the spawned mesh
    carries ``(n + 1) ** 2`` vertices: an 8x8 cloth is **81** vertices, not 64.

    Measured, not assumed — job 21233960 logged ``Particles per body: 81`` and
    ``Registered UsdGeom.Mesh: 81 vertices`` for ``resolution=8``. Reporting
    ``grid_counts(8) = 64`` for that mesh under-counts by ``2n + 1`` and puts a
    wrong vertex count in the results table, which is the one number a cloth
    throughput row is compared on.

    ``grid_counts`` keeps its own meaning — vertices per side — because that is
    correct for the OBJ writer, which places vertices directly.
    """
    return grid_counts(resolution + 1)
