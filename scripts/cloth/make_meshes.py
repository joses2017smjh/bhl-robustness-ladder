#!/usr/bin/env python3
"""Write the low-resolution cloth planes Stage 2 will actually load."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from bhl_robust.cloth.mesh import RESOLUTIONS, default_mesh_dir, write_obj


def main() -> None:
    out = default_mesh_dir(_REPO)
    rows = []
    for name, n in RESOLUTIONS:
        rows.append(write_obj(out / f"plane_{n}x{n}.obj", n))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
