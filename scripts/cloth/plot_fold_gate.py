#!/usr/bin/env python3
"""Plot measured MuJoCo flex vertices; no generated or prescribed cloth states."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    with np.load(args.trace, allow_pickle=False) as data:
        frames = data["vertices"]
        dt, width = float(data["sample_dt"]), float(data["width"])
    n = int(round(np.sqrt(frames.shape[1])))
    if n * n != frames.shape[1]:
        raise ValueError("expected square cloth grid")
    triangles = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            triangles.extend([(a, a + 1, a + n), (a + 1, a + n + 1, a + n)])
    triangles = np.asarray(triangles)
    fig = plt.figure(figsize=(13, 3.5), constrained_layout=True)
    for k, (at, title) in enumerate(((0, "Start"), (1.75, "Edge lifted"),
                                     (3.4, "Fold placed"), (4.95, "Released and settled"))):
        ax = fig.add_subplot(1, 4, k + 1, projection="3d")
        index = min(int(round(at / dt)), len(frames) - 1)
        xyz = frames[index].copy()
        xyz[:, 2] -= 0.3
        surf = Poly3DCollection(xyz[triangles], facecolors="#41b9c0", edgecolors="#19616e",
                               linewidths=0.45, alpha=0.95)
        ax.add_collection3d(surf)
        h = width * 0.64
        table = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]])
        ax.add_collection3d(Poly3DCollection([table], color="#e0d6c4", alpha=0.35))
        if at < 3.5:
            ax.scatter(*xyz[0], c="#e04a39", s=22, depthshade=False)
            ax.scatter(*xyz[n - 1], c="#346cca", s=22, depthshade=False)
        ax.set(xlim=(-h, h), ylim=(-h, h), zlim=(0, width * 0.65), title=f"{title}\nt={index*dt:.2f} s")
        ax.view_init(elev=28, azim=-58)
        ax.set_box_aspect((2, 2, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    fig.suptitle("Measured MuJoCo towel fold · idealized Cartesian pickers · no robot model", fontsize=13)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
