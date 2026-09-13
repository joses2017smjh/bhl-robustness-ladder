#!/usr/bin/env python3
"""Plan view of where the sweeping hand can touch a table, against the old cloth layout.

Reads only committed artefacts -- the fingertip contact table
(``assets/cloth/right_hand_contact_pinch.npz``) and the layout constants -- and
writes ``docs/img/cloth_reach.png``. No simulator, no GPU:

    PY=/nfs/hpc/share/$USER/Humanoid_Lite/venv/bin/python3
    PYTHONPATH=src $PY scripts/cloth/plot_reach.py

The first version plotted the hand-link origin, which bottoms out at 0.339 m and
read as "the hand cannot reach a 0.30 m table". The fingertips hang 13 cm below
that origin and can; what they cannot do is reach far enough forward.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from bhl_robust.cloth.garments import BASKET_IDS, GARMENT_BY_NAME  # noqa: E402
from bhl_robust.cloth.reach import load_contact, robot_to_world  # noqa: E402

# The old layout, frozen here: this figure documents why it was abandoned, so it
# must not move when layout.py is redesigned.
OLD_ROBOT_XY = (-0.22, 0.0)
OLD_TABLE_CENTER, OLD_TABLE_SIZE, OLD_TABLE_TOP = (0.52, 0.0), (0.70, 0.96), 0.30
OLD_BASKET_X, OLD_BASKET_Y, OLD_BASKET_INNER = 0.10, {"socks": 0.34, "shirts": 0.00, "jackets": -0.34}, (0.24, 0.28)
OLD_GARMENT_XY = (0.52, 0.0)

SURFACE, INK, INK2, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#f0efec"
REACH, TARGET = "#2a78d6", "#eb6834"


def _hull(pts: np.ndarray) -> np.ndarray:
    p = sorted(map(tuple, pts))
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])


def main() -> None:
    t = load_contact()
    assert abs(t.table_top - OLD_TABLE_TOP) < 1e-6, "contact table was not built at the old table height"
    ix, iy = np.nonzero(t.contact_mask)
    cells = np.stack([t.x[ix], t.y[iy]], 1)
    rx, ry = OLD_ROBOT_XY
    measured = np.array([robot_to_world((x, y, 0.0), robot_xy=OLD_ROBOT_XY, robot_yaw=np.pi)[:2] for x, y in cells])
    designed = np.array([robot_to_world((x, y, 0.0), robot_xy=OLD_ROBOT_XY, robot_yaw=0.0)[:2] for x, y in cells])
    fwd = float(cells[:, 0].max())
    gx, gy = OLD_GARMENT_XY
    g = GARMENT_BY_NAME["shirt_a"]
    d_garment = float(np.hypot(gx - rx, gy - ry))
    d_edge = OLD_TABLE_CENTER[0] - OLD_TABLE_SIZE[0] / 2 - rx

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                         "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.labelcolor": INK2, "text.color": INK})
    fig, a = plt.subplots(figsize=(8.6, 5.4), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    a.set_facecolor(SURFACE)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)

    hx, hy = OLD_TABLE_SIZE[0] / 2, OLD_TABLE_SIZE[1] / 2
    cx, cy = OLD_TABLE_CENTER
    a.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, fc=NEUTRAL, ec=INK2, lw=0.8, zorder=1))
    bx, by = OLD_BASKET_INNER
    for bid in BASKET_IDS:
        a.add_patch(Rectangle((OLD_BASKET_X - bx / 2, OLD_BASKET_Y[bid] - by / 2), bx, by,
                              fc="none", ec=INK2, lw=0.8, zorder=2))
    a.add_patch(Rectangle((gx - g.proxy_size[0] / 2, gy - g.proxy_size[1] / 2), g.proxy_size[0], g.proxy_size[1],
                          fc=TARGET, ec=SURFACE, lw=1.5, zorder=3))
    s = t.step
    a.add_collection(PatchCollection([Rectangle((px - s / 2, py - s / 2), s, s) for px, py in measured],
                                     fc=REACH, ec="none", alpha=0.75, zorder=4))
    h = _hull(designed)
    a.add_patch(Polygon(h, closed=True, fc="none", ec=REACH, lw=1.3, ls=(0, (4, 3)), zorder=4))
    a.plot([rx], [ry], "o", ms=6, color=INK, zorder=6)
    a.annotate("", xy=(rx - 0.15, ry), xytext=(rx, ry),
               arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, mutation_scale=12), zorder=6)
    a.plot([rx, gx], [ry, gy], color=INK2, lw=0.7, zorder=5)
    a.text(rx - 0.075, ry + 0.03, "faces −x (measured)", color=INK, fontsize=8.5, ha="center", va="bottom")
    a.text(gx, gy + g.proxy_size[1] / 2 + 0.025, f"garment, {d_garment:.2f} m from the root",
           color=INK, fontsize=8.5, ha="center", va="bottom", zorder=7)
    a.text(cx, cy + hy + 0.03, "table, top at 0.30 m", color=INK2, fontsize=8.5, ha="center")
    a.text(OLD_BASKET_X, OLD_BASKET_Y["socks"] + by / 2 + 0.03, "baskets", color=INK2, fontsize=8.5, ha="center")
    a.annotate(f"forward fingertip reach {fwd:.2f} m\ntable edge {d_edge:.2f} m away",
               xy=(float(h[:, 0].max()), float(h[np.argmax(h[:, 0]), 1])), xytext=(-0.02, -0.56),
               color=INK, fontsize=8.5, ha="center", va="top",
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    a.set_xlim(-0.72, 0.95)
    a.set_ylim(-0.66, 0.60)
    a.set_aspect("equal")
    a.set_xlabel("world x (m)")
    a.set_ylabel("world y (m)")
    handles = [
        Patch(fc=REACH, alpha=0.75, label="fingertip contact on a 0.30 m top, as it faces"),
        Line2D([], [], color=REACH, lw=1.3, ls=(0, (4, 3)), label="same, if it faced the table"),
        Patch(fc=TARGET, label="garment (C0 spawn)"),
        Patch(fc=NEUTRAL, ec=INK2, lw=0.8, label="table and baskets"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.07, 0.005), ncol=2,
               frameon=False, fontsize=8.5, handlelength=1.8, columnspacing=1.6)
    fig.text(0.015, 0.975, "The sweeping hand cannot reach the cloth-sort garment or table",
             fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.015, 0.93, "Cells where the right hand's fingertips can touch a 0.30 m table top: MuJoCo FK + IK on a hand-fixed\n"
             "contact point, 2 cm grid, legs in the controller's pinch squat, clear of the robot's thigh. Facing from Isaac 21247910.",
             fontsize=8.0, color=INK2, va="top")
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.2, top=0.84)
    out = REPO / "docs" / "img" / "cloth_reach.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}  forward={fwd:.2f}  cells={len(cells)}  garment={d_garment:.2f}  edge={d_edge:.2f}")


if __name__ == "__main__":
    main()
