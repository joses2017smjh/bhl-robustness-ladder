#!/usr/bin/env python3
"""Plan and side view of the sweeping hand's measured reach against the cloth layout.

Reads only committed artefacts -- the IK/reach table
(``assets/cloth/right_arm_ik_pinch.npz``) and the layout constants -- and writes
``docs/img/cloth_reach.png``. No simulator, no GPU:

    PY=/nfs/hpc/share/$USER/Humanoid_Lite/venv/bin/python3
    PYTHONPATH=src $PY scripts/cloth/plot_reach.py
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
from matplotlib.patches import Circle, Patch, Polygon, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from bhl_robust.cloth import layout as L  # noqa: E402
from bhl_robust.cloth.garments import BASKET_IDS, GARMENT_BY_NAME  # noqa: E402
from bhl_robust.cloth.reach import load_reach, robot_to_world  # noqa: E402

# Reference palette (light mode): slot 1 for the reach, slot 2 for the target.
SURFACE, INK, INK2, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#f0efec"
REACH, TARGET = "#2a78d6", "#eb6834"


def _hull(pts: np.ndarray) -> np.ndarray:
    """Convex hull, monotone chain. Over-states the region, which only helps the
    'as designed' outline -- and it still stops short of the table."""
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
    t = load_reach()
    ix, iy, _ = np.nonzero(t.mask)
    cells = np.unique(np.stack([t.x[ix], t.y[iy]], 1).round(3), axis=0)
    measured = np.array([robot_to_world((x, y, 0.0))[:2] for x, y in cells])
    designed = np.array([robot_to_world((x, y, 0.0), robot_yaw=0.0)[:2] for x, y in cells])

    low = t.mask[:, :, t.z <= 0.36 + 1e-9]
    lx, ly, _ = np.nonzero(low)
    r_low = float(np.hypot(t.x[lx], t.y[ly]).max())
    z_min = float(t.z[np.nonzero(t.mask.any(axis=(0, 1)))[0].min()])
    rx, ry = L.ROBOT_XY
    g = GARMENT_BY_NAME["shirt_a"]
    gx, gy = L.default_spawn_xy(g, 0, 1)
    d_garment = float(np.hypot(gx - rx, gy - ry))
    d_edge = float(L.TABLE_CENTER_XY[0] - L.TABLE_SIZE_XY[0] / 2 - rx)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
        "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
        "axes.labelcolor": INK2, "text.color": INK,
    })
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.2, 4.7), dpi=170,
                               gridspec_kw={"width_ratios": [1.6, 1.0], "wspace": 0.22})
    fig.patch.set_facecolor(SURFACE)
    for ax in (a, b):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # ---------------------------------------------------------- plan view
    hx, hy = L.TABLE_SIZE_XY[0] / 2, L.TABLE_SIZE_XY[1] / 2
    cx, cy = L.TABLE_CENTER_XY
    a.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, fc=NEUTRAL, ec=INK2, lw=0.8, zorder=1))
    bx, by = L.BASKET_INNER[0], L.BASKET_INNER[1]
    for bid in BASKET_IDS:
        a.add_patch(Rectangle((L.BASKET_X - bx / 2, L.BASKET_Y[bid] - by / 2), bx, by,
                              fc="none", ec=INK2, lw=0.8, zorder=2))
    a.add_patch(Rectangle((gx - g.proxy_size[0] / 2, gy - g.proxy_size[1] / 2),
                          g.proxy_size[0], g.proxy_size[1], fc=TARGET, ec=SURFACE, lw=1.5, zorder=3))
    s = t.step
    a.add_collection(PatchCollection(
        [Rectangle((px - s / 2, py - s / 2), s, s) for px, py in measured],
        fc=REACH, ec="none", alpha=0.75, zorder=4))
    h = _hull(designed)
    a.add_patch(Polygon(h, closed=True, fc="none", ec=REACH, lw=1.3, ls=(0, (4, 3)), zorder=4))
    a.add_patch(Circle((rx, ry), r_low, fc="none", ec=INK2, lw=0.7, zorder=2))
    a.plot([rx], [ry], "o", ms=6, color=INK, zorder=6)
    a.annotate("", xy=(rx - 0.15, ry), xytext=(rx, ry),
               arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, mutation_scale=12), zorder=6)
    a.plot([rx, gx], [ry, gy], color=INK2, lw=0.7, zorder=5)
    a.text(gx, gy + g.proxy_size[1] / 2 + 0.025, f"garment, {d_garment:.2f} m from the root",
           color=INK, fontsize=8.5, ha="center", va="bottom", zorder=7)
    ang = np.deg2rad(225.0)
    a.annotate(f"reach at z ≤ 0.36 m: {r_low:.2f} m",
               xy=(rx + r_low * np.cos(ang), ry + r_low * np.sin(ang)), xytext=(-0.61, -0.52),
               color=INK2, fontsize=8.5, ha="left", va="top",
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    a.text(rx - 0.075, ry - 0.035, "faces −x (measured)", color=INK, fontsize=8.5, ha="center", va="top")
    a.text(cx, cy + hy + 0.03, "table, top at 0.30 m", color=INK2, fontsize=8.5, ha="center")
    a.text(L.BASKET_X, L.BASKET_Y[BASKET_IDS[0]] + by / 2 + 0.03, "baskets", color=INK2, fontsize=8.5, ha="center")
    handles = [
        Patch(fc=REACH, alpha=0.75, label="right-hand reach, as it faces"),
        Line2D([], [], color=REACH, lw=1.3, ls=(0, (4, 3)), label="same reach if it faced the table"),
        Patch(fc=TARGET, label="garment (C0 spawn)"),
        Patch(fc=NEUTRAL, ec=INK2, lw=0.8, label="table and baskets, old layout"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.055, 0.005), ncol=4,
               frameon=False, fontsize=8.5, handlelength=1.8, columnspacing=1.6)
    a.set_xlim(-0.62, 0.95)
    a.set_ylim(-0.60, 0.60)
    a.set_aspect("equal")
    a.set_xlabel("world x (m)")
    a.set_ylabel("world y (m)")
    a.set_title("Plan view: the garment and the table are outside the reach", loc="left", fontsize=10, color=INK)

    # ---------------------------------------------------------- side view
    zs, fw = [], []
    for k, z in enumerate(t.z):
        m = t.mask[:, :, k]
        if m.any():
            zs.append(float(z))
            fw.append(float(t.x[np.nonzero(m)[0]].max()))
    b.plot(fw, zs, "-o", color=REACH, lw=2, ms=4.5, mec=SURFACE, mew=1.2, zorder=4)
    b.axhline(L.TABLE_TOP_Z, color=INK2, lw=0.8, zorder=2)
    b.text(0.78, L.TABLE_TOP_Z - 0.018, "old table top, 0.30 m", color=INK2, fontsize=8.5, ha="right", va="top")
    for d, lab, ha, dx in ((d_edge, f"table edge\n{d_edge:.2f} m", "left", 0.012),
                           (d_garment, f"garment\n{d_garment:.2f} m", "right", -0.012)):
        b.axvline(d, color=INK2, lw=0.8, zorder=2)
        b.text(d + dx, 0.553, lab, color=INK2, fontsize=8.5, ha=ha, va="top")
    b.annotate(f"lowest reach {z_min:.2f} m", xy=(fw[0], zs[0]), xytext=(fw[0] + 0.12, zs[0] - 0.035),
               color=INK, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    b.grid(axis="both", color="#e6e5e1", lw=0.6, zorder=0)
    b.set_xlim(0.0, 0.80)
    b.set_ylim(0.26, 0.56)
    b.set_xlabel("furthest forward of the root (m)")
    b.set_ylabel("hand-link height (m)")
    b.set_title("Side view: forward reach by height", loc="left", fontsize=10, color=INK)

    fig.text(0.012, 0.975, "The sweeping hand cannot reach the cloth-sort garment or table",
             fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.925, "Right-arm reach from MuJoCo FK + IK (2 cm voxels, hand-link heights 0.34-0.54 m), legs in the "
             "controller's pinch squat. Facing from Isaac job 21247910. The dashed outline overlaps the baskets in plan, but they sit "
             "0-0.14 m high, below the hand's 0.34 m floor.", fontsize=8.0, color=INK2, va="top", wrap=True)
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.19, top=0.80)
    out = REPO / "docs" / "img" / "cloth_reach.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}  r_low={r_low:.3f}  z_min={z_min:.2f}  garment={d_garment:.3f}  edge={d_edge:.3f}")


if __name__ == "__main__":
    main()
