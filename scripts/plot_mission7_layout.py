"""Draw scoring-only geometry; the learned policy never receives this map."""
import argparse
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1]/"results/mission7-dev/plot-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
from bhl_robust.mission.layout import generate, wall_segments, SPLITS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=SPLITS, default="train")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    if a.out.exists() or not a.out.resolve().is_relative_to(root):
        p.error("choose a fresh output inside this repository")
    layout = generate(a.split, a.index)
    fig, ax = plt.subplots(figsize=(9, 9))
    for center, size in wall_segments(layout):
        ax.add_patch(Rectangle(center-np.asarray(size), 2*size[0], 2*size[1], color="#44515d"))
    route = np.asarray([layout.xy(c) for c in layout.route])
    ax.plot(*route.T, "--", color="#276cb3", alpha=.65, label="Scoring route (not a policy input)")
    decision = 0
    for c in layout.route[1:-1]:
        if len(layout.adjacency[c]) >= 3:
            decision += 1
            ax.text(*layout.xy(c), str(decision), ha="center", va="center", fontsize=10, color="white",
                    bbox=dict(boxstyle="circle", fc="#276cb3", ec="none"))
    for i in range(2):
        center, direction = layout.door(i)
        lateral = np.array([-direction[1], direction[0]])
        endpoints = np.array([center-lateral*layout.cell_m/2, center+lateral*layout.cell_m/2])
        ax.plot(*endpoints.T, color="#d27715", linewidth=5, label="Required gate" if i == 0 else None)
        for side in (-1, 1):
            ax.scatter(*layout.plate(i, side), marker="o" if side == layout.correct_sides[i] else "s", color="#d27715", s=75)
    ax.scatter(*layout.xy(layout.route[layout.object_index]), marker="s", color="#bd4545", s=90, label="Parcel")
    ax.add_patch(Circle(route[-1], .36, fill=False, color="#23825c", linewidth=3))
    ax.annotate("Drop zone", route[-1], xytext=(10, 12), textcoords="offset points", color="#23825c")
    ax.annotate("Start", route[0], xytext=(10, 10), textcoords="offset points")
    ax.set_aspect("equal")
    ax.set(xlabel="World x (m; scoring only)", ylabel="World y (m; scoring only)",
           title=f"Mission7 {a.split} seed {layout.seed}: seven decisions, {layout.turns} turns\n"
                 f"{(len(layout.route)-1)*layout.cell_m:.1f} m route · two gates · transport abstraction")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.09), ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .012, "Generated task geometry — not a learned rollout or proof of task success", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    plt.close(fig)


if __name__ == "__main__":
    main()
