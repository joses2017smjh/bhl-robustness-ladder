"""Plot verified H1 cell summaries; no inferred intervals or new episodes."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    source = root / "paired-metrics.json"
    metrics = json.loads(source.read_text())
    assert metrics["status"] == "PASS" and metrics["verified_episodes"] == 576
    keys = [f"armV5-s{seed}-{condition}" for seed in (8, 9, 10)
            for condition in ("nominal", "drop35")]
    cells = [metrics["by_actor_condition"][key] for key in keys]
    labels = [f"s{seed} / {condition}" for seed in (8, 9, 10)
              for condition in ("nominal", "35% dropout")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.hashsalt": "bhl-h1-confirmation"})
    fig, axes = plt.subplots(1, 3, figsize=(12.7, 5.6), sharey=True)
    positions = np.arange(len(cells))
    colors = {"baseline": "#697587", "candidate": "#147d92"}
    specs = [("goals", "Goals reached / 48", (0, 53)),
             ("falls", "Episodes with falls / 48", (0, 5.5)),
             ("flips", "Mean yaw-command sign flips / s", (0, 4.5))]
    for ax, (field, title, limits) in zip(axes, specs):
        for arm, offset in (("baseline", -.17), ("candidate", .17)):
            values = [cell[arm]["gait_yaw_flips_per_s"]["mean"] if field == "flips"
                      else cell[arm][field] for cell in cells]
            bars = ax.barh(positions + offset, values, height=.30,
                           color=colors[arm], label="Baseline" if arm == "baseline" else "0.2 s filter")
            ax.bar_label(bars, labels=[f"{value:.2f}" if field == "flips" else str(value)
                                      for value in values], padding=3, fontsize=9)
        if field == "goals":
            for index, condition in enumerate(("nominal", "drop35") * 3):
                gate = 40 if condition == "nominal" else 36
                ax.plot([gate, gate], [index-.40, index+.40], color="#323944", linewidth=1)
            ax.text(.03, .02, "Ticks: frozen candidate goal gates", transform=ax.transAxes,
                    fontsize=8, color="#323944")
        if field == "falls":
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(limits)
        ax.set_title(title, pad=14, fontsize=11)
        ax.grid(axis="x", color="#e4e7eb", linewidth=.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    fig.suptitle("Frozen H1 confirmation: all three actors pass", fontsize=15, y=.98)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
               bbox_to_anchor=(.5, .045), frameon=False)
    fig.text(.5, .01, "576 MuJoCo episodes · 48 shared layouts · 12-DoF biped · oracle pose/goal · descriptive counts",
             ha="center", fontsize=9, color="#434b57")
    fig.tight_layout(rect=(0, .13, 1, .93))
    for suffix in ("png", "svg"):
        output = root / f"confirmation.{suffix}"
        fig.savefig(output, dpi=180,
                    metadata={"Software": "BHL H1 verified summary"} if suffix == "png" else {"Date": None})
        if suffix == "svg":
            output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(json.dumps({"input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                      "outputs": ["confirmation.png", "confirmation.svg"]}))


if __name__ == "__main__":
    main()
