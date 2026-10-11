#!/usr/bin/env python3
"""Render measured TUM VI tracking and the disclosed per-epoch comparison."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    evaluation = json.loads(args.evaluation.read_text())
    replay_result = json.loads((args.replay / "result.json").read_text())
    data = {mode: [json.loads(line)["native"] for line in
                   (args.replay / mode / "frames.jsonl").read_text().splitlines()]
            for mode in ("stereo", "stereo_inertial")}
    fig, (ax, comparison) = plt.subplots(1, 2, figsize=(12, 4.3),
                                        gridspec_kw={"width_ratios": [2.35, 1]})
    available, unavailable = "#2171a8", "#e4e7eb"
    series = [(data["stereo"], lambda f: f["tracked"], "Stereo: native OK"),
              (data["stereo_inertial"], lambda f: f["tracked"], "Stereo–IMU: native OK"),
              (data["stereo_inertial"], lambda f: f["inertial"]["map_imu_initialized"],
               "Stereo–IMU: initialized")]
    for row, (frames, predicate, label) in enumerate(series):
        start, value = 0., predicate(frames[0])
        segments = []
        for frame in frames[1:]:
            next_value = predicate(frame)
            if next_value != value:
                segments.append((start, frame["timestamp_s"] - start, value))
                start, value = frame["timestamp_s"], next_value
        segments.append((start, 60. - start, value))
        for start, duration, value in segments:
            ax.broken_barh([(start, duration)], (2-row-.25, .5),
                          facecolors=available if value else unavailable)
    reset = evaluation["arms"][1]["observed_epoch_boundaries"][0]["timestamp_s"]
    ax.axvline(reset, color="#bc3b2f", linestyle="--", linewidth=1.3)
    ax.annotate("Stereo–IMU active-map reset at 2.700 s\nMap ID 0 was reused", xy=(reset, 1.25),
                xytext=(11, 2.9), arrowprops={"arrowstyle": "->", "color": "#bc3b2f"},
                fontsize=9, color="#9d2f26")
    ax.set_yticks([2, 1, 0], [row[2] for row in series])
    ax.set(xlim=(0, 60), ylim=(-.7, 3.2), xlabel="Original recording time from first camera pair (s)")
    ax.grid(axis="x", alpha=.18)
    ax.legend(handles=[Patch(color=available, label="Available"),
                       Patch(color=unavailable, label="Unavailable / uninitialized")],
              loc="upper right", frameon=False, fontsize=8)
    shared = max(evaluation["shared_timestamp_comparisons"],
                 key=lambda x: x["shared_initialized_tracked_timestamps"])
    errors = [shared[mode]["ate_translation_m"]["rmse"] for mode in ("stereo", "stereo_inertial")]
    comparison.bar([0, 1], errors, color=["#2171a8", "#8761aa"], width=.6)
    for i, value in enumerate(errors):
        comparison.text(i, value+.009, f"{value:.3f} m", ha="center", fontsize=10)
    comparison.set_xticks([0, 1], ["Stereo", "Stereo–IMU"])
    comparison.set(ylabel="Translation ATE RMSE (m)", ylim=(0, max(errors)*1.25))
    support = shared["stereo_inertial"]
    comparison.set_title("Shared support after reinitialization\n"
                         f"{support['first_sensor_time_s']:.3f}–{support['last_sensor_time_s']:.3f} s; "
                         f"{support['associated_poses']:,} GT pairs", fontsize=10)
    comparison.grid(axis="y", alpha=.18)
    fig.suptitle("External TUM VI room1 recording: 1,200 stereo pairs, unchanged native ORB-SLAM3", fontsize=12)
    latency = [arm["pipeline_wall"]["p95_ms"] for arm in replay_result["arms"]]
    fig.text(.02, .06,
             "Each continuous map epoch uses rigid SE(3) alignment; scale is fixed at 1. The initial five-frame IMU epoch is too short for a trajectory claim.\n"
             f"Offline 2-CPU replay; full-pipeline p95: {latency[0]:.1f} ms stereo / {latency[1]:.1f} ms stereo–IMU. External recording, not validation of the user's robot.\n"
             "Data: Schubert et al., TUM VI Benchmark (IROS 2018), CC BY 4.0. Measured results and figure: this repository.",
             fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, .21, 1, .91))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(args.output.with_suffix(suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
