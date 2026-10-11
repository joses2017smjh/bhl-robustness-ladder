#!/usr/bin/env python3
"""Plot checksum-bound native replay poses against independent evaluator truth."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bhl_robust.research.pose_metrics import transform


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def plot(raw_archive, capture_archive, collection, output):
    raw_archive, capture_archive, collection, output = map(Path, (raw_archive, capture_archive, collection, output))
    report = json.loads((collection / "report.json").read_text())
    if digest(raw_archive) != report["raw_archive_sha256"]:
        raise ValueError("native raw archive differs from collected evidence")
    with tarfile.open(raw_archive) as native, tarfile.open(capture_archive) as capture:
        campaign = json.load(native.extractfile("campaign_result.json"))
        if digest(capture_archive) != campaign["input_artifacts"]["capture_archive"]:
            raise ValueError("capture archive differs from estimator input")
        manifest = json.load(capture.extractfile("replay/manifest.json"))
        sequences = {s["id"]: s for s in manifest["sequences"]}
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, len(campaign["runs"]), figsize=(4 * len(campaign["runs"]), 6), squeeze=False)
        counts = []
        for column, run in enumerate(campaign["runs"]):
            key = "T_W_I" if run["method"] == "lio" else "T_W_C"
            extrinsic = transform(manifest["calibration"]["T_B_I" if run["method"] == "lio" else "T_B_C"])
            inverse = np.linalg.inv(extrinsic)
            reference, stamps = [], []
            for frame in sequences[run["sequence"]]["frames"]:
                raw = capture.extractfile("replay/" + frame["truth"]).read()
                if hashlib.sha256(raw).hexdigest() != manifest["file_sha256"][frame["truth"]]:
                    raise ValueError("evaluator truth differs from captured data")
                with np.load(io.BytesIO(raw), allow_pickle=False) as truth:
                    reference.append(transform(truth[key]) @ inverse)
                stamps.append(frame["timestamp_s"])
            reference = np.asarray(reference)
            rows = [json.loads(line) for line in native.extractfile(str(Path(run["receipt"]).parent) + "/native_frames.jsonl")]
            top, lower = axes[:, column]
            top.plot(reference[:, 0, 3], reference[:, 1, 3], "k-", label="independent truth", linewidth=2)
            offset, matched = 0, 0
            for segment in run["measurement"]["segments"]:
                part = rows[offset:offset + segment["frames"]]
                offset += segment["frames"]
                metrics = segment["metrics"]
                if not metrics:
                    continue
                indices = metrics["associated_estimate_indices"]
                fit = transform(metrics["T_truthworld_estimatedworld"])
                aligned = np.asarray([fit @ transform(part[i][key]) @ inverse for i in indices])
                times = np.asarray([part[i]["timestamp_s"] for i in indices])
                ref_times = np.asarray(stamps)
                truth_xyz = np.column_stack([np.interp(times, ref_times, reference[:, j, 3]) for j in range(3)])
                errors = np.linalg.norm(aligned[:, :3, 3] - truth_xyz, axis=1)
                top.plot(aligned[:, 0, 3], aligned[:, 1, 3], "--", label="native, rigid alignment")
                lower.plot(times, 100 * errors)
                matched += len(indices)
            if not matched:
                lower.text(.5, .5, "No tracked poses; error unavailable",
                           transform=lower.transAxes, ha="center", va="center", fontsize=8)
            top.set(title=f"{run['sequence']} ({run['split']})", xlabel="world x (m)", ylabel="world y (m)")
            top.set_aspect("equal", adjustable="datalim")
            top.legend(fontsize=7)
            lower.set(xlabel="sensor time (s)", ylabel="translation error (cm)")
            for row in rows:
                if not row["tracked"]:
                    lower.axvspan(row["timestamp_s"], row["timestamp_s"] + 1 / manifest["frame_rate_hz"], color="gray", alpha=.15)
            top.grid(alpha=.2)
            lower.grid(alpha=.2)
            counts.append({"sequence": run["sequence"], "plotted_associated_poses": matched})
        scope = "SMOKE ONLY, excluded from qualification" if campaign["phase"] == "smoke" else "three simulated replay scenes; no hardware claim"
        fig.suptitle("Actual native estimator replay — " + scope, fontsize=11)
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        plt.close(fig)
    receipt = {"schema": "bhl-native-replay-figure-v1", "raw_archive_sha256": digest(raw_archive),
               "capture_archive_sha256": digest(capture_archive), "collection_report_sha256": digest(collection / "report.json"),
               "plot_script_sha256": digest(Path(__file__)), "image_sha256": digest(output), "counts": counts,
               "alignment": "same independent evaluator-only rigid transforms as reported metrics; scale exactly one",
               "truth_interpolation": "linear body translation at bracketed native timestamps; final uncovered tail excluded by metric association",
               "lost_states": "retained as gray intervals; never plotted as identity estimates"}
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-archive", type=Path, required=True)
    p.add_argument("--capture-archive", type=Path, required=True)
    p.add_argument("--collection", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(plot(a.raw_archive, a.capture_archive, a.collection, a.output), indent=2))
