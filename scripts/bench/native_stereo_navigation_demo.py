#!/usr/bin/env python3
"""Render actual original stereo inputs and estimated/true navigation traces."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np


def demo(job_root, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    job_root, output = Path(job_root), Path(output)
    completion = json.loads((job_root/"completion.json").read_text())
    raw = job_root/"outputs.tar.gz"
    with raw.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != completion["files"]["outputs.tar.gz"]["sha256"]:
        raise ValueError("original raw output differs from completed job receipt")
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(raw) as archive:
        def read(name):
            return json.load(archive.extractfile(name))
        campaign = read("campaign_result.json")
        if campaign["method"] != "orb_slam3_stereo_pose_plus_lidar_obstacle_brake":
            raise ValueError("actual stereo-navigation evidence required")
        if campaign != json.loads((job_root/"campaign_result.json").read_text()):
            raise ValueError("outer/raw scientific outcomes differ")
        rows = [read(name) for name in campaign["episode_files"]]
        cases = [case for case in ("straight", "dogleg", "occluders") if any(row["case"] == case for row in rows)]
        fig, axes = plt.subplots(1, len(cases), figsize=(5*len(cases), 4.6), squeeze=False)
        for ax, case in zip(axes[0], cases):
            for row in rows:
                if row["case"] != case:
                    continue
                prefix = f"{row['case']}-s{row['seed']}"
                native = [json.loads(line) for line in archive.extractfile(prefix+"/native/native_frames.jsonl")]
                truth_xy = np.asarray([tick["evaluator_xy"] for tick in row["trace"]])
                line, = ax.plot(truth_xy[:, 0], truth_xy[:, 1], label=f"group{row['seed']} true")
                if row["initial_registration"] is not None:
                    registration = np.asarray(row["initial_registration"])
                    imu_body = np.linalg.inv(np.asarray(row["T_body_imu"]))
                    estimated = np.asarray([(registration@np.asarray(frame["T_W_I"])@imu_body)[:2, 3]
                                            for frame in native if frame["tracked"]])
                    if len(estimated):
                        ax.plot(estimated[:, 0], estimated[:, 1], "--", color=line.get_color(), label=f"group{row['seed']} native")
                route = np.asarray(row["route_external"])
                ax.plot(route[:, 0], route[:, 1], ":", color="black", alpha=.3)
                ax.scatter(*route[-1], marker="x", color="black", s=65)
            ax.set_title(case)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel("world x (m)")
            ax.set_ylabel("world y (m)")
            ax.grid(alpha=.2)
            ax.legend(fontsize=6)
        phase = "SMOKE ONLY" if campaign["phase"] == "smoke" else "DEVELOPMENT ONLY"
        fig.suptitle(f"Actual native ORB stereo pose + LiDAR brake\n{campaign['goals']}/{campaign['episodes']} goals; {phase}", fontsize=12)
        fig.tight_layout()
        fig.savefig(output/"actual-stereo-navigation-paths.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        selected = rows[0]
        prefix = f"{selected['case']}-s{selected['seed']}"
        images = read(prefix+"/stereo/image_manifest.json")["images"]
        positions = sorted(set([0, len(images)//2, len(images)-1]))
        fig, axes = plt.subplots(len(positions), 2, figsize=(10, 3.8*len(positions)), squeeze=False)
        for axes_row, index in zip(axes, positions):
            frame = images[index]
            for ax, side in zip(axes_row, ("left", "right")):
                payload = archive.extractfile(prefix+"/stereo/"+Path(frame[side]).name).read()
                if hashlib.sha256(payload).hexdigest() != frame[side+"_sha256"]:
                    raise ValueError("original stereo image differs from capture manifest")
                ax.imshow(Image.open(io.BytesIO(payload)))
                ax.set_title(f"Original {side}, t={frame['timestamp_s']:.3f}s")
                ax.axis("off")
        fig.suptitle(f"Actual body-attached 640×480 stereo inputs; 12cm baseline, full body roll\n{selected['case']}, group{selected['seed']}; simulation", fontsize=12)
        fig.tight_layout()
        fig.savefig(output/"original-stereo-input-demo.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    receipt = {"scope": "postprocessing visualization of actual original simulation inputs/outputs; no synthetic trajectory or truth feedback",
               "generator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "raw_archive_sha256": checksum, "scientific_status": campaign["status"], "phase": campaign["phase"],
               "episodes": len(rows), "goals": campaign["goals"], "native_origin_registration": "same declared-start registration used by controller; no evaluator alignment on this path plot",
               "source_image_episode": {"case": selected["case"], "group": selected["seed"]},
               "files": {path.name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for path in output.glob("*.png")}}
    (output/"demo-receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(demo(args.job_root, args.output), indent=2))
