"""Record surviving media inputs without rerunning a simulation.

Uses the original, read-only research workspace for unpublished MP4s, frames,
event files and CSVs. Outputs portable relative paths and content hashes. The
frame tree digest hashes a sorted JSON manifest of name, size and SHA256; the
complete manifest is retained beside the stereo GIF. Requires TensorBoard,
Pillow and NumPy in the existing research environment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def file_record(path: Path, root: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size, "sha256": digest(path)}


def gif_record(path: Path, root: Path) -> dict:
    record = file_record(path, root)
    with Image.open(path) as image:
        record.update(width=image.width, height=image.height, frames=image.n_frames)
        durations = []
        for index in range(image.n_frames):
            image.seek(index)
            durations.append(image.info.get("duration", 0))
        record["duration_ms"] = sum(durations)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    repo, assets, logs = args.repo.resolve(), args.asset_root.resolve(), args.log_root.resolve()
    base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    run_root = assets / "external/Berkeley-Humanoid-Lite/logs/rsl_rl/biped"
    event_rows = []
    means = {}
    for family in ("mazefix-stereo", "maze-stereo", "maze-blind"):
        means[family] = []
        for seed in range(3):
            runs = sorted(run_root.glob(f"*_{family}-s{seed}"))
            if len(runs) != 1:
                raise ValueError(f"Expected exactly one run for {family} seed {seed}")
            events = sorted(runs[0].glob("events.out.tfevents.*"))
            if len(events) != 1:
                raise ValueError(f"Expected exactly one event file: {runs[0]}")
            accumulator = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
            accumulator.Reload()
            samples = accumulator.Scalars("Curriculum/terrain_levels")
            if len(samples) != 6000 or samples[-1].step != 5999:
                raise ValueError(f"Incomplete training event series: {events[0]}")
            mean = float(np.mean([sample.value for sample in samples[-50:]]))
            means[family].append(mean)
            event_rows.append({**file_record(events[0], assets), "family": family,
                               "seed": seed, "scalar_tag": "Curriculum/terrain_levels",
                               "samples": len(samples), "last_step": samples[-1].step,
                               "last_50_mean": mean})
    frame_root = assets / "results/clips/frames/maze_stereo_fixed"
    manifest = [file_record(path, frame_root) for path in sorted(frame_root.glob("*.png"))]
    counts = {prefix: len(list(frame_root.glob(prefix + "_*.png")))
              for prefix in ("frame", "stereo_raw", "stereo_l")}
    if any(count != 200 for count in counts.values()) or len(manifest) != 600:
        raise ValueError(f"Incomplete original frame tree: {counts}")
    manifest_sha = hashlib.sha256(encoded(manifest)).hexdigest()
    inputs_path = repo / "docs/gifs/isaac/maze_stereo_fixed.inputs.json"
    write(inputs_path, {"schema": "bhl-media-frame-manifest/v1", "directory":
                       frame_root.relative_to(assets).as_posix(), "files": manifest,
                       "digest_rule": "SHA256 of sorted JSON files array; sort_keys=True, separators=(',', ':')",
                       "tree_sha256": manifest_sha})
    capture_log = logs / "maze-fixclip-21329137.out"
    capture_text = capture_log.read_text()
    expected_run = "2026-09-13_22-58-29_mazefix-stereo-s0"
    if expected_run not in capture_text or "wrote 200 clip frames" not in capture_text:
        raise ValueError("Original capture log does not identify the expected run and frame count")
    training_log = logs / "maze-ppo-21317023_2.out"
    if "run=mazefix-stereo-s0 seed=0" not in training_log.read_text():
        raise ValueError("Training job does not identify corrected stereo seed 0")
    old_gif = subprocess.check_output(
        ["git", "show", f"{base_commit}:docs/gifs/isaac/maze_stereo_fixed.gif"], cwd=repo)
    stereo = {"schema": "bhl-media-provenance/v1", "output": gif_record(
        repo / "docs/gifs/isaac/maze_stereo_fixed.gif", repo),
        "recording_job": 21329137, "training_job": "21317023_2",
        "training_log": {"filename": training_log.name, "sha256": digest(training_log)},
        "recording_run": f"external/Berkeley-Humanoid-Lite/logs/rsl_rl/biped/{expected_run}",
        "recording_checkpoint": file_record(run_root / expected_run / "model_5999.pt", assets),
        "capture_log": {"filename": capture_log.name, "sha256": digest(capture_log)},
        "events": event_rows, "last_50_summary": {
            family: {"per_training_seed": values, "three_seed_mean": float(np.mean(values))}
            for family, values in means.items()},
        "source_frames": {"directory": frame_root.relative_to(assets).as_posix(),
                          "counts": counts, "tree_sha256": manifest_sha,
                          "manifest": inputs_path.relative_to(repo).as_posix(),
                          "manifest_sha256": digest(inputs_path)},
        "assembly": {"script": "scripts/gif_maze_stereo_fixed.py", "script_sha256":
                     digest(repo / "scripts/gif_maze_stereo_fixed.py"), "every": 2,
                     "width": 560, "duration_ms_per_frame": 80},
        "title": "Corrected stereo seed 0: 0.877 (before: 0.001)",
        "previous_output": {"git_object": f"{base_commit}:docs/gifs/isaac/maze_stereo_fixed.gif",
                            "bytes": len(old_gif), "sha256": hashlib.sha256(old_gif).hexdigest()},
        "scope": "Recorded Isaac rollout and paired ray-depth views; terrain level is a training metric, not a success rate. Corrected versus pre-fix values compare seed 0 with seed 0; three-seed means are separate.",
        "reassembly": "2026-10-08: title corrected using the existing frozen 200-frame recording; no Isaac episode rerun. The old title mixed corrected seed 0 (0.88) with the pre-fix three-seed mean (0.02).",
        "limits": "Original MP4/frame/event assets live in the research workspace; hashes identify surviving bytes. No renderer determinism or new policy result is claimed."}
    write(repo / "docs/gifs/isaac/maze_stereo_fixed.json", stereo)
    dr_inputs = []
    for family, suffix in (("dr-default-s0", "OK"), ("dr-off-s0", "FELL")):
        source = assets / f"results/video/{family}__vx+0.0_vy+0.2_wz+0.0__{suffix}.mp4"
        data = assets / f"results/raw/{family}.csv"
        with data.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        selected = [row for row in rows if
                    [float(row[key]) for key in ("command_vx", "command_vy", "command_wz")]
                    == [0.0, 0.2, 0.0] and float(row["terrain_difficulty"]) == 0.0]
        if len(selected) != 5 or sorted(int(row["seed"]) for row in selected) != list(range(5)):
            raise ValueError(f"Missing strafe rows: {data}")
        dr_inputs.append({"policy": family, "source_clip": file_record(source, assets),
                          "csv": file_record(data, assets), "matched_strafe_rows": selected,
                          "falls": sum(row["fell"].lower() == "true" for row in selected),
                          "episodes": len(selected)})
    interrupted_log = logs / "dr-ladder-20935266_4.out"
    iterations = re.findall(r"Learning iteration\s+(\d+)/6000", interrupted_log.read_text())
    last_iteration = max(int(value) for value in iterations)
    if last_iteration != 5494 or len(iterations) != 5495:
        raise ValueError(f"Unexpected last logged iteration: {last_iteration}")
    dr = {"schema": "bhl-media-provenance/v1", "output": gif_record(repo / "docs/gifs/dr_pair.gif", repo),
          "source_selection_script": {"path": "scripts/make_gifs.py", "sha256":
                                      digest(repo / "scripts/make_gifs.py")},
          "panels": dr_inputs, "command": {"vx_m_s": 0.0, "vy_m_s": 0.2, "wz_rad_s": 0.0},
          "scope": "Illustrative 12-DoF learned-policy MuJoCo strafe on flat ground; one filmed rollout per policy. CSV rows are the five matching evaluation seeds, not five filmed trials. The shorter failure panel holds its final frame.",
          "interrupted_training_seed": {"policy": "dr-default-s1", "job": "20935266_4",
                                         "last_logged_iteration": last_iteration,
                                         "logged_iterations": len(iterations),
                                         "scheduled_iterations": 6000, "log_filename": interrupted_log.name,
                                         "log_sha256": digest(interrupted_log),
                                         "note": "One policy in the historical three-seed table stopped early; the shown clip uses seed 0, not this interrupted seed."},
          "limits": "Hashes identify the surviving source MP4s selected by the committed assembly script. The GIF was not regenerated, and historical renderer/encoder determinism was not tested."}
    write(repo / "docs/gifs/dr_pair.json", dr)
    terrain = {}
    for family, relative, seeds in (("22DoF", "results/arms/arms-dr1.0-s{seed}__p0_d1.00.csv", 2),
                                    ("12DoF", "results/terrain/dr-default-s{seed}__d1.00.csv", 3)):
        records, all_rows = [], []
        for seed in range(seeds):
            path = assets / relative.format(seed=seed)
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 30 or any(float(row["terrain_difficulty"]) != 1.0 for row in rows):
                raise ValueError(f"Unexpected terrain cell: {path}")
            records.append(file_record(path, assets))
            all_rows.extend(rows)
        terrain[family] = {"training_policies": seeds, "episodes": len(all_rows),
                           "falls": sum(row["fell"].lower() == "true" for row in all_rows),
                           "inputs": records}
    receipt = {"schema": "bhl-media-task-closeout/v1", "date": "2026-10-08",
               "published_base": base_commit,
               "no_new_simulation_episodes": True, "NAV-02": {"sidecar": "docs/gifs/isaac/maze_stereo_fixed.json",
               "sidecar_sha256": digest(repo / "docs/gifs/isaac/maze_stereo_fixed.json")},
               "LOC-01": {"sidecar": "docs/gifs/dr_pair.json", "sidecar_sha256": digest(repo / "docs/gifs/dr_pair.json")},
               "LOC-06": {"terrain_difficulty": 1.0, "recounted": terrain},
               "generator_sha256": digest(Path(__file__)),
               "runtime": {"python": platform.python_version(),
                           **{name: importlib.metadata.version(name)
                              for name in ("numpy", "Pillow", "tensorboard")}},
               "visual_check": "Corrected stereo first frame inspected: the complete seed-0 title is visible within the 560-pixel image.",
               "limits": "Media provenance and reporting corrections; no new trained policy, performance improvement, real-time guarantee or hardware result."}
    write(args.receipt, receipt)
    print(json.dumps({"status": "PASS", "frame_files": len(manifest), "event_files": len(event_rows),
                      "terrain": {key: {k: v for k, v in value.items() if k != "inputs"}
                                  for key, value in terrain.items()}}))


if __name__ == "__main__":
    main()
