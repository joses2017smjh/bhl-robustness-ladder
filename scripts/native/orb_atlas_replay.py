#!/usr/bin/env python3
"""Open-loop native smoke of the Atlas-telemetry ORB wrapper on recorded v1 frames.

Replays the exact stereo pairs, capture timestamps and native settings that a
retained processing-schedule v1 bounded-latest stereo episode sent to ORB-SLAM3
(``sensors/selected-inference`` PNGs and ``sensors/processing_schedule.json``
requests), through one or more pinned native runtimes. Each runtime gets a fresh
native process. Every response is validated (including the Atlas block) and
retained; both the legacy map-point epoch tracker and the Atlas-aware tracker
are evaluated on the same responses.

This is an instrumentation/execution smoke, not navigation: the camera motion is
fixed by the recording and ORB-SLAM3 threading is nondeterministic, so a loss or
relocalization observed in closed loop need not recur. ``--blackout START END``
replaces the recorded pairs captured in [START, END) with all-black images of the
same size (the predeclared visual-blackout acquisition fault), which forces a
deterministic tracking loss: early (few keyframes) it triggers ORB-SLAM3's
same-id active-map reset, briefly later a relocalization, and for more than 3 s
later a new Atlas map. Run inside the Ubuntu SIF.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/bench"))
from bhl_robust.research.native_orb import NativeOrbClient, TRACKING_STATES, UPSTREAM_COMMIT, sha256, tracking_summary
from bhl_robust.research.navigation_sensor_controls import NativeCoordinateEpochTracker
from bhl_robust.research.stereo_relocalization import AtlasCoordinateEpochTracker


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def extract_episode(shard_archive, episode, destination):
    """Extract only one episode's settings, request ledger and selected PNG pairs."""
    wanted = (f"{episode}/native/stereo.yaml", f"{episode}/sensors/processing_schedule.json",
              f"{episode}/native/orb/native-orb.log", f"{episode}/native/native_frames.jsonl")
    prefix = f"{episode}/sensors/selected-inference/"
    with tarfile.open(shard_archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name in wanted or (m.name.startswith(prefix) and m.name.endswith(".png"))]
        for member in members:
            name = Path(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts:
                raise ValueError("unsafe recorded episode member")
            target = Path(destination) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as stream, target.open("xb") as out:
                out.write(stream.read())
    root = Path(destination) / episode
    for name in wanted:
        if not (Path(destination) / name).is_file():
            raise ValueError("recorded episode lacks " + name)
    return root


def black_out(frames, start_s, end_s, directory):
    """Replace pairs captured in [start_s, end_s) with same-size all-black PNGs."""
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=False)
    replaced = []
    for frame in frames:
        if start_s <= frame["timestamp_s"] < end_s:
            for side in ("left", "right"):
                with Image.open(frame[side]) as image:
                    size, mode = image.size, image.mode
                target = directory / f'{frame["camera_id"]:05d}-{side}.png'
                Image.new(mode, size).save(target)
                frame[side] = target
            frame["blackout"] = True
            replaced.append(frame["camera_id"])
    if not replaced:
        raise ValueError("blackout window contains no recorded frame")
    return replaced


def recorded_frames(root):
    schedule = json.loads((root / "sensors/processing_schedule.json").read_text())
    if schedule.get("pipeline") != "bounded_latest":
        raise ValueError("bounded-latest recorded stereo requests required")
    frames, previous = [], None
    for request in schedule["requests"]:
        camera = request["camera_id"]
        if camera == previous:
            continue  # only unique camera frames reach the native estimator
        previous = camera
        pair = [root / "sensors/selected-inference" / f"{camera:05d}-{side}.png" for side in ("left", "right")]
        if not all(p.is_file() for p in pair):
            raise ValueError("selected recorded stereo pair missing")
        frames.append({"timestamp_s": float(request["camera_capture_s"]), "left": pair[0], "right": pair[1],
                       "camera_id": camera})
    original = [json.loads(line) for line in (root / "native/native_frames.jsonl").read_text().splitlines()]
    original = [r for r in original if r.get("new_native_camera_frame")]
    if [round(r["timestamp_s"], 9) for r in original] != [round(f["timestamp_s"], 9) for f in frames]:
        raise ValueError("recorded requests differ from the retained native frame ledger")
    return frames, original


def unpack(archive, expected, destination):
    import importlib.util
    spec = importlib.util.spec_from_file_location("orb_replay_launcher", ROOT / "scripts/bench/h34_campaign.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    if sha256(archive) != expected:
        raise ValueError("native runtime archive checksum differs")
    destination.mkdir(parents=True)
    launcher.safe_extract(archive, destination)
    runtime = destination / "runtime"
    receipt = json.loads((runtime / "runtime.json").read_text())
    if receipt.get("status") != "PASS" or receipt.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("verified pinned native runtime required")
    for name, digest in receipt["files_sha256"].items():
        if sha256(runtime / name) != digest:
            raise ValueError("native runtime member changed: " + name)
    (runtime / "bin/orb_native").chmod(0o700)
    return runtime


def replay(runtime, root, frames, output, *, pace):
    env = dict(os.environ, LD_LIBRARY_PATH=str(runtime / "lib"), OMP_NUM_THREADS="1")
    rows = []
    legacy, atlas = NativeCoordinateEpochTracker(), AtlasCoordinateEpochTracker()
    started = time.monotonic()
    with NativeOrbClient(runtime / "bin/orb_native", runtime / "ORBvoc.txt", root / "native/stereo.yaml",
                         output / "process", environment=env) as client:
        ready = client.ready
        epoch_wall, epoch_sensor = time.monotonic(), frames[0]["timestamp_s"]
        for frame in frames:
            delay = epoch_wall + pace * (frame["timestamp_s"] - epoch_sensor) - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            response = client.track(frame["timestamp_s"], frame["left"], frame["right"])
            row = dict(response, camera_id=frame["camera_id"], legacy_epoch=legacy.observe(response))
            if response.get("atlas") is not None:
                row["atlas_epoch"] = atlas.observe(response)
            rows.append(row)
    with (output / "native_frames.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return rows, ready, legacy.receipt(), atlas.receipt(), time.monotonic() - started


def timeline(rows):
    changes, previous = [], None
    for row in rows:
        a = row.get("atlas") or {}
        key = (row["tracking_state"], row["map_id"], a.get("current_map_id"), a.get("current_map_init_kf_id"),
               a.get("atlas_map_count"), a.get("current_map_big_change_idx"), a.get("active_map_resets_total"))
        if key != previous:
            changes.append({"timestamp_s": row["timestamp_s"], "state": TRACKING_STATES[row["tracking_state"]],
                            "legacy_map_id": row["map_id"], "atlas_map_id": key[2], "init_kf_id": key[3],
                            "map_count": key[4], "big_change_idx": key[5], "resets_total": key[6],
                            "keyframes": a.get("current_map_keyframes")})
            previous = key
    return changes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-archive", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--runtime", action="append", nargs=3, metavar=("LABEL", "ARCHIVE", "SHA256"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pace", type=float, default=1.5, help="wall seconds per recorded sensor second")
    parser.add_argument("--blackout", type=float, nargs=2, metavar=("START_S", "END_S"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    work = args.output / "work"
    root = extract_episode(args.shard_archive, args.episode, work / "recorded")
    frames, original = recorded_frames(root)
    blackout = black_out(frames, *args.blackout, work / "blackout") if args.blackout else []
    original_log = (root / "native/orb/native-orb.log").read_text(errors="replace")
    original_rows = [{"tracking_state": r["native_ORB_tracking_state"], "map_id": r["native_ORB_map_id"],
                      "timestamp_s": r["timestamp_s"]} for r in original]
    legacy = NativeCoordinateEpochTracker()
    for row in original_rows:
        legacy.observe(row)
    result = {"schema": "bhl-orb-atlas-recorded-replay-v1", "status": "INCOMPLETE",
              "shard_archive": str(args.shard_archive), "shard_archive_sha256": sha256(args.shard_archive),
              "episode": args.episode, "frames": len(frames), "pace_wall_per_sensor_second": args.pace,
              "blackout_window_s": args.blackout, "blackout_camera_ids": blackout,
              "original_closed_loop": {"timeline": timeline(original_rows),
                                       "legacy_invalidations": legacy.receipt()["events"],
                                       "log_relocalized": original_log.count("Relocalized!!"),
                                       "log_new_maps": original_log.count("Creation of new map with id:") - 1,
                                       "log_active_map_resets": original_log.count("LM: Active map reset recieved")},
              "runs": {}, "scope": __doc__.strip().splitlines()[0]}
    for label, archive, expected in args.runtime:
        runtime = unpack(Path(archive), expected, work / f"runtime-{label}")
        output = args.output / label
        output.mkdir()
        rows, ready, legacy_receipt, atlas_receipt, elapsed = replay(runtime, root, frames, output, pace=args.pace)
        log = (output / "process/native-orb.log").read_text(errors="replace")
        with_atlas = sum(r.get("atlas") is not None for r in rows)
        result["runs"][label] = {
            "runtime_archive_sha256": expected, "binary_sha256": sha256(runtime / "bin/orb_native"),
            "native_library_sha256": sha256(runtime / "lib/libORB_SLAM3.so"), "ready": ready,
            "responses": len(rows), "responses_with_atlas_telemetry": with_atlas,
            "tracking": tracking_summary([{k: v for k, v in r.items() if k not in ("camera_id", "legacy_epoch", "atlas_epoch")} for r in rows]),
            "timeline": timeline(rows), "legacy_epochs": legacy_receipt,
            "atlas_epochs": atlas_receipt if with_atlas else None,
            "wrapper_reset_flags": sum(bool((r.get("atlas") or {}).get("active_map_reset")) for r in rows),
            "map_changed_flags": sum(bool((r.get("atlas") or {}).get("system_map_changed")) for r in rows),
            "inconsistent_snapshots": sum((r.get("atlas") or {}).get("snapshot_consistent") is False for r in rows),
            "legacy_vs_atlas_map_disagreements": sum(r["map_id"] is not None and r.get("atlas") is not None
                                                     and r["map_id"] != r["atlas"]["current_map_id"] for r in rows),
            "log_relocalized": log.count("Relocalized!!"),
            "log_new_maps": log.count("Creation of new map with id:") - 1,
            "log_active_map_resets": log.count("LM: Active map reset recieved"),
            "native_compute_p95_ms": 1000 * sorted(r["compute_seconds"] for r in rows)[int(.95 * (len(rows) - 1))],
            "elapsed_wall_s": elapsed, "frames_sha256": sha256(output / "native_frames.jsonl")}
    result["status"] = "PASS"
    write_json(args.output / "replay_result.json", result)
    import shutil
    shutil.rmtree(work)
    print(json.dumps({k: v for k, v in result.items() if k != "runs"} | {
        "runs": {k: {x: v[x] for x in ("responses", "responses_with_atlas_telemetry", "log_relocalized", "log_new_maps",
                                        "log_active_map_resets", "wrapper_reset_flags")} for k, v in result["runs"].items()}},
        indent=1, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
