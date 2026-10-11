#!/usr/bin/env python3
"""Fresh grouped capture -> real stereo -> ground maps -> three native estimators."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from sensor_campaign import unpack_runtime
from h34_campaign import safe_extract


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def benchmark(dataset, model_path, output, *, smoke):
    from bhl_robust.research.stereo_benchmark import (
        load_replay, load_rgb, SGBMMethod, FastFoundationStereoMethod, depth_metrics, timed_replay)
    from bhl_robust.research.terrain_methods import ground_map_metrics
    from bhl_robust.research.sensor_map_methods import stereo_points, measured_map, fuse_maps, hazard_coverage
    manifest, calibration = load_replay(dataset)
    frames = manifest["sequences"][0]["frames"]
    output.mkdir()
    methods, initialization = {}, {}
    for name, constructor in (("sgbm", lambda: SGBMMethod(calibration)),
                              ("c_fast_foundationstereo", lambda: FastFoundationStereoMethod(calibration, model_path))):
        start = time.perf_counter()
        methods[name] = constructor()
        initialization[name] = 1000*(time.perf_counter()-start)
    pairs = [(dataset/f["left"], dataset/f["right"]) for f in frames]
    latency = {name: timed_replay(method, pairs, warmup=2 if smoke else 10, timed=5 if smoke else 50)
               for name, method in methods.items()}
    rows = []
    for frame in frames:
        left, right = load_rgb(dataset/frame["left"]), load_rgb(dataset/frame["right"])
        predictions = {name: method(left, right) for name, method in methods.items()}
        common = predictions["sgbm"][1] & predictions["c_fast_foundationstereo"][1]
        with np.load(dataset/frame["lidar"], allow_pickle=False) as scan:
            lidar_map = measured_map(scan["points_xyz_m"], np.asarray(manifest["calibration"]["T_M_L"]))
        maps = {"lidar": lidar_map}
        packet = {}
        # Estimator inputs and map construction finish before evaluator truth opens.
        for name, (depth, valid) in predictions.items():
            points, variance_z, transform = stereo_points(depth, valid, manifest["calibration"])
            start = time.perf_counter()
            mapped = measured_map(points, transform, point_variance=variance_z)
            maps[name] = mapped
            maps[name+"_simple_fusion"] = fuse_maps(mapped, lidar_map, gated=False)
            maps[name+"_gated_fusion"] = fuse_maps(mapped, lidar_map, gated=True)
            packet[name] = dict(mapping_and_both_fusions_ms=1000*(time.perf_counter()-start))
        with np.load(dataset/frame["truth"], allow_pickle=False) as truth:
            for name, (depth, valid) in predictions.items():
                packet[name].update(native_support=depth_metrics(depth, valid, truth["depth_z_m"], obstacle_mask=truth["obstacle_mask"]),
                    common_support=depth_metrics(depth, valid, truth["depth_z_m"], common_mask=common, obstacle_mask=truth["obstacle_mask"]))
                np.savez_compressed(output/(frame["frame_id"]+"-"+name+"-depth.npz"), depth_m=depth, valid=valid)
            terrain = {}
            for name, snapshot in maps.items():
                terrain[name] = ground_map_metrics(snapshot, truth["ground_height_m"], truth["ground_mask"], truth_hazard=truth["ground_hazard"])
                terrain[name].update(hazard_coverage(snapshot, truth["ground_hazard"], truth["ground_mask"] & truth["ground_hazard_support"]))
                terrain[name]["sensor_conflict_cells"] = snapshot.get("sensor_conflict_cells", 0)
                np.savez_compressed(output/(frame["frame_id"]+"-"+name+"-map.npz"),
                    **{k:v for k,v in snapshot.items() if isinstance(v, np.ndarray)})
        rows.append(dict(frame_id=frame["frame_id"], depth=packet, terrain=terrain))
        print(json.dumps({"map_frame": frame["frame_id"]}), flush=True)
    result = dict(schema="bhl-sensor-map-methods-v1", status="PASS", frames=len(frames), rows=rows,
        depth_latency=latency, cold_session_creation_ms=initialization,
        provenance={name: method.provenance() for name, method in methods.items()},
        sensor_uncertainty="one-pixel disparity standard deviation projected along optical ray; fixed LiDAR, attitude and pose floors; measured interval coverage, no calibration claim",
        map_scope="instantaneous rig-local ground maps with fixed extrinsics; no ground-truth registration or estimated-pose accumulation",
        metric_scope="obstacle PIXEL recall, ground height and coverage, conditional hazard recall plus unresolved-inclusive recall; no closed-loop outcomes")
    write(output/"metrics.json", result)
    return dict(status="PASS", frames=len(frames), methods=list(methods), mapping_arms=list(maps),
                gpu_execution=methods["c_fast_foundationstereo"].provenance())


def run(protocol, cell_index, phase, output, work):
    p = json.loads(protocol.read_text())
    settings = p["sensor_methods"]
    cell = settings["cells"][cell_index]
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    result = dict(schema="bhl-sensor-methods-cell-v1", status="INCOMPLETE", phase=phase, cell=cell,
        protocol_sha256=digest(protocol), started_utc=datetime.now(timezone.utc).isoformat(),
        scientific_status="SMOKE_ONLY" if phase == "smoke" else "DEVELOPMENT_CELL_NO_GLOBAL_VERDICT",
        scope="simulated sensor replay, no physical or closed-loop improvement claim")
    write(output/"campaign_result.json", result)
    try:
        inputs = protocol.parent/"inputs"
        for name, specification in p["input_files"].items():
            if digest(inputs/name) != specification["sha256"]:
                raise ValueError("frozen sensor input changed: "+name)
        wheel = settings["ort_input"]
        unpack_runtime(inputs/wheel, p["input_files"][wheel]["sha256"], work/"ort")
        sys.path.insert(0, str(work/"ort"))
        from methods_sensor_capture import capture
        frames = settings["smoke_frames"] if phase == "smoke" else settings["frames"]
        manifest = capture(output/"replay", cell, frames=frames, protocol=protocol)
        result["capture"] = dict(frames=frames, manifest_sha256=digest(output/"replay/manifest.json"))
        result["stereo_mapping"] = benchmark(output/"replay", inputs/settings["model_input"], output/"mapping", smoke=phase == "smoke")
        for name in ("lio", "livo"):
            safe_extract(inputs/settings[name+"_input"], work/name)
        import native_livo_methods_campaign as native
        plan = dict(schema="bhl-native-livo-methods-v1", methods=["fast_lio2", "fast_livo2", "fast_livo2_no_visual"],
                    sequences=[cell["id"]], gate=settings["replay_gate"])
        write(output/"native-plan.json", plan)
        lio_runtime = work/"lio/runtime"
        if not lio_runtime.exists():
            lio_runtime = work/"lio"
        (lio_runtime/"fastlio_headless").chmod(0o700)
        (work/"livo/runtime/fastlivo_headless").chmod(0o700)
        native_result = native.run(output/"replay/manifest.json", lio_runtime, work/"livo/runtime",
                                   output/"native-plan.json", output/"native")
        result["native"] = {k:v for k,v in native_result.items() if k != "rows"}
        result["native_visual_measurement_frames"] = sum(row["native"].get("frames_with_visual_measurements", 0)
            for row in native_result["rows"] if row["method"] == "fast_livo2")
        if phase == "smoke" and result["native_visual_measurement_frames"] == 0:
            raise RuntimeError("smoke did not exercise actual native photometric measurements")
        result.update(status="PASS", acceptance_meaning="all actual methods executed; qualification retained per native cell and improvement requires complete paired cohort")
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write(output/"campaign_result.json", result)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cell", type=int, default=0)
    parser.add_argument("--phase", choices=("smoke", "run"), required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.protocol, args.cell, args.phase, Path(os.environ["H34_OUTPUT_DIR"]), Path(os.environ["H34_WORK_DIR"])))
