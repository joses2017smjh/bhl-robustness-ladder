#!/usr/bin/env python3
"""EuRoC pinhole/radtan calibration and original-clock prefix preparation.

The native runner requires a same-source 40-pair smoke before full replay.
Input preparation or synthetic tests do not count as dataset validation.
No ground truth enters this adapter or native inference.
"""
from __future__ import annotations
import argparse
import csv
import json
import hashlib
import os
import time
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bhl_robust.research.native_orb import NativeOrbClient, sha256
from bhl_robust.research.native_orb_inertial import ImuSettings, partition_imu, write_stereo_inertial_settings
from bhl_robust.research.pose_metrics import transform
from tumvi_native_validation import camera_matrix, read_imu, relative_seconds, selected_timestamps, summarize, write_json
from tumvi_native_evaluate import map_epochs

DATASET = "ETH EuRoC MAV V1_01_easy"
RUNTIME_SHA256 = "91ca4b6e4ea8f318998f400d4f7dfcd6fe8ae88b375399c79d5d633c39d71516"


def checked_transform(sensor):
    value = sensor["T_BS"]
    if value.get("rows") != 4 or value.get("cols") != 4 or len(value.get("data", [])) != 16:
        raise ValueError("EuRoC T_BS must be a declared 4x4 matrix")
    return transform(np.asarray(value["data"], dtype=float).reshape(4, 4))


def rectification(cam0, cam1, imu):
    """Return maps and T_imu_rectified_left; T_BS maps sensor into body."""
    import cv2
    for camera in (cam0, cam1):
        if camera["camera_model"] != "pinhole" or camera["distortion_model"] != "radial-tangential":
            raise ValueError("EuRoC pinhole radial-tangential calibration required")
        if camera["resolution"] != [752, 480] or camera["rate_hz"] != 20:
            raise ValueError("Declared EuRoC 752x480 at 20 Hz required")
        if len(camera["distortion_coefficients"]) != 4:
            raise ValueError("Four original radial-tangential coefficients required")
        intrinsics = np.asarray(camera["intrinsics"], dtype=float)
        if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all() or np.any(intrinsics[:2] <= 0):
            raise ValueError("Finite camera calibration with positive focal length required")
        if not np.isfinite(camera["distortion_coefficients"]).all():
            raise ValueError("Finite distortion calibration required")
    if imu["rate_hz"] != 200:
        raise ValueError("Original EuRoC 200 Hz IMU calibration required")
    t_b_c0, t_b_c1, t_b_i = map(checked_transform, (cam0, cam1, imu))
    t_c1_c0 = transform(np.linalg.inv(t_b_c1) @ t_b_c0)
    k0, k1 = camera_matrix(cam0), camera_matrix(cam1)
    d0, d1 = [np.asarray(c["distortion_coefficients"], dtype=float) for c in (cam0, cam1)]
    r0, r1, p0, p1, q, roi0, roi1 = cv2.stereoRectify(
        k0, d0, k1, d1, (752, 480), t_c1_c0[:3, :3], t_c1_c0[:3, 3],
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    baseline = -float(p1[0, 3] / p1[0, 0])
    if not np.isfinite(baseline) or baseline <= 0 or abs(p1[1, 3]) > 1e-8 or not np.allclose(p0[:3,:3], p1[:3,:3], atol=1e-8):
        raise ValueError("Positive horizontal baseline and equal rectified intrinsics required")
    original_from_rectified = np.eye(4)
    original_from_rectified[:3,:3] = r0.T
    t_i_c = transform(np.linalg.inv(t_b_i) @ t_b_c0 @ original_from_rectified)
    maps = [cv2.initUndistortRectifyMap(k, d, r, p, (752,480), cv2.CV_32FC1)
            for k,d,r,p in ((k0,d0,r0,p0),(k1,d1,r1,p1))]
    stereo = dict(fx=float(p0[0,0]),fy=float(p0[1,1]),cx=float(p0[0,2]),cy=float(p0[1,2]),
                  width=752,height=480,fps=20,baseline_m=baseline)
    receipt = {"T_imu_rectified_left_camera":t_i_c.tolist(), "T_right_original_left_camera":t_c1_c0.tolist(),
               "R_left_rectification":r0.tolist(),"R_right_rectification":r1.tolist(),
               "P_left":p0.tolist(),"P_right":p1.tolist(),"Q":q.tolist(),
               "valid_roi_left":list(roi0),"valid_roi_right":list(roi1),
               "native_stereo_settings":stereo,"rectifier":"cv2.stereoRectify, CALIB_ZERO_DISPARITY, alpha=0",
               "transform_contract":"T_BS maps original sensor to body; T_imu_rectified_camera=inverse(T_body_imu) @ T_body_originalcamera @ R_rectification.T"}
    noise = {"noise_gyro":float(imu["gyroscope_noise_density"]),
             "noise_acc":float(imu["accelerometer_noise_density"]),
             "gyro_walk":float(imu["gyroscope_random_walk"]),
             "acc_walk":float(imu["accelerometer_random_walk"])}
    if any(not np.isfinite(x) or x <= 0 for x in noise.values()):
        raise ValueError("Positive finite original IMU noise calibration required")
    return maps, stereo, t_i_c, noise, receipt


def original_image(image):
    if image is None or image.dtype != np.uint8 or image.shape != (480,752):
        raise ValueError("Original EuRoC 752x480 uint8 monochrome image required")
    return image


def camera_rows(path):
    result = {}
    with Path(path).open() as stream:
        for row in csv.reader(stream):
            if not row or row[0].startswith("#"):
                continue
            if len(row) != 2:
                raise ValueError("Exactly timestamp and image filename required")
            timestamp = int(row[0])
            filename = row[1].strip()
            if timestamp in result or filename != f"{timestamp}.png":
                raise ValueError("Duplicate camera time or noncanonical original image filename")
            result[timestamp] = filename
    return result


def bounded_path(root, relative):
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Dataset inputs must stay within the declared directory")
    return path


def prepare(dataset, protocol_path, output):
    import cv2
    import yaml
    if output.exists():
        raise ValueError("Selection output must be new")
    protocol = json.loads(protocol_path.read_text())
    if protocol["dataset"] != DATASET or protocol["runtime_sha256"] != RUNTIME_SHA256:
        raise ValueError("Exact preregistered EuRoC dataset and runtime required")
    cameras = [camera_rows(bounded_path(dataset, f"mav0/cam{c}/data.csv")) for c in (0,1)]
    origin, stamps = selected_timestamps(*cameras)
    if len(stamps) < 2:
        raise ValueError("At least two exact original camera pairs required")
    calibration = [yaml.safe_load(bounded_path(dataset, f"mav0/{name}/sensor.yaml").read_text())
                   for name in ("cam0","cam1","imu0")]
    _, _, _, noise, rect = rectification(*calibration)
    samples, imu_ns = read_imu(bounded_path(dataset,"mav0/imu0/data.csv"), origin, stamps[-1])
    _, causality = partition_imu(relative_seconds(stamps,origin), samples)
    files = {}
    names = [f"mav0/{name}/sensor.yaml" for name in ("cam0","cam1","imu0")]
    names += [f"mav0/{name}/data.csv" for name in ("cam0","cam1","imu0")]
    for c in (0,1):
        for stamp in stamps:
            name = f"mav0/cam{c}/data/{stamp}.png"
            original_image(cv2.imread(str(bounded_path(dataset,name)), cv2.IMREAD_UNCHANGED))
            names.append(name)
    for name in names:
        path = bounded_path(dataset,name)
        files[name] = {"bytes":path.stat().st_size,"sha256":sha256(path)}
    result = {"schema":"bhl-euroc-selected-input-v1","status":"INPUT_PREPARED_NATIVE_UNRUN",
              "dataset":DATASET,"protocol_sha256":sha256(protocol_path),"clock_origin_ns":origin,
              "selected_original_timestamps_ns":stamps,"paired_frames":len(stamps),
              "last_relative_seconds":(stamps[-1]-origin)*1e-9,"files":files,
              "imu_original_rows":len(imu_ns),"imu_causality":causality,"imu_noise":noise,
              "rectification":rect,"truth_inputs_to_estimator":False,"hardware_validation":False}
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


def source_identity():
    paths = sorted([*ROOT.joinpath("src/bhl_robust/research").rglob("*.py"),
                    *ROOT.joinpath("scripts/bench").glob("*.py")])
    files = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    return {"files":files,"sha256":hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}


def check_smoke(receipt, protocol, selection, fingerprint):
    if receipt is None or not receipt.is_file():
        raise ValueError("A completed same-source native smoke receipt is required")
    smoke=json.loads(receipt.read_text())
    if (smoke.get("schema") != "bhl-external-euroc-result-v1" or smoke.get("status") != "COMPLETED"
        or smoke.get("stage") != "smoke" or smoke.get("scored") is not False
        or smoke.get("paired_frames_per_arm") != 40 or smoke.get("runtime_sha256") != RUNTIME_SHA256
        or smoke.get("protocol_sha256") != sha256(protocol) or smoke.get("selection_sha256") != sha256(selection)
        or smoke.get("source_identity") != fingerprint
        or len(smoke.get("arms",[])) != 2
        or {a.get("mode") for a in smoke["arms"]} != {"stereo","stereo_inertial"}
        or any(a.get("frames") != 40 for a in smoke["arms"])):
        raise ValueError("Native smoke identity/completeness differs from the requested full replay")
    for arm in smoke["arms"]:
        work = receipt.parent / arm["mode"]
        if (sha256(work / "frames.jsonl") != arm["frames_sha256"]
            or sha256(work / "settings.yaml") != arm["settings_sha256"]
            or len((work / "frames.jsonl").read_text().splitlines()) != 40):
            raise ValueError("Native smoke trace/settings checksum or row count mismatch")
    return smoke


def run(args):
    import cv2
    import yaml
    from native_stereo_navigation_campaign import unpack_runtime
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    if protocol["runtime_sha256"] != RUNTIME_SHA256 or protocol["dataset"] != DATASET:
        raise ValueError("Exact predeclared dataset and runtime required")
    source = json.loads(args.selection.read_text())
    if source.get("schema") != "bhl-euroc-selected-input-v1" or source.get("dataset") != DATASET:
        raise ValueError("EuRoC adapter selected-input receipt required")
    stamps = source["selected_original_timestamps_ns"]
    origin, selected = selected_timestamps(*[camera_rows(bounded_path(args.dataset, f"mav0/cam{c}/data.csv")) for c in (0, 1)])
    if selected != stamps or origin != source["clock_origin_ns"]:
        raise ValueError("Dataset selection differs from declared integer-clock prefix")
    required = {f"mav0/{name}/{file}" for name in ("cam0", "cam1", "imu0")
                for file in ("sensor.yaml", "data.csv")}
    required.update(f"mav0/cam{camera}/data/{stamp}.png" for camera in (0, 1) for stamp in stamps)
    if set(source["files"]) != required or source.get("paired_frames") != len(stamps):
        raise ValueError("Complete original input manifest and matched frame count required")
    for name, expected in source["files"].items():
        path = bounded_path(args.dataset, name)
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError("Original selected input hash mismatch")
    calibration = [yaml.safe_load(bounded_path(args.dataset, f"mav0/{name}/sensor.yaml").read_text()) for name in ("cam0", "cam1", "imu0")]
    rectification_began = time.monotonic()
    maps, stereo, t_i_c, noise, rect_receipt = rectification(*calibration)
    rect_receipt["one_time_map_construction_seconds"] = time.monotonic() - rectification_began
    if source["protocol_sha256"] != sha256(args.protocol):
        raise ValueError("Selected input protocol mismatch")
    fingerprint = source_identity()
    if args.stage == "full":
        check_smoke(args.smoke_receipt, args.protocol, args.selection, fingerprint)
    else:
        stamps = stamps[:40]
        if len(stamps) != 40:
            raise ValueError("Native execution smoke requires 40 original camera pairs")
    write_json(args.output / "rectification.json", rect_receipt)
    samples, imu_ns = read_imu(args.dataset / "mav0/imu0/data.csv", origin, stamps[-1])
    times = relative_seconds(stamps, origin)
    batches, imu_receipt = partition_imu(times, samples)
    imu_receipt.update({"first_original_ns": imu_ns[0], "last_original_ns": imu_ns[-1],
                        "clock_origin_ns": origin, "no_additional_clock_offset": True,
                        "publisher_corrected_measurements": False, "measurement_source": "Original EuRoC measured SI vectors; no bias subtraction or interpolation"})
    write_json(args.output / "imu-causality.json", imu_receipt)
    runtime = unpack_runtime(args.runtime_archive, RUNTIME_SHA256, args.output / "private-runtime")
    environment = {**os.environ, "LD_LIBRARY_PATH": str(runtime / "lib"),
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    summaries = []
    for mode in ("stereo", "stereo_inertial"):
        work = args.output / mode
        work.mkdir()
        settings = work / "settings.yaml"
        if mode == "stereo_inertial":
            write_stereo_inertial_settings(settings, t_imu_camera=t_i_c,
                                           imu=ImuSettings(**noise), **stereo)
        else:
            from bhl_robust.research.native_orb import write_rectified_stereo_settings
            write_rectified_stereo_settings(settings, **stereo)
        frames, timings = [], []
        ranges = {"original_uint8_min": 255, "original_uint8_max": 0,
                  "rectified_uint8_min": 255, "rectified_uint8_max": 0,
                  "conversion": "Original uint8 preserved; no intensity rescaling"}
        began = time.monotonic()
        with NativeOrbClient(runtime / "bin/orb_native", runtime / "ORBvoc.txt", settings,
                             work / "private-native", environment=environment, sensor_mode=mode) as client:
            startup = time.monotonic() - began
            with (work / "frames.jsonl").open("x") as stream:
                for index, (stamp, timestamp) in enumerate(zip(stamps, times)):
                    began = time.monotonic()
                    outputs = []
                    for camera in (0, 1):
                        raw = cv2.imread(str(args.dataset / f"mav0/cam{camera}/data/{stamp}.png"), cv2.IMREAD_UNCHANGED)
                        converted = original_image(raw)
                        ranges["original_uint8_min"] = min(ranges["original_uint8_min"], int(raw.min()))
                        ranges["original_uint8_max"] = max(ranges["original_uint8_max"], int(raw.max()))
                        rectified = cv2.remap(converted, *maps[camera], cv2.INTER_LINEAR)
                        ranges["rectified_uint8_min"] = min(ranges["rectified_uint8_min"], int(rectified.min()))
                        ranges["rectified_uint8_max"] = max(ranges["rectified_uint8_max"], int(rectified.max()))
                        destination = work / f"current-cam{camera}.png"
                        if not cv2.imwrite(str(destination), rectified):
                            raise ValueError("Rectified image write failed")
                        outputs.append(destination)
                    image_end = time.monotonic()
                    frame = client.track(float(timestamp), *outputs,
                        **({"imu_samples": batches[index]} if mode == "stereo_inertial" else {}))
                    end = time.monotonic()
                    timing = {"pipeline_wall_s": end - began,
                              "rectification_image_io_s": image_end - began,
                              "native_client_wall_s": end - image_end}
                    frames.append(frame)
                    timings.append(timing)
                    stream.write(json.dumps({"original_timestamp_ns": stamp, "native": frame, "timing": timing}) + "\n")
                    stream.flush()
                    if (index + 1) % 100 == 0:
                        print(json.dumps({"mode": mode, "frames": index + 1}), flush=True)
            summary = summarize(frames, timings, mode)
            summary["ate"] = {"status": "UNMEASURED", "reason": "No independently verified EuRoC reference-frame and synchronized ground-truth evaluator supplied; reference data is neither read nor hashed by this runner."}
            epochs, boundaries = map_epochs(frames)
            summary.update({"coordinate_epochs": len(epochs), "observed_epoch_boundaries": boundaries,
                            "continuous_single_map_trajectory": len(epochs) == 1 and not boundaries})
            summary["pipeline_scope"] = "Original uint8 PNG read + fixed radtan rectification + PNG write + native client transport, IMU TSV I/O and native TrackStereo"
            summary.update({"startup_wall_seconds": startup, "ready": client.ready,
                            "pixel_ranges": ranges, "frames_sha256": sha256(work / "frames.jsonl"),
                            "settings_sha256": sha256(settings)})
            write_json(work / "result.json", summary)
            summaries.append(summary)
    for name, expected in source["files"].items():
        if sha256(bounded_path(args.dataset, name)) != expected["sha256"]:
            raise ValueError("Original input changed during replay; partial outputs retained")
    result = {"schema": "bhl-external-euroc-result-v1", "status": "COMPLETED",
              "scope": protocol["scope"], "protocol_sha256": sha256(args.protocol),
              "selection_sha256": sha256(args.selection), "calibration_hashes": {name: sha256(args.dataset / f"mav0/{name}/sensor.yaml") for name in ("cam0", "cam1", "imu0")},
              "stage": args.stage, "scored": args.stage == "full", "source_identity": fingerprint,
              "smoke_receipt_sha256": sha256(args.smoke_receipt) if args.stage == "full" else None,
              "source_sha256": sha256(Path(__file__)), "clock_origin_ns": origin,
              "paired_frames_per_arm": len(stamps), "window_last_sensor_time_s": float(times[-1]),
              "runtime_sha256": RUNTIME_SHA256, "arms": summaries,
              "simulation": False, "external_recorded_data": True, "own_hardware_validation": False,
              "lidar_validation": False, "physical_navigation_validation": False,
              "opencv_version": cv2.__version__, "original_data_loaded_by_estimator": "calibrated camera PNGs and genuine IMU only"}
    write_json(args.output / "result.json", result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    prep=sub.add_parser("prepare")
    replay=sub.add_parser("run")
    for command in (prep,replay):
        for name in ("dataset","protocol","output"):
            command.add_argument("--"+name,type=Path,required=True)
    for name in ("selection","runtime-archive"):
        replay.add_argument("--"+name,type=Path,required=True)
    replay.add_argument("--stage",choices=("smoke","full"),required=True)
    replay.add_argument("--smoke-receipt",type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError("Output must be new; old evidence is immutable")
    if args.command == "prepare":
        prepare(args.dataset,args.protocol,args.output)
    else:
        try:
            run(args)
        except Exception as error:
            args.output.mkdir(parents=True,exist_ok=True)
            write_json(args.output/"failure.json",{"status":"FAILED","exception":type(error).__name__,
                       "message":str(error),"stage":args.stage,"partial_native_outputs_retained":True})
            raise

if __name__ == "__main__":
    main()
