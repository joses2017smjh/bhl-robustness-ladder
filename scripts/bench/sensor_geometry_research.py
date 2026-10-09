#!/usr/bin/env python3
"""Replay a frozen rendered/physical stereo + 3D lidar dataset.

This pilot evaluates geometric perception, not robot traversal or hardware
performance. Five fusion arms reuse identical point-retention masks. Inputs
must live in an inference namespace; independent truth is read only after
each prediction has been produced. Missing capture/calibration/truth yields
INCOMPLETE and never falls back to input lidar as an accuracy label.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from bhl_robust.research.sensor_geometry import (FusionConfig, Pinhole, TerrainConfig,
    depth_metrics, elevation_map, fuse_depth, project_lidar, retained_indices,
    risk_coverage, stressed_extrinsic, terrain_metrics, transform_points, rigid_transform)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def namespaced_path(base: Path, value: str, namespace: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("dataset paths must be relative and contained")
    if relative.parts[0] != namespace:
        raise ValueError(f"{namespace} namespace required for {value}")
    cursor = base
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError("dataset paths cannot follow symbolic links")
    path = (base / relative).resolve()
    if not path.is_relative_to((base / namespace).resolve()):
        raise ValueError("dataset path escapes declared namespace")
    if not path.is_file() or path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("dataset artifact missing or too large")
    return path


def prediction_path(base: Path, value: str, dataset: Path) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("prediction path must be contained")
    if "evaluator" in relative.parts:
        raise ValueError("evaluation truth cannot be used as a prediction")
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError("prediction artifact missing or escaping namespace")
    if path.is_relative_to((dataset / "evaluator").resolve()):
        raise ValueError("prediction aliases evaluation truth")
    if path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("prediction artifact too large")
    return path


def load_array(path: Path) -> np.ndarray:
    result = np.load(path, allow_pickle=False)
    if not isinstance(result, np.ndarray) or result.size > 4_000_000:
        raise ValueError("prediction must be a bounded non-object NPY array")
    return result


def dense_truth_hazards(height: np.ndarray, x: np.ndarray, y: np.ndarray,
                       config: TerrainConfig) -> np.ndarray:
    """Evaluator-only hazard labels from independent dense vertical queries."""
    if not np.isfinite(height).all():
        raise ValueError("dense independent hazard labels require finite geometry")
    dy, dx = np.gradient(height, y, x)
    slope = np.degrees(np.arctan(np.hypot(dx, dy)))
    roughness = np.zeros(height.shape)
    step = np.zeros(height.shape)
    for yy, xx in np.ndindex(height.shape):
        ys = slice(max(0, yy - 1), min(len(y), yy + 2))
        xs = slice(max(0, xx - 1), min(len(x), xx + 2))
        patch = height[ys, xs]
        gx, gy = np.meshgrid(x[xs] - x[xx], y[ys] - y[yy])
        design = np.column_stack((gx.ravel(), gy.ravel(), np.ones(patch.size)))
        coefficients, _, rank, _ = np.linalg.lstsq(design, patch.ravel(), rcond=None)
        if rank < 3:
            raise ValueError("independent terrain grid is degenerate")
        roughness[yy, xx] = np.sqrt(np.square(patch.ravel() - design @ coefficients).mean())
        step[yy, xx] = np.max(np.abs(patch - height[yy, xx]))
    return ((slope >= config.slope_hazard_deg) | (roughness >= config.roughness_hazard_m) |
            (step >= config.step_hazard_m))


def interpolation_labels(truth: dict, terrain: dict,
                         config: TerrainConfig | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Nearest independent terrain sample, never interpolate across unknowns."""
    if str(np.asarray(truth.get("terrain_grid_frame", "")).item()) not in ("M_local", "M_local_z_up"):
        raise ValueError("terrain truth must declare the same local z-up frame")
    tx = np.asarray(truth["terrain_x_m"], float)
    ty = np.asarray(truth["terrain_y_m"], float)
    height = np.asarray(truth["terrain_height_m"], float)
    if tx.ndim == 2 and ty.ndim == 2 and tx.shape == ty.shape == height.shape:
        if not np.allclose(tx, tx[:1]) or not np.allclose(ty, ty[:, :1]):
            raise ValueError("terrain truth is not a rectilinear grid")
        tx, ty = tx[0], ty[:, 0]
    if tx.ndim != 1 or ty.ndim != 1 or height.shape != (len(ty), len(tx)):
        raise ValueError("invalid independent terrain grid")
    if min(len(tx), len(ty)) < 2 or not np.isfinite(tx).all() or not np.isfinite(ty).all():
        raise ValueError("invalid independent terrain coordinates")
    if np.any(np.diff(tx) <= 0) or np.any(np.diff(ty) <= 0):
        raise ValueError("terrain truth coordinates must be increasing")
    nx = np.argmin(np.abs(tx[:, None] - terrain["x_m"][None]), axis=0)
    ny = np.argmin(np.abs(ty[:, None] - terrain["y_m"][None]), axis=0)
    labels = height[np.ix_(ny, nx)].copy()
    outside_x = (terrain["x_m"] < tx[0]) | (terrain["x_m"] > tx[-1])
    outside_y = (terrain["y_m"] < ty[0]) | (terrain["y_m"] > ty[-1])
    outside = outside_y[:, None] | outside_x[None]
    labels[outside] = np.nan
    hazard = truth.get("terrain_hazard")
    if hazard is None and config is not None:
        hazard = dense_truth_hazards(height, tx, ty, config)
    if hazard is not None:
        hazard = np.asarray(hazard)
        if hazard.dtype != np.bool_ or hazard.shape != height.shape:
            raise ValueError("invalid independent terrain hazard grid")
        hazard = hazard[np.ix_(ny, nx)]
    return labels, hazard


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(finite_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def aggregate(records: list[dict]) -> list[dict]:
    groups = {}
    for record in records:
        key = (record["split"], record["retention"], record["stress"], record["arm"])
        groups.setdefault(key, []).append(record)
    output = []
    count_fields = ("gt_valid_pixels", "common_valid_pixels", "native_valid_pixels", "image_pixels",
                    "near_gt_pixels", "near_detected_pixels", "near_missed_pixels", "near_unknown_pixels",
                    "false_free_space_pixels", "false_obstacle_pixels", "predicted_near_pixels")
    for (split, retention, stress, arm), group in sorted(groups.items()):
        metrics = {field: sum(record["metrics"][field] for record in group) for field in count_fields}
        n = metrics["near_gt_pixels"]
        labelled = all(record["metrics"]["obstacle_metric_scope"] == "independent_obstacle_mask" for record in group)
        metrics.update({"obstacle_metric_scope": "independent_obstacle_mask" if labelled else "near_surface_proxy",
                        "obstacle_recall": metrics["near_detected_pixels"] / n if labelled and n else None,
                        "false_free_space_rate": metrics["false_free_space_pixels"] / n if labelled and n else None,
                        "missed_obstacle_rate": metrics["near_missed_pixels"] / n if labelled and n else None,
                        "near_surface_proxy_recall": metrics["near_detected_pixels"] / n if not labelled and n else None,
                        "coverage_on_gt": metrics["common_valid_pixels"] / metrics["gt_valid_pixels"]
                            if metrics["gt_valid_pixels"] else None,
                        "native_coverage": metrics["native_valid_pixels"] / metrics["image_pixels"],
                        "obstacle_precision": metrics["near_detected_pixels"] / metrics["predicted_near_pixels"]
                            if labelled and metrics["predicted_near_pixels"] else None})
        # Weighted errors derive from per-frame common-mask counts. Frames are
        # not treated as statistically independent experiment replicates.
        common = metrics["common_valid_pixels"]
        metrics["rmse_m"] = math.sqrt(sum((record["metrics"]["rmse_m"] or 0.) ** 2 *
            record["metrics"]["common_valid_pixels"] for record in group) / common) if common else None
        for field in ("mae_m", "abs_rel"):
            metrics[field] = sum((record["metrics"][field] or 0.) * record["metrics"]["common_valid_pixels"]
                                 for record in group) / common if common else None
        output.append({"split": split, "retention": retention, "stress": stress, "arm": arm,
                       "frames": len(group), "sequences": len({r["sequence_id"] for r in group}),
                       "metrics": metrics})
    return output


def run(protocol_path: Path, dataset: Path, predictions_path: Path, output: Path,
        stereo_method: str = "sgbm") -> dict:
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(protocol_path.read_text())
    config = protocol.get("sensor_geometry", {})
    if config.get("threshold_config_split", "development") != "development":
        raise ValueError("thresholds must be fixed from development before test replay")
    fusion_config = FusionConfig(**config.get("fusion", {}))
    fusion_config.validate()
    terrain_config = TerrainConfig(**config.get("terrain", {}))
    terrain_config.axes()
    retentions = config.get("retentions", [1., .5, .2])
    if retentions != [1., .5, .2]:
        raise ValueError("pilot retention matrix is fixed to 100/50/20 percent")
    stresses = config.get("extrinsic_stresses", [
        {"name": "nominal", "rotation_y_deg": 0., "translation_x_m": 0.},
        {"name": "rot1deg_trans1cm", "rotation_y_deg": 1., "translation_x_m": .01},
        {"name": "rot3deg_trans3cm", "rotation_y_deg": 3., "translation_x_m": .03}])
    if not stresses or len({s["name"] for s in stresses}) != len(stresses):
        raise ValueError("missing or duplicate extrinsic conditions")
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    declared_hashes = manifest.get("file_sha256")
    if not isinstance(declared_hashes, dict):
        raise ValueError("dataset file_sha256 manifest is required")
    def verify_dataset_file(value: str, namespace: str) -> Path:
        path = namespaced_path(dataset, value, namespace)
        expected = declared_hashes.get(value)
        actual = sha256(path)
        if not isinstance(expected, str) or len(expected) != 64 or expected != actual:
            raise ValueError(f"dataset checksum missing or mismatching: {value}")
        return path
    if manifest.get("data_kind") != "mujoco_rendered_rgb_timed_ray_lidar" and manifest.get("origin") != "physical":
        raise ValueError("rendered RGB/raw lidar replay or physical capture required")
    if manifest.get("lidar_dimension") != 3:
        raise ValueError("3D lidar capture is required for the terrain pilot")
    calibration = manifest["calibration"]
    if isinstance(calibration, str):
        calibration = json.loads(namespaced_path(dataset, calibration, "inference").read_text())
    camera = Pinhole.from_calibration(calibration)
    t_c_l = rigid_transform(np.asarray(calibration["T_C_L"], float))
    t_m_l = rigid_transform(np.asarray(calibration["T_M_L"], float)) if "T_M_L" in calibration else None
    if calibration.get("map_frame", "M_local_z_up") not in ("M_local", "M_local_z_up"):
        raise ValueError("terrain transform must declare the local z-up frame")
    predictions = json.loads(predictions_path.read_text())
    if predictions.get("schema") != "bhl-stereo-predictions-v1":
        raise ValueError("invalid stereo prediction manifest")
    prediction_map = {}
    for prediction in predictions["predictions"]:
        key = (prediction["sequence_id"], str(prediction["frame_id"]), prediction["method"])
        if key in prediction_map:
            raise ValueError("duplicate stereo prediction")
        prediction_map[key] = prediction
    records, terrain_records, common_records = [], [], []
    provenance = {str(manifest_path): sha256(manifest_path), str(predictions_path): sha256(predictions_path),
                  str(protocol_path): sha256(protocol_path)}
    input_seconds, fusion_seconds, terrain_seconds = [], [], []
    terrain_failures = []
    frame_count = 0
    for sequence in manifest["sequences"]:
        split = "development" if sequence["split"] == "dev" else sequence["split"]
        if split not in ("development", "validation", "test"):
            raise ValueError("whole-sequence split must be declared")
        if sequence.get("origin", manifest.get("origin")) not in ("simulation", "physical"):
            raise ValueError("sequence origin is missing")
        for frame in sequence["frames"]:
            start = time.perf_counter()
            for rgb in ("left", "right"):
                rgb_path = verify_dataset_file(frame[rgb], "inference")
                provenance[str(rgb_path)] = sha256(rgb_path)
            lidar_path = verify_dataset_file(frame["lidar"], "inference")
            imu_path = verify_dataset_file(frame["imu"], "inference")
            provenance[str(imu_path)] = sha256(imu_path)
            with np.load(lidar_path, allow_pickle=False) as packet:
                points = np.asarray(packet["points_xyz_m"], float)
                point_times = np.asarray(packet["point_time_s"], float)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) > 1_000_000:
                raise ValueError("invalid raw 3D point cloud")
            if point_times.shape != (len(points),) or not np.isfinite(point_times).all():
                raise ValueError("per-point scan times are required")
            provenance[str(lidar_path)] = sha256(lidar_path)
            key = (sequence["id"], str(frame["frame_id"]), stereo_method)
            if key not in prediction_map:
                raise ValueError("missing matched stereo prediction")
            prediction = prediction_map[key]
            depth_path = prediction_path(predictions_path.parent, prediction["depth"], dataset)
            confidence_path = prediction_path(predictions_path.parent, prediction["confidence"], dataset)
            valid_path = prediction_path(predictions_path.parent, prediction["valid"], dataset)
            depth, confidence, valid = map(load_array, (depth_path, confidence_path, valid_path))
            if depth.shape != (camera.height, camera.width) or valid.shape != depth.shape or confidence.shape != depth.shape:
                raise ValueError("stereo prediction/calibration dimensions disagree")
            if valid.dtype != np.bool_:
                raise ValueError("stereo validity must be boolean")
            depth = np.where(valid, depth, np.nan)
            for path in (depth_path, confidence_path, valid_path):
                provenance[str(path)] = sha256(path)
            # The only inference inputs are raw scan, stereo predictions and
            # declared calibration. We have not opened the evaluator file.
            input_seconds.append(time.perf_counter() - start)
            inference_results = []
            terrain_results = []
            stable_seed = int.from_bytes(hashlib.sha256(
                f"{config.get('dropout_seed', 220000)}:{sequence['id']}:{frame['frame_id']}".encode()).digest()[:8], "big")
            for retention in retentions:
                indices = retained_indices(len(points), retention, stable_seed)
                retained = points[indices]
                mask_hash = hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()
                for stress in stresses:
                    tic = time.perf_counter()
                    transform = stressed_extrinsic(t_c_l, stress["rotation_y_deg"], stress["translation_x_m"])
                    projected = project_lidar(retained, transform, camera)
                    fused = fuse_depth(depth, projected["depth_z_m"], confidence, fusion_config)
                    elapsed = time.perf_counter() - tic
                    fusion_seconds.append(elapsed)
                    inference_results.append((retention, stress["name"], mask_hash, projected, fused, elapsed))
                if t_m_l is not None:
                    tic = time.perf_counter()
                    terrain = elevation_map(transform_points(retained, t_m_l), terrain_config, 3)
                    terrain_seconds.append(time.perf_counter() - tic)
                    terrain_results.append((retention, mask_hash, terrain))
            # Evaluator-only truth, acquired separately from the input scan.
            truth_path = verify_dataset_file(frame["truth"], "evaluator")
            if truth_path.samefile(lidar_path) or any(truth_path.samefile(p) for p in (depth_path, confidence_path, valid_path)):
                raise ValueError("independent truth aliases an inference input")
            provenance[str(truth_path)] = sha256(truth_path)
            with np.load(truth_path, allow_pickle=False) as truth_packet:
                truth = {name: truth_packet[name] for name in truth_packet.files}
            source = str(np.asarray(truth.get("depth_truth_source", "")).item())
            if source not in ("independent_camera_geometry", "independent_ground_truth_sensor", "known_scene_geometry"):
                raise ValueError("depth truth must be independent of input lidar")
            truth_depth = np.asarray(truth["depth_z_m"], float)
            obstacle_mask = truth.get("obstacle_mask")
            if obstacle_mask is not None:
                if str(np.asarray(truth.get("obstacle_truth_source", "")).item()) != "independent_scene_geom_segmentation":
                    raise ValueError("obstacle mask must come from independent scene geometry")
            for retention, stress, mask_hash, projected, fused, elapsed in inference_results:
                base = {"sequence_id": sequence["id"], "frame_id": frame["frame_id"], "split": split,
                        "origin": sequence.get("origin", manifest.get("origin")), "retention": retention,
                        "stress": stress, "dropout_indices_sha256": mask_hash,
                        "stereo_method": stereo_method, "scan_motion_compensation": "none",
                        "projected_points": projected["projected_points"],
                        "fusion_diagnostics": fused["diagnostics"], "geometry_runtime_ms": elapsed * 1000}
                common_mask = np.isfinite(truth_depth)
                for arm_depth in fused["arms"].values():
                    common_mask &= np.isfinite(arm_depth)
                for arm, arm_depth in fused["arms"].items():
                    records.append({**base, "arm": arm,
                        "metrics": depth_metrics(arm_depth, truth_depth, fusion_config.near_threshold_m, obstacle_mask),
                        "risk_coverage": risk_coverage(arm_depth, fused["consistency_scores"][arm], truth_depth,
                                                      near_threshold_m=fusion_config.near_threshold_m,
                                                      obstacle_mask=obstacle_mask)})
                    common_records.append({**base, "arm": arm,
                        "metrics": depth_metrics(np.where(common_mask, arm_depth, np.nan),
                                                 np.where(common_mask, truth_depth, np.nan),
                                                 fusion_config.near_threshold_m, obstacle_mask)})
            for retention, mask_hash, terrain in terrain_results:
                try:
                    labels, hazards = interpolation_labels(truth, terrain, terrain_config)
                    metrics = terrain_metrics(terrain, labels, hazards)
                    terrain_records.append({"sequence_id": sequence["id"], "frame_id": frame["frame_id"],
                        "split": split, "retention": retention, "dropout_indices_sha256": mask_hash,
                        "metrics": metrics, "map_frame": "M_local_z_up", "scan_motion_compensation": "none",
                        "hazard_truth_source": "declared_independent_boolean_grid" if "terrain_hazard" in truth
                            else "independent_dense_vertical_ray_queries"})
                except (KeyError, ValueError) as error:
                    terrain_failures.append({"sequence_id": sequence["id"], "frame_id": frame["frame_id"],
                                             "reason": str(error)})
            if not terrain_results:
                terrain_failures.append({"sequence_id": sequence["id"], "frame_id": frame["frame_id"],
                                         "reason": "missing calibrated T_M_L; no oracle transform substituted"})
            frame_count += 1
    if not frame_count:
        raise ValueError("no replay frames")
    splits = {"development" if sequence["split"] == "dev" else sequence["split"]
              for sequence in manifest["sequences"]}
    if not {"development", "validation", "test"}.issubset(splits):
        raise ValueError("declared development/validation/test sequence cohorts are required")
    # Same scene cannot enter several splits. Temporal near-duplicates are not
    # counted as independent scenes simply by assigning new sequence IDs.
    scene_splits = {}
    for sequence in manifest["sequences"]:
        scene_splits.setdefault(sequence["scene_id"], set()).add(sequence["split"])
    if any(len(value) > 1 for value in scene_splits.values()):
        raise ValueError("scene leakage across evaluation splits")
    def timing(samples):
        return {"samples": len(samples), "p50_ms": float(np.quantile(samples, .5) * 1000) if samples else None,
                "p95_ms": float(np.quantile(samples, .95) * 1000) if samples else None,
                "p99_ms": float(np.quantile(samples, .99) * 1000) if samples else None}
    summary = {"schema_version": 1, "status": "PASS", "scientific_status": "PILOT_MEASUREMENT",
        "dataset_origin": manifest.get("origin"), "frames": frame_count,
        "sequences": len(manifest["sequences"]), "stereo_method": stereo_method,
        "fusion": {"status": "PASS", "summary": aggregate(records),
                   "all_arm_common_mask_summary": aggregate(common_records)},
        "terrain": {"status": "INCOMPLETE" if terrain_failures else "PASS",
                    "records": terrain_records, "failures": terrain_failures,
                    "hazard_evaluation_ready": bool(terrain_records) and all(
                        "hazard_gt_cells" in record["metrics"] for record in terrain_records),
                    "traversal_outcomes": None},
        "fixed_configuration": {"fusion": asdict(fusion_config), "terrain": asdict(terrain_config),
            "retentions": retentions, "extrinsic_stresses": stresses,
            "threshold_config_split": "development", "thresholds_tuned_on_this_dataset": False},
        "runtime": {"geometry_projection_and_all_fusion_arms": timing(fusion_seconds),
                    "local_elevation_map": timing(terrain_seconds), "input_load": timing(input_seconds),
                    "scope": "single pilot pass, no warmup; not end-to-end stereo or navigation deadline"},
        "limitations": ["No measured physical sensor accuracy or traversal outcome",
            "No scan-motion compensation; per-point timestamps are retained",
            "Confidence is a consistency score, not a calibrated probability",
            "Unknown pixels and cells are retained; sparse lidar does not certify dense free space",
            "Geometric fusion is informed by consistency research, not libSGM_lidar reproduction"],
        "inputs_sha256": provenance,
        "references": ["https://arxiv.org/abs/2504.05148", "https://github.com/yshry/libSGM_lidar"]}
    write_json(output / "fusion_frames.json", {"schema_version": 1, "records": records})
    write_json(output / "geometry_summary.json", summary)
    write_json(output / "campaign_result.json", {"status": "PASS", "scientific_status": "PILOT_MEASUREMENT",
        "route": "sensor_fusion_and_3d_terrain", "frames": frame_count,
        "fusion_status": "PASS", "terrain_status": summary["terrain"]["status"],
        "summary": "geometry_summary.json"})
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--stereo-predictions", type=Path)
    parser.add_argument("--stereo-method", default="sgbm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell")
    args = parser.parse_args(argv)
    try:
        run(args.protocol, args.dataset, args.stereo_predictions or args.dataset / "predictions.json",
            args.output, args.stereo_method)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "campaign_result.json", {"status": "INCOMPLETE",
            "route": "sensor_fusion_and_3d_terrain", "error": f"{type(error).__name__}: {error}"})
        print(f"INCOMPLETE: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
