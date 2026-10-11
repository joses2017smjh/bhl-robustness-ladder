#!/usr/bin/env python3
"""Plot verified sensor-pilot artifacts; never infer or invent missing results.

Usage: sensor_demo.py --pilot-output DIR --out NEW_DIR
The source pilot is read only. Both methods, their exact prediction digests,
independent truth, and measured CUDA execution must be available before any
figure is written. Figures explicitly retain the simulated kinematic scope.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bhl_robust.research.replay_contract import audit_replay
from bhl_robust.research.sensor_geometry import (
    Pinhole, TerrainConfig, elevation_map, project_lidar, transform_points,
)
from bhl_robust.research.stereo_benchmark import (
    FFS_MODEL_SHA256, checked_path, load_replay, load_rgb,
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def provenance_hash(provenance: dict, relative: str) -> str:
    """Find a unique digest after relocating an archived pilot output tree."""
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or ".." in parts:
        raise ValueError("unsafe relative artifact path")
    suffix = "/" + Path(relative).as_posix()
    matches = {value for name, value in provenance.items()
               if name == Path(relative).as_posix() or name.endswith(suffix)}
    if len(matches) != 1:
        raise ValueError(f"missing or ambiguous recorded digest: {relative}")
    expected = matches.pop()
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError(f"invalid recorded digest: {relative}")
    return expected


def verified_artifact(pilot: Path, relative: str, provenance: dict) -> Path:
    path = checked_path(pilot, relative)
    if path.stat().st_size > 256 * 1024**2:
        raise ValueError("demo artifact exceeds size cap")
    if sha256(path) != provenance_hash(provenance, relative):
        raise ValueError(f"artifact digest mismatch: {relative}")
    return path


def verified_p95(timing: dict) -> float:
    samples = np.asarray(timing["samples_ms"], dtype=float)
    if (samples.ndim != 1 or len(samples) != timing["timed_iterations"]
            or not len(samples) or not np.isfinite(samples).all() or np.any(samples < 0)):
        raise ValueError("latency sample evidence is missing or invalid")
    p95 = float(np.quantile(samples, .95))
    if not math.isclose(p95, float(timing["p95_ms"]), rel_tol=1.e-10, abs_tol=1.e-10):
        raise ValueError("recorded p95 disagrees with measured samples")
    if timing.get("scope") != "warm_cache_PNG_decode_preprocess_inference_depth_validity":
        raise ValueError("unrecognized timing scope")
    return p95


def load_inputs(pilot: Path) -> dict:
    """Validate real evidence first, without creating the requested output."""
    pilot = pilot.resolve()
    campaign = json.loads((pilot / "campaign_result.json").read_text())
    if campaign.get("status") != "PASS":
        raise ValueError("pilot did not complete")
    dataset = pilot / "replay"
    audit = audit_replay(dataset)
    manifest, calibration = load_replay(dataset)
    if manifest.get("origin") != "simulation":
        raise ValueError("this demo labels the current simulated pilot only")
    geometries = []
    for directory, method in (("geometry", "sgbm"), ("geometry-ffs", "c_fast_foundationstereo")):
        summary = json.loads((pilot / directory / "geometry_summary.json").read_text())
        if summary.get("status") != "PASS" or summary.get("stereo_method") != method:
            raise ValueError(f"missing completed {method} geometry output")
        geometries.append(summary)
    provenance = {}
    for summary in geometries:
        for name, value in summary["inputs_sha256"].items():
            if name in provenance and provenance[name] != value:
                raise ValueError("geometry reports disagree about source digests")
            provenance[name] = value
    verified_artifact(pilot, "replay/manifest.json", provenance)
    prediction_path = verified_artifact(pilot, "stereo/predictions.json", provenance)
    predictions = json.loads(prediction_path.read_text())
    stereo = json.loads((pilot / "stereo/campaign_result.json").read_text())
    if (predictions.get("schema") != "bhl-stereo-predictions-v1" or stereo.get("status") != "PASS"
            or predictions.get("dataset_manifest_sha256") != audit["manifest_sha256"]
            or stereo.get("dataset_manifest_sha256") != audit["manifest_sha256"]):
        raise ValueError("stereo predictions do not match the verified replay")
    methods = {row["method"]: row for row in stereo["methods"]}
    if set(methods) != {"sgbm", "c_fast_foundationstereo"}:
        raise ValueError("both actual target methods are required")
    ffs = methods["c_fast_foundationstereo"]
    evidence = ffs.get("provider_execution_evidence") or {}
    if (ffs.get("model_sha256") != FFS_MODEL_SHA256
            or ffs.get("provider") != "CUDAExecutionProvider"
            or evidence.get("status") != "VERIFIED_CUDA_KERNEL_EXECUTION"
            or evidence.get("kernel_events_by_provider", {}).get("CUDAExecutionProvider", 0) < 1):
        raise ValueError("official C-FFS checkpoint and actual CUDA execution must be verified")
    p95 = {method: verified_p95(stereo["timing"][method]) for method in methods}
    sequence = next((row for row in manifest["sequences"] if row["split"] == "test"), None)
    if sequence is None or not sequence["frames"]:
        raise ValueError("missing held-out scene frame")
    frame = sequence["frames"][0]  # Fixed selection, never chosen by error.
    shape = (calibration.image_height, calibration.image_width)
    depths = {}
    for method in methods:
        matches = [row for row in predictions["predictions"]
                   if (row["sequence_id"], row["frame_id"], row["method"])
                   == (sequence["id"], frame["frame_id"], method)]
        if len(matches) != 1:
            raise ValueError("matched prediction is missing or duplicated")
        arrays = {key: np.load(verified_artifact(pilot, "stereo/" + matches[0][key], provenance),
                               allow_pickle=False) for key in ("depth", "valid", "confidence")}
        if (any(value.shape != shape for value in arrays.values())
                or arrays["valid"].dtype != np.bool_
                or not np.isfinite(arrays["depth"][arrays["valid"]]).all()
                or np.any(arrays["depth"][arrays["valid"]] <= 0)):
            raise ValueError("invalid recorded prediction dimensions or validity")
        depths[method] = np.where(arrays["valid"], arrays["depth"], np.nan)
    with np.load(checked_path(dataset, frame["truth"], namespace="evaluator"), allow_pickle=False) as packet:
        truth = {key: packet[key] for key in packet.files}
    if (str(truth.get("depth_truth_source", "")) != "independent_camera_geometry"
            or str(truth.get("terrain_truth_source", "")) != "independent_dense_vertical_ray_queries"):
        raise ValueError("independent scene truth provenance is required")
    with np.load(checked_path(dataset, frame["lidar"], namespace="inference"), allow_pickle=False) as packet:
        points = packet["points_xyz_m"].copy()
    raw_calibration = manifest["calibration"]
    projected = project_lidar(points, np.asarray(raw_calibration["T_C_L"]),
                              Pinhole.from_calibration(raw_calibration))
    terrain_config = TerrainConfig(**geometries[0]["fixed_configuration"]["terrain"])
    terrain = elevation_map(transform_points(points, np.asarray(raw_calibration["T_M_L"])), terrain_config, 3)
    spec = importlib.util.spec_from_file_location("bhl_sensor_demo_geometry", Path(__file__).with_name("sensor_geometry_research.py"))
    geometry_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry_module)
    terrain_truth, hazard_truth = geometry_module.interpolation_labels(truth, terrain, terrain_config)
    return {"manifest": manifest, "audit": audit, "stereo": stereo, "sequence": sequence,
            "frame": frame, "left": load_rgb(checked_path(dataset, frame["left"], namespace="inference")),
            "right": load_rgb(checked_path(dataset, frame["right"], namespace="inference")),
            "truth": truth["depth_z_m"], "depths": depths, "p95": p95,
            "lidar": projected["depth_z_m"], "terrain": terrain,
            "terrain_truth": terrain_truth, "hazard_truth": hazard_truth,
            "provider": ffs["provider"], "device": ffs["cuda_runtime"]["device_name"],
            "prediction_manifest_sha256": sha256(prediction_path)}


def render(pilot: Path, output: Path) -> dict:
    data = load_inputs(pilot)
    output = output.absolute()
    if output.exists():
        raise FileExistsError("demo output must be a new directory")
    if output.resolve().is_relative_to(pilot.resolve()):
        raise ValueError("demo output must be separate from the read-only pilot")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    depth_cmap = plt.get_cmap("viridis").copy()
    depth_cmap.set_bad("#b8b8b8")
    scope = "SIMULATION | Kinematic sensor rig | No humanoid gait or navigation outcome"
    identity = f"Held-out scene: {data['sequence']['scene_id']} | first frame {data['frame']['frame_id']}"
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    for ax, image, title in ((axes[0, 0], data["left"], "Original left RGB"),
                             (axes[0, 1], data["right"], "Original right RGB")):
        ax.imshow(image)
        ax.set_title(title)
    depth_panels = ((axes[0, 2], data["truth"], "Independent scene optical depth"),
                    (axes[1, 0], data["depths"]["sgbm"], "SGBM: recorded valid depth"),
                    (axes[1, 1], data["depths"]["c_fast_foundationstereo"], "Pretrained C-Fast-FoundationStereo"),
                    (axes[1, 2], data["lidar"], "Timed 3D lidar: sparse projected endpoints"))
    for ax, image, title in depth_panels:
        artist = ax.imshow(np.ma.masked_invalid(image), cmap=depth_cmap, vmin=.1, vmax=6., interpolation="nearest")
        ax.set_title(title)
    for ax in axes.ravel():
        ax.set_axis_off()
    figure.colorbar(artist, ax=[row[0] for row in depth_panels], shrink=.8,
                    label="Optical-axis Z (m), same 0.1–6 m scale; gray = unknown", extend="max")
    figure.suptitle("Rendered stereo and 3D lidar replay\n" + scope + "\n" + identity, fontsize=13)
    caption = (f"Measured warm-cache p95: SGBM {data['p95']['sgbm']:.1f} ms (CPU), "
               f"C-FFS {data['p95']['c_fast_foundationstereo']:.1f} ms ({data['device']}, verified CUDA).\n"
               "Timing covers PNG decode, preprocessing, inference, depth and validity; capture, rectification, fusion and controller excluded. "
               "Native validity masks differ; this frame is a demo, not a superiority result.")
    figure.supxlabel(caption, fontsize=9)
    terrain = data["terrain"]
    terrain_figure, terrain_axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    x, y = terrain["x_m"], terrain["y_m"]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    extent = [x[0] - dx/2, x[-1] + dx/2, y[0] - dy/2, y[-1] + dy/2]
    panels = ((terrain_axes[0, 0], data["terrain_truth"], "Independent dense height truth", 0., 1.5, "Height (m)"),
              (terrain_axes[0, 1], terrain["height_m"], "Lidar endpoint median elevation", 0., 1.5, "Height (m)"),
              (terrain_axes[0, 2], terrain["slope_deg"], "Local plane slope", 0., 45., "Slope (degrees)"),
              (terrain_axes[1, 0], terrain["roughness_m"], "Local plane residual roughness", 0., .15, "RMS residual (m)"))
    for ax, image, title, low, high, label in panels:
        artist = ax.imshow(np.ma.masked_invalid(image), origin="lower", extent=extent,
                           cmap=depth_cmap, vmin=low, vmax=high, interpolation="nearest", aspect="equal")
        ax.set_title(title)
        terrain_figure.colorbar(artist, ax=ax, shrink=.75, label=label, extend="max")
    hazard_cmap = ListedColormap(["#4c956c", "#c94c4c"])
    hazard_cmap.set_bad("#b8b8b8")
    for ax, image, title in ((terrain_axes[1, 1], np.where(np.isfinite(data["terrain_truth"]), data["hazard_truth"], np.nan),
                             "Independent geometric hazard labels"),
                            (terrain_axes[1, 2], np.where(terrain["known"], terrain["hazard"], np.nan),
                             "Lidar hazard: supported cells only")):
        artist = ax.imshow(np.ma.masked_invalid(image), origin="lower", extent=extent,
                           cmap=hazard_cmap, norm=BoundaryNorm([-.5, .5, 1.5], 2), interpolation="nearest", aspect="equal")
        ax.set_title(title)
        colorbar = terrain_figure.colorbar(artist, ax=ax, shrink=.75, ticks=[0, 1])
        colorbar.ax.set_yticklabels(["No geometric hazard", "Hazard"])
    for ax in terrain_axes.ravel():
        ax.set_xlabel("Forward x (m)")
        ax.set_ylabel("Left y (m)")
    terrain_figure.suptitle("3D lidar elevation, slope, roughness and hazards\n" + scope + "\n" + identity, fontsize=13)
    terrain_figure.supxlabel("Gray = unknown. Sparse scans remain uncompensated; no ground segmentation or traversal result.\n"
                            "Hazards use fixed geometric thresholds; a supported green cell does not establish safe locomotion.", fontsize=10)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for fig, basename in ((figure, "stereo-lidar-demo"), (terrain_figure, "terrain-demo")):
        for extension in ("png", "svg"):
            path = output / (basename + "." + extension)
            fig.savefig(path, dpi=160)
            artifacts[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        plt.close(fig)
    result = {"status": "PASS", "scope": scope, "selection": "first frame of first predeclared held-out scene",
              "sequence_id": data["sequence"]["id"], "frame_id": data["frame"]["frame_id"],
              "dataset_manifest_sha256": data["audit"]["manifest_sha256"],
              "prediction_manifest_sha256": data["prediction_manifest_sha256"],
              "verified_provider": data["provider"], "device": data["device"], "p95_ms": data["p95"],
              "navigation_outcomes": None, "traversal_outcomes": None, "artifacts": artifacts}
    (output / "demo_result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-output", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = render(args.pilot_output, args.out)
    except Exception as error:
        print(json.dumps({"status": "INCOMPLETE", "error": f"{type(error).__name__}: {error}"}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
