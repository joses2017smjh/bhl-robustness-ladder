#!/usr/bin/env python3
"""Run actual SSD predictions through the timestamped navigation input boundary.

COCO weights do not recognize custom maze markers/buttons. Supply a trained
SSDLite state dict and its class count for those categories. No synthetic
bounding boxes are substituted if a model fails to detect an object.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from bhl_robust.sensor_io import imu_features, ssd_features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True, help="Left-camera RGB image")
    p.add_argument("--weights", required=True, help="SSDLite state dict path, or coco (downloads official weights)")
    p.add_argument("--num-classes", type=int, default=91)
    p.add_argument("--capture-time", type=float, required=True, help="RGB capture timestamp in seconds")
    p.add_argument("--depth", type=Path, help="Registered left-camera optical-axis depth .npy in meters")
    p.add_argument("--depth-time", type=float)
    p.add_argument("--depth-source", choices=("stereo_estimate", "sim_ground_truth"))
    p.add_argument("--depth-metadata", type=Path,
                   help="stereo_depth.py metadata.json; verifies image and preserves capture timestamp")
    p.add_argument("--imu", type=Path, help="ROS-unit JSON: orientation_xyzw, gyro_rad_s, specific_force_m_s2, stamp_s")
    p.add_argument("--replay", action="store_true", help="Evaluate archived inputs at their capture time; not live freshness")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.depth_metadata and any(v is not None for v in (a.depth, a.depth_time, a.depth_source)):
        p.error("Use --depth-metadata or explicit depth arguments, not both")
    if a.depth and (a.depth_time is None or a.depth_source is None):
        p.error("--depth requires its capture --depth-time and --depth-source")
    if a.out.exists():
        raise FileExistsError(f"refusing to overwrite {a.out}")
    if a.depth_metadata:
        from bhl_robust.stereo_depth import load_depth_metadata
        depth, a.depth_time, a.depth_source = load_depth_metadata(a.depth_metadata, a.image, a.capture_time)
    else:
        depth = np.load(a.depth, allow_pickle=False) if a.depth else None
    import torch
    from PIL import Image
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large,
    )
    from torchvision.transforms.functional import pil_to_tensor
    if a.weights == "coco":
        model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    else:
        model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=a.num_classes)
        state = torch.load(a.weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state), strict=True)
    model.to(a.device).eval()
    rgb = Image.open(a.image).convert("RGB")
    tensor = pil_to_tensor(rgb).to(a.device).float() / 255
    if str(a.device).startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        raw = model([tensor])[0]
    if str(a.device).startswith("cuda"):
        torch.cuda.synchronize()
    latency = time.perf_counter() - start
    pred = {key: raw[key].detach().cpu().numpy() for key in ("boxes", "labels", "scores")}
    now = a.capture_time if a.replay else time.time()
    features = ssd_features(pred, image_width=rgb.width, image_height=rgb.height,
        stamp_s=a.capture_time, now_s=now, depth_m=depth,
        depth_stamp_s=a.depth_time)
    result = {"schema": "bhl-sensor-input-v1", "detector": "SSDLite320_MobileNetV3",
        "weights": a.weights, "image": str(a.image), "capture_time": a.capture_time,
        "consumption_time": now, "replay": a.replay, "inference_s": latency,
        "depth_source": a.depth_source, "depth_capture_time": a.depth_time,
        "depth_metadata": str(a.depth_metadata) if a.depth_metadata else None,
        "detections": features.tolist(),
        "columns": ["class_id", "score", "cx", "cy", "width", "height", "depth_z_m", "depth_valid", "detection_valid"]}
    if a.imu:
        imu = json.loads(a.imu.read_text())
        result["imu"] = imu_features(imu["orientation_xyzw"], imu["gyro_rad_s"],
            imu["specific_force_m_s2"], stamp_s=imu["stamp_s"], now_s=now,
            orientation_available=imu.get("orientation_available", True)).tolist()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"valid_detections": int(features[:, -1].sum()), "inference_s": latency, "out": str(a.out)}))


if __name__ == "__main__":
    main()
