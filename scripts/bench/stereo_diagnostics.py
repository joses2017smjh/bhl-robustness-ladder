#!/usr/bin/env python3
"""Summarize genuine native ORB telemetry; never re-extract surrogate features."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from bhl_robust.research.native_orb import sha256, validate_response


def summarize(path):
    rows = [validate_response(json.loads(line), json.loads(line)["timestamp_s"])
            for line in path.read_text().splitlines() if line]
    if not rows or any("diagnostics" not in row for row in rows):
        raise ValueError("every original native frame must contain real diagnostics")
    stamps = [row["timestamp_s"] for row in rows]
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("strictly increasing original frame timestamps required")
    result = {"source": str(path), "source_sha256": sha256(path), "frames": len(rows),
              "native_tracked_frames": sum(row["tracked"] for row in rows),
              "initialization_reasons": dict(Counter(row["diagnostics"]["initialization_reason"] for row in rows)),
              "feature_threshold_exclusive": 500,
              "first_tracked_timestamp_s": next((row["timestamp_s"] for row in rows if row["tracked"]), None)}
    for name in ("features_total", "features_left", "features_right", "positive_stereo_depth_matches"):
        values = np.asarray([row["diagnostics"][name] for row in rows])
        result[name] = {"min": int(values.min()), "median": float(np.median(values)), "max": int(values.max()),
                        "p05": float(np.quantile(values, .05)), "p95": float(np.quantile(values, .95))}
    result["frames_above_native_initialization_feature_threshold"] = sum(row["diagnostics"]["features_total"] > 500 for row in rows)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--frames", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    results = [summarize(path) for path in args.frames]
    value = {"schema": "bhl-native-stereo-diagnostics-v1", "status": "MEASURED_DIAGNOSTICS",
             "streams": results, "scope": "Read-only native frame counts; no feature threshold or estimator changes; diagnostics alone are not an improvement experiment"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
