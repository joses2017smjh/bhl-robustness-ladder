"""Run a frozen sensor pilot; completion is not a hardware or navigation claim.

Uses the existing bounded, read-only H3/H4 container launcher. A pinned GPU
ONNX Runtime wheel is unpacked into private scratch, never the shared venv.
The smoke and pilot run the identical model, capture and benchmark programs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import time
import zipfile


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def unpack_runtime(wheel, expected_sha, destination):
    """Unpack only a verified wheel, bounded and with no links/path escapes."""
    wheel, destination = Path(wheel), Path(destination)
    if digest(wheel) != expected_sha:
        raise ValueError("GPU runtime wheel hash mismatch")
    with zipfile.ZipFile(wheel) as archive:
        entries, seen = archive.infolist(), set()
        if sum(row.file_size for row in entries) > 512 * 1024 ** 2:
            raise ValueError("runtime wheel exceeds 512 MiB expanded cap")
        for row in entries:
            p = PurePosixPath(row.filename)
            if (p.is_absolute() or ".." in p.parts or "\\" in row.filename
                    or row.filename in seen or stat.S_ISLNK(row.external_attr >> 16)):
                raise ValueError("unsafe or duplicate runtime member")
            if p.parts[0] not in {"onnxruntime", "onnxruntime_gpu-1.22.0.dist-info"}:
                raise ValueError("unexpected runtime wheel package")
            seen.add(row.filename)
        destination.mkdir(parents=True, exist_ok=False)
        archive.extractall(destination)
    return {"sha256": expected_sha, "bytes": wheel.stat().st_size,
            "expanded_bytes": sum(row.file_size for row in entries),
            "private_path": str(destination), "package": "onnxruntime-gpu==1.22.0"}


def stage_commands(source, protocol_path, dataset, output, model, phase, research):
    if phase not in {"smoke", "pilot"}:
        raise ValueError("phase must be smoke or pilot")
    capture = research["capture"][phase]
    timing = research["timing"][phase]
    source, dataset, output, model = map(Path, (source, dataset, output, model))
    common = ["--protocol", str(protocol_path)]
    return [
        ("capture", [str(source / "scripts/bench/sensor_capture.py"),
                     "--output", str(dataset), "--frames-per-scene", str(capture["frames_per_scene"]), *common]),
        ("stereo", [str(source / "scripts/bench/stereo_research.py"),
                    "--dataset", str(dataset), "--output", str(output / "stereo"),
                    "--model", str(model), "--provider", "CUDAExecutionProvider",
                    "--warmup", str(timing["warmup"]), "--timed", str(timing["timed"]), *common]),
        ("geometry", [str(source / "scripts/bench/sensor_geometry_research.py"),
                      "--dataset", str(dataset), "--stereo-predictions", str(output / "stereo/predictions.json"),
                      "--stereo-method", "sgbm", "--output", str(output / "geometry"), *common]),
        ("geometry_ffs", [str(source / "scripts/bench/sensor_geometry_research.py"),
                          "--dataset", str(dataset), "--stereo-predictions", str(output / "stereo/predictions.json"),
                          "--stereo-method", "c_fast_foundationstereo", "--output", str(output / "geometry-ffs"), *common]),
        ("pose_readiness", [str(source / "scripts/bench/pose_research.py"),
                            "intake", "--dataset", str(dataset), "--output", str(output / "pose"), *common]),
    ]


def validate_stage_result(name, result, geometry_summary=None):
    if result.get("status") != "PASS":
        raise ValueError(f"{name} stage result is missing or not execution PASS")
    if name in {"geometry", "geometry_ffs"}:
        if result.get("terrain_status") != "PASS":
            raise ValueError("3D map stage incomplete despite fusion completion")
        terrain = (geometry_summary or {}).get("terrain", {})
        if terrain.get("status") != "PASS" or terrain.get("hazard_evaluation_ready") is not True:
            raise ValueError("independent terrain and hazard evaluation are required")
    if name == "pose_readiness":
        if (result.get("scientific_status") != "BLOCKED_RUNTIME"
                or result.get("estimated_pose_outputs") != 0 or result.get("closed_loop_episodes") != 0):
            raise ValueError("pose pilot must preserve readiness-only status and zero unrun outcomes")


def run(protocol_path, phase, output, work):
    protocol_path, output, work = map(Path, (protocol_path, output, work))
    protocol = json.loads(protocol_path.read_text())
    research = protocol["sensor_research"]
    source = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
    inputs = protocol_path.parent / "inputs"
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    result = {"schema": "bhl-sensor-campaign-v1", "status": "INCOMPLETE", "phase": phase,
              "scope": "Simulated kinematic sensor replay pilot; no physical accuracy, SLAM drift, goals, collisions, falls, or traversal result.",
              "protocol_sha256": digest(protocol_path), "started_utc": datetime.now(timezone.utc).isoformat(),
              "stages": [], "routes": {}, "problems": []}
    started = time.monotonic()
    try:
        model_spec, runtime_spec = research["model"], research["gpu_runtime"]
        model, wheel = inputs / model_spec["input_name"], inputs / runtime_spec["input_name"]
        if digest(model) != model_spec["sha256"]:
            raise ValueError("pretrained stereo model hash mismatch")
        result["runtime"] = unpack_runtime(wheel, runtime_spec["sha256"], work / "onnx-gpu-site")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(work / "onnx-gpu-site") + ":" + env.get("PYTHONPATH", str(source / "src"))
        env["MUJOCO_GL"] = "egl"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        sys.path.insert(0, str(source / "src"))
        from bhl_robust.research.replay_contract import audit_replay
        dataset = output / "replay"
        for name, argv in stage_commands(source, protocol_path, dataset, output, model, phase, research):
            before = time.monotonic()
            command = [sys.executable, *argv]
            with (output / f"{name}.log").open("xb") as log:
                proc = subprocess.run(command, env=env, cwd=work, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=research["stage_timeout_s"])
            stage = {"name": name, "command": command, "returncode": proc.returncode,
                     "elapsed_s": time.monotonic() - before}
            result["stages"].append(stage)
            if proc.returncode:
                raise RuntimeError(f"{name} stage failed (exit {proc.returncode}); see {name}.log")
            stage_result_path = (dataset if name == "capture" else output / {
                "stereo": "stereo", "geometry": "geometry", "geometry_ffs": "geometry-ffs", "pose_readiness": "pose"}[name]) / "campaign_result.json"
            stage_result = json.loads(stage_result_path.read_text())
            stage["reported_status"] = stage_result.get("status")
            stage["scientific_status"] = stage_result.get("scientific_status")
            geometry_summary = None
            if name in {"geometry", "geometry_ffs"}:
                geometry_summary = json.loads((stage_result_path.parent / "geometry_summary.json").read_text())
            validate_stage_result(name, stage_result, geometry_summary)
            if name == "capture":
                result["replay_audit"] = audit_replay(dataset)
                write_json(output / "replay-audit.json", result["replay_audit"])
            result["routes"][name] = {"execution": "COMPLETED", "scientific_scope": "pilot only"}
        result["routes"]["estimated_pose_navigation"] = {
            "execution": "READINESS_AND_EXPORT_COMPLETED", "scientific_status": "BLOCKED_RUNTIME",
            "requirements": ["compiled ORB-SLAM3 stereo", "ROS FAST-LIO2 runtime", "longer estimator-validation sequences",
                             "estimated-pose closed-loop integration and fresh confirmation cohort"],
            "closed_loop_episodes_completed": 0}
        result["routes"]["terrain_traversal"] = {"execution": "MAP_PILOT_COMPLETED",
                                                  "scientific_status": "WAITING_LOCOMOTION_VALIDATION",
                                                  "traversal_episodes_completed": 0}
        result["status"] = "PASS"
        result["acceptance_meaning"] = "Every implemented pilot stage completed on exact frozen inputs; superiority and the four full research objectives remain unqualified."
    except Exception as exc:
        result["problems"].append(f"{type(exc).__name__}: {exc}")
    finally:
        result["elapsed_s"] = time.monotonic() - started
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "campaign_result.json", result)
    return 0 if result["status"] == "PASS" else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--phase", choices=("smoke", "pilot"), required=True)
    p.add_argument("--output", type=Path, default=os.environ.get("H34_OUTPUT_DIR"))
    p.add_argument("--work", type=Path, default=os.environ.get("H34_WORK_DIR"))
    args = p.parse_args()
    if args.output is None or args.work is None:
        p.error("--output and --work or the frozen launcher environment are required")
    return run(args.protocol, args.phase, args.output, args.work)


if __name__ == "__main__":
    raise SystemExit(main())
