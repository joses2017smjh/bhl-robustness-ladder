#!/usr/bin/env python3
"""Bounded native build wrapper; runtime construction is not SLAM accuracy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["orb", "lio"], required=True)
    a = p.parse_args()
    source = Path(__file__).resolve().parents[2]
    work, output = Path(os.environ["H34_WORK_DIR"]), Path(os.environ["H34_OUTPUT_DIR"])
    runtime = work / "native-runtime"
    command = [sys.executable, str(source / "scripts/native" / (a.method + "_build.py")), "--output", str(runtime)]
    if a.method == "orb": command += ["--work", str(work / "build"), "--jobs", "4"]
    started = time.monotonic()
    with (output / "build.log").open("x") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    result = {"route": "native_runtime_build", "method": a.method,
              "returncode": process.returncode, "wall_seconds": time.monotonic()-started,
              "command": command, "scientific_status": "BUILD_ONLY_NO_ESTIMATED_TRAJECTORIES"}
    try:
        receipt = json.loads((runtime / "runtime.json").read_text())
        if process.returncode or receipt.get("status") != "PASS": raise ValueError("native build did not pass")
        target = output / "runtime"
        target.mkdir()
        if a.method == "orb":
            for f in runtime.iterdir():
                if f.is_dir(): shutil.copytree(f, target / f.name)
                elif f.is_file(): shutil.copy2(f, target / f.name)
        else:
            for name in ["fastlio_headless", "lib", "runtime.json", "fastlio_headless.cpp", "IMU_Processing.hpp", "shim"]:
                f = runtime / name
                if f.is_dir(): shutil.copytree(f, target / name)
                else: shutil.copy2(f, target / name)
            # Retain actual unmodified upstream sources/licenses for this transport adaptation.
            import tarfile
            with tarfile.open(target / "upstream-source.tar.gz", "w:gz") as archive:
                for f in sorted((runtime / "upstream").rglob("*")):
                    if f.is_file() and ".git" not in f.relative_to(runtime / "upstream").parts:
                        archive.add(f, arcname=str(f.relative_to(runtime / "upstream")))
        files = {str(f.relative_to(target)): {"sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "bytes": f.stat().st_size}
                 for f in target.rglob("*") if f.is_file()}
        result.update(status="PASS", runtime_schema=receipt["schema"], runtime_files=files)
    except Exception as error:
        result.update(status="INCOMPLETE", error=str(error))
        if (runtime / "build_failure.json").exists(): shutil.copy2(runtime / "build_failure.json", output / "build_failure.json")
    (output / "campaign_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "runtime_files"}), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
