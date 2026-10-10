#!/usr/bin/env python3
"""Relink read-only ORB telemetry wrapper; preserve original native SLAM library.

Run in the same Ubuntu 22.04 SIF as the verified original runtime. Exact header
package versions and checksums come from that runtime's build receipt. Neither
tracking source, feature settings nor the compiled SLAM library are changed.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
from bhl_robust.research.native_orb import UPSTREAM_COMMIT, sha256


def safe_extract(archive, destination):
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
                raise ValueError("runtime/source archives require safe regular members")
        stream.extractall(destination)


def execute(argv, **kwargs):
    print("RUN", json.dumps([str(x) for x in argv]), flush=True)
    return subprocess.run([str(x) for x in argv], check=True, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--original-runtime", type=Path, required=True)
    parser.add_argument("--original-runtime-sha256", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-cache", type=Path, help="Reuse only SHA-verified original .deb packages from an earlier attempt")
    args = parser.parse_args(argv)
    if sha256(args.original_runtime) != args.original_runtime_sha256:
        raise ValueError("original native runtime archive differs")
    args.work.mkdir(parents=True, exist_ok=False)
    args.output.mkdir(parents=True, exist_ok=False)
    safe_extract(args.original_runtime, args.work)
    runtime = args.work/"runtime"
    original = json.loads((runtime/"runtime.json").read_text())
    if original.get("status") != "PASS" or original.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("successful pinned original ORB runtime required")
    for name, checksum in original["files_sha256"].items():
        member = runtime/name
        if not member.resolve().is_relative_to(runtime.resolve()) or sha256(member) != checksum:
            raise ValueError("original runtime member changed: "+name)
    source = args.work/"source"
    source.mkdir()
    safe_extract(runtime/"upstream-source.tar.gz", source)
    # Source tar was captured before CMake generated g2o/config.h. Recreate
    # its exact original bytes and verify against retained compiled-work proof.
    g2o_template = source/"Thirdparty/g2o/config.h.in"
    g2o_config = source/"Thirdparty/g2o/config.h"
    g2o_config.write_text(g2o_template.read_text().replace(
        "#cmakedefine G2O_OPENMP 1", "/* #undef G2O_OPENMP */").replace(
        "#cmakedefine G2O_SHARED_LIBS 1", "/* #undef G2O_SHARED_LIBS */"))
    compiled = json.loads((runtime/"compiled-work-provenance.json").read_text())
    generated_proof = compiled["files"]["ORB_SLAM3/Thirdparty/g2o/config.h"]
    if sha256(g2o_config) != generated_proof["sha256"] or g2o_config.stat().st_size != generated_proof["bytes"]:
        raise ValueError("reconstructed generated g2o header differs from original compiled runtime")
    header = source/"include/System.h"
    initial_header = sha256(header)
    text = header.read_text()
    if text.count("    std::vector<MapPoint*> GetTrackedMapPoints();") != 1:
        raise ValueError("pinned System header anchor differs")
    text = text.replace("    std::vector<MapPoint*> GetTrackedMapPoints();",
        "    const Tracking* BhlReadOnlyTracker() const { return mpTracker; } // wrapper-only const inspection\n"
        "    std::vector<MapPoint*> GetTrackedMapPoints();")
    header.write_text(text)
    # Restore only development headers, pinned to exactly the linked runtime.
    apt = args.work/"apt"
    (apt/"lists/partial").mkdir(parents=True)
    (apt/"cache/archives/partial").mkdir(parents=True)
    options = ["-o", f"Dir::State::Lists={apt}/lists", "-o", f"Dir::Cache={apt}/cache",
               "-o", "Debug::NoLocking=true", "-o", "APT::Sandbox::User=root"]
    execute(["apt-get", *options, "update"])
    wanted = {"libeigen3-dev", "libboost1.74-dev", "libboost-serialization1.74-dev",
              "libopencv-core-dev", "libopencv-imgproc-dev", "libopencv-imgcodecs-dev",
              "libopencv-features2d-dev", "libopencv-flann-dev", "libopencv-calib3d-dev"}
    # opencv.hpp includes every configured OpenCV module, including DNN/video.
    for row in original["ubuntu_packages"]:
        fields = dict(line.split(": ", 1) for line in row["dpkg_fields"].strip().splitlines())
        if fields["Package"].startswith("libopencv-") and fields["Package"].endswith("-dev"):
            wanted.add(fields["Package"])
    packages, receipts = args.work/"packages", []
    packages.mkdir()
    sysroot = args.work/"sysroot"
    sysroot.mkdir()
    for row in original["ubuntu_packages"]:
        fields = dict(line.split(": ", 1) for line in row["dpkg_fields"].strip().splitlines())
        if fields["Package"] not in wanted:
            continue
        package = packages/row["file"]
        cached = args.package_cache/row["file"] if args.package_cache else None
        if cached is not None and cached.is_file():
            if sha256(cached) != row["sha256"]:
                raise ValueError("cached header package checksum differs")
            shutil.copy2(cached, package)
        else:
            execute(["apt-get", *options, "download", fields["Package"]+"="+fields["Version"]], cwd=packages)
        if sha256(package) != row["sha256"]:
            raise ValueError("original header package checksum differs: "+row["file"])
        execute(["dpkg-deb", "-x", package, sysroot])
        receipts.append(row)
        wanted.remove(fields["Package"])
    if wanted:
        raise ValueError("missing original header provenance: "+repr(wanted))
    compiler = ["g++", "-std=c++11", "-O2", "-DNDEBUG", "-DBHL_ORB_DIAGNOSTICS", "-DCOMPILEDWITHC11"]
    for inc in (source, source/"include", source/"include/CameraModels", source/"Thirdparty/Sophus",
                sysroot/"usr/include", sysroot/"usr/include/eigen3", sysroot/"usr/include/opencv4"):
        compiler += ["-I", str(inc)]
    wrapper = Path(__file__).with_name("orb_runner.cc")
    binary = args.work/"orb_native"
    compiler += [str(wrapper), "-o", str(binary), "-L", str(runtime/"lib"),
                 "-Wl,-rpath,$ORIGIN/../lib", "-Wl,-rpath-link,"+str(runtime/"lib"),
                 "-lORB_SLAM3", "-l:libopencv_core.so.4.5d", "-l:libopencv_imgproc.so.4.5d",
                 "-l:libopencv_imgcodecs.so.4.5d", "-lpthread"]
    env = dict(os.environ, LD_LIBRARY_PATH=str(runtime/"lib"))
    execute(compiler, env=env)
    execute([binary, "--version"], env=env)
    output_runtime = args.output/"runtime"
    shutil.copytree(runtime, output_runtime)
    shutil.copy2(binary, output_runtime/"bin/orb_native")
    shutil.copy2(wrapper, output_runtime/"orb_runner.cc")
    provenance = {"schema": "bhl-read-only-orb-diagnostics-build-v1", "status": "PASS",
                  "original_runtime_archive_sha256": args.original_runtime_sha256,
                  "upstream_commit": UPSTREAM_COMMIT, "original_system_header_sha256": initial_header,
                  "wrapper_system_header_sha256": sha256(header), "compile_command": compiler,
                  "header_packages": receipts, "wrapper_sha256": sha256(wrapper),
                  "restored_generated_g2o_config": generated_proof,
                  "native_library_unchanged_sha256": sha256(runtime/"lib/libORB_SLAM3.so"),
                  "native_algorithm_sources_modified": False, "feature_threshold_modified": False,
                  "telemetry_scope": "const current Frame fields after synchronous TrackStereo; linked native algorithms unchanged"}
    (output_runtime/"diagnostics-build.json").write_text(json.dumps(provenance, indent=2)+"\n")
    original["diagnostic_wrapper"] = provenance
    original["files_sha256"] = {str(p.relative_to(output_runtime)): sha256(p)
        for p in sorted(output_runtime.rglob("*")) if p.is_file() and p.name != "runtime.json"}
    (output_runtime/"runtime.json").write_text(json.dumps(original, indent=2)+"\n")
    archive = args.output/"runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(output_runtime, arcname="runtime")
    receipt = {"status": "PASS", "runtime_archive": str(archive), "runtime_archive_sha256": sha256(archive),
               "binary_sha256": sha256(output_runtime/"bin/orb_native"), "provenance": provenance}
    (args.output/"build-result.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "provenance"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
