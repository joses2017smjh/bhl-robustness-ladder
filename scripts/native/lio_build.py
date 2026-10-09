#!/usr/bin/env python3
"""Build the pinned native FAST-LIO2 estimator without ROS transport services.

Run compilation on an allocated CPU node. Downloads are isolated, checksum
verified, and never installed into the system or the shared Python environment.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

COMMIT = "7cc4175de6f8ba2edf34bab02a42195b141027e9"
IKD_COMMIT = "e2e3f4e9d3b95a9e66b1ba83dc98d4a05ed8a3c4"
URL = "https://github.com/hku-mars/FAST_LIO.git"


def sha256(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def run(args, **kwargs):
    print("EXEC", args, flush=True)
    return subprocess.run(args, check=True, **kwargs)


def deb_data(path):
    raw = Path(path).read_bytes()
    if raw[:8] != b"!<arch>\n":
        raise ValueError("dependency is not a Debian archive")
    position = 8
    while position + 60 <= len(raw):
        header = raw[position:position + 60]
        size = int(header[48:58])
        name = header[:16].decode().strip().rstrip("/")
        position += 60
        data = raw[position:position + size]
        position += size + size % 2
        if name.startswith("data.tar"):
            return data
    raise ValueError("Debian archive lacks data payload")


def extract_dependency(path, target):
    payload = deb_data(path)
    if payload[:4] == b"\x28\xb5\x2f\xfd":
        executable = shutil.which("zstd")
        if not executable: raise ValueError("pinned Debian payload requires the zstd decompressor")
        payload = subprocess.check_output([executable, "--decompress", "--stdout"], input=payload)
        if len(payload) > 512 * 1024 ** 2: raise ValueError("dependency payload exceeds private extraction cap")
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
        for member in archive.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe package path")
            if member.issym() and Path(member.linkname).is_absolute():
                # Development packages contain /usr/lib absolute soname links;
                # make those private-root relative instead of escaping sysroot.
                destination = target / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.is_symlink() or destination.exists(): destination.unlink()
                destination.symlink_to(os.path.relpath(target / member.linkname.lstrip("/"), destination.parent))
            else:
                archive.extract(member, target, filter="data")


def generated_core(upstream):
    """Copy exact upstream global state and estimator callbacks, without ROS IO."""
    source = (upstream / "src/laserMapping.cpp").read_text()
    license_text = source[:source.index("#include <omp.h>")]
    sections = [
        source[source.index("#define INIT_TIME"):source.index("nav_msgs::Path path;")],
        source[source.index("void pointBodyToWorld(PointType"):source.index("void standard_pcl_cbk(")],
        source[source.index("void map_incremental()"):source.index("PointCloudXYZI::Ptr pcl_wait_pub")],
        source[source.index("void h_share_model("):source.index("int main(int argc")],
    ]
    headers = """
#include <omp.h>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include "IMU_Processing.hpp"
#include <pcl/filters/voxel_grid.h>
#include <ikd-Tree/ikd_Tree.h>
"""
    return license_text + headers + sections[0] + "\nshared_ptr<ImuProcess> p_imu(new ImuProcess());\n" + "\n".join(sections[1:]), sections


def write_shims(directory):
    """Only immutable typed message/time containers and logging; no estimator."""
    time = """#pragma once
#include <cstdio>
#include <cassert>
namespace ros { struct Time { double seconds=0; double toSec() const {return seconds;} Time& fromSec(double v) {seconds=v;return *this;} }; }
#define ROS_WARN(...) do {std::fprintf(stderr,__VA_ARGS__);} while(0)
#define ROS_ERROR(...) do {std::fprintf(stderr,__VA_ARGS__);} while(0)
#define ROS_INFO(...) do {std::fprintf(stderr,__VA_ARGS__);} while(0)
#define ROS_ASSERT(condition) assert(condition)
"""
    imu = """#pragma once
#include <memory>
#include <ros/ros.h>
namespace geometry_msgs {struct Vector3 {double x=0,y=0,z=0;};}
namespace sensor_msgs { struct Imu {using Ptr=std::shared_ptr<Imu>;using ConstPtr=std::shared_ptr<const Imu>;struct {ros::Time stamp;} header;geometry_msgs::Vector3 angular_velocity,linear_acceleration;};using ImuConstPtr=Imu::ConstPtr; }
"""
    pose = """#pragma once
namespace fast_lio {struct Pose6D {double offset_time=0;double acc[3]{},gyr[3]{},vel[3]{},pos[3]{},rot[9]{};};}
"""
    files = {"ros/ros.h": time, "sensor_msgs/Imu.h": imu, "geometry_msgs/Vector3.h": "#pragma once\n#include <sensor_msgs/Imu.h>\n",
             "fast_lio/Pose6D.h": pose, "preprocess.h": "#pragma once\nenum LID_TYPE{AVIA=1,VELO16,OUST64,MARSIM};\n"}
    for empty in ("nav_msgs/Odometry.h", "tf/transform_broadcaster.h", "eigen_conversions/eigen_msg.h", "pcl_conversions/pcl_conversions.h", "sensor_msgs/PointCloud2.h"):
        files[empty] = "#pragma once\n"
    for name, data in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)


def build(output, compiler, *, upstream_source=None):
    output = Path(output).resolve()
    if output.exists(): raise ValueError("native runtime output directory must be new")
    output.mkdir(parents=True)
    try:
        upstream = output / "upstream"
        if upstream_source:
            run(["git", "clone", "--no-hardlinks", str(Path(upstream_source).resolve()), str(upstream)])
        else:
            run(["git", "clone", "--filter=blob:none", "--no-checkout", URL, str(upstream)])
        run(["git", "-C", str(upstream), "checkout", "--detach", COMMIT])
        run(["git", "-C", str(upstream), "submodule", "update", "--init", "--recursive"])
        revision = subprocess.check_output(["git", "-C", str(upstream / "include/ikd-Tree"), "rev-parse", "HEAD"], text=True).strip()
        if revision != IKD_COMMIT: raise ValueError("upstream ikd-Tree pin differs")
        sysroot, dependencies = output / "sysroot", output / "downloads"
        sysroot.mkdir(); dependencies.mkdir()
        lock_path = Path(__file__).with_name("lio_dependencies.json")
        lock = json.loads(lock_path.read_text())
        for package in lock["packages"]:
            destination = dependencies / Path(package["url"]).name
            urllib.request.urlretrieve(package["url"], destination)
            if destination.stat().st_size != package["size_bytes"] or sha256(destination) != package["sha256"]:
                raise ValueError(f"dependency checksum mismatch: {package['package']}")
            extract_dependency(destination, sysroot)
        generated, sections = generated_core(upstream)
        shims = output / "shim"
        write_shims(shims)
        # Copy without modifying bytes so its quoted preprocess.h resolves to
        # the transport shim instead of ROS-only upstream preprocessing headers.
        shutil.copyfile(upstream / "src/IMU_Processing.hpp", output / "IMU_Processing.hpp")
        main_path = Path(__file__).with_name("lio_headless_main.cpp")
        cpp = output / "fastlio_headless.cpp"
        cpp.write_text(generated + "\n" + main_path.read_text())
        native = output / "fastlio_headless"
        compiler = str(Path(compiler).resolve())
        library = sysroot / "usr/lib/x86_64-linux-gnu"
        compiler_lib = Path(compiler).parent.parent / "lib64"
        includes = [shims, output, upstream / "include", sysroot / "usr/include", sysroot / "usr/include/eigen3", sysroot / "usr/include/pcl-1.12"]
        command = [compiler, "-std=c++14", "-O1", "-g0", "-pthread", "-fopenmp", '-DROOT_DIR="/tmp/"']
        command += [f"-I{p}" for p in includes]
        command += [str(cpp), str(upstream / "include/ikd-Tree/ikd_Tree.cpp"), f"-L{library}", f"-L{compiler_lib}",
                    "-Wl,-rpath,$ORIGIN/lib", f"-Wl,-rpath-link,{library}", "-lpcl_filters", "-lpcl_common", "-o", str(native)]
        run(command)
        lib = output / "lib"
        lib.mkdir()
        for root in (library, compiler_lib):
            for pattern in ("libpcl*.so*", "liblz4.so*", "libstdc++.so*", "libgcc_s.so*", "libgomp.so*"):
                for path in root.glob(pattern):
                    if path.is_file(): shutil.copyfile(path.resolve(), lib / path.name)
        environment = dict(os.environ, LD_LIBRARY_PATH=str(lib))
        ldd = subprocess.check_output(["ldd", str(native)], text=True, env=environment)
        if "not found" in ldd: raise ValueError("native runtime has missing dynamic libraries: " + ldd)
        receipt = {"schema": "bhl-native-lio-runtime-v1", "status": "PASS", "upstream_commit": COMMIT,
                   "ikd_tree_commit": IKD_COMMIT, "upstream_repository": URL,
                   "upstream_laser_mapping_sha256": sha256(upstream / "src/laserMapping.cpp"),
                   "upstream_imu_processing_sha256": sha256(upstream / "src/IMU_Processing.hpp"),
                   "unchanged_estimator_section_sha256": [hashlib.sha256(s.encode()).hexdigest() for s in sections],
                   "transport_main_sha256": sha256(main_path), "dependency_lock_sha256": sha256(lock_path),
                   "compiler": compiler, "compiler_version": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
                   "compile_command": command, "binary_sha256": sha256(native), "ldd": ldd,
                   "algorithm_scope": "Upstream FAST-LIO2 ImuProcess, IKFoM, ikd-Tree, map/measurement callbacks; headless ROS transport replacement only",
                   "differences": ["ROS subscriptions/publication/viewer/debug logs omitted", "feature-disabled, given-offset Velodyne conversion in transport entry point", "one explicit tracking state for every original scan, including initialization", "tracking requires >=1 final effective map match; no upstream covariance-based tracking classifier", "C++ optimization O1 instead of upstream O3; timing must disclose this"]}
        (output / "runtime.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        for directory in (dependencies, sysroot): shutil.rmtree(directory)
        return receipt
    except Exception as error:
        (output / "build_failure.json").write_text(json.dumps({"status": "INCOMPLETE", "error": str(error)}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compiler", default=shutil.which("g++"))
    parser.add_argument("--upstream-source")
    args = parser.parse_args()
    if not args.compiler: parser.error("an isolated C++14 compiler is required")
    build(args.output, args.compiler, upstream_source=args.upstream_source)
