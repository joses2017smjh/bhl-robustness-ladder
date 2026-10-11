#!/usr/bin/env python3
"""Build the pinned, genuine ORB-SLAM3 stereo runtime in a private Ubuntu sysroot.

Run inside the existing Ubuntu 22.04 SIF on an allocated CPU build job. Nothing
is installed into the SIF, shared Python environment, or system directories.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

PIN = "4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4"
PACKAGES = ["libopencv-dev", "libeigen3-dev", "libboost-serialization-dev", "libssl-dev"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(argv, *, cwd=None, env=None):
    print("RUN", json.dumps([str(x) for x in argv]), flush=True)
    return subprocess.run([str(x) for x in argv], cwd=cwd, env=env, check=True)


def cmake_text(upstream, adapter, sysroot):
    # Actual upstream sources, except the two GUI-only translation units.
    sources = sorted(str(p) for p in (upstream / "src").glob("*.cc")
                     if p.name not in {"MapDrawer.cc", "Viewer.cc"})
    sources += sorted(str(p) for p in (upstream / "src").glob("*.cpp"))
    sources += sorted(str(p) for p in (upstream / "src" / "CameraModels").glob("*.cpp"))
    sources += [str(adapter / "orb_headless.cc")]
    quoted = "\n".join(f'  "{p}"' for p in sources)
    return f'''cmake_minimum_required(VERSION 3.16)
project(BHL_ORB_NATIVE LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
include_directories("{upstream}" "{upstream}/include" "{upstream}/include/CameraModels"
 "{upstream}/Thirdparty/Sophus" "{sysroot}/usr/include/eigen3" "{sysroot}/usr/include/opencv4"
 "{sysroot}/usr/include" "{sysroot}/usr/include/x86_64-linux-gnu")
link_directories("{sysroot}/usr/lib/x86_64-linux-gnu")
add_subdirectory("{upstream}/Thirdparty/DBoW2" dbow)
add_subdirectory("{upstream}/Thirdparty/g2o" g2o)
add_library(ORB_SLAM3 SHARED
{quoted}
)
target_compile_definitions(ORB_SLAM3 PRIVATE COMPILEDWITHC11)
target_link_libraries(ORB_SLAM3 DBoW2 g2o opencv_core opencv_imgproc opencv_features2d
 opencv_flann opencv_calib3d opencv_imgcodecs boost_serialization crypto pthread)
add_executable(orb_native "{adapter}/orb_runner.cc")
target_link_libraries(orb_native ORB_SLAM3)
'''


def patch_headless(upstream):
    replacements = {
        "include/MapDrawer.h": ('#include<pangolin/pangolin.h>', 'namespace pangolin { class OpenGlMatrix; }'),
        "include/Map.h": ('#include <pangolin/pangolin.h>',
                          'using GLubyte = unsigned char; // Original OpenGL viewer-thumbnail byte type, no GUI dependency.'),
        "src/System.cc": ('#include <pangolin/pangolin.h>', '// Headless: visualization translation units are replaced.'),
    }
    patches = []
    for relative, (before, after) in replacements.items():
        path = upstream / relative
        text = path.read_text()
        if text.count(before) != 1:
            raise ValueError(f"Expected one pinned visualization include in {relative}")
        original = sha(path)
        path.write_text(text.replace(before, after))
        patches.append({"file": relative, "original_sha256": original, "modified_sha256": sha(path),
                        "before": before, "after": after})
    # Build flags cannot use -march=native: the executable moves between CPUs.
    for relative in ("Thirdparty/DBoW2/CMakeLists.txt", "Thirdparty/g2o/CMakeLists.txt"):
        path = upstream / relative
        original = sha(path)
        path.write_text(path.read_text().replace("-march=native", "").replace("-O3", "-O2"))
        patches.append({"file": relative, "original_sha256": original, "modified_sha256": sha(path),
                        "reason": "portable CPU instruction set and bounded compiler memory; no algorithm change"})
    return patches


def library_closure(executable, environment):
    output = subprocess.check_output(["ldd", str(executable)], env=environment, text=True)
    if "not found" in output:
        raise RuntimeError("Native dependency unresolved:\n" + output)
    libraries = {}
    for line in output.splitlines():
        fields = line.strip().split()
        if "=>" in fields and len(fields) >= 3 and fields[2].startswith("/"):
            libraries[fields[0]] = Path(fields[2]).resolve()
    return output, libraries


def resume_compiled(args, adapter):
    """Verify a predeclared actual failed-link tree before reusing its objects."""
    receipt = json.loads(args.resume_compiled.read_text())
    if receipt.get("schema") != "bhl-orb-compiled-work-v1" or receipt.get("upstream_commit") != PIN:
        raise ValueError("Pinned genuine compiled-work receipt required")
    if Path(receipt["root"]).resolve() != args.work.resolve() or receipt["host"] != os.uname().nodename:
        raise ValueError("Compiled work belongs to another host/path")
    for name, row in receipt["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe compiled-work member")
        path = args.work / relative
        if not path.is_file() or path.is_symlink() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("Compiled input changed: " + name)
    for name, checksum in receipt["adapters_sha256"].items():
        if sha(adapter / name) != checksum:
            raise ValueError("Compiled adapter differs from current frozen source")
    original = Path(receipt["original_partial_runtime"])
    for name, row in receipt["original_partial_files"].items():
        if sha(original / name) != row["sha256"] or (original / name).stat().st_size != row["bytes"]:
            raise ValueError("Original source/dependency evidence changed")
        shutil.copy2(original / name, args.output / name)
    shutil.copy2(args.resume_compiled, args.output / "compiled-work-provenance.json")
    print(json.dumps({"resumed_compiled_files_verified": len(receipt["files"]),
                      "source_build": receipt["source_build"], "scope": "Dependency search repair only; no native source/object modification"}), flush=True)
    dependencies = []
    for row in receipt["ubuntu_packages"]:
        item = dict(row)
        item["dpkg_fields"] = subprocess.check_output(
            ["dpkg-deb", "-f", str(args.work / "apt/cache/archives" / row["file"]),
             "Package", "Version", "Architecture"], text=True)
        dependencies.append(item)
    return (args.work / "ORB_SLAM3", args.work / "sysroot", receipt["patches"],
            dependencies, args.work / "project")


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--upstream", type=Path, help="Optional pristine already-cloned exact pinned source")
    parser.add_argument("--resume-compiled", type=Path,
                        help="Predeclared hashed genuine compiled-work receipt; final link/dependency repair only")
    args = parser.parse_args(argv)
    if args.output.exists() or (args.work.exists() and not args.resume_compiled):
        raise ValueError("Build work and output paths must be new")
    if not 1 <= args.jobs <= 8:
        raise ValueError("Bounded build requires 1..8 parallel compiler processes")
    start = time.time()
    if not args.resume_compiled:
        args.work.mkdir(parents=True)
    args.output.mkdir(parents=True)
    adapter = Path(__file__).resolve().parent
    if args.resume_compiled:
        source, sysroot, patches, dependency_receipts, project = resume_compiled(args, adapter)
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = str(sysroot / "usr/lib/x86_64-linux-gnu") + ":" + str(sysroot / "usr/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    else:
        source = args.work / "ORB_SLAM3"
        if args.upstream:
            run(["git", "clone", "--no-hardlinks", str(args.upstream), str(source)])
        else:
            run(["git", "clone", "--filter=blob:none", "--no-checkout",
                 "https://github.com/UZ-SLAMLab/ORB_SLAM3.git", str(source)])
        run(["git", "checkout", PIN], cwd=source)
        actual_pin = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        if actual_pin != PIN:
            raise RuntimeError("Upstream pin differs")
        apt = args.work / "apt"
        (apt / "lists" / "partial").mkdir(parents=True)
        (apt / "cache" / "archives" / "partial").mkdir(parents=True)
        options = ["-o", f"Dir::State::Lists={apt}/lists", "-o", f"Dir::Cache={apt}/cache",
                   "-o", "Debug::NoLocking=true", "-o", "APT::Sandbox::User=root"]
        run(["apt-get", *options, "update"])
        # Exact archive checksums and dpkg versions are recorded for reconstruction.
        uri_plan = subprocess.check_output(["apt-get", *options, "--print-uris", "--yes", "--no-install-recommends",
                                            "--download-only", "install", *PACKAGES], text=True)
        (args.output / "ubuntu-package-uris.txt").write_text(uri_plan)
        run(["apt-get", *options, "--yes", "--no-install-recommends", "--download-only", "install", *PACKAGES])
        sysroot = args.work / "sysroot"
        sysroot.mkdir()
        dependency_receipts = []
        for package in sorted((apt / "cache" / "archives").glob("*.deb")):
            info = subprocess.check_output(["dpkg-deb", "-f", str(package), "Package", "Version", "Architecture"], text=True)
            dependency_receipts.append({"file": package.name, "bytes": package.stat().st_size,
                                        "sha256": sha(package), "dpkg_fields": info})
            run(["dpkg-deb", "-x", package, sysroot])
        patches = patch_headless(source)
        # Save the precise source used, including visualization-only/portable patches.
        with tarfile.open(args.output / "upstream-source.tar.gz", "w:gz") as archive:
            for path in sorted(source.rglob("*")):
                relative = path.relative_to(source)
                # Examples contain >1 GB of unrelated timestamp/IMU recordings.
                # Retain all actual library source plus build/license metadata.
                included = (relative.parts[0] in {"src", "include", "Thirdparty", "cmake_modules"}
                            or str(relative) in {"CMakeLists.txt", "LICENSE", "Dependencies.md", "build.sh"})
                if not included or ".git" in relative.parts or not path.is_file():
                    continue
                archive.add(path, arcname=str(relative), recursive=False)
        project = args.work / "project"
        project.mkdir()
        (project / "CMakeLists.txt").write_text(cmake_text(source, adapter, sysroot))
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = str(sysroot / "usr/lib/x86_64-linux-gnu") + ":" + str(sysroot / "usr/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
        cmake_args = ["cmake", "-S", project, "-B", args.work / "build",
                      f"-DCMAKE_PREFIX_PATH={sysroot}/usr", f"-DOpenCV_DIR={sysroot}/usr/lib/x86_64-linux-gnu/cmake/opencv4",
                      f"-DEIGEN3_INCLUDE_DIR={sysroot}/usr/include/eigen3"]
        run(cmake_args, env=env)
    run(["cmake", "--build", args.work / "build", "--target", "orb_native", "--parallel", args.jobs], env=env)
    binary = args.work / "build" / "orb_native"
    ldd_output, libraries = library_closure(binary, env)
    (args.output / "ldd-build.txt").write_text(ldd_output)
    (args.output / "bin").mkdir()
    (args.output / "lib").mkdir()
    shutil.copy2(binary, args.output / "bin" / "orb_native")
    # System C/C++ runtime stays supplied by the pinned Ubuntu SIF. Ship all
    # other linked libraries so reconstruction needs no package installation.
    system_names = {"libc.so.6", "libm.so.6", "libpthread.so.0", "libdl.so.2", "librt.so.1",
                    "libstdc++.so.6", "libgcc_s.so.1"}
    linked_receipts = []
    for soname, path in libraries.items():
        linked_receipts.append({"soname": soname, "build_path": str(path), "sha256": sha(path),
                                "shipped": soname not in system_names})
        if soname not in system_names:
            shutil.copy2(path, args.output / "lib" / soname)
    with tarfile.open(source / "Vocabulary" / "ORBvoc.txt.tar.gz", "r:gz") as archive:
        member = archive.getmember("ORBvoc.txt")
        with archive.extractfile(member) as stream, (args.output / "ORBvoc.txt").open("wb") as dest:
            shutil.copyfileobj(stream, dest)
    shutil.copy2(source / "LICENSE", args.output / "ORB-SLAM3-LICENSE.txt")
    shutil.copy2(adapter / "orb_runner.cc", args.output / "orb_runner.cc")
    shutil.copy2(adapter / "orb_headless.cc", args.output / "orb_headless.cc")
    shutil.copy2(project / "CMakeLists.txt", args.output / "CMakeLists.used.txt")
    notices = args.output / "ubuntu-copyright-notices"
    notices.mkdir()
    for documentation in (Path("/usr/share/doc"), sysroot / "usr/share/doc"):
        if documentation.is_dir():
            for copyright_file in sorted(documentation.glob("*/copyright")):
                if copyright_file.is_file():
                    shutil.copy2(copyright_file, notices / (copyright_file.parent.name + ".txt"))
    smoke_env = dict(env)
    smoke_env["LD_LIBRARY_PATH"] = str(args.output / "lib")
    version = subprocess.check_output([str(args.output / "bin" / "orb_native"), "--version"], env=smoke_env, text=True)
    final_ldd, _ = library_closure(args.output / "bin" / "orb_native", smoke_env)
    (args.output / "ldd-runtime.txt").write_text(final_ldd)
    receipt = {"schema": "bhl-native-orb-runtime-v1", "status": "PASS", "upstream_commit": PIN,
               "headless_visualization_only": True, "native_algorithm_sources_modified": False,
               "patches": patches, "ubuntu_packages": dependency_receipts, "linked_libraries": linked_receipts,
               "version_smoke": version.strip(), "build_elapsed_seconds": time.time() - start,
               "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
               "os_release": Path("/etc/os-release").read_text(),
               "files_sha256": {str(p.relative_to(args.output)): sha(p) for p in sorted(args.output.rglob("*")) if p.is_file()}}
    (args.output / "runtime.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "runtime": str(args.output), "bytes": sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())}), flush=True)


if __name__ == "__main__":
    main()
