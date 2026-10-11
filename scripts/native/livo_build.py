#!/usr/bin/env python3
"""Build pinned FAST-LIVO2 native estimation units with headless ROS transport.

Preparation uses an existing Ubuntu 22.04 container, downloads private packages,
and records their checksums. Compilation must run in an allocated Slurm step.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

PINS = {
    'FAST-LIVO2': ('https://github.com/hku-mars/FAST-LIVO2.git', '0d2c0346107b75b59934975adec9a6eeeb913c64'),
    'Sophus': ('https://github.com/strasdat/Sophus.git', 'a621ff2e56c56c839a6c40418d42c3c254424b5c'),
    'rpg_vikit': ('https://github.com/xuankuzcr/rpg_vikit.git', '6c886c8e5d83997806e00294826d528cea3581dd'),
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(args, **kwargs):
    print('RUN', ' '.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def record(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def prepare(work):
    work.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, (url, pin) in PINS.items():
        dest = work / name
        if not dest.exists(): run(['git', 'clone', '--filter=blob:none', '--no-checkout', url, dest])
        run(['git', '-C', dest, 'checkout', '--detach', pin])
        actual = subprocess.check_output(['git', '-C', dest, 'rev-parse', 'HEAD'], text=True).strip()
        if actual != pin: raise ValueError('upstream pin changed: ' + name)
        sources[name] = {'repository': url, 'commit': actual}
    apt = work / 'apt'
    (apt / 'lists/partial').mkdir(parents=True, exist_ok=True)
    (apt / 'cache/archives/partial').mkdir(parents=True, exist_ok=True)
    options = ['-o', f'Dir::State::Lists={apt}/lists', '-o', f'Dir::Cache={apt}/cache',
               '-o', 'Debug::NoLocking=true', '-o', 'APT::Sandbox::User=root']
    run(['apt-get', *options, 'update'])
    packages = ['libopencv-dev', 'libeigen3-dev']
    uris = subprocess.check_output(['apt-get', *options, '--print-uris', '--yes', '--no-install-recommends',
                                    '--download-only', 'install', *packages], text=True)
    (work / 'ubuntu-package-uris.txt').write_text(uris)
    run(['apt-get', *options, '--yes', '--no-install-recommends', '--download-only', 'install', *packages])
    sysroot = work / 'sysroot'
    sysroot.mkdir(exist_ok=True)
    dependencies = []
    for archive in sorted((apt / 'cache/archives').glob('*.deb')):
        dependencies.append({'file': archive.name, 'bytes': archive.stat().st_size, 'sha256': sha(archive),
                             'package': subprocess.check_output(['dpkg-deb', '-f', archive, 'Package', 'Version'], text=True)})
        run(['dpkg-deb', '-x', archive, sysroot])
    # Reuse independently pinned PCL/Eigen/Boost packages from the prior native LIO build.
    import urllib.request
    lock = json.loads(Path(__file__).with_name('lio_dependencies.json').read_text())
    for item in lock['packages']:
        archive = apt / 'cache/archives' / Path(item['url']).name
        if not archive.exists(): urllib.request.urlretrieve(item['url'], archive)
        if archive.stat().st_size != item['size_bytes'] or sha(archive) != item['sha256']:
            raise ValueError('pinned native dependency differs: ' + item['package'])
        run(['dpkg-deb', '-x', archive, sysroot])
        dependencies.append(item)
    record(work / 'preparation.json', {'schema': 'bhl-native-livo-preparation-v1', 'sources': sources,
                                      'packages': dependencies, 'sif_expected': 'Ubuntu 22.04',
                                      'files': inventory(sysroot)})


def inventory(directory):
    return {str(p.relative_to(directory)): {'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in sorted(directory.rglob('*')) if p.is_file() and not p.is_symlink()}


def write_shims(directory):
    files = {
      'ros/ros.h': r'''#pragma once
#include <cstdio>
#include <cassert>
#include <string>
namespace ros {
struct Time {double value=0; double toSec() const {return value;} Time& fromSec(double v){value=v;return *this;}};
struct Duration {}; struct Rate {explicit Rate(int){} void sleep(){}};
struct Publisher {template<class T> void publish(const T&) const {}};
struct NodeHandle {template<class T> void param(const std::string&, T& output, const T& fallback){output=fallback;}};
}
#define ROS_WARN(...) do {std::fprintf(stderr,__VA_ARGS__);} while(0)
#define ROS_ERROR(...) do {std::fprintf(stderr,__VA_ARGS__);} while(0)
#define ROS_INFO(...) do {std::fprintf(stderr,__VA_ARGS__);} while(0)
#define ROS_ASSERT(condition) assert(condition)
''',
      'sensor_msgs/Imu.h': r'''#pragma once
#include <memory>
#include <ros/ros.h>
namespace geometry_msgs {struct Vector3 {double x=0,y=0,z=0;}; struct Quaternion {double x=0,y=0,z=0,w=1;};}
namespace sensor_msgs {struct Imu {using Ptr=std::shared_ptr<Imu>;using ConstPtr=std::shared_ptr<const Imu>;struct {ros::Time stamp;} header;geometry_msgs::Vector3 angular_velocity,linear_acceleration;};using ImuConstPtr=Imu::ConstPtr;}
''',
      'tf/transform_broadcaster.h': r'''#pragma once
#include <sensor_msgs/Imu.h>
#include <cmath>
namespace tf {inline geometry_msgs::Quaternion createQuaternionMsgFromRollPitchYaw(double r,double p,double y){
geometry_msgs::Quaternion q; double cr=cos(r/2),sr=sin(r/2),cp=cos(p/2),sp=sin(p/2),cy=cos(y/2),sy=sin(y/2);
q.w=cr*cp*cy+sr*sp*sy;q.x=sr*cp*cy-cr*sp*sy;q.y=cr*sp*cy+sr*cp*sy;q.z=cr*cp*sy-sr*sp*cy;return q;}}
''',
      'visualization_msgs/Marker.h': r'''#pragma once
#include <vector>
#include <sensor_msgs/Imu.h>
namespace visualization_msgs {struct Marker {
static const int CYLINDER=3,ADD=0; struct {std::string frame_id;ros::Time stamp;} header;
std::string ns;int id=0,type=0,action=0;struct {geometry_msgs::Vector3 position;geometry_msgs::Quaternion orientation;} pose;
geometry_msgs::Vector3 scale;struct {double r=0,g=0,b=0,a=0;} color;ros::Duration lifetime;
};struct MarkerArray {std::vector<Marker> markers;};}
''',
      'visualization_msgs/MarkerArray.h': '#pragma once\n#include <visualization_msgs/Marker.h>\n',
      'nav_msgs/Odometry.h': '#pragma once\n',
    }
    for relative, text in files.items():
        path = directory / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)


def build(work, output, jobs=2):
    if not os.environ.get('SLURM_JOB_ID'): raise ValueError('native compilation requires an allocated Slurm job')
    prep = json.loads((work / 'preparation.json').read_text())
    for name, (_, pin) in PINS.items():
        if prep['sources'][name]['commit'] != pin: raise ValueError('wrong prepared source pin')
        actual = subprocess.check_output(['git', '-C', work / name, 'rev-parse', 'HEAD'], text=True).strip()
        if actual != pin: raise ValueError('source checkout changed')
    for relative, item in prep['files'].items():
        path = work / 'sysroot' / relative
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('prepared dependency bytes changed: ' + relative)
    output.mkdir(parents=True, exist_ok=True)
    upstream, sophus, vikit = (work / n for n in ('FAST-LIVO2', 'Sophus', 'rpg_vikit'))
    compatibility = []
    shim = work / 'shim'; write_shims(shim)
    adapter = Path(__file__).resolve().parent
    sysroot = work / 'sysroot'
    include = [shim, upstream / 'include', sophus, vikit / 'vikit_common/include', sysroot / 'usr/include',
               sysroot / 'usr/include/eigen3', sysroot / 'usr/include/pcl-1.12', sysroot / 'usr/include/opencv4',
               sysroot / 'usr/include/x86_64-linux-gnu']
    sources = [upstream / 'src' / f for f in ('IMU_Processing.cpp', 'voxel_map.cpp', 'vio.cpp', 'frame.cpp', 'visual_point.cpp')]
    sources += [sophus / 'sophus' / f for f in ('so3.cpp', 'se3.cpp')]
    sources += [vikit / 'vikit_common/src' / f for f in ('pinhole_camera.cpp', 'math_utils.cpp', 'robust_cost.cpp', 'vision.cpp')]
    sources += [adapter / 'livo_headless_main.cpp']
    source_receipts = [{'file': str(p), 'sha256': sha(p)} for p in sources]
    header_receipts = {}
    for path in sorted((upstream / 'include').rglob('*')):
        if path.is_file():
            relative = path.relative_to(upstream)
            exact = subprocess.check_output(['git', '-C', upstream, 'show', f'{PINS["FAST-LIVO2"][1]}:{relative}'])
            if path.read_bytes() != exact: raise ValueError('native estimator header source modified')
            header_receipts[str(relative)] = sha(path)
    upstream_files = {}
    for path in sources:
        if upstream in path.parents:
            relative = path.relative_to(upstream)
            exact = subprocess.check_output(['git', '-C', upstream, 'show', f'{PINS["FAST-LIVO2"][1]}:{relative}'])
            if path.read_bytes() != exact: raise ValueError('native estimator algorithm source modified')
            upstream_files[str(relative)] = sha(path)
    # Upstream defines a non-inline comparator in IMU_Processing.h. Combine
    # that exact source and the transport into one translation unit, retaining
    # source bytes and avoiding duplicate linkage without algorithm edits.
    combined = work / 'imu_and_transport.cpp'
    combined.write_text(f'#include \"{upstream}/src/IMU_Processing.cpp\"\n#include \"{adapter}/livo_headless_main.cpp\"\n')
    compile_sources = [p for p in sources if p.name not in {'IMU_Processing.cpp', 'livo_headless_main.cpp'}] + [combined]
    includes = ' '.join('"' + str(p) + '"' for p in include)
    source_lines = '\n'.join('"' + str(p) + '"' for p in compile_sources)
    (work / 'CMakeLists.txt').write_text(f'''cmake_minimum_required(VERSION 3.16)
project(BHL_LIVO_NATIVE LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O1 -g0 -DNDEBUG")
include_directories({includes})
link_directories("{sysroot}/usr/lib/x86_64-linux-gnu" "{sysroot}/usr/lib")
add_executable(fastlivo_headless {source_lines})
target_compile_options(fastlivo_headless PRIVATE -fopenmp -pthread -include omp.h)
target_compile_definitions(fastlivo_headless PRIVATE ROOT_DIR="/tmp/" MP_PROC_NUM=1)
target_link_libraries(fastlivo_headless pcl_filters pcl_common opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d opencv_features2d gomp pthread)
set_target_properties(fastlivo_headless PROPERTIES BUILD_RPATH "{sysroot}/usr/lib/x86_64-linux-gnu;{sysroot}/usr/lib" INSTALL_RPATH "$ORIGIN/lib")
''')
    env = dict(os.environ, LD_LIBRARY_PATH=str(sysroot / 'usr/lib/x86_64-linux-gnu') + ':' + str(sysroot / 'usr/lib'))
    run(['cmake', '-S', work, '-B', work / 'build', '-DCMAKE_BUILD_TYPE=Release'], env=env)
    run(['cmake', '--build', work / 'build', '--parallel', str(jobs)], env=env)
    native = work / 'build/fastlivo_headless'
    ldd = subprocess.check_output(['ldd', native], env=env, text=True)
    if 'not found' in ldd: raise ValueError('native runtime dependency unresolved')
    shutil.copy2(native, output / native.name)
    (output / 'lib').mkdir(exist_ok=True)
    libraries = {}
    for line in ldd.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == '=>' and parts[2].startswith('/'):
            # Keep system glibc tied to the declared Ubuntu SIF; ship other deps.
            if parts[0] in {'libc.so.6', 'libm.so.6', 'libpthread.so.0', 'libdl.so.2', 'librt.so.1'}: continue
            path = Path(parts[2]); shutil.copy2(path.resolve(), output / 'lib' / parts[0]); libraries[parts[0]] = sha(path)
    receipt = {'schema': 'bhl-native-livo-runtime-v1', 'status': 'PASS', 'upstream_commit': PINS['FAST-LIVO2'][1],
               'sources': prep['sources'], 'binary_sha256': sha(output / native.name), 'libraries_sha256': libraries,
               'unchanged_estimator_sources': upstream_files, 'unchanged_estimator_headers': header_receipts, 'compiled_sources': source_receipts,
               'compatibility': compatibility, 'preparation_sha256': sha(work / 'preparation.json'), 'ldd': ldd,
               'compile_options': '-std=c++17 -O1 -g0 -DNDEBUG -fopenmp, one estimator thread',
               'sensors': ['left camera', 'timed 3D lidar', 'SI high-rate IMU'],
               'transport_scope': 'Typed ROS containers and publishers replaced; native IMU, voxel map and direct photometric estimator units unchanged; explicit sequential LIO/VIO orchestration follows upstream LIVMapper.',
               'native_source_license': 'GPL-2.0; upstream LICENSE retained',
               'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'slurm_step_id': os.environ.get('SLURM_STEP_ID')}
    record(output / 'runtime.json', receipt)
    shutil.copy2(upstream / 'LICENSE', output / 'FAST-LIVO2-LICENSE')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'build'))
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if args.action == 'prepare': prepare(args.work.resolve())
    else:
        if args.output is None: parser.error('--output is required for build')
        print(json.dumps(build(args.work.resolve(), args.output.resolve(), args.jobs), indent=2))


if __name__ == '__main__': main()
