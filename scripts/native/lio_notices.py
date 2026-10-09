#!/usr/bin/env python3
"""Retain exact pinned dependency copyright and GCC runtime license files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def download(row,path):
    urllib.request.urlretrieve(row["url"],path)
    if path.stat().st_size!=row["size_bytes"] or digest(path)!=row["sha256"]:
        raise ValueError("license source checksum mismatch: "+row["url"])


def retain_notices(sysroot,output):
    """Copy notices before private dependency sysroot cleanup, resolving links."""
    sysroot,output=Path(sysroot).resolve(),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    selected=list((sysroot/"usr/share/doc").glob("*/copyright"))
    selected+=list((sysroot/"usr/share/common-licenses").glob("*"))
    if not selected:raise ValueError("dependency payloads contain no notices")
    for source in sorted(selected):
        if not source.is_file():continue
        if not source.resolve().is_relative_to(sysroot):raise ValueError("notice symlink escapes sysroot")
        destination=output/"ubuntu"/source.relative_to(sysroot)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
    lock_path=Path(__file__).with_name("lio_license_sources.json")
    lock=json.loads(lock_path.read_text())
    for row in lock["gcc_files"]:
        destination=output/"gcc-12.5.0"/row["path"]
        destination.parent.mkdir(parents=True,exist_ok=True)
        download(row,destination)
    dependency_lock=Path(__file__).with_name("lio_dependencies.json")
    shutil.copyfile(dependency_lock,output/dependency_lock.name)
    shutil.copyfile(lock_path,output/lock_path.name)
    receipt={"schema":"bhl-native-lio-notices-v1","status":"PASS",
             "dependency_lock_sha256":digest(dependency_lock),"license_source_lock_sha256":digest(lock_path),
             "gcc_release":lock["gcc_release"],"gcc_source_commit":lock["gcc_commit"],
             "scope":"Copyright files from exact checksum-pinned Ubuntu build dependency payloads, common-license text from pinned base-files, GCC12.5 runtime license and source copyright headers; no runtime binary changes",
             "files":{str(p.relative_to(output)):{"sha256":digest(p),"bytes":p.stat().st_size} for p in sorted(output.rglob("*")) if p.is_file()}}
    (output/"notices-receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt


def provision(output):
    from lio_build import extract_dependency
    lock=json.loads(Path(__file__).with_name("lio_dependencies.json").read_text())
    license_lock=json.loads(Path(__file__).with_name("lio_license_sources.json").read_text())
    with tempfile.TemporaryDirectory(prefix="bhl-lio-notices-") as temporary:
        root=Path(temporary);sysroot=root/"sysroot";sysroot.mkdir()
        for row in lock["packages"]+[license_lock["common_licenses_package"]]:
            package=root/Path(row["url"]).name
            download(row,package);extract_dependency(package,sysroot)
            package.unlink()
        return retain_notices(sysroot,output)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    receipt=provision(args.output)
    print(json.dumps({k:v for k,v in receipt.items() if k!="files"}),flush=True)
