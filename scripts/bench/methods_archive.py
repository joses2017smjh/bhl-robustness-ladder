#!/usr/bin/env python3
"""Publish completed method-run archives, verify a fresh download, retain receipt.

Only new campaign outputs are eligible for local packaging retirement. Source
freezes, unfinished runs, older campaigns and upstream runtimes are untouched.
This command runs outside the credential-free scientific container.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

REPOSITORY = "joses2017smjh/bhl-robustness-ladder"
TAG = "methods-campaign-evidence-20261010"
AUTHORIZED_ROOT = Path("/nfs/stak/users/sanchej7/humanoid-methods-20261010")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gh(*args):
    result = subprocess.run(["gh", *map(str, args)], capture_output=True, env={**os.environ, "TMPDIR": "/tmp"})
    if result.returncode:
        raise RuntimeError(f"GitHub operation failed with exit {result.returncode}")
    return result.stdout


def publish(job, *, retire=False):
    job = Path(job).resolve()
    if not job.is_relative_to(AUTHORIZED_ROOT) or job == AUTHORIZED_ROOT:
        raise ValueError("only explicitly scoped new methods campaign outputs allowed")
    receipt_path = job/"publication.json"
    if receipt_path.exists() and not (job/"outputs.tar.gz").exists():
        recorded = json.loads(receipt_path.read_text())
        completed = json.loads((job/"completion.json").read_text())
        if (recorded.get("schema") != "bhl-methods-archive-v1"
                or recorded["sha256"] != completed["files"]["outputs.tar.gz"]["sha256"]
                or not recorded["local_packaging_retired"]):
            raise ValueError("retained publication/completion lineage differs")
        return recorded
    completion = json.loads((job/"completion.json").read_text())
    launch = json.loads((job/"launch.json").read_text())
    intake = json.loads((job.parent/"intake.json").read_text())
    if not completion.get("finished_utc") or completion["archive_sha256"] != launch["archive_sha256"] or launch["archive_sha256"] != intake["archive_sha256"]:
        raise ValueError("completed source lineage differs")
    for name, spec in completion["files"].items():
        if Path(name).name != name:
            raise ValueError("unsafe output filename")
        path = job/name
        if path.is_symlink() or path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
            raise ValueError("completed evidence changed: " + name)
    raw = job/"outputs.tar.gz"
    spec = completion["files"][raw.name]
    asset_name = job.parent.name + "--" + job.name + ".tar.gz"
    with tempfile.TemporaryDirectory(prefix="bhl-methods-publish-", dir="/tmp") as temporary:
        work = Path(temporary)
        release = json.loads(gh("api", f"repos/{REPOSITORY}/releases/tags/{TAG}"))
        assets = [a for a in release["assets"] if a["name"] == asset_name]
        if not assets:
            staged = work/asset_name
            shutil.copyfile(raw, staged)
            if digest(staged) != spec["sha256"]:
                raise ValueError("staged evidence changed")
            gh("release", "upload", TAG, staged, "--repo", REPOSITORY)
            staged.unlink()
            release = json.loads(gh("api", f"repos/{REPOSITORY}/releases/tags/{TAG}"))
            assets = [a for a in release["assets"] if a["name"] == asset_name]
        if len(assets) != 1 or assets[0]["size"] != spec["bytes"]:
            raise ValueError("published asset identity/size differs")
        asset = assets[0]
        if asset.get("digest") and asset["digest"] != "sha256:" + spec["sha256"]:
            raise ValueError("GitHub server checksum differs")
        gh("release", "download", TAG, "--repo", REPOSITORY, "--pattern", asset_name, "--dir", work)
        downloaded = work/asset_name
        if digest(downloaded) != spec["sha256"]:
            raise ValueError("independent fresh download checksum differs")
        retained = Path("/tmp/bhl-methods-evidence-20261010")/asset_name
        retained.parent.mkdir(exist_ok=True)
        if retained.exists() and digest(retained) != spec["sha256"]:
            raise ValueError("local verified mirror conflict")
        if not retained.exists():
            shutil.copyfile(downloaded, retained)
        record = dict(schema="bhl-methods-archive-v1", verified_utc=datetime.now(timezone.utc).isoformat(),
            job_id=launch["job_id"], scientific_status=completion["status"],
            source_archive_sha256=intake["archive_sha256"], asset_name=asset_name,
            url=asset["browser_download_url"], sha256=spec["sha256"], bytes=spec["bytes"],
            verification="completion hash + GitHub asset identity + fresh download SHA256",
            local_verified_mirror=str(retained), local_packaging_retired=bool(retire),
            scope="raw evidence publication; no scientific qualification inferred")
        temporary_receipt = receipt_path.with_suffix(".tmp")
        temporary_receipt.write_text(json.dumps(record, indent=2)+"\n")
        temporary_receipt.replace(receipt_path)
        if retire:
            if digest(raw) != spec["sha256"]:
                raise ValueError("local packaging changed during publication")
            raw.unlink()
        return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("--retire-local-packaging", action="store_true")
    args = parser.parse_args()
    print(json.dumps(publish(args.job, retire=args.retire_local_packaging), indent=2))
