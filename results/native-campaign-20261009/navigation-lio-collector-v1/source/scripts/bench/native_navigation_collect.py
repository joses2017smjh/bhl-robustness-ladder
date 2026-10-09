#!/usr/bin/env python3
"""Collect completed native navigation using a tiny checksum-pinned source bundle."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

# A collector must not mutate the exact source inventory it is verifying.
sys.dont_write_bytecode=True


def digest(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def collect(bundle,campaign_dir,job_name,output):
    bundle,campaign_dir,output=Path(bundle).resolve(),Path(campaign_dir).resolve(),Path(output).resolve()
    manifest=json.loads((bundle/"manifest.json").read_text())
    actual={str(p.relative_to(bundle)) for p in (bundle/"source").rglob("*") if p.is_file()}
    if actual!=set(manifest):raise ValueError("collector source inventory differs")
    for name,row in manifest.items():
        path=bundle/name
        if path.is_symlink() or not path.resolve().is_relative_to(bundle) or path.stat().st_size!=row["bytes"] or digest(path)!=row["sha256"]:
            raise ValueError("collector source hash differs: "+name)
    intake=json.loads((campaign_dir/"intake.json").read_text())
    submission=json.loads((campaign_dir/(job_name+"-submission.json")).read_text())
    if submission["status"]!="SUBMITTED" or submission["archive_sha256"]!=intake["archive_sha256"]:
        raise ValueError("submitted experiment pin differs")
    job_root=campaign_dir/(job_name+"-"+submission["job_id"])
    completion=json.loads((job_root/"completion.json").read_text())
    launch=json.loads((job_root/"launch.json").read_text())
    if completion["archive_sha256"]!=intake["archive_sha256"] or launch["archive_sha256"]!=intake["archive_sha256"]:
        raise ValueError("completed experiment source pin differs")
    if str(launch["job_id"])!=submission["job_id"] or launch["job"]["name"]!=job_name:
        raise ValueError("completed experiment job identity differs")
    entry=bundle/"source/scripts/bench/native_navigation_observe.py"
    spec=importlib.util.spec_from_file_location("pinned_navigation_collector_observer",entry)
    observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
    result=observer.observe(job_root,output)
    evidence=output/"evidence";evidence.mkdir()
    for name in ["completion.json","launch.json","campaign_result.json"]:
        shutil.copyfile(job_root/name,evidence/name)
    shutil.copyfile(campaign_dir/(job_name+"-submission.json"),evidence/"submission.json")
    receipt={"schema":"bhl-native-navigation-collection-v1","status":"PASS",
        "source_bundle_manifest_sha256":digest(bundle/"manifest.json"),"experiment_frozen_archive_sha256":intake["archive_sha256"],
        "job_id":submission["job_id"],"job_name":job_name,"raw_archive_path":str(job_root/"outputs.tar.gz"),
        "raw_archive_sha256":completion["files"]["outputs.tar.gz"]["sha256"],"scientific_status":result["scientific_status"],
        "files":{str(p.relative_to(output)):{"sha256":digest(p),"bytes":p.stat().st_size} for p in sorted(output.rglob("*")) if p.is_file()}}
    (output/"collection-receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,required=True);p.add_argument("--campaign-dir",type=Path,required=True)
    p.add_argument("--job-name",required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();r=collect(a.bundle,a.campaign_dir,a.job_name,a.output)
    print(json.dumps({k:v for k,v in r.items() if k!="files"}),flush=True)
