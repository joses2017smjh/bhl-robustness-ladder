#!/usr/bin/env python3
"""Collect a frozen method campaign and publish its compact, verified outcome.

Run after the serial dispatcher terminates. Missing jobs remain INCOMPLETE.
This utility never submits, reruns, tunes, or modifies scientific experiments.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h34_collect as SAFE
import methods_archive as ARCHIVE
import methods_dispatch as DISPATCH

TRACKS = {"terrain": 9, "stereo": 9, "sensors": 18, "perceptive": 3}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_plan(path, expected):
    if SAFE.digest(path) != expected:
        raise ValueError("collector plan checksum differs")
    plan = SAFE.read_json(path)
    if plan.get("schema") != "bhl-methods-finalization-v1" or plan.get("track") not in TRACKS:
        raise ValueError("unknown finalization protocol")
    root = path.parent.resolve()
    for name, sha in plan["source_sha256"].items():
        relative = SAFE.relative_name(name)
        if SAFE.digest(root/relative) != sha:
            raise ValueError("frozen collection source differs: "+name)
    if not plan["source_sha256"] or Path(__file__).resolve() != root/"scripts/bench/methods_finalize.py":
        raise ValueError("execute the inventoried collection source")
    campaign = Path(plan["campaign_dir"]).resolve()
    if campaign == DISPATCH.AUTHORIZED_ROOT or not campaign.is_relative_to(DISPATCH.AUTHORIZED_ROOT):
        raise ValueError("collection is restricted to this new methods campaign")
    intake = SAFE.read_json(campaign/"intake.json")
    if intake["archive_sha256"] != plan["archive_sha256"] or intake["protocol_sha256"] != plan["protocol_sha256"]:
        raise ValueError("collection/source lineage differs")
    for name, key in (("protocol.json", "protocol_sha256"), ("h34_campaign.sbatch", "launcher_sha256")):
        if SAFE.digest(SAFE.regular_file(campaign/name)) != intake[key]:
            raise ValueError("original scientific metadata changed: "+name)
    SAFE.regular_file(campaign/"manifest.json")
    protocol = SAFE.read_json(campaign/"protocol.json")
    jobs = [row for row in protocol["jobs"] if row["kind"] != "smoke"]
    if len(jobs) != TRACKS[plan["track"]] or [j["name"] for j in jobs] != plan["job_names"]:
        raise ValueError("exact original development cohort required")
    output = Path(plan["result_dir"]).resolve()
    if (not output.is_relative_to(DISPATCH.AUTHORIZED_ROOT/"finalization") or
            output == DISPATCH.AUTHORIZED_ROOT/"finalization"):
        raise ValueError("finalization output must use its dedicated durable directory")
    return plan, campaign, jobs


def verified_campaign_view(campaign, intake, work):
    """Rehydrate a retired source pack only; existing corrupt packs stay errors."""
    archive = campaign/"frozen-campaign.tar.gz"
    if os.path.lexists(archive):
        SAFE.verify_frozen_archive(campaign, intake)
        return campaign
    sha, size = intake["archive_sha256"], intake["archive_bytes"]
    if (not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{64}", sha) or
            type(size) is not int or not 0 < size <= SAFE.MAX_FROZEN_BYTES+16*1024**2):
        raise ValueError("invalid original source archive pin")
    if shutil.disk_usage(work).free < size+512*1024**2:
        raise ValueError("insufficient node-local space for source archive verification")
    view = work/"verified-campaign-source"
    view.mkdir(exist_ok=False)
    for name in ("intake.json", "manifest.json", "protocol.json", "h34_campaign.sbatch"):
        shutil.copyfile(SAFE.regular_file(campaign/name), view/name)
    name = "methods-source-"+sha+".tar.gz"
    ARCHIVE.gh("release", "download", ARCHIVE.TAG, "--repo", ARCHIVE.REPOSITORY,
               "--pattern", name, "--dir", view)
    downloaded = SAFE.regular_file(view/name)
    if downloaded.stat().st_size != size or SAFE.digest(downloaded) != sha:
        raise ValueError("restored source archive checksum differs from original intake")
    downloaded.rename(view/"frozen-campaign.tar.gz")
    SAFE.verify_frozen_archive(view, intake)
    return view


def obtain_raw(job, source_sha, mirror):
    completion = SAFE.read_json(job/"completion.json")
    publication = SAFE.read_json(job/"publication.json")
    raw = completion["files"]["outputs.tar.gz"]
    name = str(SAFE.relative_name(publication["asset_name"]))
    if ("/" in name or publication.get("source_archive_sha256") != source_sha or
            publication.get("sha256") != raw["sha256"] or publication.get("bytes") != raw["bytes"]):
        raise ValueError("publication/raw receipt differs")
    if shutil.disk_usage(mirror).free < raw["bytes"]+512*1024**2:
        raise ValueError("insufficient node-local space for retained raw evidence")
    target = mirror/name
    if not target.exists():
        ARCHIVE.gh("release", "download", ARCHIVE.TAG, "--repo", ARCHIVE.REPOSITORY,
                   "--pattern", name, "--dir", mirror)
    if target.stat().st_size != raw["bytes"] or SAFE.digest(target) != raw["sha256"]:
        raise ValueError("fresh raw download checksum differs")
    return target


def collect_track(track, campaign, jobs, mirror, work, result_path):
    if track in ("sensors", "perceptive"):
        module = __import__(track.replace("sensors", "sensor")+"_methods_collect")
        result = module.collect(campaign, jobs, [mirror])
        write(result_path, result)
        return result
    if track == "stereo":
        import stereo_methods_collect
        stereo_methods_collect.main(["--results", *map(str, (j/"campaign_result.json" for j in jobs)),
                                     "--output", str(result_path)])
    else:
        import terrain_methods_campaign
        roots = []
        for job in jobs:
            raw = mirror/SAFE.read_json(job/"publication.json")["asset_name"]
            extracted = work/job.name
            SAFE.safe_extract(raw, extracted, cap=2*1024**3)
            roots.append(str(extracted))
        terrain_methods_campaign.collect_campaign(campaign/"protocol.json", list(map(Path, roots)), result_path)
    return SAFE.read_json(result_path)


def verify_launch(job, declared, intake, job_id):
    launch = SAFE.read_json(job/"launch.json")
    completion = SAFE.read_json(job/"completion.json")
    pin = completion["files"]["launch.json"]
    if ((job/"launch.json").stat().st_size != pin["bytes"] or SAFE.digest(job/"launch.json") != pin["sha256"] or
            launch.get("job") != declared or launch.get("source_files_verified") != intake["source_files"] or
            str(launch.get("job_id")) != job_id):
        raise ValueError("launch does not match the exact frozen job/source receipt")


def publish_summary(output, track):
    name = "methods-final-"+track+"-20261010-"+SAFE.digest(output/"collection.json")[:12]+".tar.gz"
    with tempfile.TemporaryDirectory(prefix="bhl-method-summary-", dir="/tmp") as folder:
        packed = Path(folder)/name
        with tarfile.open(packed, "w:gz") as tar:
            for item in sorted(output.rglob("*")):
                if item.is_file():
                    tar.add(item, arcname=str(item.relative_to(output)), recursive=False)
        sha = SAFE.digest(packed)
        release = json.loads(ARCHIVE.gh("api", f"repos/{ARCHIVE.REPOSITORY}/releases/tags/{ARCHIVE.TAG}"))
        assets = [a for a in release["assets"] if a["name"] == name]
        if assets:
            raise ValueError("summary asset already exists; refusing an implicit replacement")
        ARCHIVE.gh("release", "upload", ARCHIVE.TAG, packed, "--repo", ARCHIVE.REPOSITORY)
        verify = Path(folder)/"verify"
        verify.mkdir()
        ARCHIVE.gh("release", "download", ARCHIVE.TAG, "--repo", ARCHIVE.REPOSITORY,
                   "--pattern", name, "--dir", verify)
        if SAFE.digest(verify/name) != sha:
            raise ValueError("published summary fresh-download checksum differs")
        record = {"asset_name": name, "sha256": sha, "bytes": packed.stat().st_size,
                  "url": f"https://github.com/{ARCHIVE.REPOSITORY}/releases/download/{ARCHIVE.TAG}/{name}",
                  "verified_utc": datetime.now(timezone.utc).isoformat(),
                  "verification": "fresh independent download SHA256"}
        write(output/"publication.json", record)
        return record


def run(path, expected):
    plan, campaign, declared = validate_plan(path, expected)
    intake = SAFE.read_json(campaign/"intake.json")
    output = Path(plan["result_dir"])
    output.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    status = {"schema": "bhl-methods-finalization-result-v1", "track": plan["track"],
              "plan_sha256": expected, "source_archive_sha256": plan["archive_sha256"],
              "status": "INCOMPLETE", "started_utc": started, "observed_jobs": [], "problems": [],
              "scope": "Complete matched simulation development only; no confirmation or hardware claim"}
    with tempfile.TemporaryDirectory(prefix="bhl-method-collection-", dir="/tmp") as temporary:
        work = Path(temporary)
        source_view = verified_campaign_view(campaign, intake, work)
        status["source_archive_origin"] = ("ORIGINAL_DURABLE_ARCHIVE" if source_view == campaign else
                                             "SHA_VERIFIED_RELEASE_RESTORED_IN_NODE_TMP")
        mirror = work/"raw"
        mirror.mkdir()
        jobs = []
        for declared_job in declared:
            name = declared_job["name"]
            try:
                job_id = DISPATCH.submitted_id(campaign, {"name": name}, plan["archive_sha256"])
                if job_id is None:
                    raise ValueError("declared development job was never submitted")
                state, exit_code = DISPATCH.scheduler_state(job_id)
                if state != "COMPLETED" or exit_code != "0:0":
                    raise ValueError(f"scientific scheduler state {state}, exit {exit_code}")
                job = campaign/(name+"-"+job_id)
                result, completion = DISPATCH.checked_result(job, {"name": name}, plan["archive_sha256"])
                verify_launch(job, declared_job, intake, job_id)
                if result.get("phase") == "smoke":
                    raise ValueError("smoke cannot enter a development summary")
                obtain_raw(job, plan["archive_sha256"], mirror)
                evidence = output/"jobs"/job.name
                evidence.mkdir(parents=True)
                for item in ("campaign_result.json", "completion.json", "launch.json", "publication.json"):
                    shutil.copyfile(job/item, evidence/item)
                status["observed_jobs"].append({"name": name, "job_id": job_id,
                    "result_sha256": completion["files"]["campaign_result.json"]["sha256"]})
                jobs.append(job)
            except (ValueError, OSError, KeyError, RuntimeError) as error:
                status["problems"].append({"name": name, "error": str(error)})
        if not status["problems"] and len(jobs) == len(declared):
            try:
                result = collect_track(plan["track"], source_view, jobs, mirror, work, output/"scientific-summary.json")
                status["status"] = result["status"]
                status["scientific_status"] = result.get("scientific_status", "MEASURED_DEVELOPMENT_NO_CONFIRMATION")
            except (ValueError, OSError, KeyError, RuntimeError) as error:
                status["problems"].append({"error": str(error)})
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write(output/"collection.json", status)
        (output/"report.md").write_text(
            f"# {plan['track'].capitalize()} development collection — October 10, 2026\n\n"
            f"Collection status: **{status['status']}**. Verified {len(jobs)}/{len(declared)} declared development jobs.\n\n"
            "Scientific outcomes and paired statistics are in scientific-summary.json when the complete cohort validates. "
            "Missing evidence is INCOMPLETE; an execution PASS does not establish an improvement. Negative teacher gates "
            "do not become zero-valued student results. All measurements are simulation development work.\n\n"
            + "\n".join("- "+p.get("name", "collection")+": "+p["error"] for p in status["problems"])+"\n")
        publication = publish_summary(output, plan["track"])
        print(json.dumps({"status": status["status"], "publication": publication}), flush=True)
    return 1 if status["status"] == "INCOMPLETE" else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.plan, args.plan_sha256))
