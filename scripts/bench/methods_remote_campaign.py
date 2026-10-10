#!/usr/bin/env python3
"""Run one pinned H34 job from release storage, retaining compact home receipts.

No submissions, retries, parameter edits, or automatic follow-on jobs. GitHub
operations run on the host; the original H34 runner owns scientific isolation.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
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

REPOSITORY = "joses2017smjh/bhl-robustness-ladder"
TAG = "methods-campaign-evidence-20261010"
AUTHORIZED_ROOT = Path("/nfs/stak/users/sanchej7/humanoid-methods-20261010")
METADATA = ("intake.json", "protocol.json", "h34_campaign.sbatch")
COMPACT_CAP = 1024**2


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > COMPACT_CAP:
        raise ValueError("linked or oversized JSON receipt")
    return json.loads(path.read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def scoped(path):
    original = Path(path)
    path = original.resolve()
    if original.is_symlink() or path == AUTHORIZED_ROOT or not path.is_relative_to(AUTHORIZED_ROOT):
        raise ValueError("durable output must be in this new methods campaign")
    return path


def gh(*args):
    result = subprocess.run(["gh", *map(str, args)], capture_output=True,
                            env={**os.environ, "TMPDIR": "/tmp"})
    if result.returncode:
        raise RuntimeError(f"GitHub operation failed with exit {result.returncode}")
    return result.stdout


def checked_file(path, spec):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
        raise ValueError("pinned file changed: "+path.name)


def publish_asset(path, name, expected):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.tar\.gz", name):
        raise ValueError("invalid release asset name")
    checked_file(path, expected)
    release = json.loads(gh("api", f"repos/{REPOSITORY}/releases/tags/{TAG}"))
    assets = [row for row in release["assets"] if row["name"] == name]
    with tempfile.TemporaryDirectory(prefix="bhl-remote-publication-", dir="/tmp") as directory:
        work = Path(directory)
        if not assets:
            staged = work/name
            shutil.copyfile(path, staged)
            checked_file(staged, expected)
            gh("release", "upload", TAG, staged, "--repo", REPOSITORY)
            staged.unlink()
            release = json.loads(gh("api", f"repos/{REPOSITORY}/releases/tags/{TAG}"))
            assets = [row for row in release["assets"] if row["name"] == name]
        if (len(assets) != 1 or assets[0]["size"] != expected["bytes"] or
                (assets[0].get("digest") and assets[0]["digest"] != "sha256:"+expected["sha256"])):
            raise ValueError("existing release asset differs; replacement refused")
        gh("release", "download", TAG, "--repo", REPOSITORY, "--pattern", name, "--dir", work)
        checked_file(work/name, expected)
    return dict(asset_name=name, **expected, url=assets[0]["browser_download_url"],
                verified_utc=datetime.now(timezone.utc).isoformat(),
                verification="pinned bytes + release identity + fresh independent download SHA256")


def source_identity(archive):
    with tarfile.open(archive, "r:gz") as stream:
        member = stream.getmember("manifest.json")
        if not member.isfile() or member.size > 8*1024**2:
            raise ValueError("invalid source inventory")
        inventory = json.load(stream.extractfile(member))
    # Receipts/docs do not alter scientific identity; all Python source and every
    # input byte must match across a separately declared smoke and scored plan.
    return {name: spec for name, spec in inventory.items()
            if name.startswith("inputs/") or (name.startswith("source/") and name.endswith((".py", ".cc", ".cpp", ".h", ".hpp")))}


def scientific_identity(archive, intake, protocol):
    return {"files": source_identity(archive), "runtime": protocol.get("runtime"),
            "sif_sha256": intake.get("sif_sha256")}


def identity_sha256(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def gate_receipt(directory, identity):
    directory = Path(directory)
    completed = read(directory/"completion.json")
    launch = read(directory/"launch.json")
    result = read(directory/"campaign_result.json")
    remote = read(directory/"remote-result.json")
    for name in ("launch.json", "campaign_result.json"):
        checked_file(directory/name, completed["files"][name])
    if (launch["job"]["kind"] != "smoke" or result.get("status") != "PASS" or
            completed.get("status") != "PASS" or completed.get("exit_status") != 0 or completed.get("error") or
            not completed.get("finished_utc") or remote.get("runner_returncode") != 0 or
            remote.get("runtime_exit_status") != 0 or not remote.get("publication") or
            remote.get("status") != "PASS" or remote.get("scientific_identity_sha256") != identity_sha256(identity) or
            remote.get("source_archive_sha256") != completed["archive_sha256"] or
            launch["archive_sha256"] != completed["archive_sha256"]):
        raise ValueError("matching scientific smoke readiness PASS is required; full trials remain unrun")
    return {name: {"sha256": digest(directory/name), "bytes": (directory/name).stat().st_size}
            for name in ("completion.json", "launch.json", "campaign_result.json", "remote-result.json")}


def prepare(campaign, durable, job_name, *, qualified_smoke=None):
    campaign, durable = Path(campaign).resolve(), scoped(durable)
    if durable.exists():
        raise ValueError("refusing an existing remote execution directory")
    intake, protocol = read(campaign/"intake.json"), read(campaign/"protocol.json")
    archive = campaign/"frozen-campaign.tar.gz"
    spec = {"sha256": intake["archive_sha256"], "bytes": intake["archive_bytes"]}
    checked_file(archive, spec)
    for name, key in (("protocol.json", "protocol_sha256"), ("h34_campaign.sbatch", "launcher_sha256")):
        if digest(campaign/name) != intake[key]:
            raise ValueError("original H34 metadata changed")
    jobs = [row for row in protocol["jobs"] if row["name"] == job_name]
    if len(jobs) != 1 or jobs[0]["resources"].get("array"):
        raise ValueError("exactly one declared non-array job required")
    identity = scientific_identity(archive, intake, protocol)
    gate = None
    if jobs[0]["kind"] != "smoke":
        if qualified_smoke is None:
            raise ValueError("development job requires explicit same-science qualified smoke")
        gate = {"directory": str(Path(qualified_smoke).resolve()), "files": gate_receipt(qualified_smoke, identity)}
    metadata = {name: {"sha256": digest(campaign/name), "bytes": (campaign/name).stat().st_size} for name in METADATA}
    if sum(row["bytes"] for row in metadata.values()) > COMPACT_CAP//2:
        raise ValueError("frozen metadata exceeds compact durable budget")
    publication = publish_asset(archive, "methods-source-"+spec["sha256"]+".tar.gz", spec)
    durable.mkdir(parents=True, exist_ok=False)
    for name in METADATA:
        shutil.copyfile(campaign/name, durable/name)
    bootstrap = durable/"methods_remote_campaign.py"
    shutil.copyfile(__file__, bootstrap)
    plan = dict(schema="bhl-methods-remote-job-v1", source=publication, metadata=metadata,
                job_name=job_name, job=jobs[0], bootstrap_sha256=digest(bootstrap),
                scientific_identity=identity, qualified_smoke=gate,
                scope="one frozen declared job; no submission, retry, or automatic follow-on")
    write(durable/"remote-plan.json", plan)
    write(durable/"source-publication.json", publication)
    return {"plan": str(durable/"remote-plan.json"), "plan_sha256": digest(durable/"remote-plan.json"),
            "bootstrap": str(bootstrap), "archive_sha256": spec["sha256"]}


def load_plan(path, expected):
    path = Path(path)
    root = scoped(path.parent)
    if digest(path) != expected:
        raise ValueError("remote execution plan changed")
    plan = read(path)
    if (plan.get("schema") != "bhl-methods-remote-job-v1" or
            digest(__file__) != plan["bootstrap_sha256"] or set(plan["metadata"]) != set(METADATA)):
        raise ValueError("remote bootstrap or metadata inventory changed")
    for name, spec in plan["metadata"].items():
        checked_file(root/name, spec)
    protocol = read(root/"protocol.json")
    if [row for row in protocol["jobs"] if row["name"] == plan["job_name"]] != [plan["job"]]:
        raise ValueError("declared job differs from frozen protocol")
    intake = read(root/"intake.json")
    if (plan["source"]["sha256"] != intake["archive_sha256"] or
            plan["source"]["bytes"] != intake["archive_bytes"] or
            not re.fullmatch(r"methods-source-[a-f0-9]{64}\.tar\.gz", plan["source"]["asset_name"])):
        raise ValueError("remote source identity differs")
    if plan["job"]["kind"] != "smoke":
        gate = plan["qualified_smoke"]
        if gate is None or gate_receipt(gate["directory"], plan["scientific_identity"]) != gate["files"]:
            raise ValueError("scientific smoke readiness evidence changed")
    return root, plan


def extract_runner(archive, destination):
    with tarfile.open(archive, "r:gz") as stream:
        matches = [m for m in stream.getmembers() if m.name == "source/scripts/bench/h34_campaign.py"]
        if len(matches) != 1 or not matches[0].isfile() or matches[0].size > COMPACT_CAP:
            raise ValueError("invalid pinned H34 runner member")
        with destination.open("xb") as output:
            shutil.copyfileobj(stream.extractfile(matches[0]), output)
    spec = importlib.util.spec_from_file_location("remote_pinned_h34", destination)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def retained_workspace():
    path = Path(tempfile.mkdtemp(prefix="bhl-remote-science-", dir="/tmp"))
    try:
        yield path
    except BaseException:
        print("Remote job interrupted or publication failed; original local evidence retained at "+str(path),
              file=sys.stderr, flush=True)
        raise
    else:
        shutil.rmtree(path)


def run(path, expected):
    root, plan = load_plan(path, expected)
    job_id = os.getenv("SLURM_JOB_ID", "")
    if not job_id.isdecimal():
        raise ValueError("remote scientific execution requires an allocated Slurm job")
    durable_job = root/(plan["job_name"]+"-"+job_id)
    durable_job.mkdir(exist_ok=False)  # also makes implicit retries impossible
    with retained_workspace() as local:
        write(durable_job/"remote-start.json", {"plan_sha256": expected, "job_id": job_id,
              "node_local_work": str(local), "started_utc": datetime.now(timezone.utc).isoformat()})
        source = plan["source"]
        gh("release", "download", TAG, "--repo", REPOSITORY, "--pattern", source["asset_name"], "--dir", local)
        downloaded = local/source["asset_name"]
        checked_file(downloaded, source)
        downloaded.rename(local/"frozen-campaign.tar.gz")
        for name in METADATA:
            shutil.copyfile(root/name, local/name)
        if scientific_identity(local/"frozen-campaign.tar.gz", read(local/"intake.json"),
                               read(local/"protocol.json")) != plan["scientific_identity"]:
            raise ValueError("remote scientific source inventory differs")
        helper = extract_runner(local/"frozen-campaign.tar.gz", local/"h34_campaign.py")
        rc = helper.execute(local, plan["job_name"], source["sha256"])
        job = local/durable_job.name
        completion, result = read(job/"completion.json"), read(job/"campaign_result.json")
        for name, spec in completion["files"].items():
            if Path(name).name != name:
                raise ValueError("unsafe completed evidence filename")
            checked_file(job/name, spec)
        publication = None
        raw = job/"outputs.tar.gz"
        if raw.exists():
            spec = completion["files"][raw.name]
            publication = publish_asset(raw, root.name+"--"+job.name+".tar.gz", spec)
            publication.update(schema="bhl-methods-archive-v1", source_archive_sha256=source["sha256"],
                job_id=job_id, scientific_status=result["status"], local_packaging_retired=True)
            write(durable_job/"publication.json", publication)
        # Preserve small original receipts byte-for-byte. The complete runtime
        # log is retained in a separately verified release bundle if oversized.
        compact = [job/name for name in ("completion.json", "launch.json", "campaign_result.json")]
        if sum(item.stat().st_size for item in compact) > COMPACT_CAP//2:
            raise ValueError("compact scientific receipts exceed durable budget")
        for item in compact:
            shutil.copyfile(item, durable_job/item.name)
        runtime = job/"runtime.log"
        if runtime.stat().st_size <= 128*1024:
            shutil.copyfile(runtime, durable_job/runtime.name)
        else:
            bundle = local/"execution-evidence.tar.gz"
            with tarfile.open(bundle, "w:gz") as stream:
                for item in [*compact, runtime]:
                    stream.add(item, arcname=item.name, recursive=False)
            evidence = publish_asset(bundle, root.name+"--"+job.name+"-execution.tar.gz",
                {"sha256": digest(bundle), "bytes": bundle.stat().st_size})
            write(durable_job/"execution-publication.json", evidence)
        negative_gate = (plan["job"]["kind"] == "smoke" and result.get("status") == "NEGATIVE" and
                         completion.get("exit_status") == 0 and not completion.get("error"))
        record = dict(schema="bhl-methods-remote-result-v1", plan_sha256=expected, job_id=job_id,
            source_archive_sha256=source["sha256"], scientific_identity_sha256=identity_sha256(plan["scientific_identity"]),
            status=result["status"], runner_returncode=rc, runtime_exit_status=completion.get("exit_status"),
            outcome=("FAILED_SCIENTIFIC_READINESS_GATE" if negative_gate else
                     "EXECUTION_INCOMPLETE" if result["status"] == "INCOMPLETE" else "SCIENTIFIC_EXECUTION_COMPLETED"),
            follow_on="UNRUN_AFTER_NEGATIVE_SMOKE" if negative_gate else "NO_AUTOMATIC_SUBMISSION",
            publication=publication)
        write(durable_job/"remote-result.json", record)
        print(json.dumps({key: value for key, value in record.items() if key != "scientific_identity"}), flush=True)
        return rc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--campaign-dir", required=True, type=Path)
    prep.add_argument("--durable-dir", required=True, type=Path)
    prep.add_argument("--job", required=True)
    prep.add_argument("--qualified-smoke-dir", type=Path)
    execute = sub.add_parser("run")
    execute.add_argument("--plan", required=True, type=Path)
    execute.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(args.campaign_dir, args.durable_dir, args.job,
                                 qualified_smoke=args.qualified_smoke_dir), indent=2))
    else:
        raise SystemExit(run(args.plan, args.plan_sha256))
