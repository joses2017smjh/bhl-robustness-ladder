#!/usr/bin/env python3
"""Bounded, shared-storage confirmation dispatch and strict final collection.

Only the explicitly authorized October11 shared root is writable. Scientific
source stays in the independently frozen H34 archive. Existing generic helpers
are imported without modifying their files; their authorization root is narrowed
to this explicit shared campaign before any helper mutation is called.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import time

SHARED_ROOT = Path("/nfs/hpc/share/sanchej7/humanoid-research-20261011")
CAMPAIGN = SHARED_ROOT / "terrain-confirmation-v3"
HELPERS = ("methods_dispatch.py", "methods_archive.py", "h34_campaign.py", "methods_remote_campaign.py")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_helper(path):
    spec = importlib.util.spec_from_file_location("confirmation_" + path.stem, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def validate_plan(path, expected_sha):
    path = Path(path).resolve()
    if digest(path) != expected_sha or not path.is_relative_to(SHARED_ROOT):
        raise ValueError("plan must be pinned under the explicit shared campaign root")
    plan = read(path)
    if (plan.get("schema") != "bhl-terrain-confirmation-dispatch-v1" or
            Path(plan["campaign_dir"]).resolve() != CAMPAIGN or
            plan.get("max_concurrency") not in (1, 2, 3) or
            plan.get("reserved_bytes_per_job") != 1024**3 or
            plan.get("minimum_free_bytes") != 8*1024**3 or
            plan.get("poll_seconds") != 30):
        raise ValueError("changed shared-storage scope, concurrency or reserve")
    if set(plan["source_sha256"]) != set((*HELPERS, Path(__file__).name)):
        raise ValueError("incomplete orchestration source inventory")
    for name, sha in plan["source_sha256"].items():
        if Path(name).name != name or digest(path.parent / name) != sha:
            raise ValueError("frozen dispatcher source changed")
    if Path(__file__).resolve() != path.parent / Path(__file__).name:
        raise ValueError("execute only the durable frozen dispatcher copy")
    intake, protocol = read(CAMPAIGN / "intake.json"), read(CAMPAIGN / "protocol.json")
    for name, key in (("frozen-campaign.tar.gz", "archive_sha256"), ("protocol.json", "protocol_sha256")):
        if digest(CAMPAIGN / name) != intake[key] or intake[key] != plan[key]:
            raise ValueError("scientific source changed since orchestration freeze")
    full_jobs = [j for j in protocol["jobs"] if j["kind"] == "run"]
    if [j["name"] for j in plan["jobs"]] != [j["name"] for j in full_jobs] or len(full_jobs) != 9:
        raise ValueError("all nine declared confirmation shards required exactly once")
    for job, scientific in zip(plan["jobs"], full_jobs):
        args = scientific["args"]
        actor, terrain = args[args.index("--actor")+1], args[args.index("--terrain")+1]
        expected = dict(phase="confirmation", episodes=40 if actor == "dr-default-s1" else 60,
                        actor_filter=actor, terrain_filter=terrain)
        if job.get("expected_fields") != expected:
            raise ValueError("declared shard expected identities differ")
    if plan["smoke"].get("expected_fields") != dict(phase="smoke", episodes=24, gate_to_confirmation=True):
        raise ValueError("the full same-source smoke gate is mandatory")
    return plan


def run(path, expected_sha):
    path = Path(path).resolve()
    plan = validate_plan(path, expected_sha)
    dispatcher, archiver, launcher, remote = [load_helper(path.parent/name) for name in HELPERS]
    dispatcher.AUTHORIZED_ROOT = archiver.AUTHORIZED_ROOT = SHARED_ROOT
    archiver.TAG = remote.TAG = "research-campaign-evidence-20261011"
    state_path = path.parent / "state.json"
    state = read(state_path) if state_path.exists() else dict(
        schema="bhl-terrain-confirmation-dispatch-state-v1", plan_sha256=expected_sha,
        dispatcher_job_id=os.getenv("SLURM_JOB_ID"), jobs=[], started_utc=datetime.now(timezone.utc).isoformat())
    if state["plan_sha256"] != expected_sha:
        raise ValueError("no implicit resume across changed plans")
    def save(status, **values):
        state.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **values)
        dispatcher.write_atomic(state_path, state)
    def interrupted(signum, _):
        save("INTERRUPTED", reason="submitted scientific jobs left intact")
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        smoke_id = dispatcher.submitted_id(CAMPAIGN, plan["smoke"], plan["archive_sha256"])
        smoke_dir = CAMPAIGN / (plan["smoke"]["name"] + "-" + smoke_id)
        smoke, _ = dispatcher.checked_result(smoke_dir, plan["smoke"], plan["archive_sha256"])
        if dispatcher.scheduler_state(smoke_id) != ("COMPLETED", "0:0") or smoke["status"] != "PASS":
            raise ValueError("actual completed same-source24-episode smoke required")
        archiver.publish(smoke_dir, retire=True)
        deadline = time.monotonic() + 22*3600
        while True:
            validate_plan(path, expected_sha)
            records, active, completed = [], 0, 0
            for job in plan["jobs"]:
                job_id = dispatcher.submitted_id(CAMPAIGN, job, plan["archive_sha256"])
                if not job_id:
                    records.append(dict(name=job["name"], state="UNSUBMITTED"))
                    continue
                scheduler, exit_code = dispatcher.scheduler_state(job_id)
                row = dict(name=job["name"], job_id=job_id, scheduler_state=scheduler, scheduler_exit_code=exit_code)
                if scheduler == "COMPLETED" and exit_code == "0:0":
                    directory = CAMPAIGN / (job["name"] + "-" + job_id)
                    result, _ = dispatcher.checked_result(directory, job, plan["archive_sha256"])
                    publication = archiver.publish(directory, retire=True)
                    if publication["local_packaging_retired"] is not True:
                        raise ValueError("raw output not verified remotely and retired")
                    dispatcher.release_output(CAMPAIGN, job["name"])
                    row.update(state="COMPLETE", scientific_status=result["status"], publication=publication)
                    completed += 1
                elif scheduler in dispatcher.TERMINAL:
                    row["state"] = "FAILED_NO_RETRY"
                    records.append(row)
                    state["jobs"] = records
                    raise ValueError("scientific shard execution failed; no retries or new submissions: " + job["name"])
                else:
                    active += 1
                    row["state"] = "ACTIVE"
                records.append(row)
            state["jobs"] = records
            save("RUNNING", complete_jobs=completed, active_jobs=active)
            if completed == 9:
                break
            for job in plan["jobs"]:
                if active >= plan["max_concurrency"]:
                    break
                if dispatcher.submitted_id(CAMPAIGN, job, plan["archive_sha256"]):
                    continue
                reserved, details = dispatcher.reserve_output(CAMPAIGN, job["name"], plan["reserved_bytes_per_job"], plan["minimum_free_bytes"])
                if not reserved:
                    save("WAITING_FOR_SHARED_SPACE", storage=details)
                    break
                receipt = dispatcher.submit_qualified_job(CAMPAIGN, job["name"], plan["smoke"], plan["archive_sha256"], launcher)
                if receipt["status"] != "SUBMITTED":
                    dispatcher.release_output(CAMPAIGN, job["name"])
                    raise ValueError("scheduler submission rejected; no implicit retry")
                dispatcher.reserve_output(CAMPAIGN, job["name"], plan["reserved_bytes_per_job"], plan["minimum_free_bytes"],
                                          job_id=receipt["job_id"], existing=True)
                active += 1
                save("RUNNING", active_jobs=active, last_submitted=receipt["job_id"])
            if time.monotonic() >= deadline:
                raise TimeoutError("bounded dispatcher deadline reached; active jobs retained")
            time.sleep(30)
        save("COLLECTING")
        with tempfile.TemporaryDirectory(prefix="bhl-confirmation-collect-", dir="/tmp") as temporary:
            work = Path(temporary)
            frozen = work / "frozen"
            launcher.safe_extract(CAMPAIGN / "frozen-campaign.tar.gz", frozen)
            launcher.verify_source(frozen)
            raw_roots = []
            for row in state["jobs"]:
                publication = row["publication"]
                archive = Path(publication["local_verified_mirror"])
                if not archive.exists():
                    archiver.gh("release", "download", archiver.TAG, "--repo", archiver.REPOSITORY,
                                "--pattern", publication["asset_name"], "--dir", work)
                    archive = work / publication["asset_name"]
                if archive.stat().st_size != publication["bytes"] or digest(archive) != publication["sha256"]:
                    raise ValueError("raw collection archive differs from verified publication")
                destination = work / "raw" / row["name"]
                launcher.safe_extract(archive, destination)
                raw_roots.append(str(destination))
            output = path.parent / "scientific-summary.json"
            command = [read(CAMPAIGN / "protocol.json")["runtime"]["python"],
                str(frozen / "source/scripts/bench/terrain_confirmation_campaign.py"), "--protocol",
                str(frozen / "protocol.json"), "--collect", *raw_roots, "--output", str(output)]
            environment = {**os.environ, "PYTHONPATH": str(frozen / "source/src"), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
            with (path.parent / "collection.log").open("wb") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=environment)
            result = read(output)
            if result["episodes"] != 480 or result["status"] != "PASS":
                raise ValueError("strict full-cohort collection failed")
            collection = dict(schema="bhl-terrain-confirmation-collection-v1", status="PASS", episodes=480,
                protocol_sha256=plan["protocol_sha256"], scientific_summary_sha256=digest(output),
                jobs=state["jobs"], finished_utc=datetime.now(timezone.utc).isoformat())
            dispatcher.write_atomic(path.parent / "collection.json", collection)
            compact = work / "confirmation-final.tar.gz"
            with tarfile.open(compact, "w:gz") as archive:
                for item in (output, path.parent / "collection.json", path.parent / "collection.log", CAMPAIGN / "protocol.json", CAMPAIGN / "intake.json"):
                    archive.add(item, arcname=item.name)
            spec = dict(sha256=digest(compact), bytes=compact.stat().st_size)
            publication = remote.publish_asset(compact, "terrain-confirmation-final-20261011-" + spec["sha256"][:12] + ".tar.gz", spec)
            dispatcher.write_atomic(path.parent / "publication.json", publication)
            save("COMPLETE", confirmed_combinations=result["confirmed_combinations"], final_publication=publication)
    except Exception as error:
        save("FAILED_NO_RETRY", error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    run(args.plan, args.plan_sha256)
