#!/usr/bin/env python3
"""Resume native sensor controls with bounded node-local output transport.

The scientific archive, protocol, smoke and all twenty shard identities stay
unchanged. Four verified legacy shards and one explicitly pinned packaging-only
recovery are adopted without reruns. Remaining shards publish original raw output
from node-local scratch and retain compact receipts under the shared quota.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import re
import shutil
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import time

SHARED_ROOT = Path("/nfs/hpc/share/sanchej7/humanoid-research-20261011")
CAMPAIGN = SHARED_ROOT / "native-sensor-controls-v2"
HELPERS = ("methods_dispatch.py", "methods_archive.py", "h34_campaign.py", "methods_remote_campaign.py", "native_sensor_controls_transport.py")
GIB = 1024**3
MIB = 1024**2


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_helper(path):
    spec = importlib.util.spec_from_file_location("sensor_controls_" + path.stem, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def validate_plan(path, expected_sha):
    path = Path(path).resolve()
    if digest(path) != expected_sha or not path.is_relative_to(SHARED_ROOT):
        raise ValueError("plan must be pinned under the explicit shared campaign root")
    plan = read(path)
    if (plan.get("schema") != "bhl-native-sensor-controls-dispatch-v3" or
            Path(plan["campaign_dir"]).resolve() != CAMPAIGN or
            plan.get("max_concurrency") != 1 or
            plan.get("reserved_bytes_per_job") != 16*MIB or
            plan.get("minimum_free_bytes") != 8*1024**3 or
            plan.get("poll_seconds") != 30):
        raise ValueError("changed shared-storage scope, concurrency or reserve")
    if set(plan["source_sha256"]) != set((*HELPERS, Path(__file__).name, "worker.sbatch")):
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
    if [j["name"] for j in plan["jobs"]] != [j["name"] for j in full_jobs] or len(full_jobs) != 20:
        raise ValueError("all twenty declared native sensor-control shards required exactly once")
    for job, scientific in zip(plan["jobs"], full_jobs):
        args = scientific["args"]
        seed = int(args[args.index("--seed")+1])
        expected = dict(phase="development", expected_declared_cells=10)
        if job.get("expected_fields") != expected or job.get("seed") != seed:
            raise ValueError("declared shard expected identities differ")
    if [j["seed"] for j in plan["jobs"]] != [730000]*10+[730001]*10:
        raise ValueError("calibration seed must finish and fit before any evaluation seed submission")
    if plan["smoke"].get("expected_fields") != dict(phase="smoke", episodes=14, execution_gate=True):
        raise ValueError("the complete same-source14-episode execution smoke is mandatory")
    if plan.get("transport") != dict(raw_cap_bytes=8*GIB, shared_receipt_cap_bytes=16*MIB, part_bytes=GIB):
        raise ValueError("changed node-local transport bounds")
    validate_recovery(plan)
    return plan


def validate_recovery(plan):
    spec = plan["recovery"]
    if spec.get("job_name") != "native-controls-v2-s730000-c04-20261011" or str(spec.get("job_id")) != "21776269":
        raise ValueError("only the explicit packaging failure can be recovered")
    for field in ("receipt", "result", "inventory"):
        item = Path(spec[field])
        if (item.is_symlink() or not item.resolve().is_relative_to(SHARED_ROOT) or
                digest(item) != spec[field + "_sha256"]):
            raise ValueError("pinned recovery evidence changed")
    receipt, result, inventory = (read(spec[k]) for k in ("receipt", "result", "inventory"))
    directory = CAMPAIGN / (spec["job_name"] + "-" + str(spec["job_id"]))
    completion, launch = read(directory / "completion.json"), read(directory / "launch.json")
    if (receipt.get("schema") != "bhl-native-controls-packaging-recovery-v1" or
            receipt.get("status") != "COMPLETE" or receipt.get("scientific_rerun") is not False or
            str(receipt.get("original_job_id")) != str(spec["job_id"]) or
            receipt.get("source_archive_sha256") != plan["archive_sha256"] or
            receipt.get("protocol_sha256") != plan["protocol_sha256"] or
            receipt.get("original_completion_sha256") != digest(directory / "completion.json") or
            receipt.get("result_sha256") != spec["result_sha256"] or
            receipt.get("inventory_sha256") != spec["inventory_sha256"] or
            completion.get("status") != "INCOMPLETE" or completion.get("exit_status") != 0 or
            completion.get("error") != f"output exceeds predeclared archive cap ({completion.get('output_bytes')} bytes)" or
            launch.get("archive_sha256") != plan["archive_sha256"] or
            completion.get("archive_sha256") != plan["archive_sha256"] or
            receipt.get("original_runtime_exit_status") != 0 or
            result.get("status") != "PASS" or result.get("phase") != "development" or
            result.get("expected_declared_cells") != 10 or result.get("episodes") != 10):
        raise ValueError("recovery does not prove packaging-only failure with unchanged science")
    for name, value in completion["files"].items():
        item = directory / name
        if Path(name).name != name or item.is_symlink() or digest(item) != value["sha256"] or item.stat().st_size != value["bytes"]:
            raise ValueError("original failed evidence changed")
    names = [name for part in receipt["parts"] for name in part["files"]]
    if not 2*GIB < completion["output_bytes"] <= 8*GIB:
        raise ValueError("recovery raw size does not match the declared packaging failure")
    if (len(names) != len(set(names)) or set(names) != set(inventory) or
            sum(i["bytes"] for i in inventory.values()) != completion["output_bytes"] or
            receipt["raw_output_bytes"] != completion["output_bytes"] or receipt["raw_output_bytes"] > 8*GIB):
        raise ValueError("recovery inventory is incomplete or duplicated")
    return receipt, result


def recovered_row(plan, job, scheduler, exit_code):
    spec = plan["recovery"]
    if (job["name"] != spec["job_name"] or scheduler != "FAILED" or exit_code != "1:0"):
        raise ValueError("no recovery or implicit retry for this failed job")
    receipt, result = validate_recovery(plan)
    return dict(state="COMPLETE", scientific_status=result["status"], packaging_recovered=True,
                original_scheduler_state=scheduler, recovery_receipt_sha256=spec["receipt_sha256"],
                publication=dict(schema="bhl-native-recovered-tar-volumes-v1", parts=receipt["parts"],
                    inventory=spec["inventory"], inventory_sha256=spec["inventory_sha256"],
                    raw_output_bytes=receipt["raw_output_bytes"]))


def validate_transport_result(directory, plan, result):
    receipt = read(directory / "remote-result.json")
    if (receipt.get("schema") != "bhl-native-transport-result-v1" or receipt.get("status") != result["status"] or
            receipt.get("runtime_exit_status") != 0 or receipt.get("runner_returncode") != 0 or
            receipt.get("source_archive_sha256") != plan["archive_sha256"] or
            receipt.get("protocol_sha256") != plan["protocol_sha256"] or
            receipt.get("original_max_output_mb") != 2048 or receipt.get("operational_max_output_mb") != 8192 or
            receipt.get("publication", {}).get("schema") != "bhl-native-raw-publication-v1"):
        raise ValueError("node-local transport completion differs from scientific result")
    completion = read(directory / "completion.json")
    expected = completion["files"]["outputs.tar.gz"]
    publication = receipt["publication"]
    if any(publication[k] != expected[k] for k in ("sha256", "bytes")):
        raise ValueError("publication differs from original completed output")
    return publication


def materialize(row, work, destination, transport, remote):
    publication = row["publication"]
    work, destination = Path(work), Path(destination)
    if publication.get("schema") != "bhl-native-recovered-tar-volumes-v1":
        archive = transport.download(publication, work, remote)
        transport.extract_output(archive, destination, raw_cap_bytes=8*GIB)
        archive.unlink()
        return destination
    inventory_path = Path(publication["inventory"])
    if digest(inventory_path) != publication["inventory_sha256"]:
        raise ValueError("recovery output inventory changed")
    inventory = read(inventory_path)
    destination.mkdir(parents=True, exist_ok=False)
    total = 0
    for i, part in enumerate(publication["parts"]):
        archive = transport.download(part, work / f"part-{i}", remote)
        temporary = work / f"volume-{i}"
        total += transport.extract_output(archive, temporary, raw_cap_bytes=8*GIB)
        if total > 8*GIB:
            raise ValueError("combined recovery volumes exceed raw cap")
        actual = {str(p.relative_to(temporary)) for p in temporary.rglob("*") if p.is_file()}
        if actual != set(part["files"]):
            raise ValueError("recovered volume differs from declared inventory")
        for relative in sorted(actual):
            source, target = temporary / relative, destination / relative
            expected = inventory[relative]
            if target.exists() or digest(source) != expected["sha256"] or source.stat().st_size != expected["bytes"]:
                raise ValueError("recovered file duplicated or changed")
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
        archive.unlink()
        shutil.rmtree(temporary)
    if total != publication["raw_output_bytes"]:
        raise ValueError("combined recovered output size differs")
    return destination


def submit_transport(path, plan, job, dispatcher, launcher):
    """Submit only a never-submitted frozen job after rechecking its smoke."""
    name = job["name"]
    if dispatcher.submitted_id(CAMPAIGN, job, plan["archive_sha256"]):
        raise ValueError("job already submitted; retries forbidden")
    smoke_id = dispatcher.submitted_id(CAMPAIGN, plan["smoke"], plan["archive_sha256"])
    smoke_dir = CAMPAIGN / (plan["smoke"]["name"] + "-" + smoke_id)
    result, completion = dispatcher.checked_result(smoke_dir, plan["smoke"], plan["archive_sha256"])
    if dispatcher.scheduler_state(smoke_id) != ("COMPLETED", "0:0") or result["status"] != "PASS":
        raise ValueError("same-source completed smoke required before submission")
    _, intake, protocol = launcher.load_intake(CAMPAIGN)
    if intake["archive_sha256"] != plan["archive_sha256"] or intake["protocol_sha256"] != plan["protocol_sha256"]:
        raise ValueError("frozen scientific input changed before submission")
    scientific = launcher.get_job(protocol, name)
    if scientific["kind"] != "run" or scientific["resources"].get("array"):
        raise ValueError("only predeclared individual scientific jobs allowed")
    r = scientific["resources"]
    command = ["/apps/slurm/25.11.8/bin/sbatch", "--parsable", "--ntasks=1", "--account=eecs",
        "--job-name="+name, "--partition="+r["partition"], "--cpus-per-task="+str(r["cpus"]),
        "--mem="+str(r["memory_gb"])+"G", "--time="+r["time_limit"],
        "--output="+str(CAMPAIGN/(name+"-%j.out")), "--error="+str(CAMPAIGN/(name+"-%j.out")), "--export=NONE"]
    if r.get("requeue") is False:
        command.append("--no-requeue")
    for key in ("constraint", "exclude"):
        if r.get(key):
            command.append("--"+key+"="+r[key])
    if r.get("gpus", 0):
        command.append("--gres=gpu:1")
    command.extend([str(path.parent/"worker.sbatch"), str(path), digest(path), name])
    gate = dict(smoke_job_id=smoke_id, smoke_completion_sha256=digest(smoke_dir/"completion.json"),
        smoke_result_sha256=completion["files"]["campaign_result.json"]["sha256"],
        scheduler_state="COMPLETED", scheduler_exit_code="0:0", verified_utc=dispatcher.utcnow(),
        policy="same-source complete 14-episode smoke; unchanged scientific cohort; operational transport only")
    launcher.write_json(CAMPAIGN/(name+"-submission-start.json"), dict(command=command,qualification_gate=gate,
        operational_plan_sha256=digest(path), created_utc=dispatcher.utcnow()))
    environment = {k:v for k,v in os.environ.items() if not k.startswith(("SLURM_", "APPTAINERENV_"))}
    environment["TMPDIR"] = "/tmp"
    submitted = subprocess.run(command, env=environment, capture_output=True, text=True)
    match = re.fullmatch(r"(\d+)(?:;[^\n]+)?\n?", submitted.stdout)
    receipt = dict(submitted_utc=dispatcher.utcnow(), job_id=match[1] if match and submitted.returncode == 0 else None,
        command=command, returncode=submitted.returncode, stdout=submitted.stdout, stderr=submitted.stderr,
        resources=r, archive_sha256=plan["archive_sha256"], qualification_gate=gate,
        operational_plan_sha256=digest(path), status="SUBMITTED" if match and submitted.returncode == 0 else "SUBMISSION_FAILED")
    launcher.write_json(CAMPAIGN/(name+"-submission.json"), receipt)
    return receipt


def health_stage(path, state, plan, launcher, archiver, remote, transport, *, mode):
    seed = 730000 if mode == "fit" else 730001
    wanted = {j["name"] for j in plan["jobs"] if j["seed"] == seed}
    rows_by_job = [r for r in state["jobs"] if r["name"] in wanted and r.get("state") == "COMPLETE"]
    if len(rows_by_job) != 10:
        raise ValueError("exact complete calibration/evaluation seed required")
    with tempfile.TemporaryDirectory(prefix="bhl-native-controls-health-", dir="/tmp") as temporary:
        work = Path(temporary)
        frozen = work / "frozen"
        launcher.safe_extract(CAMPAIGN / "frozen-campaign.tar.gz", frozen)
        launcher.verify_source(frozen)
        rows = []
        for row in rows_by_job:
            target = work / "raw" / row["name"]
            materialize(row, work / "downloads" / row["name"], target, transport, remote)
            result = read(target / "campaign_result.json")
            for relative in result["episode_files"]:
                episode = target / relative
                if digest(episode) != result["episode_sha256"][relative]:
                    raise ValueError("health episode differs from scientific result")
                values = read(episode)
                if values["seed"] != seed:
                    raise ValueError("health split contamination")
                selected = {key: values[key] for key in ("method", "case", "condition", "seed", "success", "full_horizon", "fell", "collision", "nonfinite", "tracking_fraction_unique_estimator_updates")}
                selected["episode_sha256"] = digest(episode)
                rows.append(selected)
            shutil.rmtree(target)
        episodes = path.parent / ("health-calibration-episodes.json" if mode == "fit" else "health-evaluation-episodes.json")
        with episodes.open("x") as stream:
            json.dump(rows, stream, indent=2, allow_nan=False)
            stream.write("\n")
        output = path.parent / ("health-model.json" if mode == "fit" else "health-evaluation.json")
        if rows:
            command = [read(CAMPAIGN / "protocol.json")["runtime"]["python"], str(frozen / "source/scripts/bench/outcome_health.py"),
                       mode, "--episodes", str(episodes), "--output", str(output)]
            if mode == "evaluate":
                command += ["--model", str(path.parent / "health-model.json")]
            with (path.parent / ("health-" + mode + ".log")).open("wb") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONPATH": str(frozen / "source/src"), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        else:
            output.write_text(json.dumps({"schema": "bhl-outcome-health-unavailable-v1", "status": "UNAVAILABLE_ALL_ARMS_GATED", "mode": mode})+"\n")
        compact = work / ("native-controls-health-" + mode + ".tar.gz")
        with tarfile.open(compact, "w:gz") as archive:
            for item in (episodes, output):
                archive.add(item, arcname=item.name)
        spec = dict(sha256=digest(compact), bytes=compact.stat().st_size)
        publication = remote.publish_asset(compact, "native-controls-health-" + mode + "-20261011-" + spec["sha256"][:12] + ".tar.gz", spec)
        dest = path.parent / ("health-model-publication.json" if mode == "fit" else "health-evaluation-publication.json")
        with dest.open("x") as stream:
            json.dump(publication, stream, indent=2)
            stream.write("\n")


def run(path, expected_sha):
    path = Path(path).resolve()
    plan = validate_plan(path, expected_sha)
    dispatcher, archiver, launcher, remote, transport = [load_helper(path.parent/name) for name in HELPERS]
    dispatcher.AUTHORIZED_ROOT = archiver.AUTHORIZED_ROOT = SHARED_ROOT
    archiver.TAG = remote.TAG = "research-campaign-evidence-20261011"
    state_path = path.parent / "state.json"
    state = read(state_path) if state_path.exists() else dict(
        schema="bhl-native-sensor-controls-dispatch-state-v3", plan_sha256=expected_sha,
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
            raise ValueError("actual completed same-source14-episode smoke required")
        qualified_smoke = path.parent / "qualified-smoke.json"
        if qualified_smoke.exists():
            if digest(qualified_smoke) != digest(smoke_dir / "campaign_result.json"):
                raise ValueError("qualified smoke copy changed")
        else:
            dispatcher.write_atomic(qualified_smoke, smoke)
        smoke_publication = archiver.publish(smoke_dir, retire=True)
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
                    if (directory / "remote-result.json").exists():
                        publication = validate_transport_result(directory, plan, result)
                    else:
                        publication = archiver.publish(directory, retire=True)
                        if publication["local_packaging_retired"] is not True:
                            raise ValueError("raw output not verified remotely and retired")
                    dispatcher.release_output(CAMPAIGN, job["name"])
                    row.update(state="COMPLETE", scientific_status=result["status"], publication=publication)
                    completed += 1
                elif job["name"] == plan["recovery"]["job_name"]:
                    if job_id != str(plan["recovery"]["job_id"]):
                        raise ValueError("recovery job identity changed")
                    row.update(recovered_row(plan, job, scheduler, exit_code))
                    dispatcher.release_output(CAMPAIGN, job["name"])
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
            if completed >= 10 and not (path.parent / "health-model-publication.json").exists():
                health_stage(path, state, plan, launcher, archiver, remote, transport, mode="fit")
            if completed == 20:
                break
            for job in plan["jobs"]:
                if active >= plan["max_concurrency"]:
                    break
                if dispatcher.submitted_id(CAMPAIGN, job, plan["archive_sha256"]):
                    continue
                if job["seed"] == 730001 and not (path.parent / "health-model-publication.json").exists():
                    break
                reserved, details = dispatcher.reserve_output(CAMPAIGN, job["name"], plan["reserved_bytes_per_job"], plan["minimum_free_bytes"])
                if not reserved:
                    save("WAITING_FOR_SHARED_SPACE", storage=details)
                    break
                receipt = submit_transport(path, plan, job, dispatcher, launcher)
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
        if shutil.disk_usage("/tmp").free < 100*GIB:
            raise ValueError("full-cohort collection requires 100 GiB node-local headroom")
        save("COLLECTING")
        health_stage(path, state, plan, launcher, archiver, remote, transport, mode="evaluate")
        with tempfile.TemporaryDirectory(prefix="bhl-native-controls-collect-", dir="/tmp") as temporary:
            work = Path(temporary)
            frozen = work / "frozen"
            launcher.safe_extract(CAMPAIGN / "frozen-campaign.tar.gz", frozen)
            launcher.verify_source(frozen)
            for row in state["jobs"]:
                materialize(row, work / "downloads" / row["name"], work / "raw" / row["name"], transport, remote)
            output = path.parent / "scientific-summary.json"
            command = [read(CAMPAIGN / "protocol.json")["runtime"]["python"],
                str(frozen / "source/scripts/bench/native_sensor_controls_campaign.py"), "--collect", str(work / "raw"),
                "--qualified-smoke", str(path.parent / "qualified-smoke.json"), "--output", str(output)]
            environment = {**os.environ, "PYTHONPATH": str(frozen / "source/src"), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
            with (path.parent / "collection.log").open("wb") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=environment)
            result = read(output)
            if result["declared_cells"] != 200 or result["status"] != "PASS":
                raise ValueError("strict full-cohort collection failed")
            compact = work / "native-controls-final.tar.gz"
            with tarfile.open(compact, "w:gz") as archive:
                for item in (output, path.parent / "collection.log", path.parent / "health-model.json",
                             path.parent / "health-evaluation.json", CAMPAIGN / "protocol.json", CAMPAIGN / "intake.json"):
                    archive.add(item, arcname=item.name)
            spec = dict(sha256=digest(compact), bytes=compact.stat().st_size)
            publication = remote.publish_asset(compact, "native-controls-final-20261011-" + spec["sha256"][:12] + ".tar.gz", spec)
            dispatcher.write_atomic(path.parent / "publication.json", publication)
            save("COMPLETE", measured_cells=result["measured_cells"], unrun_cells=result["unrun_after_nominal_gate"], final_publication=publication)
    except Exception as error:
        save("FAILED_NO_RETRY", error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    run(args.plan, args.plan_sha256)
