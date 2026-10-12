#!/usr/bin/env python3
"""Publish immutable October 11 campaign reports after strict receipt checks.

This observer performs no scientific runs, retries, parameter changes or hardware
access. It checks compact identities and the already independently collected
summary, then independently downloads its final release artifact. It does not
repeat raw-trace rescoring. Only new task-scoped files are pushed from /tmp.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import time

SHARED_ROOT = Path("/nfs/hpc/share/sanchej7/humanoid-research-20261011")
REPOSITORY = "joses2017smjh/bhl-robustness-ladder"
TAG = "research-campaign-evidence-20261011"
JSON_CAP = 8 * 1024**2
ARCHIVE_CAP = 32 * 1024**2
ARMS = {"stereo", "stereo_inertial", "lidar_inertial", "permanent_stop", "verified_lio_recovery"}
CONDITIONS = {"nominal", "camera_2_5hz", "camera_10hz", "dark_exposure_gain", "causal_40ms_exposure",
              "visual_blackout", "lidar_quarter_returns", "lidar_extrinsic_error", "imu_clock_plus80ms", "gyro_bias"}
FAILURES = {"FAILED_NO_RETRY", "FAILED", "CANCELLED", "TIMEOUT", "INCOMPLETE", "OUT_OF_MEMORY"}


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bounded_bytes(path, cap=JSON_CAP):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > cap:
        raise ValueError("missing, linked or oversized receipt: " + str(path))
    data = path.read_bytes()
    if len(data) > cap:
        raise ValueError("receipt grew beyond its size limit")
    return data


def read(path):
    return json.loads(bounded_bytes(path))


def check_hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid SHA256")
    return value


def scoped(path):
    path = Path(path)
    resolved = path.resolve()
    if path.is_symlink() or resolved == SHARED_ROOT or not resolved.is_relative_to(SHARED_ROOT):
        raise ValueError("path outside the October 11 shared research root")
    return resolved


def snapshot(path, evidence, label):
    data = bounded_bytes(path)
    if label in evidence and evidence[label] != data:
        raise ValueError("snapshot identity collision")
    evidence[label] = data
    return json.loads(data)


def checked_publication(value, *, final=False):
    name = value.get("asset_name", "")
    if (not re.fullmatch(r"[A-Za-z0-9_.-]+\.tar\.gz", name)
            or type(value.get("bytes")) is not int or value["bytes"] <= 0
            or not value.get("verified_utc") or not value.get("verification")):
        raise ValueError("unverified or malformed publication")
    check_hash(value.get("sha256"))
    expected = f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{name}"
    if value.get("url") != expected or (final and value["bytes"] > ARCHIVE_CAP):
        raise ValueError("publication URL or compact size outside scope")
    return value


def validate_plan(path, expected_sha):
    path = scoped(path)
    if digest(path) != check_hash(expected_sha):
        raise ValueError("observer plan changed")
    value = read(path)
    if value.get("schema") != "bhl-research-completion-observer-v1":
        raise ValueError("unknown observer plan")
    if (set(value.get("controllers", {})) != {"native", "terrain"}
            or value.get("poll_seconds") != 30 or not 1 <= value.get("timeout_hours", 0) <= 48):
        raise ValueError("both controllers and bounded polling are required")
    if digest(__file__) != check_hash(value["reporter_sha256"]):
        raise ValueError("observer source changed")
    scoped(value["output_dir"])
    for spec in value["controllers"].values():
        scoped(spec["directory"])
        check_hash(spec["plan_sha256"])
    gh = Path(value["gh_path"])
    if not gh.is_absolute() or gh.is_symlink() or digest(gh) != check_hash(value["gh_sha256"]):
        raise ValueError("pinned GitHub executable required")
    return value


def command(argv, *, environment=None, cwd=None, check=True):
    result = subprocess.run(list(map(str, argv)), cwd=cwd, env=environment, capture_output=True, timeout=180)
    if check and result.returncode:
        # Tool output can contain credential diagnostics; retain no authentication bytes.
        raise RuntimeError(f"{Path(str(argv[0])).name} operation failed with exit {result.returncode}")
    return result


def download_final(publication, work, gh, environment):
    checked_publication(publication, final=True)
    command([gh, "release", "download", TAG, "--repo", REPOSITORY, "--pattern", publication["asset_name"],
             "--dir", work], environment=environment)
    archive = work / publication["asset_name"]
    if archive.stat().st_size != publication["bytes"] or digest(archive) != publication["sha256"]:
        raise ValueError("fresh final artifact differs from publication")
    return archive


def verify_final_archive(archive, evidence, kind):
    # Read only bounded regular members; never extract archive paths.
    required = ("scientific-summary.json", "protocol.json")
    with tarfile.open(archive, "r:gz") as stream:
        for name in required:
            members = [member for member in stream.getmembers() if member.name == name]
            if len(members) != 1 or not members[0].isfile() or members[0].size > JSON_CAP:
                raise ValueError("missing, duplicate or unsafe final artifact member")
            data = stream.extractfile(members[0]).read(JSON_CAP + 1)
            if data != evidence[kind + "/" + name]:
                raise ValueError("published final artifact differs from local " + name)


def validate_multipart(publication, completion):
    if (publication.get("schema") != "bhl-native-raw-publication-v1"
            or not publication.get("verified_utc") or not publication.get("verification")
            or type(publication.get("bytes")) is not int or not 0 < publication["bytes"] <= 8 * 1024**3):
        raise ValueError("invalid multipart publication")
    check_hash(publication.get("sha256"))
    expected = completion["files"]["outputs.tar.gz"]
    if any(publication[k] != expected[k] for k in ("sha256", "bytes")):
        raise ValueError("multipart publication differs from completed original archive")
    parts = publication.get("parts", [])
    if not 1 <= len(parts) <= 8 or sum(part["bytes"] for part in parts) != publication["bytes"]:
        raise ValueError("multipart publication is incomplete")
    names = []
    for index, part in enumerate(parts):
        name = part.get("asset_name", "")
        if (not re.fullmatch(r"[A-Za-z0-9_.-]+\.tar\.gz\.part" + f"{index:03d}", name)
                or type(part.get("bytes")) is not int or not 0 < part["bytes"] <= 1024**3
                or part.get("url") != f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{name}"):
            raise ValueError("multipart publication part identity differs")
        check_hash(part.get("sha256"))
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("duplicate multipart publication")


def validate_recovered_shard(recovery, receipt, inventory, completion, launch, job, plan, directory):
    if (recovery["job_name"] != "native-controls-v2-s730000-c04-20261011" or str(recovery["job_id"]) != "21776269"
            or receipt.get("schema") != "bhl-native-controls-packaging-recovery-v1" or receipt.get("status") != "COMPLETE"
            or receipt.get("scientific_rerun") is not False or receipt.get("recovered_scientific_status") != "PASS"
            or str(receipt.get("original_job_id")) != str(job["job_id"])
            or receipt.get("source_archive_sha256") != plan["archive_sha256"]
            or receipt.get("protocol_sha256") != plan["protocol_sha256"]
            or receipt.get("original_completion_sha256") != digest(directory / "completion.json")
            or receipt.get("result_sha256") != recovery["result_sha256"]
            or receipt.get("inventory_sha256") != recovery["inventory_sha256"]
            or receipt.get("original_runtime_exit_status") != 0
            or completion.get("status") != "INCOMPLETE" or completion.get("exit_status") != 0
            or completion.get("archive_sha256") != plan["archive_sha256"]
            or launch.get("archive_sha256") != plan["archive_sha256"]
            or completion.get("error") != f"output exceeds predeclared archive cap ({completion.get('output_bytes')} bytes)"):
        raise ValueError("recovery does not establish the exact original packaging-only failure")
    for name, pin in completion["files"].items():
        path = directory / name
        if Path(name).name != name or path.is_symlink() or digest(path) != pin["sha256"] or path.stat().st_size != pin["bytes"]:
            raise ValueError("original failed completion evidence changed")
    for name, pin in inventory.items():
        if Path(name).is_absolute() or ".." in Path(name).parts or type(pin.get("bytes")) is not int or pin["bytes"] < 0:
            raise ValueError("invalid recovered raw inventory")
        check_hash(pin.get("sha256"))
    names = []
    for part in receipt.get("parts", []):
        checked_publication(part)
        names.extend(part["files"])
    total = completion["output_bytes"]
    if (not 2 * 1024**3 < total <= 8 * 1024**3 or len(names) != len(set(names)) or set(names) != set(inventory)
            or sum(pin["bytes"] for pin in inventory.values()) != total or receipt["raw_output_bytes"] != total):
        raise ValueError("recovery parts do not cover the original raw inventory")
    publication = job["publication"]
    if (publication.get("schema") != "bhl-native-recovered-tar-volumes-v1" or publication.get("parts") != receipt["parts"]
            or publication.get("inventory") != recovery["inventory"] or publication.get("inventory_sha256") != recovery["inventory_sha256"]
            or publication.get("raw_output_bytes") != total or job.get("packaging_recovered") is not True
            or job.get("recovery_receipt_sha256") != recovery["receipt_sha256"]):
        raise ValueError("controller recovery publication differs from pinned original evidence")


def shard_results(kind, spec, plan, state, evidence):
    expected_names = [row["name"] for row in plan["jobs"]]
    jobs = state.get("jobs", [])
    names = [row["name"] for row in jobs]
    if len(set(names)) != len(names) or set(names) != set(expected_names):
        raise ValueError("complete unique declared shard inventory required")
    campaign = scoped(plan["campaign_dir"])
    rows = []
    for job in jobs:
        name = job["name"]
        if (job.get("state") != "COMPLETE" or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
                or not re.fullmatch(r"[0-9]+", str(job.get("job_id", "")))):
            raise ValueError("noncomplete or unsafe shard identity")
        directory = scoped(campaign / (name + "-" + str(job["job_id"])))
        label = kind + "/shards/" + name
        completion = snapshot(directory / "completion.json", evidence, label + "/completion.json")
        recovery = plan.get("recovery")
        recovery = recovery if recovery and recovery.get("job_name") == name else None
        if recovery:
            # An independently reviewed packaging recovery must be explicitly pinned
            # by the observer plan; the original failed completion remains unchanged.
            if kind != "native" or str(recovery.get("job_id")) != str(job["job_id"]):
                raise ValueError("recovery belongs to a different scientific shard")
            for prefix in ("result", "receipt", "inventory"):
                path = scoped(recovery[prefix])
                if digest(path) != recovery[prefix + "_sha256"]:
                    raise ValueError("pinned recovered scientific evidence changed")
            receipt = snapshot(recovery["receipt"], evidence, label + "/recovery.json")
            inventory = snapshot(recovery["inventory"], evidence, label + "/recovery-inventory.json")
            launch = snapshot(directory / "launch.json", evidence, label + "/launch.json")
            validate_recovered_shard(recovery, receipt, inventory, completion, launch, job, plan, directory)
            result = snapshot(recovery["result"], evidence, label + "/campaign_result.json")
            if result.get("phase") != "development" or result.get("expected_declared_cells") != 10 or result.get("episodes") != 10:
                raise ValueError("recovered result is not the ten original development cells")
        else:
            result = snapshot(directory / "campaign_result.json", evidence, label + "/campaign_result.json")
            data = evidence[label + "/campaign_result.json"]
            pin = completion["files"]["campaign_result.json"]
            if (completion.get("status") != "PASS" or completion.get("exit_status") != 0
                    or completion.get("error") or completion.get("archive_sha256") != plan["archive_sha256"]
                    or pin["sha256"] != digest_bytes(data) or pin["bytes"] != len(data)):
                raise ValueError("shard result not bound to a successful completion")
            publication = job["publication"]
            if publication.get("schema") == "bhl-native-raw-publication-v1":
                transport = snapshot(directory / "remote-result.json", evidence, label + "/remote-result.json")
                if (kind != "native" or transport.get("schema") != "bhl-native-transport-result-v1"
                        or transport.get("status") != "PASS" or transport.get("runner_returncode") != 0
                        or transport.get("runtime_exit_status") != 0 or transport.get("source_archive_sha256") != plan["archive_sha256"]
                        or transport.get("protocol_sha256") != plan["protocol_sha256"] or transport.get("plan_sha256") != spec["plan_sha256"]
                        or transport.get("original_max_output_mb") != 2048 or transport.get("operational_max_output_mb") != 8192
                        or transport.get("publication") != publication):
                    raise ValueError("node-local transport result differs from completed scientific shard")
                validate_multipart(publication, completion)
            else:
                checked_publication(publication)
                if publication.get("source_archive_sha256") != plan["archive_sha256"]:
                    raise ValueError("shard publication scientific source mismatch")
        if result.get("status") != "PASS" or result.get("problems"):
            raise ValueError("scientific shard collection incomplete")
        rows.append((name, result, digest_bytes(evidence[label + "/campaign_result.json"])))
    return rows


def native_id(cell):
    return tuple(cell[key] for key in ("case", "seed", "condition", "arm"))


def validate_native(summary, rows, protocol):
    settings = []
    for job in protocol["jobs"]:
        if job["kind"] == "run":
            args = job["args"]
            condition = args[args.index("--condition") + 1]
            # Older launch protocols use names; an integer index is not inferred.
            if condition not in CONDITIONS:
                raise ValueError("native condition not explicitly named in protocol")
            settings.append((int(args[args.index("--seed") + 1]), condition))
    seeds = {seed for seed, _ in settings}
    if len(seeds) != 2 or len(settings) != 20 or set(settings) != {(s, c) for s in seeds for c in CONDITIONS}:
        raise ValueError("native protocol must declare exactly two seeds and ten conditions")
    expected = {(case, seed, condition, arm) for case in ("straight", "occluders")
                for seed, condition in settings for arm in ARMS}
    ready = summary.get("arm_development_ready", {})
    if set(ready) != ARMS or any(type(value) is not bool for value in ready.values()):
        raise ValueError("explicit per-arm nominal gates required")
    measured, unrun = [], []
    for _, result, _ in rows:
        if result.get("arm_development_ready") != ready or result.get("episodes") != len(result["completed_cells"]):
            raise ValueError("native shard counts or readiness disagree")
        for cell in result["completed_cells"]:
            if not ready[cell["arm"]]:
                raise ValueError("measured cell follows a negative nominal gate")
            measured.append(native_id(cell))
        for cell in result["unrun_cells"]:
            if ready[cell["arm"]] or cell.get("reason") != "UNRUN_AFTER_NEGATIVE_NOMINAL_ARM_SMOKE":
                raise ValueError("unrun cell lacks its declared negative gate")
            unrun.append(native_id(cell))
    all_ids = measured + unrun
    if len(all_ids) != 200 or len(set(all_ids)) != 200 or set(all_ids) != expected:
        raise ValueError("native cell identities are missing, duplicate or undeclared")
    if (summary.get("status") != "PASS" or summary.get("declared_cells") != 200
            or summary.get("measured_cells") != len(measured) or summary.get("unrun_after_nominal_gate") != len(unrun)
            or summary.get("missing_cells") != []):
        raise ValueError("native complete summary counts disagree")
    if Counter(row["sha256"] for row in summary["source_results"]) != Counter(row[2] for row in rows):
        raise ValueError("native summary does not bind every shard result")
    groups = summary["by_arm_condition"]
    if set(groups) != {arm + "/" + condition for arm in ARMS for condition in CONDITIONS}:
        raise ValueError("native arm/condition summary inventory differs")
    for key, group in groups.items():
        arm, condition = key.split("/")
        count = sum(x[3] == arm and x[2] == condition for x in measured)
        if group["episodes"] != count:
            raise ValueError("native group count differs from cell identities")
        for metric in ("clean_goals", "falls", "contacts"):
            if type(group[metric]) is not int or not 0 <= group[metric] <= count:
                raise ValueError("native outcome count outside measured denominator")
    return {"measured": len(measured), "unrun_after_negative_gate": len(unrun), "declared": 200,
            "identity_sha256": digest_bytes(json.dumps(sorted(all_ids)).encode())}


def validate_terrain(summary, rows, protocol):
    cfg = protocol["terrain_confirmation"]
    qualified = {(row["actor"], row["arm"]) for row in cfg["qualified"]}
    expected = {(actor, terrain, seed, arm) for actor in {a for a, _ in qualified}
                for terrain in cfg["terrains"] for seed in cfg["confirmation_seeds"]
                for arm in {"baseline"} | {b for a, b in qualified if a == actor}}
    identities, paths = [], set()
    for _, result, _ in rows:
        if result.get("protocol_sha256") != summary.get("protocol_sha256"):
            raise ValueError("terrain shard belongs to a different scientific protocol")
        if result.get("episodes") != len(result["episode_files"]):
            raise ValueError("terrain shard episode count differs")
        for path in result["episode_files"]:
            match = re.fullmatch(r"([A-Za-z0-9_-]+)--([A-Za-z0-9_-]+)--s([0-9]+)--([a-z_]+)/confirmation-episode.json", path)
            if not match:
                raise ValueError("unsafe or unknown terrain episode identity")
            actor, terrain, seed, arm = match.groups()
            if result.get("actor_filter") != actor or result.get("terrain_filter") != terrain:
                raise ValueError("terrain episode differs from its declared shard")
            identities.append((actor, terrain, int(seed), arm))
            paths.add(path)
    if len(expected) != 480 or len(identities) != 480 or len(set(identities)) != 480 or set(identities) != expected:
        raise ValueError("terrain episode identities are missing, duplicate or undeclared")
    sources = summary.get("episode_sources", [])
    suffixes = ["/".join(Path(row["path"]).parts[-2:]) for row in sources]
    if len(sources) != 480 or len(set(suffixes)) != 480 or set(suffixes) != paths:
        raise ValueError("terrain summary episode inventory differs")
    for row in sources:
        check_hash(row["sha256"])
    if (summary.get("status") != "PASS" or summary.get("episodes") != 480
            or summary.get("baseline_unique_episodes") != 180 or summary.get("candidate_unique_episodes") != 300):
        raise ValueError("terrain complete summary denominators differ")
    comparisons = summary["qualified_comparisons"]
    if len(comparisons) != 5 or {(r["actor"], r["arm"]) for r in comparisons} != qualified:
        raise ValueError("terrain qualification identity differs")
    for row in comparisons:
        for label in ("candidate", "matched_baseline"):
            counts = row[label]
            if counts["episodes"] != 60 or any(type(counts[k]) is not int or not 0 <= counts[k] <= 60
                                              for k in ("clean_goals", "falls", "contacts", "nonfinite")):
                raise ValueError("terrain comparison denominator differs")
        c, b = row["candidate"], row["matched_baseline"]
        gate = c["clean_goals"] >= 54 and all(c[k] == 0 for k in ("falls", "contacts", "nonfinite")) and c["clean_goals"] > b["clean_goals"]
        if type(row["confirmed"]) is not bool or row["confirmed"] != gate:
            raise ValueError("terrain reported gate disagrees with unchanged rule")
    confirmed = sum(row["confirmed"] for row in comparisons)
    if summary.get("confirmed_combinations") != confirmed:
        raise ValueError("terrain confirmation total differs")
    return {"measured": 480, "unrun_after_negative_gate": 0, "declared": 480,
            "confirmed_combinations": confirmed, "identity_sha256": digest_bytes(json.dumps(sorted(identities)).encode())}


def inspect_controller(kind, spec, evidence, work, gh, environment, downloader=download_final):
    directory = scoped(spec["directory"])
    plan = snapshot(directory / "plan.json", evidence, kind + "/plan.json")
    if digest_bytes(evidence[kind + "/plan.json"]) != spec["plan_sha256"]:
        raise ValueError("controller plan changed")
    if not (directory / "state.json").exists():
        return {"controller": str(directory), "status": "NOT_STARTED", "complete": False, "completed_shards": 0}
    state = snapshot(directory / "state.json", evidence, kind + "/state.json")
    if state.get("plan_sha256") != spec["plan_sha256"]:
        raise ValueError("controller state belongs to a different plan")
    status = state.get("status", "UNKNOWN")
    result = {"controller": str(directory), "status": status, "complete": False,
              "completed_shards": sum(row.get("state") == "COMPLETE" for row in state.get("jobs", []))}
    if status != "COMPLETE":
        if state.get("error"):
            result["error"] = str(state["error"])
        return result
    campaign = scoped(plan["campaign_dir"])
    protocol = snapshot(campaign / "protocol.json", evidence, kind + "/protocol.json")
    if digest_bytes(evidence[kind + "/protocol.json"]) != plan["protocol_sha256"]:
        raise ValueError("scientific protocol changed")
    summary = snapshot(directory / "scientific-summary.json", evidence, kind + "/scientific-summary.json")
    publication = snapshot(directory / "publication.json", evidence, kind + "/publication.json")
    checked_publication(publication, final=True)
    if state.get("final_publication") != publication:
        raise ValueError("state and final publication disagree")
    rows = shard_results(kind, spec, plan, state, evidence)
    if kind == "native":
        smoke = snapshot(directory / "qualified-smoke.json", evidence, kind + "/qualified-smoke.json")
        if summary.get("qualified_smoke_sha256") != digest_bytes(evidence[kind + "/qualified-smoke.json"]):
            raise ValueError("native summary qualified smoke changed")
        for _, shard, _ in rows:
            if any(shard.get(field) != smoke.get(field) for field in ("source_sha256", "input_sha256")):
                raise ValueError("native shard scientific identity differs from its qualified smoke")
    counts = validate_native(summary, rows, protocol) if kind == "native" else validate_terrain(summary, rows, protocol)
    if kind == "terrain" and summary.get("protocol_sha256") != plan["protocol_sha256"]:
        raise ValueError("terrain summary protocol differs")
    final_work = work / kind
    final_work.mkdir()
    archive = downloader(publication, final_work, gh, environment)
    verify_final_archive(archive, evidence, kind)
    result.update(complete=True, counts=counts, publication=publication,
                  scientific_status=summary.get("scientific_status"), summary=summary,
                  source_archive_sha256=plan["archive_sha256"], protocol_sha256=plan["protocol_sha256"],
                  final_artifact_fresh_download_verified=True)
    return result


def render_report(result, evidence_path):
    lines = ["# October 11 simulation completion evidence", "", f"Observed at {result['observed_utc']}.", "",
             "This report covers the native camera/fault controls and independent native-pose terrain confirmation. "
             "Hardware validation remains unrun; other research programs are outside this report.", "",
             f"Overall collection status: **{result['status']}**. Completed collection does not mean every method improved.", ""]
    if result.get("deadline_reached"):
        lines.extend(["The bounded observer deadline expired before both complete collections could be verified. "
                      "Active scientific jobs were left untouched.", ""])
    for kind, title in (("native", "Native camera and sensor-fault controls"), ("terrain", "Independent terrain confirmation")):
        row = result["campaigns"][kind]
        lines.extend(["## " + title, "", f"Controller status: **{row['status']}**.", ""])
        if not row.get("complete"):
            lines.append(f"Completed shards recorded by the controller: {row.get('completed_shards', 0)}. "
                         "Complete scientific collection has not been verified.")
            if row.get("error"):
                lines.append("\nRecorded failure: " + str(row["error"]).replace("\n", " ").replace("`", "'"))
            lines.append("")
            continue
        counts, summary = row["counts"], row["summary"]
        lines.extend([f"Measured **{counts['measured']}/{counts['declared']}** declared cells; "
                      f"**{counts['unrun_after_negative_gate']}** unrun after a negative prerequisite gate.", ""])
        if kind == "native":
            lines.extend(["| Estimator/controller | Measured episodes | Clean goals | Falls | Contacts |", "|---|---:|---:|---:|---:|"])
            for arm in sorted(ARMS):
                groups = [v for key, v in summary["by_arm_condition"].items() if key.startswith(arm + "/")]
                totals = [sum(v[k] for v in groups) for k in ("episodes", "clean_goals", "falls", "contacts")]
                lines.append("| " + arm + " | " + " | ".join(map(str, totals)) + " |")
            lines.extend(["", "Counts pool fixed conditions and two reset seeds for description; they do not establish population reliability. "
                          "The final archive also retains the separately frozen offline health screen; this report does not claim an online health-control improvement.", ""])
        else:
            lines.extend([f"**{counts['confirmed_combinations']}/5** prespecified actor/controller combinations confirmed.", "",
                          "| Actor / controller | Candidate clean goals | Matched baseline clean goals | Falls / contacts | Gate |", "|---|---:|---:|---:|---|"])
            for comparison in summary["qualified_comparisons"]:
                c, b = comparison["candidate"], comparison["matched_baseline"]
                lines.append(f"| {comparison['actor']} / {comparison['arm']} | {c['clean_goals']}/60 | {b['clean_goals']}/60 | "
                             f"{c['falls']} / {c['contacts']} | {'PASS' if comparison['confirmed'] else 'NEGATIVE'} |")
            lines.extend(["", "Matched baselines are reused across controllers: 180 unique baseline and 300 candidate episodes. "
                          "The cohort uses 20 shared geometry/reset-seed clusters, fixed actors and terrain families.", ""])
        lines.extend([f"[Verified final evidence]({row['publication']['url']}). Scientific source SHA256: `{row['source_archive_sha256']}`.", ""])
    lines.extend([f"[Machine-readable report and retained receipt inventory]({evidence_path}/report.json). "
                  "The observer verifies compact identities, final-artifact bytes and published summary consistency. "
                  "Raw-trace rescoring was performed by the campaign collectors, not repeated by this observer.", ""])
    return "\n".join(lines)


def write_bundle(output, result, evidence):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in evidence.items():
        path = output / "evidence" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    report_data = json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    (output / "report.json").write_bytes(report_data)
    (output / "report.md").write_text(render_report(result, "."))
    manifest = {"schema": "bhl-research-completion-evidence-v1", "files": {
        str(p.relative_to(output)): {"sha256": digest(p), "bytes": p.stat().st_size}
        for p in sorted(output.rglob("*")) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    identifier = result["observed_utc"].replace("-", "").replace(":", "").replace("+00:00", "Z").replace("+0000", "Z")
    identifier = re.sub(r"[^A-Za-z0-9]", "", identifier) + "-" + digest_bytes(report_data)[:12]
    return identifier


def push_bundle(bundle, result, identifier, gh, environment):
    # Every retry starts from current remote main in a new private checkout.
    # No force push, shared checkout edits, rebase conflict choices or existing-file changes.
    docs_path = Path("docs/research-completion-20261011") / (identifier + ".md")
    evidence_path = Path("results/research-completion-20261011") / identifier
    for attempt in range(3):
        with tempfile.TemporaryDirectory(prefix="bhl-report-git-", dir="/tmp") as temporary:
            checkout = Path(temporary) / "repo"
            git = ["/usr/bin/git", "-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential"]
            command([*git, "clone", "--depth", "1", "--filter=blob:none", "--sparse", "--branch", "main",
                     f"https://github.com/{REPOSITORY}.git", checkout], environment=environment)
            command([*git, "sparse-checkout", "set", "docs/research-completion-20261011", "results/research-completion-20261011"],
                    cwd=checkout, environment=environment)
            target, doc = checkout / evidence_path, checkout / docs_path
            if target.exists() or doc.exists():
                raise ValueError("immutable report destination already exists")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(bundle, target)
            doc.parent.mkdir(parents=True, exist_ok=True)
            doc.write_text(render_report(result, "../../" + evidence_path.as_posix()))
            command([*git, "add", "--sparse", "--", docs_path, evidence_path], cwd=checkout, environment=environment)
            changed = command([*git, "diff", "--cached", "--name-only"], cwd=checkout, environment=environment).stdout.decode().splitlines()
            if any(p != docs_path.as_posix() and not p.startswith(evidence_path.as_posix() + "/") for p in changed):
                raise ValueError("report commit escaped its new task-scoped files")
            command([*git, "-c", "user.name=BHL research reporter", "-c", "user.email=bhl-research-reporter@users.noreply.github.com",
                     "commit", "-m", "Record October 11 simulation completion evidence"], cwd=checkout, environment=environment)
            pushed = command([*git, "push", "origin", "HEAD:main"], cwd=checkout, environment=environment, check=False)
            if pushed.returncode == 0:
                commit = command([*git, "rev-parse", "HEAD"], cwd=checkout, environment=environment).stdout.decode().strip()
                return {"commit": commit, "report_url": f"https://github.com/{REPOSITORY}/blob/{commit}/{docs_path}"}
            if attempt == 2:
                raise RuntimeError("normal Git push rejected three times; no remote history was rewritten")


def run(plan_path, plan_sha, *, publish=False, once=False):
    plan = validate_plan(plan_path, plan_sha)
    output_root = scoped(plan["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + plan["timeout_hours"] * 3600
    with tempfile.TemporaryDirectory(prefix="bhl-report-tools-", dir="/tmp") as temporary:
        tools = Path(temporary)
        gh = tools / "gh"
        shutil.copyfile(plan["gh_path"], gh)
        gh.chmod(0o700)
        if digest(gh) != plan["gh_sha256"]:
            raise ValueError("copied GitHub executable changed")
        environment = {**os.environ, "PATH": str(tools) + ":/apps/slurm/25.11.8/bin:/usr/bin:/bin", "TMPDIR": "/tmp",
                       "GIT_TERMINAL_PROMPT": "0", "GH_PROMPT_DISABLED": "1"}
        while True:
            evidence = {"observer-plan.json": bounded_bytes(plan_path), "research_completion_report.py": bounded_bytes(__file__)}
            campaigns = {}
            # Poll only compact state; final archives are fetched once at the
            # terminal boundary, not repeatedly while the other campaign runs.
            for kind, spec in plan["controllers"].items():
                try:
                    state_path = scoped(spec["directory"]) / "state.json"
                    if not state_path.exists():
                        campaigns[kind] = {"status": "NOT_STARTED", "complete": False}
                        continue
                    state = read(state_path)
                    if state.get("plan_sha256") != spec["plan_sha256"]:
                        raise ValueError("controller state belongs to a different plan")
                    campaigns[kind] = {"status": state.get("status", "UNKNOWN"), "complete": False}
                except Exception as error:
                    campaigns[kind] = {"status": "INCOMPLETE", "complete": False, "error": str(error)}
            expired = time.monotonic() >= deadline
            terminal = (all(row["status"] == "COMPLETE" for row in campaigns.values())
                        or any(row["status"] in FAILURES for row in campaigns.values()) or expired or once)
            if not terminal:
                heartbeat = {"updated_utc": datetime.now(timezone.utc).isoformat(), "status": "WAITING",
                             "controllers": {key: row["status"] for key, row in campaigns.items()}}
                tmp = output_root / ".heartbeat.tmp"
                tmp.write_text(json.dumps(heartbeat, indent=2) + "\n")
                tmp.replace(output_root / "heartbeat.json")
                time.sleep(30)
                continue
            with tempfile.TemporaryDirectory(prefix="bhl-report-download-", dir="/tmp") as download_dir:
                for kind, spec in plan["controllers"].items():
                    try:
                        campaigns[kind] = inspect_controller(kind, spec, evidence, Path(download_dir), gh, environment)
                    except Exception as error:
                        campaigns[kind] = {"status": "INCOMPLETE", "complete": False,
                                           "controller": spec["directory"], "error": str(error)}
            complete = all(row.get("complete") for row in campaigns.values())
            failed = any(row["status"] in FAILURES for row in campaigns.values())
            if terminal:
                observed = datetime.now(timezone.utc).isoformat(timespec="seconds")
                result = {"schema": "bhl-research-completion-report-v1", "observed_utc": observed,
                          "status": "COMPLETE" if complete else "INCOMPLETE", "deadline_reached": expired,
                          "hardware_validation_complete": False, "campaigns": campaigns,
                          "reporter_sha256": plan["reporter_sha256"], "observer_plan_sha256": plan_sha}
                bundle = output_root / ("snapshot-" + observed.replace(":", "").replace("+", "_") + "-" + str(os.getpid()))
                identifier = write_bundle(bundle, result, evidence)
                if publish:
                    receipt = push_bundle(bundle, result, identifier, gh, environment)
                    # Keep publication receipt outside the immutable published bundle.
                    (output_root / (identifier + "-git-publication.json")).write_text(json.dumps(receipt, indent=2) + "\n")
                print(json.dumps({"status": result["status"], "bundle": str(bundle), "published": publish}), flush=True)
                return 0 if complete else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--publish", action="store_true", help="push only new immutable report/evidence paths to main")
    parser.add_argument("--once", action="store_true", help="write a current snapshot without waiting")
    args = parser.parse_args()
    raise SystemExit(run(args.plan, args.plan_sha256, publish=args.publish, once=args.once))
