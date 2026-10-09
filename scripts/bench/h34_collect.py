"""Collect complete frozen H3/H4 arrays and recompute their scientific verdicts.

No submission, Git or publishing action is performed. Original receipts,
archives and deployment bytes are preserved. A missing or malformed cell
produces an INCOMPLETE receipt, regardless of scheduler completion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

MAX_JSON_BYTES = 8 * 1024 ** 2
MAX_FROZEN_BYTES = 2 * 1024 ** 3
MAX_COLLECTION_BYTES = 4 * 1024 ** 3
MAX_MEMBERS = 20000
MAX_RECEIPT_OVERHEAD = 128 * 1024 ** 2


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative_name(name):
    if not isinstance(name, str):
        raise ValueError(f"unsafe archive or receipt path: {name!r}")
    path = PurePosixPath(name)
    if not isinstance(name, str) or not name or "\\" in name or path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError(f"unsafe archive or receipt path: {name!r}")
    return path


def regular_file(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or linked input: {path}")
    return path


def read_json(path):
    path = regular_file(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"JSON receipt exceeds cap: {path}")
    def nonfinite(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    return json.loads(path.read_text(), parse_constant=nonfinite)


def checked_members(tar, *, cap):
    members = tar.getmembers()
    if len(members) > MAX_MEMBERS:
        raise ValueError("archive member count exceeds cap")
    seen = set()
    for member in members:
        relative_name(member.name)
        if member.name in seen or not member.isfile() or member.size < 0 or member.size > 512 * 1024 ** 2:
            raise ValueError("duplicate, non-file or oversized archive member")
        seen.add(member.name)
    total = sum(member.size for member in members)
    if total > cap:
        raise ValueError("uncompressed archive exceeds cap")
    # Refuse file/path collisions before any extraction occurs.
    for name in seen:
        if any(str(parent) in seen for parent in PurePosixPath(name).parents if str(parent) != "."):
            raise ValueError("archive member is both a file and a parent directory")
    return members, total


def safe_extract(archive, destination, *, cap):
    destination = Path(destination)
    with tarfile.open(regular_file(archive), "r:gz") as tar:
        members, total = checked_members(tar, cap=cap)
        if shutil.disk_usage(destination.parent).free < total + 16 * 1024 ** 2:
            raise ValueError("insufficient space for capped extraction")
        destination.mkdir(exist_ok=False)
        hashes = {}
        for member in members:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            hashes[member.name] = {"sha256": digest(target), "bytes": member.size}
    return hashes, total


def verify_frozen_archive(persistent, intake):
    archive = regular_file(persistent / "frozen-campaign.tar.gz")
    if archive.stat().st_size > MAX_FROZEN_BYTES + 16 * 1024 ** 2:
        raise ValueError("compressed frozen archive exceeds cap")
    if digest(archive) != intake.get("archive_sha256") or archive.stat().st_size != intake.get("archive_bytes"):
        raise ValueError("frozen archive differs from intake")
    protocol_path = regular_file(persistent / "protocol.json")
    if digest(protocol_path) != intake.get("protocol_sha256"):
        raise ValueError("protocol differs from intake")
    launcher = regular_file(persistent / "h34_campaign.sbatch")
    if launcher.stat().st_size > 1024 ** 2:
        raise ValueError("launcher exceeds bounded script size")
    if digest(launcher) != intake.get("launcher_sha256"):
        raise ValueError("launcher differs from intake")
    with tarfile.open(archive, "r:gz") as tar:
        members, total = checked_members(tar, cap=MAX_FROZEN_BYTES)
        by_name = {member.name: member for member in members}
        if not {"protocol.json", "manifest.json"} <= set(by_name):
            raise ValueError("frozen archive lacks protocol/manifest")
        if by_name["protocol.json"].size > MAX_JSON_BYTES or by_name["manifest.json"].size > MAX_JSON_BYTES:
            raise ValueError("frozen protocol/manifest exceeds cap")
        with tar.extractfile(by_name["protocol.json"]) as stream:
            if hashlib.sha256(stream.read()).hexdigest() != intake["protocol_sha256"]:
                raise ValueError("archived protocol differs from durable protocol")
        with tar.extractfile(by_name["manifest.json"]) as stream:
            manifest_bytes = stream.read()
        manifest = json.loads(manifest_bytes)
        if set(by_name) != set(manifest) | {"protocol.json", "manifest.json"}:
            raise ValueError("frozen archive inventory differs from manifest")
        for name, row in manifest.items():
            relative_name(name)
            if not name.startswith(("source/", "inputs/")) or by_name[name].size != row["bytes"]:
                raise ValueError(f"frozen manifest size/path mismatch: {name}")
            h = hashlib.sha256()
            with tar.extractfile(by_name[name]) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != row["sha256"]:
                raise ValueError(f"frozen source/input checksum mismatch: {name}")
    if regular_file(persistent / "manifest.json").read_bytes() != manifest_bytes:
        raise ValueError("durable and archived manifests differ")
    if len(manifest) != intake.get("source_files") or sum(row["bytes"] for row in manifest.values()) != intake.get("uncompressed_input_bytes"):
        raise ValueError("frozen inventory count/bytes differ from intake")
    return manifest, total


def array_indices(spec):
    if not isinstance(spec, str) or not re.fullmatch(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*(?:%\d+)?", spec):
        raise ValueError("missing/invalid scored array specification")
    result = []
    for part in spec.split("%")[0].split(","):
        lo, hi = map(int, part.split("-")) if "-" in part else (int(part), int(part))
        if hi < lo or hi > 31:
            raise ValueError("invalid/unbounded array range")
        result.extend(range(lo, hi + 1))
    if len(result) != len(set(result)):
        raise ValueError("duplicate scored array index")
    return set(result)


def validate_submission(persistent, protocol, intake):
    jobs = [job for job in protocol["jobs"] if job["kind"] == "run"]
    if len(jobs) != 1:
        raise ValueError("collection requires exactly one declared scored array")
    job = jobs[0]
    if array_indices(job["resources"]["array"]) != set(range(len(protocol["cells"]))):
        raise ValueError("scored array does not cover every declared cell")
    receipt = read_json(persistent / (job["name"] + "-submission.json"))
    if receipt.get("status") != "SUBMITTED" or receipt.get("returncode") != 0 or not re.fullmatch(r"\d+", receipt.get("job_id", "")):
        raise ValueError("missing/failed full-array submission")
    if receipt.get("archive_sha256") != intake["archive_sha256"] or receipt.get("resources") != job["resources"]:
        raise ValueError("full-array submission archive/resources differ")
    command = receipt.get("command", [])
    dependency = [part for part in command if part.startswith("--dependency=afterok:")]
    if len(dependency) != 1:
        raise ValueError("full-array submission lacks a unique fresh-smoke dependency")
    smoke_id = dependency[0].split(":")[-1]
    matching_smoke = False
    for smoke in (j for j in protocol["jobs"] if j["kind"] == "smoke"):
        smoke_receipt = read_json(persistent / (smoke["name"] + "-submission.json"))
        matching_smoke |= (smoke_receipt.get("job_id") == smoke_id and smoke_receipt.get("archive_sha256") == intake["archive_sha256"]
                           and smoke_receipt.get("status") == "SUBMITTED")
    if not matching_smoke:
        raise ValueError("run dependency does not match a smoke from this frozen archive")
    return job, receipt


def check_completion(directory, job, intake, protocol):
    directory = Path(directory)
    completion = read_json(directory / "completion.json")
    if completion.get("archive_sha256") != intake["archive_sha256"]:
        raise ValueError("cell completion belongs to another source archive")
    if completion.get("status") not in {"PASS", "NEGATIVE"} or completion.get("exit_status") != 0 or completion.get("error") is not None:
        raise ValueError("cell execution is incomplete")
    files = completion.get("files", {})
    if not {"launch.json", "outputs.tar.gz", "campaign_result.json", "runtime.log"} <= set(files):
        raise ValueError("cell completion lacks required durable artifacts")
    for name, row in files.items():
        relative_name(name)
        if len(PurePosixPath(name).parts) != 1:
            raise ValueError("durable completion file must be a direct child")
        path = regular_file(directory / name)
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"durable completion checksum/size mismatch: {name}")
    launch = read_json(directory / "launch.json")
    if launch.get("archive_sha256") != intake["archive_sha256"] or launch.get("job") != job:
        raise ValueError("cell launch source/job differs from the declared full array")
    if launch.get("source_files_verified") != intake["source_files"]:
        raise ValueError("cell did not verify the complete frozen source inventory")
    if not re.fullmatch(r"\d+", launch.get("job_id", "")) or directory.name != f"{job['name']}-{launch['job_id']}":
        raise ValueError("cell durable directory differs from its recorded job")
    result = read_json(directory / "campaign_result.json")
    if result.get("phase") != "run" or result.get("status") != completion["status"]:
        raise ValueError("smoke, phase or completion/result status mismatch")
    index = result.get("cell")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(protocol["cells"]):
        raise ValueError("missing/invalid scored cell index")
    cell = protocol["cells"][index]
    if (result.get("arm"), result.get("seed")) != (cell["arm"], cell["seed"]):
        raise ValueError("receipt does not match the declared arm/seed")
    if result.get("protocol_sha256") != intake["protocol_sha256"]:
        raise ValueError("receipt belongs to another frozen protocol")
    command = launch.get("command", [])
    if command.count("--cell") != 1 or command[command.index("--cell") + 1] != str(index):
        raise ValueError("launch command and scored cell identity differ")
    if command.count("--phase") != 1 or command[command.index("--phase") + 1] != "run":
        raise ValueError("launch command is not a scored run")
    cap = job.get("max_output_mb", 512) * 1024 ** 2
    if not isinstance(completion.get("output_bytes"), int) or not 0 <= completion["output_bytes"] <= cap:
        raise ValueError("cell output exceeds its predeclared cap")
    return index, result, completion


def preserve_launch(directory, destination, *, cap, remaining):
    """Snapshot bounded original evidence even when its completion is invalid."""
    if directory.is_symlink():
        raise ValueError("linked launch directory")
    files = list(directory.iterdir())
    if len(files) > 64:
        raise ValueError("durable launch file count exceeds cap")
    total = 0
    for path in files:
        regular_file(path)
        size = path.stat().st_size
        limit = cap + 16 * 1024 ** 2 if path.name == "outputs.tar.gz" else MAX_RECEIPT_OVERHEAD
        if size > limit:
            raise ValueError("durable original file exceeds cap")
        total += size
    if total > cap + MAX_RECEIPT_OVERHEAD or total > remaining:
        raise ValueError("durable originals exceed collection cap")
    if shutil.disk_usage(destination.parent).free < total + 16 * 1024 ** 2:
        raise ValueError("insufficient space to preserve original launch evidence")
    destination.mkdir(exist_ok=False)
    for path in files:
        shutil.copyfile(path, destination / path.name)
    return total


def scientific_summary(task, protocol_path, cells):
    if task == "H3":
        from history_latency_eval import paired_summary
        return paired_summary(cells)
    if task == "H4":
        from h4_campaign_summary import summarize
        return summarize(protocol_path, cells)
    raise ValueError("collector accepts H3 or H4 only")


def collect(persistent_dir, out):
    persistent, out = Path(persistent_dir).resolve(), Path(out).resolve()
    if out == persistent or out.is_relative_to(persistent) or persistent.is_relative_to(out):
        raise ValueError("collection destination must be separate from immutable campaign inputs")
    out.mkdir(parents=True, exist_ok=False)
    problems, copied, expected, source_manifest = [], [], [], {}
    summary, intake, protocol, run_receipt = None, {}, {}, {}
    try:
        intake = read_json(persistent / "intake.json")
        source_manifest, _ = verify_frozen_archive(persistent, intake)
        protocol = read_json(persistent / "protocol.json")
        from h34_campaign import validate_protocol
        validate_protocol(protocol)
        expected = [{"cell": i, "arm": cell["arm"], "seed": cell["seed"]} for i, cell in enumerate(protocol["cells"])]
        if len(expected) != (6 if protocol["task"] == "H3" else 3 if protocol["task"] == "H4" else 0):
            raise ValueError("wrong declared H3/H4 cell count")
        job, run_receipt = validate_submission(persistent, protocol, intake)
        originals = out / "originals"
        originals.mkdir()
        original_files = [regular_file(persistent / name) for name in
                          ("intake.json", "protocol.json", "manifest.json", "h34_campaign.sbatch", "frozen-campaign.tar.gz")]
        submissions = list(persistent.glob("*-submission*.json"))
        if len(submissions) > 64:
            raise ValueError("submission receipt count exceeds cap")
        for submitted in submissions:
            read_json(submitted)
            original_files.append(submitted)
        preserved_bytes = sum(path.stat().st_size for path in original_files)
        if preserved_bytes > MAX_COLLECTION_BYTES or shutil.disk_usage(out).free < preserved_bytes + 16 * 1024 ** 2:
            raise ValueError("original intake evidence exceeds collection space cap")
        for path in original_files:
            shutil.copyfile(path, originals / path.name)
        seen, total_output = set(), 0
        candidates = sorted(persistent.glob(job["name"] + "-*"))
        for directory in candidates:
            if not directory.is_dir():
                continue
            try:
                # Preserve an invalid or failed launch as evidence before
                # deciding whether it can enter the scientific cohort.
                saved = originals / directory.name
                preserved_bytes += preserve_launch(directory, saved, cap=job.get("max_output_mb", 512) * 1024 ** 2,
                                                    remaining=MAX_COLLECTION_BYTES - preserved_bytes - total_output)
                index, result, completion = check_completion(saved, job, intake, protocol)
                if index in seen:
                    raise ValueError(f"duplicate full-array cell {index}")
                # Keep durable originals in a separate directory from the
                # extracted output; never rewrite deploy paths for relocation.
                root = out / "cells" / f"cell-{index}-{result['arm']}-s{result['seed']}"
                root.parent.mkdir(exist_ok=True)
                inventory, expanded_bytes = safe_extract(saved / "outputs.tar.gz", root,
                                                         cap=min(job.get("max_output_mb", 512) * 1024 ** 2,
                                                                 MAX_COLLECTION_BYTES - preserved_bytes - total_output))
                total_output += expanded_bytes
                if total_output + preserved_bytes > MAX_COLLECTION_BYTES or expanded_bytes != completion["output_bytes"]:
                    raise ValueError("collection or declared output byte count exceeds/differs from cap")
                # The wrapper serializes its receipt with sorted JSON keys;
                # the runner may not. Retain both bytes and compare content.
                if read_json(root / "campaign_result.json") != read_json(saved / "campaign_result.json"):
                    raise ValueError("archived and durable campaign receipts differ")
                seen.add(index)
                copied.append({"cell": index, "arm": result["arm"], "seed": result["seed"],
                               "directory": str(root.relative_to(out)), "launch": directory.name,
                               "output_bytes": expanded_bytes, "files": inventory})
            except Exception as exc:
                problems.append(f"{directory.name}: {type(exc).__name__}: {exc}")
        missing = sorted(set(range(len(expected))) - seen)
        if missing:
            problems.append(f"missing complete scored cells: {missing}")
        if not problems:
            roots = [out / row["directory"] for row in sorted(copied, key=lambda row: row["cell"])]
            summary = scientific_summary(protocol["task"], originals / "protocol.json", roots)
            if summary.get("status") not in {"PASS", "NEGATIVE", "INCOMPLETE"}:
                raise ValueError("scientific summary returned an invalid status")
            if summary.get("protocol_sha256") != intake["protocol_sha256"]:
                raise ValueError("recomputed scientific summary belongs to another protocol")
            problems.extend(summary.get("problems", []))
    except Exception as exc:
        problems.append(f"collection: {type(exc).__name__}: {exc}")
    status = "INCOMPLETE" if problems or summary is None else summary["status"]
    receipt = {"schema": "h34-durable-collection-v1", "status": status,
               "task": protocol.get("task"), "campaign_id": protocol.get("campaign_id"),
               "collected_utc": datetime.now(timezone.utc).isoformat(),
               "persistent_dir": str(persistent), "archive_sha256": intake.get("archive_sha256"),
               "protocol_sha256": intake.get("protocol_sha256"), "run_job_id": run_receipt.get("job_id"),
               "expected_cells": expected, "complete_cells": len(copied), "cells": copied,
               "scientific_summary": summary, "problems": problems,
               "collector_sha256": digest(Path(__file__)),
               "preservation": "Original archive/receipt/deploy bytes retained. Absolute deployment pointers are provenance, not relocated runnable exports.",
               "scope": "Whole-array independently recomputed scientific verdict; no scheduler-exit, smoke, Git or publication claim."}
    (out / "collection.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    if summary is not None:
        (out / "scientific-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    report = collect(args.persistent_dir, args.out)
    print(json.dumps({key: report[key] for key in ("status", "task", "complete_cells", "problems")}, indent=2))
    return 1 if report["status"] == "INCOMPLETE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
