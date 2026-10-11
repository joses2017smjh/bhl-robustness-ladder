#!/usr/bin/env python3
"""Promote an immutable source template after a genuinely successful ORB build.

This is provenance/submission orchestration only. It does not infer a scientific
result, execute physics, tune an estimator, or need a surviving Git checkout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
import tempfile


SHA_TOKEN = "__NATIVE_ORB_RUNTIME_SHA256__"
ORB_PIN = "4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_template(template, work):
    """Import the launcher from the verified template, not a mutable checkout."""
    intake = json.loads((template/"intake.json").read_text())
    for name, key in (("frozen-campaign.tar.gz", "archive_sha256"),
                      ("protocol.json", "protocol_sha256"),
                      ("h34_campaign.sbatch", "launcher_sha256")):
        if sha256(template/name) != intake[key]:
            raise ValueError(f"frozen template changed: {name}")
    archive = template/"frozen-campaign.tar.gz"
    with tarfile.open(archive, "r:gz") as stream:
        manifest = json.load(stream.extractfile("manifest.json"))
        member = stream.getmember("source/scripts/bench/h34_campaign.py")
        if not member.isfile() or member.size > 2*1024*1024:
            raise ValueError("template launcher source is invalid")
        code = stream.extractfile(member).read()
    if hashlib.sha256(code).hexdigest() != manifest[member.name]["sha256"]:
        raise ValueError("template launcher source checksum mismatch")
    module_path = work/"verified_launcher.py"
    module_path.write_bytes(code)
    spec = importlib.util.spec_from_file_location("verified_native_runtime_launcher", module_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    extracted = work/"template"
    extracted.mkdir()
    launcher.safe_extract(archive, extracted)
    launcher.verify_source(extracted)
    protocol = json.loads((extracted/"protocol.json").read_text())
    if protocol != json.loads((template/"protocol.json").read_text()):
        raise ValueError("template archive and external protocol differ")
    if manifest != json.loads((template/"manifest.json").read_text()):
        raise ValueError("template archive and external manifest differ")
    launcher.validate_protocol(protocol)
    return launcher, intake, protocol, manifest


def validate_runtime(archive, expected, work, launcher):
    if sha256(archive) != expected:
        raise ValueError("native runtime archive checksum mismatch")
    runtime_work = work/"verified-runtime"
    runtime_work.mkdir()
    launcher.safe_extract(archive, runtime_work)
    runtime = runtime_work/"runtime"
    receipt = json.loads((runtime/"runtime.json").read_text())
    if (receipt.get("schema") != "bhl-native-orb-runtime-v1"
            or receipt.get("status") != "PASS" or receipt.get("upstream_commit") != ORB_PIN
            or receipt.get("native_algorithm_sources_modified") is not False):
        raise ValueError("actual successful pinned unmodified ORB runtime required")
    inventory = receipt["files_sha256"]
    for required in ("bin/orb_native", "ORBvoc.txt", "ORB-SLAM3-LICENSE.txt", "upstream-source.tar.gz"):
        if required not in inventory:
            raise ValueError(f"native runtime lacks required executable/source/license input: {required}")
    for name, checksum in inventory.items():
        launcher.relative_name(name)
        path = runtime/name
        if not path.resolve().is_relative_to(runtime) or sha256(path) != checksum:
            raise ValueError(f"native runtime member checksum mismatch: {name}")
    actual = {str(p.relative_to(runtime)) for p in runtime.rglob("*") if p.is_file()}
    if actual != set(inventory)|{"runtime.json"}:
        raise ValueError("native runtime file inventory differs from its actual receipt")
    verified = {"runtime_receipt_sha256": sha256(runtime/"runtime.json"),
                "upstream_commit": receipt["upstream_commit"], "runtime_files": len(inventory),
                "binary_sha256": inventory["bin/orb_native"], "archive_sha256": expected}
    shutil.rmtree(runtime_work)
    return verified


def fill_runtime_sha(value, checksum):
    if isinstance(value, str):
        return value.replace(SHA_TOKEN, checksum), value.count(SHA_TOKEN)
    if isinstance(value, list):
        output, count = [], 0
        for child in value:
            replaced, changed = fill_runtime_sha(child, checksum)
            output.append(replaced)
            count += changed
        return output, count
    if isinstance(value, dict):
        output, count = {}, 0
        for key, child in value.items():
            replaced, changed = fill_runtime_sha(child, checksum)
            output[key] = replaced
            count += changed
        return output, count
    return value, 0


def promote(template, runtime_archive, runtime_sha256, persistent, *,
            runtime_input_name="runtime/orb-runtime.tar.gz", submit=False):
    template, runtime_archive, persistent = (Path(p).resolve() for p in (template, runtime_archive, persistent))
    if persistent.exists():
        raise ValueError("promotion target already exists; refusing duplicate freeze/submission")
    with tempfile.TemporaryDirectory(prefix="bhl-native-promotion-") as temporary:
        work = Path(temporary)
        launcher, parent_intake, original, manifest = load_template(template, work)
        launcher.relative_name(runtime_input_name)
        if runtime_input_name in original.get("input_files", {}):
            raise ValueError("template must omit the not-yet-existing runtime input")
        runtime_receipt = validate_runtime(runtime_archive, runtime_sha256, work, launcher)
        protocol, replacements = fill_runtime_sha(original, runtime_sha256)
        if replacements < 1:
            raise ValueError("template has no explicit deferred runtime checksum")
        protocol.setdefault("input_files", {})[runtime_input_name] = {
            "path": str(runtime_archive), "sha256": runtime_sha256}
        deferred = protocol.get("deferred_native_orb_runtime")
        if isinstance(deferred, dict):
            # A source template can predate a failed or cancelled build. Those
            # old requested IDs must never be attributed to the actual runtime.
            for old, historical in (("build_job_id", "original_requested_build_job_id"),
                                    ("build_archive_sha256", "original_requested_build_archive_sha256")):
                if old in deferred:
                    deferred[historical] = deferred.pop(old)
            deferred.update(status="VERIFIED_RUNTIME_PROMOTED", archive_sha256=runtime_sha256,
                            source_template_archive_sha256=parent_intake["archive_sha256"],
                            actual_build_provenance="verified runtime receipt and immutable controller promotion receipt; original requested build IDs are historical")
        launcher.validate_protocol(protocol)
        runtime_bytes = runtime_archive.stat().st_size
        if runtime_bytes > 512*1024**2:
            raise ValueError("compact runtime exceeds frozen per-input bound")
        total_bytes = sum(row["bytes"] for row in manifest.values())+runtime_bytes
        if total_bytes > 2*1024**3:
            raise ValueError("promoted inputs exceed frozen total bound")
        member_name = "inputs/"+runtime_input_name
        if member_name in manifest:
            raise ValueError("runtime input already present in frozen template inventory")
        promoted_manifest = dict(manifest)
        promoted_manifest[member_name] = {"sha256": runtime_sha256, "bytes": runtime_bytes}
        persistent.mkdir(parents=True, exist_ok=False)
        write_json(persistent/"protocol.json", protocol)
        write_json(persistent/"manifest.json", promoted_manifest)
        archive_path = persistent/"frozen-campaign.tar.gz"
        with tarfile.open(template/"frozen-campaign.tar.gz", "r:gz") as source, tarfile.open(archive_path, "w:gz") as target:
            for name in ("protocol.json", "manifest.json"):
                target.add(persistent/name, arcname=name, recursive=False)
            for member in source.getmembers():
                if member.name not in ("protocol.json", "manifest.json"):
                    target.addfile(member, source.extractfile(member))
            target.add(runtime_archive, arcname=member_name, recursive=False)
        shutil.copy2(template/"h34_campaign.sbatch", persistent/"h34_campaign.sbatch")
        intake = dict(parent_intake)
        intake.update(created_utc=datetime.now(timezone.utc).isoformat(),
            archive_sha256=sha256(archive_path), protocol_sha256=sha256(persistent/"protocol.json"),
            archive_bytes=archive_path.stat().st_size, source_files=len(promoted_manifest),
            uncompressed_input_bytes=total_bytes, status="FROZEN_NOT_SUBMITTED",
            parent_template_archive_sha256=parent_intake["archive_sha256"],
            native_runtime_promotion=runtime_receipt)
        write_json(persistent/"intake.json", intake)
        verified = work/"promoted"
        verified.mkdir()
        launcher.safe_extract(archive_path, verified)
        files = launcher.verify_source(verified)
        launcher.load_intake(persistent)
        proof = {"status": "PASS", "scope": "native runtime promotion only; no scientific result",
                 "parent_template_archive_sha256": parent_intake["archive_sha256"],
                 "archive_sha256": intake["archive_sha256"], "verified_files": files,
                 "all_original_source_and_actor_entries_unchanged": all(promoted_manifest[k] == v for k, v in manifest.items()),
                 "added_input": member_name, "runtime_sha_replacements": replacements,
                 "runtime_verification": runtime_receipt, "submissions": []}
        if submit:
            smokes = [job for job in protocol["jobs"] if job["kind"] == "smoke"]
            runs = [job for job in protocol["jobs"] if job["kind"] == "run"]
            if len(smokes) != 1 or len(runs) != 1:
                raise ValueError("automatic promotion requires exactly one smoke and one dependent run")
            smoke = launcher.submit(persistent, smokes[0]["name"])
            proof["submissions"].append(smoke)
            if smoke["status"] != "SUBMITTED":
                write_json(persistent/"promotion.json", proof)
                raise RuntimeError("fresh native runtime smoke submission failed")
            run = launcher.submit(persistent, runs[0]["name"], "afterok:"+smoke["job_id"])
            proof["submissions"].append(run)
            if run["status"] != "SUBMITTED":
                write_json(persistent/"promotion.json", proof)
                raise RuntimeError("dependent native runtime run submission failed")
        write_json(persistent/"promotion.json", proof)
        return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--runtime-archive-sha256", required=True)
    parser.add_argument("--persistent", type=Path, required=True)
    parser.add_argument("--runtime-input-name", default="runtime/orb-runtime.tar.gz")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    result = promote(args.template, args.runtime_archive, args.runtime_archive_sha256,
                     args.persistent, runtime_input_name=args.runtime_input_name, submit=args.submit)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
