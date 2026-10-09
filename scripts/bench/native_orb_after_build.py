#!/usr/bin/env python3
"""Promote a genuinely successful pinned ORB build into predeclared campaigns.

This CPU orchestration job never reads estimator outcomes or changes scientific
parameters. Its immutable plan pins the build and both already frozen templates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile

PIN = "4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4"
MAX_RUNTIME_BYTES = 600 * 1024 * 1024


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def regular_name(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or ".." in p.parts or str(p) != name:
        raise ValueError("unsafe artifact name")
    return p


def verify_file(path, row):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
        raise ValueError("artifact hash/size mismatch: " + str(path))


def verify_template(row):
    root = Path(row["template"])
    for name, digest in row["pins"].items():
        regular_name(name)
        if sha(root / name) != digest:
            raise ValueError("frozen template changed: " + name)
    protocol = json.loads((root / "protocol.json").read_text())
    if "__NATIVE_ORB_RUNTIME_SHA256__" not in json.dumps(protocol):
        raise ValueError("template must retain the exact deferred runtime token")
    if Path(row["persistent"]).exists():
        raise ValueError("campaign destination already exists; refusing duplicate submission")


def verify_build(plan):
    freeze = Path(plan["build_freeze"])
    intake = json.loads((freeze / "intake.json").read_text())
    for name, key in (("frozen-campaign.tar.gz", "archive_sha256"), ("protocol.json", "protocol_sha256")):
        if intake[key] != plan[key] or sha(freeze / name) != plan[key]:
            raise ValueError("build freeze changed")
    root = freeze / plan["build_result_directory"]
    protocol = json.loads((freeze / "protocol.json").read_text())
    jobs = protocol.get("jobs", [])
    if len(jobs) != 1 or jobs[0].get("entrypoint") not in {
            "scripts/bench/native_build_campaign.py", "scripts/bench/native_orb_relink_campaign.py"}:
        raise ValueError("expected one frozen native build job")
    if root.name != jobs[0]["name"] + "-" + str(plan["build_job_id"]):
        raise ValueError("unexpected actual build attempt")
    completion = json.loads((root / "completion.json").read_text())
    if completion.get("status") != "PASS" or completion.get("exit_status") != 0 or completion.get("error") is not None:
        raise ValueError("actual native build did not complete successfully")
    if completion["archive_sha256"] != plan["archive_sha256"]:
        raise ValueError("completion belongs to another frozen build")
    for name, row in completion["files"].items():
        if len(regular_name(name).parts) != 1:
            raise ValueError("completion member must be a basename")
        verify_file(root / name, row)
    result = json.loads((root / "campaign_result.json").read_text())
    if result.get("status") != "PASS" or result.get("method") != "orb" or result.get("returncode") != 0:
        raise ValueError("native build runner did not report genuine ORB success")
    if "outputs.tar.gz" not in completion["files"] or "campaign_result.json" not in completion["files"]:
        raise ValueError("completed build is missing required output receipts")
    return root, result, completion


def compact_runtime(build_root, result, destination):
    """Verify every actual build runtime byte, then retain the complete pack."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=False)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    expected = result["runtime_files"]
    if not expected or sum(row["bytes"] for row in expected.values()) > MAX_RUNTIME_BYTES:
        raise ValueError("runtime receipt exceeds predeclared compact limit")
    seen = set()
    with tarfile.open(Path(build_root) / "outputs.tar.gz", "r:gz") as source, tarfile.open(temporary, "w:gz") as target:
        members = source.getmembers()
        if len(members) > 20000 or sum(m.size for m in members) > 512 * 1024 * 1024:
            raise ValueError("build output exceeds its predeclared cap")
        names = set()
        for member in members:
            name = regular_name(member.name)
            if not member.isfile() or member.name in names:
                raise ValueError("build archive contains duplicate or non-file entries")
            names.add(member.name)
            if name.parts[0] != "runtime":
                continue
            relative = str(PurePosixPath(*name.parts[1:]))
            if relative not in expected or relative in seen:
                raise ValueError("actual runtime inventory differs from build receipt")
            row = expected[relative]
            if member.size != row["bytes"]:
                raise ValueError("actual runtime member size mismatch")
            stream = source.extractfile(member)
            h = hashlib.sha256()
            # Tar addfile streams from this bounded wrapper without duplicating
            # the vocabulary/source/dependency archive in durable storage.
            class Reader:
                def read(self, count=-1):
                    value = stream.read(count)
                    h.update(value)
                    return value
            target.addfile(member, Reader())
            if h.hexdigest() != row["sha256"]:
                raise ValueError("actual runtime member hash mismatch: " + relative)
            seen.add(relative)
        if seen != set(expected):
            raise ValueError("actual runtime file is missing")
        receipt = json.load(source.extractfile("runtime/runtime.json"))
        required = {"bin/orb_native", "ORBvoc.txt", "ORB-SLAM3-LICENSE.txt", "upstream-source.tar.gz", "ldd-runtime.txt"}
        if not required <= seen or receipt.get("schema") != "bhl-native-orb-runtime-v1" or receipt.get("status") != "PASS":
            raise ValueError("complete native ORB runtime required")
        if receipt.get("upstream_commit") != PIN or receipt.get("native_algorithm_sources_modified") is not False or receipt.get("headless_visualization_only") is not True:
            raise ValueError("native source contract differs from predeclaration")
        if set(receipt["files_sha256"]) != seen - {"runtime.json"}:
            raise ValueError("runtime manifest does not cover every retained file")
        for name, digest in receipt["files_sha256"].items():
            if digest != expected[name]["sha256"]:
                raise ValueError("runtime/build manifests disagree")
        if PIN not in receipt.get("version_smoke", ""):
            raise ValueError("actual executable version smoke was not retained")
    temporary.rename(destination)
    return {"path": str(destination), "sha256": sha(destination), "bytes": destination.stat().st_size,
            "runtime_files": len(seen), "uncompressed_runtime_bytes": sum(row["bytes"] for row in expected.values()),
            "runtime_binary_sha256": expected["bin/orb_native"]["sha256"],
            "vocabulary_sha256": expected["ORBvoc.txt"]["sha256"], "upstream_commit": PIN}


def smoke_actual_runtime(archive, plan):
    """Repeat the binary/ldd checks inside the exact pinned Ubuntu SIF."""
    sif = Path(plan["sif"])
    if sha(sif) != plan["sif_sha256"]:
        raise ValueError("runtime SIF changed")
    with tempfile.TemporaryDirectory(prefix="bhl-orb-promotion-", dir="/tmp") as tmp:
        work = Path(tmp)
        with tarfile.open(archive) as source:
            for member in source:
                name = regular_name(member.name)
                if not member.isfile() or name.parts[0] != "runtime":
                    raise ValueError("invalid compact native runtime")
                path = work / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as stream, path.open("xb") as out:
                    shutil.copyfileobj(stream, out)
                path.chmod(member.mode & 0o777)
        home = work / "home"; home.mkdir()
        binary = str(work / "runtime/bin/orb_native")
        cmd = ["apptainer", "exec", "--containall", "--cleanenv", "--home", str(home),
               "--bind", plan["share_root"] + ":" + plan["share_root"] + ":ro",
               "--bind", str(work) + ":" + str(work), "--env", "LD_LIBRARY_PATH=" + str(work / "runtime/lib"), str(sif)]
        env = {k: v for k, v in os.environ.items() if not k.startswith("APPTAINERENV_")}
        version = subprocess.check_output(cmd + [binary, "--version"], env=env, text=True, timeout=60).strip()
        ldd = subprocess.check_output(cmd + ["ldd", binary], env=env, text=True, timeout=60)
        if PIN not in version or "not found" in ldd:
            raise ValueError("actual compact executable smoke failed")
        return {"version": version, "ldd": ldd, "scope": "Actual native executable only; no estimated trajectories or scientific threshold changes"}


def promote(plan, output):
    if plan.get("schema") != "bhl-orb-post-build-plan-v1" or len(plan["campaigns"]) != 2:
        raise ValueError("exactly the two predeclared campaign templates are required")
    # Verify both templates before creating or submitting either campaign.
    for row in plan["campaigns"]:
        verify_template(row)
    build_root, result, completion = verify_build(plan)
    runtime = compact_runtime(build_root, result, plan["runtime_archive"])
    smoke = smoke_actual_runtime(runtime["path"], plan)
    report = {"schema": "bhl-orb-post-build-receipt-v1", "status": "INCOMPLETE", "runtime": runtime,
              "actual_smoke": smoke, "build_job_id": plan["build_job_id"], "build_output_archive_sha256": completion["files"]["outputs.tar.gz"]["sha256"],
              "scientific_status": "NO_RESULTS_READ_NO_PARAMETERS_TUNED", "campaigns": []}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    helper = Path(__file__).with_name("promote_native_runtime_campaign.py")
    for row in plan["campaigns"]:
        cmd = [sys.executable, str(helper), "--template", row["template"], "--runtime-archive", runtime["path"],
               "--runtime-archive-sha256", runtime["sha256"], "--persistent", row["persistent"],
               "--runtime-input-name", "runtime/orb-runtime.tar.gz", "--submit"]
        process = subprocess.run(cmd, capture_output=True, text=True)
        report["campaigns"].append({"template": row["template"], "persistent": row["persistent"], "returncode": process.returncode,
                                    "stdout": process.stdout, "stderr": process.stderr})
        (output / "promotion-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        if process.returncode:
            raise RuntimeError("predeclared campaign promotion failed; inspect retained receipt")
    report["status"] = "PASS"
    (output / "promotion-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    (Path(runtime["path"]).parent / "promotion-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    args = p.parse_args(); output = Path(os.environ["H34_OUTPUT_DIR"])
    try:
        report = promote(json.loads(args.plan.read_text()), output)
        result = {"status": "PASS", "scientific_status": "ORCHESTRATION_ONLY", "runtime_archive_sha256": report["runtime"]["sha256"],
                  "campaigns_promoted": len(report["campaigns"])}
    except Exception as error:
        result = {"status": "INCOMPLETE", "error": str(error), "scientific_status": "NO_ESTIMATOR_RESULT_INFERRED"}
    (output / "campaign_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
