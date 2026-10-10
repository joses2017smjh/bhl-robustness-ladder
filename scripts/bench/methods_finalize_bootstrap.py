#!/usr/bin/env python3
"""Unpack one pinned collector snapshot into node scratch and collect four tracks."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import tarfile


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(path, expected):
    if digest(path) != expected:
        raise ValueError("bootstrap plan changed")
    plan = json.loads(path.read_text())
    if (plan["schema"] != "bhl-methods-finalization-bootstrap-v1" or
            digest(Path(__file__)) != plan["bootstrap_sha256"]):
        raise ValueError("bootstrap source changed")
    rows = plan["collections"]
    if len(rows) != 4 or {row["track"] for row in rows} != {"terrain", "stereo", "sensors", "perceptive"}:
        raise ValueError("exactly four distinct declared collection tracks required")
    for row in rows:
        child = Path(row["plan_path"])
        if digest(child) != row["plan_sha256"] or json.loads(child.read_text())["track"] != row["track"]:
            raise ValueError("individual collection plan identity changed")
    source = Path(plan["source_archive"])
    if digest(source) != plan["source_archive_sha256"]:
        raise ValueError("collector source archive changed")
    results = []
    with tempfile.TemporaryDirectory(prefix="bhl-finalize-source-", dir="/tmp") as folder:
        work = Path(folder)
        with tarfile.open(source, "r:gz") as archive:
            members = archive.getmembers()
            names = set()
            if len(members) > 5000 or sum(m.size for m in members) > 50*1024**2:
                raise ValueError("collector source archive exceeds cap")
            for member in members:
                relative = PurePosixPath(member.name)
                if (not member.isfile() or member.name in names or relative.is_absolute() or
                        ".." in relative.parts or "\\" in member.name or str(relative) != member.name):
                    raise ValueError("unsafe collector source member")
                names.add(member.name)
            for member in members:
                target = work/member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
        for row in plan["collections"]:
            original = Path(row["plan_path"])
            if digest(original) != row["plan_sha256"]:
                raise ValueError("individual collection plan changed")
            local = work/(row["track"]+"-plan.json")
            shutil.copyfile(original, local)
            environment = {**os.environ, "TMPDIR": "/tmp", "PYTHONDONTWRITEBYTECODE": "1",
                           "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
            completed = subprocess.run([plan["python"], str(work/"scripts/bench/methods_finalize.py"),
                "--plan", str(local), "--plan-sha256", row["plan_sha256"]], env=environment)
            results.append({"track": row["track"], "returncode": completed.returncode})
    receipt = {"plan_sha256": expected, "job_id": os.getenv("SLURM_JOB_ID"), "collections": results,
               "status": "PASS" if all(r["returncode"] == 0 for r in results) else "INCOMPLETE"}
    with (path.parent/"bootstrap-result.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.plan, args.plan_sha256))
