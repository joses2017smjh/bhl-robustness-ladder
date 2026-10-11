"""Submit an already frozen H1 archive; refuse duplicate submission receipts.

The persistent directory must hold intake.json, protocol.json, the frozen
archive and h1_navigation.sbatch. All hashes are checked before sbatch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--persistent-dir", type=Path, required=True)
    a = ap.parse_args()
    root = a.persistent_dir.resolve()
    intake = json.loads((root / "intake.json").read_text())
    for name, key in (("frozen-campaign.tar.gz", "archive_sha256"),
                      ("protocol.json", "protocol_sha256"), ("h1_navigation.sbatch", "launcher_sha256")):
        if digest(root / name) != intake[key]:
            raise ValueError(f"frozen submission input mismatch: {name}")
    command = ["sbatch", "--parsable", f"--output={root}/slurm-%j.out", f"--error={root}/slurm-%j.out",
               str(root / "h1_navigation.sbatch"), str(root / "frozen-campaign.tar.gz"),
               intake["archive_sha256"], str(root)]
    # Exclusive reservation precedes the scheduler mutation, so a repeated
    # invocation cannot submit a second campaign accidentally.
    with (root / "submission-start.json").open("x") as stream:
        json.dump({"created_utc": datetime.now(timezone.utc).isoformat(), "command": command}, stream, indent=2)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    proc = subprocess.run(command, env=env, capture_output=True, text=True)
    match = re.fullmatch(r"(\d+)(?:;[^\n]+)?\n?", proc.stdout)
    receipt = {"submitted_utc": datetime.now(timezone.utc).isoformat(), "command": command,
               "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
               "job_id": match[1] if match and proc.returncode == 0 else None,
               "protocol_sha256": intake["protocol_sha256"], "archive_sha256": intake["archive_sha256"],
               "launcher_sha256": intake["launcher_sha256"], "submitter_sha256": digest(__file__),
               "resources": {"partition": "share", "constraint": "el9", "cpus": 2, "memory_gb": 8,
                             "wall_time_h": 12, "gpus": 0, "episodes": 576, "max_evaluators": 2},
               "status": "SUBMITTED" if match and proc.returncode == 0 else "SUBMISSION_FAILED",
               "scope": "Queued/running is not a completed experiment. Fresh matched CPU NavGym confirmation; no training/Mission 7/hardware."}
    with (root / "submission.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["job_id"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
