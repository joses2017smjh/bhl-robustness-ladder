#!/usr/bin/env python3
"""Collect completed native replay evidence without rerunning an estimator."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import tarfile

import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_run(run, rows, receipt):
    method = run["method"]
    key = "T_W_C" if method == "orb" else "T_W_I"
    if method not in {"orb", "lio"} or receipt.get("ground_truth_inputs") != []:
        raise ValueError("truth-free native estimator receipt required")
    times = []
    for row in rows:
        if type(row["tracked"]) is not bool or (not row["tracked"] and row[key] is not None):
            raise ValueError("native lost states must retain explicit null poses")
        if row["tracked"]:
            pose = np.asarray(row[key], float)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError("invalid native tracked pose")
        stamp, seconds = float(row["timestamp_s"]), float(row["compute_seconds"])
        if not math.isfinite(stamp) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid native timing")
        times.append(stamp)
    if not rows or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("strictly increasing native timestamps required")
    if len(rows) != run["frames"] or sum(r["tracked"] for r in rows) != run["tracked_frames"]:
        raise ValueError("native frame counts differ from scientific result")
    measurement = run["measurement"]
    observed = 1000 * float(np.quantile([r["compute_seconds"] for r in rows], .95))
    if not math.isclose(observed, measurement["native_compute_p95_ms"], rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError("native timing differs from scientific result")
    for segment in measurement["segments"]:
        metrics = segment["metrics"]
        if metrics and metrics.get("scale") != 1.0:
            raise ValueError("metric scale must remain one")
    return {"sequence": run["sequence"], "split": run["split"], "method": method,
            "frames": len(rows), "tracked_frames": sum(r["tracked"] for r in rows),
            "measurement": measurement}


def collect(campaign_dir, job_name, output, allocation_step=None):
    campaign_dir, output = Path(campaign_dir).resolve(), Path(output).resolve()
    intake = json.loads((campaign_dir / "intake.json").read_text())
    submission_path = Path(allocation_step) if allocation_step else campaign_dir / (job_name + "-submission.json")
    submission = json.loads(submission_path.read_text())
    expected_status = "ALLOCATION_STEP_COMPLETE" if allocation_step else "SUBMITTED"
    if submission["status"] != expected_status or submission["archive_sha256"] != intake["archive_sha256"]:
        raise ValueError("submitted source pin differs")
    if allocation_step and (submission["job_name"] != job_name or submission["returncode"] != 0):
        raise ValueError("successful recorded allocation step required")
    job_root = campaign_dir / (job_name + "-" + submission["job_id"])
    completion = json.loads((job_root / "completion.json").read_text())
    launch = json.loads((job_root / "launch.json").read_text())
    if completion["status"] not in {"PASS", "NEGATIVE"} or completion["exit_status"] != 0:
        raise ValueError("completed valid native replay required")
    if any(r["archive_sha256"] != intake["archive_sha256"] for r in (launch, completion)):
        raise ValueError("completed source pin differs")
    if str(launch["job_id"]) != submission["job_id"] or launch["job"]["name"] != job_name:
        raise ValueError("completed job identity differs")
    for name, row in completion["files"].items():
        if PurePosixPath(name).name != name or name in {".", ".."}:
            raise ValueError("unsafe completion evidence name")
        p = job_root / name
        if p.is_symlink() or p.stat().st_size != row["bytes"] or digest(p) != row["sha256"]:
            raise ValueError("completion evidence hash differs: " + name)
    runs = []
    with tarfile.open(job_root / "outputs.tar.gz") as archive:
        members = archive.getmembers()
        if len({m.name for m in members}) != len(members) or any(not m.isfile() for m in members):
            raise ValueError("duplicate or non-file raw evidence")
        campaign = json.load(archive.extractfile("campaign_result.json"))
        if campaign != json.loads((job_root / "campaign_result.json").read_text()):
            raise ValueError("outer scientific result differs from raw archive")
        if campaign["schema"] != "bhl-native-slam-campaign-v1" or campaign["closed_loop_navigation_episodes"] != 0:
            raise ValueError("native replay campaign required")
        for run in campaign["runs"]:
            prefix = str(PurePosixPath(run["receipt"]).parent)
            rows = [json.loads(line) for line in archive.extractfile(prefix + "/native_frames.jsonl")]
            receipt = json.load(archive.extractfile(run["receipt"]))
            if json.load(archive.extractfile(prefix + "/metrics.json")) != run["measurement"]:
                raise ValueError("raw metrics differ from scientific result")
            runs.append(audit_run(run, rows, receipt))
    scientific = "SMOKE_ONLY" if campaign["phase"] == "smoke" else campaign["status"]
    result = {"schema": "bhl-native-slam-collection-v1", "audit_status": "PASS",
              "scientific_status": scientific, "runs": runs,
              "experiment_source_sha256": intake["archive_sha256"],
              "raw_archive_sha256": completion["files"]["outputs.tar.gz"]["sha256"],
              "scope": "Actual native simulated replay; smoke excluded; zero closed-loop episodes or hardware claims"}
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Actual native estimator replay", "", "Scientific status: **" + scientific + "**.", "",
             "Smoke is excluded from scientific qualification. These are simulated replay measurements, with zero closed-loop navigation episodes.", "",
             "|Scene|Method|Tracked / frames|Post-init tracking|ATE RMSE (m)|Native p95 (ms)|Readiness|",
             "|---|---|---:|---:|---:|---:|---|"]
    for run in runs:
        m = run["measurement"]
        metrics = m["segments"][0]["metrics"] if len(m["segments"]) == 1 else None
        ate = f"{metrics['ate_translation_m']['rmse']:.4f}" if metrics else "unscorable"
        lines.append(f"|{run['sequence']}|{run['method']}|{run['tracked_frames']}/{run['frames']}|{m['post_initialization_tracked_fraction']:.3f}|{ate}|{m['native_compute_p95_ms']:.2f}|{m['replay_readiness_gate']}|")
    lines.extend(["", "ATE uses independent evaluator-only rigid alignment with scale fixed to one. Native compute excludes capture, validation/export, process startup and imposed replay waits; it is distinct from client wall latency.", "",
                  "Raw archive SHA256: `" + result["raw_archive_sha256"] + "`.", ""])
    (output / "report.md").write_text("\n".join(lines))
    evidence = output / "evidence"
    evidence.mkdir()
    for name in ["launch.json", "completion.json", "campaign_result.json"]:
        shutil.copyfile(job_root / name, evidence / name)
    shutil.copyfile(submission_path, evidence / ("allocation-step.json" if allocation_step else "submission.json"))
    result_receipt = {k: v for k, v in result.items() if k != "runs"}
    result_receipt["files"] = {str(p.relative_to(output)): {"sha256": digest(p), "bytes": p.stat().st_size}
                               for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "collection-receipt.json").write_text(json.dumps(result_receipt, indent=2) + "\n")
    return result_receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allocation-step", type=Path)
    args = parser.parse_args()
    print(json.dumps(collect(args.campaign_dir, args.job_name, args.output, args.allocation_step), indent=2))
