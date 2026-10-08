"""Explicit, source-frozen Mission7 submissions. Defaults to a dry run.

Only smoke and the one-seed validation are initially submitted. Pilot, later
curriculum stages, and held-out evaluation require their measured JSON gates.
No existing jobs or results are modified.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("blind", "lidar", "stereo", "both")
STAGES = ("approach", "branches", "navigation", "doors", "transport")
BUDGETS = (400, 800, 1600, 1600, 1600)


def require_gate(path, kind, **fields):
    data = json.loads(path.read_text())
    if data.get("passed") is not True or data.get("kind") != kind or any(data.get(k) != v for k, v in fields.items()):
        raise SystemExit(f"Required measured gate has not passed: {path}")
    if kind == "policy_validation":
        if data.get("split") != "validation" or data.get("episode_count", 0) < 16:
            raise SystemExit("Validation evidence is incomplete")
        if hashlib.sha256(Path(data["checkpoint"]).read_bytes()).hexdigest() != data["checkpoint_sha256"]:
            raise SystemExit("Checkpoint no longer matches passed gate")
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("smoke", "validation", "pilot", "stage", "eval"))
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--after", help="numeric afterok scheduler prerequisite")
    p.add_argument("--arm", choices=ARMS, default="both")
    p.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--stage", choices=STAGES, default="approach")
    p.add_argument("--failure", choices=("normal", "lidar_missing", "stereo_missing", "both_missing",
                                        "lidar_stale", "stereo_stale", "noisy", "occluded", "intermittent"), default="normal")
    p.add_argument("--submit", action="store_true")
    a = p.parse_args()
    out = a.campaign.resolve()
    if not out.is_relative_to(ROOT):
        p.error("campaign must remain inside this repository")
    snapshot = out/"source"
    infrastructure = out/"cluster-smoke/smoke.json"
    if a.phase == "smoke":
        name, mode, destination = "m7-smoke", "smoke", out/"cluster-smoke"
        args = []
    elif a.phase == "validation":
        name, mode, destination = "m7-learn-both-s0", "train", out/"validation-both-s0"
        args = ["--arm", "both", "--seed", "0", "--stage", "approach", "--updates", "400",
                "--infrastructure-gate", str(infrastructure)]
        if not a.after:
            require_gate(infrastructure, "infrastructure_smoke")
    else:
        require_gate(infrastructure, "infrastructure_smoke")
        require_gate(out/"validation-both-s0/gate.json", "policy_validation", arm="both", seed=0, stage="approach")
        name = f"m7-{a.stage}-{a.arm}-s{a.seed}"
        destination = out/f"{a.stage}-{a.arm}-s{a.seed}"
        mode, args = "train", ["--stage", a.stage, "--arm", a.arm, "--seed", str(a.seed),
                                "--updates", str(BUDGETS[STAGES.index(a.stage)]),
                                "--infrastructure-gate", str(infrastructure)]
        if a.phase == "pilot":
            if a.stage != "approach" or a.seed != 0:
                p.error("pilot is a fresh approach training at seed 0")
        if a.phase in ("stage", "eval"):
            for arm in ARMS:
                require_gate(out/f"approach-{arm}-s0/gate.json", "policy_validation", arm=arm, seed=0, stage="approach")
        if a.phase == "stage" and a.stage != "approach":
            prior = STAGES[STAGES.index(a.stage)-1]
            gate_path = out/f"{prior}-{a.arm}-s{a.seed}/gate.json"
            require_gate(gate_path, "policy_validation", arm=a.arm, seed=a.seed, stage=prior)
            args += ["--previous-gate", str(gate_path)]
        if a.phase == "eval":
            gate = require_gate(destination/"gate.json", "policy_validation", arm=a.arm, seed=a.seed, stage=a.stage)
            mode, name = "eval", name+"-"+a.failure
            destination = out/f"heldout-{a.stage}-{a.arm}-s{a.seed}-{a.failure}.json"
            args = ["--stage", a.stage, "--arm", a.arm, "--seed", str(a.seed),
                    "--checkpoint", gate["checkpoint"], "--split", "test", "--episodes", "64", "--failure", a.failure]
    if destination.exists():
        p.error("destination exists; preserve it and use a new campaign for a retry")
    cmd = ["sbatch", "--parsable", f"--job-name={name}", "--account=eecs", "--partition=share",
           "--cpus-per-task=2", "--mem=12G", "--time="+("00:30:00" if a.phase == "smoke" else "04:00:00"),
           f"--chdir={ROOT}", f"--output={out}/%x-%j.out", f"--error={out}/%x-%j.out"]
    if a.after:
        if not a.after.isdigit():
            p.error("dependency ID must be numeric")
        cmd += [f"--dependency=afterok:{a.after}", "--kill-on-invalid-dep=yes"]
    cmd += [str(snapshot/"slurm/mission7_cpu.sbatch"), str(ROOT), str(snapshot), mode,
            "--out", str(destination), *args]
    print(shlex.join(cmd), flush=True)
    if not a.submit:
        return
    out.mkdir(parents=True, exist_ok=True)
    receipts = out/"submissions.jsonl"
    if receipts.exists():
        for line in receipts.read_text().splitlines():
            if json.loads(line)["destination"] == str(destination):
                raise SystemExit("this destination was already submitted; do not duplicate pending jobs")
    if not snapshot.exists():
        snapshot.mkdir()
        files = list((ROOT/"src/bhl_robust/mission").glob("*.py")) + [ROOT/path for path in (
            "src/bhl_robust/__init__.py", "src/bhl_robust/sensor_io.py", "src/bhl_robust/eval/__init__.py",
            "src/bhl_robust/eval/multi_robot.py", "src/bhl_robust/eval/mjcf_assets.py",
            "src/bhl_robust/eval/livery.py", "src/bhl_robust/eval/team_sensors.py",
            "scripts/mission7.py", "slurm/mission7_cpu.sbatch")]
        hashes = {}
        for source in files:
            relative = source.relative_to(ROOT)
            target = snapshot/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
        (snapshot/"sha256.json").write_text(json.dumps(hashes, indent=2)+"\n")
    clean = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")
             and k not in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "TMPDIR")}
    receipt = subprocess.check_output(cmd, env=clean, text=True).strip()
    job = receipt.split(";")[0]
    if not job.isdigit():
        raise RuntimeError("unrecognized sbatch receipt: "+receipt)
    row = {"job_id": job, "phase": a.phase, "destination": str(destination),
           "dependency": "afterok:"+a.after if a.after else None, "array": None,
           "gpus": 0, "cpus": 2, "source_snapshot": str(snapshot), "command": cmd,
           "submitted_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    with receipts.open("a") as stream:
        stream.write(json.dumps(row)+"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print("SUBMITTED_JOB_ID="+job, flush=True)


if __name__ == "__main__":
    main()
