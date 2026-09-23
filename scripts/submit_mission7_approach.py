"""Submit a source-frozen privileged Approach evaluation (world -x workstream)."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--controller", choices=("recovery", "guard", "pulse", "recovery030", "settle14", "settle16", "center"),
                        default="recovery")
    parser.add_argument("--directions", default="", help="';'-separated subset; default all four")
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--node", default=None)
    parser.add_argument("--constraint", default="haswell&el8")
    parser.add_argument("--time", default="02:00:00")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if not campaign.is_relative_to(ROOT):
        parser.error("campaign must remain inside this repository")
    out = campaign / args.output_name
    snapshot = out / "source"
    if out.exists():
        parser.error(f"destination exists; choose a new output name: {out}")
    probe_args = ["--controller", args.controller]
    if args.directions:
        probe_args.append(f"--directions={args.directions}")
    if args.controls:
        probe_args.append("--controls")
    if args.smoke:
        probe_args.append("--smoke")
    placement = [f"--nodelist={args.node}"] if args.node else [f"--constraint={args.constraint}"]
    if not args.submit:
        print(json.dumps({"planned_output": str(out), "probe_args": probe_args,
                          "placement": placement}, indent=2))
        return
    snapshot.mkdir(parents=True)
    files = list((ROOT / "src/bhl_robust/mission").glob("*.py"))
    files += [ROOT / name for name in (
        "src/bhl_robust/__init__.py", "src/bhl_robust/sensor_io.py",
        "src/bhl_robust/eval/__init__.py", "src/bhl_robust/eval/multi_robot.py",
        "src/bhl_robust/eval/mjcf_assets.py", "src/bhl_robust/eval/livery.py",
        "src/bhl_robust/eval/team_sensors.py", "slurm/mission7_approach_followup.sbatch",
    )]
    files += list((ROOT / "scripts").glob("mission7*.py"))
    hashes = {}
    for source in files:
        relative = source.relative_to(ROOT)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    (snapshot / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    command = [
        "sbatch", "--parsable", f"--job-name=m7-approach-{args.controller}", "--account=eecs",
        "--partition=share", "--cpus-per-task=2", "--mem=12G", f"--time={args.time}",
        *placement, f"--chdir={ROOT}",
        f"--output={out}/m7-approach-%j.out", f"--error={out}/m7-approach-%j.out",
        str(snapshot / "slurm/mission7_approach_followup.sbatch"),
        str(ROOT), str(snapshot), str(campaign), str(out), *probe_args,
    ]
    clean_env = {key: value for key, value in os.environ.items()
                 if not key.startswith("SLURM_") and key not in ("TMPDIR", "CUDA_VISIBLE_DEVICES")}
    receipt = subprocess.check_output(command, env=clean_env, text=True).strip()
    job_id = receipt.split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"unrecognized sbatch receipt: {receipt}")
    row = {
        "job_id": job_id, "status": "SUBMITTED", "controller": args.controller,
        "directions": args.directions or "all", "controls": args.controls, "smoke": args.smoke,
        "requested_node": args.node, "requested_constraint": None if args.node else args.constraint,
        "destination": str(out), "source_snapshot": str(snapshot), "source_sha256": hashes,
        "probe_args": probe_args, "command": command,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "submitted_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (out / "submission.json").write_text(json.dumps(row, indent=2) + "\n")
    with (ROOT / "SLURM_JOBS.md").open("a") as stream:
        stream.write(
            f"\nMission7 privileged Approach arm (2026-09-23): **SUBMITTED** `{job_id}` — "
            f"controller `{args.controller}`, directions `{args.directions or 'all'}`, "
            f"{'node `' + args.node + '`' if args.node else 'constraint `' + args.constraint + '`'}"
            f", 2 CPUs / 12 GB / 0 GPUs / {args.time}; geometry, spawn and predicates unchanged; "
            f"receipt/source hashes: `{out.relative_to(ROOT)}/submission.json`.\n")
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
