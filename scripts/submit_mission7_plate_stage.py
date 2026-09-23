"""Submit the source-frozen exact ten-fall replay gate (staged plate maneuver).

The gate is the documented 10/10 upright regression: unchanged geometry,
recorded activation schedule and fall predicate over the ten retained falls.
Any change to PlateStage must be re-run through it.  Default arguments
reproduce job 21397732; probe flags after the paths are forwarded verbatim.
"""
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
DEFAULT_CAMPAIGN = "results/mission7-replay-smoke-20260921"
DEFAULT_BASELINE = "results/mission7-approach-followup-20260922/replay-diagnose-cn-c22"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-campaign", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--campaign", type=Path, default=Path(DEFAULT_CAMPAIGN))
    parser.add_argument("--baseline", type=Path, default=Path(DEFAULT_BASELINE))
    parser.add_argument("--stage-lateral", type=float, default=None)
    parser.add_argument("--wait-open", type=float, default=0.)
    parser.add_argument("--press-hold", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--node", default="cn-c22",
                        help="the replay gate is bitwise; it stays pinned to the original physics node")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    out_campaign = args.out_campaign.resolve()
    if not out_campaign.is_relative_to(ROOT):
        parser.error("out-campaign must remain inside this repository")
    out = out_campaign / args.output_name
    snapshot = out / "source"
    if out.exists():
        parser.error(f"destination exists; choose a new output name: {out}")
    campaign = (ROOT / args.campaign).resolve()
    baseline = (ROOT / args.baseline).resolve()
    for path, label in ((campaign / "fullroute/legacy-doors.json", "campaign"), (baseline / "result.json", "baseline")):
        if not path.exists():
            parser.error(f"{label} missing: {path}")
    probe_args = []
    if args.stage_lateral is not None:
        probe_args.append(f"--stage-lateral={args.stage_lateral}")
    if args.wait_open:
        probe_args.append(f"--wait-open={args.wait_open}")
    if args.press_hold:
        probe_args.append("--press-hold")
    if args.smoke:
        probe_args.append("--smoke")
    if not args.submit:
        print(json.dumps({"planned_output": str(out), "campaign": str(campaign), "baseline": str(baseline),
                          "probe_args": probe_args, "node": args.node}, indent=2))
        return
    snapshot.mkdir(parents=True)
    files = list((ROOT / "src/bhl_robust/mission").glob("*.py"))
    files += [ROOT / name for name in (
        "src/bhl_robust/__init__.py", "src/bhl_robust/sensor_io.py",
        "src/bhl_robust/eval/__init__.py", "src/bhl_robust/eval/multi_robot.py",
        "src/bhl_robust/eval/mjcf_assets.py", "src/bhl_robust/eval/livery.py",
        "src/bhl_robust/eval/team_sensors.py", "slurm/mission7_plate_stage.sbatch",
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
        "sbatch", "--parsable", "--job-name=m7-replay-gate", "--account=eecs",
        "--partition=share", "--cpus-per-task=2", "--mem=12G", "--time=02:00:00",
        f"--nodelist={args.node}", f"--chdir={ROOT}",
        f"--output={out}/m7-replay-gate-%j.out", f"--error={out}/m7-replay-gate-%j.out",
        str(snapshot / "slurm/mission7_plate_stage.sbatch"),
        str(ROOT), str(snapshot), str(campaign), str(baseline), str(out), *probe_args,
    ]
    clean_env = {key: value for key, value in os.environ.items()
                 if not key.startswith("SLURM_") and key not in ("TMPDIR", "CUDA_VISIBLE_DEVICES")}
    receipt = subprocess.check_output(command, env=clean_env, text=True).strip()
    job_id = receipt.split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"unrecognized sbatch receipt: {receipt}")
    row = {"job_id": job_id, "status": "SUBMITTED", "stage_lateral_m": args.stage_lateral, "wait_open_s": args.wait_open, "smoke": args.smoke,
           "requested_node": args.node, "campaign": str(campaign), "baseline": str(baseline),
           "destination": str(out), "source_snapshot": str(snapshot), "source_sha256": hashes,
           "probe_args": probe_args, "command": command,
           "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
           "submitted_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    (out / "submission.json").write_text(json.dumps(row, indent=2) + "\n")
    with (ROOT / "SLURM_JOBS.md").open("a") as stream:
        stream.write(
            f"\nMission7 exact replay gate (2026-09-23): **SUBMITTED** `{job_id}` — ten-fall staged replay pinned to "
            f"`{args.node}`, PlateStage lateral `{'plate centre' if args.stage_lateral is None else f'{args.stage_lateral} m'}`, wait-open `{args.wait_open} s`; "
            f"geometry, activation schedule and fall predicate unchanged; receipt/source hashes: "
            f"`{out.relative_to(ROOT)}/submission.json`.\n")
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
