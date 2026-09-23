"""Submit the small, source-frozen Mission 7 route handoff probe."""
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
    parser.add_argument("--stage", choices=("doors", "transport", "both"), required=True)
    parser.add_argument("--indices", required=True)
    # The `share` partition is heterogeneous: sandybridge through skylake, and
    # both el8 and el9.  Different ISA (AVX vs AVX2 vs AVX-512) and different
    # glibc change floating-point results, so a bare submission is not
    # comparable to the completed cn-c22 jobs.  Constrain to cn-c22's hardware
    # and OS class -- 8 nodes instead of 1 -- and keep --node for the exact
    # bitwise replay gate, where the single original physics node is the point.
    parser.add_argument("--node", default=None,
                        help="pin to one node; only needed for exact bitwise replay")
    parser.add_argument("--constraint", default="haswell&el8",
                        help="node feature constraint used when --node is not given")
    parser.add_argument("--allow-inactive-intervention", action="store_true",
                        help="let the probe record a requested-but-ineffective "
                             "intervention as a null result instead of failing")
    parser.add_argument("--handoff", choices=("switch", "early"), default="switch")
    parser.add_argument("--output-name", default="route-handoff-probe-cn-c22")
    parser.add_argument("--rejoin-diagnostic", action="store_true")
    parser.add_argument("--rejoin-fix", choices=("none", "forward_pulse", "prev_actions_reset"),
                        default="none")
    parser.add_argument("--chain-trace", action="store_true",
                        help="record the 25 Hz command-to-motion chain (Campaign A)")
    parser.add_argument("--stage-lateral", type=float, default=None,
                        help="PlateStage body-centre lateral offset (m); default plate centre")
    parser.add_argument("--stage-activate", action="store_true")
    parser.add_argument("--stage-press-hold", action="store_true")
    parser.add_argument("--stage-wait-open", type=float, default=None)
    parser.add_argument("--exit-ramp", type=float, default=0.)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if not campaign.is_relative_to(ROOT):
        parser.error("campaign must remain inside this repository")
    out = campaign / args.output_name
    snapshot = out / "source"
    if out.exists():
        parser.error(f"destination exists; preserve it and choose a new campaign: {out}")
    if not args.submit:
        print(json.dumps({"planned_output": str(out), "stage": args.stage,
                          "indices": args.indices, "node": args.node,
                          "constraint": None if args.node else args.constraint},
                         indent=2))
        return
    snapshot.mkdir(parents=True)
    files = list((ROOT / "src/bhl_robust/mission").glob("*.py"))
    files += [ROOT / name for name in (
        "src/bhl_robust/__init__.py", "src/bhl_robust/sensor_io.py",
        "src/bhl_robust/eval/__init__.py", "src/bhl_robust/eval/multi_robot.py",
        "src/bhl_robust/eval/mjcf_assets.py", "src/bhl_robust/eval/livery.py",
        "src/bhl_robust/eval/team_sensors.py", "slurm/mission7_route_handoff_probe.sbatch",
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
    placement = ([f"--nodelist={args.node}"] if args.node
                 else [f"--constraint={args.constraint}"])
    # Everything after the three paths is forwarded to the probe verbatim, so a
    # new probe flag needs no change here or in the sbatch.  The old eight
    # positional slots had to agree across all three files, and when they
    # stopped agreeing the job still ran and still exited 0.
    probe_args = ["--stage", args.stage, "--indices", args.indices,
                  "--handoff", args.handoff, "--rejoin-fix", args.rejoin_fix]
    if args.rejoin_diagnostic:
        probe_args.append("--rejoin-diagnostic")
    if args.chain_trace:
        probe_args.append("--chain-trace")
    if args.stage_lateral is not None:
        probe_args.append(f"--stage-lateral={args.stage_lateral}")
    if args.stage_activate:
        probe_args.append("--stage-activate")
    if args.stage_press_hold:
        probe_args.append("--stage-press-hold")
    if args.stage_wait_open is not None:
        probe_args.append(f"--stage-wait-open={args.stage_wait_open}")
    if args.exit_ramp:
        probe_args.append(f"--exit-ramp={args.exit_ramp}")
    if args.allow_inactive_intervention:
        probe_args.append("--allow-inactive-intervention")
    command = [
        "sbatch", "--parsable", "--job-name=m7-handoff-probe", "--account=eecs",
        "--partition=share", "--cpus-per-task=2", "--mem=12G", "--time=02:00:00",
        *placement, f"--chdir={ROOT}",
        f"--output={out}/m7-handoff-probe-%j.out",
        f"--error={out}/m7-handoff-probe-%j.out",
        str(snapshot / "slurm/mission7_route_handoff_probe.sbatch"),
        str(ROOT), str(snapshot), str(out), *probe_args,
    ]
    clean_env = {key: value for key, value in os.environ.items()
                 if not key.startswith("SLURM_") and key not in ("TMPDIR", "CUDA_VISIBLE_DEVICES")}
    receipt = subprocess.check_output(command, env=clean_env, text=True).strip()
    job_id = receipt.split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"unrecognized sbatch receipt: {receipt}")
    row = {
        "job_id": job_id,
        "status": "SUBMITTED",
        "stage": args.stage,
        "indices": args.indices,
        "handoff": args.handoff,
        "rejoin_diagnostic": args.rejoin_diagnostic,
        "rejoin_fix": args.rejoin_fix,
        "chain_trace": args.chain_trace,
        "stage_lateral_m": args.stage_lateral,
        "stage_activate": args.stage_activate,
        "stage_press_hold": args.stage_press_hold,
        "stage_wait_open_s": args.stage_wait_open,
        "exit_ramp_s": args.exit_ramp,
        "allow_inactive_intervention": args.allow_inactive_intervention,
        "requested_node": args.node,
        "requested_constraint": None if args.node else args.constraint,
        "probe_args": probe_args,
        "destination": str(out),
        "source_snapshot": str(snapshot),
        "source_sha256": hashes,
        "command": command,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "submitted_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (out / "submission.json").write_text(json.dumps(row, indent=2) + "\n")
    with (ROOT / "SLURM_JOBS.md").open("a") as stream:
        stream.write(
            f"\nMission7 route handoff probe (2026-09-22): **SUBMITTED** `{job_id}` — "
            f"{args.stage} layouts `{args.indices}`, "
            f"{'node `' + args.node + '`' if args.node else 'constraint `' + args.constraint + '`'}"
            f", 2 CPUs / 12 GB / 0 GPUs / 2 h; "
            f"unchanged PlateStage with `{args.handoff}` route handoff"
            f" and rejoin diagnostic `{args.rejoin_diagnostic}`, chain trace `{args.chain_trace}`, "
            f"fix `{args.rejoin_fix}`; "
            f"receipt/source hashes: `{out.relative_to(ROOT)}/submission.json`.\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
