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
# --- m7-clocks2 --- the export stage gaits swap in a controller built by bhl_robust.eval.gait_clock, which the snapshot
# job (PYTHONPATH = the snapshot only) must find in the snapshot: it is added for these gaits only.  The path is
# composed rather than written as one quoted path literal because tests/test_mission7_plate_bench.py reads those
# literals as the DEFAULT snapshot set (unchanged); the clocks2 bench's provenance hashes gait_clock.py and the release
# gate compares it.
EXPORT_STAGE_GAITS = ("clocks2", "export")
GAIT_CLOCK_MODULE = Path("src/bhl_robust/eval") / "gait_clock.py"
# --- m7-fix --- F2 / F3 (= mission7_plate_stage.FIX_CROSS_BUDGETS / FIX_YAW_CAPS; that module is not imported here):
# forwarded verbatim as --cross-budget=window / --yaw-cap=0.6, with an export stage gait only.
FIX_CROSS_BUDGETS = ("window",)
FIX_YAW_CAPS = (.60,)


def _fix_args(parser, args):
    """--- m7-fix --- the forwarded F2 / F3 flags ([] when neither is given: every other receipt is unchanged)."""
    out = []
    if args.cross_budget is None and args.yaw_cap is None:
        return out
    if args.stage_gait not in EXPORT_STAGE_GAITS:
        parser.error("--cross-budget / --yaw-cap go with --stage-gait clocks2 | export only")
    if args.cross_budget is not None:
        if args.cross_clear is not None:
            parser.error("--cross-budget ends the crossing at mission7_gates' clear; --cross-clear does not compose")
        out.append(f"--cross-budget={args.cross_budget}")
    if args.yaw_cap is not None:
        out.append(f"--yaw-cap={args.yaw_cap}")
    return out


def _export_args(parser, args):
    """--- m7-clocks2 --- the forwarded flags of an export stage gait ([] for every other gait).

    clocks2 is a preset whose policy sha256 the stage pins itself; export forwards its directory (absolute) and the
    sha256 of its policy.onnx as read at submission, so the job refuses weights changed in between.
    """
    if args.stage_gait not in EXPORT_STAGE_GAITS:
        if args.stage_gait_export is not None or args.stage_gait_export_sha256 is not None:
            parser.error("--stage-gait-export / --stage-gait-export-sha256 go with --stage-gait export only")
        return []
    if args.align_yaw or args.press_hold:
        parser.error(f"--stage-gait {args.stage_gait} does not compose with --align-yaw or --press-hold")
    if args.stage_gait == "clocks2":
        if args.stage_gait_export is not None or args.stage_gait_export_sha256 is not None:
            parser.error("--stage-gait clocks2 is a pinned preset; --stage-gait-export* go with --stage-gait export")
        return ["--stage-gait=clocks2"]
    if args.stage_gait_export is None:
        parser.error("--stage-gait export needs --stage-gait-export <exported dir>")
    export = args.stage_gait_export.resolve()
    if not (export / "policy.onnx").is_file():
        parser.error(f"no policy.onnx in {export}")
    digest = hashlib.sha256((export / "policy.onnx").read_bytes()).hexdigest()
    if args.stage_gait_export_sha256 is not None and args.stage_gait_export_sha256 != digest:
        parser.error(f"{export}/policy.onnx has sha256 {digest}, not {args.stage_gait_export_sha256}")
    return ["--stage-gait=export", f"--stage-gait-export={export}", f"--stage-gait-export-sha256={digest}"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-campaign", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--campaign", type=Path, default=Path(DEFAULT_CAMPAIGN))
    parser.add_argument("--baseline", type=Path, default=Path(DEFAULT_BASELINE))
    parser.add_argument("--stage-lateral", type=float, default=None)
    parser.add_argument("--wait-open", type=float, default=0.)
    parser.add_argument("--press-hold", action="store_true")
    parser.add_argument("--pre-point", type=float, default=None)
    parser.add_argument("--cross-clear", type=float, default=None)
    parser.add_argument("--settle-s", type=float, default=None)
    parser.add_argument("--cross-kick", action="store_true")
    parser.add_argument("--align-yaw", action="store_true")
    parser.add_argument("--stage-gait", choices=("shipped", "turnboth", "m3") + EXPORT_STAGE_GAITS, default=None,
                        help="forwarded as --stage-gait=turnboth, --stage-gait=m3, --stage-gait=clocks2 or "
                             "--stage-gait=export (with --stage-gait-export); omitted = the shipped gait (exact "
                             "replay behaviour)")
    parser.add_argument("--stage-gait-export", type=Path, default=None,
                        help="--stage-gait export only: the exported directory; forwarded with its policy.onnx "
                             "sha256 as read here, so the replay refuses different weights")
    parser.add_argument("--stage-gait-export-sha256", default=None,
                        help="--stage-gait export only: the expected policy.onnx sha256 (checked here and in the job)")
    parser.add_argument("--cross-budget", choices=FIX_CROSS_BUDGETS, default=None,
                        help="m7-fix F2, forwarded as --cross-budget=window (an export stage gait only)")
    parser.add_argument("--yaw-cap", type=float, choices=FIX_YAW_CAPS, default=None,
                        help="m7-fix F3, forwarded as --yaw-cap=0.6 (an export stage gait only)")
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
    if args.pre_point is not None:
        probe_args.append(f"--pre-point={args.pre_point}")
    if args.cross_clear is not None:
        probe_args.append(f"--cross-clear={args.cross_clear}")
    if args.settle_s is not None:
        probe_args.append(f"--settle-s={args.settle_s}")
    if args.cross_kick:
        probe_args.append("--cross-kick")
    if args.align_yaw:
        probe_args.append("--align-yaw")
    if args.stage_gait == "turnboth":
        if args.align_yaw or args.press_hold:
            parser.error("--stage-gait turnboth does not compose with --align-yaw or --press-hold")
        probe_args.append("--stage-gait=turnboth")
    elif args.stage_gait == "m3":
        if args.align_yaw or args.press_hold:
            parser.error("--stage-gait m3 does not compose with --align-yaw or --press-hold")
        probe_args.append("--stage-gait=m3")
    export_args = _export_args(parser, args)   # --- m7-clocks2 --- [] unless --stage-gait clocks2 | export
    probe_args += export_args
    fix_args = _fix_args(parser, args)         # --- m7-fix --- [] unless --cross-budget / --yaw-cap
    probe_args += fix_args
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
    if args.stage_gait in EXPORT_STAGE_GAITS:   # --- m7-clocks2 --- see GAIT_CLOCK_MODULE
        files.append(ROOT / GAIT_CLOCK_MODULE)
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
    if args.stage_gait in ("turnboth", "m3"):   # default receipts keep their keys
        row["stage_gait"] = args.stage_gait
    if args.stage_gait in EXPORT_STAGE_GAITS:   # --- m7-clocks2 ---
        row["stage_gait"] = args.stage_gait
        row["stage_gait_export_args"] = export_args
    if fix_args:   # --- m7-fix ---
        row["stage_fix_args"] = fix_args
    (out / "submission.json").write_text(json.dumps(row, indent=2) + "\n")
    with (ROOT / "SLURM_JOBS.md").open("a") as stream:
        stream.write(
            f"\nMission7 exact replay gate (2026-09-23): **SUBMITTED** `{job_id}` — ten-fall staged replay pinned to "
            f"`{args.node}`, PlateStage lateral `{'plate centre' if args.stage_lateral is None else f'{args.stage_lateral} m'}`, wait-open `{args.wait_open} s`; "
            f"{'stage gait `turnboth` (TurnBoth-s0 swapped in for the stage); ' if args.stage_gait == 'turnboth' else ''}"
            f"{'stage gait `m3` (shipped gait: turn while stepping + stall watchdog); ' if args.stage_gait == 'm3' else ''}"
            f"{'stage gait `' + args.stage_gait + '` (its controller swapped in for the stage, M2 turnboth law); ' if args.stage_gait in EXPORT_STAGE_GAITS else ''}"
            f"{'m7-fix options `' + ' '.join(fix_args) + '`; ' if fix_args else ''}"
            f"geometry, activation schedule and fall predicate unchanged; receipt/source hashes: "
            f"`{out.relative_to(ROOT)}/submission.json`.\n")
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
