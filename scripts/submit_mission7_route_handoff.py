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
# --- m7-clocks2 --- the export stage gaits swap in a controller built by bhl_robust.eval.gait_clock, which the snapshot
# job (PYTHONPATH = the snapshot only) must find in the snapshot: it is added for these gaits only (composed path, as in
# scripts/submit_mission7_plate_stage.py; the default snapshot set is unchanged).
EXPORT_STAGE_GAITS = ("clocks2", "export")
GAIT_CLOCK_MODULE = Path("src/bhl_robust/eval") / "gait_clock.py"
# --- m7-fix --- F2 / F3 (= mission7_plate_stage.FIX_CROSS_BUDGETS / FIX_YAW_CAPS; that module is not imported here):
# forwarded verbatim as --cross-budget=window / --yaw-cap=0.6, with an export stage gait only.
FIX_CROSS_BUDGETS = ("window",)
FIX_YAW_CAPS = (.60,)


def _check_fix_args(parser, args):
    """--- m7-fix --- validate --cross-budget / --yaw-cap (no-op when neither is given)."""
    if args.cross_budget is None and args.yaw_cap is None:
        return
    if args.stage_gait not in EXPORT_STAGE_GAITS:
        parser.error("--cross-budget / --yaw-cap go with --stage-gait clocks2 | export only")
    if args.cross_budget is not None and args.cross_clear is not None:
        parser.error("--cross-budget ends the crossing at mission7_gates' clear; --cross-clear does not compose")


def _check_export_args(parser, args):
    """--- m7-clocks2 --- validate the export flags; for --stage-gait export resolve the directory and pin the sha256
    of its policy.onnx as read here (a given --stage-gait-export-sha256 must match it).  No-op for other gaits."""
    if args.stage_gait not in EXPORT_STAGE_GAITS or args.stage_gait == "clocks2":
        if args.stage_gait_export is not None or args.stage_gait_export_sha256 is not None:
            parser.error("--stage-gait-export / --stage-gait-export-sha256 go with --stage-gait export only")
        if args.stage_gait == "clocks2" and (args.align_yaw or args.stage_press_hold):
            parser.error("--stage-gait clocks2 does not compose with --align-yaw or --stage-press-hold")
        return
    if args.align_yaw or args.stage_press_hold:
        parser.error("--stage-gait export does not compose with --align-yaw or --stage-press-hold")
    if args.stage_gait_export is None:
        parser.error("--stage-gait export needs --stage-gait-export <exported dir>")
    args.stage_gait_export = args.stage_gait_export.resolve()
    policy = args.stage_gait_export / "policy.onnx"
    if not policy.is_file():
        parser.error(f"no policy.onnx in {args.stage_gait_export}")
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    if args.stage_gait_export_sha256 is not None and args.stage_gait_export_sha256 != digest:
        parser.error(f"{policy} has sha256 {digest}, not {args.stage_gait_export_sha256}")
    args.stage_gait_export_sha256 = digest


def _probe_args(args):
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
    if args.pre_point is not None:
        probe_args.append(f"--pre-point={args.pre_point}")
    if args.cross_clear is not None:
        probe_args.append(f"--cross-clear={args.cross_clear}")
    if args.stall_min_s is not None:
        probe_args.append(f"--stall-min-s={args.stall_min_s}")
    if args.settle_s is not None:
        probe_args.append(f"--settle-s={args.settle_s}")
    if args.cross_kick:
        probe_args.append("--cross-kick")
    if args.rejoin_advance:
        probe_args.append("--rejoin-advance")
    if args.exit_ramp_center:
        probe_args.append("--exit-ramp-center")
    if args.align_yaw:
        probe_args.append("--align-yaw")
    if args.exit_ramp:
        probe_args.append(f"--exit-ramp={args.exit_ramp}")
    if args.allow_inactive_intervention:
        probe_args.append("--allow-inactive-intervention")
    if args.stage_gait == "turnboth":
        probe_args.append("--stage-gait=turnboth")
    elif args.stage_gait == "m3":
        probe_args.append("--stage-gait=m3")
    elif args.stage_gait == "clocks2":   # --- m7-clocks2 --- a pinned preset
        probe_args.append("--stage-gait=clocks2")
    elif args.stage_gait == "export":    # --- m7-clocks2 --- main() resolves the directory and its policy sha256
        probe_args += ["--stage-gait=export", f"--stage-gait-export={getattr(args, 'stage_gait_export', None)}",
                       f"--stage-gait-export-sha256={getattr(args, 'stage_gait_export_sha256', None)}"]
    if getattr(args, "cross_budget", None) is not None:   # --- m7-fix --- F2
        probe_args.append(f"--cross-budget={args.cross_budget}")
    if getattr(args, "yaw_cap", None) is not None:        # --- m7-fix --- F3
        probe_args.append(f"--yaw-cap={args.yaw_cap}")
    return probe_args


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
    parser.add_argument("--pre-point", type=float, default=None)
    parser.add_argument("--cross-clear", type=float, default=None)
    parser.add_argument("--stall-min-s", type=float, default=None)
    parser.add_argument("--settle-s", type=float, default=None)
    parser.add_argument("--cross-kick", action="store_true")
    parser.add_argument("--rejoin-advance", action="store_true")
    parser.add_argument("--exit-ramp-center", action="store_true")
    parser.add_argument("--align-yaw", action="store_true")
    parser.add_argument("--exit-ramp", type=float, default=0.)
    parser.add_argument("--stage-gait", choices=("shipped", "turnboth", "m3") + EXPORT_STAGE_GAITS, default=None,
                        help="forwarded as --stage-gait=turnboth, --stage-gait=m3, --stage-gait=clocks2 or "
                             "--stage-gait=export (with --stage-gait-export); omitted = the shipped stage gait")
    parser.add_argument("--stage-gait-export", type=Path, default=None,
                        help="--stage-gait export only: the exported directory; forwarded with its policy.onnx "
                             "sha256 as read here, so the probe refuses different weights")
    parser.add_argument("--stage-gait-export-sha256", default=None,
                        help="--stage-gait export only: the expected policy.onnx sha256 (checked here and in the job)")
    parser.add_argument("--cross-budget", choices=FIX_CROSS_BUDGETS, default=None,
                        help="m7-fix F2, forwarded as --cross-budget=window (an export stage gait only)")
    parser.add_argument("--yaw-cap", type=float, choices=FIX_YAW_CAPS, default=None,
                        help="m7-fix F3, forwarded as --yaw-cap=0.6 (an export stage gait only)")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if not campaign.is_relative_to(ROOT):
        parser.error("campaign must remain inside this repository")
    out = campaign / args.output_name
    snapshot = out / "source"
    if out.exists():
        parser.error(f"destination exists; preserve it and choose a new campaign: {out}")
    if args.stage_gait == "turnboth" and (args.align_yaw or args.stage_press_hold):
        parser.error("--stage-gait turnboth does not compose with --align-yaw or --stage-press-hold")
    if args.stage_gait == "m3" and (args.align_yaw or args.stage_press_hold):
        parser.error("--stage-gait m3 does not compose with --align-yaw or --stage-press-hold")
    _check_export_args(parser, args)   # --- m7-clocks2 --- (no-op for every other gait)
    _check_fix_args(parser, args)      # --- m7-fix --- (no-op without --cross-budget / --yaw-cap)
    probe_args = _probe_args(args)
    if not args.submit:
        plan = {"planned_output": str(out), "stage": args.stage,
                "indices": args.indices, "node": args.node,
                "constraint": None if args.node else args.constraint}
        if args.stage_gait in ("turnboth", "m3"):   # the default plan is unchanged
            plan["stage_gait"] = args.stage_gait
            plan["probe_args"] = probe_args
        if args.stage_gait in EXPORT_STAGE_GAITS:   # --- m7-clocks2 ---
            plan["stage_gait"] = args.stage_gait
            plan["probe_args"] = probe_args
        print(json.dumps(plan, indent=2))
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
    placement = ([f"--nodelist={args.node}"] if args.node
                 else [f"--constraint={args.constraint}"])
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
        "pre_point_m": args.pre_point,
        "cross_clear_m": args.cross_clear,
        "stall_min_s": args.stall_min_s,
        "settle_s": args.settle_s,
        "cross_kick": args.cross_kick,
        "rejoin_advance": args.rejoin_advance,
        "exit_ramp_center": args.exit_ramp_center,
        "align_yaw": args.align_yaw,
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
    if args.stage_gait in ("turnboth", "m3"):   # default receipts keep their keys
        row["stage_gait"] = args.stage_gait
    if args.stage_gait in EXPORT_STAGE_GAITS:   # --- m7-clocks2 ---
        row["stage_gait"] = args.stage_gait
    if args.cross_budget is not None or args.yaw_cap is not None:   # --- m7-fix ---
        row.update(cross_budget=args.cross_budget, yaw_cap=args.yaw_cap)
    (out / "submission.json").write_text(json.dumps(row, indent=2) + "\n")
    with (ROOT / "SLURM_JOBS.md").open("a") as stream:
        stream.write(
            f"\nMission7 route handoff probe (2026-09-22): **SUBMITTED** `{job_id}` — "
            f"{args.stage} layouts `{args.indices}`, "
            f"{'node `' + args.node + '`' if args.node else 'constraint `' + args.constraint + '`'}"
            f", 2 CPUs / 12 GB / 0 GPUs / 2 h; "
            f"{'PlateStage on the `turnboth` stage gait (TurnBoth-s0 swapped in for the stage)' if args.stage_gait == 'turnboth' else 'PlateStage on the `m3` stage path (shipped gait: turn while stepping + stall watchdog)' if args.stage_gait == 'm3' else 'PlateStage on the `' + str(args.stage_gait) + '` stage gait (its controller swapped in for the stage, M2 turnboth law)' if args.stage_gait in EXPORT_STAGE_GAITS else 'unchanged PlateStage'}"
            f"{'; m7-fix options cross budget `' + str(args.cross_budget) + '`, yaw cap `' + str(args.yaw_cap) + '`' if args.cross_budget is not None or args.yaw_cap is not None else ''}"
            f" with `{args.handoff}` route handoff"
            f" and rejoin diagnostic `{args.rejoin_diagnostic}`, chain trace `{args.chain_trace}`, "
            f"fix `{args.rejoin_fix}`; "
            f"receipt/source hashes: `{out.relative_to(ROOT)}/submission.json`.\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
