"""Resolve fresh campaign checkpoints or prepare a gated stage resume.

This helper does not import torch/Isaac. Generated overrides are runtime output
for train.sh; only the exact checkpoint from a passed policy gate is resumed.
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
sub = parser.add_subparsers(dest="mode", required=True)
resolve = sub.add_parser("resolve")
resolve.add_argument("--log-root", type=Path, required=True)
resolve.add_argument("--run-name", required=True)
resolve.add_argument("--newer-than", type=float, required=True)
resume = sub.add_parser("resume")
resume.add_argument("--gate", type=Path, required=True)
resume.add_argument("--task", required=True)
resume.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

if args.mode == "resolve":
    candidates = []
    for directory in args.log_root.glob(f"*_{args.run_name}"):
        for checkpoint in directory.glob("model_*.pt"):
            suffix = checkpoint.stem.removeprefix("model_")
            if suffix.isdigit() and checkpoint.stat().st_mtime >= args.newer_than:
                candidates.append((directory.name, int(suffix), checkpoint))
    if not candidates:
        raise SystemExit(f"No fresh checkpoint from run {args.run_name}")
    print(max(candidates, key=lambda row: row[:2])[2].resolve())
else:
    gate = json.loads(args.gate.read_text())
    if not (gate.get("passed") is True and gate.get("gate_kind") == "policy_evaluation"
            and gate.get("task") == args.task
            and gate.get("first_episode_success_rate", 0) >= 0.30
            and gate.get("first_episodes_completed", 0) == gate.get("num_envs", -1)):
        raise SystemExit(f"Previous stage has not passed its policy gate: {args.gate}")
    checkpoint = Path(gate["checkpoint"]).resolve(strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("agent.resume=true\n"
        f"agent.load_run={checkpoint.parent.name}\n"
        f"agent.load_checkpoint={checkpoint.name}\n")
    print(checkpoint)
