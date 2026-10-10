#!/usr/bin/env python3
"""Combine all nine declared stereo-method pairs with grouped uncertainty."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from bhl_robust.research.native_orb import sha256
from bhl_robust.research.stereo_recovery_metrics import aggregate


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.results) != 9:
        raise ValueError("exactly nine paired shard results required")
    rows, sources, inputs = [], {}, None
    for path in args.results:
        result = json.loads(path.read_text())
        if result.get("status") != "PASS" or result.get("scientific_status") != "DEVELOPMENT_PAIR_NO_GLOBAL_VERDICT" or result.get("episodes") != 2:
            raise ValueError("every shard must be a completed two-episode measured pair")
        hashes = sorted(result["input_sha256"].values())
        if inputs is not None and hashes != inputs:
            raise ValueError("frozen actor/native input differs between paired shards")
        inputs = hashes
        rows.extend(result["development_pair_outcomes"])
        sources[str(path)] = sha256(path)
    combined = aggregate(rows)
    combined.update(source_results_sha256=sources, input_sha256=inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(combined, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(combined), flush=True)


if __name__ == "__main__": main()
