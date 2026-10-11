#!/usr/bin/env python3
"""Fit/freeze or independently evaluate an offline episode health screen."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bhl_robust.research.outcome_health import evaluate, fit, protocol


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=("protocol", "fit", "evaluate"))
    p.add_argument("--episodes", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.mode == "protocol":
        result = protocol()
    else:
        if a.episodes is None:
            p.error("--episodes required")
        rows = json.loads(a.episodes.read_text())
        if a.mode == "fit":
            if a.model:
                p.error("fit reads only calibration episodes, not a previous model")
            result = fit(rows)
        else:
            if a.model is None:
                p.error("--model required")
            result = evaluate(json.loads(a.model.read_text()), rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"schema": result["schema"], "output": str(a.output)}))


if __name__ == "__main__":
    main()
