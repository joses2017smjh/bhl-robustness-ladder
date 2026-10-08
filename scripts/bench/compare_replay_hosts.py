"""Compare two hosts' nominal traces using the frozen bundle's original gate.

Run in the bundle's pinned Python environment. Receipt and scenario contracts
are checked before comparing the raw physics evidence; timing is not compared.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "reference", "candidate", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists; retain the previous comparison")
    bundle = args.bundle.resolve()
    sys.path.insert(0, str(bundle / "src"))
    from bhl_robust.eval.replay_gate import compare_replay, verify_bundle

    manifest = verify_bundle(bundle)
    manifest_sha = digest(bundle / "manifest.json")
    receipts = {}
    for label, directory in (("reference", args.reference), ("candidate", args.candidate)):
        path = directory / "receipt.json"
        receipt = json.loads(path.read_text())
        if receipt["manifest_sha256"] != manifest_sha or receipt["packages"] != manifest["packages"]:
            raise ValueError(f"{label}: frozen manifest/runtime receipt mismatch")
        if receipt["source"] != manifest["source"]:
            raise ValueError(f"{label}: source receipt mismatch")
        receipts[label] = {"host": receipt["host"], "receipt_sha256": digest(path),
                           "installed_packages": receipt["installed_packages"]}
    if receipts["reference"]["host"] == receipts["candidate"]["host"]:
        raise ValueError("Cross-host comparison requires two different recorded hosts")
    expected_steps = int(manifest["seconds"] / manifest["policy_dt"])
    cases = []
    for scenario in manifest["scenarios"]:
        name = f"seed{scenario['seed']}-nominal.json"
        paths = [args.reference / name, args.candidate / name]
        traces = [json.loads(path.read_text()) for path in paths]
        for label, trace in zip(("reference", "candidate"), traces):
            if (trace["scenario"] != scenario or trace["fault"] is not None
                    or trace["runtime_error"] is not None or len(trace["trace"]) != expected_steps):
                raise ValueError(f"{label}: invalid nominal scenario {name}")
            if [step["step"] for step in trace["trace"]] != list(range(expected_steps)):
                raise ValueError(f"{label}: incomplete step sequence {name}")
            if not compare_replay(trace, trace)["passed"]:
                raise ValueError(f"{label}: invalid physical/contact trace {name}")
        cases.append({"file": name, "reference_sha256": digest(paths[0]),
                      "candidate_sha256": digest(paths[1]), **compare_replay(*traces)})
    result = {"schema": "bhl-cross-host-replay/v1", "manifest_sha256": manifest_sha,
              "gate_sha256": digest(bundle / "src/bhl_robust/eval/replay_gate.py"),
              "receipts": receipts, "cases": cases, "passed": sum(c["passed"] for c in cases),
              "total": len(cases), "status": "PASS" if all(c["passed"] for c in cases) else "NEGATIVE",
              "scope": "Same frozen trained checkpoint, controller, assets and numerical releases. "
                       "Original 1e-9 atol/rtol gate compares qpos, qvel, joint targets and physical contacts. "
                       "Exact agreement is limited to these five two-second seeded trajectories; "
                       "wall time, startup, long-term CI stability and hardware behavior are excluded."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "passed", "total")}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
