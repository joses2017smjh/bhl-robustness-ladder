#!/usr/bin/env python3
"""Recount the published V6 outcome snapshot; no simulator or third-party packages."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
manifest = json.loads((root / "snapshot-hashes.json").read_text())
for name, expected in manifest.items():
    path = (root / name).resolve()
    assert path.is_relative_to(root), name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
rows = json.loads((root / "confirmation-episodes.json").read_text())
audit = json.loads((root / "audit.json").read_text())
assert len(rows) == 1152
assert len({(r["cell"], r["seed"]) for r in rows}) == 1152
assert len({r["cell"] for r in rows}) == 24
assert set(Counter(r["cell"] for r in rows).values()) == {48}
assert {r["seed"] for r in rows} == set(range(2110000, 2110048))
layouts = defaultdict(set)
groups = defaultdict(list)
for row in rows:
    layouts[row["seed"]].add(row["layout_sha256"])
    groups[row["method"]].append(row)
    assert row["outcome"] in {"goal", "fall", "time_out"}
    assert row["success"] == (row["outcome"] == "goal")
    assert not row["clean_success"] or (
        row["success"] and row["wall_contact_steps"] == 0
    )
    assert len(row["original_record_sha256"]) == 64
assert all(len(values) == 1 for values in layouts.values())
assert len({next(iter(values)) for values in layouts.values()}) == 48
assert set(groups) == {"baseline", "yaw_only", "hybrid_adaptive", "budget_adaptive"}
counts = {}
for method, records in sorted(groups.items()):
    counts[method] = {
        "episodes": len(records),
        "goals": sum(r["success"] for r in records),
        "clean_goals": sum(r["clean_success"] for r in records),
        "falls": sum(r["outcome"] == "fall" for r in records),
        "timeouts": sum(r["outcome"] == "time_out" for r in records),
        "wall_contact_episodes": sum(r["wall_contact_steps"] > 0 for r in records),
        "invalid": sum(r["invalid"] for r in records),
        "recovery_episodes": sum(r["recovery"]["recovery_steps"] > 0 for r in records),
        "model_influence_steps": sum(r["counters"]["model_influence_steps"] for r in records),
    }
assert counts == audit["counts"], (counts, audit["counts"])
assert counts["budget_adaptive"]["goals"] == 288
assert counts["budget_adaptive"]["model_influence_steps"] == 0
assert audit["overall_v6_status"] == "INCOMPLETE"
print(json.dumps({"verified_files": len(manifest), "layout_clusters": len(layouts), "counts": counts}, indent=2))
