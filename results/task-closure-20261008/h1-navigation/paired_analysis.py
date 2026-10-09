"""Recompute descriptive H1 comparisons from all frozen matched raw records.

Run after independent contract verification. No gate, input or episode changes.
Shared layout IDs remain explicit; there is no independent-policy inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

ACTORS = ("armV5-s8", "armV5-s9", "armV5-s10")
CONDITIONS = ("nominal", "drop35")
SEEDS = tuple(range(120000, 120048))
PROTOCOL_SHA = "7b0b404ccac476340550166158b652b3d0eebb4a560aa0b397bf58eb75cd9112"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(values):
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None}


def summarize(pairs):
    def arm(index):
        episodes = [p[index] for p in pairs]
        flips = [e["cmd_filter"]["gait_wz"]["flips_per_s"] for e in episodes]
        return {"episodes": len(episodes), "goals": sum(e["success"] for e in episodes),
                "clean_goals": sum(e["clean_success"] for e in episodes),
                "falls": sum(e["outcome"] == "fall" for e in episodes),
                "wall_contact_steps": sum(e["wall_contact_steps"] for e in episodes),
                "gait_yaw_flips_per_s": describe(flips)}
    baseline, candidate = arm(3), arm(4)
    deltas = [c["cmd_filter"]["gait_wz"]["flips_per_s"] -
              b["cmd_filter"]["gait_wz"]["flips_per_s"] for _, _, _, b, c in pairs]
    times = [c["completion_s"] - b["completion_s"] for _, _, _, b, c in pairs
             if b["success"] and c["success"]]
    return {"matched_pairs": len(pairs), "baseline": baseline, "candidate": candidate,
            "goal_rate_difference_percentage_points": 100 * (candidate["goals"] - baseline["goals"]) / len(pairs),
            "mean_yaw_flips_reduction_percent": 100 * (1 - candidate["gait_yaw_flips_per_s"]["mean"] /
                                                           baseline["gait_yaw_flips_per_s"]["mean"]),
            "paired_yaw_flips_delta_per_s": describe(deltas),
            "paired_goal_outcomes": {
                "both_goal": sum(b["success"] and c["success"] for _, _, _, b, c in pairs),
                "baseline_only_goal": sum(b["success"] and not c["success"] for _, _, _, b, c in pairs),
                "candidate_only_goal": sum(c["success"] and not b["success"] for _, _, _, b, c in pairs),
                "neither_goal": sum(not b["success"] and not c["success"] for _, _, _, b, c in pairs)},
            "paired_success_completion_delta_s": describe(times),
            "time_scope": "Candidate minus baseline; only pairs where both succeed. Conditional subset, not all-episode speed."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs-csv", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists() or a.pairs_csv.exists():
        ap.error("choose fresh output paths")
    if digest(a.campaign / "protocol.json") != PROTOCOL_SHA:
        ap.error("trusted submitted protocol hash differs")
    verified = json.loads((a.campaign / "independent-verification.json").read_text())
    if verified["status"] != "PASS" or verified["valid_episode_records"] != 576 or verified["problems"]:
        ap.error("strict verification must establish PASS with all 576 records")
    pairs = []
    for actor in ACTORS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                values = []
                for mode in ("none", "wz-lpf"):
                    f = a.campaign / "runs" / f"{actor}-{mode}-{condition}" / f"seed{seed}.json"
                    if digest(f) != verified["raw_episode_sha256"][str(f.relative_to(a.campaign))]:
                        ap.error(f"raw evidence differs after verification: {f}")
                    values.append(json.loads(f.read_text()))
                if values[0]["maze"]["layout_sha256"] != values[1]["maze"]["layout_sha256"]:
                    ap.error("matched layouts differ")
                pairs.append((actor, condition, seed, *values))
    a.pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["actor", "condition", "seed", "layout_sha256", "baseline_outcome", "candidate_outcome",
              "baseline_yaw_flips_per_s", "candidate_yaw_flips_per_s", "yaw_flips_delta_per_s",
              "baseline_wall_contact_steps", "candidate_wall_contact_steps", "baseline_completion_s",
              "candidate_completion_s", "paired_success_completion_delta_s"]
    with a.pairs_csv.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for actor, condition, seed, b, c in pairs:
            bf, cf = [e["cmd_filter"]["gait_wz"]["flips_per_s"] for e in (b, c)]
            writer.writerow(dict(zip(fields, [actor, condition, seed, b["maze"]["layout_sha256"],
                b["outcome"], c["outcome"], bf, cf, cf - bf, b["wall_contact_steps"], c["wall_contact_steps"],
                b["completion_s"], c["completion_s"],
                c["completion_s"] - b["completion_s"] if b["success"] and c["success"] else None])))
    report = json.loads((a.campaign / "confirmation_report.json").read_text())
    result = {"schema": "bhl-h1-paired-descriptive-v1", "status": "PASS", "planned_episodes": 576,
        "verified_episodes": 576, "protocol_sha256": PROTOCOL_SHA,
        "analysis_source_sha256": digest(__file__), "paired_csv_sha256": digest(a.pairs_csv),
        "independent_verification_sha256": digest(a.campaign / "independent-verification.json"),
        "command_metric_definition": {"raw_key": "cmd_filter.gait_wz.flips_per_s",
            "numerator": "Consecutive nonzero opposite-sign final gait yaw commands; transitions involving zero do not count.",
            "denominator": "Command sample steps times policy_dt=0.04 s, after the 1 s settling period.",
            "aggregation": "Arithmetic mean of all 288 rounded per-episode rates per arm; each episode rate was rounded to three decimal places in the frozen producer.",
            "interpretation": "Final commanded yaw after the LPF and speed brake, not measured physical yaw oscillation."},
        "candidate_rule": report["candidate_rule"], "per_actor_pass": report["per_actor_pass"],
        "all_pairs": summarize(pairs),
        "by_condition": {c: summarize([p for p in pairs if p[1] == c]) for c in CONDITIONS},
        "by_actor_condition": {f"{actor}-{condition}": summarize([p for p in pairs if p[:2] == (actor, condition)])
                               for actor in ACTORS for condition in CONDITIONS},
        "layout_blocks": [{"seed": seed, "layout_sha256": next(p[3]["maze"]["layout_sha256"] for p in pairs if p[2] == seed),
                           **summarize([p for p in pairs if p[2] == seed])} for seed in SEEDS],
        "statistical_scope": {"shared_layout_blocks": 48, "trained_navigation_actors": 3,
            "sensing_conditions": 2, "pairs_per_layout": 6,
            "interpretation": "Descriptive matched simulation comparison. Repeated layouts across actors and conditions are clustered; 288 pairs are not 288 independent layouts or trained policies. No formal significance or hardware performance claim."},
        "scope": "Existing scripted 0.2 s yaw filter; frozen learned gait/navigation actors; oracle pose/goal; simulated lidar and 35% packet dropout. No stereo, physical lidar, new training, or hardware evaluation."}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "all_pairs": result["all_pairs"]}, indent=2))


if __name__ == "__main__":
    main()
