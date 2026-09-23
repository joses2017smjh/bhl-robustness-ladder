"""Aggregate Mission 7 campaign probe outputs into compact, comparable summaries.

Two modes over one or more probe output directories (each holding a compact
result.json and per-episode <stage>-<index>.json records):

  campaign-a  --dirs D1 D2 ...  --out summary.json
      Coverage by stage / layout / world direction, failure phase, mechanism
      label with its raw statistics, and the exposure list (episodes whose
      post-stage stall condition became true, with branch time and fingerprint)
      that Campaign B reruns to.

  campaign-b  --baseline D... --arm D... --out summary.json
      Pair baseline and intervention episodes by (stage, layout); verify the
      pre-intervention stall fingerprints match; report per-pair delivery
      state, outcome change, post-stage progress and falls; keep the route
      denominator whole.  Pairs whose fingerprints differ are reported, not
      silently compared.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


HEAVY_EPISODE_KEYS = (
    "plate_contact_events", "rejoin_diagnostic", "diagnostic_trace",
    "decision_rewards", "route_phase_history", "trace",
    "guarded_stage_history", "guarded_stage_phase_history",
    "stage_entry_states", "world_contact_samples", "chain",
)


def load_dir(directory):
    """Compact summaries for a probe output dir.

    Falls back to the per-episode records when result.json is absent, which is
    what a job that died mid-run (21400755, out of memory) leaves behind; the
    episodes it did write are still valid and stay in the denominator.
    """
    directory = Path(directory)
    result_path = directory / "result.json"
    rows = []
    if result_path.exists():
        result = json.loads(result_path.read_text())
        summaries = result["episodes"]
    else:
        result = {"status": "PARTIAL_NO_RESULT_JSON", "episodes": []}
        summaries = []
        for record in sorted(directory.glob("*-[0-9]*.json")):
            if record.name == "submission.json":
                continue
            row = json.loads(record.read_text())
            summary = {k: v for k, v in row.items() if k not in HEAVY_EPISODE_KEYS}
            summary["full_episode_record"] = record.name
            summary["partial_dir"] = True
            summaries.append(summary)
        result["episodes"] = summaries
    for summary in summaries:
        record = directory / summary.get("full_episode_record", f"{summary['stage']}-{summary['layout_index']}.json")
        rows.append({"summary": summary, "record_path": str(record), "dir": str(directory)})
    return result, rows


def world_direction(summary):
    entry = summary.get("stage_entry_state") or {}
    return entry.get("world_direction")


def episode_key(summary):
    return f"{summary['stage']}/{summary['layout_index']}"


def compact_stats(stats):
    if not stats:
        return None
    keys = ("mechanism", "window_s", "ticks", "target_range_ratio_vs_reference",
            "tracking_err_ratio_vs_reference", "double_support_frac", "stance_slip_m",
            "base_progress_m", "base_speed_mean_mps", "ctrl_saturation_frac", "cmd_in_mean_mps")
    return {k: stats.get(k) for k in keys}


def campaign_a(args):
    episodes = []
    for directory in args.dirs:
        result, rows = load_dir(directory)
        for row in rows:
            s = row["summary"]
            episodes.append({
                "key": episode_key(s), "stage": s["stage"], "layout_index": s["layout_index"],
                "world_direction": world_direction(s),
                "success": bool(s["success"]), "fall": bool(s.get("fall")), "timeout": bool(s.get("timeout")),
                "failure_phase": s["failure_phase"], "elapsed_s": s["elapsed_s"],
                "stage_activated": bool(s["guarded_stage_activated"]),
                "stage_completed": bool(s["guarded_stage_completed"]),
                "exposure": s.get("intervention_exposure"),
                "stall_branch_time_s": s.get("stall_branch_time_s"),
                "stall_fingerprint": (s.get("stall_fingerprint") or {}).get("sha256_16"),
                "mechanism": s.get("mechanism"),
                "stall_window": compact_stats(s.get("chain_stall_window_summary")),
                "post_stage": compact_stats(s.get("chain_post_stage_summary")),
                "walking_reference": compact_stats(s.get("chain_walking_reference_summary")),
                "post_stage_contacts": s.get("post_stage_contact_summary"),
                "source_dir": row["dir"],
            })
    episodes.sort(key=lambda e: (e["stage"], e["layout_index"]))
    by = lambda field: {k: {"n": len(v), "success": sum(e["success"] for e in v),
                            "falls": sum(e["fall"] for e in v), "timeouts": sum(e["timeout"] for e in v),
                            "failure_phases": dict(Counter(e["failure_phase"] for e in v)),
                            "mechanisms": dict(Counter(str(e["mechanism"]) for e in v))}
                        for k, v in sorted(_group(episodes, field).items(), key=lambda kv: str(kv[0]))}
    exposed = [e for e in episodes if e["exposure"] == "exposed"]
    summary = {
        "episodes": len(episodes),
        "success": sum(e["success"] for e in episodes),
        "falls": sum(e["fall"] for e in episodes),
        "timeouts": sum(e["timeout"] for e in episodes),
        "by_stage": by("stage"),
        "by_world_direction": by("world_direction"),
        "by_failure_phase": dict(Counter(e["failure_phase"] for e in episodes)),
        "mechanism_counts_stall_window": dict(Counter(str((e["stall_window"] or {}).get("mechanism")) for e in episodes if e["stall_window"])),
        "mechanism_counts_post_stage": dict(Counter(str((e["post_stage"] or {}).get("mechanism")) for e in episodes if e["post_stage"])),
        "exposed_episodes": len(exposed),
        "exposure_list": [{"key": e["key"], "stage": e["stage"], "layout_index": e["layout_index"],
                           "world_direction": e["world_direction"], "stall_branch_time_s": e["stall_branch_time_s"],
                           "stall_fingerprint": e["stall_fingerprint"], "outcome": ("success" if e["success"] else e["failure_phase"]),
                           "mechanism": e["mechanism"]} for e in exposed],
        "per_episode": episodes,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("episodes", "success", "falls", "timeouts", "exposed_episodes",
                                              "mechanism_counts_stall_window", "by_failure_phase")}, sort_keys=True))


def _group(episodes, field):
    groups = defaultdict(list)
    for e in episodes:
        groups[e[field]].append(e)
    return groups


def campaign_b(args):
    def index(dirs):
        table = {}
        for directory in dirs:
            _, rows = load_dir(directory)
            for row in rows:
                s = row["summary"]
                table[episode_key(s)] = s
        return table
    base, arm = index(args.baseline), index(args.arm)
    pairs = []
    for key in sorted(set(base) | set(arm)):
        b, a = base.get(key), arm.get(key)
        if b is None or a is None:
            pairs.append({"key": key, "status": "unpaired", "in_baseline": b is not None, "in_arm": a is not None})
            continue
        bf, af = (b.get("stall_fingerprint") or {}).get("sha256_16"), (a.get("stall_fingerprint") or {}).get("sha256_16")
        fire = (a.get("rejoin_fix_events") or [{}])[0]
        fire_fp = (fire.get("fingerprint_at_fire") or {}).get("sha256_16")
        pairs.append({
            "key": key, "stage": b["stage"], "layout_index": b["layout_index"],
            "world_direction": world_direction(b),
            "baseline_exposure": b.get("intervention_exposure"), "arm_exposure": a.get("intervention_exposure"),
            "baseline_stall_time_s": b.get("stall_branch_time_s"), "arm_stall_time_s": a.get("stall_branch_time_s"),
            "fingerprint_baseline": bf, "fingerprint_arm": af, "fingerprint_at_fire": fire_fp,
            "fingerprints_match": (bf is not None and bf == af == (fire_fp or af)),
            "delivery": a.get("intervention_delivery"),
            "baseline_outcome": "success" if b["success"] else b["failure_phase"],
            "arm_outcome": "success" if a["success"] else a["failure_phase"],
            "baseline_success": bool(b["success"]), "arm_success": bool(a["success"]),
            "baseline_fall": bool(b.get("fall")), "arm_fall": bool(a.get("fall")),
            "baseline_elapsed_s": b["elapsed_s"], "arm_elapsed_s": a["elapsed_s"],
            "baseline_post_stage_progress_m": ((b.get("chain_post_stage_summary") or {}).get("base_progress_m")),
            "arm_post_stage_progress_m": ((a.get("chain_post_stage_summary") or {}).get("base_progress_m")),
            "arm_mechanism": a.get("mechanism"), "baseline_mechanism": b.get("mechanism"),
            # The comparable column: the window that starts at the stall
            # condition in each arm (present in records written after the
            # 2026-09-23 review; older records need the offline recompute in
            # campaign-b-stall-anchored.json).
            "baseline_stall_anchored": compact_stats(b.get("chain_stall_anchored_summary")),
            "arm_stall_anchored": compact_stats(a.get("chain_stall_anchored_summary")),
        })
    paired = [p for p in pairs if p.get("status") != "unpaired"]
    exposed = [p for p in paired if p["arm_exposure"] == "exposed"]
    applied = [p for p in exposed if p["delivery"] == "APPLIED"]
    matched = [p for p in applied if p["fingerprints_match"]]
    summary = {
        "pairs": len(paired), "unpaired": [p["key"] for p in pairs if p.get("status") == "unpaired"],
        "route_denominator": len(paired),
        "baseline_success": sum(p["baseline_success"] for p in paired),
        "arm_success": sum(p["arm_success"] for p in paired),
        "baseline_falls": sum(p["baseline_fall"] for p in paired),
        "arm_falls": sum(p["arm_fall"] for p in paired),
        "not_exposed": [p["key"] for p in paired if p["arm_exposure"] != "exposed"],
        "exposed": len(exposed),
        "delivery_counts": dict(Counter(p["delivery"] for p in exposed)),
        "fingerprint_mismatches": [p["key"] for p in applied if not p["fingerprints_match"]],
        "matched_applied": len(matched),
        "matched_improved": [p["key"] for p in matched if p["arm_success"] and not p["baseline_success"]],
        "matched_regressed": [p["key"] for p in matched if p["baseline_success"] and not p["arm_success"]],
        "matched_unchanged": [p["key"] for p in matched if p["arm_success"] == p["baseline_success"]],
        "matched_new_falls": [p["key"] for p in matched if p["arm_fall"] and not p["baseline_fall"]],
        # Outcome flips over EVERY pair, whatever the delivery state: for a
        # controller arm with no intervention (delivery NOT_REQUESTED) the
        # matched_* lists above are empty by construction, and the in-stage
        # arms were first read with those empty lists while pairs_detail showed
        # Doors/8 turning into a success.  Success counts were always over all
        # pairs; these make the per-layout flips visible too.
        "all_improved": [p["key"] for p in paired if p["arm_success"] and not p["baseline_success"]],
        "all_regressed": [p["key"] for p in paired if p["baseline_success"] and not p["arm_success"]],
        "all_falls_removed": [p["key"] for p in paired if p["baseline_fall"] and not p["arm_fall"]],
        "all_falls_added": [p["key"] for p in paired if p["arm_fall"] and not p["baseline_fall"]],
        "pairs_detail": pairs,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("pairs", "baseline_success", "arm_success", "exposed", "delivery_counts",
                                              "matched_applied", "matched_improved", "matched_regressed", "fingerprint_mismatches",
                                              "all_improved", "all_regressed", "all_falls_removed", "all_falls_added")}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("campaign-a"); a.add_argument("--dirs", nargs="+", required=True); a.add_argument("--out", required=True)
    b = sub.add_parser("campaign-b"); b.add_argument("--baseline", nargs="+", required=True); b.add_argument("--arm", nargs="+", required=True); b.add_argument("--out", required=True)
    args = parser.parse_args()
    (campaign_a if args.mode == "campaign-a" else campaign_b)(args)
