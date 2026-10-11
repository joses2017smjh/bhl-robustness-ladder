#!/usr/bin/env python3
"""Waiter phase 3 gate verdict (rule predeclared in SLURM_JOBS.md and frozen in a rule file before the scored run).

Reads the frozen rule file and the two scored rollout results written by rollout_vla.py --purpose scored:
  own       every seed 4000-4099 once, with its layout's own instruction (the natural mix of templates)
  held_out  every seed 4000-4099 once, with the held-out instruction template
checks that both runs are the ones the rule describes (purpose, gate file hash, checkpoint, seeds, completeness,
stance lock on) and writes the verdict once (never overwritten). It only counts the environment's own success flag.

usage: phase3_gate.py --rule <rule.json> --own <own.json> --held-out <held_out.json> --out <verdict.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def check_run(res: dict, rule: dict, rule_sha: str, name: str, want_ids) -> list[str]:
    """Problems with one scored result against the frozen rule ([] = none)."""
    p = []
    if res.get("purpose") != "scored" or res.get("env") != "waiter" or res.get("synthetic"):
        p.append(f"{name}: not a scored waiter run")
    if (res.get("gate") or {}).get("sha256") != rule_sha:
        p.append(f"{name}: gate file hash differs from the rule file")
    if Path(str(res.get("checkpoint"))).name != Path(rule["checkpoint"]).name or \
            res.get("checkpoint_data_hash") != rule["checkpoint_data_hash"]:
        p.append(f"{name}: checkpoint {res.get('checkpoint')} is not the frozen one")
    seeds = list(range(rule["seed0"], rule["seed1"] + 1))
    if list(res.get("seeds", [])) != seeds:
        p.append(f"{name}: seeds are not {rule['seed0']}-{rule['seed1']}")
    if list(res.get("instruction_ids", [])) != list(want_ids):
        p.append(f"{name}: instruction ids {res.get('instruction_ids')} != {list(want_ids)}")
    eps = res.get("episodes", [])
    if sorted(e["seed"] for e in eps) != seeds or not res.get("completed", True):
        p.append(f"{name}: {len(eps)} episodes, expected one per seed")
    if res.get("counterfactual_slot"):
        p.append(f"{name}: a counterfactual run cannot be scored")
    if res.get("stance_lock") is not True or res.get("fixed_base"):
        p.append(f"{name}: not the contract env (stance_lock {res.get('stance_lock')}, fixed_base {res.get('fixed_base')})")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", type=Path, required=True)
    ap.add_argument("--own", type=Path, required=True)
    ap.add_argument("--held-out", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"PHASE3 GATE: REFUSED ({a.out} exists; never overwritten)")
    rule = json.loads(a.rule.read_text())
    rule_sha = hashlib.sha256(a.rule.read_bytes()).hexdigest()
    own, held = json.loads(a.own.read_text()), json.loads(a.held_out.read_text())
    problems = check_run(own, rule, rule_sha, "own", [None]) + check_run(held, rule, rule_sha, "held_out",
                                                                         [rule["held_out_instruction_id"]])
    bad_tpl = [e["seed"] for e in held.get("episodes", []) if e.get("template_id") != rule["held_out_template"]]
    if bad_tpl:
        problems.append(f"held_out: {len(bad_tpl)} episodes are not template {rule['held_out_template']}")
    if problems:
        raise SystemExit("PHASE3 GATE: INVALID inputs, no verdict written: " + "; ".join(problems))
    k_own = sum(bool(e["success"]) for e in own["episodes"])
    k_held = sum(bool(e["success"]) for e in held["episodes"])
    n = rule["seed1"] - rule["seed0"] + 1
    seen = [e for e in own["episodes"] if not e.get("held_out_template")]
    verdict = "PASS" if k_own >= rule["min_own"] and k_held >= rule["min_held_out"] else "FAIL"
    res = {"gate": "Waiter phase 3 (SmolVLA supervised fine-tune)", "verdict": verdict, "rule": rule["rule"],
           "rule_file": str(a.rule), "rule_sha256": rule_sha,
           "own_instruction": {"successes": k_own, "n": n, "wilson95": wilson(k_own, n),
                               "falls": sum(bool(e["fell"]) for e in own["episodes"]),
                               "seen_templates_only": {"successes": sum(bool(e["success"]) for e in seen), "n": len(seen)}},
           "held_out_template": {"successes": k_held, "n": n, "wilson95": wilson(k_held, n),
                                 "falls": sum(bool(e["fell"]) for e in held["episodes"])},
           "inputs": {"own": str(a.own), "held_out": str(a.held_out)}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as f:
        json.dump(res, f, indent=1)
    print(f"WAITER PHASE 3 GATE: {verdict} (own instruction {k_own}/{n}, need >= {rule['min_own']}; held-out template "
          f"{k_held}/{n}, need >= {rule['min_held_out']}) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
