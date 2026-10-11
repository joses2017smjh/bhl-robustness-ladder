#!/usr/bin/env python3
"""Waiter phase 4 gate verdict (rule predeclared in SLURM_JOBS.md and frozen in a rule file before any phase 4
training or evaluation seed was run).

Reads the frozen rule file, the scored evaluation written by eval_dsrl.py --scored-phase4-eval (base and DSRL arms on
seeds 6000-6099, one episode per seed per arm, paired by seed) and the training run directory. Checks that the
evaluation is the one the rule describes (rule hash, seeds, executed steps per chunk, frozen base model, contract env,
unshaped training on the training range, and that the SAC checkpoint evaluated is the run's final one), recomputes
the exact one-sided McNemar p from the per-episode success flags and writes the verdict once (never overwritten).

usage: phase4_gate.py --rule <rule.json> --eval <scored.json> --out <verdict.json> [--run-dir <dir>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path


def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                return h.hexdigest()
            h.update(block)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def mcnemar_one_sided(b: int, c: int) -> float:
    """b = seeds only the treatment solved, c = seeds only the control solved; exact p for treatment > control."""
    n = b + c
    return 1.0 if n == 0 else sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n


def check(res: dict, rule: dict, rule_sha: str, run_dir: Path) -> list[str]:
    """Problems with the scored evaluation against the frozen rule ([] = none)."""
    p = []
    if not res.get("scored_phase4_eval") or res.get("env") != "waiter" or "TOY" in res.get("labels", []):
        p.append("not a scored waiter evaluation")
    if "TRAINED_WITH_SHAPING" in res.get("labels", []):
        p.append("the run was trained with a shaped reward")
    if (res.get("gate") or {}).get("sha256") != rule_sha:
        p.append("gate file hash differs from the rule file")
    seeds = list(range(rule["seed0"], rule["seed1"] + 1))
    if list(res.get("seeds", [])) != seeds:
        p.append(f"seeds are not {rule['seed0']}-{rule['seed1']}")
    for arm in ("base", "dsrl"):
        eps = (res.get("episodes") or {}).get(arm, [])
        if [e.get("seed") for e in eps] != seeds:
            p.append(f"{arm}: {len(eps)} episodes, expected one per seed in order")
    if res.get("exec_horizon") != rule["exec_horizon"]:
        p.append(f"executed steps per chunk {res.get('exec_horizon')} != {rule['exec_horizon']}")
    if res.get("noise_seed") != rule["noise_seed"]:
        p.append(f"noise seed {res.get('noise_seed')} != {rule['noise_seed']}")
    if res.get("policy_model_sha256") != rule["base_model_sha256"]:
        p.append("the base policy is not the frozen checkpoint")
    kw = res.get("env_kwargs") or {}
    if kw.get("stance_lock", True) is not True or kw.get("fixed_base") or kw.get("lock_start_only") or \
            kw.get("stance_hold", True) is not True or kw.get("stance_mode", "clock") != "clock" or \
            "time_limit_s" in kw or kw.get("render", True) is not True:
        p.append(f"not the contract env ({kw})")
    if Path(str(kw.get("deploy"))).name != Path(rule["deploy"]).name:
        p.append(f"deploy {kw.get('deploy')} is not the frozen one")
    if Path(str(res.get("run_dir"))).name != Path(rule["run_dir"]).name:
        p.append(f"run dir {res.get('run_dir')} is not the frozen one")
    # the training run: unshaped, on the training range, finished, within its budget; its final checkpoint evaluated
    try:
        cfg = json.loads((run_dir / "config.json").read_text())
        args, tr = cfg["args"], rule["train"]
        if cfg.get("labels") or args.get("reward") != "sparse" or args.get("shaping") != "none" or \
                float(args.get("fall_penalty", 0.0)) != 0.0:
            p.append("training was not the unshaped sparse-reward run")
        if args.get("train_seeds") != tr["seeds"] or int(args.get("n_envs", -1)) != tr["n_envs"] or \
                int(args.get("seed", -1)) != tr["seed"] or cfg.get("exec_horizon") != rule["exec_horizon"]:
            p.append("training seeds / envs / seed / executed steps differ from the rule")
        for key, want in (tr.get("args") or {}).items():           # the declared design, as train_dsrl.py recorded it
            if str(args.get(key)) != str(want):
                p.append(f"training argument {key} = {args.get(key)}, the rule says {want}")
        counters = json.loads((run_dir / "summary.json").read_text())["counters"]
        if counters["decisions"] > tr["max_decisions"] + tr["n_envs"]:
            p.append(f"training ran {counters['decisions']} decisions, over the {tr['max_decisions']} budget")
    except FileNotFoundError as e:
        p.append(f"training run incomplete ({Path(e.filename).name} missing)")
    ck = run_dir / "ckpt" / "sac.pt"
    if not ck.exists() or not res.get("sac_ckpt_sha256") or sha256_file(ck) != res["sac_ckpt_sha256"]:
        p.append("the evaluated SAC checkpoint is not the run's final checkpoint")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True, dest="eval_json")
    ap.add_argument("--run-dir", type=Path, help="default: the rule file's run_dir")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"PHASE4 GATE: REFUSED ({a.out} exists; never overwritten)")
    rule = json.loads(a.rule.read_text())
    rule_sha = sha256_file(a.rule)
    res = json.loads(a.eval_json.read_text())
    problems = check(res, rule, rule_sha, a.run_dir or Path(rule["run_dir"]))
    if problems:
        raise SystemExit("PHASE4 GATE: INVALID inputs, no verdict written: " + "; ".join(problems))
    base = [bool(e["success"]) for e in res["episodes"]["base"]]
    dsrl = [bool(e["success"]) for e in res["episodes"]["dsrl"]]
    n, k_base, k_dsrl = len(base), sum(base), sum(dsrl)
    b = sum(d and not c for d, c in zip(dsrl, base))
    c = sum(c and not d for d, c in zip(dsrl, base))
    pval = mcnemar_one_sided(b, c)
    verdict = "PASS" if k_dsrl > k_base and pval <= rule["alpha"] else "FAIL"
    out = {"gate": "Waiter phase 4 (DSRL: SAC over the frozen VLA's initial noise)", "verdict": verdict,
           "rule": rule["rule"], "rule_file": str(a.rule), "rule_sha256": rule_sha,
           "base": {"successes": k_base, "n": n, "wilson95": wilson(k_base, n),
                    "falls": sum(bool(e["fell"]) for e in res["episodes"]["base"])},
           "dsrl": {"successes": k_dsrl, "n": n, "wilson95": wilson(k_dsrl, n),
                    "falls": sum(bool(e["fell"]) for e in res["episodes"]["dsrl"])},
           "paired": {"both": sum(d and c for d, c in zip(dsrl, base)), "dsrl_only": b, "base_only": c,
                      "neither": sum(not d and not c for d, c in zip(dsrl, base)),
                      "mcnemar_p_one_sided": pval, "alpha": rule["alpha"]},
           "reported_not_gated": {arm: {"successes": sum(bool(e["success"]) for e in eps), "n": len(eps)}
                                  for arm, eps in res["episodes"].items() if arm not in ("base", "dsrl")},
           "sac_ckpt_sha256": res["sac_ckpt_sha256"], "inputs": {"eval": str(a.eval_json)}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as f:
        json.dump(out, f, indent=1)
    print(f"WAITER PHASE 4 GATE: {verdict} (DSRL {k_dsrl}/{n} vs base {k_base}/{n}; DSRL only {b}, base only {c}; "
          f"one-sided exact McNemar p = {pval:.4f}, need DSRL > base and p <= {rule['alpha']}) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
