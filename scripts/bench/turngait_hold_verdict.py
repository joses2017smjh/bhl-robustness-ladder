"""Arm verdict and env.yaml recipe check of the turning follow-up R1H (slurm/repo20260923/gpu_turngait_hold.sbatch).

R1H = Velocity-BHL-Arms-TurnGaitClockHold-v0 = R1 (Velocity-BHL-Arms-TurnGaitClock-v0) + exactly two changes, both
chosen, not tuned (SLURM_JOBS.md "(C') revised design", 2026-10-02 14:44): (i) per resample 30 % of envs leave
heading control and take EXPLICIT commands (vx, vy, wz from R1's own ranges, wz = 0 exactly with probability 0.5);
(ii) the heading_hold reward, weight 1.0, exp(-(dpsi / 0.2 rad)^2), only in those explicit envs while |wz_cmd| < 0.05.

PREDECLARED RULE (frozen 2026-10-02, before any R1H run exists; v5's joint rule verbatim):
  PASS iff >= 2/3 seeds both PASS turn_test v2 AND are QUALIFIED by cpu_turn_qualify's unchanged rule (v2x >= 9/10
  on reset seeds 10-14, walk <= 15 deg on >= 2/3, push <= 9/60); else FAIL; INCOMPLETE if any JSON is missing. A seed
  that fails v2 does not count even if it qualifies through v2x. Training counts only if the run dir holds
  model_5999.pt and the training log shows the arm's task id, the feet_gait term, the heading_hold term, the
  explicit-command mix and the push event.
A new task, never presented as a fine-tune of R1. Labels: LEARNED gait; MuJoCo gates.
Disclosure: the arm changes the command mix (the training distribution), so it tests explicit straight-walk commands
+ the hold reward together, not the reward alone.

Read from JSON only (never exit codes): per seed <res>/training/<run>.json (written by the launcher from the fresh
training log), <res>/turn-test-v2/<run>.json (turn_test.py --protocol v2) and <res>/qualify/<run>__qualify.json
(cpu_turn_qualify.sbatch). An unreadable JSON, or one that is not the declared protocol / not this run, counts as
missing. The verdict JSON is written only when the arm is complete, and an existing one is never overwritten.

    python scripts/bench/turngait_hold_verdict.py verdict --res <results dir>
    python scripts/bench/turngait_hold_verdict.py check-recipe --env-yaml <R1H run>/params/env.yaml \
        --ref-env-yaml <arms-turngait-clock-s0 run>/params/env.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RULE = ("PASS iff >= 2/3 seeds both PASS turn_test v2 AND are QUALIFIED by cpu_turn_qualify's unchanged rule "
        "(v2x >= 9/10 on reset seeds 10-14, walk <= 15 deg on >= 2/3, push <= 9/60); else FAIL; INCOMPLETE if "
        "any JSON is missing. A seed that fails v2 does not count even if it qualifies through v2x.")
TRAINING_RULE = ("Training counts only if the run dir holds model_5999.pt and the training log shows the arm's task "
                 "id, the feet_gait term, the heading_hold term, the explicit-command mix and the push event.")
LABEL = "LEARNED gait; MuJoCo gates"
DISCLOSURE = ("the arm changes the command mix (the training distribution), so it tests explicit straight-walk "
              "commands + the hold reward together, not the reward alone.")
ARM = {"name": "R1H", "task": "Velocity-BHL-Arms-TurnGaitClockHold-v0", "prefix": "arms-turngait-hold",
       "what": "R1 + 30 % explicit-command envs (half at wz = 0) + heading_hold (weight 1.0, |wz_cmd| < 0.05)"}
SEEDS = (0, 1, 2)
FINAL_CKPT = "model_5999.pt"
# turn_test.py --protocol v2 as the launcher runs it (defaults; v5 and R1/R2 identical)
V2_RULE = {"turn_min_deg": 150.0, "drift_max_deg": 15.0, "seconds": 6.0, "turn_seeds": [0, 1, 2], "turn_wz": 0.6,
           "turn_warm_s": 3.0, "walk_cmd": [0.35, 0.0, 0.0], "walk_warm_s": 1.0, "walk_seed": 0}
NEED = 2
# The training clause's six flags, as the launcher writes them into the training record.
TRAINING_FLAGS = ("final_ckpt_present", "task_id_in_log", "feet_gait_in_log", "heading_hold_in_log",
                  "command_mix_in_log", "push_robot_in_log")


def training_counts(t: dict) -> bool:
    """The frozen training clause, from the launcher's training JSON."""
    return bool(t.get("final_ckpt") == FINAL_CKPT and all(t.get(k) is True for k in TRAINING_FLAGS))


def seed_counts(training_ok, v2, qualify) -> bool:
    """A seed counts iff its training counts, it PASSES v2 AND it is QUALIFIED."""
    return training_ok is True and v2 == "PASS" and qualify == "QUALIFIED"


def arm_verdict(seeds: list[dict | None]) -> dict:
    """Pure rule. seeds: one entry per seed (exactly 3), each {"training": bool | None, "v2": "PASS" | "FAIL" | None,
    "qualify": "QUALIFIED" | "NOT QUALIFIED" | None}; None = that JSON is missing (or the whole seed is None)."""
    if len(seeds) != len(SEEDS):
        raise ValueError(f"expected {len(SEEDS)} seeds, got {len(seeds)}")
    rows = [s or {"training": None, "v2": None, "qualify": None} for s in seeds]
    missing = [i for i, s in enumerate(rows) if s.get("training") is None or s.get("v2") is None or s.get("qualify") is None]
    k = sum(seed_counts(s.get("training"), s.get("v2"), s.get("qualify")) for s in rows)
    if missing:
        return {"verdict": "INCOMPLETE", "n_counted": k, "need": NEED, "of": len(SEEDS), "missing_seeds": missing}
    return {"verdict": "PASS" if k >= NEED else "FAIL", "n_counted": k, "need": NEED, "of": len(SEEDS),
            "missing_seeds": []}


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:                                               # noqa: BLE001
        return None


def read_seed(res: Path, run: str) -> dict:
    """{training, v2, qualify} for one run from its three JSONs (None = missing / unreadable / not this run)."""
    t = _read(res / "training" / f"{run}.json")
    training = training_counts(t) if (isinstance(t, dict) and t.get("run") == run) else None
    d = _read(res / "turn-test-v2" / f"{run}.json")
    v2 = None
    if isinstance(d, dict) and d.get("protocol") == "v2" and f"_{run}/" in str(d.get("deploy", "")) \
            and all((d.get("rule") or {}).get(k) == v for k, v in V2_RULE.items()) and d.get("verdict") in ("PASS", "FAIL"):
        v2 = d["verdict"]
    q = _read(res / "qualify" / f"{run}__qualify.json")
    qualify = q["verdict"] if (isinstance(q, dict) and q.get("run") == run
                               and q.get("verdict") in ("QUALIFIED", "NOT QUALIFIED")) else None
    return {"run": run, "training": training, "v2": v2, "qualify": qualify,
            "push": (q.get("clauses") or {}).get("push") if isinstance(q, dict) else None}


def write_verdict(res: Path, out: Path | None = None) -> int:
    out = out or res / "verdict" / f"{ARM['name']}.json"
    if out.exists():
        d = _read(out) or {}
        print(f"ARM-HOLD {ARM['name']}: {d.get('verdict')} (already recorded, not overwritten: {out})")
        return 0
    rows = [read_seed(res, f"{ARM['prefix']}-s{s}") for s in SEEDS]
    v = arm_verdict(rows)
    rec = {"arm": ARM["name"], "task": ARM["task"], "what": ARM["what"], **v, "seeds": rows, "rule": RULE,
           "training_rule": TRAINING_RULE, "label": LABEL, "disclosure": DISCLOSURE,
           "note": "a new task trained from scratch (no parent checkpoint); never presented as a fine-tune of R1"}
    print(f"ARM-HOLD {ARM['name']} ({ARM['task']}): {v['verdict']} ({v['n_counted']}/3 seeds count, need {NEED}; "
          f"per seed (training, v2, qualify) {[(r['training'], r['v2'], r['qualify']) for r in rows]}) [{LABEL}]")
    if v["verdict"] != "INCOMPLETE":
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(out, "x") as f:                # O_EXCL: two array tasks finishing together cannot both write
                f.write(json.dumps(rec, indent=2) + "\n")
            print(f"wrote {out}")
        except FileExistsError:
            print(f"verdict already recorded by another task, not overwritten: {out}")
    return 0


# ---------------------------------------------------------------- the env.yaml recipe: exactly R1 + two changes
# Mirrors arms_env_cfg.HumanoidTurnGaitClockHoldCfg, turn_command.TurnHoldMixVelocityCommandCfg and
# gait_clock_mdp.HOLD_*. An R1H run's params/env.yaml must be R1's -- the reference is an R1 run's own
# params/env.yaml (arms-turngait-clock-s0) -- plus exactly two changes:
#   (i)  commands.base_velocity is TurnHoldMixVelocityCommand: class_type and the cfg keys HOLD_NEW_COMMAND_KEYS
#        with the values below; every key it shares with R1's UniformVelocityCommandCfg is unchanged;
#   (ii) the new rewards.heading_hold term.
# R1's own recipe (bhl_robust.eval.gait_clock.check_recipe(env, "R1"), unchanged) must still hold on it, and the
# reference must itself be an R1 env.yaml. Kept here, not in eval/gait_clock.py, so that file (a hash-frozen
# source of the Mission 7 clocks2 replay snapshot) is not touched by this workstream.
HOLD_RECIPE = {
    "command_class": "bhl_robust.tasks.turn_command:TurnHoldMixVelocityCommand",
    "rel_pure_turn_envs": 0.0,
    "rel_direct_envs": 0.3,
    "direct_zero_wz_prob": 0.5,
    "pure_turn_ang_vel_abs": [0.3, 1.0],         # TurnMix's default, never read (rel_pure_turn_envs = 0)
    "direct_ranges": {"direct_lin_vel_x": "lin_vel_x", "direct_lin_vel_y": "lin_vel_y", "direct_ang_vel_z": "ang_vel_z"},
    "heading_mode": {"heading_command": True, "rel_heading_envs": 1.0, "rel_standing_envs": 0.02},   # R1's, kept
    "heading_hold": {"func": "bhl_robust.tasks.gait_clock_mdp:heading_hold", "weight": 1.0,
                     "params": {"command_name": "base_velocity", "std": 0.2, "wz_threshold": 0.05}},
}
HOLD_NEW_COMMAND_KEYS = ("rel_pure_turn_envs", "rel_direct_envs", "pure_turn_ang_vel_abs", "direct_lin_vel_x",
                         "direct_lin_vel_y", "direct_ang_vel_z", "direct_zero_wz_prob")
HOLD_CHANGES_VS_R1 = frozenset({("commands", "base_velocity", "class_type"), ("rewards", "heading_hold")}
                               | {("commands", "base_velocity", k) for k in HOLD_NEW_COMMAND_KEYS})
R1_COMMAND_CLASS = "isaaclab.envs.mdp.commands.velocity_command:UniformVelocityCommand"
# the env.yaml sections compared (eval/gait_clock.RECIPE_SECTIONS, R1/R2's check)
RECIPE_SECTIONS = ("rewards", "events", "commands", "observations", "terminations", "actions", "curriculum")


def _num_eq(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) <= 1e-9
    except (TypeError, ValueError):
        return False


def _list_eq(a, b) -> bool:
    try:
        return len(a) == len(b) and all(_num_eq(x, y) for x, y in zip(a, b))
    except TypeError:
        return False


def _r1_recipe(env: dict) -> list[str]:
    from bhl_robust.eval import gait_clock as gc          # numpy-only import; R1's frozen recipe check, unchanged
    return gc.check_recipe(env, "R1")


def check_recipe_hold(env: dict) -> list[str]:
    """Problems with an R1H run's env.yaml against the frozen R1H changes and R1's own recipe; [] = the recipe."""
    p = [f"R1 recipe: {x}" for x in _r1_recipe(env)]
    cmd = (env.get("commands") or {}).get("base_velocity") or {}
    want = HOLD_RECIPE
    if str(cmd.get("class_type")) != want["command_class"]:
        p.append(f"command class_type {cmd.get('class_type')!r} != {want['command_class']!r}")
    for k in ("rel_pure_turn_envs", "rel_direct_envs", "direct_zero_wz_prob"):
        if not _num_eq(cmd.get(k), want[k]):
            p.append(f"command {k} {cmd.get(k)!r} != {want[k]}")
    if not _list_eq(cmd.get("pure_turn_ang_vel_abs") or [], want["pure_turn_ang_vel_abs"]):
        p.append(f"command pure_turn_ang_vel_abs {cmd.get('pure_turn_ang_vel_abs')!r} != {want['pure_turn_ang_vel_abs']}")
    ranges = cmd.get("ranges") or {}
    for dk, rk in want["direct_ranges"].items():
        if not (ranges.get(rk) is not None and _list_eq(cmd.get(dk) or [], ranges[rk])):
            p.append(f"command {dk} {cmd.get(dk)!r} != the task's own ranges.{rk} {ranges.get(rk)!r}")
    hm = want["heading_mode"]
    if cmd.get("heading_command") is not True:
        p.append(f"command heading_command {cmd.get('heading_command')!r} (the other envs must keep heading mode)")
    for k in ("rel_heading_envs", "rel_standing_envs"):
        if not _num_eq(cmd.get(k), hm[k]):
            p.append(f"command {k} {cmd.get(k)!r} != {hm[k]}")
    h = (env.get("rewards") or {}).get("heading_hold") or {}
    wh = want["heading_hold"]
    if str(h.get("func")) != wh["func"]:
        p.append(f"heading_hold func {h.get('func')!r}")
    if not _num_eq(h.get("weight"), wh["weight"]):
        p.append(f"heading_hold weight {h.get('weight')!r} != {wh['weight']}")
    hp = h.get("params") or {}
    if sorted(hp) != sorted(wh["params"]):
        p.append(f"heading_hold params {sorted(hp)} != {sorted(wh['params'])}")
    if hp.get("command_name") != wh["params"]["command_name"]:
        p.append(f"heading_hold command_name {hp.get('command_name')!r}")
    for k in ("std", "wz_threshold"):
        if not _num_eq(hp.get(k), wh["params"][k]):
            p.append(f"heading_hold {k} {hp.get(k)!r} != {wh['params'][k]}")
    return p


def check_hold_reference(ref: dict) -> list[str]:
    """Problems with the reference env.yaml as an R1 run's (R1's recipe, upstream command, no heading_hold)."""
    p = [f"reference is not R1: {x}" for x in _r1_recipe(ref)]
    cls = str(((ref.get("commands") or {}).get("base_velocity") or {}).get("class_type"))
    if cls != R1_COMMAND_CLASS:
        p.append(f"reference command class_type {cls!r} != {R1_COMMAND_CLASS!r}")
    if "heading_hold" in (ref.get("rewards") or {}):
        p.append("reference already has a heading_hold term")
    return p


def recipe_diff_hold(env: dict, ref_r1: dict) -> list[str]:
    """Paths in RECIPE_SECTIONS where env differs from the R1 reference env.yaml, other than the two declared R1H
    changes (HOLD_CHANGES_VS_R1), plus any declared change that is missing. [] = R1 + those two changes ONLY."""
    allowed = HOLD_CHANGES_VS_R1
    out: list[str] = []

    def walk(a, b, path):
        if path in allowed:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b), key=str):
                if k not in a or k not in b:
                    if path + (k,) not in allowed:
                        out.append(".".join(map(str, path + (k,))) + (" (new)" if k not in b else " (missing)"))
                else:
                    walk(a[k], b[k], path + (k,))
        elif a != b:
            out.append(".".join(map(str, path)) + f": {a!r} != R1 {b!r}")

    for sec in RECIPE_SECTIONS:
        walk(env.get(sec), ref_r1.get(sec), (sec,))
    cmd = (env.get("commands") or {}).get("base_velocity") or {}
    for k in HOLD_NEW_COMMAND_KEYS:
        if k not in cmd:
            out.append(f"commands.base_velocity.{k} (declared change missing)")
    if "heading_hold" not in (env.get("rewards") or {}):
        out.append("rewards.heading_hold (declared change missing)")
    return out


def check_recipe_cli(env_yaml: Path, ref_env_yaml: Path) -> int:
    """HOLD RECIPE: OK iff the run's env.yaml = the R1 reference's + the two declared R1H changes only."""
    try:
        from bhl_robust.eval import gait_clock as gc
        env, ref = gc.load_env_yaml(env_yaml), gc.load_env_yaml(ref_env_yaml)
        probs = check_recipe_hold(env) + check_hold_reference(ref)
        probs += [f"differs from R1: {d}" for d in recipe_diff_hold(env, ref)]
    except Exception as exc:                                        # noqa: BLE001
        print(f"HOLD RECIPE: FAIL {env_yaml}: {exc!r}")
        return 1
    if probs:
        print(f"HOLD RECIPE: FAIL {env_yaml}: " + "; ".join(probs[:12])
              + (f" (+{len(probs) - 12} more)" if len(probs) > 12 else ""))
        return 1
    print(f"HOLD RECIPE: OK {env_yaml} (= R1 {ref_env_yaml} + the two declared changes only)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verdict", help="the arm verdict from the per-seed JSONs (written once)")
    v.add_argument("--res", type=Path, required=True, help="the launcher's results directory")
    v.add_argument("--out", type=Path, default=None, help="verdict JSON (default <res>/verdict/R1H.json)")
    r = sub.add_parser("check-recipe", help="R1H env.yaml = R1's + the two declared changes only")
    r.add_argument("--env-yaml", type=Path, required=True)
    r.add_argument("--ref-env-yaml", type=Path, required=True, help="an R1 run's params/env.yaml (arms-turngait-clock-s0)")
    args = ap.parse_args(argv)
    if args.cmd == "verdict":
        return write_verdict(args.res, args.out)
    return check_recipe_cli(args.env_yaml, args.ref_env_yaml)


if __name__ == "__main__":
    sys.exit(main())
