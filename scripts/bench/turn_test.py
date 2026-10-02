"""Does a gait checkpoint turn when told to? (MuJoCo, CPU, no render)

Measured the same way for every checkpoint: 1 s of standing, then a constant
body-frame command for `--seconds`; report the yaw turned, the displacement
and whether it fell. The three predeclared checks for a turning gait:

  turn    (0, 0, 0.6) rad/s for 6 s  ->  |yaw| >= 150 deg, no fall
  walk    (0.35, 0, 0) for 6 s       ->  |yaw drift| <= 15 deg, no fall
  arc     (0.2, 0, 0.8)              ->  reported, not gated

Exit status 0 = PASS on turn and walk, 1 = FAIL, 2 = could not run.

--protocol v1 (default) is the above, unchanged. --protocol v2 (2026-09-26)
tests turning from a SETTLED stand, because the v1 turn command arrives 1 s
after reset while some gaits are still stepping off the reset transient:

  turn    for reset seed in 0, 1, 2: 3.0 s standing, then (0, 0, +0.6) and
          (0, 0, -0.6) for 6 s each  ->  all six reach >= 150 deg IN THE
          COMMANDED DIRECTION with no fall
  walk    (0.35, 0, 0), 1.0 s warm-up, reset seed --seed (as v1)
          ->  |yaw drift| <= 15 deg, no fall

v2 PASS iff all six turn runs and the walk run pass; the JSON records every run.
(--warm is not used by v2; its warm-ups are fixed as above.)

--protocol v2x (2026-09-27, "v2-extended", single-checkpoint qualification) is v2
on more, FRESH reset seeds, with a count rule instead of all-must-pass:

  turn    for each reset seed in --turn-seeds (default 10 11 12 13 14): 3.0 s
          standing, then (0, 0, +0.6) and (0, 0, -0.6) for 6 s each
          ->  a run is ok iff >= 150 deg IN THE COMMANDED DIRECTION, no fall
  walk    (0.35, 0, 0), 1.0 s warm-up, for each reset seed in --walk-seeds
          (default 10 11 12)  ->  a run is ok iff |yaw drift| <= 15 deg, no fall

v2x PASS iff >= --min-turn-ok (default 9) turn runs are ok AND >= --min-walk-ok
(default 2) walk runs are ok. The default seeds (10+) are disjoint from the
seeds v1/v2 used (0-2), so a checkpoint inspected under v2 is scored on runs
nobody has looked at. v1 and v2 are unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]


def yaw_of(q):
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def run_command(deploy: Path, upstream: Path, cache: Path, variant: str, cmd, seconds: float, warm: float, seed: int):
    import mujoco
    from omegaconf import OmegaConf
    from bhl_robust.eval.gait_clock import make_controller   # RlController(cfg) unless deploy.yaml has a gait_clock block
    from bhl_robust.eval.multi_robot import build_multi
    from team_airlock import ContactRunner, CpuPolicy
    cfg = OmegaConf.load(deploy)
    policy = CpuPolicy(cfg.policy_checkpoint_path)
    model, slots = build_multi(upstream, cache / variant, 1, ["t"], variant=variant, world="flat")
    ctrl = make_controller(cfg)
    ctrl.policy = policy
    runner = ContactRunner(model, slots, [cfg], [ctrl])
    rng = np.random.default_rng(seed)
    runner.reset(rng)
    slot = slots[0]
    owners = np.array([0 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "").startswith(slot.prefix)
                       else -1 for g in range(model.ngeom)])
    runner.configure_contacts(owners)
    dt = float(cfg.policy_dt)
    yaw0 = yaw_of(runner.d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7])
    xy0 = runner.d.xpos[slot.body_id, :2].copy()
    fell, yaws = None, []
    for step in range(int((warm + seconds) / dt)):
        now = step * dt
        c = np.zeros(3) if now < warm else np.asarray(cmd, dtype=float)
        obs = runner.observe(0, c)
        runner.step([ctrl.update(obs)])
        yaws.append(yaw_of(runner.d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7]))
        if runner.tilt(0) >= 0.78:
            fell = now
            break
    dyaw = float(np.unwrap(np.array(yaws))[-1] - yaw0)
    xy1 = runner.d.xpos[slot.body_id, :2]
    return {"cmd": [float(v) for v in cmd], "fell_at_s": fell, "yaw_deg": round(math.degrees(dyaw), 1),
            "displacement_m": round(float(np.linalg.norm(xy1 - xy0)), 3), "seconds": seconds}


V2_TURN_SEEDS = (0, 1, 2)
V2_TURN_WZ = 0.6
V2_TURN_WARM_S = 3.0
V2_WALK_CMD = (0.35, 0.0, 0.0)
V2_WALK_WARM_S = 1.0


def v2_judge(turns: list[dict], walk: dict, turn_min_deg: float, drift_max_deg: float) -> dict:
    """Pure v2 rule: every turn run turns >= turn_min_deg in the sign of its commanded wz without
    falling, and the walk run drifts <= drift_max_deg without falling."""
    for r in turns:
        r["ok"] = bool(r["fell_at_s"] is None and r["yaw_deg"] * math.copysign(1.0, r["cmd"][2]) >= turn_min_deg)
    walk["ok"] = bool(walk["fell_at_s"] is None and abs(walk["yaw_deg"]) <= drift_max_deg)
    turn_ok = bool(turns) and all(r["ok"] for r in turns)
    return {"turn_ok": turn_ok, "walk_ok": walk["ok"], "n_turn_ok": sum(r["ok"] for r in turns),
            "n_turn": len(turns), "verdict": "PASS" if (turn_ok and walk["ok"]) else "FAIL"}


def main_v2(args) -> int:
    try:
        turns = []
        for seed in V2_TURN_SEEDS:
            for sign in (1.0, -1.0):
                r = run_command(args.deploy, args.upstream, args.cache_dir, args.variant, (0.0, 0.0, sign * V2_TURN_WZ),
                                args.seconds, V2_TURN_WARM_S, seed)
                turns.append({"name": f"turn{'+' if sign > 0 else '-'}_s{seed}", "seed": seed, "warm_s": V2_TURN_WARM_S, **r})
                print(f"  {turns[-1]['name']}: yaw {r['yaw_deg']} deg, fell {r['fell_at_s']}", flush=True)
        walk = {"name": f"walk_s{args.seed}", "seed": args.seed, "warm_s": V2_WALK_WARM_S,
                **run_command(args.deploy, args.upstream, args.cache_dir, args.variant, V2_WALK_CMD,
                              args.seconds, V2_WALK_WARM_S, args.seed)}
        print(f"  {walk['name']}: drift {walk['yaw_deg']} deg, fell {walk['fell_at_s']}", flush=True)
    except Exception as exc:                                        # noqa: BLE001
        print(f"TURN-TEST RESULT: ERROR {exc!r}")
        return 2
    judged = v2_judge(turns, walk, args.turn_min_deg, args.drift_max_deg)
    out = {"deploy": str(args.deploy), "variant": args.variant, "protocol": "v2", "turns": turns, "walk": walk, **judged,
           "rule": {"turn_min_deg": args.turn_min_deg, "drift_max_deg": args.drift_max_deg, "seconds": args.seconds,
                    "turn_seeds": list(V2_TURN_SEEDS), "turn_wz": V2_TURN_WZ, "turn_warm_s": V2_TURN_WARM_S,
                    "walk_cmd": list(V2_WALK_CMD), "walk_warm_s": V2_WALK_WARM_S, "walk_seed": args.seed}}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"TURN-TEST v2 RESULT: {judged['verdict']} (turns ok {judged['n_turn_ok']}/{judged['n_turn']}: "
          f"{[r['yaw_deg'] for r in turns]} deg; walk drift {walk['yaw_deg']} deg; "
          f"falls {[r['fell_at_s'] for r in turns + [walk]]})")
    return 0 if judged["verdict"] == "PASS" else 1


V2X_TURN_SEEDS = (10, 11, 12, 13, 14)
V2X_WALK_SEEDS = (10, 11, 12)


def v2x_judge(turns: list[dict], walks: list[dict], turn_min_deg: float, drift_max_deg: float,
              min_turn_ok: int, min_walk_ok: int) -> dict:
    """Pure v2x rule: count turn runs that turn >= turn_min_deg in the sign of their commanded wz without
    falling, and walk runs that drift <= drift_max_deg without falling; PASS iff both counts reach their bars."""
    for r in turns:
        r["ok"] = bool(r["fell_at_s"] is None and r["yaw_deg"] * math.copysign(1.0, r["cmd"][2]) >= turn_min_deg)
    for w in walks:
        w["ok"] = bool(w["fell_at_s"] is None and abs(w["yaw_deg"]) <= drift_max_deg)
    n_t, n_w = sum(r["ok"] for r in turns), sum(w["ok"] for w in walks)
    turn_ok, walk_ok = bool(turns) and n_t >= min_turn_ok, bool(walks) and n_w >= min_walk_ok
    return {"turn_ok": turn_ok, "walk_ok": walk_ok, "n_turn_ok": n_t, "n_turn": len(turns), "n_walk_ok": n_w,
            "n_walk": len(walks), "n_fell": sum(r["fell_at_s"] is not None for r in turns + walks),
            "verdict": "PASS" if (turn_ok and walk_ok) else "FAIL"}


def main_v2x(args) -> int:
    turn_seeds = tuple(args.turn_seeds) if args.turn_seeds else V2X_TURN_SEEDS
    walk_seeds = tuple(args.walk_seeds) if args.walk_seeds else V2X_WALK_SEEDS
    try:
        turns, walks = [], []
        for seed in turn_seeds:
            for sign in (1.0, -1.0):
                r = run_command(args.deploy, args.upstream, args.cache_dir, args.variant, (0.0, 0.0, sign * V2_TURN_WZ),
                                args.seconds, V2_TURN_WARM_S, seed)
                turns.append({"name": f"turn{'+' if sign > 0 else '-'}_s{seed}", "seed": seed, "warm_s": V2_TURN_WARM_S, **r})
                print(f"  {turns[-1]['name']}: yaw {r['yaw_deg']} deg, fell {r['fell_at_s']}", flush=True)
        for seed in walk_seeds:
            walks.append({"name": f"walk_s{seed}", "seed": seed, "warm_s": V2_WALK_WARM_S,
                          **run_command(args.deploy, args.upstream, args.cache_dir, args.variant, V2_WALK_CMD,
                                        args.seconds, V2_WALK_WARM_S, seed)})
            print(f"  {walks[-1]['name']}: drift {walks[-1]['yaw_deg']} deg, fell {walks[-1]['fell_at_s']}", flush=True)
    except Exception as exc:                                        # noqa: BLE001
        print(f"TURN-TEST RESULT: ERROR {exc!r}")
        return 2
    judged = v2x_judge(turns, walks, args.turn_min_deg, args.drift_max_deg, args.min_turn_ok, args.min_walk_ok)
    out = {"deploy": str(args.deploy), "variant": args.variant, "protocol": "v2x", "turns": turns, "walks": walks, **judged,
           "rule": {"turn_min_deg": args.turn_min_deg, "drift_max_deg": args.drift_max_deg, "seconds": args.seconds,
                    "turn_seeds": list(turn_seeds), "turn_wz": V2_TURN_WZ, "turn_warm_s": V2_TURN_WARM_S,
                    "walk_cmd": list(V2_WALK_CMD), "walk_warm_s": V2_WALK_WARM_S, "walk_seeds": list(walk_seeds),
                    "min_turn_ok": args.min_turn_ok, "min_walk_ok": args.min_walk_ok}}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"TURN-TEST v2x RESULT: {judged['verdict']} (turns ok {judged['n_turn_ok']}/{judged['n_turn']} need {args.min_turn_ok}: "
          f"{[r['yaw_deg'] for r in turns]} deg; walks ok {judged['n_walk_ok']}/{judged['n_walk']} need {args.min_walk_ok}: "
          f"{[w['yaw_deg'] for w in walks]} deg; falls {judged['n_fell']})")
    return 0 if judged["verdict"] == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deploy", type=Path, required=True)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--variant", choices=("biped", "humanoid"), required=True)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--warm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--turn-min-deg", type=float, default=150.0)
    ap.add_argument("--drift-max-deg", type=float, default=15.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--protocol", choices=("v1", "v2", "v2x"), default="v1",
                    help="v1 = the original three-command test (default); v2 = settled-stand turns; "
                         "v2x = v2 on fresh seeds with a count rule (see above)")
    ap.add_argument("--turn-seeds", type=int, nargs="+", default=None, help="v2x only (default 10 11 12 13 14)")
    ap.add_argument("--walk-seeds", type=int, nargs="+", default=None, help="v2x only (default 10 11 12)")
    ap.add_argument("--min-turn-ok", type=int, default=9, help="v2x only: turn runs that must be ok")
    ap.add_argument("--min-walk-ok", type=int, default=2, help="v2x only: walk runs that must be ok")
    args = ap.parse_args()
    if args.protocol == "v2":
        return main_v2(args)
    if args.protocol == "v2x":
        return main_v2x(args)
    try:
        res = {name: run_command(args.deploy, args.upstream, args.cache_dir, args.variant, cmd, args.seconds, args.warm, args.seed)
               for name, cmd in (("turn", (0.0, 0.0, 0.6)), ("walk", (0.35, 0.0, 0.0)), ("arc", (0.2, 0.0, 0.8)))}
    except Exception as exc:                                        # noqa: BLE001
        print(f"TURN-TEST RESULT: ERROR {exc!r}")
        return 2
    turn_ok = res["turn"]["fell_at_s"] is None and abs(res["turn"]["yaw_deg"]) >= args.turn_min_deg
    walk_ok = res["walk"]["fell_at_s"] is None and abs(res["walk"]["yaw_deg"]) <= args.drift_max_deg
    verdict = "PASS" if (turn_ok and walk_ok) else "FAIL"
    out = {"deploy": str(args.deploy), "variant": args.variant, "results": res, "turn_ok": turn_ok, "walk_ok": walk_ok,
           "verdict": verdict, "rule": {"turn_min_deg": args.turn_min_deg, "drift_max_deg": args.drift_max_deg, "seconds": args.seconds}}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("results", "turn_ok", "walk_ok")}))
    print(f"TURN-TEST RESULT: {verdict} (turn {res['turn']['yaw_deg']} deg, walk drift {res['walk']['yaw_deg']} deg, "
          f"falls {[res[k]['fell_at_s'] for k in ('turn', 'walk', 'arc')]})")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
