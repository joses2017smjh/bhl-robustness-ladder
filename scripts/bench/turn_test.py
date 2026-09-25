"""Does a gait checkpoint turn when told to? (MuJoCo, CPU, no render)

Measured the same way for every checkpoint: 1 s of standing, then a constant
body-frame command for `--seconds`; report the yaw turned, the displacement
and whether it fell. The three predeclared checks for a turning gait:

  turn    (0, 0, 0.6) rad/s for 6 s  ->  |yaw| >= 150 deg, no fall
  walk    (0.35, 0, 0) for 6 s       ->  |yaw drift| <= 15 deg, no fall
  arc     (0.2, 0, 0.8)              ->  reported, not gated

Exit status 0 = PASS on turn and walk, 1 = FAIL, 2 = could not run.
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
    from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
    from bhl_robust.eval.multi_robot import build_multi
    from team_airlock import ContactRunner, CpuPolicy
    cfg = OmegaConf.load(deploy)
    policy = CpuPolicy(cfg.policy_checkpoint_path)
    model, slots = build_multi(upstream, cache / variant, 1, ["t"], variant=variant, world="flat")
    ctrl = RlController(cfg)
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
    args = ap.parse_args()
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
