#!/usr/bin/env python3
"""Waiter phase 1 gate Q2, upper body under load (docs/WAITER_PROGRAM.md; frozen 2026-10-05).

12 episodes = 4 arm trajectories (q2_trajectories.TRAJECTORIES) x reset seeds 0-2, on the MuJoCo Waiter model with
0.5 kg added to each hand link: 3 s standing (command 0) while the trajectory runs, then 5 s walking at
(0.25, 0, 0) while it continues. Fall = tilt >= 0.78 rad (turn_test's criterion). Heading drift = yaw at 8 s minus
yaw at 3 s. PASS iff 0 falls in 12 AND |drift| <= 15 deg on >= 10 of the 12 walking segments.
Same runner path as turn_test.py (build_multi, ContactRunner, CpuPolicy, gait_clock.make_controller).
Reported, not gated: arm tracking error (mean |q - target| over the 10 arm joints) and a push-with-load reading
(T1 carry pose held, 0.5 kg per hand, 12 s standing with 0.5 m/s shoves every 3 s, reset seeds 20-39).

usage: q2_gate.py --deploy <deploy.yaml> --upstream <Berkeley-Humanoid-Lite> --cache-dir <dir> --out <json>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/bench"))
sys.path.insert(0, str(HERE))
import q2_trajectories as T  # noqa: E402

PAYLOAD_KG = 0.5
SEEDS = (0, 1, 2)
STAND_S, WALK_S = 3.0, 5.0
WALK_CMD = (0.25, 0.0, 0.0)
TILT_FALL = 0.78
DRIFT_MAX_DEG = 15.0
MIN_DRIFT_OK = 10
PUSH_SEEDS = tuple(range(20, 40))
PUSH_SPEED, PUSH_EVERY_S, PUSH_EPISODE_S = 0.5, 3.0, 12.0
RULE = ("PASS iff 0 falls in 12 (4 trajectories x seeds 0-2, 0.5 kg per hand, 3 s stand + 5 s walk at 0.25 m/s) "
        "and |heading drift| <= 15 deg on >= 10 of the 12 walking segments")


def yaw_of(q):
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def make(deploy: Path, upstream: Path, cache: Path):
    import mujoco
    from omegaconf import OmegaConf
    from bhl_robust.eval.gait_clock import make_controller
    from bhl_robust.eval.multi_robot import build_multi
    from team_airlock import ContactRunner, CpuPolicy
    cfg = OmegaConf.load(deploy)
    if "waiter_wbc" not in cfg:
        raise SystemExit("Q2: REFUSED (not a stamped Waiter WBC deploy.yaml)")
    model, slots = build_multi(upstream, cache / "waiter", 1, ["t"], variant="waiter", world="flat")
    for side in ("left", "right"):
        model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"r0_arm_{side}_hand_link")] += PAYLOAD_KG
    ctrl = make_controller(cfg)
    ctrl.policy = CpuPolicy(cfg.policy_checkpoint_path)
    runner = ContactRunner(model, slots, [cfg], [ctrl])
    runner.configure_contacts(np.zeros(model.ngeom, dtype=int))
    return cfg, model, slots, ctrl, runner


def episode(cfg, slots, ctrl, runner, traj, seed: int) -> dict:
    runner.reset(np.random.default_rng(seed))
    slot = slots[0]
    dt = float(cfg.policy_dt)
    n = int(round((STAND_S + WALK_S) / dt))
    fell, yaw_walk0, errs = None, None, []
    for k in range(n):
        t = k * dt
        ub = traj(t)
        ctrl.set_upper_body(ub)
        cmd = np.zeros(3) if t < STAND_S else np.asarray(WALK_CMD)
        if yaw_walk0 is None and t >= STAND_S:
            yaw_walk0 = yaw_of(runner.d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7])
        runner.step([ctrl.update(runner.observe(0, cmd))])
        q = runner.d.sensordata[slot.jpos_adr]
        errs.append(float(np.mean(np.abs(q[:10] - ub[:10]))))
        if runner.tilt(0) >= TILT_FALL:
            fell = round(t, 2)
            break
    drift = None
    if fell is None:
        drift = math.degrees(math.remainder(yaw_of(runner.d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7]) - yaw_walk0,
                                            2 * math.pi))
    return {"seed": seed, "fell_at_s": fell, "walk_drift_deg": None if drift is None else round(drift, 1),
            "drift_ok": drift is not None and abs(drift) <= DRIFT_MAX_DEG,
            "arm_track_err_rad": round(float(np.mean(errs)), 4)}


def push_with_load(cfg, slots, ctrl, runner) -> dict:
    dt = float(cfg.policy_dt)
    falls = 0
    for seed in PUSH_SEEDS:
        rng = np.random.default_rng(seed)
        runner.reset(rng)
        for k in range(int(round(PUSH_EPISODE_S / dt))):
            t = k * dt
            if k > 0 and k % int(round(PUSH_EVERY_S / dt)) == 0:
                runner.push_all(PUSH_SPEED, rng)
            ctrl.set_upper_body(T.t1_carry(t))
            runner.step([ctrl.update(runner.observe(0, np.zeros(3)))])
            if runner.tilt(0) >= TILT_FALL:
                falls += 1
                break
    return {"falls": falls, "n": len(PUSH_SEEDS), "protocol": "T1 carry held, 0.5 kg per hand, standing 12 s, "
            "0.5 m/s shoves every 3 s, reset seeds 20-39 (reported, not gated)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deploy", type=Path, required=True)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-push", action="store_true", help="skip the report-only push-with-load reading")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"Q2: REFUSED ({a.out} exists; never overwritten)")
    assert T.within_limits(), "a Q2 trajectory leaves the joint limits"
    cfg, model, slots, ctrl, runner = make(a.deploy, a.upstream, a.cache_dir)
    eps = []
    for name, traj in T.TRAJECTORIES.items():
        for seed in SEEDS:
            r = episode(cfg, slots, ctrl, runner, traj, seed)
            r["trajectory"] = name
            eps.append(r)
            print(json.dumps(r), flush=True)
    falls = sum(e["fell_at_s"] is not None for e in eps)
    drift_ok = sum(e["drift_ok"] for e in eps)
    verdict = "PASS" if falls == 0 and drift_ok >= MIN_DRIFT_OK else "FAIL"
    out = {"deploy": str(a.deploy), "gate": "Q2", "rule": RULE, "verdict": verdict, "falls": falls, "n": len(eps),
           "drift_ok": drift_ok, "episodes": eps, "payload_kg_per_hand": PAYLOAD_KG,
           "trajectories": T.__doc__,
           "push_with_load": None if a.no_push else push_with_load(cfg, slots, ctrl, runner)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"WAITER Q2: {verdict} (falls {falls}/12, drift ok {drift_ok}/12; need 0 and >= {MIN_DRIFT_OK}) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
