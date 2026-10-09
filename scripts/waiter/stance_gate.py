#!/usr/bin/env python3
"""Waiter phase 1d gate Q3, standing still while the arms work (SLURM_JOBS.md "Predeclared now ... Waiter phase 1d";
frozen 2026-10-09, before any 1d checkpoint exists).

12 episodes = the 4 Q2 arm trajectories (q2_trajectories.TRAJECTORIES) x reset seeds 0-2, on the MuJoCo Waiter model
with 0.5 kg added to each hand link, base command 0 for 8 s, the gait-clock input HELD at clock step 10 (phase 0.5)
from the first time the clock reaches it: WaiterEnv's stance hold (bhl_robust/eval/waiter_env.py). Drift = base
horizontal displacement and yaw change between t = 1.0 s and t = 8.0 s. Fall = tilt >= 0.78 rad (turn_test's rule).
PASS iff 0 falls in 12 AND drift <= 0.05 m AND |yaw drift| <= 5 deg on >= 10 of the 12.
Same runner path as q2_gate.py (build_multi, ContactRunner, CpuPolicy, gait_clock.make_controller).

usage: stance_gate.py --deploy <deploy.yaml> --upstream <Berkeley-Humanoid-Lite> --cache-dir <dir> --out <json>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parents[1] / "scripts/bench"))
sys.path.insert(0, str(HERE))
import q2_gate as Q  # noqa: E402
import q2_trajectories as T  # noqa: E402

HOLD_STEP = 10                     # clock step (of 20) held: phase 0.5 (gait_clock_mdp.STANCE_PHASE)
SEEDS = (0, 1, 2)
EPISODE_S, T0_S = 8.0, 1.0
MAX_DRIFT_M, MAX_YAW_DEG, MIN_OK = 0.05, 5.0, 10
RULE = ("PASS iff 0 falls in 12 (4 Q2 arm trajectories x seeds 0-2, 0.5 kg per hand, command 0, clock held at "
        "phase 0.5) and base drift <= 0.05 m and |yaw drift| <= 5 deg (t 1-8 s) on >= 10 of the 12")


def episode(cfg, slots, ctrl, runner, traj, seed: int) -> dict:
    runner.reset(np.random.default_rng(seed))
    ctrl.reset_phase()                      # the clock starts at 0 every episode (as WaiterEnv.reset)
    slot = slots[0]
    dt = float(cfg.policy_dt)
    held, p0, y0, fell = None, None, None, None
    for k in range(int(round(EPISODE_S / dt))):
        t = k * dt
        ctrl.set_upper_body(traj(t))
        if held is None and ctrl.clock_step % 20 == HOLD_STEP:
            held = HOLD_STEP
        if held is not None:
            ctrl.clock_step = held
        runner.step([ctrl.update(runner.observe(0, np.zeros(3)))])
        if p0 is None and t >= T0_S:
            p0 = runner.d.qpos[slot.qpos_adr:slot.qpos_adr + 2].copy()
            y0 = Q.yaw_of(runner.d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7])
        if runner.tilt(0) >= Q.TILT_FALL:
            fell = round(t, 2)
            break
    if fell is not None:
        return {"seed": seed, "fell_at_s": fell, "drift_m": None, "yaw_drift_deg": None, "ok": False}
    drift = float(np.linalg.norm(runner.d.qpos[slot.qpos_adr:slot.qpos_adr + 2] - p0))
    yaw = math.degrees(math.remainder(Q.yaw_of(runner.d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7]) - y0, 2 * math.pi))
    return {"seed": seed, "fell_at_s": None, "drift_m": round(drift, 4), "yaw_drift_deg": round(yaw, 2),
            "ok": drift <= MAX_DRIFT_M and abs(yaw) <= MAX_YAW_DEG}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deploy", type=Path, required=True)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"Q3: REFUSED ({a.out} exists; never overwritten)")
    cfg, model, slots, ctrl, runner = Q.make(a.deploy, a.upstream, a.cache_dir)
    eps = []
    for name, traj in T.TRAJECTORIES.items():
        for seed in SEEDS:
            r = episode(cfg, slots, ctrl, runner, traj, seed)
            r["trajectory"] = name
            eps.append(r)
            print(json.dumps(r), flush=True)
    falls = sum(e["fell_at_s"] is not None for e in eps)
    n_ok = sum(e["ok"] for e in eps)
    verdict = "PASS" if falls == 0 and n_ok >= MIN_OK else "FAIL"
    out = {"deploy": str(a.deploy), "gate": "Q3 stance", "rule": RULE, "verdict": verdict, "falls": falls,
           "n": len(eps), "n_ok": n_ok, "episodes": eps, "payload_kg_per_hand": Q.PAYLOAD_KG, "hold_step": HOLD_STEP}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"WAITER Q3: {verdict} (falls {falls}/12, still {n_ok}/12; need 0 and >= {MIN_OK}) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
