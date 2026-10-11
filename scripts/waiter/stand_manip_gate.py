#!/usr/bin/env python3
"""Waiter phase 1e gate Q4, standing still while the arms do the phase 2 expert's work (SLURM_JOBS.md "Predeclared now
... Waiter phase 1e"; frozen 2026-10-10, before any 1e checkpoint exists).

12 episodes = 4 arm trajectories x reset seeds 0-2, on the MuJoCo Waiter model with 0.5 kg added to each hand (as
Q2/Q3), base command 0, the gait-clock input HELD at clock step 10 (phase 0.5) from the first time the clock reaches
it (WaiterEnv's stance hold, as Q3), FREE base (no stance lock). The trajectories move the arms through the phase 2
scripted expert's key poses, computed once by its IK (base upright at the station, nominal layout;
campaign-waiter/phase2/dev2/t_q4poses.py) and frozen here:
  P1 tumbler-left:  left arm to the tumbler insertion, lift, over the tray, back (right arm at default);
  P2 tumbler-right: the same with the right arm (mirrored by negation);
  P3 tray-hook:     both arms to the tray-handle insertion, then lifted, held;
  P4 pick-then-hook: P1's carry, then P3 (16 s).
Measured from t = 1.0 s to the end: max horizontal base displacement, max |yaw change| and max |pitch change|,
each relative to the base pose at t = 1.0 s. Fall = tilt >= 0.78 rad.
PASS iff 0 falls in 12 AND displacement <= 0.03 m AND |yaw| <= 3 deg AND |pitch| <= 3 deg on >= 10 of the 12.
Same runner path as q2_gate.py / stance_gate.py (build_multi, ContactRunner, CpuPolicy, gait_clock.make_controller).

usage: stand_manip_gate.py --deploy <deploy.yaml> --upstream <Berkeley-Humanoid-Lite> --cache-dir <dir> --out <json>
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

HOLD_STEP = 10                     # clock step (of 20) held: phase 0.5 (gait_clock_mdp.STANCE_PHASE)
SEEDS = (0, 1, 2)
T0_S = 1.0
MAX_DISP_M, MAX_YAW_DEG, MAX_PITCH_DEG, MIN_OK = 0.03, 3.0, 3.0, 10
# the expert's left-arm poses (rad; deploy order shoulder pitch, roll, yaw, elbow pitch, roll); right = -left
POSE = {"tumbler_ins": [-0.185, 0.323, -0.128, 1.272, -0.541], "tumbler_lift": [-0.184, 0.583, -0.015, 1.416, -0.785],
        "tumbler_over_tray": [-0.608, -0.084, 0.785, 1.447, -0.543], "hook_ins": [-0.848, 0.126, 0.59, 0.902, -0.675],
        "hook_lift": [-0.959, 0.219, 0.692, 1.005, -0.785]}
DEFAULT = [0.0] * 5
RULE = ("PASS iff 0 falls in 12 (4 expert-pose arm trajectories x seeds 0-2, 0.5 kg per hand, command 0, clock held at "
        "phase 0.5, free base) and max displacement <= 0.03 m, max |yaw| <= 3 deg and max |pitch| <= 3 deg (t >= 1 s, "
        "relative to t = 1 s) on >= 10 of the 12")


def _smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * u)


def _piecewise(t: float, keys) -> np.ndarray:
    """keys: [(t_i, pose_i)] with t_0 = 0; raised-cosine between consecutive keys, held after the last."""
    for (ta, pa), (tb, pb) in zip(keys[:-1], keys[1:]):
        if t <= tb:
            u = _smooth((t - ta) / max(1e-9, tb - ta))
            return np.asarray(pa, float) + (np.asarray(pb, float) - np.asarray(pa, float)) * u
    return np.asarray(keys[-1][1], float)


def _pack(left, right) -> np.ndarray:
    return np.r_[left, right, 0.0, 0.0].astype(np.float32)


PICK = [(0.0, DEFAULT), (2.0, POSE["tumbler_ins"]), (3.0, POSE["tumbler_ins"]), (4.2, POSE["tumbler_lift"]),
        (7.2, POSE["tumbler_over_tray"]), (9.2, POSE["tumbler_over_tray"]), (11.2, DEFAULT)]
HOOK = [(0.0, DEFAULT), (2.5, POSE["hook_ins"]), (4.5, POSE["hook_lift"])]


def p1(t):
    return _pack(_piecewise(t, PICK), DEFAULT)


def p2(t):
    return _pack(DEFAULT, -_piecewise(t, PICK))


def p3(t):
    q = _piecewise(t, HOOK)
    return _pack(q, -q)


def p4(t):
    if t < 11.2:
        return p1(t)
    q = _piecewise(t - 11.2, [(0.0, DEFAULT), (2.5, POSE["hook_ins"]), (4.5, POSE["hook_lift"])])
    return _pack(q, -q)


TRAJECTORIES = {"P1_tumbler_left": (p1, 12.0), "P2_tumbler_right": (p2, 12.0), "P3_tray_hook": (p3, 12.0),
                "P4_pick_then_hook": (p4, 16.0)}


def pitch_of(q) -> float:
    w, x, y, z = q
    return math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))


def episode(cfg, slots, ctrl, runner, traj, dur: float, seed: int) -> dict:
    runner.reset(np.random.default_rng(seed))
    ctrl.reset_phase()                      # the clock starts at 0 every episode (as WaiterEnv.reset)
    slot = slots[0]
    dt = float(cfg.policy_dt)
    held, p0, y0, pt0 = None, None, None, None
    disp = yaw = pitch = 0.0
    for k in range(int(round(dur / dt))):
        t = k * dt
        ctrl.set_upper_body(traj(t))
        if held is None and ctrl.clock_step % 20 == HOLD_STEP:
            held = HOLD_STEP
        if held is not None:
            ctrl.clock_step = held
        runner.step([ctrl.update(runner.observe(0, np.zeros(3)))])
        quat = runner.d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7]
        if runner.tilt(0) >= Q.TILT_FALL:
            return {"seed": seed, "fell_at_s": round(t, 2), "disp_m": None, "yaw_deg": None, "pitch_deg": None,
                    "ok": False}
        if t >= T0_S:
            xy = runner.d.qpos[slot.qpos_adr:slot.qpos_adr + 2]
            if p0 is None:
                p0, y0, pt0 = xy.copy(), Q.yaw_of(quat), pitch_of(quat)
            disp = max(disp, float(np.linalg.norm(xy - p0)))
            yaw = max(yaw, abs(math.degrees(math.remainder(Q.yaw_of(quat) - y0, 2 * math.pi))))
            pitch = max(pitch, abs(math.degrees(pitch_of(quat) - pt0)))
    ok = disp <= MAX_DISP_M and yaw <= MAX_YAW_DEG and pitch <= MAX_PITCH_DEG
    return {"seed": seed, "fell_at_s": None, "disp_m": round(disp, 4), "yaw_deg": round(yaw, 2),
            "pitch_deg": round(pitch, 2), "ok": bool(ok)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deploy", type=Path, required=True)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"Q4: REFUSED ({a.out} exists; never overwritten)")
    cfg, model, slots, ctrl, runner = Q.make(a.deploy, a.upstream, a.cache_dir)
    eps = []
    for name, (traj, dur) in TRAJECTORIES.items():
        for seed in SEEDS:
            r = episode(cfg, slots, ctrl, runner, traj, dur, seed)
            r["trajectory"] = name
            eps.append(r)
            print(json.dumps(r), flush=True)
    falls = sum(e["fell_at_s"] is not None for e in eps)
    n_ok = sum(e["ok"] for e in eps)
    verdict = "PASS" if falls == 0 and n_ok >= MIN_OK else "FAIL"
    out = {"deploy": str(a.deploy), "gate": "Q4 standing manipulation", "rule": RULE, "verdict": verdict,
           "falls": falls, "n": len(eps), "n_ok": n_ok, "episodes": eps, "payload_kg_per_hand": Q.PAYLOAD_KG,
           "hold_step": HOLD_STEP, "poses": POSE}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"WAITER Q4: {verdict} (falls {falls}/12, ok {n_ok}/12; need 0 and >= {MIN_OK}) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
