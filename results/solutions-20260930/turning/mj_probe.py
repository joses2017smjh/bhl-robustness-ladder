"""Light CPU MuJoCo probe of turning checkpoints (read-only on the repo; learned policy replayed, nothing scripted).

Same replay path as scripts/bench/turn_test.py / turn_diagnose.py (build_multi + RlController + ContactRunner),
but the actor is read straight from an RSL-RL checkpoint (model_XXXX.pt) so intermediate checkpoints can be
probed without an ONNX export, and the action can optionally carry the checkpoint's own Gaussian exploration
noise (std * mult), fed back as last_action exactly like a training rollout.

Protocol per run: reset seed, `warm` s of zero command, then (0, 0, wz) for `seconds`.
Reports: yaw turned in the commanded direction, fall, foot lift-offs in the last 2 s of the settle and during
the turn, raw |action| statistics and the fraction of leg-joint torque samples at the effort limit.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder")
UPSTREAM = REPO / "external/Berkeley-Humanoid-Lite"
LOG = UPSTREAM / "logs/rsl_rl/humanoid"
sys.path.insert(0, str(REPO / "scripts/bench"))
sys.path.insert(0, str(REPO / "src"))
DEPLOY = LOG / "2026-09-25_18-54-44_arms-turn-turnboth-s0/exported/deploy.yaml"   # config shared by all 22-DoF runs
SAMPLE_EVERY = 10
MIN_AIR_S = 0.020


def yaw_of(q):
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def count_liftoffs(contact, dt, min_air_s=MIN_AIR_S):
    n, i, seen = 0, 0, False
    c = np.asarray(contact, dtype=bool)
    while i < len(c):
        if c[i]:
            seen = True
            i += 1
            continue
        j = i
        while j < len(c) and not c[j]:
            j += 1
        if seen and (j - i) * dt >= min_air_s - 1e-12:
            n += 1
        i = j
    return n


class NpActor:
    """RSL-RL ActorCritic actor (ELU MLP) in numpy, optional Gaussian noise with the checkpoint's std."""

    def __init__(self, ckpt: str, noise_mult: float = 0.0, seed: int = 0, joints: str = "all"):
        import torch
        torch.set_num_threads(1)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        self.W = [sd[f"actor.{i}.weight"].numpy().astype(np.float32) for i in (0, 2, 4, 6)]
        self.b = [sd[f"actor.{i}.bias"].numpy().astype(np.float32) for i in (0, 2, 4, 6)]
        self.std = sd["std"].numpy().astype(np.float64)
        self.noise_mult = noise_mult
        mask = np.ones(22, dtype=np.float32)
        if joints == "legs":
            mask[:10] = 0.0
        elif joints == "arms":
            mask[10:] = 0.0
        self.mask = mask
        self.rng = np.random.default_rng(10_000 + seed)

    def mean(self, obs):
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        for k in range(3):
            x = self.W[k] @ x + self.b[k]
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
        return self.W[3] @ x + self.b[3]

    def forward(self, obs):
        a = self.mean(obs)
        if self.noise_mult > 0:
            a = a + self.mask * self.noise_mult * self.std * self.rng.standard_normal(a.shape)
        return a.astype(np.float32)[None, :]


_MODEL = None


def run(ckpt, wz, warm=3.0, seconds=6.0, seed=0, noise_mult=0.0, vx=0.0, joints="all"):
    import mujoco
    from omegaconf import OmegaConf
    from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
    from bhl_robust.eval.multi_robot import build_multi
    from team_airlock import ContactRunner

    cache = Path(os.environ.get("MJ_CACHE", "/scratch/sanchej7/tmp/claude-19646/mjcache"))
    cfg = OmegaConf.load(DEPLOY)
    actor = NpActor(ckpt, noise_mult, seed, joints)
    global _MODEL
    if _MODEL is None:
        _MODEL = build_multi(UPSTREAM, cache / "humanoid", 1, ["t"], variant="humanoid", world="flat")
    model, slots = _MODEL
    ctrl = RlController(cfg)
    ctrl.policy = actor
    runner = ContactRunner(model, slots, [cfg], [ctrl])
    runner.reset(np.random.default_rng(seed))
    slot = slots[0]
    owners = np.array([0 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "").startswith(slot.prefix)
                       else -1 for g in range(model.ngeom)])
    runner.configure_contacts(owners)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    feet = {f: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{slot.prefix}leg_{f}_ankle_roll") for f in ("left", "right")}
    fgeoms = {f: {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == feet[f] and model.geom_contype[g]} for f in feet}
    tr = {"t": [], "left": [], "right": [], "sat": []}
    clk = {"t": 0.0, "k": 0}
    sub_dt = float(cfg.physics_dt)
    leg_ctrl = np.array(slot.ctrl)[10:]

    def hook(d):
        clk["k"] += 1
        clk["t"] += sub_dt
        if clk["k"] % SAMPLE_EVERY:
            return
        on = {"left": False, "right": False}
        for c in range(d.ncon):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            o = g2 if g1 == floor else g1 if g2 == floor else None
            if o is None:
                continue
            for f in on:
                if o in fgeoms[f]:
                    on[f] = True
        tr["t"].append(clk["t"])
        tr["left"].append(on["left"])
        tr["right"].append(on["right"])
        tr["sat"].append(float(np.mean(np.abs(d.ctrl[leg_ctrl]) >= 0.999 * runner.eff[10:])) if np.ndim(runner.eff) else np.nan)

    runner.substep_hook = hook
    dt = float(cfg.policy_dt)
    yaws, acts, fell = [], [], None
    yaw0 = None
    for step in range(int(round((warm + seconds) / dt))):
        now = step * dt
        cmd = np.zeros(3) if now < warm else np.array([vx, 0.0, wz])
        obs = runner.observe(0, cmd)
        target = ctrl.update(obs)
        acts.append((now, ctrl.policy_actions[0].copy()))
        runner.step([target])
        y = yaw_of(runner.d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7])
        yaws.append((now, y))
        if runner.tilt(0) >= 0.78:
            fell = round(now, 2)
            break
    t = np.array(tr["t"])
    samp = sub_dt * SAMPLE_EVERY
    tm = t >= warm
    sm = (t >= warm - 2.0) & (t < warm)
    yv = np.unwrap(np.array([y for _, y in yaws]))
    ty = np.array([tt for tt, _ in yaws])
    i0 = int(np.argmax(ty >= warm)) if (ty >= warm).any() else len(yv) - 1
    turned = math.degrees(yv[-1] - yv[i0]) * (math.copysign(1.0, wz) if wz != 0 else 1.0)
    xy_disp = None
    A = np.array([a for _, a in acts])
    At = np.array([a for tt, a in acts if tt >= warm]) if any(tt >= warm for tt, _ in acts) else A
    sat = np.array(tr["sat"])
    return {
        "seed": seed, "wz": wz, "vx": vx, "noise_mult": noise_mult, "noise_joints": joints, "fell_at_s": fell,
        "yaw_deg_cmd_dir": round(turned, 1),
        "lo_settle": [count_liftoffs(np.array(tr[f])[sm], samp) for f in ("left", "right")],
        "lo_turn": [count_liftoffs(np.array(tr[f])[tm], samp) for f in ("left", "right")],
        "abs_a_arm": round(float(np.abs(At[:, :10]).mean()), 2), "abs_a_leg": round(float(np.abs(At[:, 10:]).mean()), 2),
        "abs_a_leg_perjoint": [round(float(v), 2) for v in np.abs(At[:, 10:16]).mean(0)],
        "leg_sat_frac_turn": round(float(np.nanmean(sat[tm])), 3) if tm.any() else None,
        "leg_sat_frac_settle": round(float(np.nanmean(sat[sm])), 3) if sm.any() else None,
    }


def ckpt_path(run: str, it: int | None):
    d = sorted(glob.glob(str(LOG / f"*_{run}")))[-1]
    if it is None:
        return sorted(glob.glob(d + "/model_*.pt"), key=lambda q: int(q.split("_")[-1][:-3]))[-1]
    return f"{d}/model_{it}.pt"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--iters", type=int, nargs="+", default=[-1])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--wz", type=float, nargs="+", default=[0.6, -0.6])
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0])
    ap.add_argument("--warm", type=float, default=3.0)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--noise-joints", type=str, default="all", choices=("all", "legs", "arms"))
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()
    res = []
    for r in a.runs:
        for it in a.iters:
            ck = ckpt_path(r, None if it < 0 else it)
            for nm in a.noise:
                for s in a.seeds:
                    for w in a.wz:
                        o = run(ck, w, a.warm, a.seconds, s, nm, a.vx, a.noise_joints)
                        o.update(run_name=r, ckpt=os.path.basename(ck))
                        res.append(o)
                        print(f"{r:28s} {os.path.basename(ck):15s} noise{nm:4.2f}{a.noise_joints[0]} s{s} wz{w:+.1f}: yaw {o['yaw_deg_cmd_dir']:+7.1f} "
                              f"fell {o['fell_at_s']} lo_settle {o['lo_settle']} lo_turn {o['lo_turn']} |a|arm {o['abs_a_arm']} "
                              f"|a|leg {o['abs_a_leg']} sat {o['leg_sat_frac_settle']}/{o['leg_sat_frac_turn']}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
