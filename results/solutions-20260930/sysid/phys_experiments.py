"""Physics (MuJoCo, CPU) system-identification runs of the frozen biped gait dr-default-s0 on the open floor.

Usage: phys_experiments.py <block> ; blocks: steps, speed, square, prbs, pwm, replay, brake
Every run is saved as data/phys_<block>_<name>_s<seed>.npz (per-policy-step arrays) + a JSON index.
Gait reset seeds only (runner.reset rng); no maze is involved.
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np
from sysid_lib import BipedSim

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)
DT = 0.04
SETTLE = 3.0


def save(block, name, seed, r, meta):
    f = OUT / f"phys_{block}_{name}_s{seed}.npz"
    np.savez(f, **{k: v for k, v in r.items() if isinstance(v, np.ndarray)}, meta=json.dumps(meta),
             fell=np.array(-1.0 if r["fell_at_s"] is None else r["fell_at_s"]))
    return f.name


def block_steps(sim):
    """yaw-rate steps: settle 3 s, (vx,0,0) 3 s, (vx,0,wz) 6 s, (vx,0,0) 3 s; step instant jittered 0-4 steps by seed."""
    idx = []
    conds = [(vx, wz) for vx in (0.0, 0.15, 0.35) for wz in (0.3, 0.6, 1.0, -1.0)] + [(0.0, 1.5), (0.35, 1.5)]
    for vx, wz in conds:
        for seed in (0, 1, 2):
            t_on = SETTLE + 3.0 + DT * (seed * 2 % 5)
            t_off = t_on + 6.0
            sched = lambda t, s, vx=vx, wz=wz, t_on=t_on, t_off=t_off: (
                0.0 if t < SETTLE - 1e-9 else vx, 0.0, wz if t_on - 1e-9 <= t < t_off - 1e-9 else 0.0)
            t0 = time.time()
            r = sim.run(sched, t_off + 3.0, seed=seed)
            name = f"vx{vx:.2f}_wz{wz:+.1f}"
            meta = {"vx": vx, "wz": wz, "t_walk": SETTLE, "t_on": t_on, "t_off": t_off, "seed": seed}
            idx.append({"file": save("steps", name, seed, r, meta), **meta, "fell": r["fell_at_s"], "wall": round(time.time() - t0, 1)})
            print(idx[-1], flush=True)
    return idx


def block_speed(sim):
    """forward-speed steps: settle 3 s, (vx,0,0) 10 s, zero 3 s."""
    idx = []
    for vx in (0.15, 0.25, 0.35):
        for seed in range(5):
            t_on = SETTLE + DT * (seed * 2 % 5)
            t_off = t_on + 10.0
            sched = lambda t, s, vx=vx, t_on=t_on, t_off=t_off: (vx if t_on - 1e-9 <= t < t_off - 1e-9 else 0.0, 0.0, 0.0)
            t0 = time.time()
            r = sim.run(sched, t_off + 3.0, seed=seed)
            meta = {"vx": vx, "t_on": t_on, "t_off": t_off, "seed": seed}
            idx.append({"file": save("speed", f"vx{vx:.2f}", seed, r, meta), **meta, "fell": r["fell_at_s"], "wall": round(time.time() - t0, 1)})
            print(idx[-1], flush=True)
    return idx


def square(t, f):
    return 1.0 if math.sin(2 * math.pi * f * t + 1e-9) >= 0 else -1.0


def block_square(sim):
    """wz = +-1 square wave (f Hz) with vx 0.3: settle 3 s, (0.3,0,0) 3 s, square 12 s, (0.3,0,0) 2 s."""
    idx = []
    for f in (0.5, 1.5, 3.0, 6.25, 12.5):
        for seed in (0, 1, 2):
            t_on = SETTLE + 3.0 + DT * (seed * 2 % 5)
            t_off = t_on + 12.0
            def sched(t, s, f=f, t_on=t_on, t_off=t_off):
                if t < SETTLE - 1e-9:
                    return (0.0, 0.0, 0.0)
                if t_on - 1e-9 <= t < t_off - 1e-9:
                    # sample the square wave at the step's own phase (k steps after t_on)
                    k = int(round((t - t_on) / DT))
                    return (0.3, 0.0, square(k * DT, f))
                return (0.3, 0.0, 0.0)
            t0 = time.time()
            r = sim.run(sched, t_off + 2.0, seed=seed)
            meta = {"f_hz": f, "vx": 0.3, "t_on": t_on, "t_off": t_off, "seed": seed}
            idx.append({"file": save("square", f"f{f}", seed, r, meta), **meta, "fell": r["fell_at_s"], "wall": round(time.time() - t0, 1)})
            print(idx[-1], flush=True)
    return idx


def markov_pm1(n, p_flip, rng, p_plus=None):
    """+-1 sequence: Markov chain with per-step flip probability p_flip (mean 0), or i.i.d. with P(+1)=p_plus."""
    if p_plus is not None:
        return np.where(rng.random(n) < p_plus, 1.0, -1.0)
    s = np.empty(n); s[0] = 1.0 if rng.random() < 0.5 else -1.0
    fl = rng.random(n) < p_flip
    for i in range(1, n):
        s[i] = -s[i - 1] if fl[i] else s[i - 1]
    return s


def block_prbs(sim):
    """PRBS wz = +-1 with vx 0.3: per-step flip probability 0.7 (the v4 actors' 0.52-0.73 in the gym), 0.5 (i.i.d.),
    0.1 and 0.02 (slow); 3 seeds; settle 3 s, walk 3 s, PRBS 20 s."""
    idx = []
    for p in (0.7, 0.5, 0.1, 0.02):
        for seed in (0, 1, 2):
            rng = np.random.default_rng(1000 + seed + int(p * 100))
            seq = markov_pm1(int(20.0 / DT), p, rng)
            t_on = SETTLE + 3.0
            t_off = t_on + 20.0
            def sched(t, s, seq=seq, t_on=t_on, t_off=t_off):
                if t < SETTLE - 1e-9:
                    return (0.0, 0.0, 0.0)
                if t_on - 1e-9 <= t < t_off - 1e-9:
                    return (0.3, 0.0, float(seq[int(round((t - t_on) / DT))]))
                return (0.3, 0.0, 0.0)
            t0 = time.time()
            r = sim.run(sched, t_off + 1.0, seed=seed)
            meta = {"p_flip": p, "vx": 0.3, "t_on": t_on, "t_off": t_off, "seed": seed}
            idx.append({"file": save("prbs", f"p{p}", seed, r, meta), **meta, "fell": r["fell_at_s"], "wall": round(time.time() - t0, 1)})
            print(idx[-1], flush=True)
    return idx


def block_pwm(sim):
    """Duty-cycle chatter vs its mean as a constant command, vx 0.3 (and 0.0): i.i.d. +-1 with P(+1)=(1+m)/2
    ('chat'), the deterministic fastest PWM pattern of the same mean ('pwm'), and the constant m ('const');
    settle 3 s, walk 3 s, 8 s of the pattern."""
    idx = []
    pwm_patterns = {0.0: [1, -1], 0.5: [1, 1, 1, -1], 0.333: [1, 1, -1], 0.75: [1, 1, 1, 1, 1, 1, 1, -1], -0.5: [-1, -1, -1, 1]}
    for vx in (0.3, 0.0):
        for m in (0.0, 0.333, 0.5, 0.75, -0.5):
            for kind in ("chat", "pwm", "const"):
                if kind == "const" and m == 0.0:
                    continue
                for seed in ((0, 1) if vx == 0.3 else (0,)):
                    rng = np.random.default_rng(2000 + seed + int(m * 1000))
                    n = int(8.0 / DT)
                    if kind == "chat":
                        seq = markov_pm1(n, None, rng, p_plus=(1 + m) / 2)
                    elif kind == "pwm":
                        pat = pwm_patterns[m]
                        seq = np.array([pat[i % len(pat)] for i in range(n)], float)
                    else:
                        seq = np.full(n, m)
                    t_on = SETTLE + 3.0
                    t_off = t_on + 8.0
                    def sched(t, s, seq=seq, t_on=t_on, t_off=t_off, vx=vx):
                        if t < SETTLE - 1e-9:
                            return (0.0, 0.0, 0.0)
                        if t_on - 1e-9 <= t < t_off - 1e-9:
                            return (vx, 0.0, float(seq[int(round((t - t_on) / DT))]))
                        return (vx, 0.0, 0.0)
                    t0 = time.time()
                    r = sim.run(sched, t_off + 1.0, seed=seed)
                    meta = {"m": m, "kind": kind, "vx": vx, "t_on": t_on, "t_off": t_off, "seed": seed, "seq_mean": float(seq.mean())}
                    idx.append({"file": save("pwm", f"vx{vx}_m{m}_{kind}", seed, r, meta), **meta, "fell": r["fell_at_s"],
                                "wall": round(time.time() - t0, 1)})
                    print(idx[-1], flush=True)
    return idx


def block_replay(sim, arms=("armV4-s5", "armV4-s6"), mazes=(9000, 9003)):
    """Open-loop replay of the gym actors' own (vx, wz) command sequences (data/gym_actor_<arm>_maze<k>.npz) on the
    open floor: settle 3 s, then the recorded sequence (first 60 s)."""
    from bhl_robust.navgym.env import V_MAX, W_MAX
    idx = []
    for arm in arms:
        for mz in mazes:
            A = np.load(OUT / f"gym_actor_{arm}_maze{mz}.npz")["A"][: int(60.0 / DT)]
            v = (A[:, 0] + 1.0) * 0.5 * V_MAX; w = A[:, 1] * W_MAX
            t_on = SETTLE
            t_off = t_on + len(A) * DT
            def sched(t, s, v=v, w=w, t_on=t_on, t_off=t_off):
                if t < t_on - 1e-9 or t >= t_off - 1e-9:
                    return (0.0, 0.0, 0.0)
                k = int(round((t - t_on) / DT))
                return (float(v[k]), 0.0, float(w[k]))
            t0 = time.time()
            r = sim.run(sched, t_off, seed=0)
            meta = {"arm": arm, "maze": mz, "t_on": t_on, "t_off": t_off, "seed": 0}
            idx.append({"file": save("replay", f"{arm}_maze{mz}", 0, r, meta), **meta, "fell": r["fell_at_s"], "wall": round(time.time() - t0, 1)})
            print(idx[-1], flush=True)
    return idx


def block_brake(sim):
    """Flat-floor brake artefact: the speed-step and PRBS protocols with the team_sensors brake ON (no walls)."""
    idx = []
    for vx in (0.15, 0.35):
        for seed in (0, 1):
            sched = lambda t, s, vx=vx: (0.0 if t < SETTLE else vx, 0.0, 0.0)
            r = sim.run(sched, SETTLE + 10.0, seed=seed, brake=True)
            meta = {"vx": vx, "kind": "straight", "seed": seed, "sensor_stats": r["sensor_stats"]}
            idx.append({"file": save("brake", f"straight_vx{vx}", seed, r, meta), **meta, "fell": r["fell_at_s"]})
            print(idx[-1], flush=True)
    for seed in (0, 1):
        rng = np.random.default_rng(3000 + seed)
        seq = markov_pm1(int(12.0 / DT), 0.7, rng)
        def sched(t, s, seq=seq):
            if t < SETTLE:
                return (0.0, 0.0, 0.0)
            k = int(round((t - SETTLE) / DT))
            return (0.35, 0.0, float(seq[k]) if k < len(seq) else 0.0)
        r = sim.run(sched, SETTLE + 12.0, seed=seed, brake=True)
        meta = {"vx": 0.35, "kind": "prbs0.7", "seed": seed, "sensor_stats": r["sensor_stats"]}
        idx.append({"file": save("brake", "prbs0.7_vx0.35", seed, r, meta), **meta, "fell": r["fell_at_s"]})
        print(idx[-1], flush=True)
    return idx


if __name__ == "__main__":
    block = sys.argv[1]
    sim = BipedSim()
    t0 = time.time()
    idx = globals()[f"block_{block}"](sim)
    (OUT / f"index_{block}.json").write_text(json.dumps(idx, indent=1))
    print(f"BLOCK {block} done in {time.time() - t0:.0f} s", flush=True)
