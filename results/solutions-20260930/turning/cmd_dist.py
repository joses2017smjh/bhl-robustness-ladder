"""Monte-Carlo of the command stream the stock/TurnBoth env trains on (heading mode, 10 s resample, 2 % standing).
Assumes the robot tracks the heading-derived yaw rate perfectly (upper bound on how fast wz decays)
or not at all (wz stays at its initial value): the truth is in between."""
import numpy as np
rng = np.random.default_rng(0)
N, T, dt = 200000, 10.0, 0.04
steps = int(T / dt)
vx = rng.uniform(-1, 1, N); vy = rng.uniform(-0.5, 0.5, N)
e0 = rng.uniform(-np.pi, np.pi, N); stand = rng.uniform(0, 1, N) <= 0.02
t = np.arange(steps) * dt
for label, decay in (("tracks heading (err*exp(-0.5t))", True), ("no yaw tracking (err const)", False)):
    err = e0[:, None] * (np.exp(-0.5 * t)[None, :] if decay else np.ones_like(t)[None, :])
    wz = np.clip(0.5 * err, -1.5, 1.5)
    v = np.hypot(vx, vy)[:, None] * np.ones_like(t)[None, :]
    wz[stand] = 0.0; v[stand] = 0.0
    pure = (v < 0.1) & (np.abs(wz) >= 0.3)
    pure2 = (v < 0.1) & (np.abs(wz) >= 0.1)
    gate_lin = v > 0.1
    gate_full = np.sqrt(v ** 2 + wz ** 2) > 0.1
    zero = (v < 0.1) & (np.abs(wz) < 0.1)
    print(f"{label}: pure turn (|v|<0.1,|wz|>=0.3) {pure.mean()*100:.2f}% of steps; (|wz|>=0.1) {pure2.mean()*100:.2f}%; "
          f"near-zero cmd {zero.mean()*100:.2f}%; stock air-time gate open {gate_lin.mean()*100:.1f}%; TurnBoth gate open {gate_full.mean()*100:.1f}%; "
          f"median |wz| {np.median(np.abs(wz)):.2f}")
