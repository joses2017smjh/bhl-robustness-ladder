"""Static PNG figures of the identification (light surface, reference palette, one y-axis per panel)."""
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sysid_fit import gym_yaw_fast, DT
from analyze_steps import load
from analyze_bangbang import step_rates

D = Path(__file__).resolve().parent / "data"; P = Path(__file__).resolve().parent / "plots"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURF, INK, INK2, MUTED, GRID, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#d9d8d2"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "legend.frameon": False, "lines.linewidth": 2})


def gym_rate(u, K, tau, lat):
    y, _, _ = gym_yaw_fast(u, np.zeros(len(u)), K, tau, lat)
    return np.diff(np.concatenate([[0.0], y])) / DT


# 1. yaw step responses, small multiples by vx (wz = +1.0), physics mean of 3 seeds vs the gym's range
fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
pre, post = 10, 40
for ax, vx in zip(axs, (0.0, 0.15, 0.35)):
    R = []
    for s in (0, 1, 2):
        r = load(D / f"phys_steps_vx{vx:.2f}_wz+1.0_s{s}.npz"); m = r["meta"]
        k0 = int(round(m["t_on"] / DT))
        R.append(step_rates(r)[k0 - pre:k0 + post])
    R = np.array(R); tt = (np.arange(-pre, post) + 0.5) * DT
    u = np.r_[np.zeros(pre), np.ones(post)]
    lo = np.minimum.reduce([gym_rate(u, K, tau, lat) for K in (0.9, 1.4) for tau in (0.15, 0.45) for lat in (0, 2)])
    hi = np.maximum.reduce([gym_rate(u, K, tau, lat) for K in (0.9, 1.4) for tau in (0.15, 0.45) for lat in (0, 2)])
    ax.fill_between(tt, lo, hi, color=BAND, lw=0, label="gym randomization range", step="mid")
    ax.step(tt, gym_rate(u, 1.17, 0.25, 1), where="mid", color=ORANGE, label="gym nominal (K 1.17, tau 0.25 s, 1 step)")
    ax.step(tt, u, where="mid", color=MUTED, lw=1, ls="--", label="command wz = 1.0")
    for s in range(3):
        ax.step(tt, R[s], where="mid", color=BLUE, lw=0.8, alpha=0.35)
    ax.step(tt, R.mean(0), where="mid", color=BLUE, label="physics (mean of 3 reset seeds)")
    ax.set_title(f"vx = {vx} m/s", color=INK, fontsize=10, loc="left")
    ax.set_xlabel("time from step (s)")
axs[0].set_ylabel("yaw rate over each 0.04 s step (rad/s)")
axs[0].legend(loc="lower right", fontsize=7.5)
fig.suptitle("Yaw-rate step response: the gait turns within 1-2 policy steps; the gym lags 0.1-0.5 s", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(P / "yaw_step_response.png", dpi=150); plt.close(fig)

# 2. frequency response: gain and phase (two panels, one axis each)
sq = json.load(open(D / "bangbang_summary.json"))["square"]
fs = sorted(set(x["f_hz"] for x in sq))
g_ph = [np.mean([x["gain"] for x in sq if x["f_hz"] == f]) for f in fs]
p_ph = [np.mean([x["phase_deg"] for x in sq if x["f_hz"] == f]) if f < 12.5 else np.nan for f in fs]
fr = np.linspace(0.2, 12.5, 200); w_ = 2 * np.pi * fr * DT
def disc_resp(K, tau, lat):
    a = DT / max(tau, DT); z = np.exp(1j * w_)
    return K * a / (1 - (1 - a) / z) * z ** (-lat)
h = json.load(open(D / "bangbang_summary.json"))["prbs"]["0.7"]["fir_h"]
Hf = np.array([sum(hj * np.exp(-1j * wk * j) for j, hj in enumerate(h)) for wk in w_])
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
env_g = np.array([np.abs(disc_resp(K, tau, lat)) for K in (0.9, 1.4) for tau in (0.15, 0.45) for lat in (0, 2)])
env_p = np.array([np.degrees(np.unwrap(np.angle(disc_resp(1, tau, lat)))) for tau in (0.15, 0.45) for lat in (0, 2)])
axs[0].fill_between(fr, env_g.min(0), env_g.max(0), color=BAND, lw=0, label="gym range")
axs[0].plot(fr, np.abs(disc_resp(1.17, 0.25, 1)), color=ORANGE, label="gym nominal")
axs[0].plot(fr, np.abs(Hf), color=AQUA, label="physics, PRBS FIR (p_flip 0.7)")
axs[0].plot(fs, g_ph, "o", color=BLUE, ms=6, label="physics, +-1 square waves")
axs[0].set_xscale("log"); axs[0].set_xlabel("frequency (Hz)"); axs[0].set_ylabel("|yaw rate| / |command| (fundamental)")
axs[0].set_title("gain", loc="left", color=INK); axs[0].legend(fontsize=7.5, loc="lower left")
axs[1].fill_between(fr, env_p.min(0), env_p.max(0), color=BAND, lw=0)
axs[1].plot(fr, np.degrees(np.unwrap(np.angle(disc_resp(1.17, 0.25, 1)))), color=ORANGE)
axs[1].plot(fr, np.degrees(np.unwrap(np.angle(Hf))), color=AQUA)
axs[1].plot(fs[:-1], p_ph[:-1], "o", color=BLUE, ms=6)
axs[1].set_xscale("log"); axs[1].set_xlabel("frequency (Hz)"); axs[1].set_ylabel("phase (deg)")
axs[1].set_title("phase", loc="left", color=INK)
fig.suptitle("Yaw channel frequency response: physics bandwidth ~4-6x the gym's (the actor chatters at 6-12 Hz)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(P / "yaw_frequency_response.png", dpi=150); plt.close(fig)

# 3. static maps: yaw rate vs command (by vx) and forward speed vs command
st = json.load(open(D / "steps_fits.json")); sp = json.load(open(D / "speed_fits.json"))
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
u = np.linspace(0, 1.5, 50)
axs[0].fill_between(u, 0.9 * u, 1.4 * u, color=BAND, lw=0, label="gym range (K 0.9-1.4)")
for vx, c in ((0.0, YELLOW), (0.15, AQUA), (0.35, BLUE)):
    rr = [x for x in st if x["vx"] == vx]
    us = sorted(set(abs(x["wz"]) for x in rr))
    ys = [np.mean([abs(x["w_ss"] - x["baseline_rate"]) for x in rr if abs(x["wz"]) == a and x["wz"] > 0]) for a in us]
    axs[0].plot(us, ys, "o-", color=c, ms=6, label=f"physics vx = {vx}")
axs[0].set_xlabel("constant wz command (rad/s)"); axs[0].set_ylabel("steady yaw rate (rad/s)"); axs[0].legend(fontsize=7.5)
axs[0].set_title("yaw: concave gain, no sustained in-place turn below ~0.4", loc="left", color=INK)
vc = np.linspace(0, 0.35, 50)
axs[1].fill_between(vc, 0.85 * vc, 1.05 * vc, color=BAND, lw=0, label="gym range (v_gain 0.85-1.05)")
vs = sorted(set(x["vx"] for x in sp))
axs[1].plot(vs, [np.mean([x["v_ss"] for x in sp if x["vx"] == v]) for v in vs], "o-", color=BLUE, ms=6, label="physics steady speed")
axs[1].plot(vc, np.maximum(0, 1.19 * vc - 0.063), color=MUTED, lw=1, ls="--", label="fit max(0, 1.19 v - 0.063)")
axs[1].set_xlabel("vx command (m/s)"); axs[1].set_ylabel("forward speed (m/s)"); axs[1].legend(fontsize=7.5)
axs[1].set_title("forward: dead zone below ~0.05 m/s, gain 0.77 at 0.15", loc="left", color=INK)
fig.tight_layout(); fig.savefig(P / "static_maps.png", dpi=150); plt.close(fig)

# 4. transfer traces: braked fraction and revisit fraction by outcome
tl = json.load(open(D / "trace_loops.json"))
fig, axs = plt.subplots(1, 2, figsize=(10, 3.0), sharey=True)
arms = ["armV4-s5", "armV4-s6", "astar"]
for ax, key, lab in ((axs[0], "braked_frac", "fraction of policy steps the brake scaled vx"),
                     (axs[1], "revisit_frac", "fraction of time within 0.3 m of a spot visited >= 10 s earlier")):
    for i, a in enumerate(arms):
        for ok, c, mk in ((True, BLUE, "o"), (False, ORANGE, "X")):
            xs = [r[key] for r in tl if r["arm"] == a and (r["outcome"] == "goal") == ok]
            ax.scatter(xs, np.full(len(xs), i) + (0.12 if ok else -0.12), color=c, marker=mk, s=40,
                       label=("reached goal" if ok else "time-out / stuck") if i == 0 else None, edgecolor=SURF, linewidth=0.8)
    ax.set_yticks(range(3)); ax.set_yticklabels(["v4 s5", "v4 s6", "A*"]); ax.set_xlabel(lab); ax.set_xlim(-0.02, 1.0)
axs[0].legend(fontsize=7.5, loc="upper right")
fig.suptitle("Physics transfer (maze seeds 50000-50011, existing traces): failures brake more and loop", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(P / "transfer_brake_loops.png", dpi=150); plt.close(fig)
print("ok", sorted(p.name for p in P.glob("*.png")))
