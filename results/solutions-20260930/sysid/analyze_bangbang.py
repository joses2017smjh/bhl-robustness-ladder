"""Bang-bang yaw commands in physics: square waves (fundamental gain/phase), PRBS (FIR impulse response and best-fit gym
parameters), duty-cycle chatter vs constant commands (does the mean survive?), forward-speed drop and tilt.
Per-step yaw increments are aligned with the per-step commands (command k acts over [kDT, (k+1)DT]; the
increment yaw_k - yaw_{k-1} is the mean yaw rate over the same interval), so no half-step phase bias."""
import json, math
from pathlib import Path
import numpy as np
from sysid_fit import gym_yaw_fast, sinusoid_fit, DT
from sysid_lib import fwd_speed
from analyze_steps import load

D = Path(__file__).resolve().parent / "data"


def step_rates(r):
    yaw = r["yaw"]
    inc = np.diff(np.concatenate([[yaw[0]], yaw]))
    return inc / DT                       # rate_k over step k (first entry 0)


def window(r, t0, t1):
    t = r["t"] - DT                       # start time of step k
    return (t >= t0 - 1e-9) & (t < t1 - 1e-9)


def gym_rate(cmd, w_gain, tau, lat):
    yaw, _, w = gym_yaw_fast(cmd, np.zeros(len(cmd)), w_gain, tau, lat)
    return np.diff(np.concatenate([[0.0], yaw])) / DT


def fundamental(tk, x, f):
    amp, ph, c = sinusoid_fit(tk, x, f)
    return amp, ph


def analyse_square(r):
    m = r["meta"]; f = m["f_hz"]
    w = window(r, m["t_on"] + 2.0, m["t_off"])
    tk = (np.arange(len(r["t"]))[w] + 0.5) * DT
    cmd = r["cmd"][w, 2]; rate = step_rates(r)[w]
    out = {"f_hz": f, "seed": m["seed"]}
    if f >= 12.5 - 1e-9:
        # Nyquist: the only resolvable component alternates with the command; project onto the command's sign pattern
        s = np.sign(cmd)
        out.update(gain=float(np.dot(rate - rate.mean(), s) / np.dot(s, s)), phase_deg=None, lag_s=None)
    else:
        a_c, p_c = fundamental(tk, cmd, f); a_y, p_y = fundamental(tk, rate, f)
        dph = (p_y - p_c + np.pi) % (2 * np.pi) - np.pi
        out.update(gain=a_y / a_c, phase_deg=math.degrees(dph), lag_s=-dph / (2 * np.pi * f), cmd_fund=a_c, rate_fund=a_y)
    pre = window(r, m["t_on"] - 2.0, m["t_on"])
    v = fwd_speed(r)
    out.update(v_pre=float(v[pre].mean()), v_sq=float(v[w].mean()), v_ratio=float(v[w].mean() / v[pre].mean()),
               net_rate=float(rate.mean()), rate_rms=float(rate.std()), max_tilt=float(r["tilt"][w].max()),
               max_tilt_pre=float(r["tilt"][pre].max()), fell=float(r["fell"]))
    # the gym model on the same command (nominal and the lag range ends)
    full = r["cmd"][:, 2]
    for name, (K, tau, lat) in {"gym_nom": (1.17, 0.25, 1), "gym_fast": (1.17, 0.15, 0), "gym_slow": (1.17, 0.45, 2)}.items():
        g = gym_rate(full, K, tau, lat)[w]
        if f >= 12.5 - 1e-9:
            s = np.sign(cmd); out[name + "_gain"] = float(np.dot(g - g.mean(), s) / np.dot(s, s)); out[name + "_phase_deg"] = None
        else:
            a_c, p_c = fundamental(tk, cmd, f); a_g, p_g = fundamental(tk, g, f)
            out[name + "_gain"] = a_g / a_c
            out[name + "_phase_deg"] = math.degrees((p_g - p_c + np.pi) % (2 * np.pi) - np.pi)
    return out


def fir(rows_u, rows_y, n_lags=15):
    """Least-squares FIR: y_k = sum_{j=0}^{n-1} h_j u_{k-j} + c (pooled over runs)."""
    X, Y = [], []
    for u, y in zip(rows_u, rows_y):
        for k in range(n_lags, len(u)):
            X.append(np.concatenate([u[k - n_lags + 1:k + 1][::-1], [1.0]])); Y.append(y[k])
    X, Y = np.asarray(X), np.asarray(Y)
    h, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ h
    return h[:-1], float(res.std()), float(1 - res.var() / Y.var())


def best_gym_fit(rows_u, rows_y, Ks=np.arange(0.3, 2.01, 0.01), taus=np.arange(0.04, 1.201, 0.01), lats=range(0, 7)):
    """Grid search the gym's own (w_gain, tau, latency) that best predicts the per-step yaw rate on these inputs."""
    best = None
    Y = np.concatenate(rows_y)
    for lat in lats:
        for tau in taus:
            G = np.concatenate([gym_rate(u, 1.0, tau, lat) for u in rows_u])
            K = float(G @ Y / (G @ G))
            sse = float(np.sum((Y - K * G) ** 2))
            if best is None or sse < best[0]:
                best = (sse, K, float(tau), lat)
    sse, K, tau, lat = best
    return {"w_gain": K, "tau": tau, "latency": lat, "rmse": math.sqrt(sse / len(Y)), "r2": 1 - sse / float(np.sum((Y - Y.mean()) ** 2))}


def analyse_prbs(files):
    """Group by p_flip; the input is the command over the PRBS window plus 15 steps of pre-history."""
    groups = {}
    for f in files:
        r = load(f); m = r["meta"]
        w = window(r, m["t_on"] - 1.0, m["t_off"])
        groups.setdefault(m["p_flip"], []).append((r["cmd"][w, 2], step_rates(r)[w], r, m))
    out = {}
    for p, rr in sorted(groups.items()):
        U = [x[0] for x in rr]; Y = [x[1] for x in rr]
        h, res, r2 = fir(U, Y)
        g = best_gym_fit(U, Y)
        v = [fwd_speed(x[2])[window(x[2], x[3]["t_on"] + 1.0, x[3]["t_off"])].mean() / fwd_speed(x[2])[window(x[2], x[3]["t_on"] - 2.0, x[3]["t_on"])].mean() for x in rr]
        tilt = [x[2]["tilt"][window(x[2], x[3]["t_on"], x[3]["t_off"])].max() for x in rr]
        # gym nominal FIR for reference
        hg, _, _ = fir(U, [gym_rate(u, 1.17, 0.25, 1) for u in U])
        out[str(p)] = {"fir_h": [round(float(x), 4) for x in h], "fir_step": [round(float(x), 4) for x in np.cumsum(h)],
                       "fir_resid_rms": res, "fir_r2": r2, "gym_best": g, "gym_nominal_fir_step": [round(float(x), 4) for x in np.cumsum(hg)],
                       "v_ratio": [round(float(x), 3) for x in v], "max_tilt": [round(float(x), 3) for x in tilt],
                       "falls": sum(float(x[2]["fell"]) >= 0 for x in rr)}
        print(f"PRBS p_flip={p}: FIR step response (per-lag cumulative) {np.round(np.cumsum(h)[:10], 2).tolist()} r2={r2:.2f}; "
              f"gym-best {g}; v_ratio {np.round(v, 3).tolist()}; max tilt {np.round(tilt, 3).tolist()}")
    return out


def analyse_pwm(files):
    rows = []
    for f in files:
        r = load(f); m = r["meta"]
        w = window(r, m["t_on"] + 1.0, m["t_off"])
        rate = step_rates(r)[w]
        pre = window(r, m["t_on"] - 2.0, m["t_on"])
        v = fwd_speed(r)
        rows.append({"vx": m["vx"], "m": m["m"], "kind": m["kind"], "seed": m["seed"], "seq_mean": float(r["cmd"][w, 2].mean()),
                     "w_mean": float(rate.mean()), "gain": float(rate.mean() / r["cmd"][w, 2].mean()) if abs(r["cmd"][w, 2].mean()) > 1e-6 else None,
                     "v_mean": float(v[w].mean()), "v_pre": float(v[pre].mean()), "max_tilt": float(r["tilt"][w].max()), "fell": float(r["fell"]),
                     "rate_rms": float(rate.std())})
    return rows


if __name__ == "__main__":
    res = {}
    sq = [analyse_square(load(f)) for f in sorted(D.glob("phys_square_*.npz"))]
    res["square"] = sq
    for f in sorted(set(x["f_hz"] for x in sq)):
        rr = [x for x in sq if x["f_hz"] == f]
        def mm(k):
            v = [x[k] for x in rr if x.get(k) is not None]
            return (round(float(np.mean(v)), 3), round(float(np.min(v)), 3), round(float(np.max(v)), 3)) if v else None
        print(f"square {f} Hz: gain {mm('gain')} phase {mm('phase_deg')} lag_s {mm('lag_s')} | gym nom gain {mm('gym_nom_gain')} "
              f"ph {mm('gym_nom_phase_deg')} fast {mm('gym_fast_gain')} slow {mm('gym_slow_gain')} | v_ratio {mm('v_ratio')} "
              f"net_rate {mm('net_rate')} max_tilt {mm('max_tilt')} (pre {mm('max_tilt_pre')}) falls {sum(x['fell'] >= 0 for x in rr)}")
    files = sorted(D.glob("phys_prbs_*.npz"))
    if files:
        res["prbs"] = analyse_prbs(files)
    files = sorted(D.glob("phys_pwm_*.npz"))
    if files:
        rows = analyse_pwm(files)
        res["pwm"] = rows
        for vx in (0.3, 0.0):
            print(f"-- PWM vx={vx}")
            for m in (0.0, 0.333, 0.5, 0.75, -0.5):
                for kind in ("const", "pwm", "chat"):
                    rr = [x for x in rows if x["vx"] == vx and x["m"] == m and x["kind"] == kind]
                    if rr:
                        print(f"   m={m:+.3f} {kind:5s}: seq_mean {np.mean([x['seq_mean'] for x in rr]):+.3f} w_mean {np.mean([x['w_mean'] for x in rr]):+.3f} "
                              f"v_mean {np.mean([x['v_mean'] for x in rr]):.3f} (pre {np.mean([x['v_pre'] for x in rr]):.3f}) "
                              f"rate_rms {np.mean([x['rate_rms'] for x in rr]):.2f} max_tilt {np.max([x['max_tilt'] for x in rr]):.3f} falls {sum(x['fell'] >= 0 for x in rr)}")
    (D / "bangbang_summary.json").write_text(json.dumps(res, indent=1, default=float))
