"""Model fitting helpers: continuous FOPDT on integrated responses, and the gym's own discrete dynamics."""
import math
import numpy as np

DT = 0.04
V_MAX, W_MAX = 0.35, 1.0


def fopdt_int(t, tau, L):
    """Integral of a unit-gain FOPDT step response (step at t=0, input held): (t-L) - tau(1-exp(-(t-L)/tau)), 0 before L."""
    s = np.maximum(np.asarray(t, float) - L, 0.0)
    return s - tau * (1.0 - np.exp(-s / tau))


def fit_fopdt_integrated(t, y, u, taus=None, Ls=None):
    """Least-squares (K, tau, L) for y(t) = K*u*fopdt_int(t, tau, L) (t measured from the step instant).
    Returns dict with K, tau, L, rmse and the 63 % rise time L + tau."""
    taus = np.concatenate([np.arange(0.01, 0.2, 0.005), np.arange(0.2, 1.0, 0.01), np.arange(1.0, 3.01, 0.05)]) if taus is None else taus
    Ls = np.arange(0.0, 0.6001, 0.005) if Ls is None else Ls
    best = None
    for tau in taus:
        for L in Ls:
            g = u * fopdt_int(t, tau, L)
            gg = float(g @ g)
            if gg <= 0:
                continue
            K = float(g @ y) / gg
            r = y - K * g
            sse = float(r @ r)
            if best is None or sse < best[0]:
                best = (sse, K, tau, L)
    sse, K, tau, L = best
    return {"K": K, "tau": float(tau), "L": float(L), "rise63": float(tau + L), "rmse": math.sqrt(sse / len(y))}


def gym_response(v_cmd, w_cmd, w_gain=1.17, v_gain=1.0, tau=0.25, drift=0.0, latency=1, v0=0.0, w0=0.0, dt=DT):
    """The gym's MazeNavEnv.step dynamics, noise-free: latency queue, Euler first-order lag (alpha = dt / max(tau, dt)) on
    both channels, drift scaled by v / V_MAX, yaw += w*dt, position += v*dt*(cos, sin)(new yaw).
    Inputs are PHYSICAL commands (m/s, rad/s) per step. Returns (yaw, x, y, v, w) arrays, entry k = state after step k."""
    n = len(w_cmd)
    q = [np.zeros(2) for _ in range(int(latency))]
    v, w, yaw, x, y = v0, w0, 0.0, 0.0, 0.0
    alpha = dt / max(tau, dt)
    out = np.zeros((n, 5))
    for k in range(n):
        q.append(np.array([v_cmd[k], w_cmd[k]]))
        vc, wc = q.pop(0)
        v += alpha * (v_gain * vc - v)
        w += alpha * (w_gain * wc - w)
        wt = w + drift * (v / V_MAX)
        yaw += wt * dt
        x += v * dt * math.cos(yaw); y += v * dt * math.sin(yaw)
        out[k] = (yaw, x, y, v, wt)
    return out


def gym_yaw_fast(w_cmd, v_cmd, w_gain, tau, latency, drift=0.0, v_gain=1.0, dt=DT, w0=0.0, v0=0.0):
    """Vectorisable yaw-only version of gym_response (same equations) for grid searches."""
    n = len(w_cmd)
    L = int(latency)
    wc = np.concatenate([np.zeros(L), np.asarray(w_cmd, float)])[:n]
    vc = np.concatenate([np.zeros(L), np.asarray(v_cmd, float)])[:n]
    a = dt / max(tau, dt)
    # first-order recursions via scipy.signal.lfilter: w_k = (1-a) w_{k-1} + a K u_k
    from scipy.signal import lfilter
    w, _ = lfilter([a * w_gain], [1.0, -(1.0 - a)], wc, zi=[(1.0 - a) * w0])
    v, _ = lfilter([a * v_gain], [1.0, -(1.0 - a)], vc, zi=[(1.0 - a) * v0])
    wt = w + drift * v / V_MAX
    return np.cumsum(wt) * dt, v, wt


def sinusoid_fit(t, y, f):
    """Least-squares a*sin(2 pi f t) + b*cos(2 pi f t) + c -> amplitude, phase (rad), offset."""
    X = np.column_stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = coef
    return float(math.hypot(a, b)), float(math.atan2(b, a)), float(c)


def central_rate(t, yaw):
    """Central-difference yaw rate at the sample times (edges one-sided)."""
    return np.gradient(yaw, t)
