"""Calibrate command dead-reckoning error against evaluator truth on v1 bounded traces (prior data only)."""
import json, sys, glob, math, os
import numpy as np
V = sys.argv[1]
rows = []
def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
stats = {}
allrec = []
for ep in sorted(glob.glob(os.path.join(V, 'collector-inputs/*/bounded_latest-*'))):
    e = json.load(open(os.path.join(ep, 'episode.json')))
    truth = np.load(os.path.join(ep, 'evaluator_truth.npz'))
    T = truth['T_W_I']; ts = truth['timestamp_s']
    Tb = np.linalg.inv(np.asarray(e['T_body_imu']))
    body = T @ Tb  # body pose
    trace = e['trace']
    assert len(trace) == len(ts)
    cmd = np.array([t['command'] for t in trace])  # applied during step k (from decision k to k+1)
    dt = e['latency_quantization_s']
    xy = body[:, :2, 3]; yaw = np.array([yaw_of(b) for b in body])
    n = len(trace)
    dec = np.array([t['decision_timestamp_s'] for t in trace])
    # truth state at decision k is body[k-1] (after step k-1); use state index k-1 as start of step k
    for W in (1, 2, 3, 5, 8, 13, 25, 50, 75, 100, 125):  # windows in policy steps (dt=0.04 s)
        for s in range(260, n - W, 3):  # start after nav begins (decision >= 10.4 s)
            a, b = s - 1, s - 1 + W
            x, y, th = xy[a, 0], xy[a, 1], yaw[a]
            L = 0.0; rot = 0.0
            for k in range(s, s + W):
                vx, vy, wz = cmd[k]
                x += (vx * math.cos(th) - vy * math.sin(th)) * dt
                y += (vx * math.sin(th) + vy * math.cos(th)) * dt
                th += wz * dt
                L += math.hypot(vx, vy) * dt; rot += abs(wz) * dt
            ep_err = math.hypot(xy[b, 0] - x, xy[b, 1] - y)
            yaw_err = abs(wrap(yaw[b] - th))
            allrec.append((e['method'], e['condition'], W * dt, L, rot, ep_err, yaw_err, float(np.abs(cmd[s:s+W]).sum() == 0)))
A = np.array([(r[2], r[3], r[4], r[5], r[6], r[7]) for r in allrec])
print('records', len(A))
for zero in (1.0, 0.0):
    sel = A[A[:, 5] == zero]
    print('ZERO-COMMAND windows' if zero else 'WINDOWS WITH NONZERO COMMANDS')
    for W in sorted(set(np.round(sel[:, 0], 3))):
        s2 = sel[np.isclose(sel[:, 0], W)]
        if len(s2) < 20: continue
        print('  T=%5.2fs n=%5d  pos err p50=%.3f p95=%.3f p99=%.3f max=%.3f | yaw err deg p50=%.1f p95=%.1f p99=%.1f max=%.1f | L mean=%.2f' % (
            W, len(s2), *np.quantile(s2[:, 3], [.5, .95, .99]), s2[:, 3].max(), *np.degrees(np.quantile(s2[:, 4], [.5, .95, .99])), np.degrees(s2[:, 4].max()), s2[:, 1].mean()))
np.save(sys.argv[2], A)
