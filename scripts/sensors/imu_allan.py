#!/usr/bin/env python3
"""Measure IMU noise parameters from ROS 2 recordings (or CSV) for the simulator.

Written for the probable Hiwonder IM10A (JY901B-class) 10-axis module, but
nothing below depends on that identification. Recording procedure and how the
output plugs into ``bhl_robust.eval.imu_sim`` / ``bhl_robust.fusion.attitude``:
docs/IMU_RECORDING.md. One subcommand per capture there:

  stationary  >= 2 h still: overlapping Allan deviation per axis, noise terms,
              timing/dropouts/duplicates, biases, gravity, magnetometer,
              barometer, quaternion sanity
              -> imu_measured.json   (drop-in keys of imu_sim.IM10A_DATASHEET,
                                      ImuNoise fields, Kalibr noise values)
                 imu_allan_report.json (everything, incl. the Allan curves)
                 imu_allan.png         (needs matplotlib; skipped otherwise)
  rotations   six-face 90-degree-hold sequence: axis/sign check, gyro-vs-accel
              frame consistency, quaternion convention, six-position
              accelerometer bias/scale            -> rotations_report.json
  tap         taps seen by this IMU and a reference sensor: time offset
                                                   -> tap_report.json
  export-csv  bag -> CSV in the format below

Usage (no ROS and no repo package needed; numpy is the only hard dependency):
    python scripts/sensors/imu_allan.py stationary <bag dir|csv> --nominal-rate 200 \\
        --trim-start-s 900 --out-dir <dir>
    python scripts/sensors/imu_allan.py --doc        # these conventions

INPUT is a CSV file (always supported) or a rosbag2 directory (sqlite3 .db3 or
.mcap storage; a ROS 1 .bag also works). Bags are read with the pure-Python
``rosbags`` package, which is OPTIONAL: ``pip install rosbags`` (no ROS install
needed). Without it, a bag input stops with that instruction; CSV keeps working.
pandas (faster CSV reading) and matplotlib (the plot) are optional too. CSV
format, a header row then one sample per row::

    t,gx,gy,gz,ax,ay,az[,mx,my,mz][,p][,qx,qy,qz,qw][,t_recv][,temp]

Write t with enough digits (epoch seconds ~1.7e9 need >= 16 significant digits;
"%.9f" is safe). CSV units default to the ROS ones (s, rad/s, m/s^2, tesla, pascal, quaternion
body->world x,y,z,w); override with --gyro-unit/--accel-unit/--mag-unit/
--baro-unit. Channels that update more slowly may leave their cells empty (NaN).

CONVENTIONS AND UNIT CONVERSIONS (every number in the outputs follows these)
---------------------------------------------------------------------------
Allan deviation: IEEE Std 952-1997 (Annex C), overlapping estimator on the
rate samples y_k, taken as uniformly spaced at the device rate fs (tau0=1/fs):
    theta_j = (1/fs) * sum_{k<j} y_k                         (j = 0..n)
    AVAR(tau = m/fs) = sum_j (theta_{j+2m} - 2 theta_{j+m} + theta_j)^2
                       / (2 tau^2 (n + 1 - 2m))
Relative error of ADEV ~ 1/sqrt(2 (n/m - 1)) (El-Sheimy et al. 2008, IEEE
TIM); drawn as the band on the plot and used as fit weights. Host timestamps
give the delivered rate, dropouts and jitter; they do not re-time samples
(the device samples on its own clock). Short dropouts are filled by linear
interpolation before the Allan computation unless receive bunching makes gap
detection unreliable.

White noise N (gyro: angle random walk; accel: velocity random walk)
    = the slope -1/2 line sigma(tau) = N/sqrt(tau) read at tau = 1 s. The white
    region is the longest run of points with local log-log slope -0.5 +- 0.1 below
    the minimum (slopes from a local linear fit over +-0.15 decade), started at its
    first point within +-0.02 of -0.5: behind the device low-pass filter (IM10A
    "Bandwidth", default 20 Hz) the curve approaches the line from below, and
    starting at the +-0.1 edge read N ~5 % low (20 Hz at 200 Hz, 20 synthetic
    seeds: median 0.950); the tighter start reads it ~2 % low there (median 0.981,
    range 0.959-1.003) and changes nothing for unfiltered noise. The line's
    level is the -1/2 term of the least-squares fit below (fitted from that region
    on), because a pure slope-window read is biased high by the flicker floor near
    the region's end (+1.5-2 % in the tests); the slope-window value is kept as
    N_slope and used when the fit is poor or disagrees by > 15 %.
    Units (rad/s)/sqrt(Hz) = rad/sqrt(s), (m/s^2)/sqrt(Hz).
  Per-sample standard deviation at rate f (what SimIM10A and ImuNoise add):
        sigma_d(f) = N * sqrt(f)
  Why: i.i.d. samples of std sigma_d at rate f have AVAR(tau) = sigma_d^2/(f tau)
  exactly (the mean of m samples has variance sigma_d^2/m and the Allan
  variance of white noise equals that variance), so N^2 = sigma_d^2/f. In PSD
  terms N^2 is the two-sided white level; the one-sided PSD is 2 N^2, so the
  rms in a one-sided band B is N*sqrt(2B) = N*sqrt(f) at the Nyquist band
  B = f/2. "N*sqrt(f/2)" is right only if N denotes the one-sided density
  sqrt(2)*N; with the Allan N it understates the per-sample noise by sqrt(2)
  (tests/test_imu_allan.py checks this numerically). The IM10A datasheet
  quotes rms "at Band width=100Hz" (manual s1.1.6); for an IDEAL (brick-wall)
  one-sided 100 Hz band that is N*sqrt(200), reported as
  *_datasheet_equiv_100hz_bw for comparison with the vendor's 0.028-0.07 deg/s
  and 0.75-1 mg ranges only (a real filter's noise bandwidth differs).
  gyro_noise_dps / accel_noise_mg in imu_measured.json = max over the three
  axes of N*sqrt(fs) at the MEASURED rate fs (key "rate_hz"), i.e. the rms over
  the measured Nyquist bandwidth; "per_sample_std_at" repeats it at 25/100/200
  Hz, the simulator's call rates. If the device low-pass filter (vendor
  "Bandwidth", default 20 Hz) is below Nyquist, the directly measured sample
  std is smaller than N*sqrt(fs) (reported as sample_std_over_white); the
  white-equivalent value is the one that reproduces the angle random walk in a
  simulator that adds independent noise per sample.
Bias instability B (flicker floor, IEEE 952): sigma_min = B*sqrt(2 ln2/pi)
    = 0.6643 B, so B = sigma_min/0.6643, at the minimum of the lightly smoothed
    curve within tau <= max_tau_frac * T. Flagged as an upper bound when the
    minimum sits at the edge of that range. When white/rate-random-walk terms
    overlap the floor, sigma_min lies above it and B is overstated; the
    cross-check is a nonnegative least-squares fit of the IEEE 952 C.2 form
    AVAR = N^2/tau + (0.6643 B)^2 + K^2 tau/3 + R^2 tau^2/2 ("fit" in the report).
Rate random walk K: the slope +1/2 line sigma(tau) = K*sqrt(tau/3), read at
    tau = 3 s (sigma(3 s) = K on that line). "Visible" only when >= 3 points past
    the minimum rise above it by more than 2 relative errors with slope > 0.25.
    The line's level is then taken from the least-squares fit above (headline
    "K"; it separates the floor), with the graphical read kept as "K_graph" =
    median over the rising points of sqrt((sigma^2 - sigma_min^2) * 3/tau), i.e.
    the +1/2 line through the floor-subtracted points, evaluated at tau = 3 s.
    When not visible, K is null and "K_upper" is an APPROXIMATE one-sided 95 %
    bound: the minimum over tau >= tau_min of the per-tau 95 % bound
    sigma(tau)*sqrt(edf/chi2_0.05(edf))*sqrt(3/tau) (edf for random-walk noise,
    NIST SP 1065). Taking the minimum over several tau lowers the coverage below
    95 % (a 16-seed synthetic Monte Carlo at 200 Hz covered the true K in 94 % of
    2 h records and 100 % of 30 min records).
    Units (rad/s)/sqrt(s). Sampling scatter is large: near tau = T/5 an Allan
    point has only ~3 degrees of freedom, so K from a 2 h record scatters by
    tens of percent.
    This IS ImuNoise.gyro_bias_walk: ImuNoise adds N(0,1)*gyro_bias_walk*sqrt(dt)
    to its bias every step, a Wiener process with diffusion K, ADEV K*sqrt(tau/3).
Latency: the tap capture gives the OFFSET of this IMU's stamps relative to a
    reference sensor for the same physical tap. Only offset + the reference's own
    latency (--ref-latency-ms, which you must know from elsewhere) is an absolute
    stamp latency; ImuNoise.delay_steps is filled from that sum and is 0 otherwise.
    The bag's receive-minus-stamp time is reported separately (transport only).
Constants: 1 deg = pi/180 rad; deg/h = deg/s * 3600; deg/sqrt(h) =
    (deg/s)/sqrt(Hz) * 60; 1 mg = 1e-3 * 9.80665 m/s^2 (standard gravity =
    imu_sim.G); 1 T = 1e4 G = 1e7 mGauss = 1e6 uT; pressure in Pa. Local g from
    --local-g, or WGS84 normal gravity from --latitude-deg/--height-m, else 9.80665.
Stationary bias: mean over the (trimmed) record, per axis. It includes Earth
    rotation (<= 0.0042 deg/s) and is ONE sample of the turn-on bias. The
    accelerometer mean contains gravity; from one pose only the along-gravity
    part |mean| - g is observable, so the per-axis accelerometer bias comes from
    the six-face rotation capture (rotations -> stationary --rotations-report).
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # login nodes cap threads/virtual memory per user

import argparse
import datetime as _dt
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

__version__ = "1.2.0"

G0 = 9.80665
DEG = math.pi / 180.0
MG = 1e-3 * G0
MGAUSS_PER_T = 1e7
UT_PER_T = 1e6
FLICKER = math.sqrt(2.0 * math.log(2.0) / math.pi)  # 0.6643: sigma_min = B * FLICKER
EARTH_RATE_DPS = 360.0 / 86164.0905
#: |local log-log slope + 0.5| allowed at the FIRST point of the white-noise region (the rest of
#: the region allows 0.1); see fit_allan
WHITE_START_SLOPE_TOL = 0.02

#: Copy of bhl_robust.eval.imu_sim.IM10A_DATASHEET (repo, 2026-09-29), kept here so
#: this script needs neither the repo package nor mujoco. tests/test_imu_allan.py
#: asserts it still equals the repo dict. Used only as the fallback for keys a
#: recording cannot measure; see "provenance".
IM10A_DATASHEET = {
    "gyro_noise_dps": 0.07,
    "gyro_bias_dps": 1.0,
    "accel_noise_mg": 1.0,
    "accel_bias_mg": 40.0,
    "mag_resolution_mgauss": 0.0667,
    "baro_noise_pa": 0.5,
    "max_rate_hz": 200.0,
    "source": "Hiwonder IMU module user manual s1 (datasheet upper bounds), retrieved 2026-09-29",
}

TYPE_IMU = ("sensor_msgs/msg/Imu", "sensor_msgs/Imu")
TYPE_MAG = ("sensor_msgs/msg/MagneticField", "sensor_msgs/MagneticField",
            "geometry_msgs/msg/Vector3Stamped", "geometry_msgs/Vector3Stamped")
TYPE_BARO = ("sensor_msgs/msg/FluidPressure", "sensor_msgs/FluidPressure")
TYPE_TEMP = ("sensor_msgs/msg/Temperature", "sensor_msgs/Temperature")

DEFAULT_FACES = "+Z,+Y,+Z,-Y,+Z,-X,+Z,+X,+Z,+Z,+Z,-Z,+Z"
DEFAULT_MOVES = "+X90,-X90,-X90,+X90,+Y90,-Y90,-Y90,+Y90,+Z90,-Z90,+X180,-X180"


# ----------------------------------------------------------------------------- small helpers
def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _odd(w: int) -> int:
    w = max(1, int(w))
    return w if w % 2 else w + 1


def _moving_mean(x, w: int):
    """Centered moving mean along axis 0 with a shrinking window at the edges."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if w <= 1 or n == 0:
        return x.copy()
    h = w // 2
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)], axis=0)
    i = np.arange(n)
    lo, hi = np.clip(i - h, 0, n), np.clip(i + h + 1, 0, n)
    cnt = (hi - lo).astype(float).reshape((n,) + (1,) * (x.ndim - 1))
    return (c[hi] - c[lo]) / cnt


def _runs(mask):
    """(start, stop) pairs, stop exclusive, of the True runs of a boolean array."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _pct(x, qs=(1, 5, 50, 95, 99)):
    x = np.asarray(x, float)
    return {f"p{q:02d}": float(np.percentile(x, q)) for q in qs}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(_jsonable(obj), indent=2) + "\n")


def normal_gravity(lat_deg: float, h_m: float = 0.0) -> float:
    """WGS84 Somigliana normal gravity with the free-air gradient (m/s^2)."""
    s2 = math.sin(math.radians(lat_deg)) ** 2
    g = 9.7803253359 * (1 + 0.00193185265241 * s2) / math.sqrt(1 - 0.00669437999013 * s2)
    return g - 3.086e-6 * h_m


def _local_g(args):
    if getattr(args, "local_g", None):
        return float(args.local_g), "--local-g"
    if getattr(args, "latitude_deg", None) is not None:
        return normal_gravity(args.latitude_deg, args.height_m or 0.0), \
            f"WGS84 normal gravity at {args.latitude_deg} deg, {args.height_m or 0.0} m"
    return G0, "standard gravity 9.80665 (pass --local-g or --latitude-deg for local g)"


# ----------------------------------------------------------------------------- readers
class _Grow:
    def __init__(self, cols: int, cap: int):
        self.a = np.empty((max(int(cap), 1024), cols))
        self.n = 0

    def add(self, row):
        if self.n == self.a.shape[0]:
            self.a = np.concatenate([self.a, np.empty_like(self.a)])
        self.a[self.n] = row
        self.n += 1

    def get(self):
        return self.a[: self.n].copy()


def _bag_dir(p: Path) -> Path:
    if p.is_file() and p.suffix in (".mcap", ".db3") and (p.parent / "metadata.yaml").exists():
        return p.parent
    return p


def _any_reader(path: Path):
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise SystemExit("reading a ROS bag needs the pure-Python 'rosbags' package: pip install rosbags "
                         "(no ROS install needed) - or pass a CSV (see --help)") from exc
    try:
        from rosbags.typesys import Stores, get_typestore
        return AnyReader([path], default_typestore=get_typestore(Stores.ROS2_HUMBLE))
    except (ImportError, TypeError):  # older rosbags without typestores
        return AnyReader([path])


def _pick_topic(by_topic: dict, requested, types, kind: str, required: bool):
    listing = ", ".join(f"{k} [{v}]" for k, v in sorted(by_topic.items()))
    if requested:
        if requested not in by_topic:
            raise SystemExit(f"{kind} topic {requested!r} is not in the bag; it has: {listing}")
        return requested
    cands = sorted(t for t, ty in by_topic.items() if ty in types)
    if len(cands) == 1:
        return cands[0]
    if required:
        what = "no" if not cands else "several"
        raise SystemExit(f"{what} {kind} topics of type {types[0]}; pass --{kind}-topic. Bag topics: {listing}")
    if cands:
        _log(f"note: several {kind} topics {cands}; none used (pass --{kind}-topic)")
    return None


def _stamp(msg) -> float:
    s = msg.header.stamp
    return float(s.sec) + float(s.nanosec) * 1e-9


def read_bag(path, imu_topic=None, mag_topic=None, baro_topic=None, temp_topic=None, aux=True) -> dict:
    bag = _bag_dir(Path(path))
    with _any_reader(bag) as reader:
        conns = list(reader.connections)
        by_topic = {}
        for c in conns:
            by_topic.setdefault(c.topic, c.msgtype)
        sel = {"imu": _pick_topic(by_topic, imu_topic, TYPE_IMU, "imu", True)}
        if aux:
            sel["mag"] = _pick_topic(by_topic, mag_topic, TYPE_MAG, "mag", False)
            sel["baro"] = _pick_topic(by_topic, baro_topic, TYPE_BARO, "baro", False)
            sel["temp"] = _pick_topic(by_topic, temp_topic, TYPE_TEMP, "temp", False)
        use = [c for c in conns if c.topic in {v for v in sel.values() if v}]
        cap = {}
        for c in use:
            cap[c.topic] = cap.get(c.topic, 0) + int(getattr(c, "msgcount", 0) or 0)
        width = {"imu": 13, "mag": 5, "baro": 3, "temp": 3}
        kind_of = {v: k for k, v in sel.items() if v}
        buf = {k: _Grow(width[k], cap.get(sel[k], 0)) for k in kind_of.values()}
        frames, cov_first, total = set(), None, 0
        for conn, ts, raw in reader.messages(connections=use):
            msg = reader.deserialize(raw, conn.msgtype)
            k = kind_of[conn.topic]
            tr = ts * 1e-9
            th = _stamp(msg)
            if k == "imu":
                w, a, q = msg.angular_velocity, msg.linear_acceleration, msg.orientation
                oc = msg.orientation_covariance
                buf[k].add((th, tr, w.x, w.y, w.z, a.x, a.y, a.z, q.x, q.y, q.z, q.w, float(oc[0])))
                frames.add(msg.header.frame_id)
                if cov_first is None:
                    cov_first = {"orientation": [float(v) for v in oc],
                                 "angular_velocity": [float(v) for v in msg.angular_velocity_covariance],
                                 "linear_acceleration": [float(v) for v in msg.linear_acceleration_covariance]}
            elif k == "mag":
                v = msg.magnetic_field if hasattr(msg, "magnetic_field") else msg.vector
                buf[k].add((th, tr, v.x, v.y, v.z))
            elif k == "baro":
                buf[k].add((th, tr, msg.fluid_pressure))
            else:
                buf[k].add((th, tr, msg.temperature))
            total += 1
            if total % 250000 == 0:
                _log(f"  read {total} messages")
    A = buf["imu"].get()
    if A.shape[0] < 3:
        raise SystemExit(f"IMU topic {sel['imu']} has only {A.shape[0]} messages")
    quat = A[:, 8:12]
    out = {"source": f"rosbag2:{bag}", "kind": "bag", "topics": by_topic, "selected": sel,
           "imu": {"t": A[:, 0], "t_recv": A[:, 1], "gyro": A[:, 2:5], "accel": A[:, 5:8],
                   "quat": None if np.all(quat == 0) else quat,
                   "orientation_unavailable_frac": float(np.mean(A[:, 12] == -1.0)),
                   "frame_ids": sorted(frames), "cov_first": cov_first}}
    for k in ("mag", "baro", "temp"):
        if k in buf:
            B = buf[k].get()
            out[k] = {"t": B[:, 0], "t_recv": B[:, 1], "v": B[:, 2:] if k == "mag" else B[:, 2]}
        else:
            out[k] = None
    return out


def from_arrays(t, gyro, accel, quat_xyzw=None, t_recv=None, source="arrays") -> dict:
    """In-memory input in the reader format. SI units: t in s, gyro in rad/s, accel (specific
    force) in m/s^2, quaternion columns x,y,z,w body->world. Rows with a non-finite t/gyro/accel
    value are dropped. Used by read_csv and by callers that already hold the samples."""
    t = np.asarray(t, float).reshape(-1)
    G = np.asarray(gyro, float).reshape(-1, 3)
    A = np.asarray(accel, float).reshape(-1, 3)
    ok = np.isfinite(t) & np.all(np.isfinite(G), 1) & np.all(np.isfinite(A), 1)
    imu = {"t": t[ok], "t_recv": None if t_recv is None else np.asarray(t_recv, float).reshape(-1)[ok],
           "gyro": G[ok], "accel": A[ok], "quat": None, "orientation_unavailable_frac": None,
           "frame_ids": [], "cov_first": None}
    if quat_xyzw is not None:
        Q = np.asarray(quat_xyzw, float).reshape(-1, 4)[ok]
        if np.all(np.isfinite(Q)) and not np.all(Q == 0):
            imu["quat"] = Q
    return {"source": source, "kind": "arrays", "topics": {}, "selected": {}, "imu": imu,
            "mag": None, "baro": None, "temp": None}


def read_csv(path, gyro_unit="rad/s", accel_unit="m/s2", mag_unit="T", baro_unit="Pa") -> dict:
    path = Path(path)
    try:
        import pandas as pd
        df = pd.read_csv(path, comment="#", skipinitialspace=True)
        cols = {str(c).strip().lower(): df[c].to_numpy(dtype=float) for c in df.columns}
    except ImportError:
        arr = np.genfromtxt(path, delimiter=",", names=True, comments="#", dtype=float, autostrip=True)
        cols = {n.lower(): np.asarray(arr[n], float) for n in arr.dtype.names}
    need = ("t", "gx", "gy", "gz", "ax", "ay", "az")
    miss = [c for c in need if c not in cols]
    if miss:
        raise SystemExit(f"CSV {path} lacks column(s) {miss}; the header must contain t,gx,gy,gz,ax,ay,az")
    gs = {"rad/s": 1.0, "deg/s": DEG}[gyro_unit]
    as_ = {"m/s2": 1.0, "g": G0}[accel_unit]
    ms = {"T": 1.0, "uT": 1e-6, "G": 1e-4, "mG": 1e-7}[mag_unit]
    bs = {"Pa": 1.0, "hPa": 100.0}[baro_unit]
    t = cols["t"]
    G = np.stack([cols["gx"], cols["gy"], cols["gz"]], 1) * gs
    A = np.stack([cols["ax"], cols["ay"], cols["az"]], 1) * as_
    Q = (np.stack([cols["qx"], cols["qy"], cols["qz"], cols["qw"]], 1)
         if all(c in cols for c in ("qx", "qy", "qz", "qw")) else None)
    out = from_arrays(t, G, A, Q, cols.get("t_recv"), source=f"csv:{path}")
    out["kind"] = "csv"
    if all(c in cols for c in ("mx", "my", "mz")):
        M = np.stack([cols["mx"], cols["my"], cols["mz"]], 1) * ms
        k = np.isfinite(t) & np.all(np.isfinite(M), 1)
        if k.sum() > 10:
            out["mag"] = {"t": t[k], "t_recv": None, "v": M[k]}
    if "p" in cols:
        k = np.isfinite(t) & np.isfinite(cols["p"])
        if k.sum() > 10:
            out["baro"] = {"t": t[k], "t_recv": None, "v": cols["p"][k] * bs}
    if "temp" in cols:
        k = np.isfinite(t) & np.isfinite(cols["temp"])
        if k.sum() > 1:
            out["temp"] = {"t": t[k], "t_recv": None, "v": cols["temp"][k]}
    return out


def load(path, args, imu_topic=None, aux=True) -> dict:
    p = Path(path)
    if p.is_file() and p.suffix.lower() in (".csv", ".txt"):
        return read_csv(p, args.gyro_unit, args.accel_unit, args.mag_unit, args.baro_unit)
    return read_bag(p, imu_topic if imu_topic is not None else args.imu_topic,
                    getattr(args, "mag_topic", None), getattr(args, "baro_topic", None),
                    getattr(args, "temp_topic", None), aux=aux)


def choose_time(imu: dict, pref: str, warnings: list) -> tuple[np.ndarray, str]:
    """Measurement time: header stamps unless unusable (zero/constant/non-monotonic)."""
    th, tr = imu["t"], imu.get("t_recv")
    if pref == "receive":
        if tr is None:
            raise SystemExit("--time-source receive needs bag receive times (or a t_recv CSV column)")
        return tr, "receive"
    bad = (not np.all(np.isfinite(th))) or np.all(th == 0) or np.ptp(th) == 0
    back = float(np.mean(np.diff(th) < 0)) if th.size > 1 else 0.0
    if pref == "header" or (not bad and back <= 0.01):
        if back > 0:
            warnings.append(f"{back:.2%} of header stamps step backwards")
        return th, "header"
    if tr is None:
        raise SystemExit("header stamps are unusable and there are no receive times")
    warnings.append("header stamps unusable (zero, constant or >1% non-monotonic): using bag receive time")
    return tr, "receive"


# ----------------------------------------------------------------------------- timing / data quality
def timing_stats(t, t_recv=None, nominal=None) -> dict:
    t = np.asarray(t, float)
    n = t.size
    dt = np.diff(t)
    dur = float(t[-1] - t[0])
    rate = (n - 1) / dur
    med = float(np.median(dt))
    bunched = med < 0.5 / rate
    nominal_ok = bool(nominal) and abs(rate / nominal - 1) <= 0.05   # else the module lowered its rate
    ref = 1.0 / nominal if nominal_ok else (1.0 / rate if bunched else med)
    gaps = dt > 1.5 * ref
    missing = int(np.sum(np.maximum(np.rint(dt[gaps] / ref) - 1, 0)))
    out = {"samples": n, "duration_s": dur, "rate_hz": rate,
           "median_dt_rate_hz": (1.0 / med) if med > 0 else None,
           "dt_ms": {"median": med * 1e3, "mean": float(dt.mean() * 1e3), "std": float(dt.std() * 1e3),
                     "min": float(dt.min() * 1e3), "max": float(dt.max() * 1e3),
                     **{k: v * 1e3 for k, v in _pct(dt, (1, 99)).items()}},
           "nonmonotonic_steps": int(np.sum(dt <= 0)),
           "receive_bunching": bool(bunched),
           "gap_threshold_ms": 1.5 * ref * 1e3,
           "gaps": int(gaps.sum()), "missing_samples_est": missing,
           "missing_frac": missing / (n + missing), "longest_gap_s": float(dt.max())}
    if nominal:
        out.update(nominal_rate_hz=float(nominal), delivered_over_nominal=rate / nominal,
                   missing_vs_nominal_frac=float(1.0 - n / (dur * nominal + 1)))
    if t_recv is not None and np.all(np.isfinite(t_recv)):
        lat = (np.asarray(t_recv) - t) * 1e3
        out["recv_minus_stamp_ms"] = {"median": float(np.median(lat)), "min": float(lat.min()),
                                      "max": float(lat.max()), **_pct(lat, (5, 95))}
    return out


def fill_gaps(t, Y, ref_dt):
    """Put samples on the device-index grid, linearly interpolating detected gaps."""
    steps = np.maximum(np.rint(np.diff(t) / ref_dt).astype(np.int64), 1)
    idx = np.concatenate([[0], np.cumsum(steps)])
    grid = np.arange(idx[-1] + 1)
    out = np.empty((grid.size, Y.shape[1]))
    for j in range(Y.shape[1]):
        out[:, j] = np.interp(grid, idx, Y[:, j])
    return out, int(grid.size - Y.shape[0])


def duplicate_stats(G, A) -> dict:
    same = np.all(G[1:] == G[:-1], 1) & np.all(A[1:] == A[:-1], 1)
    per = [float(np.mean(c[1:] == c[:-1])) for c in (*G.T, *A.T)]
    chance = float(np.prod(per))
    frac = float(same.mean())
    return {"full_duplicates": int(same.sum()), "fraction": frac, "chance_fraction": chance,
            "per_channel_repeat_fraction": dict(zip(("gx", "gy", "gz", "ax", "ay", "az"), per)),
            "suspicious": bool(frac > 2 * chance + 1e-3)}


def quant_step(y) -> dict:
    y = np.asarray(y, float)
    u = np.unique(y)
    if u.size < 3:
        return {"quantized": True, "unique_values": int(u.size), "step": None}
    if u.size > 0.5 * y.size:
        return {"quantized": False, "unique_values": int(u.size), "step": None}
    d = np.diff(u)
    q = float(np.median(d))
    mult = float(np.mean(np.abs(d / q - np.rint(d / q)) < 0.05)) if q > 0 else 0.0
    return {"quantized": mult > 0.9, "unique_values": int(u.size), "step": q if mult > 0.9 else None}


def hf_std(y, fs, window_s=1.0) -> float:
    """Std of y minus its centered moving mean (bias-corrected): per-sample noise
    without the slow wander. Equals sigma_d for white noise."""
    w = _odd(round(window_s * fs))
    if w < 3 or y.size < 3 * w:
        return float(np.std(np.diff(y)) / math.sqrt(2))
    r = y - _moving_mean(y, w)
    return float(np.std(r) * math.sqrt(w / (w - 1)))


def drift_checks(Y, fs, white_N, block_s=5.0, side=3) -> dict:
    """Warm-up trend and abrupt bias steps (e.g. vendor gyro auto-calibration)."""
    blk = max(1, int(round(block_s * fs)))
    nb = Y.shape[0] // blk
    if nb < 4 * side:
        return {"block_s": block_s, "blocks": int(nb), "steps": [], "note": "record too short"}
    M = Y[: nb * blk].reshape(nb, blk, 3).mean(1)
    tb = (np.arange(nb) + 0.5) * block_s
    k10 = max(1, nb // 10)
    trend = np.polyfit(tb / 3600.0, M, 1)[0]
    c = np.concatenate([np.zeros((1, 3)), np.cumsum(M, 0)])
    b = np.arange(side, nb - side + 1)               # boundary before block b
    D = (c[b + side] - c[b]) / side - (c[b] - c[b - side]) / side
    steps = []
    for j in range(3):
        med = np.median(D[:, j])
        mad = 1.4826 * np.median(np.abs(D[:, j] - med))
        wn = math.sqrt(2.0) * white_N[j] / math.sqrt(side * block_s)
        thr = max(8.0 * mad, 6.0 * wn)
        hit = np.abs(D[:, j] - med) > thr
        for s, e in _runs(hit):
            i = s + int(np.argmax(np.abs(D[s:e, j] - med)))
            steps.append({"axis": "xyz"[j], "t_s": float(b[i] * block_s), "size": float(D[i, j] - med)})
    return {"block_s": block_s, "first_tenth_minus_last_tenth": (M[:k10].mean(0) - M[-k10:].mean(0)),
            "trend_per_hour": trend, "steps": steps, "block_t_s": tb, "block_means": M}


# ----------------------------------------------------------------------------- Allan deviation
def overlapping_adev(y, fs, per_decade=20):
    """Overlapping Allan deviation of rate samples y at rate fs (see module docstring)."""
    y = np.asarray(y, float)
    y = y - y.mean()
    n = y.size
    theta = np.zeros(n + 1)
    np.cumsum(y, out=theta[1:])
    theta /= fs
    m_max = n // 2
    if m_max < 1:
        raise ValueError("need at least 2 samples")
    num = max(2, int(math.log10(max(m_max, 1)) * per_decade) + 1)
    ms = np.unique(np.rint(np.logspace(0, math.log10(m_max), num)).astype(np.int64))
    taus = ms / fs
    avar = np.empty(ms.size)
    for i, m in enumerate(ms):
        d = theta[2 * m:] - 2.0 * theta[m:n + 1 - m] + theta[:n + 1 - 2 * m]
        avar[i] = np.dot(d, d) / (2.0 * taus[i] ** 2 * d.size)
    return taus, np.sqrt(avar), ms


def edf_rwfm(n, m):
    """Equivalent degrees of freedom of the overlapping AVAR for random-walk (RRW-type) noise,
    NIST SP 1065 (Riley 2008) Table 5, with N = n + 1 phase points. The smallest edf of the
    usual noise types, hence the conservative choice for bounds in the RRW region."""
    N = n + 1.0
    m = np.asarray(m, float)
    return np.maximum((N - 2) / m * ((N - 1) ** 2 - 3 * m * (N - 1) + 4 * m ** 2) / (N - 3) ** 2, 1.0)


def chi2_lower(nu, z=1.6449):
    """Lower 5 % quantile of chi-square(nu) (Wilson-Hilferty; conservative for nu near 1)."""
    nu = np.asarray(nu, float)
    c = 1 - 2 / (9 * nu) - z * np.sqrt(2 / (9 * nu))
    return nu * np.maximum(c, 1e-3) ** 3


def fit_model(taus, adev, rel, mask):
    """Nonnegative least squares of AVAR = sum_p C_p tau^p, p in (-1, 0, 1, 2)."""
    t, v, w = taus[mask], adev[mask] ** 2, 1.0 / (2.0 * rel[mask])
    if t.size < 3:
        return None
    powers = (-1, 0, 1, 2)
    cols = np.stack([t ** p / v for p in powers], 1) * w[:, None]
    scale = np.linalg.norm(cols, axis=0)
    cols = cols / scale
    best = None
    for r in range(1, 5):
        for sub in itertools.combinations(range(4), r):
            c, *_ = np.linalg.lstsq(cols[:, sub], w, rcond=None)
            if np.any(c <= 0):
                continue
            res = float(np.sum((cols[:, sub] @ c - w) ** 2))
            if best is None or res < best[0]:
                best = (res, sub, c)
    if best is None:
        return None
    C = np.zeros(4)
    for j, cj in zip(best[1], best[2]):
        C[j] = cj / scale[j]
    model = np.sqrt(sum(C[j] * taus ** p for j, p in enumerate(powers)))
    rr = model[mask] / adev[mask] - 1.0
    chi = float(np.sqrt(np.mean((rr / rel[mask]) ** 2)))
    return {"N": math.sqrt(C[0]), "B": math.sqrt(C[1]) / FLICKER, "K": math.sqrt(3.0 * C[2]),
            "R": math.sqrt(2.0 * C[3]), "terms": [powers[j] for j in best[1]],
            "tau_range_s": [float(t[0]), float(t[-1])], "rms_rel_resid": float(np.sqrt(np.mean(rr ** 2))),
            "rms_resid_in_rel_errors": chi,
            "model_adev": model}


def _loglog_local(lt, ls, half=0.15):
    """Local linear fit of log10 ADEV on log10 tau within +-half decade (>= 3 points):
    smoothed curve and local slope. Unlike an index-space moving average it keeps power
    laws exact where the tau grid is uneven (m = 1..10)."""
    sm, sl = np.empty_like(ls), np.empty_like(ls)
    for i in range(lt.size):
        k = np.abs(lt - lt[i]) <= half
        if k.sum() < 3:
            k = np.zeros(lt.size, bool)
            k[np.argsort(np.abs(lt - lt[i]))[:3]] = True
        x = lt[k] - lt[i]
        c = np.linalg.lstsq(np.stack([np.ones_like(x), x], 1), ls[k], rcond=None)[0]
        sm[i], sl[i] = c
    return sm, sl


def fit_allan(taus, adev, ms, n, fs, max_tau_frac=0.2) -> dict:
    rel = 1.0 / np.sqrt(2.0 * (n / ms - 1.0))
    lt, ls = np.log10(taus), np.log10(adev)
    if lt.size >= 3:
        sm, slope = _loglog_local(lt, ls)
    else:
        sm, slope = ls.copy(), np.zeros_like(lt)
    T = n / fs
    valid = taus <= max_tau_frac * T
    valid[: min(5, valid.size)] = True
    iv = np.flatnonzero(valid)
    imin = int(iv[np.argmin(sm[iv])])
    s_min = float(10 ** sm[imin])
    at_edge = bool(imin >= iv[-1] - 1)
    cand = valid & (taus < taus[imin]) & (np.abs(slope + 0.5) <= 0.1)
    runs = [r for r in _runs(cand) if r[1] - r[0] >= 2]
    if runs:
        a, b = max(runs, key=lambda r: r[1] - r[0])
        # Start the white region where the curve has settled onto the -1/2 line. Behind a device
        # low-pass filter (IM10A "Bandwidth", default 20 Hz) sigma*sqrt(tau) is still rising by a
        # few % where the slope first enters the +-0.1 window (0.05 s at 200 Hz), which biased N
        # low by ~5 % in the synthetic 20 Hz low-pass test; tightening only the start removes that.
        a2 = a
        while a2 < b - 3 and abs(slope[a2] + 0.5) > WHITE_START_SLOPE_TOL:
            a2 += 1
        if abs(slope[a2] + 0.5) <= WHITE_START_SLOPE_TOL:
            a = a2
        N_slope = float(10 ** np.median(ls[a:b] + 0.5 * lt[a:b]))
        white = [float(taus[a]), float(taus[b - 1])]
        n_method = "slope-window -1/2 line read at tau = 1 s"
    else:
        pool = np.flatnonzero(valid & (taus <= taus[imin]))
        j = int(pool[np.argmin(np.abs(slope[pool] + 0.5))])
        N_slope = float(adev[j] * math.sqrt(taus[j]))
        white = [float(taus[j]), float(taus[j])]
        n_method = "point closest to slope -1/2 (no clean white region)"
    B = s_min / FLICKER
    post = valid & (taus > taus[imin])
    rising = post & (adev > s_min * (1.0 + 2.0 * rel)) & (slope > 0.25)
    visible = int(rising.sum()) >= 3
    K_graph = None
    if visible:
        K_graph = float(np.median(np.sqrt(np.maximum(adev[rising] ** 2 - s_min ** 2, 0) * 3.0 / taus[rising])))
    tail = valid & (taus >= taus[imin])
    ci = np.minimum(np.sqrt(edf_rwfm(n, ms[tail]) / chi2_lower(edf_rwfm(n, ms[tail]))), 20.0)
    K_upper = float(np.min(adev[tail] * ci * np.sqrt(3.0 / taus[tail])))
    lo = white[0] if runs else 3.0 / fs
    model = fit_model(taus, adev, rel, valid & (taus >= lo))
    K, K_method = None, "not visible: see K_upper"
    if visible:
        if model and model["K"] > 0:
            K, K_method = model["K"], "+1/2 line at tau = 3 s from the IEEE 952 least-squares fit"
        else:
            K, K_method = K_graph, "+1/2 line at tau = 3 s through the floor-subtracted rising points"
    use_fit = bool(model and model["N"] > 0 and model["rms_resid_in_rel_errors"] < 5.0 and runs
                   and abs(model["N"] / N_slope - 1) < 0.15)
    N = model["N"] if use_fit else N_slope
    if use_fit:
        n_method = ("-1/2 line at tau = 1 s from the IEEE 952 least-squares fit (separates the floor that "
                    "biases a pure slope-window read high)")
    return {"tau_s": taus, "adev": adev, "rel_err": rel, "slope": slope, "adev_smoothed": 10 ** sm,
            "N": N, "N_slope": N_slope, "N_method": n_method, "white_region_s": white,
            "B": B, "tau_B_s": float(taus[imin]), "adev_min": s_min, "B_is_upper_bound": at_edge,
            "K": K, "K_method": K_method, "K_visible": visible, "K_graph": K_graph, "K_upper": K_upper,
            "fit": model, "tau_max_fit_s": float(max_tau_frac * T)}


# ----------------------------------------------------------------------------- quaternions
def _qmul(a, b):
    aw, ax, ay, az = np.moveaxis(np.asarray(a, float), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, float), -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def _rot_inv(q, v):
    """R(q)^T v for (...,4) wxyz quaternions (body->world), i.e. world vector in body frame."""
    q = np.asarray(q, float)
    w, u = q[..., :1], -q[..., 1:]
    v = np.broadcast_to(np.asarray(v, float), u.shape)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


QUAT_HYPOTHESES = {
    "xyzw fields, body->world (ROS / IMU_INPUT.md contract)": ("xyzw", False),
    "xyzw fields, world->body (conjugate)": ("xyzw", True),
    "w stored in x slot (wxyz order), body->world": ("wxyz", False),
    "w stored in x slot (wxyz order), world->body": ("wxyz", True),
}


def _as_wxyz(Q, order, conj):
    Q = np.asarray(Q, float)
    q = np.stack([Q[:, 3], Q[:, 0], Q[:, 1], Q[:, 2]], 1) if order == "xyzw" else Q.copy()
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    if conj:
        q[:, 1:] *= -1
    return q


def _angle_deg(u, v):
    u = u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-12)
    v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    return np.degrees(np.arccos(np.clip(np.sum(u * v, -1), -1.0, 1.0)))


def quat_checks_static(Q, accel, fs) -> dict:
    qn = np.linalg.norm(Q, axis=1)
    blk = max(1, int(round(fs)))
    nb = Q.shape[0] // blk
    if nb < 2:
        return {"note": "too short"}
    A = accel[: nb * blk].reshape(nb, blk, 3).mean(1)
    Qb = Q[blk // 2: nb * blk: blk][:nb]
    err = {}
    for name, (order, conj) in QUAT_HYPOTHESES.items():
        up = _rot_inv(_as_wxyz(Qb, order, conj), [0.0, 0.0, 1.0])
        err[name] = float(np.median(_angle_deg(A, up)))
    q = _as_wxyz(Qb, "xyzw", False)
    w, x, y, z = q.T
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    tb = np.arange(nb) / 3600.0
    return {"norm": {"min": float(qn.min()), "max": float(qn.max()), "mean": float(qn.mean())},
            "tilt_error_deg_by_interpretation": err,
            "best_interpretation": min(err, key=err.get),
            "note": "one pose cannot separate the conjugate/order hypotheses when the tilt is small; "
                    "the rotations capture decides",
            "yaw_change_deg": float(np.degrees(yaw[-1] - yaw[0])),
            "yaw_trend_deg_per_h": float(np.degrees(np.polyfit(tb, yaw, 1)[0])) if nb > 2 else None}


# ----------------------------------------------------------------------------- stationary
def _series_stats(t, v, fs_hint=None, window_s=10.0):
    """Update rate (distinct consecutive values), mean, and high-passed noise of a slow channel."""
    v = np.asarray(v, float)
    v2 = v.reshape(v.shape[0], -1)
    changed = np.concatenate([[True], np.any(v2[1:] != v2[:-1], 1)])
    dur = float(t[-1] - t[0]) if t.size > 1 else float("nan")
    fs = (v.shape[0] - 1) / dur if dur > 0 else float("nan")
    noise = [hf_std(v2[:, j], fs, window_s) for j in range(v2.shape[1])]
    return {"messages": int(v.shape[0]), "message_rate_hz": fs,
            "distinct_update_rate_hz": (int(changed.sum()) - 1) / dur if dur > 0 else None,
            "mean": v2.mean(0), "noise_hp": noise, "quant": [quant_step(v2[:, j]) for j in range(v2.shape[1])]}


def analyse_stationary(data: dict, args) -> tuple[dict, dict, dict]:
    warnings: list[str] = []
    imu = data["imu"]
    t_all, tsrc = choose_time(imu, args.time_source, warnings)
    t0, t1 = t_all[0] + args.trim_start_s, t_all[-1] - args.trim_end_s
    keep = (t_all >= t0) & (t_all <= t1)
    if keep.sum() < 100:
        raise SystemExit("fewer than 100 IMU samples left after trimming")
    t = t_all[keep]
    G, A = imu["gyro"][keep], imu["accel"][keep]
    Q = imu["quat"][keep] if imu["quat"] is not None else None
    t_recv = imu["t_recv"][keep] if (imu.get("t_recv") is not None and tsrc == "header") else None
    timing = timing_stats(t, t_recv, args.nominal_rate)
    if args.nominal_rate and abs(timing["delivered_over_nominal"] - 1) > 0.05:
        warnings.append(f"delivered rate {timing['rate_hz']:.2f} Hz differs from the configured "
                        f"{args.nominal_rate:g} Hz by >5% (manual s1.6.1: too much output content or too low a "
                        "baud makes the module lower its rate)")
    dup = duplicate_stats(G, A)
    if dup["suspicious"]:
        warnings.append(f"{dup['fraction']:.2%} consecutive samples are exact duplicates on all six channels "
                        f"(chance level {dup['chance_fraction']:.2%}): the publisher repeats samples; "
                        "consider --drop-duplicates")
    if args.drop_duplicates and dup["full_duplicates"]:
        k = np.concatenate([[True], ~(np.all(G[1:] == G[:-1], 1) & np.all(A[1:] == A[:-1], 1))])
        t, G, A = t[k], G[k], A[k]
        Q = Q[k] if Q is not None else None
        t_recv = t_recv[k] if t_recv is not None else None
        timing = timing_stats(t, t_recv, args.nominal_rate)
        timing["duplicates_dropped"] = int((~k).sum())
    nominal_ok = bool(args.nominal_rate) and abs(timing["rate_hz"] / args.nominal_rate - 1) <= 0.05
    ref_dt = 1.0 / args.nominal_rate if nominal_ok else timing["dt_ms"]["median"] / 1e3
    can_fill = (not timing["receive_bunching"]) and timing["dt_ms"]["p01"] > 0.5 * ref_dt * 1e3
    nfill = 0
    Gu, Au = G, A
    if timing["missing_samples_est"] > 0:
        if can_fill:
            Gu, nfill = fill_gaps(t, G, ref_dt)
            Au, _ = fill_gaps(t, A, ref_dt)
            if nfill > 0.01 * G.shape[0]:
                warnings.append(f"{nfill} missing samples ({nfill / Gu.shape[0]:.2%}) were interpolated")
        else:
            warnings.append("receive-time bunching/jitter: gaps not filled; the Allan tau axis assumes "
                            "the delivered samples are consecutive")
    fs = (Gu.shape[0] - 1) / timing["duration_s"]
    n = Gu.shape[0]
    T = n / fs
    if T < 1800:
        warnings.append(f"record is {T / 60:.0f} min; bias instability/rate random walk need >= 2 h "
                        "(30 min minimum, see docs/IMU_RECORDING.md)")

    allan = {"gyro": {}, "accel": {}}
    for name, Y in (("gyro", Gu), ("accel", Au)):
        for i, ax in enumerate("xyz"):
            taus, adev, ms = overlapping_adev(Y[:, i], fs, args.points_per_decade)
            f = fit_allan(taus, adev, ms, n, fs, args.max_tau_frac)
            f["sample_std"] = hf_std(Y[:, i], fs)
            f["sample_std_over_white"] = f["sample_std"] / (f["N"] * math.sqrt(fs))
            allan[name][ax] = f
    for name in ("gyro", "accel"):
        for ax, f in allan[name].items():
            r = f["sample_std_over_white"]
            if r < 0.8:
                warnings.append(f"{name} {ax}: per-sample std is {r:.2f} x N*sqrt(fs): samples are correlated "
                                "(device low-pass 'Bandwidth' below Nyquist, manual s1.6.1); noise keys use the "
                                "white-equivalent N*sqrt(fs)")
            elif r > 1.25:
                warnings.append(f"{name} {ax}: per-sample std is {r:.2f} x N*sqrt(fs): excess fast noise "
                                "(quantization steps, vibration, or steps in the record)")
            if f["B_is_upper_bound"]:
                warnings.append(f"{name} {ax}: Allan minimum at the edge of the usable tau range: B is an upper "
                                "bound (record too short)")
            if f["fit"] and f["N_slope"] > 0 and abs(f["fit"]["N"] / f["N_slope"] - 1) > 0.25:
                warnings.append(f"{name} {ax}: slope-window N and least-squares N differ by "
                                f">25% ({f['N_slope']:.3g} vs {f['fit']['N']:.3g}); using {f['N']:.3g}")

    quant = {c: quant_step(col) for c, col in zip(("gx", "gy", "gz", "ax", "ay", "az"), (*G.T, *A.T))}
    for c, qd in quant.items():
        if qd["step"]:
            sensor = "gyro" if c[0] == "g" else "accel"
            sd = allan[sensor][c[1]]["sample_std"]
            if sd < 2 * qd["step"]:
                warnings.append(f"{c}: noise ({sd:.3g}) is < 2 output steps ({qd['step']:.3g}): the white-noise "
                                "figure is quantization-limited")
    gN = [allan["gyro"][a]["N"] for a in "xyz"]
    aN = [allan["accel"][a]["N"] for a in "xyz"]
    drift = {"gyro": drift_checks(Gu, fs, gN), "accel": drift_checks(Au, fs, aN)}
    if drift["gyro"]["steps"]:
        warnings.append(f"{len(drift['gyro']['steps'])} abrupt gyro bias step(s) (vendor gyro auto-calibration "
                        "left on? manual s1.7.3)")

    g_mean, a_mean = G.mean(0), A.mean(0)
    if np.all(np.abs(g_mean) < 0.02 * DEG):
        warnings.append("gyro mean bias < 0.02 deg/s on every axis (datasheet zero drift is 0.5-1 deg/s): "
                        "vendor gyro auto-calibration was probably ON; the bias/B/K then describe the "
                        "auto-calibrated output")
    g_loc, g_src = _local_g(args)
    amag = float(np.linalg.norm(a_mean))
    iup = int(np.argmax(np.abs(a_mean)))
    grav = {"mean_specific_force_m_s2": a_mean, "magnitude_m_s2": amag, "local_g_m_s2": g_loc,
            "local_g_source": g_src, "magnitude_minus_local_g_mg": (amag - g_loc) / MG,
            "up_axis": ("+" if a_mean[iup] > 0 else "-") + "XYZ"[iup],
            "tilt_from_up_axis_deg": float(np.degrees(np.arccos(min(1.0, abs(a_mean[iup]) / amag))))}
    if 0.8 < amag < 1.2:
        warnings.append("accel magnitude ~1: data look like g units, ROS requires m/s^2 (use --accel-unit g "
                        "for a CSV, fix the driver for a bag)")
    elif not 9.0 < amag < 10.6:
        warnings.append(f"accel magnitude {amag:.3g} m/s^2 is far from g: check units/scaling")
    if np.linalg.norm(g_mean) > 0.5:
        warnings.append("stationary gyro magnitude > 0.5 rad/s: deg/s published as rad/s?")
    if grav["up_axis"] != "+Z":
        warnings.append(f"the up axis reads {grav['up_axis']} (REP-145: a sensor lying Z-up reports +g on Z). "
                        "Expected if it did not lie Z-up; otherwise the gravity sign/axis is wrong")

    mag = baro = temp = None
    for key in ("mag", "baro", "temp"):
        ch = data.get(key)
        if ch is None:
            continue
        tt = ch["t"]
        k = (tt >= t0) & (tt <= t1)
        if k.sum() < 10:
            continue
        st = _series_stats(tt[k], ch["v"][k])
        if key == "mag":
            m_ut = ch["v"][k] * UT_PER_T
            mm = np.linalg.norm(m_ut, axis=1)
            steps = [q["step"] for q in st["quant"] if q["step"]]
            mag = {**st, "field_magnitude_ut": {"mean": float(mm.mean()), "std": float(mm.std())},
                   "mean_ut": m_ut.mean(0), "noise_hp_ut": [x * UT_PER_T for x in st["noise_hp"]],
                   "step_mgauss": (float(np.median(steps)) * MGAUSS_PER_T) if steps else None}
            if args.mag_ref_ut and abs(mm.mean() / args.mag_ref_ut - 1) > 0.1:
                warnings.append(f"mag field {mm.mean():.1f} uT differs from the reference {args.mag_ref_ut} uT "
                                "by >10%: uncalibrated magnetometer or local distortion (steel, electronics)")
        elif key == "baro":
            v = ch["v"][k]
            baro = {**st, "noise_hp_pa": st["noise_hp"][0], "mean_pa": float(v.mean()),
                    "trend_pa_per_h": float(np.polyfit((tt[k] - tt[k][0]) / 3600, v, 1)[0]) if v.size > 2 else None}
        else:
            v = ch["v"][k]
            temp = {"start_c": float(v[: max(1, v.size // 20)].mean()), "end_c": float(v[-max(1, v.size // 20):].mean()),
                    "min_c": float(v.min()), "max_c": float(v.max())}
    qchk = None
    if Q is not None:
        if imu.get("orientation_unavailable_frac"):
            qchk = {"available": False, "reason": "orientation_covariance[0] == -1 (ROS: orientation not provided)"}
        else:
            qchk = {"available": True, **quat_checks_static(Q, A, fs)}

    six = lat = None
    if args.rotations_report:
        rr = json.loads(Path(args.rotations_report).read_text())
        six = rr.get("six_position")
    if args.tap_report:
        lat = json.loads(Path(args.tap_report).read_text())

    measured = compose_measured(data, args, allan, fs, g_mean, grav, six, mag, baro, lat, T, warnings)
    report = {"tool": f"imu_allan.py {__version__}", "created": _dt.datetime.now().isoformat(timespec="seconds"),
              "input": data["source"], "selected_topics": data.get("selected"), "time_source": tsrc,
              "trim_s": [args.trim_start_s, args.trim_end_s],
              "frame_ids": imu.get("frame_ids"), "first_message_covariance": imu.get("cov_first"),
              "orientation_unavailable_frac": imu.get("orientation_unavailable_frac"),
              "timing": timing, "allan_rate_hz": fs, "samples_interpolated": nfill,
              "duplicates": dup, "quantization": quant,
              "gyro_mean_rad_s": g_mean, "gyro_mean_dps": g_mean / DEG,
              "earth_rate_dps_max": EARTH_RATE_DPS,
              "accel_mean_m_s2": a_mean, "gravity": grav,
              "drift": {k: {kk: vv for kk, vv in d.items() if kk not in ("block_means", "block_t_s")}
                        for k, d in drift.items()},
              "allan": {s: {a: {k: v for k, v in f.items() if k not in ("slope",)} for a, f in d.items()}
                        for s, d in allan.items()},
              "magnetometer": mag, "barometer": baro, "temperature": temp, "quaternion": qchk,
              "six_position_from": args.rotations_report, "tap_from": args.tap_report,
              "warnings": warnings}
    plot = {"allan": allan, "drift": drift, "fs": fs, "T": T, "measured": measured,
            "title": f"{args.label or Path(str(data['source']).split(':', 1)[-1]).name}: "
                     f"{T / 3600:.2f} h at {fs:.1f} Hz"}
    return measured, report, plot


def compose_measured(data, args, allan, fs, g_mean, grav, six, mag, baro, lat, T, warnings) -> dict:
    prov = {}
    gN = max(allan["gyro"][a]["N"] for a in "xyz")          # rad/s/sqrt(Hz)
    aN = max(allan["accel"][a]["N"] for a in "xyz")         # m/s^2/sqrt(Hz)
    gB = max(allan["gyro"][a]["B"] for a in "xyz")
    aB = max(allan["accel"][a]["B"] for a in "xyz")

    short = T < 3600.0

    def kpick(sensor):
        """(K or None, kind, walk): walk = max over axes of K where visible, else that axis's upper bound."""
        per = [(allan[sensor][a]["K"] if (allan[sensor][a]["K_visible"] and allan[sensor][a]["K"])
                else None, allan[sensor][a]["K_upper"]) for a in "xyz"]
        vis = [k for k, _ in per if k is not None]
        walk = max((k if k is not None else ub) for k, ub in per)
        low = (" - LOW CONFIDENCE: record < 1 h (synthetic 30 min records that showed K were off by up to "
               "~50 %)") if short else ""
        if len(vis) == 3:
            return max(vis), "measured (max over axes)" + low, walk
        if vis:
            return max(vis), f"measured on {len(vis)}/3 axes; upper bound used for the others" + low, walk
        return None, "upper bound (rate random walk not visible in this record)", walk

    gK, gK_kind, g_walk = kpick("gyro")
    aK, aK_kind, a_walk = kpick("accel")
    out = {}
    out["gyro_noise_dps"] = gN * math.sqrt(fs) / DEG
    prov["gyro_noise_dps"] = f"measured: max-axis Allan N * sqrt({fs:.2f} Hz) (per-sample std at the measured rate)"
    out["gyro_bias_dps"] = float(np.linalg.norm(g_mean) / DEG)
    prov["gyro_bias_dps"] = "measured: |mean gyro vector| over the stationary record (one power cycle; incl. Earth rate)"
    out["accel_noise_mg"] = aN * math.sqrt(fs) / MG
    prov["accel_noise_mg"] = f"measured: max-axis Allan N * sqrt({fs:.2f} Hz)"
    if six and six.get("bias_norm_mg") is not None:
        out["accel_bias_mg"] = float(six["bias_norm_mg"])
        prov["accel_bias_mg"] = "measured: six-position |bias| from the rotations capture"
    else:
        out["accel_bias_mg"] = abs(float(grav["magnitude_minus_local_g_mg"]))
        prov["accel_bias_mg"] = ("measured, PARTIAL: | |mean specific force| - local g |, the along-gravity component "
                                 "only (a lower bound on |bias|; also absorbs scale error). Run the rotations capture "
                                 "and pass --rotations-report for the per-axis bias")
    if mag and mag.get("step_mgauss"):
        out["mag_resolution_mgauss"] = mag["step_mgauss"]
        prov["mag_resolution_mgauss"] = "measured: output step of the magnetometer channel"
    else:
        out["mag_resolution_mgauss"] = IM10A_DATASHEET["mag_resolution_mgauss"]
        prov["mag_resolution_mgauss"] = "datasheet fallback (no magnetometer data or no visible quantization)"
    if baro:
        out["baro_noise_pa"] = float(baro["noise_hp_pa"])
        prov["baro_noise_pa"] = "measured: std of pressure minus its 10 s moving mean"
    else:
        out["baro_noise_pa"] = IM10A_DATASHEET["baro_noise_pa"]
        prov["baro_noise_pa"] = "datasheet fallback (no pressure data in the recording)"
    out["max_rate_hz"] = IM10A_DATASHEET["max_rate_hz"]
    prov["max_rate_hz"] = ("datasheet device capability (SimIM10A rejects rate_hz above it; the delivered rate "
                           "is 'rate_hz')")
    out["source"] = (f"measured with imu_allan.py {__version__} from {data['source']} "
                     f"({T / 3600:.2f} h at {fs:.1f} Hz), {_dt.date.today().isoformat()}")
    prov["source"] = "provenance string of this file"
    out["rate_hz"] = fs
    out["noise_density"] = {"gyro_rad_s_rthz": gN, "gyro_dps_rthz": gN / DEG, "gyro_arw_deg_rth": gN / DEG * 60,
                            "accel_m_s2_rthz": aN, "accel_mg_rthz": aN / MG}
    out["gyro_noise_dps_datasheet_equiv_100hz_bw"] = gN * math.sqrt(200.0) / DEG
    out["accel_noise_mg_datasheet_equiv_100hz_bw"] = aN * math.sqrt(200.0) / MG
    out["datasheet_equiv_note"] = ("N*sqrt(2*100 Hz): rms in an ideal one-sided 100 Hz band, for comparison with the "
                                   "manual's 'Band width=100Hz' rms (0.028-0.07 deg/s, 0.75-1 mg) only")
    out["per_sample_std_at"] = {str(r): {"gyro_dps": gN * math.sqrt(r) / DEG, "gyro_rad_s": gN * math.sqrt(r),
                                         "accel_mg": aN * math.sqrt(r) / MG, "accel_m_s2": aN * math.sqrt(r)}
                                for r in (25, 50, 100, 200)}
    out["gyro_bias_instability_rad_s"] = gB
    out["gyro_bias_instability_dps"] = gB / DEG
    out["gyro_bias_instability_deg_h"] = gB / DEG * 3600
    out["gyro_bias_instability_is_upper_bound"] = any(allan["gyro"][a]["B_is_upper_bound"] for a in "xyz")
    out["gyro_rate_random_walk_rad_s_rtsec"] = gK
    out["gyro_bias_walk_rad_s_rtsec"] = g_walk
    out["gyro_bias_walk_kind"] = gK_kind
    out["accel_bias_instability_mg"] = aB / MG
    out["accel_bias_instability_m_s2"] = aB
    out["accel_rate_random_walk_m_s2_rtsec"] = aK
    out["accel_bias_walk_m_s2_rtsec"] = a_walk
    out["accel_bias_walk_kind"] = aK_kind
    off_ms = abs_ms = None
    if lat and lat.get("offset_ms", {}).get("median") is not None:
        off_ms = float(lat["offset_ms"]["median"])
        if lat.get("latency_ms_estimate") is not None:
            abs_ms = float(lat["latency_ms_estimate"])
    out["imu_tap_offset_ms"] = off_ms
    out["imu_latency_ms"] = abs_ms
    if abs_ms is not None:
        out["imu_latency_note"] = (f"tap offset {off_ms:.2f} ms + reference latency {lat['ref_latency_ms']:g} ms "
                                   "(tap --ref-latency-ms): stamp latency; add the consumer's processing for "
                                   "end-to-end. imu_noise.delay_steps = round(imu_latency_ms * rate / 1000)")
    elif off_ms is not None:
        out["imu_latency_note"] = ("only the RELATIVE tap offset is known (imu_tap_offset_ms; the reference's own "
                                   "latency was not given): imu_noise.delay_steps left at 0. Rerun tap with "
                                   "--ref-latency-ms to make it absolute")
    else:
        out["imu_latency_note"] = "not measured (run the tap capture, pass --tap-report): imu_noise.delay_steps = 0"
    sim_rate = float(args.sim_rate)

    def imu_noise_at(r):
        d = {"gyro_std": gN * math.sqrt(r), "accel_std": aN * math.sqrt(r),
             "gyro_bias": float(np.linalg.norm(g_mean)), "gyro_bias_walk": float(g_walk),
             "delay_steps": int(round(max(abs_ms, 0.0) * r / 1000.0)) if abs_ms is not None else 0}
        return d
    out["imu_noise"] = imu_noise_at(sim_rate)
    out["imu_noise_rate_hz"] = sim_rate
    out["imu_noise_by_rate"] = {str(r): imu_noise_at(r) for r in (25, 50, 100, 200)}
    out["kalibr_imu"] = {"gyroscope_noise_density": gN, "gyroscope_random_walk": float(g_walk),
                         "accelerometer_noise_density": aN, "accelerometer_random_walk": float(a_walk),
                         "update_rate": fs,
                         "note": "Kalibr IMU noise model (github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model), same "
                                 "conventions: noise density = Allan N at tau = 1 s (discrete std = N/sqrt(dt)); "
                                 "random walk = the +1/2 line at tau = 3 s (discrete step = K*sqrt(dt)), here K where "
                                 "measured else its upper bound (see *_bias_walk_kind). Kalibr advises inflating "
                                 "static values for low-cost IMUs."}
    out["per_axis"] = {s: {a: {"N": allan[s][a]["N"], "B": allan[s][a]["B"], "tau_B_s": allan[s][a]["tau_B_s"],
                               "K": allan[s][a]["K"], "K_upper": allan[s][a]["K_upper"],
                               "sample_std": allan[s][a]["sample_std"]} for a in "xyz"}
                       for s in ("gyro", "accel")}
    out["per_axis_units"] = "SI: N in unit/sqrt(Hz), B and sample_std in unit, K in unit/sqrt(s); unit = rad/s or m/s^2"
    out["provenance"] = prov
    out["conversions"] = ("per-sample std = N*sqrt(rate) (Allan N, two-sided white level); B = ADEV_min/0.6643; "
                          "K from sigma = K*sqrt(tau/3) = ImuNoise.gyro_bias_walk; 1 mg = 9.80665e-3 m/s^2; "
                          "1 mGauss = 1e-7 T. See imu_allan.py docstring.")
    out["warnings"] = list(warnings)
    return out


# ----------------------------------------------------------------------------- plotting
def plot_stationary(path: Path, pdata: dict) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _log("matplotlib not installed: no plot")
        return False
    INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    COL = {"x": "#2a78d6", "y": "#eb6834", "z": "#1baf7a"}
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9.5, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.titlecolor": INK, "text.color": INK})
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.6), dpi=110, gridspec_kw={"height_ratios": [3, 2]})
    fig.patch.set_facecolor(SURF)
    m = pdata["measured"]
    spec = (("gyro", "deg/s", 1 / DEG), ("accel", "mg", 1 / MG))
    for col, (sensor, unit, sc) in enumerate(spec):
        ax = axs[0, col]
        ax.set_facecolor(SURF)
        worst = max("xyz", key=lambda a: pdata["allan"][sensor][a]["N"])
        for a in "xyz":
            f = pdata["allan"][sensor][a]
            tau, sig, rel = np.asarray(f["tau_s"]), np.asarray(f["adev"]) * sc, np.asarray(f["rel_err"])
            ax.fill_between(tau, sig * (1 - rel), sig * (1 + rel), color=COL[a], alpha=0.12, lw=0)
            ax.loglog(tau, sig, color=COL[a], lw=1.6, label=f"{a} axis")
            ax.annotate(a, (tau[-1], sig[-1]), xytext=(4, 0), textcoords="offset points", color=INK2,
                        fontsize=9, va="center")
        f = pdata["allan"][sensor][worst]
        tau = np.asarray(f["tau_s"])
        t1 = tau[tau <= f["tau_B_s"] * 3]
        ax.loglog(t1, f["N"] * sc / np.sqrt(t1), color=INK2, lw=1.0, ls="--")
        ax.axhline(f["B"] * FLICKER * sc, color=INK2, lw=1.0, ls="--")
        if f["K"]:
            t2 = tau[tau >= f["tau_B_s"] / 3]
            ax.loglog(t2, f["K"] * sc * np.sqrt(t2 / 3.0), color=INK2, lw=1.0, ls="--")
        ax.axvline(f["tau_max_fit_s"], color=MUTED, lw=0.8, ls=":")
        if sensor == "gyro":
            txt = (f"fits ({worst} axis, dashed):\nN = {f['N'] / DEG:.3g} deg/s/sqrt(Hz) = {f['N'] / DEG * 60:.3g} deg/sqrt(h)\n"
                   f"B = {f['B'] / DEG * 3600:.3g} deg/h at tau = {f['tau_B_s']:.0f} s"
                   + (" (upper bound)" if f["B_is_upper_bound"] else "") + "\n"
                   + (f"K = {f['K']:.3g} rad/s/sqrt(s)" if f["K"] else f"K not visible (<= {f['K_upper']:.2g} rad/s/sqrt(s))")
                   + f"\nper-sample std at {pdata['fs']:.0f} Hz = {m['gyro_noise_dps']:.3g} deg/s")
        else:
            txt = (f"fits ({worst} axis, dashed):\nN = {f['N'] / MG * 1e3:.3g} ug/sqrt(Hz)\n"
                   f"B = {f['B'] / MG:.3g} mg at tau = {f['tau_B_s']:.0f} s"
                   + (" (upper bound)" if f["B_is_upper_bound"] else "") + "\n"
                   + (f"K = {f['K']:.3g} m/s^2/sqrt(s)" if f["K"] else f"K not visible (<= {f['K_upper']:.2g} m/s^2/sqrt(s))")
                   + f"\nper-sample std at {pdata['fs']:.0f} Hz = {m['accel_noise_mg']:.3g} mg")
        ax.text(0.02, 0.03, txt, transform=ax.transAxes, fontsize=8.5, color=INK, va="bottom",
                bbox={"facecolor": SURF, "edgecolor": GRID, "boxstyle": "round,pad=0.4"})
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.set_xlabel("averaging time tau (s)   [dotted: fit limit]")
        ax.set_ylabel(f"Allan deviation ({unit})")
        ax.set_title(f"{'Gyroscope' if sensor == 'gyro' else 'Accelerometer'} Allan deviation", loc="left")
        ax.legend(frameon=False, loc="upper right")
        for s in ax.spines.values():
            s.set_color(AXIS)
        axb = axs[1, col]
        axb.set_facecolor(SURF)
        d = pdata["drift"][sensor]
        if "block_means" in d:
            Mb = np.asarray(d["block_means"]) * sc
            tb = np.asarray(d["block_t_s"]) / 60.0
            for i, a in enumerate("xyz"):
                axb.plot(tb, Mb[:, i] - Mb[:, i].mean(), color=COL[a], lw=1.2, label=f"{a} axis")
            for st in d["steps"]:
                axb.axvline(st["t_s"] / 60.0, color=MUTED, lw=0.8, ls=":")
        axb.grid(True, color=GRID, lw=0.6)
        axb.set_xlabel("time (min)")
        axb.set_ylabel(f"{d.get('block_s', 5):g} s mean minus record mean ({unit})")
        axb.set_title("Slow drift (warm-up, temperature, auto-calibration steps)", loc="left", fontsize=9.5)
        axb.legend(frameon=False, loc="upper right", ncol=3)
        for s in axb.spines.values():
            s.set_color(AXIS)
    fig.suptitle(f"IMU stationary record - {pdata['title']}", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------- rotations
def _signed_perms():
    out = []
    for p in itertools.permutations(range(3)):
        for s in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for r, c in enumerate(p):
                M[r, c] = s[r]
            out.append(M)
    return out


def _integrate_batch(W, fs):
    """Integrate body rates W (H, n, 3) -> quaternions (H, 4) mapping end-body to start-body."""
    H = W.shape[0]
    q = np.tile([1.0, 0.0, 0.0, 0.0], (H, 1))
    dt = 1.0 / fs
    for k in range(W.shape[1]):
        w = W[:, k, :]
        nw = np.linalg.norm(w, axis=1)
        th = nw * dt
        ax = w / np.maximum(nw, 1e-15)[:, None]
        dq = np.concatenate([np.cos(th / 2)[:, None], np.sin(th / 2)[:, None] * ax], 1)
        q = _qmul(q, dq)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _mean_quat_wxyz(q):
    q = q * np.sign(q @ q[0])[:, None]
    m = q.mean(0)
    return m / np.linalg.norm(m)


def _parse_move(s):
    s = s.strip().upper()
    return (1 if s[0] == "+" else -1), s[1], float(s[2:])


def analyse_rotations(data, args) -> dict:
    warnings = []
    imu = data["imu"]
    t, tsrc = choose_time(imu, args.time_source, warnings)
    G, A, Q = imu["gyro"], imu["accel"], imu["quat"]
    fs = (t.size - 1) / (t[-1] - t[0])
    w = _odd(round(0.5 * fs))
    gm = _moving_mean(G, w)
    gsd = np.sqrt(np.maximum(_moving_mean(G * G, w) - gm * gm, 0)).max(1)
    am = _moving_mean(A, w)
    asd = np.sqrt(np.maximum(_moving_mean(A * A, w) - am * am, 0)).max(1)
    static = (gsd < args.static_gyro_std) & (asd < args.static_accel_std) & \
             (np.linalg.norm(gm, axis=1) < args.static_rate)
    trim = int(round(0.25 * fs))
    holds = []
    for s, e in _runs(static):
        s2, e2 = s + trim, e - trim
        if e2 - s2 < args.min_hold_s * fs:
            continue
        f = A[s2:e2].mean(0)
        i = int(np.argmax(np.abs(f)))
        holds.append({"start": s2, "stop": e2, "t0_s": float(t[s2] - t[0]), "t1_s": float(t[e2 - 1] - t[0]),
                      "face": ("+" if f[i] > 0 else "-") + "XYZ"[i],
                      "tilt_deg": float(np.degrees(np.arccos(min(1.0, abs(f[i]) / np.linalg.norm(f))))),
                      "specific_force_m_s2": f, "gyro_mean_rad_s": G[s2:e2].mean(0), "samples": e2 - s2})
    if len(holds) < 2:
        raise SystemExit(f"found {len(holds)} static holds; need >= 2 (check --static-* thresholds / the capture)")
    g_loc, g_src = _local_g(args)
    faces = {}
    for h in holds:
        if h["tilt_deg"] < args.max_face_tilt_deg:
            faces.setdefault(h["face"], []).append((h["specific_force_m_s2"], h["samples"]))
    F = {k: np.average(np.stack([v[0] for v in vs]), axis=0, weights=[v[1] for v in vs]) for k, vs in faces.items()}
    six = {"faces_seen": sorted(F), "local_g_m_s2": g_loc, "local_g_source": g_src, "axes": {}}
    bias = np.full(3, np.nan)
    for i, a in enumerate("XYZ"):
        if f"+{a}" in F and f"-{a}" in F:
            fp, fm = F[f"+{a}"][i], F[f"-{a}"][i]
            bias[i] = 0.5 * (fp + fm)
            six["axes"][a] = {"bias_m_s2": bias[i], "bias_mg": bias[i] / MG,
                              "scale_error_pct": 100 * ((fp - fm) / (2 * g_loc) - 1)}
    six["bias_mg"] = bias / MG
    six["bias_norm_mg"] = float(np.linalg.norm(bias) / MG) if np.all(np.isfinite(bias)) else None
    if six["bias_norm_mg"] is None:
        warnings.append(f"six-position incomplete: faces seen {sorted(F)}; need +-X, +-Y and +-Z holds")
    # accelerations corrected with the six-position bias/scale, so the gyro-vs-accel check below
    # measures gyro/frame errors rather than the accelerometer's own bias (a 25 mg bias alone tilts
    # the gravity direction by ~1.4 deg)
    b_corr = np.where(np.isfinite(bias), bias, 0.0)
    s_corr = np.array([six['axes'][a]['scale_error_pct'] / 100 if a in six['axes'] else 0.0 for a in 'XYZ'])
    for h in holds:
        h['specific_force_corrected_m_s2'] = (h['specific_force_m_s2'] - b_corr) / (1.0 + s_corr)
    six['used_to_correct_consistency_check'] = bool(np.any(np.isfinite(bias)))

    perms = _signed_perms()
    scales = [(1.0, "rad/s"), (DEG, "deg/s published in a rad/s field"),
              (1.0 / DEG, "57.3x too small (deg->rad conversion applied twice)")]
    hyps = [(P, s, lab) for s, lab in scales for P in perms]
    ident = next(i for i, (P, s, _) in enumerate(hyps) if s == 1.0 and np.allclose(P, np.eye(3)))
    Pst = np.stack([h[0] * h[1] for h in hyps])
    moves, errs = [], []
    for h0, h1 in zip(holds[:-1], holds[1:]):
        s, e = h0["stop"], h1["start"]
        bias = 0.5 * (h0["gyro_mean_rad_s"] + h1["gyro_mean_rad_s"])
        seg = G[s:e] - bias
        ang = seg.sum(0) / fs
        dom = int(np.argmax(np.abs(ang)))
        sgn = 1 if ang[dom] >= 0 else -1
        W = np.einsum("hij,nj->hni", Pst, seg)
        qrel = _integrate_batch(W, fs)
        fpred = _rot_inv(qrel, h0["specific_force_corrected_m_s2"])
        err = _angle_deg(fpred, np.broadcast_to(h1["specific_force_corrected_m_s2"], fpred.shape))
        raw = _angle_deg(_rot_inv(qrel[ident], h0["specific_force_m_s2"]), h1["specific_force_m_s2"])
        errs.append(err)
        mv = {"from_hold": len(moves) + 1, "t0_s": float(t[s] - t[0]), "t1_s": float(t[e - 1] - t[0]),
              "integrated_deg": np.degrees(ang), "label": f"{'+' if sgn > 0 else '-'}{'XYZ'[dom]}{abs(np.degrees(ang[dom])):.0f}",
              "dominant_axis": "XYZ"[dom], "signed_angle_deg": float(np.degrees(ang[dom])),
              "axis_purity": float(abs(ang[dom]) / max(np.linalg.norm(ang), 1e-12)),
              "gravity_prediction_error_deg": float(err[ident]),
              "gravity_prediction_error_raw_accel_deg": float(raw),
              "q_rel_gyro_wxyz": qrel[ident]}
        moves.append(mv)
    E = np.stack(errs, 1)                        # (H, moves)
    rms = np.sqrt(np.mean(E ** 2, 1))
    best = int(np.argmin(rms))
    Pb, sb, lb = hyps[best]
    raw_rms = float(np.sqrt(np.mean([m["gravity_prediction_error_raw_accel_deg"] ** 2 for m in moves])))
    consistency = {"identity_rms_error_deg": float(rms[ident]), "best_rms_error_deg": float(rms[best]),
                   "identity_rms_error_raw_accel_deg": raw_rms,
                   "accel_corrected_with_six_position": six["used_to_correct_consistency_check"],
                   "best_gyro_to_accel_matrix": Pb, "best_scale": lb,
                   "identity_is_best": bool(best == ident or rms[ident] <= rms[best] + 0.5)}
    if not consistency["identity_is_best"] and rms[ident] > 3.0:
        warnings.append(f"gyro and accel frames disagree: gyro' = {Pb.astype(int).tolist()} x gyro ({lb}) "
                        f"explains the gravity changes with {rms[best]:.1f} deg rms vs {rms[ident]:.1f} deg as published")

    checks = {}
    ef = [x.strip().upper() for x in args.expect_faces.split(",")] if args.expect_faces else None
    em = [_parse_move(x) for x in args.expect_moves.split(",")] if args.expect_moves else None
    if ef is not None:
        if len(ef) == len(holds):
            checks["faces"] = [{"hold": i + 1, "expected": x, "seen": h["face"], "ok": x == h["face"]}
                               for i, (x, h) in enumerate(zip(ef, holds))]
        else:
            checks["faces_note"] = f"{len(holds)} holds found, {len(ef)} expected: compare the table by hand"
    if em is not None:
        if len(em) == len(moves):
            res = []
            for i, ((sg, ax, deg), mv) in enumerate(zip(em, moves)):
                ok = (mv["dominant_axis"] == ax and np.sign(mv["signed_angle_deg"]) == sg
                      and abs(abs(mv["signed_angle_deg"]) - deg) < args.angle_tol_deg and mv["axis_purity"] > 0.9)
                res.append({"move": i + 1, "expected": f"{'+' if sg > 0 else '-'}{ax}{deg:g}", "seen": mv["label"],
                            "ok": bool(ok)})
            checks["moves"] = res
        else:
            checks["moves_note"] = f"{len(moves)} moves found, {len(em)} expected: compare the table by hand"
    if checks.get("faces") and not checks["faces"][0]["ok"]:
        warnings.append(f"first hold reads {holds[0]['face']} but the protocol starts Z-up (+Z): gravity sign or "
                        "axis labelling differs from REP-145")

    qres = None
    if Q is not None and not imu.get("orientation_unavailable_frac"):
        qres = {}
        for name, (order, conj) in QUAT_HYPOTHESES.items():
            qw = _as_wxyz(Q, order, conj)
            tilt, rot = [], []
            hq = [_mean_quat_wxyz(qw[h["start"]:h["stop"]]) for h in holds]
            for h, q in zip(holds, hq):
                tilt.append(float(_angle_deg(h["specific_force_m_s2"], _rot_inv(q, [0.0, 0.0, 1.0]))))
            for i, mv in enumerate(moves):
                q_meas = _qmul(np.array([hq[i][0], *(-hq[i][1:])]), hq[i + 1])
                dq = _qmul(np.array([mv["q_rel_gyro_wxyz"][0], *(-np.asarray(mv["q_rel_gyro_wxyz"][1:]))]), q_meas)
                rot.append(float(np.degrees(2 * np.arctan2(np.linalg.norm(dq[1:]), abs(dq[0])))))
            qres[name] = {"hold_tilt_error_deg_median": float(np.median(tilt)),
                          "move_rotation_mismatch_deg_median": float(np.median(rot)) if rot else None}
        score = {k: v["hold_tilt_error_deg_median"] + (v["move_rotation_mismatch_deg_median"] or 0) for k, v in qres.items()}
        qres = {"by_interpretation": qres, "best": min(score, key=score.get)}
        if not qres["best"].startswith("xyzw fields, body->world"):
            warnings.append(f"quaternion matches '{qres['best']}', not the body->world xyzw contract of IMU_INPUT.md")
    for mv in moves:
        mv["q_rel_gyro_wxyz"] = np.asarray(mv["q_rel_gyro_wxyz"]).round(6)
    return {"tool": f"imu_allan.py {__version__}", "input": data["source"], "time_source": tsrc, "rate_hz": fs,
            "frame_ids": imu.get("frame_ids"), "holds": holds, "moves": moves,
            "gyro_accel_consistency": consistency, "six_position": six, "expected_sequence": checks,
            "quaternion": qres, "warnings": warnings}


# ----------------------------------------------------------------------------- tap
def tap_envelope(A, fs):
    """Energy envelope of the high-passed accelerometer: sqrt(15 ms moving mean of |a - 0.1 s mean|^2)."""
    hp = A - _moving_mean(A, _odd(max(3, round(0.1 * fs))))
    e2 = np.sum(hp * hp, axis=1)
    return np.sqrt(_moving_mean(e2, _odd(max(1, round(0.015 * fs)))))


def detect_taps(t, e, k=8.0, refractory_s=0.3):
    med = float(np.median(e))
    mad = 1.4826 * float(np.median(np.abs(e - med))) + 1e-12
    thr = med + k * mad
    ev, last = [], -np.inf
    for s, _ in _runs(e > thr):
        if t[s] - last < refractory_s:
            continue
        if s > 0 and e[s] > e[s - 1]:
            fr = (thr - e[s - 1]) / (e[s] - e[s - 1])
            ev.append(float(t[s - 1] + fr * (t[s] - t[s - 1])))
        else:
            ev.append(float(t[s]))
        last = t[s]
    return np.asarray(ev), thr


def xcorr_offset(t1, e1, t2, e2, c1, c2, half=0.25, grid=2.5e-4, max_lag=0.12):
    """Lag (s) of signal 1 relative to signal 2 around a paired event (positive: 1 later)."""
    tg = np.arange(c2 - half, c2 + half, grid)
    s2 = np.interp(tg, t2, e2)
    s2 = s2 - s2.mean()
    base = c1 - c2
    lags = base + np.arange(-max_lag, max_lag + grid / 2, grid)
    best, bl = -np.inf, 0.0
    for lag in lags:
        s1 = np.interp(tg + lag, t1, e1)
        s1 = s1 - s1.mean()
        den = np.linalg.norm(s1) * np.linalg.norm(s2)
        c = float(s1 @ s2 / den) if den > 0 else -np.inf
        if c > best:
            best, bl = c, lag
    return float(bl), best


def analyse_tap(args) -> dict:
    warnings = []
    main = load(args.input, args, aux=False)
    t1, src1 = choose_time(main["imu"], args.time_source, warnings)
    fs1 = (t1.size - 1) / (t1[-1] - t1[0])
    e1 = tap_envelope(main["imu"]["accel"], fs1)
    on1, thr1 = detect_taps(t1, e1, args.k_mad, args.refractory_s)
    out = {"tool": f"imu_allan.py {__version__}", "input": main["source"], "time_source": src1,
           "main": {"topic": main["selected"].get("imu"), "rate_hz": fs1, "taps": int(on1.size),
                    "stamp_latency": timing_stats(t1, main["imu"]["t_recv"] if src1 == "header" else None)
                    .get("recv_minus_stamp_ms")}}
    ref_t = ref_e = None
    if args.ref_topic or args.ref_csv:
        if args.ref_topic:
            ref = read_bag(args.input, imu_topic=args.ref_topic, aux=False)
        else:
            ref = read_csv(args.ref_csv, args.gyro_unit, args.accel_unit)
        ref_t, src2 = choose_time(ref["imu"], args.time_source, warnings)
        fs2 = (ref_t.size - 1) / (ref_t[-1] - ref_t[0])
        ref_e = tap_envelope(ref["imu"]["accel"], fs2)
        on2, _ = detect_taps(ref_t, ref_e, args.k_mad, args.refractory_s)
        out["reference"] = {"source": ref["source"], "topic": ref["selected"].get("imu"), "rate_hz": fs2,
                            "taps": int(on2.size), "stamp_latency": timing_stats(
                                ref_t, ref["imu"]["t_recv"] if src2 == "header" else None).get("recv_minus_stamp_ms")}
        method = "cross-correlation of high-passed |accel| envelopes around each paired tap"
    elif args.ref_events:
        ev = np.loadtxt(args.ref_events, delimiter=",", skiprows=1, ndmin=2)[:, 0]
        on2 = np.sort(ev)
        out["reference"] = {"source": f"events:{args.ref_events}", "taps": int(on2.size)}
        method = "tap onset (threshold crossing) minus reference event time"
    else:
        raise SystemExit("tap needs --ref-topic, --ref-csv or --ref-events")
    pairs = []
    for ta in on1:
        if on2.size == 0:
            break
        j = int(np.argmin(np.abs(on2 - ta)))
        if abs(on2[j] - ta) > args.max_offset_s:
            continue
        p = {"t_main_s": float(ta - t1[0]), "onset_offset_ms": float((ta - on2[j]) * 1e3)}
        if ref_t is not None:
            lag, c = xcorr_offset(t1, e1, ref_t, ref_e, ta, on2[j], max_lag=args.max_offset_s / 2)
            p.update(xcorr_offset_ms=lag * 1e3, xcorr_peak=c)
        pairs.append(p)
    key = "xcorr_offset_ms" if ref_t is not None else "onset_offset_ms"
    vals = np.asarray([p[key] for p in pairs])

    def stats(v):
        if v.size == 0:
            return {"n": 0, "median": None}
        return {"n": int(v.size), "median": float(np.median(v)), "mean": float(v.mean()),
                "std": float(v.std(ddof=1)) if v.size > 1 else None, "min": float(v.min()), "max": float(v.max())}
    out.update(method=method, pairs=pairs, offset_ms=stats(vals),
               onset_offset_ms=stats(np.asarray([p["onset_offset_ms"] for p in pairs])),
               interpretation="positive = this IMU's stamps lag the reference for the same physical tap; "
                              "IMU stamp latency = offset + the reference's own latency (+ consumer processing "
                              "for end-to-end)",
               warnings=warnings)
    rl = getattr(args, "ref_latency_ms", None)
    med = out["offset_ms"]["median"]
    out["ref_latency_ms"] = rl
    out["latency_ms_estimate"] = (med + rl) if (rl is not None and med is not None) else None
    if rl is None:
        warnings.append("--ref-latency-ms not given: only the relative offset is known, so stationary "
                        "--tap-report leaves imu_noise.delay_steps at 0")
    if len(pairs) < 5:
        warnings.append(f"only {len(pairs)} paired taps: give >= 10 sharp taps ~2 s apart")
    return out


# ----------------------------------------------------------------------------- printing / CLI
def _fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}g}"


def print_stationary(measured, report):
    tm = report["timing"]
    print(f"\n== {report['input']}  (time source: {report['time_source']})")
    print(f"samples {tm['samples']}, {tm['duration_s'] / 3600:.3f} h, delivered rate {tm['rate_hz']:.3f} Hz "
          f"(median-dt rate {_fmt(tm['median_dt_rate_hz'], 6)} Hz), Allan rate {report['allan_rate_hz']:.3f} Hz")
    dtm = tm["dt_ms"]
    print(f"dt ms: median {dtm['median']:.3f} std {dtm['std']:.3f} p01 {dtm['p01']:.3f} p99 {dtm['p99']:.3f} "
          f"max {dtm['max']:.1f}; gaps {tm['gaps']}, missing ~{tm['missing_samples_est']} ({tm['missing_frac']:.3%}); "
          f"duplicates {report['duplicates']['full_duplicates']} ({report['duplicates']['fraction']:.3%})")
    if "recv_minus_stamp_ms" in tm:
        r = tm["recv_minus_stamp_ms"]
        print(f"bag receive - header stamp: median {r['median']:.2f} ms, p95 {r['p95']:.2f} ms, max {r['max']:.1f} ms")
    print(f"{'':6}{'N':>14}{'B':>12}{'tau_B':>8}{'K':>12}{'K<=':>11}{'mean':>12}{'hp std':>10}")
    for s, unit, sc, bu in (("gyro", "deg/s", 1 / DEG, 3600 / DEG), ("accel", "mg", 1 / MG, 1 / MG)):
        for a in "xyz":
            f = report["allan"][s][a]
            mean = (report["gyro_mean_dps"] if s == "gyro" else np.asarray(report["accel_mean_m_s2"]) / MG)["xyz".index(a)]
            print(f"{s[0]}{a:<5}{f['N'] * sc:>14.5g}{f['B'] * bu:>12.4g}{f['tau_B_s']:>8.0f}"
                  f"{_fmt(f['K'] * sc if f['K'] else None):>12}{f['K_upper'] * sc:>11.3g}{mean:>12.5g}"
                  f"{f['sample_std'] * sc:>10.4g}")
    print("units: gyro N (deg/s)/sqrt(Hz), B deg/h, K (deg/s)/sqrt(s), mean deg/s; accel N mg/sqrt(Hz), B mg, "
          "K mg/sqrt(s), mean mg (incl. gravity)")
    g = report["gravity"]
    print(f"gravity |f| = {g['magnitude_m_s2']:.5f} m/s^2 vs local {g['local_g_m_s2']:.5f} "
          f"({g['magnitude_minus_local_g_mg']:+.2f} mg), up axis {g['up_axis']}, tilt {g['tilt_from_up_axis_deg']:.2f} deg")
    if report["magnetometer"]:
        mm = report["magnetometer"]
        print(f"mag |B| = {mm['field_magnitude_ut']['mean']:.2f} +- {mm['field_magnitude_ut']['std']:.2f} uT, "
              f"noise {np.round(mm['noise_hp_ut'], 4).tolist()} uT, step {_fmt(mm['step_mgauss'])} mGauss, "
              f"updates {_fmt(mm['distinct_update_rate_hz'])} Hz")
    if report["barometer"]:
        b = report["barometer"]
        print(f"baro noise {b['noise_hp_pa']:.3f} Pa (hp), mean {b['mean_pa']:.1f} Pa, updates "
              f"{_fmt(b['distinct_update_rate_hz'])} Hz")
    if report["quaternion"]:
        print(f"quaternion: {json.dumps(_jsonable(report['quaternion']))[:300]}")
    print("\nimu_measured.json headline:")
    for k in IM10A_DATASHEET:
        print(f"  {k:24s} {measured[k]!r:>28}   [{measured['provenance'][k][:70]}]")
    print(f"  imu_noise @ {measured['imu_noise_rate_hz']:g} Hz: {json.dumps(_jsonable(measured['imu_noise']))}")
    for w in report["warnings"]:
        print(f"WARNING: {w}")


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser (also used by tests to get default analysis arguments)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Conventions: see the module docstring (python imu_allan.py --doc).")
    ap.add_argument("--doc", action="store_true", help="print the conventions/units docstring and exit")
    sub = ap.add_subparsers(dest="cmd")

    def common(p, topics=True):
        p.add_argument("input", help="rosbag2 directory (or its .mcap/.db3), ROS1 .bag, or CSV")
        p.add_argument("--imu-topic", default=None, help="sensor_msgs/Imu topic (auto if exactly one)")
        if topics:
            p.add_argument("--mag-topic", default=None)
            p.add_argument("--baro-topic", default=None)
            p.add_argument("--temp-topic", default=None)
        p.add_argument("--time-source", choices=("auto", "header", "receive"), default="auto")
        p.add_argument("--gyro-unit", choices=("rad/s", "deg/s"), default="rad/s", help="CSV only")
        p.add_argument("--accel-unit", choices=("m/s2", "g"), default="m/s2", help="CSV only")
        p.add_argument("--mag-unit", choices=("T", "uT", "G", "mG"), default="T", help="CSV only")
        p.add_argument("--baro-unit", choices=("Pa", "hPa"), default="Pa", help="CSV only")
        p.add_argument("--out-dir", default=".", help="where outputs are written")
        p.add_argument("--label", default=None)
        p.add_argument("--local-g", type=float, default=None, help="local gravity, m/s^2")
        p.add_argument("--latitude-deg", type=float, default=None, help="for WGS84 normal gravity")
        p.add_argument("--height-m", type=float, default=0.0)

    s = sub.add_parser("stationary", help="Allan analysis of a still record")
    common(s)
    s.add_argument("--trim-start-s", type=float, default=0.0, help="drop warm-up / handling at the start")
    s.add_argument("--trim-end-s", type=float, default=0.0, help="drop handling at the end")
    s.add_argument("--nominal-rate", type=float, default=None, help="configured output rate, Hz")
    s.add_argument("--max-tau-frac", type=float, default=0.2, help="largest tau used for fits, fraction of T")
    s.add_argument("--points-per-decade", type=int, default=20)
    s.add_argument("--drop-duplicates", action="store_true", help="remove exact repeated samples first")
    s.add_argument("--rotations-report", default=None, help="rotations_report.json -> six-position accel bias")
    s.add_argument("--tap-report", default=None, help="tap_report.json -> latency")
    s.add_argument("--sim-rate", type=float, default=200.0, help="call rate for the imu_noise block (Hz)")
    s.add_argument("--mag-ref-ut", type=float, default=None, help="expected local field magnitude, uT")
    s.add_argument("--no-plot", action="store_true")

    r = sub.add_parser("rotations", help="axis/sign check and six-position accelerometer bias")
    common(r, topics=False)
    r.add_argument("--static-gyro-std", type=float, default=0.01, help="rad/s, 0.5 s moving std")
    r.add_argument("--static-accel-std", type=float, default=0.05, help="m/s^2, 0.5 s moving std")
    r.add_argument("--static-rate", type=float, default=0.05, help="rad/s, 0.5 s moving |mean|")
    r.add_argument("--min-hold-s", type=float, default=3.0)
    r.add_argument("--max-face-tilt-deg", type=float, default=15.0)
    r.add_argument("--angle-tol-deg", type=float, default=15.0)
    r.add_argument("--expect-faces", default=DEFAULT_FACES, help="'' to skip")
    r.add_argument("--expect-moves", default=DEFAULT_MOVES, help="'' to skip")

    tp = sub.add_parser("tap", help="time offset against a reference sensor")
    common(tp, topics=False)
    tp.add_argument("--ref-topic", default=None, help="reference sensor_msgs/Imu topic in the same bag")
    tp.add_argument("--ref-csv", default=None, help="reference IMU as CSV (same format)")
    tp.add_argument("--ref-events", default=None, help="CSV with header and a first column of event times (s)")
    tp.add_argument("--k-mad", type=float, default=8.0, help="detection threshold in robust sigmas")
    tp.add_argument("--refractory-s", type=float, default=0.3)
    tp.add_argument("--max-offset-s", type=float, default=0.25)
    tp.add_argument("--ref-latency-ms", type=float, default=None,
                    help="the reference sensor's own stamp latency in ms, if known from elsewhere; makes "
                         "latency_ms_estimate (and imu_noise.delay_steps) absolute")

    ex = sub.add_parser("export-csv", help="write the IMU topic of a bag as CSV (mag/baro are not exported)")
    common(ex)
    ex.add_argument("--output", required=True)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.doc:
        print(__doc__)
        return 0
    if not args.cmd:
        ap.print_help()
        return 2
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "stationary":
        data = load(args.input, args)
        measured, report, pdata = analyse_stationary(data, args)
        _write_json(out / "imu_measured.json", measured)
        _write_json(out / "imu_allan_report.json", report)
        if not args.no_plot and plot_stationary(out / "imu_allan.png", pdata):
            _log(f"wrote {out / 'imu_allan.png'}")
        print_stationary(measured, report)
        _log(f"wrote {out / 'imu_measured.json'} and {out / 'imu_allan_report.json'}")
    elif args.cmd == "rotations":
        data = load(args.input, args, aux=False)
        rep = analyse_rotations(data, args)
        _write_json(out / "rotations_report.json", rep)
        print(f"\n== {rep['input']}: {len(rep['holds'])} holds, {len(rep['moves'])} moves at {rep['rate_hz']:.1f} Hz")
        for i, h in enumerate(rep["holds"]):
            f = h["specific_force_m_s2"]
            print(f"hold {i + 1:2d} {h['t0_s']:7.1f}-{h['t1_s']:7.1f} s  {h['face']}  tilt {h['tilt_deg']:4.1f} deg  "
                  f"f = [{f[0]:+7.3f} {f[1]:+7.3f} {f[2]:+7.3f}] m/s^2")
        for m in rep["moves"]:
            d = m["integrated_deg"]
            print(f"move {m['from_hold']:2d}  {m['label']:>6}  integrated [{d[0]:+7.1f} {d[1]:+7.1f} {d[2]:+7.1f}] deg  "
                  f"purity {m['axis_purity']:.2f}  gravity-prediction error {m['gravity_prediction_error_deg']:.2f} deg")
        for k in ("faces", "moves"):
            if k in rep["expected_sequence"]:
                bad = [x for x in rep["expected_sequence"][k] if not x["ok"]]
                print(f"expected {k}: {'ALL PASS' if not bad else 'FAIL at ' + str(bad)}")
            elif f"{k}_note" in rep["expected_sequence"]:
                print(rep["expected_sequence"][f"{k}_note"])
        c = rep["gyro_accel_consistency"]
        print(f"gyro-vs-accel: identity {c['identity_rms_error_deg']:.2f} deg rms (six-position-corrected accel; "
              f"{c['identity_rms_error_raw_accel_deg']:.2f} with raw accel); best {c['best_rms_error_deg']:.2f} "
              f"deg with {np.asarray(c['best_gyro_to_accel_matrix']).astype(int).tolist()} ({c['best_scale']})")
        sp = rep["six_position"]
        print(f"six-position: faces {sp['faces_seen']}, bias {np.round(sp['bias_mg'], 2).tolist()} mg, "
              f"|bias| {_fmt(sp['bias_norm_mg'])} mg, "
              f"scale error % {[round(v['scale_error_pct'], 3) for v in sp['axes'].values()]}")
        if rep["quaternion"]:
            print(f"quaternion best interpretation: {rep['quaternion']['best']}")
        for w in rep["warnings"]:
            print(f"WARNING: {w}")
        _log(f"wrote {out / 'rotations_report.json'}")
    elif args.cmd == "tap":
        rep = analyse_tap(args)
        _write_json(out / "tap_report.json", rep)
        o = rep["offset_ms"]
        print(f"\n== tap: {rep['main']['taps']} IMU taps, {rep['reference']['taps']} reference taps, {o['n']} pairs")
        print(f"offset (IMU - reference): median {_fmt(o['median'])} ms, mean {_fmt(o.get('mean'))} ms, "
              f"std {_fmt(o.get('std'))} ms  [{rep['method']}]")
        print(f"stamp latency estimate (offset + --ref-latency-ms): {_fmt(rep['latency_ms_estimate'])} ms")
        for w in rep["warnings"]:
            print(f"WARNING: {w}")
        _log(f"wrote {out / 'tap_report.json'}")
    elif args.cmd == "export-csv":
        data = load(args.input, args)
        imu = data["imu"]
        cols = [imu["t"], *imu["gyro"].T, *imu["accel"].T]
        names = ["t", "gx", "gy", "gz", "ax", "ay", "az"]
        if imu["quat"] is not None and not imu.get("orientation_unavailable_frac"):
            # orientation_covariance[0] == -1 marks "no orientation" (ROS): do not export a placeholder quaternion
            cols += list(imu["quat"].T)
            names += ["qx", "qy", "qz", "qw"]
        if imu.get("t_recv") is not None:
            cols.append(imu["t_recv"])
            names.append("t_recv")
        fmt = ["%.9f" if nm in ("t", "t_recv") else "%.17g" for nm in names]   # epoch seconds need ns digits
        np.savetxt(args.output, np.stack(cols, 1), delimiter=",", header=",".join(names), comments="", fmt=fmt)
        _log(f"wrote {args.output} ({imu['t'].size} IMU rows; mag/baro not exported)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
