"""Identified-model table: physics (MuJoCo biped, frozen gait dr-default-s0) vs the NavGym v2/v4 randomization box,
built only from the saved summaries in data/ (no simulation). Writes data/identified_model_table.json and prints it.

Physics sources: steps_summary.json (yaw steps), speed_summary.json (forward steps), bangbang_summary.json (square /
PRBS / PWM chatter), replay_summary.json (the actors' own gym command sequences replayed open-loop on the flat
floor), index_brake.json (flat-floor brake), trace_loops.json (read-only stats of the scored 50000-50011 traces),
physloop/armV4-s5_none/seed900{0,1,2}_log.npz (training-range closed loop).
Gym source: bhl_robust.navgym.env.sample_dynamics (w_gain 0.9-1.4, v_gain 0.85-1.05, tau 0.15-0.45 s on BOTH channels,
drift +-0.06 rad/s at V_MAX, latency 0-2 steps, w_noise 0-0.10, v_noise 0-0.04), V_MAX 0.35, W_MAX 1.0, DT 0.04."""
import json
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent / "data"
st = json.loads((D / "steps_summary.json").read_text())
ph, gapp = st["physics"], st["gym_apparent"]
sp = json.loads((D / "speed_summary.json").read_text())
bb = json.loads((D / "bangbang_summary.json").read_text())
rp = json.loads((D / "replay_summary.json").read_text())
br = json.loads((D / "index_brake.json").read_text())
tl = json.loads((D / "trace_loops.json").read_text())
fits = json.loads((D / "steps_fits.json").read_text())


def rng_of(keys, field, idx=(1, 2)):
    v = [ph[k][field][i] for k in keys for i in idx if ph[k][field] is not None]
    return [round(min(v), 3), round(max(v), 3)]


walk1 = ["vx0.15_wz1.0", "vx0.15_wz-1.0", "vx0.35_wz1.0", "vx0.35_wz-1.0"]
place1 = ["vx0.0_wz1.0", "vx0.0_wz-1.0"]
sq = bb["square"]


def sqm(f, k):
    v = [x[k] for x in sq if x["f_hz"] == f and x.get(k) is not None]
    return round(float(np.mean(v)), 3) if v else None


prbs = bb["prbs"]
pwm = bb["pwm"]


def pwm_rate(vx, m, kind):
    v = [x["w_mean"] for x in pwm if x["vx"] == vx and x["m"] == m and x["kind"] == kind]
    return round(float(np.mean(v)), 3) if v else None


# standing windows of the s5 training-maze physics closed loop (maze 9001, braked to standstill 65 % of steps)
z = np.load(D / "physloop/armV4-s5_none/seed9001_log.npz")
L = z["log"]; c = {k: L[:, i] for i, k in enumerate(list(z["cols"]))}
sel = c["t"] >= 1.0 - 1e-9
yaw = np.unwrap(c["yaw"]); i0 = int(np.flatnonzero(sel)[0])
yp = yaw[i0 + 1:] - yaw[i0]; v = c["out_vx"][sel][:len(yp)]; w = c["out_wz"][sel][:len(yp)]
n = 12; K = (len(yp) // n) * n
idx = [j for j in range(0, K - n, n) if (v[j:j + n] < 0.053).all()]
dp = np.array([yp[j + n - 1] - (yp[j - 1] if j > 0 else 0.0) for j in idx]); mc = np.array([w[j:j + n].mean() for j in idx])
stand = {"windows_0.48s": len(idx), "slope_net_rate_per_mean_cmd": round(float(np.polyfit(mc, dp / (n * 0.04), 1)[0]), 2),
         "mean_abs_net_rate_rad_s": round(float(np.mean(np.abs(dp / (n * 0.04)))), 3), "mean_abs_cmd_mean": round(float(np.mean(np.abs(mc))), 3)}


def mean_braked(arm):
    return round(float(np.mean([r["braked_frac"] for r in tl if r["arm"] == arm])), 3)


rows = [
    # ---------------- yaw channel
    {"ch": "yaw", "q": "steady-state gain at |u|=1 (walking 0.15-0.35 m/s)", "phys": rng_of(walk1, "K_ss"), "gym": [0.9, 1.4], "verdict": "inside"},
    {"ch": "yaw", "q": "steady-state gain at |u|=1 (in place)", "phys": rng_of(place1, "K_ss"), "gym": [0.9, 1.4], "verdict": "inside"},
    {"ch": "yaw", "q": "gain at |u|=0.6 (all speeds)", "phys": rng_of(["vx0.0_wz0.6", "vx0.15_wz0.6", "vx0.35_wz0.6"], "K_ss"), "gym": [0.9, 1.4], "verdict": "inside"},
    {"ch": "yaw", "q": "gain at |u|=0.3 walking (concave input map)", "phys": rng_of(["vx0.15_wz0.3", "vx0.35_wz0.3"], "K_ss"), "gym": [0.9, 1.4],
     "verdict": "outside (above at 0.15 m/s; gym is linear)"},
    {"ch": "yaw", "q": "gain at |u|=0.3 in place (dead zone)", "phys": rng_of(["vx0.0_wz0.3"], "K_ss"), "gym": [0.9, 1.4], "verdict": "outside (no turn)"},
    {"ch": "yaw", "q": "first-order tau, continuous FOPDT on the integrated step (s)", "phys": rng_of(walk1 + place1, "tau"),
     "gym": [gapp["K1.17_tau0.15_lat0"]["tau"], gapp["K1.17_tau0.45_lat2"]["tau"]], "gym_note": "gym fitted by the same procedure",
     "verdict": "outside (faster)"},
    {"ch": "yaw", "q": "tau in the gym's own discrete form, PRBS fit (s) / latency (steps)",
     "phys": [min(prbs[p]["gym_best"]["tau"] for p in prbs), max(prbs[p]["gym_best"]["tau"] for p in prbs)],
     "phys_latency": sorted({prbs[p]["gym_best"]["latency"] for p in prbs}), "gym": [0.15, 0.45], "gym_latency": [0, 2], "verdict": "outside (faster)"},
    {"ch": "yaw", "q": "dead time (s), continuous fit", "phys": rng_of(walk1 + place1, "L"), "gym": [0.0, 0.08], "verdict": "inside (lower edge: latency 0 only)"},
    {"ch": "yaw", "q": "FIR step response after 1 / 2 policy steps (PRBS p=0.02-0.7)",
     "phys": [[min(prbs[p]["fir_step"][0] for p in prbs), max(prbs[p]["fir_step"][0] for p in prbs)],
              [min(prbs[p]["fir_step"][1] for p in prbs), max(prbs[p]["fir_step"][1] for p in prbs)]],
     "gym_nominal": [prbs["0.7"]["gym_nominal_fir_step"][0], prbs["0.7"]["gym_nominal_fir_step"][1]], "verdict": "outside (faster)"},
    {"ch": "yaw", "q": "max rate at the interface limit |u|=1 (rad/s)", "phys": [round(min(abs(r["w_ss"]) for r in fits if abs(r["wz"]) == 1.0), 3), round(max(abs(r["w_ss"]) for r in fits if abs(r["wz"]) == 1.0), 3)],
     "phys_at_1.5": {f"vx{vx}": round(float(np.mean([r["w_ss"] for r in fits if r["wz"] == 1.5 and r["vx"] == vx])), 3) for vx in (0.0, 0.35)}, "gym": [0.9, 1.4],
     "note": "|u|=1.5 (not reachable through W_MAX=1): 1.50 in place, 1.31 at 0.35 m/s -> the gait is not rate-saturated at 1 rad/s",
     "verdict": "inside"},
    {"ch": "yaw", "q": "bang-bang square wave +-1 at 0.35 m/s: gain of the fundamental 0.5/1.5/3/6.25 Hz, every-step flip (12.5 Hz)",
     "phys": [sqm(f, "gain") for f in (0.5, 1.5, 3.0, 6.25, 12.5)],
     "gym": [[sqm(f, "gym_slow_gain") for f in (0.5, 1.5, 3.0, 6.25, 12.5)], [sqm(f, "gym_fast_gain") for f in (0.5, 1.5, 3.0, 6.25, 12.5)]],
     "gym_note": "slow corner (1.17, 0.45 s, 2 steps) and fast corner (1.17, 0.15 s, 0 steps)", "verdict": "outside (4-6x bandwidth)"},
    {"ch": "yaw", "q": "square-wave phase lag at 0.5/1.5/3/6.25 Hz (deg)", "phys": [sqm(f, "phase_deg") for f in (0.5, 1.5, 3.0, 6.25)],
     "gym": [[sqm(f, "gym_fast_phase_deg") for f in (0.5, 1.5, 3.0, 6.25)], [sqm(f, "gym_slow_phase_deg") for f in (0.5, 1.5, 3.0, 6.25)]],
     "verdict": "outside (less lag)"},
    {"ch": "yaw", "q": "mean of a +-1 chatter while walking (0.3 m/s): realized rate for duty-cycle mean m=0.33/0.5/0.75, iid chatter",
     "phys": [pwm_rate(0.3, m, "chat") for m in (0.333, 0.5, 0.75)], "phys_const": [pwm_rate(0.3, m, "const") for m in (0.333, 0.5, 0.75)],
     "gym": "linear: w_gain * mean", "verdict": "inside (mean survives, 0.8-1.1x of const)"},
    {"ch": "yaw", "q": "in place: realized rate for const / fastest-PWM / iid chatter of mean 0.33, 0.5, 0.75",
     "phys": {"const": [pwm_rate(0.0, m, "const") for m in (0.333, 0.5, 0.75)], "pwm": [pwm_rate(0.0, m, "pwm") for m in (0.333, 0.5, 0.75)],
              "chat": [pwm_rate(0.0, m, "chat") for m in (0.333, 0.5, 0.75)]},
     "gym": "linear", "verdict": "outside (dead zone; PWM of mean <= 0.5 does not turn)"},
    {"ch": "yaw", "q": "braked to standstill in a maze (s5, training maze 9001, physics closed loop): 0.48 s windows", "phys": stand,
     "gym": "no brake, no standstill mode", "verdict": "outside"},
    {"ch": "yaw", "q": "per-step heading change under the actors' own chatter (deg, median / p90)",
     "phys": [[min(x["per_step_abs_dyaw_deg"]["median"] for x in rp), max(x["per_step_abs_dyaw_deg"]["median"] for x in rp)],
              [min(x["per_step_abs_dyaw_deg"]["p90"] for x in rp), max(x["per_step_abs_dyaw_deg"]["p90"] for x in rp)]],
     "gym": "s5 0.32 / 2.00, s6 0.77 / 2.14 (gym_actor_*_maze9000-9005)", "verdict": "outside (1.3-3x more jitter)"},
    {"ch": "yaw", "q": "stride wobble: yaw-rate RMS (rad/s) / heading p-p (deg) / freq (Hz) at 0.35 m/s",
     "phys": [sp["0.35"]["yawrate_rms"][0], sp["0.35"]["wobble_yaw_pp_deg"][0], sp["0.35"]["stride_peak_hz"][0]],
     "gym": "white w_noise 0-0.10 rad/s", "verdict": "outside (periodic, larger RMS; heading effect only +-2 deg)"},
    {"ch": "yaw", "q": "heading drift walking straight (rad/s) at 0.15 / 0.25 / 0.35 m/s",
     "phys": [sp[k]["drift_rad_s"][0] for k in ("0.15", "0.25", "0.35")], "phys_deg_per_m": [sp[k]["drift_deg_per_m"][0] for k in ("0.15", "0.25", "0.35")],
     "gym": "+-0.06 * v/V_MAX (+-0.026 at 0.15, +-0.06 at 0.35)", "verdict": "inside"},
    # ---------------- forward channel
    {"ch": "fwd", "q": "steady-state gain at 0.15 / 0.25 / 0.35 m/s", "phys": [sp[k]["gain_ss"][0] for k in ("0.15", "0.25", "0.35")],
     "map": "v = max(0, 1.19 v_cmd - 0.063): no motion below 0.053 m/s", "gym": [0.85, 1.05], "verdict": "inside at 0.25-0.35; outside at 0.15 and below 0.053 (dead zone)"},
    {"ch": "fwd", "q": "start tau (s) from standstill at 0.25 / 0.35 m/s", "phys": [sp["0.25"]["tau_v"], sp["0.35"]["tau_v"]], "gym": [0.15, 0.45],
     "verdict": "inside (lower edge)"},
    {"ch": "fwd", "q": "start from standstill at 0.15 m/s", "phys": {"tau_fit_s": sp["0.15"]["tau_v"][0], "gain_ss": sp["0.15"]["gain_ss"][0]},
     "gym": [0.15, 0.45], "verdict": "outside (slower, ~1.5 s to start stepping)"},
    {"ch": "fwd", "q": "stop tau (s) from 0.25-0.35 m/s", "phys": [min(sp["0.25"]["stop_tau"][1], sp["0.35"]["stop_tau"][1]), max(sp["0.25"]["stop_tau"][2], sp["0.35"]["stop_tau"][2])],
     "gym": [0.15, 0.45], "verdict": "outside (faster; gym lags the stop like the start)"},
    {"ch": "fwd", "q": "dead time (s)", "phys": [min(sp[k]["L_v"][1] for k in ("0.25", "0.35")), max(sp[k]["L_v"][2] for k in ("0.25", "0.35"))],
     "gym": [0.0, 0.08], "verdict": "inside"},
    {"ch": "fwd", "q": "max speed at v_cmd = V_MAX (m/s)", "phys": sp["0.35"]["v_ss"][1:], "gym": [0.2975, 0.3675], "verdict": "inside"},
    # ---------------- coupling
    {"ch": "coupling", "q": "speed ratio during a constant turn at 0.35 m/s, |u|=1 / |u|=1.5", "phys": [ph["vx0.35_wz1.0"]["v_ratio"][0], ph["vx0.35_wz-1.0"]["v_ratio"][0], ph["vx0.35_wz1.5"]["v_ratio"][0]],
     "gym": "none", "verdict": "inside (<= 3 % at |u|<=1)"},
    {"ch": "coupling", "q": "speed ratio under +-1 chatter: PRBS p=0.5/0.7 (0.3 m/s) / 3 Hz square / the actors' own sequences",
     "phys": {"prbs0.5": prbs["0.5"]["v_ratio"], "prbs0.7": prbs["0.7"]["v_ratio"], "square3Hz": sqm(3.0, "v_ratio"),
              "actor_replays_phys_over_cmd": [round(x["speed_phys_mean"] / x["speed_cmd_mean"], 3) for x in rp]},
     "gym": "none", "verdict": "outside (5-10 % loss under PRBS, 26 % at 3 Hz; 1-3 % on the actors' own sequences)"},
    {"ch": "coupling", "q": "lateral sway: lateral-speed RMS (m/s) at 0.35 m/s", "phys": sp["0.35"]["lat_speed_rms"][0], "gym": "none (unicycle)",
     "verdict": "outside (about +-3 cm of sway)"},
    # ---------------- brake (deployment, not gait)
    {"ch": "brake", "q": "team_sensors speed brake: v *= clip((clear - 0.42)/0.48, 0, 1), +-25 deg forward cone, 10 Hz packets (+ depth term)",
     "phys": {"maze_braked_frac_s5": mean_braked("armV4-s5"), "maze_braked_frac_s6": mean_braked("armV4-s6"), "maze_braked_frac_astar": mean_braked("astar"),
              "flat_floor_false_brake_straight_0.35": round(br[2]["sensor_stats"]["braked_robot_steps"] / br[2]["sensor_stats"]["sampled_robot_steps"], 4),
              "flat_floor_false_brake_prbs0.7_0.35": round(br[4]["sensor_stats"]["braked_robot_steps"] / br[4]["sensor_stats"]["sampled_robot_steps"], 4)},
     "gym": "absent (collision ends the episode at 0.22 m)", "verdict": "outside (absent from the gym)"},
    {"ch": "obs", "q": "lidar/map pipeline: 10 Hz packets held 2-3 steps; biped --policy path integrates each packet at the NEXT step's pose",
     "phys": "maze_explore.py run_seed (biped branch, learned['emap'].update at the loop top)", "gym": "fresh scan each step at the current pose",
     "verdict": "outside (pipeline)"},
]
out = {"source": __doc__.split("\n")[0], "rows": rows}
(D / "identified_model_table.json").write_text(json.dumps(out, indent=1, default=float))
for r in rows:
    print(f"[{r['ch']}] {r['q']}: phys={r['phys']} | gym={r.get('gym')} -> {r['verdict']}")
