"""Final consolidation (no simulation): (1) the identified-model table with the follow-up corrections, (2) the gym
re-evaluation per arm x maze block x config, pooled over blocks, paired McNemar vs the default-gym 'base' run, and the
predeclared checks of data/predeclared_gymcheck_20261001.json.  Reads only saved files in data/.

Writes data/identified_model_table_final.json and data/gym_reeval_final.json."""
import glob, json, math
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent / "data"
FULL = "dyn+brake+obs10+h0+settle"


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


# ------------------------------------------------------------------ (1) table
tab = json.loads((D / "identified_model_table.json").read_text())
rows = tab["rows"]
sp = json.loads((D / "speed_summary.json").read_text())
fixes = {
    "yaw|first-order tau, continuous FOPDT on the integrated step (s)":
        {"note": "most walking fits sit at the 0.01 s grid floor (response complete within the first 40 ms step); range shown is "
                 "|u|=1 walking and in place; in-place |u|=0.6 is slower (0.26 s) and excluded; |u|=0.3 in place does not turn"},
    "yaw|bang-bang square wave +-1 at 0.35 m/s: gain of the fundamental 0.5/1.5/3/6.25 Hz, every-step flip (12.5 Hz)":
        {"q": "bang-bang square wave +-1 at 0.30 m/s: gain of the fundamental 0.5/1.5/3/6.25 Hz, every-step flip (12.5 Hz)",
         "note": "run at vx 0.30 (phys_experiments.block_square), not 0.35; the 1.5 Hz point (0.74) is non-monotonic -- likely "
                 "interference with the 1.71 Hz stride wobble (0.28 rad/s RMS) in a 10 s window; 3 seeds agree within 0.02. "
                 "-3 dB bandwidth: physics ~6 Hz (0.76 at 6.25 Hz relative to 1.02 at 0.5 Hz) vs the gym's first-order corner "
                 "1/(2 pi tau) = 0.35-1.06 Hz (0.64 nominal)",
         "verdict": "outside (bandwidth ~6 Hz vs 0.35-1.1 Hz: 3x the fast-corner gain at 6.25 Hz, 1.5x at every-step flips)"},
    "yaw|FIR step response after 1 / 2 policy steps (PRBS p=0.02-0.7)":
        {"gym_fast_corner (tau 0.15, latency 0), fraction of steady state": [0.27, 0.46],
         "note": "phys values are per unit command (steady-state gain 1.02-1.11), i.e. ~0.6 / ~1.0 of the steady-state rate after "
                 "1 / 2 steps; gym nominal (0.25 s, 1 step) 0 / 0.16 of steady state"},
    "yaw|per-step heading change under the actors' own chatter (deg, median / p90)":
        {"verdict": "outside (median 1.3-4x the gym's, p90 1.4x; max 5.9 deg per 40 ms step)"},
    "yaw|square-wave phase lag at 0.5/1.5/3/6.25 Hz (deg)":
        {"note": "gym lists are [fast corner (1.17, 0.15 s, 0 steps)], [slow corner (1.17, 0.45 s, 2 steps)]; 6.25 Hz slow-corner value is wrapped (= -222 deg)"},
    "fwd|start from standstill at 0.15 m/s":
        {"note": "not first-order: the FOPDT fit sits at its 3.0 s bound; the gait shuffles ~1.5-3.5 s before stepping"},
    "fwd|stop tau (s) from 0.25-0.35 m/s":
        {"q": "stop: travel after a stop command from 0.25 / 0.35 m/s (cm, signed, along heading, 2 s)",
         "phys": [-2.7, -1.1], "phys_range": [-4.1, -0.4],
         "gym": "v0 (tau - dt + latency dt): 3.3 (0.85, 0.15 s, 0) .. 8.8 nominal .. 18.0 cm (1.05, 0.45 s, 2)",
         "note": "the previous FOPDT stop fit (tau 0.01-0.03 s) is on stride-noisy per-step speed; the coast distance is the "
                 "robust number: the gait stops within one step (even steps back 1-4 cm)",
         "verdict": "outside (faster: stops in < 1 step; the gym lags the stop like the start)"},
    "brake|team_sensors speed brake: v *= clip((clear - 0.42)/0.48, 0, 1), +-25 deg forward cone, 10 Hz packets (+ depth term)":
        {"phys_step_weighted_scored": {"armV4-s5": 0.397, "armV4-s6": 0.248, "astar": 0.051},
         "note": "maze_braked_frac_* are per-episode means over the 12 scored traces (read-only); step-weighted: s5 0.397, s6 0.248, "
                 "A* 0.051; IMU term fired on 5 of 27685 s6 steps, 0 for s5/A*; the gym re-creation (sysid_env.py) omits the "
                 "stereo-depth term (unverified, likely minor: flat-floor false braking 0.6 % straight / 3 % under chatter)"},
}
for r in rows:
    k = f"{r['ch']}|{r['q']}"
    if k in fixes:
        r.update(fixes[k])
rows.insert(next(i for i, r in enumerate(rows) if r["q"].startswith("dead time (s)") and r["ch"] == "fwd"),
            {"ch": "fwd", "q": "start delay-equivalent (v_ss*3s - s(3s))/v_ss from standstill at 0.25 / 0.35 m/s (s)",
             "phys": [0.126, 0.105], "phys_range": [0.049, 0.188], "gym": "latency*dt + tau - dt: 0.11 .. 0.49 s",
             "verdict": "inside (fast edge)"})
rows.insert(0, {"ch": "yaw", "q": "FOPDT model (walking 0.15-0.35 m/s, |u| = 1)",
                "phys": {"K": [0.98, 1.13], "tau_continuous_s": "<= 0.01-0.10", "L_s": [0.0, 0.005],
                         "gym_form_fit (PRBS, MazeNavEnv.step equations)": {"w_gain": [1.02, 1.11], "tau_s": [0.06, 0.07], "latency_steps": 0},
                         "gym_form_fit (actors' own sequences replayed)": {"K": [1.02, 1.08], "tau_s": [0.07, 0.08], "latency_steps": 0}},
                "gym": {"w_gain": [0.9, 1.4], "tau_s": [0.15, 0.45], "latency_steps": [0, 2]},
                "verdict": "gain inside; tau outside (gym-form 0.06-0.08 s vs the gym's 0.15 s floor: 2-2.5x faster than its fastest "
                           "draw, 3-4x faster than nominal; ~60 % of the steady-state response inside the first 40 ms step vs 0 % nominal, 27 % fastest draw); latency at the gym's lower edge (0)"})
rows.insert(next(i for i, r in enumerate(rows) if r["ch"] == "fwd"),
            {"ch": "fwd", "q": "FOPDT model (start from standstill, 0.25-0.35 m/s)",
             "phys": {"K": [0.95, 1.01], "map": "v_ss = max(0, 1.19 v_cmd - 0.063)", "tau_s": [0.12, 0.17], "L_s": [0.0, 0.025],
                      "stop": "within one 40 ms step (coast -4..+0.7 cm)"},
             "gym": {"v_gain": [0.85, 1.05], "tau_s": [0.15, 0.45], "latency_steps": [0, 2], "stop": "same lag as the start"},
             "verdict": "start inside (fast edge); stop outside (faster); low-speed dead zone outside"})
(D / "identified_model_table_final.json").write_text(json.dumps({"source": tab["source"] + " + follow-up corrections (build_final.py)",
                                                                  "rows": rows}, indent=1, default=float))

# ------------------------------------------------------------------ (2) gym re-evaluation
runs = {}
meta = {}
for f in sorted(glob.glob(str(D / "gymcheck" / "*.json"))):
    d = json.loads(Path(f).read_text())
    s = d["summary"]
    key = (s["arm"], s["maze_seeds"][0])
    runs.setdefault(key, {})[s["config"]] = {e["maze_seed"]: e for e in d["episodes"]}
    meta[(s["arm"], s["maze_seeds"][0], s["config"])] = s
block_label = {9000: "9000-9023 (exploratory)", 9024: "9024-9047 (replicate)", 9048: "9048-9071 (fresh replicate)"}
out = {"per_block": {}, "pooled": {}, "predeclared": {}}
for (arm, start), cfgs in sorted(runs.items()):
    base = cfgs.get("base")
    blk = {}
    for c, eps in sorted(cfgs.items()):
        seeds = sorted(eps)
        oc = [eps[s]["outcome"] for s in seeds]
        row = {"goal": oc.count("goal"), "collision": oc.count("collision"), "time_out": oc.count("time_out"), "n": len(seeds),
               "mean_braked_frac": round(float(np.mean([eps[s]["braked_frac"] for s in seeds])), 3),
               "mean_flips_per_s": round(float(np.mean([eps[s]["flips_per_s"] for s in seeds])), 1)}
        if base is not None and c != "base":
            lost = sum(base[s]["outcome"] == "goal" and eps[s]["outcome"] != "goal" for s in seeds)
            won = sum(base[s]["outcome"] != "goal" and eps[s]["outcome"] == "goal" for s in seeds)
            row.update(lost_vs_base=lost, won_vs_base=won, mcnemar_p=round(mcnemar(lost, won), 3))
        blk[c] = row
    out["per_block"][f"{arm} {block_label.get(start, start)}"] = blk
for arm in ("armV4-s5", "armV4-s6"):
    blocks = sorted(s for (a, s) in runs if a == arm)
    common = set.intersection(*[set(runs[(arm, s)]) for s in blocks]) if blocks else set()
    for c in sorted(common):
        g = n = lost = won = 0
        for s in blocks:
            eps = runs[(arm, s)][c]; base = runs[(arm, s)].get("base")
            g += sum(e["outcome"] == "goal" for e in eps.values()); n += len(eps)
            if base is not None and c != "base":
                lost += sum(base[k]["outcome"] == "goal" and eps[k]["outcome"] != "goal" for k in eps)
                won += sum(base[k]["outcome"] != "goal" and eps[k]["outcome"] == "goal" for k in eps)
        row = {"blocks": [block_label.get(s, s) for s in blocks], "goal": g, "n": n}
        if c != "base":
            row.update(lost_vs_base=lost, won_vs_base=won, mcnemar_p=round(mcnemar(lost, won), 4))
        out["pooled"][f"{arm} {c}"] = row
    # replicate-only pooling (excludes the exploratory 9000 block)
    rep = [s for s in blocks if s != 9000]
    if rep:
        common_r = set.intersection(*[set(runs[(arm, s)]) for s in rep])
        for c in sorted(common_r):
            g = n = lost = won = 0
            for s in rep:
                eps = runs[(arm, s)][c]; base = runs[(arm, s)].get("base")
                g += sum(e["outcome"] == "goal" for e in eps.values()); n += len(eps)
                if base is not None and c != "base":
                    lost += sum(base[k]["outcome"] == "goal" and eps[k]["outcome"] != "goal" for k in eps)
                    won += sum(base[k]["outcome"] != "goal" and eps[k]["outcome"] == "goal" for k in eps)
            row = {"blocks": [block_label.get(s, s) for s in rep], "goal": g, "n": n}
            if c != "base":
                row.update(lost_vs_base=lost, won_vs_base=won, mcnemar_p=round(mcnemar(lost, won), 4))
            out["pooled"][f"{arm} {c} [replicates only]"] = row


def goals(arm, start, cfg):
    e = runs.get((arm, start), {}).get(cfg)
    return None if e is None else sum(x["outcome"] == "goal" for x in e.values())


def blk_row(arm, start, cfg):
    return out["per_block"].get(f"{arm} {block_label.get(start, start)}", {}).get(cfg)


pd = {}
p1 = []
for (arm, start) in sorted(runs):
    gb, gd = goals(arm, start, "base"), goals(arm, start, "dyn")
    if gb is not None and gd is not None:
        p1.append({"arm": arm, "block": start, "base": gb, "dyn": gd, "holds": abs(gd - gb) <= 2})
pd["P1_dynamics_alone_no_gap (|dyn-base| <= 2 per arm/block)"] = p1
r = blk_row("armV4-s5", 9048, FULL)
gb = goals("armV4-s5", 9048, "base")
if r is not None and gb is not None:
    pd["P2_s5_full_gap_on_9048 (full <= base-3; mostly time-outs; braked >= 0.25)"] = {
        "base": gb, "full": r["goal"], "collision": r["collision"], "time_out": r["time_out"], "mean_braked_frac": r["mean_braked_frac"],
        "holds": r["goal"] <= gb - 3 and r["time_out"] > r["collision"] and r["mean_braked_frac"] >= 0.25}
gf, gb = goals("armV4-s6", 9024, FULL), goals("armV4-s6", 9024, "base")
if gf is not None and gb is not None:
    pd["P3_s6_full_no_gap_on_9024 (full >= base-1)"] = {"base": gb, "full": gf, "holds": gf >= gb - 1}
gx, gf = goals("armV4-s6", 9000, FULL + "+fix"), goals("armV4-s6", 9000, FULL)
if gx is not None and gf is not None:
    pd["P4_s6_capture_fix_harmless_on_9000 (full+fix >= full-1)"] = {"full": gf, "full+fix": gx, "holds": gx >= gf - 1}
gx, gf = goals("armV4-s5", 9048, FULL + "+fix"), goals("armV4-s5", 9048, FULL)
if gx is not None and gf is not None:
    pd["P5_s5_capture_fix_helps_on_9048 (full+fix >= full+2; addendum_1)"] = {"full": gf, "full+fix": gx, "holds": gx >= gf + 2}
gx, gf = goals("armV4-s6", 9024, FULL + "+fix"), goals("armV4-s6", 9024, FULL)
if gx is not None and gf is not None:
    pd["P6_s6_capture_fix_harmless_on_9024 (full+fix >= full-1; addendum_2)"] = {"full": gf, "full+fix": gx, "holds": gx >= gf - 1}
# capture fix vs full pipeline, paired, pooled over every block where both ran (the fix is the R1 candidate)
fx = {}
for arm in ("armV4-s5", "armV4-s6"):
    lost = won = gF = gX = n = 0
    blocks = []
    for (a, s), cfgs in sorted(runs.items()):
        if a != arm or FULL not in cfgs or FULL + "+fix" not in cfgs:
            continue
        F, X = cfgs[FULL], cfgs[FULL + "+fix"]
        blocks.append(block_label.get(s, s))
        gF += sum(F[k]["outcome"] == "goal" for k in F); gX += sum(X[k]["outcome"] == "goal" for k in X); n += len(F)
        lost += sum(F[k]["outcome"] == "goal" and X[k]["outcome"] != "goal" for k in F)
        won += sum(F[k]["outcome"] != "goal" and X[k]["outcome"] == "goal" for k in F)
    if blocks:
        fx[arm] = {"blocks": blocks, "n": n, "full": gF, "full+fix": gX, "fix_lost": lost, "fix_won": won, "mcnemar_p": round(mcnemar(lost, won), 4)}
out["capture_fix_vs_full"] = fx
out["predeclared"] = pd
# physics closed loop (MuJoCo, maze_explore.run_seed), training-range mazes; gym outcomes of the same maze for comparison
phys = []
for f in sorted(glob.glob(str(D / "physloop" / "*" / "seed*.json"))):
    d = json.loads(Path(f).read_text())
    st = d["sensor_stats"]
    variant = Path(f).parent.name
    arm = variant.split("_")[0]
    gym = {c: runs.get((arm, 9000), {}).get(c, {}).get(d["seed"], {}).get("outcome") for c in ("base", FULL, FULL + "+fix", FULL + "+lpf0.15")}
    phys.append({"variant": variant, "seed": d["seed"], "outcome": d["outcome_class"], "completion_s": d["completion_s"],
                 "wall_contact_steps": d["wall_contact_steps"], "braked_frac": round(st["braked_robot_steps"] / st["sampled_robot_steps"], 3),
                 "path_m": d["path_length_m"], "capture_pose_substitution": d.get("capture_pose_substitution"), "gym_same_maze": gym})
out["physics_closed_loop_training_mazes"] = phys
(D / "gym_reeval_final.json").write_text(json.dumps(out, indent=1))
for r in phys:
    print("phys", r)

for k, blk in out["per_block"].items():
    print("==", k)
    for c, row in sorted(blk.items(), key=lambda kv: (kv[0] != "base", len(kv[0]), kv[0])):
        tail = f" | lost {row['lost_vs_base']} won {row['won_vs_base']} p={row['mcnemar_p']}" if "lost_vs_base" in row else ""
        print(f"   {c:40s} G{row['goal']:3d} C{row['collision']:3d} T{row['time_out']:3d}  braked {row['mean_braked_frac']:.3f} flips/s {row['mean_flips_per_s']:5.1f}{tail}")
print("== pooled")
for k, row in out["pooled"].items():
    print("  ", k, row)
print("== predeclared")
for k, v in pd.items():
    print("  ", k, v)
