"""Per-checkpoint push fall rate vs measurable quantities (push CSV, MuJoCo diagnose JSON, TB, checkpoint std)."""
import csv
import glob
import json
import os
import pickle

import numpy as np
import torch

torch.set_num_threads(1)
R = "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/results/repo-gpu-20260923/turn-20260927"
LOG = "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/external/Berkeley-Humanoid-Lite/logs/rsl_rl/humanoid"
HERE = os.path.dirname(os.path.abspath(__file__))
tb = pickle.load(open(os.path.join(HERE, "tb_scalars.pkl"), "rb"))

rows = []
for p in sorted(glob.glob(f"{R}/qualify/*__p0.5_n60.csv")):
    run = os.path.basename(p).split("__")[0]
    eps = list(csv.DictReader(open(p)))
    fell = np.array([e["fell"] == "True" for e in eps])
    by_cmd = {}
    for e in eps:
        k = (e["command_vx"], e["command_vy"], e["command_wz"])
        by_cmd.setdefault(k, []).append(e["fell"] == "True")
    surv = [e for e in eps if e["fell"] == "False"]
    h = np.mean([float(e["mean_height_m"]) for e in surv])
    tilt = np.mean([float(e["mean_tilt_rad"]) for e in surv])
    ylin = np.mean([float(e["lin_vel_err"]) for e in surv])
    yyaw = np.mean([float(e["yaw_rate_err"]) for e in surv])
    # diagnose JSON (MuJoCo, pure turn after 3 s settle, reset seeds 0-2)
    dj = None
    for cand in (f"{R}/diagnose/{run}.json",):
        if os.path.exists(cand):
            dj = json.load(open(cand))
    if dj is None:
        for cand in glob.glob(f"{R}/diagnose/mujoco_*.json"):
            d = json.load(open(cand))
            if run in d["runs"]:
                dj = d
    settle_lo = turn_lo = clear = None
    if dj is not None and run in dj["runs"]:
        rr = [r for r in dj["runs"][run] if "feet" in r]
        settle_lo = np.mean([(r["feet"]["left"]["liftoffs_last2s_settle"] + r["feet"]["right"]["liftoffs_last2s_settle"]) / 2 / 2.0 for r in rr])
        turn_lo = np.mean([(r["feet"]["left"]["liftoffs_turn"] + r["feet"]["right"]["liftoffs_turn"]) / 2 / 6.0 for r in rr])
        clear = np.mean([max(r["feet"]["left"]["peak_clearance_m_turn"], r["feet"]["right"]["peak_clearance_m_turn"]) for r in rr])
    # checkpoint std
    d = sorted(glob.glob(f"{LOG}/*_{run}"))[-1]
    last = sorted(glob.glob(d + "/model_*.pt"), key=lambda q: int(q.split("_")[-1][:-3]))[-1]
    s = torch.load(last, map_location="cpu", weights_only=False)["model_state_dict"]["std"]
    legstd = float(s[10:].mean())
    # TB at the end (last 100 its)
    t = tb[run]

    def endv(tag):
        a = np.array(t[tag])
        return float(a[-100:, 1].mean())
    eplen = endv("Train/mean_episode_length")
    rows.append(dict(run=run.replace("arms-turn-", ""), push=fell.mean(), n=len(eps),
                     push_turn=np.mean(by_cmd.get(("0.0", "0.0", "0.5"), [np.nan])),
                     height=h, tilt=tilt, linerr=ylin, yawerr=yyaw,
                     settle_steps_hz=settle_lo, turn_steps_hz=turn_lo, clear=clear, legstd=legstd,
                     tb_fall=endv("Episode_Termination/base_orientation"), tb_eplen=eplen,
                     tb_air_per_step=endv("Episode_Reward/feet_air_time") / (eplen / 500.0),
                     tb_arate_per_step=endv("Episode_Reward/action_rate_l2") / (eplen / 500.0),
                     tb_yaw_per_step=endv("Episode_Reward/track_ang_vel_z_exp") / (eplen / 500.0)))

keys = [k for k in rows[0] if k != "run"]
print("run".ljust(24) + "".join(k[:11].rjust(12) for k in keys))
for r in rows:
    print(r["run"][:23].ljust(24) + "".join((f"{r[k]:12.4f}" if r[k] is not None else "        None") for k in keys))
print("\nSpearman / Pearson correlation with push fall rate (n = %d checkpoints; NOT independent: one lineage)" % len(rows))
from scipy.stats import pearsonr, spearmanr
pf = np.array([r["push"] for r in rows])
for k in keys:
    if k in ("push", "n"):
        continue
    v = np.array([r[k] if r[k] is not None else np.nan for r in rows], dtype=float)
    m = np.isfinite(v)
    if m.sum() < 4:
        continue
    print(f"  {k:18s} spearman {spearmanr(pf[m], v[m])[0]:+.2f}  pearson {pearsonr(pf[m], v[m])[0]:+.2f}  (n={m.sum()})")
json.dump(rows, open(os.path.join(HERE, "push_corr.json"), "w"), indent=1, default=float)
