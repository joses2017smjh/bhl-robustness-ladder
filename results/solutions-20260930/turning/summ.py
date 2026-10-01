"""Tabulate TensorBoard scalars of the turning runs at chosen iterations (windowed means)."""
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
data = pickle.load(open(os.path.join(HERE, "tb_scalars.pkl"), "rb"))

BASE = ["arms-dr1.0-s0"] + [f"arms-turn-{a}-s{s}" for a in ("turnhip", "turntrack", "turnboth", "turncmd") for s in (0, 1, 2)]
FT = [f"arms-turn-{a}-s{s}" for a in ("turnrest-ft", "turnboth-cont", "turnrestpush-ft") for s in (0, 1, 2)]

SHORT = {
    "Policy/mean_noise_std": "std",
    "Train/mean_episode_length": "eplen",
    "Episode_Termination/base_orientation": "fall",
    "Episode_Reward/feet_air_time": "air",
    "Episode_Reward/track_ang_vel_z_exp": "yawR",
    "Episode_Reward/track_lin_vel_xy_exp": "linR",
    "Metrics/base_velocity/error_vel_yaw": "eyaw",
    "Metrics/base_velocity/error_vel_xy": "exy",
    "Episode_Reward/action_rate_l2": "arate",
    "Episode_Reward/joint_deviation_hip": "dhip",
    "Episode_Reward/joint_deviation_ankle_roll": "dankr",
    "Episode_Reward/joint_deviation_shoulder": "dsho",
    "Episode_Reward/joint_deviation_elbow": "delb",
    "Episode_Reward/dof_pos_limits": "plim",
    "Episode_Reward/feet_slide": "slide",
    "Episode_Reward/termination_penalty": "termP",
    "Episode_Reward/dof_acc_l2": "acc",
    "Episode_Reward/ang_vel_xy_l2": "angxy",
    "Episode_Reward/flat_orientation_l2": "flat",
    "Train/mean_reward": "R",
    "Loss/entropy": "ent",
    "Loss/learning_rate": "lr",
    "Loss/value_function": "vloss",
}


def series(run, tag):
    s = data[run].get(tag)
    if not s:
        return None, None
    a = np.array(s)
    return a[:, 0], a[:, 1]


def at(run, tag, it, w=50):
    st, v = series(run, tag)
    if st is None:
        return np.nan
    m = (st >= it - w) & (st <= it + w)
    return float(np.mean(v[m])) if m.any() else np.nan


def table(runs, tag, its):
    print(f"\n### {tag}  (window +-50 it)")
    print("run".ljust(30) + "".join(f"{i:>9d}" for i in its))
    for r in runs:
        if r not in data:
            continue
        print(r.ljust(30) + "".join(f"{at(r, tag, i):9.3f}" for i in its))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "base"
    if which == "base":
        its = [50, 150, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 5950]
        runs = BASE
    else:
        its = [6050, 6300, 6600, 7000, 7500, 8000, 8500, 8950]
        runs = FT + ["arms-turn-turnboth-s0"]
    for tag in SHORT:
        table(runs, tag, its)
