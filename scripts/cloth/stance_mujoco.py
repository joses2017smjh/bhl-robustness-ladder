#!/usr/bin/env python3
"""Why the cloth-sort squat falls on a free base, measured in MuJoCo.

On the free base the robot falls backward within a second even with its arm held
still (21317172). This reproduces that fall offline and takes it apart. The
MuJoCo robot is the crew model the reach tables were built from, driven by
Isaac's actuator model: PD at 20 N m/rad, 2 N m s/rad, 6 N m on the legs and 10,
2, 4 on the arms, rotor inertia as in HUMANOID_LITE_CFG, torques applied at a
0.5 ms step with targets held for Isaac's 5 ms.

    fall      pitch curve of the pinch squat, arm still, against Isaac's trace
    sole      the foot box at the spawn pose: sole pitch, penetration, flat-sole ankle
    knees     leg torques and joint sag while the squat collapses
    map       quasi-static leg torques over level, flat-soled stances
    search    IMU ankle pitch/roll feedback gains on the knee-1.0 stance, with
              Isaac's +-0.04 rad reset noise
    stress    the deployed controller (bhl_robust.cloth.balance, through the same
              function the Isaac term mirrors): noisy resets, arm still and arm
              playing the scripted sweep schedule
    robust    gain search with the arm sweeping in the objective: each gain set
              over noisy resets with the scripted sweep playing, ranked by stands
              then worst tilt (the Isaac probe fell 1 of 4 with the arm moving)

Writes ``results/cloth/stance/<command>.json``. CPU only:

    /nfs/hpc/share/$USER/Humanoid_Lite/venv/bin/python3 scripts/cloth/stance_mujoco.py all
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from bhl_robust.cloth.layout import ROBOT_ROOT_Z  # noqa: E402
from bhl_robust.eval.coop_replay import JOINTS, PINCH_POSE, build_crew  # noqa: E402

OUT = REPO / "results" / "cloth" / "stance"
DT, SUB = 0.005, 10
KP = np.array([10.0 if j.startswith("arm") else 20.0 for j in JOINTS])
KD = np.full(len(JOINTS), 2.0)
EFFORT = np.array([4.0 if j.startswith("arm") else 6.0 for j in JOINTS])
J = {j: i for i, j in enumerate(JOINTS)}


class Rig:
    """Robot 0 of the crew alone: robot 1 and the crate are parked 6 m away."""

    def __init__(self):
        m, slots, crates = build_crew(REPO / "external" / "Berkeley-Humanoid-Lite",
                                      Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/mjcf_cache"), 2,
                                      ego_camera=False, payload="cube")
        m.opt.timestep = DT / SUB
        m.actuator_gainprm[:] = 0.0
        m.actuator_biasprm[:] = 0.0
        self.s, self.s1, self.crate = slots[0], slots[1], crates[0]
        for i, j in enumerate(JOINTS):
            dof = int(self.s.jnt_qvel[i])
            m.dof_damping[dof] = 0.0
            m.dof_frictionloss[dof] = 0.0
            m.dof_armature[dof] = 0.002 if (j.startswith("arm") or "ankle" in j) else 0.007
        self.m = m
        self.foot = [g for g in range(m.ngeom)
                     if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "")
                     in ("r0_leg_left_ankle_roll", "r0_leg_right_ankle_roll") and m.geom_contype[g] != 0]
        self.mass = float(m.body_subtreemass[self.s.body_id])

    def data(self, q, root_z):
        d = mujoco.MjData(self.m)
        s = self.s
        d.qpos[s.qpos_adr:s.qpos_adr + 7] = [0, 0, root_z, 1, 0, 0, 0]
        d.qpos[self.s1.qpos_adr:self.s1.qpos_adr + 7] = [0, 6.0, 0.5, 1, 0, 0, 0]
        d.qpos[self.crate.qpos_adr:self.crate.qpos_adr + 7] = [0, -6.0, 0.2, 1, 0, 0, 0]
        d.qpos[s.jnt_qpos] = q
        mujoco.mj_forward(self.m, d)
        return d

    def box(self, d, g):
        size, R = self.m.geom_size[g], d.geom_xmat[g].reshape(3, 3)
        c = np.array([[a, b, cc] for a in (-1, 1) for b in (-1, 1) for cc in (-1, 1)]) * size
        return d.geom_xpos[g] + c @ R.T

    def tilt(self, d):
        R = Rot.from_quat(d.qpos[self.s.qpos_adr + 3:self.s.qpos_adr + 7], scalar_first=True).as_matrix()
        up = R[:, 2]
        w = R @ d.qvel[self.s.qvel_adr + 3:self.s.qvel_adr + 6]
        return (float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))),
                float(np.arctan2(up[0], up[2])), float(np.arctan2(-up[1], up[2])), w)

    def run(self, q0, root_z, target, T, log_every=None, stop_deg=None):
        """Step with ``target(t, d) -> (q_target, qd_target)``; returns the log and the worst tilt."""
        d = self.data(q0, root_z)
        s, log, worst = self.s, [], 0.0
        for k in range(int(round(T / DT))):
            qt, vt = target(k * DT, d)
            for _ in range(SUB):
                q, v = d.qpos[s.jnt_qpos], d.qvel[s.jnt_qvel]
                tau = np.clip(KP * (qt - q) + KD * (vt - v), -EFFORT, EFFORT)
                d.qfrc_applied[s.jnt_qvel] = tau
                mujoco.mj_step(self.m, d)
            tilt, pitch, roll, _ = self.tilt(d)
            worst = max(worst, tilt)
            if log_every and k % log_every == 0:
                log.append({"t": round(k * DT, 3), "tilt_deg": tilt, "pitch_deg": float(np.degrees(pitch)),
                            "root_z": float(d.qpos[s.qpos_adr + 2]), "tau": tau.tolist(),
                            "q_minus_target": (d.qpos[s.jnt_qpos] - qt).tolist()})
            if stop_deg and tilt > stop_deg:
                break
        return log, worst


def pinch() -> np.ndarray:
    return np.array([PINCH_POSE.get(j, 0.0) for j in JOINTS])


def stance(hip, knee, ankle) -> np.ndarray:
    q = pinch()
    for side in ("left", "right"):
        q[J[f"leg_{side}_hip_pitch_joint"]] = hip
        q[J[f"leg_{side}_knee_pitch_joint"]] = knee
        q[J[f"leg_{side}_ankle_pitch_joint"]] = ankle
    return q


def cmd_fall(rig: Rig) -> dict:
    q = pinch()
    log, worst = rig.run(q, ROBOT_ROOT_Z, lambda t, d: (q, np.zeros_like(q)), T=1.2, log_every=10)
    isaac = json.load(open(REPO / "results" / "cloth" / "isaac_c0_hold_trace.json"))
    S = isaac["samples"]
    rq = np.array([x["root_quat"] for x in S])
    R = Rot.from_quat(rq, scalar_first=(isaac["quat_order"] == "wxyz"))
    pitch_isaac = (R[0].inv() * R).as_euler("xyz", degrees=True)[:, 1]
    t_isaac = np.array([x["t"] for x in S])
    rows = []
    for row in log:
        k = int(np.argmin(np.abs(t_isaac - row["t"])))
        rows.append({"t": row["t"], "mujoco_pitch_deg": row["pitch_deg"], "isaac_pitch_deg": float(pitch_isaac[k]),
                     "mujoco_root_z": row["root_z"]})
    for r in rows[::2]:
        print(f"  t={r['t']:.2f}  MuJoCo pitch {r['mujoco_pitch_deg']:+6.1f}  Isaac (21317172) {r['isaac_pitch_deg']:+6.1f}")
    return {"robot_mass_kg": rig.mass, "rows": rows,
            "note": "MuJoCo pitch: + = forward lean. Isaac pitch: body-frame euler relative to spawn, "
                    "negative = backward for a robot facing -x."}


def cmd_sole(rig: Rig) -> dict:
    out = []
    ia = [J["leg_left_ankle_pitch_joint"], J["leg_right_ankle_pitch_joint"]]
    for ankle in np.round(np.arange(-0.70, -0.449, 0.025), 3):
        q = pinch()
        q[ia] = ankle
        d = rig.data(q, 0.0)
        c = rig.box(d, rig.foot[0])
        bottom = c[np.argsort(c[:, 2])[:4]]
        heel, toe = bottom[np.argmin(bottom[:, 0])], bottom[np.argmax(bottom[:, 0])]
        zmin = min(rig.box(d, g)[:, 2].min() for g in rig.foot)
        out.append({"ankle": float(ankle),
                    "sole_pitch_deg_toe_down": float(np.degrees(np.arctan2(heel[2] - toe[2], toe[0] - heel[0]))),
                    "root_z_sole_on_floor": float(-zmin), "com_x": float(d.subtree_com[rig.s.body_id][0]),
                    "sole_x": [float(bottom[:, 0].min()), float(bottom[:, 0].max())]})
    spawn = next(r for r in out if abs(r["ankle"] + 0.55) < 1e-6)
    flat = min(out, key=lambda r: abs(r["sole_pitch_deg_toe_down"]))
    print(f"  spawn pose (ankle -0.55): sole {spawn['sole_pitch_deg_toe_down']:+.2f} deg toe-down; at root "
          f"{ROBOT_ROOT_Z} its lowest point is {1000 * (spawn['root_z_sole_on_floor'] - ROBOT_ROOT_Z):.1f} mm into the floor")
    print(f"  flat sole: ankle {flat['ankle']:+.3f}, root z {flat['root_z_sole_on_floor']:+.4f}, COM x "
          f"{flat['com_x']:+.3f} inside sole x {flat['sole_x']}")
    return {"spawn_root_z": ROBOT_ROOT_Z, "spawn_penetration_m": spawn["root_z_sole_on_floor"] - ROBOT_ROOT_Z,
            "flat": flat, "rows": out}


def cmd_knees(rig: Rig) -> dict:
    q = pinch()
    q[[J["leg_left_ankle_pitch_joint"], J["leg_right_ankle_pitch_joint"]]] = -0.60
    log, _ = rig.run(q, -0.1179, lambda t, d: (q, np.zeros_like(q)), T=0.6, log_every=20)
    legs = [J[f"leg_left_{n}_joint"] for n in ("hip_pitch", "knee_pitch", "ankle_pitch")]
    rows = [{"t": r["t"], "root_z": r["root_z"],
             "tau_hip_knee_ankle": [r["tau"][i] for i in legs],
             "sag_hip_knee_ankle": [r["q_minus_target"][i] for i in legs]} for r in log]
    for r in rows:
        print(f"  t={r['t']:.2f} root z {r['root_z']:+.3f}  tau hip/knee/ankle "
              + " ".join(f"{v:+.2f}" for v in r["tau_hip_knee_ankle"])
              + "  sag " + " ".join(f"{v:+.3f}" for v in r["sag_hip_knee_ankle"]))
    return {"stance": "pinch squat with flat soles (ankle -0.60), root -0.1179", "rows": rows,
            "knee_effort_limit": 6.0}


def cmd_map(rig: Rig) -> dict:
    W = rig.mass * 9.81
    jid = lambda n: mujoco.mj_name2id(rig.m, mujoco.mjtObj.mjOBJ_JOINT, "r0_" + n)
    out = []
    for knee in np.round(np.arange(0.3, 1.51, 0.1), 2):
        for hip in np.round(np.arange(-1.0, 0.001, 0.05), 2):
            ankle = -(hip + knee)
            d = rig.data(stance(hip, knee, ankle), 0.0)
            c = rig.box(d, rig.foot[0])
            com = d.subtree_com[rig.s.body_id]
            cop = np.array([com[0], d.geom_xpos[rig.foot[0]][1], c[:, 2].min()])
            tau = []
            for jn in ("leg_left_hip_pitch_joint", "leg_left_knee_pitch_joint", "leg_left_ankle_pitch_joint"):
                j = jid(jn)
                b = int(rig.m.jnt_bodyid[j])
                axis = d.xmat[b].reshape(3, 3) @ rig.m.jnt_axis[j]
                tau.append(float(np.cross(cop - d.xpos[b], [0.0, 0.0, W / 2]) @ axis))
            out.append({"hip": float(hip), "knee": float(knee), "ankle": float(ankle),
                        "root_z": float(-c[:, 2].min()), "com_x": float(com[0]),
                        "com_margin": float(min(com[0] - c[:, 0].min(), c[:, 0].max() - com[0])),
                        "tau_hip_knee_ankle": tau, "peak_tau": max(abs(t) for t in tau)})
    inside = [r for r in out if r["com_margin"] > 0.05]
    for knee in (0.4, 0.7, 1.0, 1.3, 1.5):
        best = min((r for r in inside if abs(r["knee"] - knee) < 1e-6), key=lambda r: r["peak_tau"], default=None)
        if best:
            print(f"  knee {knee:.1f}: least peak leg torque {best['peak_tau']:.2f} N m at hip {best['hip']:+.2f}, "
                  f"root z {best['root_z']:+.3f}")
    return {"robot_mass_kg": rig.mass, "assumption": "vertical ground reaction under the COM, half per foot",
            "rows": out}


def cmd_search(rig: Rig) -> dict:
    hip, knee, ankle, root_z = -0.50, 1.00, -0.50, -0.0775
    q_st = stance(hip, knee, ankle)
    # Feedforward: the settled leg torques of a run that stood without noise, over Kp.
    ap = [J["leg_left_ankle_pitch_joint"], J["leg_right_ankle_pitch_joint"]]
    ar = [J["leg_left_ankle_roll_joint"], J["leg_right_ankle_roll_joint"]]

    def controller(ff, kpp, kdp, kpr, kdr):
        def f(t, d):
            _, pitch, roll, w = rig.tilt(d)
            qt = q_st + ff
            qt[ap] += kpp * pitch + kdp * w[1]
            qt[ar] += kpr * roll - kdr * w[0]
            return qt, np.zeros_like(qt)
        return f

    legs = [i for i, j in enumerate(JOINTS) if j.startswith("leg")]
    log, worst0 = rig.run(q_st, root_z, controller(np.zeros(len(JOINTS)), 2.0, 0.2, 0.0, 0.0), T=3.0, log_every=1)
    settled = np.mean([r["tau"] for r in log if r["t"] > 1.5], axis=0)
    ff = np.zeros(len(JOINTS))
    ff[legs] = settled[legs] / KP[legs]
    rng = np.random.default_rng(0)
    results = []
    for kpp, kdp, sign, kpr, kdr in itertools.product((1.5, 2.0, 2.5, 3.0), (0.1, 0.2, 0.3), (+1, -1),
                                                      (1.0, 2.0, 3.0), (0.05, 0.15)):
        worst = 0.0
        for _ in range(2):
            q0 = q_st + rng.uniform(-0.04, 0.04, len(q_st))
            _, w = rig.run(q0, root_z, controller(ff, kpp, kdp, sign * kpr, sign * kdr), T=6.0, stop_deg=45.0)
            worst = max(worst, w)
            if w > 45.0:
                break
        results.append({"pitch_kp": kpp, "pitch_kd": kdp, "roll_kp": sign * kpr, "roll_kd": sign * kdr,
                        "worst_tilt_deg": worst})
    stood = [r for r in results if r["worst_tilt_deg"] < 30.0]
    print(f"  {len(stood)} of {len(results)} gain sets stood 6 s under +-0.04 rad reset noise")
    return {"stance": {"hip": hip, "knee": knee, "ankle": ankle, "root_z": root_z},
            "noise_free_reference_worst_tilt_deg": worst0, "feedforward_rad": ff.tolist(),
            "gain_sets": len(results), "stood": len(stood), "results": results}


def cmd_stress(rig: Rig) -> dict:
    from bhl_robust.cloth import balance, layout, schedule, scripted
    from bhl_robust.cloth.garments import GARMENT_BY_NAME

    q_st = pinch()
    for j, v in balance.STANCE.items():
        q_st[J[j]] = v
    ff = np.zeros(len(q_st))
    for j, v in balance.FEEDFORWARD.items():
        ff[J[j]] = v
    ap = [J["leg_left_ankle_pitch_joint"], J["leg_right_ankle_pitch_joint"]]
    ar = [J["leg_left_ankle_roll_joint"], J["leg_right_ankle_roll_joint"]]
    arm = [J[j] for j in schedule.load_contact().joints]
    spec = GARMENT_BY_NAME["shirt_a"]
    g = np.array(layout.default_spawn_xy(spec))
    sch = schedule.build_schedule(g, spec, scripted.scripted_action(g, spec), DT, int(round(schedule.MACRO_STEP_S / DT)))

    def controller(with_arm):
        def f(t, d):
            M = Rot.from_quat(d.qpos[rig.s.qpos_adr + 3:rig.s.qpos_adr + 7], scalar_first=True).as_matrix()
            w = M @ d.qvel[rig.s.qvel_adr + 3:rig.s.qvel_adr + 6]
            po, ro = balance.ankle_offsets(M[0, 2], M[1, 2], M[2, 2], w[0], w[1], float(np.arctan2(M[1, 0], M[0, 0])))
            qt, vt = q_st + ff, np.zeros_like(q_st)
            qt[ap] += po
            qt[ar] += ro
            if with_arm:
                k = min(int(round(t / DT)), len(sch.q_cmd) - 1)
                qt[arm], vt[arm] = sch.q_cmd[k], sch.qd[k]
                qt[ap] += sch.ankle_ff[k, 0]
                qt[ar] += sch.ankle_ff[k, 1]
            return qt, vt
        return f

    rng = np.random.default_rng(7)
    out = {}
    for label, with_arm, T in (("arm_still", False, 10.0), ("arm_sweeping", True, 6.0)):
        worst = []
        for _ in range(8):
            q0 = q_st + rng.uniform(-0.04, 0.04, len(q_st))
            _, w = rig.run(q0, balance.ROOT_Z, controller(with_arm), T=T, stop_deg=45.0)
            worst.append(w)
        out[label] = {"seconds": T, "trials": len(worst), "stood": int(sum(w < 30.0 for w in worst)),
                      "worst_tilt_deg": worst}
        print(f"  {label}: stood {out[label]['stood']}/{len(worst)}, worst tilt median {np.median(worst):.1f} "
              f"max {max(worst):.1f} deg")
    return {"gains": list(balance.GAINS), "root_z": balance.ROOT_Z, "schedule_valid": bool(sch.valid), **out}


def cmd_robust(rig: Rig) -> dict:
    from bhl_robust.cloth import balance, layout, schedule, scripted
    from bhl_robust.cloth.garments import GARMENT_BY_NAME

    q_st = pinch()
    for j, v in balance.STANCE.items():
        q_st[J[j]] = v
    ff = np.zeros(len(q_st))
    for j, v in balance.FEEDFORWARD.items():
        ff[J[j]] = v
    ap = [J["leg_left_ankle_pitch_joint"], J["leg_right_ankle_pitch_joint"]]
    ar = [J["leg_left_ankle_roll_joint"], J["leg_right_ankle_roll_joint"]]
    arm = [J[j] for j in schedule.load_contact().joints]
    spec = GARMENT_BY_NAME["shirt_a"]
    g = np.array(layout.default_spawn_xy(spec))
    sch = schedule.build_schedule(g, spec, scripted.scripted_action(g, spec), DT, int(round(schedule.MACRO_STEP_S / DT)))

    def controller(gains):
        def f(t, d):
            M = Rot.from_quat(d.qpos[rig.s.qpos_adr + 3:rig.s.qpos_adr + 7], scalar_first=True).as_matrix()
            w = M @ d.qvel[rig.s.qvel_adr + 3:rig.s.qvel_adr + 6]
            po, ro = balance.ankle_offsets(M[0, 2], M[1, 2], M[2, 2], w[0], w[1],
                                           float(np.arctan2(M[1, 0], M[0, 0])), gains)
            qt, vt = q_st + ff, np.zeros_like(q_st)
            qt[ap] += po
            qt[ar] += ro
            k = min(int(round(t / DT)), len(sch.q_cmd) - 1)
            qt[arm], vt[arm] = sch.q_cmd[k], sch.qd[k]
            return qt, vt
        return f

    results = []
    for gains in itertools.product((1.0, 1.5, 2.0, 2.5), (0.3, 0.45, 0.6), (1.5, 2.0, 3.0), (0.05, 0.15)):
        rng = np.random.default_rng(11)          # the same resets for every gain set
        worst, stood = [], 0
        for _ in range(6):
            q0 = q_st + rng.uniform(-0.04, 0.04, len(q_st))
            _, w = rig.run(q0, balance.ROOT_Z, controller(gains), T=6.0, stop_deg=45.0)
            worst.append(w)
            stood += w < 30.0
        results.append({"gains": list(gains), "stood": int(stood), "trials": 6, "worst_tilt_deg": worst,
                        "max_tilt_stood_deg": max([w for w in worst if w < 30.0], default=None)})
    results.sort(key=lambda r: (-r["stood"], r["max_tilt_stood_deg"] if r["max_tilt_stood_deg"] is not None else 99))
    for r in results[:8]:
        print(f"  gains {r['gains']}: stood {r['stood']}/6, max tilt among stands {r['max_tilt_stood_deg']}")
    return {"current_gains": list(balance.GAINS), "results": results}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("fall", "sole", "knees", "map", "search", "stress", "robust", "all"))
    args = p.parse_args()
    rig = Rig()
    OUT.mkdir(parents=True, exist_ok=True)
    cmds = {"fall": cmd_fall, "sole": cmd_sole, "knees": cmd_knees, "map": cmd_map, "search": cmd_search,
            "stress": cmd_stress, "robust": cmd_robust}
    for name in (cmds if args.command == "all" else [args.command]):
        print(f"== {name}")
        (OUT / f"{name}.json").write_text(json.dumps(cmds[name](rig), indent=1))
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
