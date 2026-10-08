"""G-T1 and G-T3: is GRASP_Z reachable standing, and unreachable collapsed?

Pure kinematics -- no policy, no training, no GPU. The redesign's whole claim is
that putting the payload at 0.30 m makes standing instrumentally necessary. That
claim is falsifiable in a forward-kinematics sweep, and it costs seconds, so it
gets checked before anything trains on it.

G-T1 passes when a posture exists that puts both hands at GRASP_Z with the base
still above `--min-base`, i.e. a squat rather than a collapse.
G-T3 passes when no posture reaches GRASP_Z from a collapsed base, so the
behaviour the old tasks rewarded now forfeits the payload.

The knees are swept along with the arms, which the first reach probe did not do
-- it varied arm joints only and so could not see a squat at all.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import mujoco
import numpy as np

from bhl_robust.eval.coop_replay import ARM_JOINTS, JOINTS, PINCH_POSE, build_crew
from bhl_robust.reach_band import COLLAPSE_HAND_HI, GRASP_Z


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--grasp-z", type=float, default=GRASP_Z)
    p.add_argument("--max-drop", type=float, default=0.25,
                   help="descent from standing beyond which a posture is a "
                        "collapse rather than a squat; the policies drop 0.41")
    p.add_argument("--collapse-drop", type=float, default=0.41,
                   help="the descent the current cube policies actually perform")
    p.add_argument("--tol", type=float, default=0.03)
    p.add_argument("--grid", type=int, default=5)
    p.add_argument("--shelf-deck", type=float, default=0.38)
    p.add_argument("--net-rim", type=float, default=0.60)
    p.add_argument("--wall-contact", type=float, default=0.50)
    args = p.parse_args()

    model, slots, _ = build_crew(args.upstream, args.cache_dir, 2,
                                 ego_camera=False, payload="cube")
    d = mujoco.MjData(model)
    s = slots[0]
    jid = lambda j: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, s.prefix + j)
    adr = {j: int(model.jnt_qposadr[jid(j)]) for j in JOINTS}
    rng = {j: model.jnt_range[jid(j)] for j in JOINTS}
    base = {j: PINCH_POSE.get(j, 0.0) for j in JOINTS}

    geoms = [g for g in range(model.ngeom)
             if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                   int(model.geom_bodyid[g])) or "").startswith(s.prefix)]

    def pose(over: dict, root_z: float, plant: bool = False):
        """Forward kinematics; `plant` re-seats the feet on the floor.

        Planting is the difference between a squat and a joke. With the root
        pinned, bending the knees lifts the feet into the air and the hands do
        not descend at all -- which is why the first version of this gate found
        0.30 m unreachable while reporting a closest approach of 0.407 m, the
        arms-only figure. A real squat keeps the feet down and lowers the torso.
        """
        def _set(rz):
            d.qpos[:] = 0.0
            d.qpos[s.qpos_adr + 2] = rz
            d.qpos[s.qpos_adr + 3] = 1.0
            for j, v in {**base, **over}.items():
                d.qpos[adr[j]] = v
            mujoco.mj_forward(model, d)
            return min(float(d.geom_xpos[g][2]) for g in geoms)

        low = _set(root_z)
        if plant:
            low = _set(root_z - low)      # drop until the lowest geom is at 0
        return float(d.xpos[s.hands][:, 2].mean()), low, float(d.xpos[s.body_id][2])

    _, low0, _ = pose({}, 0.0)
    stand_z = -low0                       # plant the feet

    # Sweep the joints that can lower a hand: shoulder pitch, elbow pitch, and
    # -- the part the first probe missed -- hip and knee pitch.
    keys = ["arm_left_shoulder_pitch_joint", "arm_left_elbow_pitch_joint",
            "leg_left_hip_pitch_joint", "leg_left_knee_pitch_joint"]
    grids = []
    for k in keys:
        lo, hi = rng[k]
        grids.append(np.linspace(lo, hi, args.grid) if hi > lo
                     else np.linspace(-2.0, 2.0, args.grid))

    def sweep(root_z: float, plant: bool):
        hits = []
        for combo in itertools.product(*grids):
            over = dict(zip(keys, combo))
            # Mirror the leg joints so the squat is symmetric and the robot
            # does not "reach" by falling sideways onto one hip.
            for k, v in list(over.items()):
                if k.startswith("leg_left"):
                    over[k.replace("leg_left", "leg_right")] = v
            hz, low, bz = pose(over, root_z, plant=plant)
            if low < -0.02:               # posture drives a geom through the floor
                continue
            hits.append((abs(hz - args.grasp_z), hz, bz, low))
        hits.sort()
        return hits

    print(f"grasp height under test: {args.grasp_z:.3f} m "
          f"(tolerance +/-{args.tol:.3f})")
    print(f"standing root_z = {stand_z:+.3f}\n")

    _, _, base_stand = pose({}, stand_z, plant=True)
    st = sweep(stand_z, plant=True)
    ok_st = [h for h in st
             if h[0] <= args.tol and (base_stand - h[2]) <= args.max_drop]
    print(f"G-T1  standing (feet planted): {len(st)} legal postures, "
          f"{len(ok_st)} put both hands within tolerance of {args.grasp_z:.2f} m "
          f"while descending no more than {100*args.max_drop:.0f} cm")
    if st:
        _, hz, bz, low = st[0]
        print(f"      closest overall: hands {hz:.3f} m, base {bz:+.3f} m, "
              f"lowest geom {low:+.3f} m")
    if ok_st:
        _, hz, bz, low = ok_st[0]
        print(f"      best qualifying: hands {hz:.3f} m, "
              f"descent {100*(base_stand - bz):.1f} cm -> a squat, not a collapse")

    co = sweep(stand_z - args.collapse_drop, plant=False)
    ok_co = [h for h in co if h[0] <= args.tol]
    print(f"\nG-T3  collapsed by {100*args.collapse_drop:.0f} cm: "
          f"{len(co)} legal postures, {len(ok_co)} reach {args.grasp_z:.2f} m")
    if co:
        print(f"      closest: hands {co[0][1]:.3f} m "
              f"(collapsed ceiling measured at {COLLAPSE_HAND_HI:.3f} m)")

    # ---- G-T2: are the target interaction heights reachable standing? -----
    # A payload you can pick up and then cannot place is a task gated on the
    # shelf rather than on the policy. Each target is the height the payload's
    # *centre* must reach, so the hands must reach it too.
    reach_lo = min(h[1] for h in st)
    reach_hi = max(h[1] for h in st)
    targets = [
        ("cube -> shelf slot", args.shelf_deck + 0.14),
        ("ball -> net rim", args.net_rim),
        ("plank -> wall contact", args.wall_contact),
    ]
    print(f"\nG-T2  standing hands span {reach_lo:.3f} .. {reach_hi:.3f} m "
          f"with feet planted")
    t2 = True
    for name, z in targets:
        ok = reach_lo - args.tol <= z <= reach_hi + args.tol
        t2 &= ok
        print(f"      {name:26} needs {z:.3f} m  {'ok' if ok else 'OUT OF REACH'}")

    print()
    t1 = len(ok_st) > 0
    t3 = len(ok_co) == 0
    print(f"G-T1 {'PASS' if t1 else 'FAIL'} | "
          f"{'a standing squat reaches the payload' if t1 else 'no standing posture reaches it -- GRASP_Z is wrong'}")
    print(f"G-T2 {'PASS' if t2 else 'FAIL'} | "
          f"{'every target height is reachable standing' if t2 else 'a target is outside the standing envelope'}")
    print(f"G-T3 {'PASS' if t3 else 'FAIL'} | "
          f"{'a collapsed robot cannot reach it' if t3 else 'a collapsed robot still reaches it -- raising the payload bought nothing'}")
    raise SystemExit(0 if (t1 and t2 and t3) else 1)


if __name__ == "__main__":
    main()
