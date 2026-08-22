"""How far does each policy get across the composed lab floor, and what stops it?

The lab clip is the one place in this repo where a GIF could be read as a claim
that nobody checked. The camera tracks the pack, so a robot stalled against a
threshold and a robot walking look similar for the second or two that a loop
lasts, and "four policies on a composed floor" says nothing about whether any
of them crossed it.

Two things this reports that the clip cannot:

* the ray-cast height profile of the lane, in centimetres and as a fraction of
  leg length, so the course is stated rather than eyeballed;
* per policy, the furthest x reached, which feature it was at when it fell, and
  whether it finished.

Run it whenever the scene geometry changes. The first version of `_LAB_WORLD`
omitted `<compiler angle="radian">` while writing its eulers in radians, which
turned a 5 cm cable into a 5.2 m vertical pole and a ramp into a flat slab --
and the clip looked entirely plausible the whole time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from omegaconf import OmegaConf

from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
from bhl_robust.eval.multi_robot import _WORLDS, MultiRunner, build_multi

TILT_LIMIT = 0.78
# 0.12 m thigh + 0.16 m shank. Obstacle difficulty for a legged machine
# normalises on leg length, not on absolute centimetres.
LEG_M = 0.28


def height_profile(x0: float = -0.4, x1: float = 5.2, step: float = 0.02):
    """Walking-surface height along the lane centreline, by ray-cast.

    Rays are cast straight down from above and the *first* hit is the surface a
    foot would land on, which is the quantity that matters and is not the same
    as any single geom's size attribute.
    """
    path = Path("/tmp/bhl_lab_profile.xml")
    path.write_text(_WORLDS["lab"])
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    xs = np.arange(x0, x1, step)
    zs = np.zeros_like(xs)
    names = []
    gid = np.zeros(1, dtype=np.int32)
    for i, x in enumerate(xs):
        dist = mujoco.mj_ray(m, d, np.array([x, 0.0, 2.0]),
                             np.array([0.0, 0.0, -1.0]), None, 1, -1, gid)
        zs[i] = 2.0 - dist if dist >= 0 else 0.0
        names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(gid[0]))
                     if gid[0] >= 0 else "-")
    return xs, zs, names


def _local_half_extent(model: mujoco.MjModel, g: int) -> np.ndarray:
    """Half-extents of geom `g` in its own frame, as an (x, y, z) box.

    `geom_size` is not a half-extent vector for every type -- a cylinder stores
    (radius, half_length) with the length along local z, so reading it as a box
    reported the lane's 2.5 cm cable as 2.6 m tall once it was rotated flat.
    """
    s = model.geom_size[g]
    t = int(model.geom_type[g])
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return s[:3].copy()
    if t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return np.array([s[0], s[0], s[1]])
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return np.array([s[0], s[0], s[1] + s[0]])
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.array([s[0], s[0], s[0]])
    raise ValueError(f"no half-extent rule for geom type {t}")


def features():
    """(name, leading edge x, trailing edge x, top height) per obstacle."""
    path = Path("/tmp/bhl_lab_feat.xml")
    path.write_text(_WORLDS["lab"])
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    out = []
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name in ("floor", ""):
            continue
        pos = d.geom_xpos[g]
        R = d.geom_xmat[g].reshape(3, 3)
        ext = np.abs(R) @ _local_half_extent(m, g)
        out.append((name, float(pos[0] - ext[0]), float(pos[0] + ext[0]),
                    float(pos[2] + ext[2])))
    return sorted(out, key=lambda r: r[1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deploy", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--variant", choices=("biped", "humanoid"), default="biped")
    p.add_argument("--seconds", type=float, default=24.0)
    p.add_argument("--vx", type=float, default=0.40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=== lane height profile (ray-cast, y=0) ===")
    xs, zs, names = height_profile()
    prev, prev_name = 0.0, "floor"
    for x, z, nm in zip(xs, zs, names):
        if abs(z - prev) > 0.004 or nm != prev_name:
            print(f"  x={x:5.2f}  surface {z * 100:5.1f} cm  "
                  f"({z / LEG_M * 100:4.0f}% leg)  step {(z - prev) * 100:+5.1f} cm  [{nm}]")
            prev, prev_name = z, nm
    print("\n=== features (leading edge, top) ===")
    feats = features()
    for nm, x_lead, x_trail, z_top in feats:
        print(f"  {nm:<11} x {x_lead:5.2f} -> {x_trail:5.2f}  top {z_top * 100:5.1f} cm "
              f"({z_top / LEG_M * 100:4.0f}% leg)")
    # Clearing the course means walking off the far edge of the last feature,
    # not merely touching its leading edge.
    finish = max(f[2] for f in feats)

    n = len(args.deploy)
    cfgs = [OmegaConf.load(d) for d in args.deploy]
    model, slots = build_multi(args.upstream, args.cache_dir, n, args.labels,
                               variant=args.variant, world="lab")
    ctrls = []
    for c in cfgs:
        rc = RlController(c)
        rc.load_policy()
        ctrls.append(rc)

    run = MultiRunner(model, slots, cfgs, ctrls)
    rng = np.random.default_rng(args.seed)
    run.reset(rng)

    dt = cfgs[0].policy_dt
    steps = int(args.seconds / dt)
    start = np.array([run.d.xpos[s.body_id][0] for s in slots])
    peak = start.copy()
    fell = [None] * n

    for t in range(steps):
        targets = [ctrls[i].update(run.observe(i, (args.vx, 0.0, 0.0))) for i in range(n)]
        run.step(targets)
        for i, s in enumerate(slots):
            x = float(run.d.xpos[s.body_id][0])
            peak[i] = max(peak[i], x)
            if fell[i] is None and run.tilt(i) > TILT_LIMIT:
                fell[i] = (t * dt, x)

    print(f"\n=== traversal: variant={args.variant} vx={args.vx} "
          f"{args.seconds:.0f}s  finish line x={finish:.2f} ===")
    print(f"{'policy':<20}{'peak x':>9}{'progress':>10}  outcome")
    for i, s in enumerate(slots):
        if fell[i] is not None:
            at = "flat"
            for nm, x_lead, _, _ in feats:
                if fell[i][1] >= x_lead:
                    at = nm
            verdict = f"FELL t={fell[i][0]:5.1f}s x={fell[i][1]:5.2f} (on {at})"
        elif peak[i] >= finish:
            verdict = "FINISHED the course"
        else:
            stuck = "flat"
            for nm, x_lead, _, _ in feats:
                if peak[i] >= x_lead:
                    stuck = nm
            verdict = f"upright, stalled past {stuck}"
        print(f"{s.label:<20}{peak[i]:>9.2f}{peak[i] - start[i]:>10.2f}  {verdict}")


if __name__ == "__main__":
    main()
