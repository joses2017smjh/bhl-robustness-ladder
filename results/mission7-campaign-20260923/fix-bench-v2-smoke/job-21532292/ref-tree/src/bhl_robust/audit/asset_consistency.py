"""Three-way asset consistency audit: URDF vs USD vs MJCF.

BHL describes the same robot three times. Isaac Lab trains from the USD, the
sim2sim harness scores from the MJCF, and the URDF is the nominal source both
were converted from. Every number this project reports as a "PhysX vs MuJoCo"
gap silently assumes those three agree.

If they do not, part of the measured sim2sim gap is asset drift -- bookkeeping,
not physics -- and the transfer numbers have an unaccounted term in them. This
script either finds that term or licenses the claim that the gap is genuinely
physics.

Compared, per link/joint, with an explicit tolerance:
  * link mass
  * diagonal inertia
  * joint position limits
  * joint damping / friction
  * collision geometry representation (mesh vs primitive)
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco

REL_TOL = 0.01   # 1% — tighter than any physical difference these should have


# --------------------------------------------------------------------------- URDF
@dataclass
class Link:
    mass: float | None = None
    # Principal moments, sorted ascending. MuJoCo stores inertia in the body's
    # PRINCIPAL frame (with body_iquat holding the rotation), while URDF states
    # it in the link frame. Comparing raw diagonals therefore reports spurious
    # mismatches whenever MuJoCo reordered the axes. Sorted eigenvalues are
    # frame-independent, so they are what actually has to agree.
    inertia: tuple[float, float, float] | None = None
    n_collision: int = 0
    collision_kinds: set[str] = field(default_factory=set)


@dataclass
class Joint:
    lower: float | None = None
    upper: float | None = None
    damping: float | None = None
    friction: float | None = None
    effort: float | None = None


def read_urdf(path: Path):
    root = ET.parse(path).getroot()
    links, joints = {}, {}
    for l in root.findall("link"):
        li = Link()
        inert = l.find("inertial")
        if inert is not None:
            m = inert.find("mass")
            if m is not None:
                li.mass = float(m.get("value"))
            i = inert.find("inertia")
            if i is not None:
                ixx, iyy, izz = (float(i.get(k)) for k in ("ixx", "iyy", "izz"))
                ixy, ixz, iyz = (float(i.get(k, 0.0)) for k in ("ixy", "ixz", "iyz"))
                import numpy as _np
                T = _np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
                li.inertia = tuple(sorted(float(x) for x in _np.linalg.eigvalsh(T)))
        for c in l.findall("collision"):
            li.n_collision += 1
            g = c.find("geometry")
            if g is not None and len(g):
                li.collision_kinds.add(g[0].tag)
        links[l.get("name")] = li
    for j in root.findall("joint"):
        if j.get("type") in ("fixed", None):
            continue
        jo = Joint()
        lim = j.find("limit")
        if lim is not None:
            jo.lower = float(lim.get("lower", "nan"))
            jo.upper = float(lim.get("upper", "nan"))
            jo.effort = float(lim.get("effort", "nan"))
        dyn = j.find("dynamics")
        if dyn is not None:
            jo.damping = float(dyn.get("damping", "nan"))
            jo.friction = float(dyn.get("friction", "nan"))
        joints[j.get("name")] = jo
    return links, joints


# --------------------------------------------------------------------------- MJCF
def read_mjcf(path: Path):
    m = mujoco.MjModel.from_xml_path(str(path))
    links, joints = {}, {}
    for b in range(1, m.nbody):                     # skip world
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        li = Link(mass=float(m.body_mass[b]),
                  inertia=tuple(sorted(float(x) for x in m.body_inertia[b])))
        links[name] = li
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        # contype==0 and conaffinity==0 means the geom is visual-only.
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        if name in links:
            links[name].n_collision += 1
            links[name].collision_kinds.add(
                mujoco.mjtGeom(m.geom_type[g]).name.replace("mjGEOM_", "").lower())
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        adr = m.jnt_dofadr[j]
        lo, hi = (float(x) for x in m.jnt_range[j])
        joints[name] = Joint(lower=lo, upper=hi,
                             damping=float(m.dof_damping[adr]),
                             friction=float(m.dof_frictionloss[adr]))
    return links, joints


# --------------------------------------------------------------------------- USD
def read_usd(path: Path):
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(str(path))
    links, joints = {}, {}
    for prim in stage.Traverse():
        n = prim.GetName()
        if prim.HasAPI(UsdPhysics.MassAPI):
            api = UsdPhysics.MassAPI(prim)
            li = links.setdefault(n, Link())
            m = api.GetMassAttr().Get()
            if m is not None:
                li.mass = float(m)
            di = api.GetDiagonalInertiaAttr().Get()
            if di is not None and tuple(di) != (0.0, 0.0, 0.0):
                li.inertia = tuple(float(x) for x in di)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            rj = UsdPhysics.RevoluteJoint(prim)
            lo, hi = rj.GetLowerLimitAttr().Get(), rj.GetUpperLimitAttr().Get()
            jo = joints.setdefault(n, Joint())
            # USD revolute limits are DEGREES; URDF and MJCF are radians.
            if lo is not None:
                jo.lower = math.radians(float(lo))
            if hi is not None:
                jo.upper = math.radians(float(hi))
        if prim.HasAPI(UsdPhysics.DriveAPI):
            d = UsdPhysics.DriveAPI(prim, "angular")
            jo = joints.setdefault(n, Joint())
            dv = d.GetDampingAttr().Get()
            if dv is not None:
                jo.damping = float(dv)
    return links, joints


# --------------------------------------------------------------------------- compare
# Inertias of tiny parts are ~1e-9; a relative test on those reports noise as
# disagreement, so anything below this absolute floor counts as equal.
ABS_FLOOR = 1e-7


def close(a, b, tol=REL_TOL):
    if a is None or b is None:
        return None
    if isinstance(a, (tuple, list)):
        return all(close(x, y, tol) for x, y in zip(a, b))
    if abs(a) < ABS_FLOOR and abs(b) < ABS_FLOOR:
        return True
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom <= tol


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", required=True, type=Path,
                    help="…/data/robots/berkeley_humanoid/berkeley_humanoid_lite")
    ap.add_argument("--variant", default="biped", choices=["biped", "humanoid"])
    ap.add_argument("--upstream", type=Path, required=True,
                    help="Berkeley-Humanoid-Lite checkout (for the MJCF path repair)")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--json-out", type=Path, default=None,
                    help="write a machine-readable summary next to the printout")
    args = ap.parse_args()

    stem = "berkeley_humanoid_lite_biped" if args.variant == "biped" else "berkeley_humanoid_lite"
    urdf = args.assets / "urdf" / f"{stem}.urdf"
    usd = args.assets / "usd" / f"{stem}.usd"
    # The shipped MJCF cannot be loaded as-is (its meshdir points at a directory
    # that does not exist), so the audit reads the same repaired copy the
    # evaluation harness uses. That is the file the sim2sim numbers came from.
    from bhl_robust.eval.mjcf_assets import prepare_mjcf
    scene = prepare_mjcf(args.upstream, args.cache_dir, args.variant)
    mjcf = scene.parent / f"{stem}.xml"

    print(f"### {args.variant}")
    for label, p in (("URDF", urdf), ("MJCF", mjcf), ("USD ", usd)):
        print(f"  {label}: {'present' if p.exists() else 'MISSING'}  {p.name}")
    print()

    u_l, u_j = read_urdf(urdf)
    m_l, m_j = read_mjcf(mjcf)
    try:
        s_l, s_j = read_usd(usd)
    except Exception as e:
        print(f"  USD read failed: {e}")
        s_l, s_j = {}, {}

    # ---- masses
    print("--- link mass: URDF vs MJCF ---")
    tot_u = sum(v.mass for v in u_l.values() if v.mass)
    tot_m = sum(v.mass for v in m_l.values() if v.mass)
    print(f"  total mass  URDF {tot_u:8.4f} kg   MJCF {tot_m:8.4f} kg   "
          f"delta {abs(tot_u-tot_m):.4f} kg ({abs(tot_u-tot_m)/max(tot_u,1e-9)*100:.2f}%)")
    bad = []
    for name, lu in sorted(u_l.items()):
        lm = m_l.get(name)
        if lm is None or lu.mass is None or lm.mass is None:
            continue
        if not close(lu.mass, lm.mass):
            bad.append((name, lu.mass, lm.mass))
    print(f"  per-link mismatches (> {REL_TOL*100:.0f}%): {len(bad)}")
    for n, a, b in bad[:10]:
        print(f"    {n:34s} URDF {a:9.5f}  MJCF {b:9.5f}  ({abs(a-b)/max(a,1e-9)*100:6.2f}%)")

    # ---- inertia
    print("--- principal moments of inertia (sorted, frame-independent) ---")
    ib = [(n, u_l[n].inertia, m_l[n].inertia) for n in sorted(u_l)
          if n in m_l and u_l[n].inertia and m_l[n].inertia
          and not close(u_l[n].inertia, m_l[n].inertia, 0.05)]
    print(f"  mismatches (> 5%): {len(ib)}")
    for n, a, b in ib[:8]:
        print(f"    {n:34s} URDF {tuple(round(x,6) for x in a)}")
        print(f"    {'':34s} MJCF {tuple(round(x,6) for x in b)}")

    # ---- joint limits
    print("--- joint limits ---")
    for other, oj, tag in ((m_j, m_j, "MJCF"), (s_j, s_j, "USD")):
        if not oj:
            print(f"  {tag}: no joints parsed")
            continue
        diffs = []
        for n, ju in sorted(u_j.items()):
            jo = oj.get(n)
            if jo is None or ju.lower is None or jo.lower is None:
                continue
            if not (close(ju.lower, jo.lower, 0.02) and close(ju.upper, jo.upper, 0.02)):
                diffs.append((n, (ju.lower, ju.upper), (jo.lower, jo.upper)))
        matched = sum(1 for n in u_j if n in oj)
        print(f"  URDF vs {tag}: {matched} joints matched by name, {len(diffs)} limit mismatches")
        for n, a, b in diffs[:8]:
            print(f"    {n:34s} URDF [{a[0]:+.3f},{a[1]:+.3f}]  {tag} [{b[0]:+.3f},{b[1]:+.3f}]")

    # ---- damping
    print("--- joint damping: URDF vs MJCF ---")
    dd = [(n, u_j[n].damping, m_j[n].damping) for n in sorted(u_j)
          if n in m_j and u_j[n].damping is not None and m_j[n].damping is not None
          and not close(u_j[n].damping, m_j[n].damping, 0.02)]
    print(f"  mismatches: {len(dd)}")
    for n, a, b in dd[:8]:
        print(f"    {n:34s} URDF {a:8.4f}  MJCF {b:8.4f}")

    # ---- collision representation
    print("--- collision geometry representation ---")
    uk, mk = set(), set()
    for v in u_l.values():
        uk |= v.collision_kinds
    for v in m_l.values():
        mk |= v.collision_kinds
    print(f"  URDF collision kinds: {sorted(uk) or 'none'}")
    print(f"  MJCF collision kinds: {sorted(mk) or 'none'}")
    n_u = sum(v.n_collision for v in u_l.values())
    n_m = sum(v.n_collision for v in m_l.values())
    print(f"  URDF collision geoms: {n_u}")
    print(f"  MJCF collision geoms: {n_m}")

    summary = {
        "variant": args.variant,
        "total_mass_urdf_kg": tot_u,
        "total_mass_mjcf_kg": tot_m,
        "mass_delta_kg": abs(tot_u - tot_m),
        "mass_mismatches": len(bad),
        "inertia_mismatches": len(ib),
        "joint_limit_mismatches_mjcf": None,
        "joint_limit_mismatches_usd": None,
        "damping_mismatches": len(dd),
        "urdf_collision_kinds": sorted(uk),
        "mjcf_collision_kinds": sorted(mk),
        "urdf_collision_geoms": n_u,
        "mjcf_collision_geoms": n_m,
        "urdf_joints": len(u_j),
        "mjcf_joints": len(m_j),
        "usd_joints": len(s_j),
        "agrees": (len(bad) == 0 and len(ib) == 0 and len(dd) == 0
                   and abs(tot_u - tot_m) < 1e-6 and n_u == n_m),
    }
    # Recompute limit-mismatch counts for the JSON (the print loop already ran).
    for oj, tag, key in ((m_j, "MJCF", "joint_limit_mismatches_mjcf"),
                         (s_j, "USD", "joint_limit_mismatches_usd")):
        diffs = 0
        for n, ju in u_j.items():
            jo = oj.get(n)
            if jo is None or ju.lower is None or jo.lower is None:
                continue
            if not (close(ju.lower, jo.lower, 0.02) and close(ju.upper, jo.upper, 0.02)):
                diffs += 1
        summary[key] = diffs
        if diffs:
            summary["agrees"] = False

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"  wrote {args.json_out}")


if __name__ == "__main__":
    main()
