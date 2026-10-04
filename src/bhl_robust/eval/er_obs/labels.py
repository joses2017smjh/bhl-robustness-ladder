"""ORACLE labels of ER-OBS-1 (pure numpy kernels + the read-only MuJoCo extraction).

Clauses as applied: `bhl_robust.eval.er_obs.CLAUSES_AS_APPLIED` (labels (1)-(4) and C2). The kernels take plain
arrays so the tests can put a synthetic cube exactly on each boundary (2 cm, 8 deg, 1 N, any face down). The
extraction reads one MuJoCo snapshot and never writes to the simulation's data.
"""

from __future__ import annotations

import math

import numpy as np

from bhl_robust.eval.er_obs import (C2_LIFT_CENTRE_M, C2_PLACE_HEIGHT_TOL_M, CONTACT_FORCE_N, LIFT_CLEARANCE_M,
                                    PLINTH_HALF_M, SEAT_TILT_MAX_DEG)


# ------------------------------------------------------------------ pure kernels

def cube_half_extents(R, half: float) -> np.ndarray:
    """(hx, hy, hz): half-extents of the rotated cube's world AABB. R is the body-to-world rotation (MuJoCo xmat,
    row-major 3x3): along world axis i the cube reaches half * sum_j |R[i, j]| from its centre (Stand4's
    `cube_half_extents` with R from the quaternion)."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    return float(half) * np.abs(R).sum(axis=1)


def lowest_corner_z(centre, R, half: float) -> float:
    """World z of the cube's lowest corner: centre z - half * (|R20| + |R21| + |R22|)."""
    return float(np.asarray(centre, dtype=float)[2] - cube_half_extents(R, half)[2])


def support_clearance(centre, R, half: float, supports, floor_z: float = 0.0) -> float:
    """Lowest corner minus the highest support surface the cube could rest on (Stand4's `support_clearance`):
    the floor always; each raised support (x0, x1, y0, y1, top) whose footprint strictly overlaps the world-xy AABB
    of the cube's corners (touching edges do not count)."""
    c = np.asarray(centre, dtype=float)
    hx, hy, _ = cube_half_extents(R, half)
    low = lowest_corner_z(c, R, half)
    clear = low - float(floor_z)
    for x0, x1, y0, y1, top in supports:
        under = (c[0] + hx > x0) and (c[0] - hx < x1) and (c[1] + hy > y0) and (c[1] - hy < y1)
        if under:
            clear = min(clear, low - float(top))
    return float(clear)


def face_tilt_deg(R) -> float:
    """Minimum over the cube's six body-axis directions (+-x, +-y, +-z) of the angle to world up, in degrees:
    arccos(max_i |R[2, i]|). 0 for any face flat down; a 90-deg roll onto another face also reads 0."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    return float(math.degrees(math.acos(min(1.0, max(-1.0, float(np.abs(R[2, :]).max()))))))


def body_z_tilt_deg(R) -> float:
    """Stand4's strict tilt: the angle between the cube's OWN z axis and world up, arccos(R[2, 2]), in degrees
    (a cube rolled 90 deg onto another face reads 90)."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    return float(math.degrees(math.acos(min(1.0, max(-1.0, float(R[2, 2]))))))


def lifted_clear(clearance_m: float) -> bool:
    """(1) lowest corner >= 2 cm above every support."""
    return bool(clearance_m >= LIFT_CLEARANCE_M)


def robot_contact(robot_force_n: float) -> bool:
    """(3) summed robot normal force on the cube > 1 N (strict)."""
    return bool(robot_force_n > CONTACT_FORCE_N)


def over_plinth(offset_xy, half: float = PLINTH_HALF_M) -> bool:
    """Cube centre over the plinth top: |dx|, |dy| <= 0.09 m from the plinth centre."""
    dx, dy = (float(v) for v in offset_xy)
    return bool(max(abs(dx), abs(dy)) <= half)


def seated_flat(plinth_contact: bool, offset_xy, tilt_deg: float) -> bool:
    """(4) cube-plinth contact AND centre over the plinth top AND tilt <= 8 deg. With `face_tilt_deg` this is the
    any-face-down label; with `body_z_tilt_deg` it is Stand4's strict variant (reported, never scored)."""
    return bool(plinth_contact) and over_plinth(offset_xy) and bool(tilt_deg <= SEAT_TILT_MAX_DEG)


def c2_lifted(dz_m: float) -> bool:
    """C2 status quo: cube centre >= 0.05 m above the harness's rest height (LIFT_PLACE_RULE lift_hold_m)."""
    return bool(dz_m >= C2_LIFT_CENTRE_M)


def c2_placed(dz_m: float, offset_xy) -> bool:
    """C2 status quo: summarize_lift_place's placed_on_plinth (|dz| <= 0.03 m and centre over the plinth top);
    no orientation clause, no contact clause."""
    return bool(abs(float(dz_m)) <= C2_PLACE_HEIGHT_TOL_M) and over_plinth(offset_xy)


# ------------------------------------------------------------------ MuJoCo extraction (read-only)

def plinth_geometry(model, plinth_geom: int) -> dict:
    """Centre xy, half-width and top z of the box plinth, read from the compiled model."""
    pos = np.asarray(model.geom_pos[plinth_geom], dtype=float)
    size = np.asarray(model.geom_size[plinth_geom], dtype=float)
    return {"xy": [float(pos[0]), float(pos[1])], "half": float(size[0]), "half_y": float(size[1]),
            "top": float(pos[2] + size[2])}


def contact_partners(model, data, cube_geom: int) -> set:
    """Geom ids touching the cube in MuJoCo's current contact list (no recomputation)."""
    out = set()
    for k in range(int(data.ncon)):
        c = data.contact[k]
        if int(c.geom1) == cube_geom:
            out.add(int(c.geom2))
        elif int(c.geom2) == cube_geom:
            out.add(int(c.geom1))
    return out


def frame_state(model, data, runner, pair, floor_geom: int) -> dict:
    """Every number and strict label of one snapshot. Reads `data` only; the robot forces come from the harness's
    own `CarryRunner.cube_forces` (mj_contactForce, read-only)."""
    cube_geom = int(pair.cube_geom)
    half = float(model.geom_size[cube_geom][0])
    centre = np.asarray(data.xpos[pair.cube_body], dtype=float).copy()
    R = np.asarray(data.xmat[pair.cube_body], dtype=float).reshape(3, 3).copy()
    pl = plinth_geometry(model, int(pair.plinth_geom))
    supports = [(pl["xy"][0] - pl["half"], pl["xy"][0] + pl["half"], pl["xy"][1] - pl["half_y"],
                 pl["xy"][1] + pl["half_y"], pl["top"])]
    floor_z = float(model.geom_pos[floor_geom][2])
    clearance = support_clearance(centre, R, half, supports, floor_z)
    partners = contact_partners(model, data, cube_geom)
    floor_contact = floor_geom in partners
    plinth_contact = int(pair.plinth_geom) in partners
    robot_geoms = sorted(g for g in partners if runner.owner[g] >= 0)
    forces = runner.cube_forces(pair)
    robot_force = float(forces["a"]) + float(forces["b"])
    offset = [float(centre[0] - pl["xy"][0]), float(centre[1] - pl["xy"][1])]
    ft, bz = face_tilt_deg(R), body_z_tilt_deg(R)
    labels = {
        "lifted_clear": lifted_clear(clearance),
        "on_floor": bool(floor_contact),
        "robot_contact": robot_contact(robot_force),
        "seated_flat": seated_flat(plinth_contact, offset, ft),
    }
    strict = seated_flat(plinth_contact, offset, bz)
    return {
        "labels": labels,
        "seated_flat_strict_body_z": bool(strict),
        "body_z_disagrees": bool(strict != labels["seated_flat"]),
        "state": {
            "cube_centre_m": [round(float(v), 6) for v in centre],
            "cube_xmat": [round(float(v), 6) for v in R.ravel()],
            "lowest_corner_z_m": round(lowest_corner_z(centre, R, half), 6),
            "support_clearance_m": round(clearance, 6),
            "face_tilt_deg": round(ft, 4),
            "body_z_tilt_deg": round(bz, 4),
            "offset_from_plinth_xy_m": [round(v, 6) for v in offset],
            "floor_contact": bool(floor_contact),
            "plinth_contact": bool(plinth_contact),
            "robot_contact_geoms": len(robot_geoms),
            "robot_normal_force_n": round(robot_force, 4),
            "normal_force_n": {k: round(float(v), 4) for k, v in forces.items()},
        },
        # raw values the C2 post-pass needs (the rest height is known only after settling)
        "_z": float(centre[2]), "_offset": offset,
    }
