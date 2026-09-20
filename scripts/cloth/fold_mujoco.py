#!/usr/bin/env python3
"""Bounded deformable-towel fold diagnostic with two idealized Cartesian pickers.

This is a cloth-physics/controller gate, not a gripper or humanoid benchmark.
Both picker attachments release before scoring. There is no cloth teleport,
fixed cloth vertex, or scoring from a prescribed target configuration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np


def model_xml(n=9, width=0.18, mass=0.016, friction=0.8, dt=0.001):
    if n < 5 or n % 2 == 0:
        raise ValueError("odd resolution >=5 required to represent the crease")
    h, z = width / 2, 0.3045
    return f"""<mujoco model="released_towel_fold">
      <option timestep="{dt}" integrator="implicitfast" solver="Newton" iterations="50">
        <flag autoreset="disable"/>
      </option>
      <size memory="128M"/>
      <default><joint damping="0.0001"/></default>
      <worldbody>
        <geom name="table" type="box" pos="0 0 0.285" size="0.3 0.3 0.015"
              friction="{friction} 0.005 0.0001" rgba="0.35 0.30 0.25 1"/>
        <light pos="0 0 1"/>
        <body name="picker_l" mocap="true" pos="{-h} {-h} {z}">
          <geom type="sphere" size="0.004" contype="0" conaffinity="0" rgba="1 0.2 0.2 1"/>
        </body>
        <body name="picker_r" mocap="true" pos="{-h} {h} {z}">
          <geom type="sphere" size="0.004" contype="0" conaffinity="0" rgba="0.2 0.2 1 1"/>
        </body>
        <flexcomp name="towel" type="grid" count="{n} {n} 1" spacing="{width/(n-1)} {width/(n-1)} 1"
                  pos="0 0 {z}" dim="2" mass="{mass}" radius="0.0015" rgba="0.2 0.65 0.75 1">
          <edge equality="true" damping="0.001" solref="0.008 1"/>
          <elasticity young="1000" poisson="0.2" thickness="0.0003" elastic2d="bend"/>
          <contact selfcollide="auto" internal="false" friction="{friction} 0.005 0.0001" solref="0.006 1"/>
        </flexcomp>
      </worldbody>
      <equality>
        <connect name="grasp_l" body1="towel_0" body2="picker_l" anchor="0 0 0" solref="0.004 1"/>
        <connect name="grasp_r" body1="towel_{n-1}" body2="picker_r" anchor="0 0 0" solref="0.004 1"/>
      </equality>
    </mujoco>"""


def fold_metrics(vertices, n, width, released, warning_count):
    finite = bool(np.isfinite(vertices).all())
    if not finite:
        return {"finite": False, "success": False}
    grid = vertices.reshape(n, n, 3)
    pairs = grid[:n//2, :, :2] - grid[:n//2:-1, :, :2]
    pair_rmse = float(np.sqrt(np.mean(np.sum(pairs * pairs, axis=-1))))
    footprint = np.ptp(vertices[:, :2], axis=0)
    height = float(np.max(vertices[:, 2]) - 0.3)
    # All mirrored material points must align; collapsed crumples cannot pass
    # the preserved-width/half-length checks. Cloth must rest on the table.
    success = bool(released and warning_count == 0 and pair_rmse < 0.025
                   and 0.30 * width < footprint[0] < 0.70 * width
                   and 0.75 * width < footprint[1] < 1.2 * width
                   and height < 0.035 and vertices[:, 2].min() > 0.295)
    return {"finite": finite, "success": success, "released": bool(released),
            "mirror_pair_rmse_m": pair_rmse, "footprint_xy_m": footprint.tolist(),
            "max_height_above_table_m": height, "warning_count": warning_count}


def run(seed=0, resolution=9, control="fold", trace=None):
    rng = np.random.default_rng(seed)
    width = float(rng.uniform(0.16, 0.20))
    friction = float(rng.uniform(0.7, 1.0))
    mass = float(rng.uniform(0.012, 0.020))
    model = mujoco.MjModel.from_xml_string(model_xml(resolution, width, mass, friction))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial = data.flexvert_xpos.copy()
    # Resolve actual corner coordinates from compiled model, avoiding an assumed
    # grid ordering if MuJoCo changes its macro expansion.
    grasp_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
                          for name in ("grasp_l", "grasp_r")])
    start = initial[[0, resolution-1]].copy()
    data.mocap_pos[:] = start
    samples, began, finite = [], time.monotonic(), True
    duration, settle, release = 5.0, 0.5, 3.5
    for i in range(round(duration / model.opt.timestep)):
        t = i * model.opt.timestep
        if control == "hold":
            data.eq_active[grasp_ids] = False
        elif t >= release:
            data.eq_active[grasp_ids] = False
        else:
            phase = np.clip((t - settle) / (release - settle - 0.5), 0, 1)
            smooth = phase * phase * (3 - 2 * phase)
            theta = np.pi * smooth
            data.mocap_pos[:] = start
            data.mocap_pos[:, 0] = -width / 2 * np.cos(theta)
            data.mocap_pos[:, 2] = start[:, 2] + width / 2 * np.sin(theta) + 0.003 * smooth
        mujoco.mj_step(model, data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                and np.isfinite(data.flexvert_xpos).all()):
            finite = False
            break
        if i % 50 == 0:
            samples.append(data.flexvert_xpos.copy())
    warning_count = sum(int(w.number) for w in data.warning)
    result = fold_metrics(data.flexvert_xpos, resolution, width,
                          not bool(data.eq_active[grasp_ids].any()), warning_count)
    result.update({"seed": seed, "control": control, "width_m": width, "friction": friction,
                   "mass_kg": mass, "resolution": resolution, "simulated_s": float(data.time),
                   "wall_s": time.monotonic()-began, "finite": finite and result["finite"],
                   "physics": f"MuJoCo {mujoco.__version__} 2D flex with self collision",
                   "actuation": "two idealized Cartesian attachments; no robot/gripper model",
                   "max_vertex_displacement_m": float(np.linalg.norm(data.flexvert_xpos-initial, axis=1).max())})
    if trace:
        np.savez_compressed(trace, vertices=np.asarray(samples), initial=initial,
                            final=data.flexvert_xpos.copy(), width=width, sample_dt=0.05)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resolution", type=int, default=9)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.episodes < 1:
        p.error("episodes must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.episodes):
        for control in ("hold", "fold"):
            row = run(args.seed+i, args.resolution, control,
                      args.out.with_name(f"{args.out.stem}_{i}_{control}.npz"))
            rows.append(row)
            print(json.dumps(row), flush=True)
            args.out.write_text(json.dumps({"rows": rows, "completed": False}, indent=2, allow_nan=False))
    finite = all(r["finite"] and r["warning_count"] == 0 for r in rows)
    positive = [r for r in rows if r["control"] == "fold"]
    negative = [r for r in rows if r["control"] == "hold"]
    gate = finite and any(r["success"] for r in positive) and not any(r["success"] for r in negative)
    args.out.write_text(json.dumps({"rows": rows, "completed": True, "gate_pass": gate,
        "fold_success_rate": float(np.mean([r["success"] for r in positive])),
        "scope": "physics diagnostic with idealized pickers, not a humanoid folding result"}, indent=2, allow_nan=False))
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
