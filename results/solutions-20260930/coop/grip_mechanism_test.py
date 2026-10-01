"""Mechanism test (exploration seeds >= 100 only; never a scored seed): does the cube's ~90 deg
roll in the scripted pinch come from the absence of torsional grip at the pad-cube contact?

Same harness as results/scripted-carry-20260926 lift_hold (scripted_carry.run_episode,
protocol lift_hold, crew 2, frozen arms-dr1.0-s0 gait, kp-30 grasping arm, 4 Nm cap), with ONE
modelling change per variant, applied to the hand pads only:
  base      : as published (condim 3 = sliding friction only; mu 1.0 vs cube 1.2 -> 1.2)
  tors02    : pads condim 4, torsional friction 0.02 m  (~ mu x mean radius of a 3-4 cm patch)
  tors04    : pads condim 4, torsional friction 0.04 m  (~ a flush 6x13 cm pad face)
Reports lift, longest >= 5 cm hold, cube tilt, contact point height relative to the cube
centre, normal forces, floor contact.  Scratch outputs only.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "bench"))
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from bhl_robust.eval import scripted_carry as sc  # noqa: E402
from team_airlock import CpuPolicy  # noqa: E402

OUT = Path(__file__).resolve().parent
U = REPO / "external" / "Berkeley-Humanoid-Lite"
DEPLOY = U / "logs/rsl_rl/humanoid/2026-08-18_20-57-50_arms-dr1.0-s0/exported/deploy.yaml"
_orig_add = sc.add_hand_pads


def make_pad_patch(condim, tors, slide=None):
    def patched(spec, frames, friction, prefix=""):
        _orig_add(spec, frames, friction if slide is None else slide, prefix)
        for side in frames:
            body = spec.body(f"{prefix}arm_{side}_hand_link")
            for g in body.geoms:
                if g.name.endswith("_hand_pad"):
                    g.condim = condim
                    g.friction = [friction if slide is None else slide, tors, 0.0001]
    return patched


VARIANTS = {
    "base": None,
    "tors02": (4, 0.02),
    "tors04": (4, 0.04),
    "base_ell": None,          # solver: elliptic cone + impratio 10 (MuJoCo's grasping advice)
    "tors04_ell": (4, 0.04),
    "base_noslip": None,       # solver: noslip_iterations 10 (removes soft-contact creep)
}
SOLVER = {"base_ell": {"cone": "elliptic", "impratio": 10.0}, "tors04_ell": {"cone": "elliptic", "impratio": 10.0},
          "base_noslip": {"noslip": 10}}


def run(variant, seeds, seconds=20.0):
    if VARIANTS[variant] is None:
        sc.add_hand_pads = _orig_add
    else:
        sc.add_hand_pads = make_pad_patch(*VARIANTS[variant])
    cfg = OmegaConf.load(DEPLOY)
    policy = CpuPolicy(cfg.policy_checkpoint_path)
    p = sc.CarryParams()
    model, slots, pairs = sc.build_carry(U, OUT / "mjcf_cache_carry", 1, p)
    so = SOLVER.get(variant, {})
    if so.get("cone") == "elliptic":
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    if "impratio" in so:
        model.opt.impratio = so["impratio"]
    if "noslip" in so:
        model.opt.noslip_iterations = so["noslip"]
    pad_ids = [g for g in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("_hand_pad")]
    pads_info = [{"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g), "condim": int(model.geom_condim[g]),
                  "friction": model.geom_friction[g].round(4).tolist()} for g in pad_ids]
    pr = pairs[0]
    rows = []
    for seed in seeds:
        log = []

        def hook(step, t, runner, **kw):
            if step % 5:
                return
            d = runner.d
            c = d.xpos[pr.cube_body].copy()
            R = d.xmat[pr.cube_body].reshape(3, 3)
            pts = []
            f6 = np.zeros(6)
            for k in range(d.ncon):
                con = d.contact[k]
                if pr.cube_geom not in (con.geom1, con.geom2):
                    continue
                other = con.geom2 if con.geom1 == pr.cube_geom else con.geom1
                if other in pad_ids or runner.owner[other] >= 0:
                    mujoco.mj_contactForce(model, d, k, f6)
                    rel = R.T @ (con.pos - c)            # contact point in the cube frame
                    pts.append({"z_rel_world": round(float(con.pos[2] - c[2]), 4), "rel_cube": np.round(rel, 3).tolist(),
                                "fn": round(float(f6[0]), 2), "pad": bool(other in pad_ids)})
            log.append({"t": round(float(t), 2), "lift": round(float(kw["lift"][0]), 4),
                        "tilt": round(runner.cube_tilt(pr), 3), "cube": np.round(c, 3).tolist(), "contacts": pts})

        t0 = time.time()
        res = sc.run_episode(model, slots, pairs, cfg, policy, seed, p, frame_hook=hook,
                             rule=dict(sc.LIFT_HOLD_RULE, episode_s=seconds), protocol="lift_hold")
        wall = time.time() - t0
        row = res["pairs"][0]
        tilts = np.array([q["tilt"] for q in log])
        # contact geometry during the lift / early hold (t 5-9 s)
        zrel = [c["z_rel_world"] for q in log if 5.0 <= q["t"] <= 9.0 for c in q["contacts"] if c["pad"]]
        ncon = [len([c for c in q["contacts"] if c["pad"]]) for q in log if 5.0 <= q["t"] <= 9.0]
        fn = [sum(c["fn"] for c in q["contacts"]) for q in log if 5.0 <= q["t"] <= 9.0]
        rows.append({"variant": variant, "seed": seed, "wall_s": round(wall, 1),
                     "lift_peak_m": row["lift_peak_m"], "lift_hold_s": row["lift_hold_s"],
                     "hold_end_s": row.get("hold_end_s"), "floor": row["cube_floor_contact"],
                     "max_robot_tilt": row["max_tilt_rad"], "success_lift_hold_rule": row["success"],
                     "cube_tilt_at_t": {f"{tt:.0f}s": float(tilts[min(len(tilts) - 1, int(tt / 0.2))]) for tt in (5, 6, 7, 8, 10, 12, 15, 19)},
                     "max_cube_tilt_rad_before_9s": float(tilts[: int(9 / 0.2)].max()),
                     "pad_contact_z_rel_cube_centre_5to9s": None if not zrel else
                         {"p10": round(float(np.percentile(zrel, 10)), 3), "p50": round(float(np.median(zrel)), 3),
                          "p90": round(float(np.percentile(zrel, 90)), 3)},
                     "pad_contact_points_per_frame_5to9s_median": float(np.median(ncon)) if ncon else 0,
                     "total_normal_N_5to9s_median": round(float(np.median(fn)), 2) if fn else 0.0})
        print(json.dumps(rows[-1]), flush=True)
    return {"variant": variant, "pads": pads_info, "solver": {"cone": int(model.opt.cone), "impratio": float(model.opt.impratio), "noslip_iterations": int(model.opt.noslip_iterations)}, "rows": rows}


if __name__ == "__main__":
    variants = sys.argv[1].split(",") if len(sys.argv) > 1 else list(VARIANTS)
    seeds = [int(s) for s in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["100", "101", "102"])]
    for s in seeds:
        assert s >= 100, "exploration seeds only"
    out = [run(v, seeds) for v in variants]
    tag = "_".join(variants) + "_" + "-".join(map(str, seeds))
    (OUT / f"grip_mechanism_{tag}.json").write_text(json.dumps(out, indent=1))
