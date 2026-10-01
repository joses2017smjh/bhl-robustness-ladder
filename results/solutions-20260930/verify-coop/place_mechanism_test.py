"""lift_place (scripted_carry.run_place_episode, PlaceParams frozen defaults) on exploration seeds
>= 100 with the pad/solver variants of grip_mechanism_test.py. Scratch only; never a scored seed."""
import json, os, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
import grip_mechanism_test as g
import mujoco
import numpy as np
from omegaconf import OmegaConf
sc = g.sc

def run(variant, seeds, lower='tuned'):
    sc.add_hand_pads = g._orig_add if g.VARIANTS[variant] is None else g.make_pad_patch(*g.VARIANTS[variant])
    cfg = OmegaConf.load(g.DEPLOY); policy = g.CpuPolicy(cfg.policy_checkpoint_path)
    p = sc.CarryParams()
    model, slots, pairs = sc.build_carry(g.U, g.OUT / "mjcf_cache_carry", 1, p)
    so = g.SOLVER.get(variant, {})
    if so.get("cone") == "elliptic": model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    if "impratio" in so: model.opt.impratio = so["impratio"]
    rows = []
    for s in seeds:
        assert s >= 100
        t0 = time.time()
        q = sc.PlaceParams() if lower == 'tuned' else sc.PlaceParams(lower_to=None if lower == 'reverse' else tuple(float(x) for x in lower.split(':')))
        res = sc.run_place_episode(model, slots, pairs, cfg, policy, s, p, q=q, rule=dict(sc.LIFT_PLACE_RULE, seeds=[s]))
        r = res["pairs"][0]
        rows.append({"variant": variant, "lower": lower, "seed": s, "success": r["success"], "first_failed": r["first_failed_check"],
                     "lift_peak_m": r["lift_peak_m"], "hold_s": r["lift_hold_s"], "final_dz_m": r["final_dz_m"],
                     "final_offset_xy_m": r["final_offset_xy_m"], "final_cube_tilt_rad": r["final_cube_tilt_rad"],
                     "max_robot_tilt": r["max_tilt_rad"], "floor": r["cube_floor_contact"],
                     "floor_first": r["floor_contact_first"], "wall_s": round(time.time() - t0, 1)})
        print(json.dumps(rows[-1]), flush=True)
    return rows

if __name__ == "__main__":
    v = sys.argv[1]; seeds = [int(x) for x in sys.argv[2].split(",")]; lower = sys.argv[3] if len(sys.argv) > 3 else 'tuned'
    rows = run(v, seeds, lower)
    json.dump(rows, open(g.OUT / f"place_mechanism_{v}_{lower.replace(':','_')}_{'-'.join(map(str, seeds))}.json", "w"), indent=1)
