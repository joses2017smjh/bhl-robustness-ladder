#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Does the v2 spawn put the robot through the floor?

The Isaac replay shows both robots flat on the ground within a second and
staying there, which is not what ~8-step episodes should look like -- at 8 steps
a 300-step clip should reset them upright dozens of times. `_PINCH_ROOT_Z` is
-0.07, and the shipped asset spawns at z = 0.0 with its root frame at ground
level, so the crouch pose is applied 7 cm *below* the plane.

This measures it instead of arguing about it: where every body starts, how far
the lowest one is under z = 0, and what the solver does about it over the first
few steps with a zero action. Depenetration at max_depenetration_velocity = 1.0
is what launched the plank payload 22 cm, so the same question is asked here.
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args([]); a.headless = True; a.enable_cameras = True
app = AppLauncher(a)

import gymnasium as gym, torch
import bhl_robust.tasks  # noqa: F401
from bhl_robust.tasks.coop_lift_mdp import _t

# The control. "19 of 27 bodies below z = 0" is a fact about this asset's frame
# convention until a task that demonstrably works reports a smaller number --
# the same mistake G-B2 made when it measured its own iteration budget and
# called it a terrain verdict.
# Candidate spawn quaternions, swept in-task. This probe's burial counts are the
# only Isaac numbers in this investigation that matched what the render shows --
# it said 27 of 27 below ground and the video's first frames are empty -- so it
# is the one to sweep with.
#
# MuJoCo puts this robot's bodies up to 0.77 m *above* the root; Isaac puts them
# 0.78 m below it. Same magnitude, opposite sign, so the USD wants a 180 degree
# flip on top of the yaw the config already has.
import itertools as _it
QUATS = {
    "stand only   180y": (0.0, 0.0, 1.0, 0.0),
    "180y then yaw  90": (0.0, -0.70710678, 0.70710678, 0.0),
    "180y then yaw 180": (0.0, -1.0, 0.0, 0.0),
    "180y then yaw 270": (0.0, -0.70710678, -0.70710678, 0.0),
}

TASKS = [
    ("v2 CubeToShelf (suspect)", "TaskV2-BHL-CubeToShelf-Blind-v0", ("robot_a", "robot_b")),
    ("22-DoF locomotion (control)", "Velocity-BHL-Arms-PushAdaptive-v0", ("robot",)),
]

def report(tag, u, robots):
    print(f"\n=== {tag} ===")
    for name in robots:
        r = u.scene[name]
        root = _t(r.data.root_pos_w)
        bodies = _t(r.data.body_pos_w)            # (envs, bodies, 3)
        # env origins are added into world pos; subtract to get z above terrain
        zmin, zi = bodies[..., 2].min(dim=1)
        names = r.body_names
        print(f"  {name}: root_z={root[0,2]:+.4f}  "
              f"lowest body={names[zi[0]]!r} at z={zmin[0]:+.4f}")
        under = (bodies[..., 2] < 0.0).sum(dim=1)
        print(f"          bodies below z=0: {under.tolist()} of {len(names)}")
        # Where the design says these should be: feet on the floor (z~0) and
        # hands spanning 0.404-0.610 standing, reaching down to GRASP_Z=0.30.
        for key in ("hand", "foot", "ankle"):
            idx = [i for i, n in enumerate(names) if key in n.lower()]
            if idx:
                zs = bodies[0, idx, 2]
                print(f"          {key:6}: " + "  ".join(
                    f"{names[i]}={bodies[0,i,2]:+.3f}" for i in idx[:2]))

# Sweep the candidates on the suspect task before the usual two-task report.
import gymnasium as _gym
print(f"\n{'quaternion':32} {'below':>7} {'ankle':>8} {'shoulder':>9} {'base':>8}  verdict")
for qlabel, rot in QUATS.items():
    try:
        cfg = _gym.spec(TASKS[0][1]).kwargs["env_cfg_entry_point"]()
        cfg.scene.num_envs = 2
        for r in (cfg.scene.robot_a, cfg.scene.robot_b):
            r.init_state = r.init_state.replace(rot=rot)
        e = _gym.make(TASKS[0][1], cfg=cfg, disable_env_checker=True)
        e.reset(); uu = e.unwrapped
        rr = uu.scene["robot_a"]; rr.update(dt=0.0)
        rb = uu.scene["robot_b"]; rb.update(dt=0.0)
        nm = rr.body_names
        b = _t(rr.data.body_pos_w)[0]
        org = uu.scene.env_origins[0]
        h = lambda k: float(b[[i for i, n in enumerate(nm) if k in n], 2].mean() - org[2])
        below = int(((b[:, 2] - org[2]) < 0).sum())
        ank, sho, bas = h("ankle_roll"), h("shoulder_pitch"), h("base")
        # facing: the two robots must look at each other across the payload,
        # so their base-to-hand vectors should oppose in y.
        bb = _t(rb.data.body_pos_w)[0]
        nmb = rb.body_names
        hy = lambda arr, nn, k: float(arr[[i for i, n in enumerate(nn) if k in n], 1].mean())
        fa = hy(b, nm, "hand_link") - hy(b, nm, "base")
        fb = hy(bb, nmb, "hand_link") - hy(bb, nmb, "base")
        good = below <= 2 and ank < sho
        print(f"{qlabel:32} {below:3d}/{len(nm):<3d} {ank:8.3f} {sho:9.3f} {bas:8.3f}"
              f"  {'STANDS' if good else 'no':6}  face_a {fa:+.3f} face_b {fb:+.3f}")
    except Exception as exc:
        print(f"{qlabel:32}  failed: {str(exc)[:44]}")
    finally:
        try:
            e.close()
        except Exception:
            pass
print("\nMuJoCo reference: ankle +0.140, shoulder +0.737, base -0.027, 1 of 26 below.\n")

for label, task, robots in TASKS:
    print(f"\n{'#'*66}\n# {label}: {task}\n{'#'*66}")
    if task not in gym.registry:
        print(f"  not registered -- skipped"); continue
    try:
        cfg = gym.spec(task).kwargs["env_cfg_entry_point"]()
        cfg.scene.num_envs = 4
        env = gym.make(task, cfg=cfg, disable_env_checker=True)
        env.reset()
        u = env.unwrapped
        report("at reset, before any step", u, robots)
        if "object" in u.scene.keys():
            print(f"  object z at reset: {_t(u.scene['object'].data.root_pos_w)[0,2]:+.4f}"
                  f"   (GRASP_Z = 0.30)")
        zero = torch.zeros((4, u.action_space.shape[-1]), device=u.device)
        for i in range(1, 11):
            env.step(zero)
            if i in (1, 2, 5, 10):
                report(f"after step {i} (zero action)", u, robots)
        print("\n  termination terms after 10 zero-action steps:")
        tm = u.termination_manager
        for term in tm.active_terms:
            print(f"    {term:14} {tm.get_term(term).float().mean().item():.4f}")
        env.close()
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"  FAILED: {exc!r}")
app.app.close()
PY
