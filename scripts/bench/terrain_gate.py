"""G-B1 and G-B2: are the new terrains actually different from the baseline?

G-B1 is the one worth being pedantic about. `slippery` is a *negative control* --
the whole depth claim rests on it being difficulty a depth camera cannot see. If
the friction override silently failed, the arm would train as a copy of the
bumpy rung, come back with the same numbers, and read as "depth does not help on
slippery" when what actually happened is that slippery was never built. Section
5 lost three ablation arms to exactly that shape of failure.

So this resolves the constructed env and compares against the baseline rather
than trusting the config file.
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--check", choices=["friction", "stairs"], required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import berkeley_humanoid_lite.tasks  # noqa: F401,E402
import bhl_robust.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

OUT = os.environ.get("BENCH_OUT", "/tmp/terrain_gate.txt")
lines, ok = [], True


def log(s):
    lines.append(s)
    print(s, flush=True)


def friction_of(task):
    cfg = parse_env_cfg(task, device="cuda:0", num_envs=4)
    p = cfg.events.physics_material.params
    return tuple(p["static_friction_range"]), tuple(p["dynamic_friction_range"]), cfg


if args.check == "friction":
    base_s, base_d, _ = friction_of("Velocity-BHL-Biped-Bumpy-v0")
    slip_s, slip_d, slip_cfg = friction_of("Velocity-BHL-Biped-Slippery-v0")
    log(f"CHECK | bumpy    static={base_s} dynamic={base_d}")
    log(f"CHECK | slippery static={slip_s} dynamic={slip_d}")

    differs = slip_s != base_s
    below = slip_s[1] < base_s[0]          # whole range under upstream's floor
    dyn_lower = slip_d[1] <= slip_s[1]     # sliding friction under breakaway
    for good, msg in (
        (differs, "slippery friction differs from bumpy"),
        (below, f"slippery max {slip_s[1]} is below bumpy min {base_s[0]} -- a different regime, not the tail of the same one"),
        (dyn_lower, "dynamic friction does not exceed static"),
    ):
        log(("PASS  | " if good else "FAIL  | ") + msg)
        ok = ok and good

    # Geometry must be untouched: same generator object, same sub-terrains.
    depth_slip = parse_env_cfg("Velocity-BHL-Biped-Slippery-Depth-v0", device="cuda:0", num_envs=4)
    same_gen = (sorted(slip_cfg.scene.terrain.terrain_generator.sub_terrains)
                == sorted(depth_slip.scene.terrain.terrain_generator.sub_terrains))
    log(("PASS  | " if same_gen else "FAIL  | ") + "slippery and slippery-depth share one terrain menu")
    ok = ok and same_gen

else:
    cfg = parse_env_cfg("Velocity-BHL-Biped-Stairs-v0", device="cuda:0", num_envs=4)
    subs = cfg.scene.terrain.terrain_generator.sub_terrains
    log(f"CHECK | stairs sub-terrains: {sorted(subs)}")
    LEG = 0.28
    for name, sub in subs.items():
        hi = sub.step_height_range[1]
        log(f"CHECK | {name}: riser 0 -> {hi*100:.1f} cm ({hi/LEG*100:.0f}% leg), "
            f"tread {sub.step_width*100:.0f} cm")
        for good, msg in (
            (hi <= 0.03, f"{name} riser {hi*100:.1f} cm within the 3 cm ceiling G-B2 settled on"),
            (sub.step_height_range[0] == 0.0, f"{name} starts at zero, so difficulty 0 is flat"),
            (sub.step_width >= 0.30, f"{name} tread {sub.step_width*100:.0f} cm fits a foot"),
        ):
            log(("PASS  | " if good else "FAIL  | ") + msg)
            ok = ok and good
    has_both = len(subs) == 2 and any("down" in k for k in subs)
    log(("PASS  | " if has_both else "FAIL  | ") + "both ascent and descent present")
    ok = ok and has_both

log(f"VERDICT | {'ALL PASS' if ok else 'FAILURES ABOVE'}")
with open(OUT, "a") as f:
    f.write("\n".join(lines) + "\n")
app.close()
