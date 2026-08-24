"""Where does the pair's observation group survive into a crew config?

Four gate attempts died on `policy/object_pos_a: scene entity 'robot_a' does not
exist`, on a config whose class declares its own observations. Two hypotheses
remain and they need opposite fixes, so this measures which one is true instead
of inferring it from a stack trace for a fifth time:

  A. clean at plain instantiation, dirty after Hydra  -> the to_dict/from_dict
     round trip is reintroducing the parent's field.
  B. dirty at plain instantiation                     -> configclass inheritance
     is merging the parent's `observations` rather than replacing it, and no
     restructuring of the subclass will help; the crew configs must not inherit
     CoopLiftEnvCfg at all.

Run as a job, not in a shell: the two previous attempts at this were killed with
their sessions before reporting.
"""
import os

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import berkeley_humanoid_lite.tasks  # noqa: F401,E402
import bhl_robust.tasks  # noqa: F401,E402
from bhl_robust.tasks.coop_crew_generated import Crew3Cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

OUT = os.environ.get("BENCH_OUT", "/tmp/crew_diag.txt")
lines = []


def look(tag, cfg):
    o = cfg.observations
    pol = o.policy
    names = sorted(k for k, v in vars(pol).items() if v is not None and not k.startswith("_"))
    pair = [n for n in names if n.endswith(("_a", "_b"))]
    crew = [n for n in names if n[-1].isdigit()]
    lines.append(f"{tag}: observations={type(o).__name__} policy={type(pol).__name__}")
    lines.append(f"{tag}: {len(names)} terms | pair-style {pair[:4]} | crew-style {crew[:4]}")
    lines.append(f"{tag}: VERDICT {'DIRTY (pair terms present)' if pair else 'CLEAN'}")
    return bool(pair)


dirty_plain = look("A plain Crew3Cfg()", Crew3Cfg())
dirty_parsed = look("B parse_env_cfg", parse_env_cfg("CoopLift-BHL-Cube-Crew3-v0",
                                                     device="cuda:0", num_envs=4))

if dirty_plain:
    lines.append("CONCLUSION | configclass inheritance merges the parent's "
                 "observations. Crew configs must stop inheriting CoopLiftEnvCfg.")
elif dirty_parsed:
    lines.append("CONCLUSION | plain instantiation is clean, parse_env_cfg is not: "
                 "the registry/copy path reintroduces the parent field.")
else:
    lines.append("CONCLUSION | both paths clean; the leak is downstream of cfg "
                 "construction, i.e. in Hydra's to_dict/from_dict round trip.")

for ln in lines:
    print(ln, flush=True)
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
app.close()
