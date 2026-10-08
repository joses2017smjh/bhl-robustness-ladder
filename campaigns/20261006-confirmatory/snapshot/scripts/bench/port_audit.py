"""What breaks if the overlays are imported under Isaac Lab 3.x?

Scoping by measurement. Isaac Lab went 2.3.2 -> 3.0.0b2 and the config classes,
the mdp namespace and the manager API all moved between those majors, but "a
major version changed" is not an estimate of work -- it is an excuse to guess.
This imports each overlay module on its own under the 6.0 stack and reports the
first thing that fails, so the port can be planned against a list of real
breakages instead of a feeling about semver.

Each module is imported in a subprocess: the first failure in a shared
interpreter would mask every module after it, and a hang in one would take the
rest with it.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODULES = [
    "bhl_robust.terrains.bumpy",
    "bhl_robust.tasks.push_env_cfg",
    "bhl_robust.tasks.terrain_env_cfg",
    "bhl_robust.tasks.arms_env_cfg",
    "bhl_robust.tasks.collision_env_cfg",
    "bhl_robust.tasks.depth_env_cfg",
    "bhl_robust.tasks.scan_env_cfg",
    "bhl_robust.tasks.symmetry",
    "bhl_robust.tasks.coop_lift_mdp",
    "bhl_robust.tasks.coop_lift_env_cfg",
    "bhl_robust.tasks.coop_depth_env_cfg",
    "bhl_robust.tasks.coop_hard_env_cfg",
    "bhl_robust.tasks.coop_crew_generated",
    "bhl_robust.eval.harness",
    "bhl_robust.eval.depth",
]

# Substituted with str.replace, not str.format: the stub is full of f-string
# braces and format() tries to interpolate every one of them.
STUB = '''
import sys
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
try:
    from bhl_robust import compat
    shims = compat.apply()
    if shims:
        print("SHIMS " + "; ".join(shims), flush=True)
except Exception as e:
    print(f"SHIM_FAIL {type(e).__name__}: {e}"[:200], flush=True)
try:
    import berkeley_humanoid_lite.tasks  # noqa: F401
except Exception as e:
    print(f"UPSTREAM_FAIL {type(e).__name__}: {e}"[:300], flush=True)
try:
    __import__("__MOD__")
    print("OK", flush=True)
except Exception as e:
    print(f"FAIL {type(e).__name__}: {e}"[:300], flush=True)
app.close()
'''

py = sys.executable
rows = []
for mod in MODULES:
    r = subprocess.run([py, "-c", STUB.replace("__MOD__", mod)],
                       capture_output=True, text=True, timeout=900,
                       cwd=str(REPO / "external/Berkeley-Humanoid-Lite"))
    out = [l for l in r.stdout.splitlines() if l.startswith(("OK", "FAIL", "UPSTREAM_FAIL"))]
    verdict = out[-1] if out else "NO VERDICT (crashed before reporting)"
    rows.append((mod, verdict))
    print(f"{mod:<42} {verdict[:110]}", flush=True)

ok = sum(1 for _, v in rows if v == "OK")
print(f"\nPORT AUDIT | {ok}/{len(rows)} modules import unchanged on Isaac Lab 3.x")
for mod, v in rows:
    if v != "OK":
        print(f"  NEEDS WORK  {mod}: {v[:150]}")
