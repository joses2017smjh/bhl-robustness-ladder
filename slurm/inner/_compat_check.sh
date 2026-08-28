#!/bin/bash
# Does `import bhl_robust.tasks` still work on THIS stack?
#
# The physx shim fired on v51 -- @configclass gives the field a default_factory
# so it never becomes a class attribute, and hasattr said it was missing -- then
# imported a 3.x-only module and broke every v51 task. A shim that is meant to
# be a no-op on one stack has to be tested on that stack.
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args([]); a.headless = True
app = AppLauncher(a)
from bhl_robust import compat
print("shims applied:", compat.apply() or "(none needed)")
import bhl_robust.tasks as T
import gymnasium as gym
ids = sorted(i for i in gym.registry if "BHL" in i)
print(f"registered task ids: {len(ids)}")
print("  v2 ids:", [i for i in ids if i.startswith("TaskV2")][:3], "...")
app.app.close()
PY
