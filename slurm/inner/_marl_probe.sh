#!/bin/bash
# G-B4 step 0: does this stack actually have the MARL pieces, before anything
# is designed around them? The work order assumes DirectMARL + skrl; skrl 1.4.3
# is on disk with IPPO and MAPPO, but the isaaclab directory on disk is a
# launcher, not the package, so the import is the only honest check.
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
import importlib
for mod, names in (("isaaclab.envs", ["DirectMARLEnv", "DirectMARLEnvCfg",
                                      "multi_agent_to_single_agent"]),
                   ("isaaclab_rl.skrl", ["SkrlVecEnvWrapper"]),
                   ("skrl.multi_agents.torch.mappo", ["MAPPO"]),
                   ("skrl.multi_agents.torch.ippo", ["IPPO"])):
    try:
        m = importlib.import_module(mod)
        have = [n for n in names if hasattr(m, n)]
        miss = [n for n in names if not hasattr(m, n)]
        print(f"  {mod:34} OK   have={have} missing={miss}")
    except Exception as e:
        print(f"  {mod:34} FAIL {type(e).__name__}: {e}")
import isaaclab, skrl
print(f"\n  isaaclab {isaaclab.__version__} at {isaaclab.__file__}")
print(f"  skrl     {skrl.__version__}")
PY
