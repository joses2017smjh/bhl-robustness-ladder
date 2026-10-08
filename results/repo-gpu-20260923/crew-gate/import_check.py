import os, sys, traceback
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
out = os.environ["CREW_GATE_OUT"]
verdict = "import FAIL"
try:
    import berkeley_humanoid_lite.tasks, bhl_robust.tasks  # noqa: F401
    import gymnasium as gym
    from isaaclab.managers import ObservationTermCfg
    from isaaclab_tasks.utils import parse_env_cfg
    ids = sorted(i for i in gym.registry if "Crew" in i)
    print("REGISTERED", ids, flush=True)
    clean = True
    for i in ids:
        cfg = parse_env_cfg(i, device="cuda:0", num_envs=8)
        pol = cfg.observations.policy
        n_terms = sum(1 for v in vars(pol).values() if isinstance(v, ObservationTermCfg))
        pair = sorted(k for k, v in vars(pol).items() if v is not None and k.endswith(("_a", "_b")))
        robots = sorted(k for k in vars(cfg.scene) if k.startswith("robot_"))
        clean = clean and not pair and "robot_a" not in robots
        print(f"CONSTRUCTED {i} crew_size={getattr(cfg, 'crew_size', None)} policy_terms={n_terms} "
              f"robots={robots} pair_terms={pair}", flush=True)
    verdict = "import OK" if (len(ids) == 4 and clean) else "import FAIL"
    if len(ids) != 4:
        print(f"expected 4 crew tasks, got {len(ids)}", flush=True)
except Exception:
    traceback.print_exc()
    sys.stderr.flush()
with open(os.path.join(out, "import_verdict.txt"), "w") as f:
    f.write(verdict + "\n")
print("IMPORT", verdict, flush=True)
sys.stdout.flush()
app.close()
