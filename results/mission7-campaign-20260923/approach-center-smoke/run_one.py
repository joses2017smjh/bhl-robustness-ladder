import sys, json
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from mission7_approach_followup import CONTROLLERS, run_controller
from bhl_robust.mission.approach_debug import DebugEnv
idx = int(sys.argv[1]); arm = sys.argv[2]
env = DebugEnv("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder", f"/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/results/mission7-campaign-20260923/approach-center-smoke/cache-{arm}-{idx}", stage="approach", split="test", seed=2300, approach_distance=.65)
env.reset(idx); c = CONTROLLERS[arm](env); row = run_controller(env, c)
ev = getattr(c, "center_events", [])
print(json.dumps({"arm": arm, "layout": idx, "success": row["success"], "failure": row["failure"], "collisions": row["collisions"], "elapsed_s": round(row["elapsed_s"], 2), "center_ticks": len(ev), "lateral_err_range": [round(min(e["lateral_err_m"] for e in ev), 3), round(max(e["lateral_err_m"] for e in ev), 3)] if ev else None}))
