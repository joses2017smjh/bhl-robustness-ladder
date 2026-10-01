"""verify-turning: re-measure the near-threshold F3 (obs-noise) runs with turn_test's yaw origin (reset) instead of
mj_probe2's (first policy step after the command).  Exact replay: same mj_probe2.run call as v2_obsnoise.py (same
NpActor RNG stream), with yaw_of logged via monkeypatch and the post-reset yaw captured. Reset seeds 0-2 only."""
import json, math, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/turning")
sys.path.insert(0, "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/scripts/bench")
os.environ.setdefault("MJ_CACHE", str(HERE / "mjcache"))
import mj_probe2 as P
import team_airlock
_orig_yaw = P.yaw_of
LOG = {"reset": None, "yaws": []}
def logged_yaw(q):
    y = _orig_yaw(q); LOG["yaws"].append(y); return y
P.yaw_of = logged_yaw
_orig_reset = team_airlock.ContactRunner.reset
def reset_hook(self, rng):
    out = _orig_reset(self, rng)
    s = self.slots[0]
    LOG["reset"] = _orig_yaw(self.d.qpos[s.qpos_adr + 3:s.qpos_adr + 7]); return out
team_airlock.ContactRunner.reset = reset_hook
cases = [  # (run, seed, wz, vx, warm, saved probe value from v2_obsnoise_x1.0.json)
    ("arms-turn-turncmd-s1", 1, -0.6, 0.0, 3.0, 156.2),
    ("arms-turn-turnhip-s1", 2, 0.6, 0.0, 3.0, 160.4),
    ("arms-turn-turncmd-s2", 0, 0.0, 0.35, 1.0, 12.4),
    ("arms-turn-turnhip-s1", 0, 0.0, 0.35, 1.0, -10.3),
]
which = [int(a) for a in sys.argv[1:]] or range(len(cases))
for i in which:
    run, seed, wz, vx, warm, saved = cases[i]
    LOG["yaws"].clear(); LOG["reset"] = None
    o = P.run(P.ckpt_path(run, None), wz, warm, 6.0, seed, 0.0, vx, "all", 1e9, 3.0, 1.0)
    yv = np.unwrap(np.array([LOG["reset"]] + LOG["yaws"]))
    sgn = math.copysign(1.0, wz) if wz != 0 else 1.0
    from_reset = math.degrees(yv[-1] - yv[0]) * sgn
    rec = {"run": run, "reset_seed": seed, "wz": wz, "vx": vx, "obs_noise": 1.0, "probe_yaw_cmd_dir": o["yaw_deg_cmd_dir"],
           "saved_probe_value": saved, "reproduced": abs(o["yaw_deg_cmd_dir"] - saved) < 0.05,
           "turn_test_origin_yaw_cmd_dir": round(from_reset, 1), "fell_at_s": o["fell_at_s"]}
    print(json.dumps(rec), flush=True)
    with open(HERE / "origin_check.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
