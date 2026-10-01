"""ONE physics closed-loop episode of a NavGym actor through maze_explore.run_seed (imported, unmodified) with the R1
candidate: the learned ego map integrates each lidar packet at the pose the packet was CAPTURED at (maze_explore's
humanoid branch already does this; its biped --policy branch integrates at the next step's loop-top pose).
Training-range maze seeds only (< 10 000). Nothing in the repository is modified: the harness of phys_closedloop.py is
imported (its TeamSensors wrapper logs every step, no command filter), the capture pose is recorded by one more
TeamSensors.filter_commands wrapper, and bhl_robust.navgym.env.EgoMap (imported by run_seed at call time) is replaced
by a subclass whose update() substitutes the recorded capture pose.

usage: phys_closedloop_capfix.py <arm> <maze_seed>
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import phys_closedloop as pc                      # installs the logging wrapper (FILT None -> no filter)
import maze_explore as mx
from bhl_robust.eval import team_sensors as ts
import bhl_robust.navgym.env as ngenv
from omegaconf import OmegaConf
from team_airlock import CpuPolicy

CAP = {"pose": None, "used": 0, "shift_m": [], "shift_deg": []}
_inner = ts.TeamSensors.filter_commands           # pc._patched: logs, then the original filter_commands


def _patched_capfix(self, data, commands, now):
    before = self.packets[0]["stamp_s"] if self.packets[0] is not None else None
    out = _inner(self, data, commands, now)
    pkt = self.packets[0]
    if pkt is not None and pkt["stamp_s"] != before:
        # runner.d is unchanged since the loop top of this step: the same pose maze_explore read there
        slot = self.slots[0]
        q = data.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7]
        CAP["pose"] = (float(data.xpos[slot.body_id, 0]), float(data.xpos[slot.body_id, 1]), float(mx.yaw_of(q)))
    return out


ts.TeamSensors.filter_commands = _patched_capfix
mx.TeamSensors.filter_commands = _patched_capfix


class EgoMapCapturePose(ngenv.EgoMap):
    def update(self, x, y, yaw, angles, ranges):
        if CAP["pose"] is not None:
            cx, cy, cyaw = CAP["pose"]
            CAP["used"] += 1
            CAP["shift_m"].append(math.hypot(x - cx, y - cy))
            CAP["shift_deg"].append(abs(math.degrees(math.atan2(math.sin(yaw - cyaw), math.cos(yaw - cyaw)))))
            x, y, yaw = cx, cy, cyaw
        return super().update(x, y, yaw, angles, ranges)


ngenv.EgoMap = EgoMapCapturePose

if __name__ == "__main__":
    arm, seed = sys.argv[1], int(sys.argv[2])
    assert seed < 10_000, "training-range maze seeds only"
    out_dir = pc.HERE / "data" / "physloop" / f"{arm}_capfix"
    out_dir.mkdir(parents=True, exist_ok=True)
    args = pc.make_args(arm, out_dir, pc.HERE / "mjcf_cache" / "maze")
    cfg = OmegaConf.load(Path(args.upstream) / mx.GAIT_DEFAULT)
    policy = CpuPolicy(cfg.policy_checkpoint_path)
    pc.LOG.clear()
    t0 = time.time()
    res = mx.run_seed(args, cfg, policy, seed)
    L = np.array(pc.LOG)
    np.savez(out_dir / f"seed{seed}_log.npz", log=L, cols=np.array(["t", "raw_vx", "raw_wz", "pre_vx", "pre_wz", "out_vx", "out_wz", "x", "y",
                                                                   "yaw", "range_scale", "imu_scale", "stale_stop", "pkt_stamp"]))
    res["wall_seconds_total"] = round(time.time() - t0, 1)
    res["filter"] = "none"
    res["map_integration"] = "capture_pose (R1 candidate)"
    sm, sd = np.array(CAP["shift_m"]), np.array(CAP["shift_deg"])
    res["capture_pose_substitution"] = {"updates": CAP["used"], "pose_shift_m_median": float(np.median(sm)) if sm.size else None,
                                        "pose_shift_m_max": float(sm.max()) if sm.size else None,
                                        "yaw_shift_deg_median": float(np.median(sd)) if sd.size else None,
                                        "yaw_shift_deg_p90": float(np.percentile(sd, 90)) if sd.size else None,
                                        "yaw_shift_deg_max": float(sd.max()) if sd.size else None}
    (out_dir / f"seed{seed}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("seed", "outcome", "outcome_class", "completion_s", "wall_contact_steps", "path_length_m", "sensor_stats")}
                     | {"map_integration": res["map_integration"], "capture_pose_substitution": res["capture_pose_substitution"],
                        "wall": res["wall_seconds_total"]}, default=float), flush=True)
