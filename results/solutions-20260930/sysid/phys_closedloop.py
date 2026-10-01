"""Physics closed-loop runs of a NavGym actor through maze_explore.run_seed itself (imported, unmodified), with
full-rate logging and an optional deployment-time command filter, on TRAINING-RANGE maze seeds only (< 10 000).

The filter and the logger are installed by wrapping TeamSensors.filter_commands (the call maze_explore makes once per
policy step with the actor's raw command): pre-filter command -> [filter] -> brake -> gait. The actor's own
prev_action observation is unchanged (maze_explore keeps it). Nothing in the repository is modified.

usage: phys_closedloop.py <arm> <seed_start> <n> <filter> [tag]
  filter: none | lpfTAU (first-order low-pass on wz, e.g. lpf0.2) | maN (moving average of the last N wz) | rlR (rate limit R rad/s^2)
"""
import argparse, json, math, os, sys, time
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
from pathlib import Path
import numpy as np
import sysid_lib  # noqa: F401  (sets sys.path for the repo's src and scripts/bench)
import maze_explore as mx
from bhl_robust.eval import team_sensors as ts
from omegaconf import OmegaConf
from team_airlock import CpuPolicy

REPO = sysid_lib.REPO
HERE = Path(__file__).resolve().parent
DT = 0.04


class Filt:
    def __init__(self, spec):
        self.spec = spec
        self.reset()

    def reset(self):
        self.w = 0.0; self.buf = []

    def __call__(self, raw):
        s = self.spec
        out = np.asarray(raw, float).copy()
        if s == "none":
            return out
        if s.startswith("lpf"):
            a = DT / max(float(s[3:]), DT)
            self.w += a * (out[2] - self.w); out[2] = self.w
        elif s.startswith("ma"):
            self.buf.append(out[2]); self.buf = self.buf[-int(s[2:]):]; out[2] = float(np.mean(self.buf))
        elif s.startswith("rl"):
            d = float(s[2:]) * DT
            self.w = float(np.clip(out[2], self.w - d, self.w + d)); out[2] = self.w
        else:
            raise ValueError(s)
        return out


LOG = []
FILT = None
_orig = ts.TeamSensors.filter_commands


def _patched(self, data, commands, now):
    raw = np.asarray(commands[0], float)
    pre = FILT(raw) if FILT is not None else raw.copy()
    out = _orig(self, data, [pre], now)
    slot = self.slots[0]
    q = data.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7]
    pkt = self.packets[0]
    b = self.latest[0]["brake"]
    LOG.append([now, raw[0], raw[2], pre[0], pre[2], out[0][0], out[0][2], float(data.xpos[slot.body_id, 0]), float(data.xpos[slot.body_id, 1]),
                mx.yaw_of(q), b["range_scale"], b["imu_scale"], float(b["stale_stop"]), -1.0 if pkt is None else pkt["stamp_s"]])
    return out


ts.TeamSensors.filter_commands = _patched
mx.TeamSensors.filter_commands = _patched


def make_args(arm, out_dir, cache_dir):
    ap_defaults = dict(upstream=REPO / "external/Berkeley-Humanoid-Lite", cache_dir=cache_dir, gait=None, seeds=1, seed_start=0, n=6, m=6,
                       extra_openings=1, time_limit=180.0, settle_s=1.0, cruise=0.30, turn_rate=0.6, inflate=0.30, replan_s=0.4, map_res=0.10,
                       random_heading=False, sensor_mode="reactive", dropout_probability=0.35, out_dir=out_dir, render=False, no_render=True,
                       frames_root=str(HERE / "frames"), width=1280, height=720, distance=0.0, azimuth=90.0, elevation=-66.0, stride=2,
                       plain_world=False, gif=None, gif_speed=4.0, gif_fps=8, gif_width=860, tag="", imu_panel=False, imu_panel_rate=100.0,
                       rig_panels=False, rig_res="160x120", policy=REPO / f"results/navgym-v4-20260928/{arm}/actor.onnx", no_overwrite=False,
                       variant="biped", turn_enter=None, turn_exit=None, wz_walk=None, waypoint_radius=None, pose_yaw_deg=0.0, pose_bias_m=0.0,
                       pose_noise_m=0.0, imu_source="truth", imu_filter="mahony", imu_init="accel", imu_kp=1.0, imu_ki=0.1, imu_beta=0.1,
                       imu_gyro_std=0.0, imu_accel_std=0.0, imu_gyro_bias=0.0, imu_gyro_bias_walk=0.0, imu_delay_steps=0, imu_rate_hz=200.0,
                       imu_delay_ms=0.0)
    return argparse.Namespace(**ap_defaults)


if __name__ == "__main__":
    arm, s0, n, spec = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    tag = sys.argv[5] if len(sys.argv) > 5 else ""
    out_dir = HERE / "data" / "physloop" / f"{arm}_{spec}{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    args = make_args(arm, out_dir, HERE / "mjcf_cache" / "maze")
    gait = Path(args.upstream) / mx.GAIT_DEFAULT
    cfg = OmegaConf.load(gait)
    policy = CpuPolicy(cfg.policy_checkpoint_path)
    for seed in range(s0, s0 + n):
        assert seed < 10_000, "training-range maze seeds only"
        LOG.clear()
        FILT = Filt(spec)
        globals()["FILT"] = FILT
        t0 = time.time()
        res = mx.run_seed(args, cfg, policy, seed)
        L = np.array(LOG)
        np.savez(out_dir / f"seed{seed}_log.npz", log=L, cols=np.array(["t", "raw_vx", "raw_wz", "pre_vx", "pre_wz", "out_vx", "out_wz", "x", "y", "yaw",
                                                                       "range_scale", "imu_scale", "stale_stop", "pkt_stamp"]))
        res["wall_seconds_total"] = round(time.time() - t0, 1)
        res["filter"] = spec
        (out_dir / f"seed{seed}.json").write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps({k: res[k] for k in ("seed", "outcome", "outcome_class", "completion_s", "wall_contact_steps", "path_length_m", "sensor_stats")}
                         | {"filter": spec, "wall": res["wall_seconds_total"]}, default=float), flush=True)
