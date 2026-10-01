"""System identification harness for the frozen biped gait dr-default-s0 in MuJoCo.

Built exactly the way scripts/bench/maze_explore.py::run_seed builds its runner (build_multi(...,
variant='biped'), RlController + CpuPolicy on the exported ONNX deploy config, ContactRunner,
runner.reset(default_rng(seed)), spawn heading set through the free-joint quaternion, TeamSensors
with the biped mounts, the speed brake applied to the command with sensors.filter_commands), but on
the open 'flat' world (no walls) unless a world XML is given. Nothing in the repository is modified;
the MJCF cache lives under the scratch directory.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

REPO = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder")
sys.path.insert(0, str(REPO / "scripts/bench"))
sys.path.insert(0, str(REPO / "src"))

import numpy as np                                                   # noqa: E402
import mujoco                                                        # noqa: E402
from omegaconf import OmegaConf                                      # noqa: E402
from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController  # noqa: E402
from team_airlock import ContactRunner, CpuPolicy                    # noqa: E402
from bhl_robust.eval.multi_robot import build_multi, _WORLDS         # noqa: E402
from bhl_robust.eval.team_sensors import TeamSensors                 # noqa: E402

UPSTREAM = REPO / "external/Berkeley-Humanoid-Lite"
GAIT = UPSTREAM / "logs/rsl_rl/biped/2026-08-17_09-54-10_dr-default-s0/exported/deploy.yaml"
LIDAR_MOUNT_BIPED = (0.0, 0.0, 0.34)          # maze_explore.py constants (copied, not imported: that module pulls in the recorder)
STEREO_CENTER_BIPED = (0.12, 0.0, 0.30)
SCRATCH = Path(__file__).resolve().parent
CACHE = SCRATCH / "mjcf_cache"


def yaw_of(q):
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


class BipedSim:
    """One biped on the flat floor; `run(schedule)` returns per-policy-step arrays."""

    def __init__(self, world: str = "flat"):
        self.cfg = OmegaConf.load(GAIT)
        assert self.cfg.num_actions == 12
        self.policy = CpuPolicy(self.cfg.policy_checkpoint_path)
        cache = CACHE / world
        cache.mkdir(parents=True, exist_ok=True)
        self.model, self.slots = build_multi(UPSTREAM, cache, 1, ["explorer"], variant="biped", world=world)
        self.slot = self.slots[0]
        self.dt = float(self.cfg.policy_dt)

    def run(self, schedule, seconds: float, seed: int = 0, brake: bool = False, yaw0: float = 0.0, xy0=(0.0, 0.0),
            record_contacts: bool = False, sensor_mode: str = "reactive"):
        """schedule(t, state) -> (vx, vy, wz) raw command at policy time t (state: dict of the latest measurements).
        brake=True applies TeamSensors.filter_commands exactly as maze_explore does."""
        model, slot, cfg = self.model, self.slot, self.cfg
        controller = RlController(cfg)
        controller.policy = self.policy
        runner = ContactRunner(model, self.slots, [cfg], [controller])
        rng = np.random.default_rng(seed)
        runner.reset(rng)
        runner.d.qpos[slot.qpos_adr:slot.qpos_adr + 2] = (xy0[0] + rng.normal(0, 0.03), xy0[1] + rng.normal(0, 0.03))
        runner.d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7] = (math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2))
        mujoco.mj_forward(model, runner.d)
        owners = np.array([0 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "").startswith(slot.prefix)
                           else -1 for g in range(model.ngeom)])
        runner.configure_contacts(owners)
        sensors = TeamSensors(model, self.slots, owners, mode=sensor_mode, seed=seed, dropout_probability=0.35,
                              lidar_mount=LIDAR_MOUNT_BIPED, stereo_center=STEREO_CENTER_BIPED) if brake else None
        n = int(round(seconds / self.dt))
        T = np.zeros(n); CR = np.zeros((n, 3)); CA = np.zeros((n, 3)); XY = np.zeros((n, 2)); YAW = np.zeros(n)
        VW = np.zeros((n, 3)); WB = np.zeros((n, 3)); TILT = np.zeros(n); BR = np.ones((n, 2)); Z = np.zeros(n)
        fell = None
        state = {}
        for k in range(n):
            now = k * self.dt
            q = runner.d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7]
            state = {"t": now, "xy": runner.d.xpos[slot.body_id, :2].copy(), "yaw": yaw_of(q)}
            raw = np.asarray(schedule(now, state), dtype=float)
            if sensors is not None:
                command = sensors.filter_commands(runner.d, [raw], now)[0]
                b = sensors.latest[0]["brake"]
                BR[k] = (b["range_scale"], b["imu_scale"])
            else:
                command = raw
            obs = runner.observe(0, command)
            target = controller.update(obs)
            runner.step([target])
            T[k] = now
            CR[k] = raw; CA[k] = command
            XY[k] = runner.d.xpos[slot.body_id, :2]
            Z[k] = runner.d.xpos[slot.body_id, 2]
            YAW[k] = yaw_of(runner.d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7])
            VW[k] = runner.d.qvel[slot.qvel_adr:slot.qvel_adr + 3]
            WB[k] = runner.d.qvel[slot.qvel_adr + 3:slot.qvel_adr + 6]
            TILT[k] = runner.tilt(0)
            if TILT[k] >= 0.78:
                fell = now
                n = k + 1
                break
        out = {"t": T[:n] + self.dt, "cmd_raw": CR[:n], "cmd": CA[:n], "xy": XY[:n], "yaw": np.unwrap(YAW[:n]), "v_world": VW[:n],
               "w_body": WB[:n], "tilt": TILT[:n], "brake": BR[:n], "z": Z[:n], "fell_at_s": fell, "dt": self.dt, "seed": seed}
        if sensors is not None:
            out["sensor_stats"] = dict(sensors.stats)
        return out


def fwd_speed(r):
    """Forward speed in the heading frame from the base's world velocity, per step."""
    v = r["v_world"]
    return v[:, 0] * np.cos(r["yaw"]) + v[:, 1] * np.sin(r["yaw"])


def lat_speed(r):
    v = r["v_world"]
    return -v[:, 0] * np.sin(r["yaw"]) + v[:, 1] * np.cos(r["yaw"])


def yaw_rate_fd(r):
    """Finite-difference yaw rate of the base between policy steps (rad/s)."""
    y = np.concatenate([[r["yaw"][0]], r["yaw"]])
    return np.diff(y) / r["dt"]
