"""Run several policies side by side in ONE MuJoCo scene.

Every comparison elsewhere in this repo is two separate rollouts composited into
a split screen. That is honest but it is still two simulations. Here the robots
share a world, a ground plane, a solver, and a clock, so what you see is not an
edit -- they are genuinely diverging under identical conditions at the same
instant.

Composition uses MuJoCo's MjSpec attach API rather than XML text munging: each
robot is attached under its own frame with a name prefix (`r0_`, `r1_`, ...),
which keeps actuator/sensor names unique and lets every per-robot index be
resolved by name instead of by assuming a memory layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from bhl_robust.eval.livery import apply_livery
from bhl_robust.eval.mjcf_assets import prepare_mjcf

# Distinct tints so a four-robot shot is readable without reading a caption.
# Order is the slot order passed to build_multi.
PALETTE = [
    (0.22, 0.78, 0.45, 1.0),   # green
    (0.90, 0.24, 0.22, 1.0),   # red
    (0.22, 0.52, 0.95, 1.0),   # blue
    (0.96, 0.72, 0.14, 1.0),   # amber
    (0.75, 0.40, 0.90, 1.0),
    (0.20, 0.82, 0.82, 1.0),
]

# Minimal world. The robots supply everything else.
_FLAT_WORLD = """<mujoco model="bhl-multi">
  <statistic center="1.5 0 0.4" extent="4.0"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="140" elevation="-16" offwidth="1920" offheight="1080"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.28 0.42 0.58" rgb2="0.05 0.07 0.10"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.22 0.32 0.42" rgb2="0.12 0.20 0.28" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.15"/>
  </asset>
  <worldbody>
    <light pos="0 0 3.0" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""

# Structured real-world geometry rather than parameterized noise: a lab floor
# with a cable run, a door threshold, and a ramp. The question this asks is
# whether a terrain policy generalizes to composed geometry or only to the
# noise distribution it was scored on.
_LAB_WORLD = """<mujoco model="bhl-lab">
  <!-- angle="radian" is load-bearing. MuJoCo's default is DEGREES, and the
       first version of this scene omitted the tag while writing its eulers in
       radians. `euler="1.5708 0 0"` was therefore 1.57 degrees, not 90: the
       cable stayed a 5.2 m vertical pole instead of lying across the lane, and
       the ramp stayed a flat slab. Three of four policies were stopping on
       geometry that was not what it claimed to be. -->
  <compiler angle="radian"/>
  <statistic center="1.8 0 0.4" extent="4.2"/>
  <visual>
    <headlight diffuse="0.65 0.65 0.65" ambient="0.32 0.32 0.32" specular="0 0 0"/>
    <rgba haze="0.18 0.22 0.28 1"/>
    <global azimuth="125" elevation="-14" offwidth="1920" offheight="1080"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.58 0.62" rgb2="0.12 0.13 0.15"
             width="512" height="3072"/>
    <texture type="2d" name="tile" builtin="checker" mark="edge"
             rgb1="0.78 0.78 0.76" rgb2="0.62 0.62 0.60" markrgb="0.45 0.45 0.44"
             width="300" height="300"/>
    <texture type="2d" name="carpet" builtin="flat" rgb1="0.42 0.28 0.22" rgb2="0.42 0.28 0.22"
             width="128" height="128"/>
    <material name="tile" texture="tile" texuniform="true" texrepeat="8 8" reflectance="0.22"/>
    <material name="carpet" texture="carpet" texuniform="true" reflectance="0.04"/>
  </asset>
  <worldbody>
    <light pos="1.5 0 3.2" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="tile"/>
    <!-- Heights are set against the robot, not against a real building. BHL's
         leg is 0.12 m thigh + 0.16 m shank = 0.28 m, and it trained on discrete
         obstacles capped at 4 cm (14% of leg length). Blind legged locomotion
         is reliable to roughly 10-20% of leg length; the first version of this
         scene ran 27-61%, which is why nothing crossed it.

         Spacing is set against the *clip*, not the building either. These
         policies make about 0.28 m/s of real ground speed against a 0.40 m/s
         command, so the 6.4 m course the heights were first fixed on took
         ~24 s to walk -- and a GIF long enough to show it is too large for
         GitHub to inline. Every height and grade below is unchanged; only the
         gaps between features are shorter, which puts the whole course inside
         a 16 s clip. -->
    <!-- visual-only carpet strip: friction is not the claim, the geometry is -->
    <geom name="carpet" type="box" size="0.50 2.60 0.004" pos="0.30 0 0.004"
          material="carpet" contype="0" conaffinity="0"/>
    <!-- cable run across the lane: 2.5 cm, 9% of leg length -->
    <geom name="cable" type="cylinder" size="0.0125 2.6" pos="0.90 0 0.0125"
          euler="1.5708 0 0" rgba="0.10 0.10 0.10 1"/>
    <!-- door threshold at exactly the training obstacle ceiling, 4 cm = 14% -->
    <geom name="threshold" type="box" size="0.07 2.6 0.02" pos="1.70 0 0.02"
          rgba="0.48 0.38 0.26 1"/>
    <!-- a real wedge: 3.7 deg over 1.3 m, leading edge 2 cm off the floor,
         summit 10.5 cm. Training saw slopes to 15 deg, so the grade is far
         inside the envelope; only the lip is a step at all. -->
    <geom name="ramp" type="box" size="0.65 2.4 0.010" pos="2.95 0 0.0525"
          euler="0 -0.0653 0" rgba="0.58 0.58 0.54 1"/>
    <!-- landing flush with the ramp summit and overlapping it, so the exit off
         the wedge is not a second step -->
    <geom name="landing" type="box" size="0.60 2.4 0.0525" pos="4.10 0 0.0525"
          rgba="0.52 0.52 0.50 1"/>
  </worldbody>
</mujoco>
"""

_WORLDS = {"flat": _FLAT_WORLD, "lab": _LAB_WORLD}


@dataclass
class Slot:
    """Everything needed to drive and read one robot in the shared model."""
    prefix: str
    label: str
    ctrl: np.ndarray          # actuator indices
    jpos_adr: np.ndarray      # sensordata addresses, joint order
    jvel_adr: np.ndarray
    quat_adr: int
    gyro_adr: int
    qpos_adr: int             # free-joint qpos start
    qvel_adr: int
    body_id: int


def colorize_robots(model: mujoco.MjModel, n: int, hero: int | None = None) -> None:
    """Tint every geom that belongs to robot i with PALETTE[i].

    `hero` instead gets the orange-shell/black-joint livery. A flat tint is the
    right label when the question is which robot fell, and the wrong one when
    the question is what a single robot's legs are doing — so the robot that
    finishes the course is painted to be watched, and the rest stay labels.
    """
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        for i in range(n):
            if name.startswith(f"r{i}_"):
                model.geom_rgba[g] = PALETTE[i % len(PALETTE)]
                break
    if hero is not None:
        n_shell, n_joint = apply_livery(model, f"r{hero}_")
        if n_shell == 0 or n_joint == 0:
            raise RuntimeError(
                f"livery partition is degenerate for r{hero}_ "
                f"({n_shell} shell, {n_joint} joint geoms); the asset's "
                "collision groups changed"
            )


def build_multi(upstream: Path, cache_dir: Path, n: int, labels: list[str],
                variant: str = "biped", spacing: float = 1.1, world: str = "flat",
                ego_camera: bool = False, hero: int | None = None):
    """Compose `n` robots into one model. Returns (model, slots).

    With `ego_camera`, each robot carries the base-mounted depth camera from
    `prepare_mjcf`. MjSpec prefixes attached bodies, so the cameras come out as
    `r0_ego_depth`, `r1_ego_depth`, ... and a renderer can be pointed at any one
    of them to show what that particular policy is walking into.

    `hero` is the slot index painted in the orange/black livery instead of a
    flat palette tint.
    """
    scene = prepare_mjcf(upstream, cache_dir, variant, ego_camera=ego_camera)
    robot_xml = scene.parent / ("berkeley_humanoid_lite_biped.xml" if variant == "biped"
                                else "berkeley_humanoid_lite.xml")

    xml = _WORLDS.get(world)
    if xml is None:
        raise ValueError(f"unknown world {world!r}; expected one of {sorted(_WORLDS)}")
    world_path = cache_dir / f"multi_world_{world}.xml"
    world_path.write_text(xml)
    spec = mujoco.MjSpec.from_file(str(world_path))

    # Lay the robots out along +y, centred on the origin, so one camera frames all.
    y0 = -spacing * (n - 1) / 2.0
    for i in range(n):
        child = mujoco.MjSpec.from_file(str(robot_xml))
        frame = spec.worldbody.add_frame()
        frame.pos = [0.0, y0 + i * spacing, 0.0]
        frame.attach_body(child.bodies[1], f"r{i}_", "")

    model = spec.compile()

    def sid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)

    slots = []
    for i in range(n):
        p = f"r{i}_"
        act = [j for j in range(model.nu)
               if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) or "").startswith(p)]
        joints = [(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) or "") for j in act]
        jpos, jvel = [], []
        for jn in joints:
            base = jn[len(p):].replace("_joint", "")
            a_p, a_v = sid(f"{p}{base}_pos"), sid(f"{p}{base}_vel")
            if a_p < 0 or a_v < 0:
                raise RuntimeError(f"cannot resolve sensors for {jn}")
            jpos.append(model.sensor_adr[a_p]); jvel.append(model.sensor_adr[a_v])

        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{p}base")
        jid = model.body_jntadr[bid]
        slots.append(Slot(
            prefix=p, label=labels[i] if i < len(labels) else p,
            ctrl=np.asarray(act, dtype=int),
            jpos_adr=np.asarray(jpos, dtype=int), jvel_adr=np.asarray(jvel, dtype=int),
            quat_adr=int(model.sensor_adr[sid(f"{p}imu_quat")]),
            gyro_adr=int(model.sensor_adr[sid(f"{p}imu_gyro")]),
            qpos_adr=int(model.jnt_qposadr[jid]), qvel_adr=int(model.jnt_dofadr[jid]),
            body_id=int(bid),
        ))
    colorize_robots(model, n, hero=hero)
    return model, slots


class MultiRunner:
    """Steps N independent policies inside one shared MuJoCo model."""

    def __init__(self, model, slots, cfgs, controllers):
        self.m, self.slots, self.cfgs, self.ctrls = model, slots, cfgs, controllers
        self.d = mujoco.MjData(model)
        c0 = cfgs[0]
        self.m.opt.timestep = c0.physics_dt
        self.substeps = int(round(c0.policy_dt / c0.physics_dt))
        self.kp = np.asarray(c0.joint_kp, dtype=np.float32)
        self.kd = np.asarray(c0.joint_kd, dtype=np.float32)
        self.eff = np.asarray(c0.effort_limits, dtype=np.float32)
        self.qdefault = np.asarray(c0.default_joint_positions, dtype=np.float32)
        self.alive = [True] * len(slots)

    def reset(self, rng):
        mujoco.mj_resetData(self.m, self.d)
        for s in self.slots:
            # Each robot's lateral spawn offset is baked into its free joint's
            # qpos0 by the attach frame. Zeroing position here would stack every
            # robot at the origin, where they collide and immediately topple.
            self.d.qpos[s.qpos_adr:s.qpos_adr + 3] = self.m.qpos0[s.qpos_adr:s.qpos_adr + 3]
            self.d.qpos[s.qpos_adr + 3:s.qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
            n = len(s.ctrl)
            self.d.qpos[s.qpos_adr + 7:s.qpos_adr + 7 + n] = (
                self.qdefault + rng.normal(0.0, 0.02, n).astype(np.float32))
        mujoco.mj_forward(self.m, self.d)
        self.alive = [True] * len(self.slots)
        for c in self.ctrls:
            c.prev_actions[:] = 0.0
            c.policy_observations[:] = 0.0

    def observe(self, i, command):
        s = self.slots[i]
        sd = self.d.sensordata
        return np.concatenate([
            sd[s.quat_adr:s.quat_adr + 4],
            sd[s.gyro_adr:s.gyro_adr + 3],
            sd[s.jpos_adr], sd[s.jvel_adr],
            np.array([3.0], dtype=np.float32),
            np.asarray(command, dtype=np.float32),
        ]).astype(np.float32)

    def tilt(self, i):
        s = self.slots[i]
        q = self.d.qpos[s.qpos_adr + 3:s.qpos_adr + 7]
        up = 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)
        return float(np.arccos(np.clip(up, -1.0, 1.0)))

    def step(self, targets_per_robot):
        """targets_per_robot[i] = scaled joint position targets for robot i."""
        for _ in range(self.substeps):
            for i, s in enumerate(self.slots):
                jp = self.d.sensordata[s.jpos_adr]
                jv = self.d.sensordata[s.jvel_adr]
                tau = self.kp * (targets_per_robot[i] - jp) + self.kd * (-jv)
                self.d.ctrl[s.ctrl] = np.clip(tau, -self.eff, self.eff)
            mujoco.mj_step(self.m, self.d)

    def push_all(self, speed, rng):
        ang = rng.uniform(0, 2 * np.pi)
        for s in self.slots:
            self.d.qvel[s.qvel_adr + 0] += speed * np.cos(ang)
            self.d.qvel[s.qvel_adr + 1] += speed * np.sin(ang)
