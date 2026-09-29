"""Two (or four) BHL humanoids lift a cube together and carry it -- scripted arms.

Every learned cooperative-lift rollout in this repo failed in MuJoCo (the best,
`carry_cube_pov`, raised the cube 7.8 cm out of a 41 cm collapse). This module
asks the narrower question of whether the *robots* can do it when the arms are
not learned. Label on every output: `LABEL` + `LABEL_DETAIL`.

* legs: the FROZEN learned 22-DoF locomotion gait (`arms-dr1.0-s0`), queried
  every policy step with its normal, unmodified 75-dim observation;
* arms: each robot's GRASPING arm (the one next to the cube) follows a SCRIPTED
  joint trajectory -- reach -> squeeze -> lift -> hold -- that replaces the
  policy's output for those five joints before the PD loop. The robot's other
  (outer) arm keeps the gait policy's own output (see "why one arm");
* cube: pose read from the simulator (ORACLE) -- it gates the carry on the lift
  and stops it after the carry distance; robot base poses (oracle) keep the two
  walkers level with each other along the carry direction.

Physics is plain MuJoCo contact: no welds, no kinematic attachment, no mocap
body. The cube (0.28 m, 0.5 kg, mu 1.2 -- `coop_replay.CUBE`) is a free body on
a static plinth. Modelling choices, all recorded in the outputs:

* hand pads: upstream's hands are visual-only, so the only arm geometry that
  can touch anything is a rounded forearm cylinder, i.e. a point contact. A box
  collision geom the size of each hand mesh's own AABB is added to each hand
  link (`add_hand_pads`) so the squeeze is a contact patch;
* grasping-arm PD kp 30 instead of deploy.yaml's 10 (kd 2 and the 4 Nm arm
  effort cap unchanged; the legs and the outer arm keep deploy.yaml gains);
* the grasping arm is spawned in the script's rest pose (hanging, slightly
  adducted) so the hand does not start inside the cube;
* the cube rotates in the hands during the lift (the pads transmit the hands'
  pitch rotation): the rule scores the cube centre's height, not its attitude.

Layout. The two robots of a pair stand side by side, both facing +y, one on
each side of the cube (x = -/+ `side_off`). Robot b (at -x) grasps with its
right hand, robot a (at +x) with its left, closing on the two cube faces normal
to x. The pair then walks BACKWARD (-y) together, which carries the cube off
its plinth without either robot stepping past the plinth.

Login-node exploration (2026-09-26; all on seeds >= 100, never on the scored
seeds 0-9):

* Facing each other across the cube (the `coop_replay.build_crew` layout) with
  BOTH arms scripted, the pair lifted the cube 13 cm and held it, but could not
  walk: with both arms held forward the gait stops stepping (0.00-0.05 m in 6 s
  at 0.25-0.5 m/s) or falls (tilt > 1.5 rad), and the sideways gait has a
  direction-dependent deadband (arms free, 6 s: robot b 1.14 m at vy = -0.2,
  robot a 0.01 m at vy = +0.2).
* One robot walking backward at -0.4 m/s with BOTH arms held in grasp-like
  poses froze in 3 of 7 poses (<= 0.05 m in 6 s); feeding the policy the
  script as its previous action, or spoofing its arm-state inputs, made it
  fall. Holding ONE arm and leaving the other to the policy: 1.74-2.46 m in
  6 s, tilt <= 0.21 rad, in all three poses.
* Forearm-cylinder point contacts: the cube pivots about the line joining the
  two contacts; best case 16 cm lift, 3.8 s hold, 7-8 N per side, dropped
  within 0.5 s of gait onset (contact slid to the rear edge).
* Hand pads + kp 30 (frozen configuration): seed 100 lifted 21 cm and held
  7.6 s, seed 101 16.8 cm / 2.8 s; the drop comes at the carry. Without
  synchronisation one robot steps and the other does not; with it (`k_sync`)
  they stay within 2-4 cm but under the loaded, held arm the gait nearly stops
  stepping at -0.4 m/s (cube carried 0.24-0.28 m before it slips) and at
  -0.6/-0.8 m/s both robots fall (tilt 1.8 rad). A lower lift pose did not
  lift (1.8 cm); script + policy arm swing ("residual") did not lift (0.4 cm).

Protocols. `run_episode(protocol=...)`:

* "carry" (default, the original experiment, `SUCCESS_RULE`): lift, then the
  oracle-gated backward carry described above;
* "lift_hold" (`LIFT_HOLD_RULE`, a separate experiment predeclared in
  SLURM_JOBS.md before any scored carry seed existed): the same harness,
  layout, hand pads, kp-30 grasping arm, arm script and label, with the carry
  phase removed -- every robot's velocity command is zero for the whole
  episode, so after the lift the pair stands and holds the cube until the end
  of a 20 s episode. The oracle cube pose is used only to score. Outputs of
  this protocol say `LIFT_HOLD_NOTE` ("carry not achieved").
* "lift_place" (`LIFT_PLACE_RULE`, `run_place_episode`, 2026-09-27): lift,
  hold, lower the cube back onto its plinth, open the hands, stand; scored on
  seeds 10-19 only. See the lift-place section below.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import mujoco
import numpy as np

from bhl_robust.eval.coop_replay import CUBE, _WORLD
from bhl_robust.eval.livery import PAYLOAD_RGBA, apply_livery
from bhl_robust.eval.mjcf_assets import prepare_mjcf
from bhl_robust.eval.multi_robot import MultiRunner, Slot

LABEL = "learned gait (frozen) + scripted arms + oracle cube pose"
LABEL_DETAIL = (
    "legs + outer arm: frozen learned PPO gait, unmodified observation | grasping arm "
    "(one per robot): scripted joint targets, PD kp 30 (deploy 10), 4 Nm cap | "
    "cube + robot poses: simulator oracle (carry gate, stop, pair sync) | contact: "
    "MuJoCo, added box pads = hand-mesh AABB, no welds")

ARM_JOINTS_L = ["arm_left_shoulder_pitch_joint", "arm_left_shoulder_roll_joint",
                "arm_left_shoulder_yaw_joint", "arm_left_elbow_pitch_joint",
                "arm_left_elbow_roll_joint"]
ARM_JOINTS_R = [j.replace("_left_", "_right_") for j in ARM_JOINTS_L]
LEFT = slice(0, 5)                # deploy.yaml order: left arm 0..4, right arm 5..9
RIGHT = slice(5, 10)
N_ARM = 10

CUBE_HALF = float(CUBE.size[0])   # 0.14 m: the 0.28 m box of CoopLiftEnvCfg
CUBE_MASS = float(CUBE.mass)      # 0.5 kg
CUBE_FRICTION = float(CUBE.friction)
TILT_LIMIT = 0.78

# ------------------------------------------------------------------ predeclared
#: Success rule, fixed before any scored seed was run (see the launcher header).
SUCCESS_RULE = {
    "episode_s": 30.0,
    "lift_peak_m": 0.10,        # cube rises >= 10 cm above its plinth rest height
    "lift_hold_m": 0.05,        # ...and stays >= 5 cm up
    "lift_hold_s": 3.0,         # ...for >= 3 s contiguous
    "carry_m": 1.0,             # horizontal distance from rest, reached while >= 5 cm up
    "tilt_rad": TILT_LIMIT,     # no robot of the pair tilts past this at any policy step
    "floor_contact": False,     # the cube never touches the floor (every physics step)
    "seeds": list(range(10)),   # reset-jitter seeds
    "pass_min": 8,              # PASS iff >= 8/10 seeds succeed (per pair for crew 4)
}

PROTOCOLS = ("carry", "lift_hold", "lift_place")

#: Cooperative LIFT-AND-HOLD rule, a separate experiment predeclared in
#: SLURM_JOBS.md ("cooperative LIFT-AND-HOLD, a separate experiment") before
#: any scored seed of the carry run existed. No carry clause: the pair never
#: receives a velocity command.
LIFT_HOLD_RULE = {
    "protocol": "lift_hold",
    "episode_s": 20.0,
    "lift_peak_m": 0.10,        # cube rises >= 10 cm above its plinth rest height
    "lift_hold_m": 0.05,        # ...and stays >= 5 cm up
    "lift_hold_s": 5.0,         # ...for >= 5 s continuously
    "tilt_rad": TILT_LIMIT,     # no robot of the pair tilts past this at any policy step
    "floor_contact": False,     # the cube never touches the floor (every physics step)
    "seeds": list(range(10)),   # reset-jitter seeds
    "pass_min": 8,              # PASS iff >= 8/10 seeds succeed (per pair for crew 4)
}

#: Said by every lift-and-hold output (frame, banner, sidecar, verdict line).
LIFT_HOLD_NOTE = "cooperative lift and hold — carry not achieved"
LIFT_HOLD_LABEL_DETAIL = (
    "legs + outer arm: frozen learned PPO gait, unmodified observation, zero velocity "
    "command all episode | grasping arm (one per robot): scripted joint targets, PD kp 30 "
    "(deploy 10), 4 Nm cap | cube pose: simulator oracle (scoring only; no carry) | contact: "
    "MuJoCo, added box pads = hand-mesh AABB, no welds")

#: Joint-space keyframes for the LEFT grasping arm (robot a); the right arm
#: (robot b) is the exact mirror, q_right = -q_left (checked in the tests).
#: Hand-pad centre in the robot frame (x forward, y left, z up from the floor)
#: from `ArmIK.pad_fk`, unobstructed -- the cube stops the squeeze and lift
#: poses short of these:
#:   rest     (0.005, 0.237, 0.374)  hanging; outer pad corner 0.284 < 0.30 = face
#:   reach    (0.095, 0.200, 0.395)  elbow 0.5, still adducted
#:   squeeze  (0.095, 0.481, 0.596)  roll 0.9: target 18 cm inside the face
#:   lift     (0.167, 0.436, 0.757)  shoulder pitch + elbow: +16 cm
KEYFRAMES_LEFT = {
    "rest": [0.0, -0.1, 0.0, 0.0, 0.0],
    "reach": [0.0, -0.2, 0.0, 0.5, 0.0],
    "squeeze": [0.0, 0.9, 0.0, 0.5, 0.0],
    "lift": [-0.5, 1.1, 0.0, 1.0, 0.0],
}


@dataclass
class CarryParams:
    """Everything the script decides (frozen 2026-09-26). Recorded in every output."""
    # world (m)
    side_off: float = 0.44          # robot base to cube centre along x
    cube_fwd: float = 0.10          # cube centre ahead of the robots' bases (y)
    cube_z: float = 0.33            # cube centre on the plinth (plinth top 0.19 m)
    plinth_half: float = 0.09       # plinth half-width, smaller than the cube
    pair_pitch: float = 1.50        # x distance between pair centres (crew 4)
    # contact model
    hand_pads: bool = True          # box collision pad per hand (hand-mesh AABB)
    pad_friction: float = 1.0       # MuJoCo default; the cube's 1.2 wins (max rule)
    # timeline (s)
    t_settle: float = 1.5
    t_reach: float = 1.5
    t_squeeze: float = 1.5
    t_lift: float = 2.5
    t_hold: float = 1.5
    # carry (oracle-gated)
    carry_speed: float = 0.40       # backward command, m/s (gait deadband ~0.25-0.3)
    carry_goal_m: float = 1.25      # stop once the cube is this far from rest
    carry_gate_lift_m: float = 0.08  # carry starts only once the cube is this high
    k_sync: float = 5.0             # extra backward speed per metre a robot lags its partner
    v_max: float = 0.8              # backward command cap (training range +/-1.0)
    v_min: float = 0.25             # commands below this are sent as 0 (the robot waits)
    # arms
    grasp_kp: float | None = 30.0   # PD kp of the scripted arm; None = deploy.yaml (10)
    grasp_arm_mode: str = "override"  # "override" | "residual" (script + policy swing)
    outer_arm_mode: str = "policy"    # "policy" | "override" (held at 0)
    reset_jitter: float = 0.02        # joint noise at reset (rad, normal)
    xy_jitter: float = 0.01           # base xy noise at reset (m, normal)


@dataclass
class Pair:
    """One cube and the two robots on it."""
    cube_body: int
    cube_geom: int
    plinth_geom: int
    robot_b: int              # at -x, grasps with its RIGHT hand
    robot_a: int              # at +x, grasps with its LEFT hand
    centre_x: float


def _yaw_quat(yaw: float) -> list[float]:
    return [float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0))]


def pair_centres(n_pairs: int, pitch: float) -> list[float]:
    return [(-0.5 * (n_pairs - 1) + k) * pitch for k in range(n_pairs)]


# ------------------------------------------------------------------ model

def hand_pad_frames(robot_xml: Path) -> dict:
    """Box pads matching each hand mesh's bounding box, in the hand-link frame."""
    m = mujoco.MjSpec.from_file(str(robot_xml)).compile()
    out = {}
    for side in ("left", "right"):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"arm_{side}_hand_link")
        gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
        if len(gs) != 1:
            raise RuntimeError(f"expected one visual geom on arm_{side}_hand_link, got {gs}")
        g = gs[0]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        centre = m.geom_pos[g] + R.reshape(3, 3) @ m.geom_aabb[g][:3]
        out[side] = {"pos": centre.tolist(), "quat": m.geom_quat[g].tolist(),
                     "size": m.geom_aabb[g][3:].tolist()}
    return out


def add_hand_pads(spec, frames: dict, friction: float, prefix: str = "") -> None:
    """Add one invisible box collision geom per hand link (massless: the hand
    link's inertial already carries the hand)."""
    for side, f in frames.items():
        body = spec.body(f"{prefix}arm_{side}_hand_link")
        if body is None:
            raise RuntimeError(f"no {prefix}arm_{side}_hand_link in spec")
        g = body.add_geom()
        g.name = f"{prefix}arm_{side}_hand_pad"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.pos = f["pos"]
        g.quat = f["quat"]
        g.size = f["size"]
        g.group = 3
        g.mass = 0.0
        g.friction = [friction, 0.005, 0.0001]
        g.rgba = [0.1, 0.1, 0.1, 0.0]


def lowest_point(model, data, geoms) -> float:
    """Exact lowest world z over the given collision geoms. The conservative
    centre-minus-largest-half-extent rule puts a 0.11 m foot box 9 cm high."""
    lo = np.inf
    for g in geoms:
        c = data.geom_xpos[g]
        R = data.geom_xmat[g].reshape(3, 3)
        sz = model.geom_size[g]
        t = int(model.geom_type[g])
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            z = c[2] - np.abs(R[2, :]) @ sz
        elif t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            ax = R[:, 2]
            r, h = sz[0], sz[1]
            if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                z = c[2] - abs(ax[2]) * h - r
            else:
                z = c[2] - abs(ax[2]) * h - r * np.sqrt(max(0.0, 1.0 - ax[2] ** 2))
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            z = c[2] - sz[0]
        else:
            z = c[2] - sz.max()
        lo = min(lo, float(z))
    return lo


def build_carry(upstream: Path, cache_dir: Path, n_pairs: int, p: CarryParams):
    """Compose 2*n_pairs humanoids, one plinth and one free cube per pair.

    Returns (model, slots, pairs). `slots` are `multi_robot.Slot`s, so
    `MultiRunner`'s observation and PD code drive them unchanged.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    scene = prepare_mjcf(upstream, cache_dir, "humanoid")
    robot_xml = scene.parent / "berkeley_humanoid_lite.xml"
    world_path = cache_dir / "scripted_carry_world.xml"
    world_path.write_text(_WORLD)
    spec = mujoco.MjSpec.from_file(str(world_path))

    centres = pair_centres(n_pairs, p.pair_pitch)
    pads = hand_pad_frames(robot_xml) if p.hand_pads else None
    for i, c in enumerate(centres):
        for j, sx in enumerate((-1.0, +1.0)):          # b then a
            child = mujoco.MjSpec.from_file(str(robot_xml))
            if pads:
                add_hand_pads(child, pads, p.pad_friction)
            frame = spec.worldbody.add_frame()
            frame.pos = [c + sx * p.side_off, 0.0, 0.0]
            frame.quat = _yaw_quat(np.pi / 2)            # facing +y
            frame.attach_body(child.bodies[1], f"r{2 * i + j}_", "")

    plinth_top = p.cube_z - CUBE_HALF
    for k, c in enumerate(centres):
        g = spec.worldbody.add_geom()
        g.name = f"plinth{k}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [p.plinth_half, p.plinth_half, 0.5 * plinth_top]
        g.pos = [c, p.cube_fwd, 0.5 * plinth_top]
        g.rgba = [0.45, 0.45, 0.47, 1.0]
        g.friction = [1.0, 0.005, 0.0001]
        body = spec.worldbody.add_body()
        body.name = f"cube{k}"
        body.pos = [c, p.cube_fwd, p.cube_z]
        body.add_freejoint(name=f"cube{k}_free")
        cg = body.add_geom()
        cg.name = f"cube{k}_g"
        cg.type = mujoco.mjtGeom.mjGEOM_BOX
        cg.size = [CUBE_HALF] * 3
        cg.mass = CUBE_MASS
        cg.friction = [CUBE_FRICTION, 0.005, 0.0001]
        cg.rgba = PAYLOAD_RGBA

    model = spec.compile()

    def sid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)

    slots = []
    for i in range(2 * n_pairs):
        pre = f"r{i}_"
        act = [j for j in range(model.nu)
               if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) or "").startswith(pre)]
        jpos, jvel = [], []
        for j in act:
            base = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)[len(pre):]
            base = base.replace("_joint", "")
            a_p, a_v = sid(f"{pre}{base}_pos"), sid(f"{pre}{base}_vel")
            if a_p < 0 or a_v < 0:
                raise RuntimeError(f"cannot resolve sensors for {pre}{base}")
            jpos.append(model.sensor_adr[a_p])
            jvel.append(model.sensor_adr[a_v])
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{pre}base")
        jid = model.body_jntadr[bid]
        slots.append(Slot(
            prefix=pre, label=f"r{i}", ctrl=np.asarray(act, dtype=int),
            jpos_adr=np.asarray(jpos, dtype=int), jvel_adr=np.asarray(jvel, dtype=int),
            quat_adr=int(model.sensor_adr[sid(f"{pre}imu_quat")]),
            gyro_adr=int(model.sensor_adr[sid(f"{pre}imu_gyro")]),
            qpos_adr=int(model.jnt_qposadr[jid]), qvel_adr=int(model.jnt_dofadr[jid]),
            body_id=int(bid)))
        apply_livery(model, pre)

    pairs = []
    for k, c in enumerate(centres):
        pairs.append(Pair(
            cube_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cube{k}"),
            cube_geom=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"cube{k}_g"),
            plinth_geom=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"plinth{k}"),
            robot_b=2 * k, robot_a=2 * k + 1, centre_x=float(c)))
    return model, slots, pairs


def actuator_joint_names(model, slot: Slot) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(a))[len(slot.prefix):]
            for a in slot.ctrl]


# ------------------------------------------------------------------ arm kinematics

class ArmIK:
    """Forward kinematics of the arm contact geometry on one standing robot.

    Legs at the deploy defaults, base at the origin facing +x, feet planted on
    the floor (exact lowest point), so positions are in the robot's own frame
    with z from the floor. The simulated stance matches this to ~5 mm in height;
    the gait leans ~0.1 rad forward, moving the hands 3-6 cm ahead of it.
    """

    def __init__(self, upstream: Path, cache_dir: Path, leg_default: dict[str, float],
                 hand_pads: bool = True):
        scene = prepare_mjcf(upstream, cache_dir, "humanoid")
        spec = mujoco.MjSpec.from_file(str(scene))
        if hand_pads:
            add_hand_pads(spec, hand_pad_frames(scene.parent / "berkeley_humanoid_lite.xml"), 1.0)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.d.qpos[:] = 0.0
        self.d.qpos[3] = 1.0
        for k, v in leg_default.items():
            self.d.qpos[self.m.jnt_qposadr[self._jid(k)]] = v
        mujoco.mj_forward(self.m, self.d)
        self.d.qpos[2] -= lowest_point(self.m, self.d,
                                       [g for g in range(self.m.ngeom)
                                        if self.m.geom_contype[g] and self.m.geom_bodyid[g] > 0])
        mujoco.mj_forward(self.m, self.d)
        self.pad = {s: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f"arm_{s}_hand_pad")
                    for s in ("left", "right")}

    def _jid(self, n):
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)

    def _set(self, side, q):
        names = ARM_JOINTS_L if side == "left" else ARM_JOINTS_R
        adr = [self.m.jnt_qposadr[self._jid(n)] for n in names]
        self.d.qpos[adr] = q
        mujoco.mj_kinematics(self.m, self.d)

    def pad_fk(self, side: str, q):
        """(centre, rotation matrix, half-sizes) of that side's hand pad."""
        g = self.pad[side]
        if g < 0:
            raise RuntimeError("ArmIK was built without hand pads")
        self._set(side, np.asarray(q, dtype=float))
        return (self.d.geom_xpos[g].copy(), self.d.geom_xmat[g].reshape(3, 3).copy(),
                self.m.geom_size[g].copy())

    def pad_lateral_extent(self, side: str, q) -> float:
        """Outermost lateral reach of the pad toward its own side (m)."""
        c, R, h = self.pad_fk(side, q)
        sgn = 1.0 if side == "left" else -1.0
        corners = [c + R @ (h * np.array([sx, sy, sz]))
                   for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        return float(max(sgn * pt[1] for pt in corners))


def _smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return float(0.5 - 0.5 * np.cos(np.pi * u))


class ArmScript:
    """Open-loop grasping-arm schedule: rest -> reach -> squeeze -> lift, then held.
    Returns the five LEFT-arm targets; the right grasping arm uses -q.
    `hold_forever` (lift-and-hold protocol) only relabels the phase after
    `t_carry` as "hold"; the joint targets are the same."""

    PHASES = ("settle", "reach", "squeeze", "lift")

    def __init__(self, p: CarryParams, keyframes: dict = KEYFRAMES_LEFT, hold_forever: bool = False):
        self.hold_forever = bool(hold_forever)
        k = {n: np.asarray(v, dtype=float) for n, v in keyframes.items()}
        t0 = p.t_settle
        t1 = t0 + p.t_reach
        t2 = t1 + p.t_squeeze
        t3 = t2 + p.t_lift
        self.t_lift_done = t3
        self.t_carry = t3 + p.t_hold
        self.segs = [(0.0, t0, k["rest"], k["rest"]), (t0, t1, k["rest"], k["reach"]),
                     (t1, t2, k["reach"], k["squeeze"]), (t2, t3, k["squeeze"], k["lift"])]
        self.final = k["lift"]

    def phase(self, t: float) -> str:
        for (_, b, _, _), n in zip(self.segs, self.PHASES):
            if t < b:
                return n
        return "hold" if (self.hold_forever or t < self.t_carry) else "carry"

    def __call__(self, t: float) -> np.ndarray:
        for a, b, q0, q1 in self.segs:
            if t < b:
                return q0 + _smooth((t - a) / max(b - a, 1e-6)) * (q1 - q0)
        return self.final.copy()


def compose_arm_targets(policy_q, left_q, grasp_side: str, p: CarryParams) -> np.ndarray:
    """The 22 joint targets sent to the PD loop for one robot: the gait policy's
    output with the grasping arm's five entries replaced by the script (or, in
    "residual" mode, offset by it); the outer arm keeps the policy's output
    unless `outer_arm_mode == "override"`."""
    q = np.asarray(policy_q, dtype=float).copy()
    g_sl, o_sl, s = (LEFT, RIGHT, 1.0) if grasp_side == "left" else (RIGHT, LEFT, -1.0)
    script = s * np.asarray(left_q, dtype=float)
    if p.grasp_arm_mode == "override":
        q[g_sl] = script
    elif p.grasp_arm_mode == "residual":
        q[g_sl] = q[g_sl] + script
    else:
        raise ValueError(p.grasp_arm_mode)
    if p.outer_arm_mode == "override":
        q[o_sl] = 0.0
    elif p.outer_arm_mode != "policy":
        raise ValueError(p.outer_arm_mode)
    return q


def sync_speed(lag: float, p: CarryParams) -> float:
    """Backward command magnitude for a robot `lag` metres behind its partner
    (negative = ahead). Below `v_min` the gait would not step anyway, so the
    robot that is ahead is told to wait (0) instead of creeping."""
    v = float(np.clip(p.carry_speed + p.k_sync * lag, 0.0, p.v_max))
    return v if v >= p.v_min else 0.0


# ------------------------------------------------------------------ runner

class CarryRunner(MultiRunner):
    """`MultiRunner` that keeps spawn yaws, plants the feet, gives the scripted
    arm its own kp, and checks cube-floor contact at every physics step."""

    def __init__(self, model, slots, pairs, cfg, controllers, p: CarryParams,
                 grasp_rest: dict | None = None):
        super().__init__(model, slots, [cfg] * len(slots), controllers)
        self.pairs, self.p = pairs, p
        # robot index -> (slice of its grasping arm in joint order, 5 rest angles)
        self.grasp_rest = grasp_rest or {}
        self.kp_r = [self.kp.copy() for _ in slots]
        if p.grasp_kp is not None:
            for i, (sl, _) in self.grasp_rest.items():
                self.kp_r[i][sl] = p.grasp_kp
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.cube_geoms = np.array([pr.cube_geom for pr in pairs])
        self.cube_floor = np.zeros(len(pairs), dtype=bool)
        owner = np.full(model.ngeom, -1, dtype=int)
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
            for i, s in enumerate(slots):
                if name.startswith(s.prefix):
                    owner[g] = i
        self.owner = owner

    def reset(self, rng):
        mujoco.mj_resetData(self.m, self.d)
        for s in self.slots:
            self.d.qpos[s.qpos_adr:s.qpos_adr + 7] = self.m.qpos0[s.qpos_adr:s.qpos_adr + 7]
            self.d.qpos[s.qpos_adr:s.qpos_adr + 2] += rng.normal(0.0, self.p.xy_jitter, 2)
            n = len(s.ctrl)
            q = self.qdefault + rng.normal(0.0, self.p.reset_jitter, n).astype(np.float32)
            self.d.qpos[s.qpos_adr + 7:s.qpos_adr + 7 + n] = q
        for i, (sl, q5) in self.grasp_rest.items():
            a = self.slots[i].qpos_adr + 7
            self.d.qpos[a + sl.start:a + sl.stop] = (
                np.asarray(q5) + rng.normal(0.0, self.p.reset_jitter, 5))
        mujoco.mj_forward(self.m, self.d)
        for i, s in enumerate(self.slots):       # plant the feet on the floor
            mine = [g for g in np.where(self.owner == i)[0] if self.m.geom_contype[g]]
            self.d.qpos[s.qpos_adr + 2] -= lowest_point(self.m, self.d, mine) - 0.002
        mujoco.mj_forward(self.m, self.d)
        self.alive = [True] * len(self.slots)
        self.cube_floor[:] = False
        for c in self.ctrls:
            c.prev_actions[:] = 0.0
            c.policy_observations[:] = 0.0

    def step(self, targets_per_robot):
        for _ in range(self.substeps):
            for i, s in enumerate(self.slots):
                jp = self.d.sensordata[s.jpos_adr]
                jv = self.d.sensordata[s.jvel_adr]
                tau = self.kp_r[i] * (targets_per_robot[i] - jp) - self.kd * jv
                self.d.ctrl[s.ctrl] = np.clip(tau, -self.eff, self.eff)
            mujoco.mj_step(self.m, self.d)
            n = self.d.ncon
            if n:
                g1 = self.d.contact.geom1[:n]
                g2 = self.d.contact.geom2[:n]
                for k, cg in enumerate(self.cube_geoms):
                    if (((g1 == cg) & (g2 == self.floor)) | ((g2 == cg) & (g1 == self.floor))).any():
                        self.cube_floor[k] = True

    def yaw(self, i):
        q = self.d.qpos[self.slots[i].qpos_adr + 3:self.slots[i].qpos_adr + 7]
        return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))

    def xy(self, i):
        return self.d.xpos[self.slots[i].body_id, :2].copy()

    def cube_tilt(self, pair: Pair) -> float:
        R = self.d.xmat[pair.cube_body].reshape(3, 3)
        return float(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))

    def cube_forces(self, pair: Pair) -> dict:
        """Normal force on the cube (N) from robot b, robot a, and the plinth."""
        out = {"b": 0.0, "a": 0.0, "plinth": 0.0}
        f6 = np.zeros(6)
        for k in range(self.d.ncon):
            c = self.d.contact[k]
            if pair.cube_geom not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == pair.cube_geom else c.geom1
            mujoco.mj_contactForce(self.m, self.d, k, f6)
            if other == pair.plinth_geom:
                out["plinth"] += float(f6[0])
            elif self.owner[other] == pair.robot_b:
                out["b"] += float(f6[0])
            elif self.owner[other] == pair.robot_a:
                out["a"] += float(f6[0])
        return out


def body_frame_command(yaw: float, v_world) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * v_world[0] + s * v_world[1], -s * v_world[0] + c * v_world[1]])


def run_episode(model, slots, pairs, cfg, policy, seed: int, p: CarryParams,
                frame_hook=None, rule: dict | None = None,
                keyframes: dict = KEYFRAMES_LEFT, protocol: str = "carry") -> dict:
    """One seeded episode for every pair in the model.

    protocol "carry" (default): `SUCCESS_RULE`, lift then oracle-gated carry.
    protocol "lift_hold": `LIFT_HOLD_RULE`, no velocity command ever (the
    pair stands and holds after the lift), scored by `score_lift_hold`.
    `rule` defaults to the protocol's own rule."""
    if protocol == "lift_place":
        # separate code path (the carry / lift_hold paths below are untouched)
        return run_place_episode(model, slots, pairs, cfg, policy, seed, p, frame_hook=frame_hook,
                                 rule=rule, keyframes=keyframes)
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}")
    lift_hold = protocol == "lift_hold"
    if rule is None:
        rule = LIFT_HOLD_RULE if lift_hold else SUCCESS_RULE
    if rule.get("protocol") == "lift_place" or lift_hold != (rule.get("protocol") == "lift_hold"):
        raise ValueError(f"rule does not belong to protocol {protocol!r}")
    from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController

    controllers = [RlController(cfg) for _ in slots]
    for c in controllers:
        c.policy = policy
    rest_l = np.asarray(keyframes["rest"], dtype=float)
    grasp_rest, grasp_side = {}, {}
    for pr in pairs:
        grasp_rest[pr.robot_b], grasp_side[pr.robot_b] = (RIGHT, -rest_l), "right"
        grasp_rest[pr.robot_a], grasp_side[pr.robot_a] = (LEFT, rest_l), "left"
    runner = CarryRunner(model, slots, pairs, cfg, controllers, p, grasp_rest)
    for s in slots:
        if actuator_joint_names(model, s) != list(cfg.joints):
            raise RuntimeError(f"{s.prefix} actuator order differs from deploy.yaml joints")
    runner.reset(np.random.default_rng(seed))
    dt = float(cfg.policy_dt)
    script = ArmScript(p, keyframes, hold_forever=lift_hold)
    n_steps = int(round(rule["episode_s"] / dt))

    rest_z = rest_xy = None
    lift = np.zeros((len(pairs), n_steps))
    horiz = np.zeros((len(pairs), n_steps))
    tilt_series = np.zeros((len(slots), n_steps))
    carry_on = [False] * len(pairs)
    carry_done = [False] * len(pairs)
    carry_started = [None] * len(pairs)
    carry_stopped = [None] * len(pairs)
    trace, failed, n_done, t = [], None, 0, 0.0
    for step in range(n_steps):
        t = step * dt
        if not (np.isfinite(runner.d.qpos).all() and np.isfinite(runner.d.qvel).all()):
            failed = "nonfinite_state"
            break
        tilt_series[:, step] = [runner.tilt(i) for i in range(len(slots))]
        cube = np.array([runner.d.xpos[pr.cube_body].copy() for pr in pairs])
        if rest_z is None and t >= p.t_settle - 1e-9:
            rest_z, rest_xy = cube[:, 2].copy(), cube[:, :2].copy()   # on the plinth, settled
        if rest_z is not None:
            lift[:, step] = cube[:, 2] - rest_z
            horiz[:, step] = np.linalg.norm(cube[:, :2] - rest_xy, axis=1)

        commands = [np.zeros(3) for _ in slots]
        if not lift_hold and rest_z is not None and t >= script.t_carry:
            for k, pr in enumerate(pairs):
                if not carry_on[k] and not carry_done[k] and lift[k, step] >= p.carry_gate_lift_m:
                    carry_on[k], carry_started[k] = True, round(t, 3)
                if carry_on[k] and horiz[k, step] >= p.carry_goal_m:
                    carry_on[k], carry_done[k], carry_stopped[k] = False, True, round(t, 3)
                if not carry_on[k]:
                    continue
                for i, j in ((pr.robot_a, pr.robot_b), (pr.robot_b, pr.robot_a)):
                    lag = runner.xy(i)[1] - runner.xy(j)[1]     # + = i is behind (walking -y)
                    cmd = body_frame_command(runner.yaw(i), (0.0, -sync_speed(lag, p)))
                    commands[i] = np.array([cmd[0], cmd[1], 0.0])

        left_q = script(t)
        targets = []
        for i, c in enumerate(controllers):
            q = c.update(runner.observe(i, commands[i]))
            targets.append(compose_arm_targets(q, left_q, grasp_side[i], p))
        if not all(np.isfinite(q).all() for q in targets):
            failed = "nonfinite_action"
            break
        runner.step(targets)
        n_done = step + 1
        if frame_hook is not None:
            frame_hook(step=step, t=t, runner=runner, script=script, pairs=pairs,
                       lift=lift[:, step].copy(), horiz=horiz[:, step].copy(),
                       carry_on=list(carry_on), carry_done=list(carry_done), commands=commands)
        if step % 25 == 0:
            trace.append({
                "t": round(t, 2), "phase": script.phase(t),
                "cube": cube.round(4).tolist(),
                "lift_m": lift[:, step].round(4).tolist(),
                "cube_tilt_rad": [round(runner.cube_tilt(pr), 3) for pr in pairs],
                "robots_xy": [runner.xy(i).round(3).tolist() for i in range(len(slots))],
                "tilt": tilt_series[:, step].round(3).tolist(),
                "yaw": [round(runner.yaw(i), 3) for i in range(len(slots))],
                "cube_normal_N": [{kk: round(v, 2) for kk, v in runner.cube_forces(pr).items()}
                                  for pr in pairs],
                "commands": [np.round(c, 3).tolist() for c in commands]})
    rows = []
    for k, pr in enumerate(pairs):
        mt = float(tilt_series[[pr.robot_a, pr.robot_b], :n_done].max()) if n_done else 0.0
        if lift_hold:
            row = score_lift_hold(lift[k, :n_done], horiz[k, :n_done], dt, max_tilt=mt,
                                  floor=bool(runner.cube_floor[k]), failed=failed, rule=rule)
        else:
            row = score_pair(lift[k, :n_done], horiz[k, :n_done], dt, max_tilt=mt,
                             floor=bool(runner.cube_floor[k]), failed=failed, rule=rule)
        row.update({"pair": k, "seed": seed, "carry_started_s": carry_started[k],
                    "carry_stopped_s": carry_stopped[k],
                    "lift_series_m": lift[k, :n_done:5].round(4).tolist(),
                    "horiz_series_m": horiz[k, :n_done:5].round(4).tolist(),
                    "series_dt_s": round(5 * dt, 3)})
        rows.append(row)
    out = {"seed": seed, "pairs": rows, "failed": failed, "steps": n_done,
           "elapsed_s": round(n_done * dt, 3), "trace": trace}
    if lift_hold:
        out["protocol"] = "lift_hold"
    return out


def score_pair(lift, horiz, dt: float, *, max_tilt: float, floor: bool,
               failed: str | None, rule: dict = SUCCESS_RULE) -> dict:
    """The predeclared rule, applied to one pair's per-step series."""
    lift = np.asarray(lift, dtype=float)
    horiz = np.asarray(horiz, dtype=float)
    up = lift >= rule["lift_hold_m"]
    best = run = 0
    for u in up:
        run = run + 1 if u else 0
        best = max(best, run)
    hold_s = best * dt
    peak = float(lift.max()) if lift.size else 0.0
    carry_m = float(horiz[up].max()) if up.any() else 0.0
    checks = {
        "lift_peak": peak >= rule["lift_peak_m"],
        "lift_hold": hold_s >= rule["lift_hold_s"] - 1e-9,
        "carry": carry_m >= rule["carry_m"],
        "no_fall": max_tilt <= rule["tilt_rad"],
        "no_floor_contact": not floor,
        "finite": failed is None,
        "full_episode": lift.size * dt >= rule["episode_s"] - 1e-6,
    }
    success = all(checks.values())
    completion = None
    if success:
        # first time the cube stood >= carry_m from rest while >= lift_hold_m up
        completion = round(float(np.where(up & (horiz >= rule["carry_m"]))[0][0] * dt), 3)
    return {"success": bool(success),
            "first_failed_check": None if success else next(k for k, v in checks.items() if not v),
            "checks": checks, "lift_peak_m": round(peak, 4), "lift_hold_s": round(hold_s, 3),
            "carry_m": round(carry_m, 4), "max_tilt_rad": round(max_tilt, 4),
            "cube_floor_contact": bool(floor), "completion_s": completion,
            "final_lift_m": round(float(lift[-1]), 4) if lift.size else None}


def summarize(episodes: list[dict], n_pairs: int, rule: dict = SUCCESS_RULE) -> dict:
    """Per-pair success counts and the PASS verdict under the predeclared rule.
    Complete only when every predeclared seed has been run."""
    seeds_run = sorted({e["seed"] for e in episodes})
    complete = all(s in seeds_run for s in rule["seeds"])
    per_pair = []
    for k in range(n_pairs):
        rows = [e["pairs"][k] for e in episodes if e["seed"] in rule["seeds"]]
        wins = [r for r in rows if r["success"]]
        failed = sorted({r["first_failed_check"] for r in rows} - {None})
        per_pair.append({
            "pair": k, "episodes": len(rows), "successes": len(wins),
            "pass": bool(complete and len(wins) >= rule["pass_min"]),
            "first_failed_check_counts": {c: sum(1 for r in rows if r["first_failed_check"] == c)
                                          for c in failed},
            "lift_peak_m_median": float(np.median([r["lift_peak_m"] for r in rows])) if rows else None,
            "lift_hold_s_median": float(np.median([r["lift_hold_s"] for r in rows])) if rows else None,
            "carry_m_median": float(np.median([r["carry_m"] for r in rows])) if rows else None,
            "carry_m_max": float(max(r["carry_m"] for r in rows)) if rows else None,
        })
    return {"per_pair": per_pair, "complete": bool(complete), "seeds_run": seeds_run,
            "pass": bool(complete and all(pp["pass"] for pp in per_pair))}


def median_seed(episodes: list[dict]) -> int | None:
    """Seed of the median-by-completion-time successful episode (every pair
    successful; an episode completes when its slowest pair does). With an even
    count the lower median is taken, so the choice is never the faster one."""
    done = sorted((max(r["completion_s"] for r in e["pairs"]), e["seed"])
                  for e in episodes if all(r["success"] for r in e["pairs"]))
    if not done:
        return None
    return done[(len(done) - 1) // 2][1]


def params_dict(p: CarryParams) -> dict:
    return asdict(p)


# ------------------------------------------------------------------ lift and hold

def _longest_run(mask) -> tuple[int, int]:
    """(length, start index) of the longest run of True; the first one on ties."""
    best = run = 0
    start = best_start = 0
    for i, u in enumerate(mask):
        if u:
            if run == 0:
                start = i
            run += 1
            if run > best:
                best, best_start = run, start
        else:
            run = 0
    return best, best_start


def score_lift_hold(lift, horiz, dt: float, *, max_tilt: float, floor: bool,
                    failed: str | None, rule: dict = LIFT_HOLD_RULE) -> dict:
    """`LIFT_HOLD_RULE` applied to one pair's per-step series. Same clauses
    and the same longest-contiguous-run hold as `score_pair`, without the
    carry clause; `horiz` is reported (cube drift) but never scored."""
    lift = np.asarray(lift, dtype=float)
    horiz = np.asarray(horiz, dtype=float)
    up = lift >= rule["lift_hold_m"]
    best, start = _longest_run(up)
    hold_s = best * dt
    peak = float(lift.max()) if lift.size else 0.0
    checks = {
        "lift_peak": peak >= rule["lift_peak_m"],
        "lift_hold": hold_s >= rule["lift_hold_s"] - 1e-9,
        "no_fall": max_tilt <= rule["tilt_rad"],
        "no_floor_contact": not floor,
        "finite": failed is None,
        "full_episode": lift.size * dt >= rule["episode_s"] - 1e-6,
    }
    success = all(checks.values())
    return {"success": bool(success),
            "first_failed_check": None if success else next(k for k, v in checks.items() if not v),
            "checks": checks, "lift_peak_m": round(peak, 4), "lift_hold_s": round(hold_s, 3),
            "hold_start_s": round(start * dt, 3) if best else None,
            "hold_end_s": round((start + best) * dt, 3) if best else None,
            "max_tilt_rad": round(max_tilt, 4), "cube_floor_contact": bool(floor),
            "horiz_max_m": round(float(horiz.max()), 4) if horiz.size else 0.0,
            "final_lift_m": round(float(lift[-1]), 4) if lift.size else None}


def summarize_lift_hold(episodes: list[dict], n_pairs: int, rule: dict = LIFT_HOLD_RULE) -> dict:
    """Per-pair success counts and the PASS verdict under `LIFT_HOLD_RULE`.
    Complete only when every predeclared seed has been run."""
    seeds_run = sorted({e["seed"] for e in episodes})
    complete = all(s in seeds_run for s in rule["seeds"])
    per_pair = []
    for k in range(n_pairs):
        rows = [e["pairs"][k] for e in episodes if e["seed"] in rule["seeds"]]
        wins = [r for r in rows if r["success"]]
        failed = sorted({r["first_failed_check"] for r in rows} - {None})
        per_pair.append({
            "pair": k, "episodes": len(rows), "successes": len(wins),
            "pass": bool(complete and len(wins) >= rule["pass_min"]),
            "first_failed_check_counts": {c: sum(1 for r in rows if r["first_failed_check"] == c)
                                          for c in failed},
            "lift_peak_m_median": float(np.median([r["lift_peak_m"] for r in rows])) if rows else None,
            "lift_hold_s_median": float(np.median([r["lift_hold_s"] for r in rows])) if rows else None,
            "lift_hold_s_max": float(max(r["lift_hold_s"] for r in rows)) if rows else None,
            "falls": sum(1 for r in rows if r["max_tilt_rad"] > rule["tilt_rad"]),
            "floor_contacts": sum(1 for r in rows if r["cube_floor_contact"]),
        })
    return {"per_pair": per_pair, "complete": bool(complete), "seeds_run": seeds_run,
            "pass": bool(complete and all(pp["pass"] for pp in per_pair))}


def median_seed_by_hold(episodes: list[dict]) -> int | None:
    """Seed of the median-by-hold-duration successful episode (every pair
    successful; an episode's hold is its weakest pair's longest hold). Sorted
    ascending with the lower median on an even count, so the choice is never
    the longer-holding (better-looking) one; ties break on the lower seed."""
    done = sorted((min(r["lift_hold_s"] for r in e["pairs"]), e["seed"])
                  for e in episodes if all(r["success"] for r in e["pairs"]))
    if not done:
        return None
    return done[(len(done) - 1) // 2][1]


def lift_hold_verdict_line(payload: dict, crew=None) -> str:
    """`COOP-LIFT-HOLD crew N: PASS|NEGATIVE|INCOMPLETE|INVALID ...`, computed
    from a lift-and-hold score JSON (its episodes, re-summarised here)."""
    crew = payload.get("crew", crew) if crew is None else crew
    label = payload.get("label", LABEL)
    if payload.get("protocol") != "lift_hold" or payload.get("rule") != LIFT_HOLD_RULE:
        return (f"COOP-LIFT-HOLD crew {crew}: INVALID (not a lift_hold score JSON under the "
                f"predeclared LIFT_HOLD_RULE) | {LIFT_HOLD_NOTE} | {label}")
    n_pairs = int(payload.get("pairs", int(crew) // 2))
    s = summarize_lift_hold(payload.get("episodes", []), n_pairs, LIFT_HOLD_RULE)
    median = median_seed_by_hold([e for e in payload.get("episodes", [])
                                  if e["seed"] in LIFT_HOLD_RULE["seeds"]])
    if not (payload.get("scored_run") and s["complete"]):
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS" if s["pass"] else "NEGATIVE"
    pp = " ".join(
        f"pair{p['pair']}={p['successes']}/{p['episodes']} (first failed: {p['first_failed_check_counts']}, "
        f"median lift {p['lift_peak_m_median']:.3f} m, median hold {p['lift_hold_s_median']:.2f} s, "
        f"max hold {p['lift_hold_s_max']:.2f} s, falls {p['falls']}, floor contacts {p['floor_contacts']})"
        for p in s["per_pair"] if p["episodes"])
    return (f"COOP-LIFT-HOLD crew {crew}: {verdict} | {pp or 'no scored episodes'} | "
            f"median_seed_by_hold={median} | {LIFT_HOLD_NOTE} | {label}")


def enforce_gif_budget(gif: dict | None, gif_path, sidecar_path=None) -> dict:
    """Render gate: a GIF over the committable budget is never left behind.
    Deletes `gif_path` (and `sidecar_path`) unless `gif["within_budget"]`."""
    gif_path = Path(gif_path)
    if gif is not None and gif.get("within_budget"):
        return {"kept": True, "deleted": []}
    deleted = []
    for f in (gif_path, Path(sidecar_path) if sidecar_path else None):
        if f is not None and f.exists():
            f.unlink()
            deleted.append(str(f))
    return {"kept": False, "deleted": deleted}


# ------------------------------------------------------------------ lift, hold and place
#
# A third, separate experiment (`protocol="lift_place"`): the same harness,
# layout, hand pads, kp-30 grasping arm, cube and label, but after the lift the
# pair holds the cube briefly, LOWERS it back onto its plinth by reversing the
# lift's shoulder pitch only (pitch-only lowering: lift keyframe -> PlaceParams.
# lower_to = (0.6, 1.1, 0, 1.0, 0), i.e. shoulder pitch -0.5 -> +0.6, past the
# squeeze keyframe's 0.0, with the lift keyframe's roll 1.1 and elbow 1.0 held;
# a TUNED target, not a plain reverse of the keyframes -- see "exploration"
# below), opens the hands (shoulder roll -> the reach keyframe's -0.2), lets the
# arms hang (-> rest) and stands to the end of a 20 s episode. The robots never
# walk.
#
# Modelling choices of this protocol (recorded in every output):
# * lowering time: the lowering starts early enough that the cube is back on
#   the plinth before the earliest grip slip seen under the lift_hold protocol
#   on exploration seeds 100-119 (slip = end of the longest >= 5 cm run; 20
#   seeds, 11.36-18.92 s, never a scored seed 0-19);
# * lowering target: pitch-only lowering from the lift keyframe to
#   PlaceParams.lower_to (tuned on exploration seeds 100-102 and 110-114; pure
#   reverse-keyframe lowering, lift -> squeeze, placed 0/11 and was rejected);
# * station keeping (optional, `PlaceParams.station_keep`): small velocity
#   commands from the ORACLE robot base poses that pull each robot back toward
#   the xy it stood on at the end of settling.
#
# Exploration (2026-09-27, seeds >= 100 only; no seed 0-19 of this protocol run):
# * slip times under lift_hold, crew 2, seeds 100-119 (end of the longest >= 5 cm
#   run): 11.36, 11.68, 11.80, 12.44, 12.84, 13.04, 13.24, 14.04, 14.36, 15.28,
#   15.64, 15.96, 16.08, 16.32, 16.64, 16.84, 17.32, 17.52, 17.88, 18.92 s. The
#   cube is >= 5 cm up from 6.2-6.6 s; the arm keeps rising until ~9 s (seed
#   101: 0.095 m at 8 s), so lowering runs 9.2 -> 10.7 s. Robot bases drift only
#   3-5 cm after settling; the 0.36-0.42 m "drift" in the lift_hold notes is the
#   cube's horizontal travel (it rolls ~90 deg about x and moves ~0.25 m forward,
#   y 0.10 -> 0.35, during the lift).
# * lowering target (crew 2): pure reverse (lift -> squeeze) placed 0/11 (three
#   timings, seeds 100-102 and 110-114): the cube comes down ~12 cm forward of the plinth centre,
#   lands on the front edge and tips off at release. Also tried and rejected:
#   an elbow-first via point (grip lost mid-air), roll-held targets, pitched-back
#   targets with the squeeze elbow, a slower lowering, a slower opening, two
#   "unload" via points before opening, and station keeping. Best: shoulder pitch
#   reversed past the squeeze pitch with elbow and roll held, -> (0.6, 1.1, 0, 1.0,
#   0): 1/5 on seeds 110-114 (tie with pitch 0.3 broken by fewer floor contacts),
#   frozen as `PlaceParams` defaults.
# * pilot of the frozen setting: crew 2 seeds 100-109 1/10 (station keeping on:
#   0/10; robot-a drift hold->release max 0.19 m off, 0.07 m on); crew 4 seeds
#   100-104 0/5 per pair. The hand that ends on the cube's top face drags it off
#   the plinth as it opens, often pulling that robot over.

LIFT_PLACE_NOTE = "cooperative lift, hold and place — scripted arms, frozen learned gait; carry not achieved"
LIFT_PLACE_LABEL_DETAIL = (
    "legs + outer arm: frozen learned PPO gait, unmodified observation | grasping arm (one per "
    "robot): scripted joint targets reach/squeeze/lift/hold, pitch-only lowering (tuned target), open, rest; "
    "PD kp 30 (deploy 10), 4 Nm cap | cube pose: simulator oracle (scoring only) | robot base "
    "poses: simulator oracle (station keeping, when on) | contact: MuJoCo, added box pads = "
    "hand-mesh AABB, no welds")

#: Cooperative LIFT, HOLD and PLACE rule. Proposed after the pilot on
#: seeds >= 100 and recorded in SLURM_JOBS.md before any scored seed 10-19 ran.
LIFT_PLACE_RULE = {
    "protocol": "lift_place",
    "episode_s": 20.0,
    "lift_peak_m": 0.10,          # cube rises >= 10 cm above its plinth rest height
    "lift_hold_m": 0.05,          # ...and stays >= 5 cm up
    "lift_hold_s": 3.0,           # ...for >= 3.0 s continuously
    "final_height_tol_m": 0.03,   # final step: |cube z - rest z| <= 3 cm
    "final_speed_mps": 0.05,      # final step: cube linear speed < 5 cm/s
    "plinth_half_m": 0.09,        # final step: cube centre within the plinth top (|dx|, |dy| <= 9 cm)
    "released": True,            # final step: no robot geom touches the cube
    "tilt_rad": TILT_LIMIT,       # no robot of the pair tilts past this at any policy step
    "floor_contact": False,       # the cube never touches the floor (every physics step)
    "seeds": list(range(10, 20)),  # reset-jitter seeds reserved for this protocol
    "pass_min": 8,                # PASS iff >= 8/10 seeds succeed (per pair for crew 4)
}


@dataclass
class PlaceParams:
    """What the lift-hold-place script adds to `CarryParams` (which it uses
    unchanged). Recorded in every output."""
    t_hold: float = 2.2             # hold after the lift keyframe is reached: lowering 9.2 -> 10.7 s
    t_via: float = 0.0              # lift -> lower_via (0 = no via point)
    t_lower: float = 1.5            # (lower_via or lift) -> lower_to
    t_unload: float = 0.0           # lower_to -> release_via (0 = no via point)
    t_release: float = 1.0          # -> (release_via or lower_to) with the reach keyframe's roll (hands open)
    t_retract: float = 1.0          # -> rest (arms hang), then stand
    lower_via: tuple | None = None  # optional intermediate LEFT-arm pose (e.g. pull back at height)
    # None = the squeeze keyframe (pure reverse-keyframe lowering). Frozen choice: reverse the lift's
    # shoulder pitch only, past the squeeze pitch (-0.5 -> +0.6), elbow and roll held at the lift
    # keyframe's values -- see "exploration" in the lift-place section header.
    lower_to: tuple | None = (0.6, 1.1, 0.0, 1.0, 0.0)
    release_via: tuple | None = None  # optional pose after lowering, before the hands open
    station_keep: bool = False      # oracle-pose velocity commands toward each robot's settle xy
    k_station: float = 2.0          # command (m/s) per metre of displacement
    v_station_max: float = 0.30     # command cap (m/s)
    v_station_min: float = 0.0      # commands below this are sent as 0
    station_from: str = "settle"    # station keeping on from t_settle ...
    station_until: str = "release"  # ... to the end of this phase ("release" | "episode")


_ARM_JOINTS = ("shoulder pitch", "shoulder roll", "shoulder yaw", "elbow pitch", "elbow roll")  # = ARM_JOINTS_L order


def describe_lower_target(target) -> str:
    """Human-readable provenance of a LEFT-arm lowering target: for each joint,
    whether it is held at the lift keyframe's value, equals the squeeze
    keyframe's, or moves to a new value. Computed from the numbers, so the
    recorded description can never disagree with the target."""
    t = np.asarray(target, dtype=float)
    lift, sq = np.asarray(KEYFRAMES_LEFT["lift"]), np.asarray(KEYFRAMES_LEFT["squeeze"])
    if np.allclose(t, sq):
        return "= the squeeze keyframe (pure reverse-keyframe lowering, lift -> squeeze)"
    moved = [i for i in range(len(t)) if not np.isclose(t[i], lift[i])]
    parts = []
    for i, name in enumerate(_ARM_JOINTS):
        if np.isclose(t[i], lift[i]) and np.isclose(t[i], sq[i]):
            parts.append(f"{name} {t[i]:g} (lift = squeeze keyframe)")
        elif np.isclose(t[i], lift[i]):
            parts.append(f"{name} {t[i]:g} held at the lift keyframe's value")
        elif np.isclose(t[i], sq[i]):
            parts.append(f"{name} {lift[i]:g} -> {t[i]:g} (the squeeze keyframe's value)")
        else:
            parts.append(f"{name} {lift[i]:g} -> {t[i]:g} (the squeeze keyframe has {sq[i]:g})")
    kind = ("pitch-only lowering" if moved == [0] else
            "lowering moving " + ", ".join(_ARM_JOINTS[i] for i in moved))
    return f"from the lift keyframe, {kind} (tuned target): " + "; ".join(parts)


def place_params_dict(q: PlaceParams) -> dict:
    return asdict(q)


class PlaceScript:
    """Grasping-arm schedule of the lift-hold-place protocol (LEFT-arm targets;
    the right grasping arm uses -q). Identical to `ArmScript` up to the end of
    the lift, then hold -> lower -> release -> retract -> stand."""

    PHASES = ("settle", "reach", "squeeze", "lift", "hold", "lower_via", "lower", "unload", "release",
              "retract")

    def __init__(self, p: CarryParams, q: PlaceParams, keyframes: dict = KEYFRAMES_LEFT):
        k = {n: np.asarray(v, dtype=float) for n, v in keyframes.items()}
        lower = k["squeeze"].copy() if q.lower_to is None else np.asarray(q.lower_to, dtype=float)
        via = k["lift"].copy() if q.lower_via is None else np.asarray(q.lower_via, dtype=float)
        if q.lower_via is None and q.t_via != 0.0:
            raise ValueError("t_via without lower_via")
        unload = lower.copy() if q.release_via is None else np.asarray(q.release_via, dtype=float)
        if q.release_via is None and q.t_unload != 0.0:
            raise ValueError("t_unload without release_via")
        opened = unload.copy()
        opened[1] = k["reach"][1]                      # open: the reach keyframe's shoulder roll
        self.lower_target, self.via_target, self.open_target = lower, via, opened
        ts = np.cumsum([p.t_settle, p.t_reach, p.t_squeeze, p.t_lift, q.t_hold, q.t_via, q.t_lower,
                        q.t_unload, q.t_release, q.t_retract])
        self.t_end = {n: float(t) for n, t in zip(self.PHASES, ts)}
        pts = [k["rest"], k["rest"], k["reach"], k["squeeze"], k["lift"], k["lift"], via, lower,
               unload, opened, k["rest"]]
        starts = [0.0] + [float(t) for t in ts[:-1]]
        self.segs = [(a, float(b), pts[i], pts[i + 1]) for i, (a, b) in enumerate(zip(starts, ts))]
        self.t_lift_done = self.t_end["lift"]
        self.t_lower_start = self.t_end["hold"]
        self.t_released = self.t_end["release"]
        self.final = k["rest"]

    def phase(self, t: float) -> str:
        for (_, b, _, _), n in zip(self.segs, self.PHASES):
            if t < b:
                return n
        return "stand"

    def __call__(self, t: float) -> np.ndarray:
        for a, b, q0, q1 in self.segs:
            if t < b:
                return q0 + _smooth((t - a) / max(b - a, 1e-6)) * (q1 - q0)
        return self.final.copy()


def station_command(xy, ref, yaw: float, q: PlaceParams) -> np.ndarray:
    """Body-frame (vx, vy, wz) pulling a robot back toward `ref` (oracle pose)."""
    v = -q.k_station * (np.asarray(xy, dtype=float) - np.asarray(ref, dtype=float))
    n = float(np.linalg.norm(v))
    if n > q.v_station_max:
        v = v * (q.v_station_max / n)
        n = q.v_station_max
    if n < q.v_station_min:
        return np.zeros(3)
    b = body_frame_command(yaw, v)
    return np.array([b[0], b[1], 0.0])


def _cube_robot_contact(runner, pair: Pair) -> bool:
    """True iff any robot geom touches this pair's cube right now."""
    for k in range(runner.d.ncon):
        c = runner.d.contact[k]
        if pair.cube_geom == c.geom1 and runner.owner[c.geom2] >= 0:
            return True
        if pair.cube_geom == c.geom2 and runner.owner[c.geom1] >= 0:
            return True
    return False


def run_place_episode(model, slots, pairs, cfg, policy, seed: int, p: CarryParams,
                      q: PlaceParams | None = None, frame_hook=None, rule: dict | None = None,
                      keyframes: dict = KEYFRAMES_LEFT) -> dict:
    """One seeded lift-hold-place episode for every pair in the model, scored
    by `score_lift_place` under `rule` (default `LIFT_PLACE_RULE`)."""
    q = PlaceParams() if q is None else q
    rule = LIFT_PLACE_RULE if rule is None else rule
    if rule.get("protocol") != "lift_place":
        raise ValueError("rule does not belong to protocol 'lift_place'")
    from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController

    controllers = [RlController(cfg) for _ in slots]
    for c in controllers:
        c.policy = policy
    rest_l = np.asarray(keyframes["rest"], dtype=float)
    grasp_rest, grasp_side = {}, {}
    for pr in pairs:
        grasp_rest[pr.robot_b], grasp_side[pr.robot_b] = (RIGHT, -rest_l), "right"
        grasp_rest[pr.robot_a], grasp_side[pr.robot_a] = (LEFT, rest_l), "left"
    runner = CarryRunner(model, slots, pairs, cfg, controllers, p, grasp_rest)
    for s in slots:
        if actuator_joint_names(model, s) != list(cfg.joints):
            raise RuntimeError(f"{s.prefix} actuator order differs from deploy.yaml joints")
    runner.reset(np.random.default_rng(seed))
    dt = float(cfg.policy_dt)
    script = PlaceScript(p, q, keyframes)
    n_steps = int(round(rule["episode_s"] / dt))
    t_on = p.t_settle if q.station_from == "settle" else script.t_lift_done
    t_off = script.t_released if q.station_until == "release" else np.inf

    rest_z = rest_xy = ref_xy = None
    floor_at = [None] * len(pairs)                     # (t, phase) of the first cube-floor contact
    grip_lost_at = [None] * len(pairs)                 # first step after the lift with no robot contact
    lift = np.zeros((len(pairs), n_steps))
    horiz = np.zeros((len(pairs), n_steps))
    tilt_series = np.zeros((len(slots), n_steps))
    disp = np.zeros((len(slots), n_steps))
    trace, failed, n_done, t = [], None, 0, 0.0
    for step in range(n_steps):
        t = step * dt
        if not (np.isfinite(runner.d.qpos).all() and np.isfinite(runner.d.qvel).all()):
            failed = "nonfinite_state"
            break
        tilt_series[:, step] = [runner.tilt(i) for i in range(len(slots))]
        cube = np.array([runner.d.xpos[pr.cube_body].copy() for pr in pairs])
        if rest_z is None and t >= p.t_settle - 1e-9:
            rest_z, rest_xy = cube[:, 2].copy(), cube[:, :2].copy()   # on the plinth, settled
            ref_xy = [runner.xy(i) for i in range(len(slots))]        # station-keeping reference
        if rest_z is not None:
            lift[:, step] = cube[:, 2] - rest_z
            horiz[:, step] = np.linalg.norm(cube[:, :2] - rest_xy, axis=1)
            disp[:, step] = [np.linalg.norm(runner.xy(i) - ref_xy[i]) for i in range(len(slots))]

        commands = [np.zeros(3) for _ in slots]
        if q.station_keep and ref_xy is not None and t_on - 1e-9 <= t < t_off:
            commands = [station_command(runner.xy(i), ref_xy[i], runner.yaw(i), q)
                        for i in range(len(slots))]

        left_q = script(t)
        targets = []
        for i, c in enumerate(controllers):
            a = c.update(runner.observe(i, commands[i]))
            targets.append(compose_arm_targets(a, left_q, grasp_side[i], p))
        if not all(np.isfinite(x).all() for x in targets):
            failed = "nonfinite_action"
            break
        runner.step(targets)
        n_done = step + 1
        for k, pr in enumerate(pairs):
            if floor_at[k] is None and runner.cube_floor[k]:
                floor_at[k] = (round(t, 3), script.phase(t))
            if (grip_lost_at[k] is None and script.t_lift_done <= t < script.t_lower_start + q.t_via
                    + q.t_lower and not _cube_robot_contact(runner, pr)):
                grip_lost_at[k] = (round(t, 3), script.phase(t))
        if frame_hook is not None:
            frame_hook(step=step, t=t, runner=runner, script=script, pairs=pairs,
                       lift=lift[:, step].copy(), horiz=horiz[:, step].copy(),
                       carry_on=[False] * len(pairs), carry_done=[False] * len(pairs),
                       commands=commands)
        if step % 25 == 0:
            trace.append({
                "t": round(t, 2), "phase": script.phase(t),
                "cube": cube.round(4).tolist(),
                "lift_m": lift[:, step].round(4).tolist(),
                "cube_tilt_rad": [round(runner.cube_tilt(pr), 3) for pr in pairs],
                "robots_xy": [runner.xy(i).round(3).tolist() for i in range(len(slots))],
                "tilt": tilt_series[:, step].round(3).tolist(),
                "station_disp_m": disp[:, step].round(3).tolist(),
                "cube_normal_N": [{kk: round(v, 2) for kk, v in runner.cube_forces(pr).items()}
                                  for pr in pairs],
                "commands": [np.round(c, 3).tolist() for c in commands]})
    rows = []
    i_hold = int(round(script.t_lift_done / dt))
    i_rel = int(round(script.t_released / dt))
    for k, pr in enumerate(pairs):
        mt = float(tilt_series[[pr.robot_a, pr.robot_b], :n_done].max()) if n_done else 0.0
        plinth_xy = model.geom_pos[pr.plinth_geom][:2]
        cube_now = runner.d.xpos[pr.cube_body]
        jadr = model.body_jntadr[pr.cube_body]
        vel = runner.d.qvel[model.jnt_dofadr[jadr]:model.jnt_dofadr[jadr] + 3]
        final = {
            "dz_m": float(cube_now[2] - rest_z[k]) if rest_z is not None else None,
            "speed_mps": float(np.linalg.norm(vel)),
            "offset_xy_m": [float(v) for v in (cube_now[:2] - plinth_xy)],
            "cube_tilt_rad": runner.cube_tilt(pr),
            "robot_contact": _cube_robot_contact(runner, pr),
            "normal_N": {kk: round(v, 2) for kk, v in runner.cube_forces(pr).items()},
        }
        row = score_lift_place(lift[k, :n_done], dt, max_tilt=mt, floor=bool(runner.cube_floor[k]),
                               failed=failed, final=final, rule=rule)
        both = [pr.robot_b, pr.robot_a]
        row.update({
            "pair": k, "seed": seed,
            # diagnostics (never scored): when the cube first touched the floor, and the first
            # step between the end of the lift and the end of the lowering with no hand on it
            "floor_contact_first": None if floor_at[k] is None else
                {"t_s": floor_at[k][0], "phase": floor_at[k][1]},
            "grip_lost_first": None if grip_lost_at[k] is None else
                {"t_s": grip_lost_at[k][0], "phase": grip_lost_at[k][1]},
            "station_disp_max_m": {
                "hold_to_release": [round(float(disp[i, i_hold:min(i_rel, n_done)].max()), 4)
                                    if n_done > i_hold else None for i in both],
                "episode": [round(float(disp[i, :n_done].max()), 4) if n_done else None for i in both]},
            "lift_series_m": lift[k, :n_done:5].round(4).tolist(),
            "horiz_series_m": horiz[k, :n_done:5].round(4).tolist(),
            "series_dt_s": round(5 * dt, 3)})
        rows.append(row)
    return {"seed": seed, "pairs": rows, "failed": failed, "steps": n_done,
            "elapsed_s": round(n_done * dt, 3), "trace": trace, "protocol": "lift_place",
            "place_params": place_params_dict(q),
            "script_times_s": {n: round(v, 3) for n, v in script.t_end.items()}}


def score_lift_place(lift, dt: float, *, max_tilt: float, floor: bool, failed: str | None,
                     final: dict, rule: dict = LIFT_PLACE_RULE) -> dict:
    """`LIFT_PLACE_RULE` applied to one pair: the lift-hold clauses (longest
    contiguous run, as `score_lift_hold`) plus the final cube state."""
    lift = np.asarray(lift, dtype=float)
    up = lift >= rule["lift_hold_m"]
    best, start = _longest_run(up)
    hold_s = best * dt
    peak = float(lift.max()) if lift.size else 0.0
    dz = final.get("dz_m")
    off = final.get("offset_xy_m") or [np.inf, np.inf]
    checks = {
        "lift_peak": peak >= rule["lift_peak_m"],
        "lift_hold": hold_s >= rule["lift_hold_s"] - 1e-9,
        "on_plinth_height": dz is not None and abs(dz) <= rule["final_height_tol_m"],
        "at_rest": final.get("speed_mps", np.inf) < rule["final_speed_mps"],
        "on_plinth_footprint": max(abs(off[0]), abs(off[1])) <= rule["plinth_half_m"],
        "released": (not final.get("robot_contact", True)) if rule.get("released") else True,
        "no_fall": max_tilt <= rule["tilt_rad"],
        "no_floor_contact": not floor,
        "finite": failed is None,
        "full_episode": lift.size * dt >= rule["episode_s"] - 1e-6,
    }
    success = all(checks.values())
    return {"success": bool(success),
            "first_failed_check": None if success else next(k for k, v in checks.items() if not v),
            "checks": checks, "lift_peak_m": round(peak, 4), "lift_hold_s": round(hold_s, 3),
            "hold_start_s": round(start * dt, 3) if best else None,
            "hold_end_s": round((start + best) * dt, 3) if best else None,
            "max_tilt_rad": round(max_tilt, 4), "cube_floor_contact": bool(floor),
            "final_dz_m": None if dz is None else round(float(dz), 4),
            "final_speed_mps": round(float(final.get("speed_mps", np.nan)), 4),
            "final_offset_xy_m": [round(float(v), 4) for v in off],
            "final_cube_tilt_rad": round(float(final.get("cube_tilt_rad", np.nan)), 4),
            "final_robot_contact": bool(final.get("robot_contact", True)),
            "final_normal_N": final.get("normal_N"),
            "final_lift_m": round(float(lift[-1]), 4) if lift.size else None}


def summarize_lift_place(episodes: list[dict], n_pairs: int, rule: dict = LIFT_PLACE_RULE) -> dict:
    """Per-pair success counts and the PASS verdict under `LIFT_PLACE_RULE`.
    Complete only when every predeclared seed has been run."""
    seeds_run = sorted({e["seed"] for e in episodes})
    complete = all(s in seeds_run for s in rule["seeds"])
    per_pair = []
    for k in range(n_pairs):
        rows = [e["pairs"][k] for e in episodes if e["seed"] in rule["seeds"]]
        wins = [r for r in rows if r["success"]]
        failed = sorted({r["first_failed_check"] for r in rows} - {None})
        per_pair.append({
            "pair": k, "episodes": len(rows), "successes": len(wins),
            "pass": bool(complete and len(wins) >= rule["pass_min"]),
            "first_failed_check_counts": {c: sum(1 for r in rows if r["first_failed_check"] == c)
                                          for c in failed},
            "lift_peak_m_median": float(np.median([r["lift_peak_m"] for r in rows])) if rows else None,
            "lift_hold_s_median": float(np.median([r["lift_hold_s"] for r in rows])) if rows else None,
            "placed_on_plinth": sum(1 for r in rows if r["checks"]["on_plinth_height"]
                                    and r["checks"]["on_plinth_footprint"]),
            "falls": sum(1 for r in rows if r["max_tilt_rad"] > rule["tilt_rad"]),
            "floor_contacts": sum(1 for r in rows if r["cube_floor_contact"]),
        })
    return {"per_pair": per_pair, "complete": bool(complete), "seeds_run": seeds_run,
            "pass": bool(complete and all(pp["pass"] for pp in per_pair))}


def lift_place_verdict_line(payload: dict, crew=None) -> str:
    """`COOP-LIFT-PLACE crew N: PASS|NEGATIVE|INCOMPLETE|INVALID ...`, computed
    from a lift-hold-place score JSON (its episodes, re-summarised here)."""
    crew = payload.get("crew", crew) if crew is None else crew
    label = payload.get("label", LABEL)
    if payload.get("protocol") != "lift_place" or payload.get("rule") != LIFT_PLACE_RULE:
        return (f"COOP-LIFT-PLACE crew {crew}: INVALID (not a lift_place score JSON under the "
                f"predeclared LIFT_PLACE_RULE) | {LIFT_PLACE_NOTE} | {label}")
    n_pairs = int(payload.get("pairs", int(crew) // 2))
    eps = payload.get("episodes", [])
    s = summarize_lift_place(eps, n_pairs, LIFT_PLACE_RULE)
    median = median_seed_by_hold([e for e in eps if e["seed"] in LIFT_PLACE_RULE["seeds"]])
    if not (payload.get("scored_run") and s["complete"]):
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS" if s["pass"] else "NEGATIVE"
    pp = " ".join(
        f"pair{p['pair']}={p['successes']}/{p['episodes']} (first failed: {p['first_failed_check_counts']}, "
        f"median lift {p['lift_peak_m_median']:.3f} m, median hold {p['lift_hold_s_median']:.2f} s, "
        f"placed {p['placed_on_plinth']}, falls {p['falls']}, floor contacts {p['floor_contacts']})"
        for p in s["per_pair"] if p["episodes"])
    return (f"COOP-LIFT-PLACE crew {crew}: {verdict} | {pp or 'no scored episodes'} | "
            f"median_seed_by_hold={median} | {LIFT_PLACE_NOTE} | {label}")
