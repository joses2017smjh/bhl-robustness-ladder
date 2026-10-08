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
    flush = is_flushpad(p)        # opt-in C1 flush-pad variant (FlushPadParams); False for a stock CarryParams
    if flush:
        pads = flush_pad_frames(robot_xml, p)["frames"]
    for i, c in enumerate(centres):
        for j, sx in enumerate((-1.0, +1.0)):          # b then a
            child = mujoco.MjSpec.from_file(str(robot_xml))
            if pads:
                add_hand_pads(child, pads, p.pad_friction)
                if flush:
                    set_flushpad_contact(child, pads, p)
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

    if flush:
        set_flushpad_harness_options(spec, p)
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


# ------------------------------------------------------------------ flush-pad wrist variant (C1, opt-in)
#
# OPT-IN, 2026-10-02: C1 of docs/SOLUTIONS_2026-10-01.md (section 4); launcher
# slurm/repo20260923/cpu_coop_flushpad.sbatch. A MODIFIED END-EFFECTOR, not the stock robot. Nothing in this
# section runs unless the caller passes a `FlushPadParams`: `build_carry` branches only on
# `pad_geometry == "flush"`, a field a stock `CarryParams` does not have, so the stock model, scripts, rules and
# outputs are unchanged (tests/test_coop_flushpad.py: the stock model is byte-identical to commit 094797e's, a short
# stock episode is identical step for step, and this file's diff to 094797e is insertions only).
#
# * Pad geometry (`flush_pad_frames`; kinematics only: no episode, no seed, no tuned number). The stock pad box
#   (hand-mesh AABB half-sizes and centre in the hand-link frame, both unchanged) is re-oriented about its centre in
#   the hand-link frame so that at the SQUEEZE POSE its faces are parallel to the cube's faces. The squeeze pose is
#   the squeeze keyframe (0, 0.9, 0, 0.5, 0) with the shoulder roll stopped where the pad's outer face reaches the
#   cube face plane (side_off - CUBE_HALF = 0.30 m from the base's sagittal plane; base upright, as spawned): the
#   cube stops the squeeze short of its keyframe, so this is the pose in which the pad presses on the cube. The
#   target attitude is the cube-aligned attitude nearest to the stock pad's at the stock pad's own contact roll (the
#   proper signed axis permutation with the smallest rotation), so the pad keeps its stock contact face. Left and
#   right are computed independently by the same rule; both hands of every robot get the flush pad.
# * Pad contact: condim 4 (sliding + torsional friction), torsional friction 0.04 m; sliding (pad 1.0, the cube's
#   1.2 wins) and rolling terms unchanged.
# * HARNESS CHANGE: elliptic friction cone and impratio 10. These are global MuJoCo options, so they also change
#   the foot-floor contact of the frozen gait; the robot-fall clauses are kept unchanged.
# * Placement lowering (lift_place): the reverse keyframe, `PlaceParams.lower_to = None` (lift -> squeeze), FROZEN
#   2026-10-02 before any probe or scored episode of this variant (`FlushPadPlaceParams`).
# * Unchanged: arm keyframes, timeline, grasping-arm kp 30, 4 Nm arm cap, cube, plinth, layout, the gait.
# * Cube tilt (ORACLE, scoring only): `CarryRunner.cube_tilt`, the angle between the cube's body z axis and world
#   z (the measure behind the "held rolled 0.63-0.93 rad" finding), at every policy-step state (`CubeTiltRecorder`).
# * Numerical blow-ups (review fix 2026-10-02, before any probe or scored episode): MuJoCo resets the simulation by
#   itself (qpos0, time 0) when qpos, qvel or qacc holds a NaN or a value beyond mjMAXVAL = 1e10, and keeps stepping,
#   so the stock non-finite check almost never fires and a blow-up would put the cube back flat on its plinth.
#   `CubeTiltRecorder` detects that reset after every policy step (`sim_reset_signal`) and the episode records it
#   (`sim_reset`); a detected reset fails the episode (check `no_sim_reset`). A non-finite stop's NaNs are written as
#   null (`_json_safe`), so the score JSON holds the episode and it is scored as the failure it is.

FLUSHPAD_VARIANT = "flushpad_v1"
FLUSHPAD_TILT_MAX_RAD = 0.35
#: The declared tuning/exploration seeds of this harness (module docstring and lift-place notes): the only seeds a
#: flush-pad smoke may run. 120-124 belong to the probe and 20-39 to the scored stage.
FLUSHPAD_SMOKE_SEEDS = tuple(range(100, 120))

FLUSHPAD_NOTE = "MODIFIED END-EFFECTOR (flush hand pads) — not the stock robot"
FLUSHPAD_HARNESS_NOTE = ("HARNESS CHANGE: elliptic friction cone + impratio 10 are global MuJoCo options and also "
                         "change the foot-floor contact of the frozen gait; the robot-fall clauses are kept unchanged")
FLUSHPAD_LABEL = ("LEARNED gait (frozen arms-dr1.0-s0) + SCRIPTED arms + ORACLE cube pose (scoring only) | "
                  + FLUSHPAD_NOTE)
FLUSHPAD_LABEL_DETAIL = (
    "legs + outer arm: frozen learned PPO gait arms-dr1.0-s0, unmodified observation, zero velocity command all "
    "episode | grasping arm (one per robot): scripted joint keyframes (unchanged), PD kp 30 (deploy 10), 4 Nm cap | "
    "cube pose: simulator oracle, scoring only | contact: MuJoCo, no welds | MODIFIED END-EFFECTOR: hand pad = "
    "hand-mesh AABB box re-oriented flush to the cube faces at the squeeze pose, condim 4, torsional friction "
    "0.04 m | HARNESS CHANGE: elliptic friction cone, impratio 10 (global; also foot-floor contact)")

#: The C1 PREDECLARED RULES, verbatim (FROZEN 2026-10-02; the launcher header repeats them word for word).
FLUSHPAD_PROBE_RULE_TEXT = (
    "Probe on exploration seeds 120-124 (state the grep evidence that they were never used). PROCEED iff the cube "
    "tilt stays <= 0.35 rad throughout lift-hold on >= 4/5 seeds, with LIFT_HOLD_RULE's robot-fall clauses "
    "unchanged. Otherwise STOP: NEGATIVE, the flush pad does not stop the roll.")
FLUSHPAD_SCORED_RULE_TEXT = (
    "Scored, only after PROCEED (state the evidence that seeds 20-39 were never used): lift-hold on seeds 20-29 and "
    "lift-place on seeds 30-39, each with its UNCHANGED rule (LIFT_HOLD_RULE / LIFT_PLACE_RULE, pass_min 8) PLUS "
    "cube tilt <= 0.35 rad (hold: throughout the hold; place: at release and when seated), >= 8/10 per crew or pair "
    "as the existing rules define them. PASS iff every crew/pair meets it; otherwise NEGATIVE; INCOMPLETE if any "
    "episode is missing.")

#: How the frozen rules are applied (decided 2026-10-02, before any episode of this variant). Each reading is
#: stricter than or equal to the rule text; none is chosen from results.
FLUSHPAD_CLAUSES_AS_APPLIED = (
    "cube tilt = the angle between the cube's own z axis and world z (CarryRunner.cube_tilt; a cube rolled 90 deg "
    "reads 1.571 rad), logged at every policy-step state (t = i * 0.04 s, i = 0..500) in full float precision; "
    "<= 0.35 rad passes, anything larger or non-finite fails",
    "probe 'throughout lift-hold' = every policy-step state of the 20 s lift_hold episode, t = 0 to 20 s inclusive "
    "(stricter than the hold alone). A probe seed passes iff that tilt clause, LIFT_HOLD_RULE's robot-fall clause "
    "unchanged (no robot of the pair tilts > 0.78 rad at any policy step) and the validity clauses (finite state, "
    "no MuJoCo auto-reset, full 20 s) all hold; crew 2 (one pair), seeds 120-124, PROCEED iff >= 4 of 5",
    "probe disclosure: the rule text names only the robot-fall clauses of LIFT_HOLD_RULE, so its lift, hold and "
    "floor clauses are not gated in the probe (they are reported per seed as diagnostics). A probe episode that "
    "never lifts the cube therefore passes the probe's tilt clause; the scored stage keeps every LIFT_HOLD_RULE "
    "clause",
    "hold 'throughout the hold' = every policy-step state of the hold LIFT_HOLD_RULE scores (its longest contiguous "
    ">= 0.05 m run, hold_start_s inclusive to hold_end_s exclusive); no hold = the clause fails",
    "place 'at release' = every policy-step state from the first at or after the end of the lowering "
    "(PlaceScript t_end['unload'] = 10.7 s: state 268, t = 10.72 s) through the first at or after the hands are "
    "fully open (t_end['release'] = 11.7 s: state 293, t = 11.72 s), inclusive; 'when seated' = the final state "
    "(t = 20 s), where LIFT_PLACE_RULE checks the seated cube",
    "crews as the existing rules and launchers define them: crew 2 (one pair) and crew 4 (two pairs, two cubes, one "
    "world), >= 8/10 per pair; PASS iff all six pairs (lift-hold crew 2, crew 4 pair 0, crew 4 pair 1; lift-place "
    "the same) meet it",
    "an episode stopped by a non-finite state is a scored failure (as cpu_coop_lift_place.sbatch reads it); its "
    "non-finite numbers are written as null so the score JSON holds it. A numerical blow-up that MuJoCo catches "
    "itself is a scored failure too, in every stage (check no_sim_reset): MuJoCo 3.3.5 resets the state "
    "automatically (qpos0, time 0) when qpos, qvel or qacc holds a NaN or a value beyond 1e10, so the harness's own "
    "non-finite check never sees it; it is detected after every policy step from the simulated time (not "
    "steps x policy step) and MuJoCo's bad-qpos/qvel/qacc warning counters. An episode that is absent, shorter "
    "than 20 s without a recorded failure, or has no tilt log or no auto-reset record is MISSING (INCOMPLETE)",
    "INVALID (never PROCEED, never PASS): a score JSON that is not this variant's or its stage's, not under the "
    "frozen rule, parameters, keyframes or lowering, not a scored run, or whose compiled model does not show the "
    "designed flush pads with condim 4, torsional friction 0.04 m, elliptic cone and impratio 10",
)

#: Seed-use evidence (grep of results, Slurm logs, the ledger and the investigators' files, 2026-10-02), recorded in
#: every flush-pad verdict JSON.
FLUSHPAD_SEED_EVIDENCE = (
    "results/scripted-carry-20260926/score_crew2.json, score_crew4.json, score_lifthold_crew2.json and "
    "score_lifthold_crew4.json hold seeds 0-9 only; results/scripted-carry-20260927/score_liftplace_crew2.json and "
    "score_liftplace_crew4.json hold seeds 10-19 only",
    "Slurm logs of this harness: scripted-carry-21434982 and coop-lift-hold-21435079 ran seeds 0-9, "
    "coop-lift-place-21443282 seeds 10-19; the render jobs 21435080 and 21443283 skipped (no PASS)",
    "exploration: this module's docstring and lift-place notes name seeds 100-119 only (100-102, 100-109, 110-114, "
    "100-104 and the slip times on 100-119); the 2026-09-30 investigators' grip and place mechanism JSONs "
    "(solutions-20260930/coop and verify-coop) hold seeds 100-104 only",
    "the ledger (SLURM_JOBS.md) names seeds 0-9 (carry, lift_hold), 10-19 (lift_place) and exploration seeds "
    ">= 100 for this harness; no score JSON, Slurm log, ledger entry or investigator file of this harness "
    "contains seed 120-124 or any seed 20-39",
)

#: Probe (stage 1): crew 2, lift_hold protocol (20 s), seeds 120-124 (see FLUSHPAD_CLAUSES_AS_APPLIED).
FLUSHPAD_PROBE_RULE = {
    **LIFT_HOLD_RULE,
    "variant": FLUSHPAD_VARIANT, "stage": "probe", "clauses": "probe",
    "cube_tilt_max_rad": FLUSHPAD_TILT_MAX_RAD,
    "cube_tilt_window": "every policy-step state of the 20 s lift-hold episode, t = 0 to 20 s inclusive",
    "gated_checks": ["cube_tilt", "no_fall", "finite", "no_sim_reset", "full_episode"],
    "seeds": [120, 121, 122, 123, 124],
    "crews": [2],
    "pass_min": 4,
}
#: Scored lift-hold (stage 2): LIFT_HOLD_RULE unchanged (every clause, pass_min 8) on seeds 20-29, crews 2 and 4
#: (crew 4: each pair), PLUS cube tilt <= 0.35 rad at every policy-step state of the scored hold.
FLUSHPAD_HOLD_RULE = {
    **LIFT_HOLD_RULE,
    "variant": FLUSHPAD_VARIANT, "stage": "hold", "clauses": "hold",
    "cube_tilt_max_rad": FLUSHPAD_TILT_MAX_RAD,
    "cube_tilt_window": ("every policy-step state of the hold that LIFT_HOLD_RULE scores (the longest contiguous "
                         ">= 0.05 m run, hold_start_s inclusive to hold_end_s exclusive); no hold = clause fails"),
    "seeds": list(range(20, 30)),
    "crews": [2, 4],
}
#: Scored lift-place (stage 2): LIFT_PLACE_RULE unchanged (every clause, pass_min 8) on seeds 30-39, crews 2 and 4,
#: with the frozen reverse-keyframe lowering, PLUS cube tilt <= 0.35 rad at release and when seated.
FLUSHPAD_PLACE_RULE = {
    **LIFT_PLACE_RULE,
    "variant": FLUSHPAD_VARIANT, "stage": "place", "clauses": "place",
    "cube_tilt_max_rad": FLUSHPAD_TILT_MAX_RAD,
    "cube_tilt_window": ("at release: every policy-step state from the first at or after the end of the lowering "
                         "(PlaceScript t_end['unload']) through the first at or after the hands are fully open "
                         "(t_end['release']), inclusive; when seated: the final state, where LIFT_PLACE_RULE checks "
                         "the seated cube"),
    "seeds": list(range(30, 40)),
    "crews": [2, 4],
}
FLUSHPAD_STAGES = ("probe", "hold", "place", "smoke")


@dataclass
class FlushPadParams(CarryParams):
    """OPT-IN C1 flush-pad wrist variant: `CarryParams` (every stock field and default unchanged) plus the
    MODIFIED END-EFFECTOR and the HARNESS CHANGE. Frozen 2026-10-02, before any episode of this variant."""
    pad_geometry: str = "flush"             # hand-mesh AABB box re-oriented flush at the squeeze pose
    pad_condim: int = 4                     # sliding + torsional friction
    pad_torsional_friction: float = 0.04    # m
    solver_cone: str = "elliptic"           # HARNESS CHANGE (global option; also foot-floor contact)
    solver_impratio: float = 10.0           # HARNESS CHANGE (global option)


@dataclass
class FlushPadPlaceParams(PlaceParams):
    """`PlaceParams` with the lowering FROZEN 2026-10-02 for the flush-pad variant: the reverse keyframe
    (lift -> squeeze); every other field keeps the stock default."""
    lower_to: tuple | None = None


def is_flushpad(p) -> bool:
    """True only for the opt-in flush-pad variant (a stock `CarryParams` has no `pad_geometry`)."""
    return getattr(p, "pad_geometry", None) == "flush"


def proper_signed_permutations() -> list:
    """The 24 rotations that map each coordinate axis onto a +/- coordinate axis (the cube's symmetries)."""
    import itertools
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            P = np.zeros((3, 3))
            for col, (row, s) in enumerate(zip(perm, signs)):
                P[row, col] = s
            if np.linalg.det(P) > 0:
                out.append(P)
    return out


def nearest_axis_rotation(R) -> np.ndarray:
    """The axis-aligned attitude closest to rotation matrix `R` (max trace(P^T R) = smallest rotation angle)."""
    R = np.asarray(R, dtype=float)
    return max(proper_signed_permutations(), key=lambda P: float(np.trace(P.T @ R)))


def rotation_angle_deg(Ra, Rb) -> float:
    """Angle (deg) of the rotation taking attitude Ra to attitude Rb."""
    c = (float(np.trace(np.asarray(Ra, dtype=float).T @ np.asarray(Rb, dtype=float))) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _bisect_increasing(f, lo: float, hi: float, n: int = 80) -> float:
    flo, fhi = f(lo), f(hi)
    if not (flo < 0.0 < fhi):
        raise RuntimeError(f"no sign change on [{lo}, {hi}]: f = {flo:.4f}, {fhi:.4f}")
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class PadFK:
    """Arm forward kinematics of one robot with the given hand-pad frames: free base at the origin, upright, every
    joint at 0 except the arm being posed. Robot frame: x forward, y left, origin at the base body. The arms hang
    off the base body, so the pad's pose relative to the base does not depend on the legs: the flush design needs
    neither deploy.yaml nor the stance height."""

    def __init__(self, robot_xml: Path, frames: dict):
        spec = mujoco.MjSpec.from_file(str(robot_xml))
        add_hand_pads(spec, frames, 1.0)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.d.qpos[:] = 0.0
        for j in range(self.m.njnt):
            if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.d.qpos[self.m.jnt_qposadr[j] + 3] = 1.0
        self.geom = {s: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f"arm_{s}_hand_pad")
                     for s in ("left", "right")}
        self.body = {s: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"arm_{s}_hand_link")
                     for s in ("left", "right")}

    def pose(self, side: str, q):
        """(pad centre, pad rotation, hand-link body rotation, pad half-sizes) with that arm at joints `q`."""
        names = ARM_JOINTS_L if side == "left" else ARM_JOINTS_R
        adr = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
        self.d.qpos[adr] = np.asarray(q, dtype=float)
        mujoco.mj_kinematics(self.m, self.d)
        g, b = self.geom[side], self.body[side]
        return (self.d.geom_xpos[g].copy(), self.d.geom_xmat[g].reshape(3, 3).copy(),
                self.d.xmat[b].reshape(3, 3).copy(), self.m.geom_size[g].copy())

    def lateral_extent(self, side: str, q) -> float:
        """Outermost reach of the pad box toward its own side (m)."""
        c, R, _, h = self.pose(side, q)
        sgn = 1.0 if side == "left" else -1.0
        return float(max(sgn * (c + R @ (h * np.array([sx, sy, sz])))[1]
                         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)))


def flushpad_squeeze_pose(side: str, roll: float, keyframes: dict = KEYFRAMES_LEFT) -> np.ndarray:
    """That arm's joints at the squeeze keyframe with the shoulder roll set to `roll` (left order; right = -q)."""
    q = np.asarray(keyframes["squeeze"], dtype=float).copy()
    q[1] = roll
    return q if side == "left" else -q


def flush_pad_frames(robot_xml: Path, p: CarryParams, keyframes: dict = KEYFRAMES_LEFT) -> dict:
    """The flush-pad frames (hand-link frame; same keys as `hand_pad_frames`) and their design record.

    Per side: (1) the stock pad's contact roll r0 (its outermost corner reaches the cube face plane, squeeze
    keyframe otherwise); (2) P = the cube-aligned attitude nearest to the stock pad's at r0; the pad axis P maps
    onto the pinch axis (+/- y) is the contact axis, half-size h; (3) the flush contact roll r* solves
    pad-centre lateral offset + h = side_off - CUBE_HALF; (4) the pad's hand-link rotation becomes
    R_hand(r*)^T P, so at the squeeze pose (squeeze keyframe, roll r*) the pad's attitude is exactly P."""
    if not p.hand_pads:
        raise ValueError("the flush-pad variant needs hand_pads=True")
    stock = hand_pad_frames(robot_xml)
    fk = PadFK(robot_xml, stock)
    face = float(p.side_off - CUBE_HALF)
    frames, design = {}, {}
    for side in ("left", "right"):
        sgn = 1.0 if side == "left" else -1.0
        half = np.asarray(stock[side]["size"], dtype=float)
        r0 = _bisect_increasing(lambda r: fk.lateral_extent(side, flushpad_squeeze_pose(side, r, keyframes)) - face,
                                -0.6, 1.2)
        _, R0, _, _ = fk.pose(side, flushpad_squeeze_pose(side, r0, keyframes))
        P = nearest_axis_rotation(R0)
        axis = int(np.argmax(np.abs(P[1, :])))           # pad axis mapped onto the pinch axis (robot y)
        rs = _bisect_increasing(
            lambda r: sgn * fk.pose(side, flushpad_squeeze_pose(side, r, keyframes))[0][1] + half[axis] - face,
            -0.6, 1.2)
        c1, R1, Rb, _ = fk.pose(side, flushpad_squeeze_pose(side, rs, keyframes))
        if not np.array_equal(nearest_axis_rotation(R1), P):
            raise RuntimeError(f"{side}: the stock pad's nearest cube-aligned attitude differs between r0 and r*")
        R_local = Rb.T @ P
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R_local.flatten())
        frames[side] = {"pos": list(stock[side]["pos"]), "quat": quat.tolist(), "size": list(stock[side]["size"])}
        _, Rk, _, _ = fk.pose(side, sgn * np.asarray(keyframes["squeeze"], dtype=float))
        design[side] = {                          # joint values in full precision: the pose the pad is flush at
            "squeeze_pose_joints": flushpad_squeeze_pose(side, rs, keyframes).tolist(),
            "flush_contact_roll_rad": float(rs),
            "stock_contact_roll_rad": float(r0),
            "pad_axes_at_squeeze_robot_frame": P.astype(int).tolist(),
            "contact_axis": axis, "contact_half_m": round(float(half[axis]), 6),
            "pad_centre_at_squeeze_base_frame_m": c1.round(4).tolist(),
            "stock_pad_off_cube_attitude_deg_at_stock_contact": round(rotation_angle_deg(P, R0), 2),
            "stock_pad_off_cube_attitude_deg_at_flush_contact": round(rotation_angle_deg(P, R1), 2),
            "stock_pad_contact_normal_off_pinch_axis_deg_at_squeeze_keyframe":
                round(float(np.degrees(np.arccos(np.clip(np.max(np.abs(Rk[1, :])), -1.0, 1.0)))), 2),
            "pad_rotation_in_hand_frame_deg": round(rotation_angle_deg(R1, P), 2),
        }
    return {"frames": frames, "design": design, "face_lateral_m": round(face, 6),
            "squeeze_pose": ("squeeze keyframe with the shoulder roll stopped where the pad's outer face reaches the "
                             "cube face plane (base upright at the origin, robot frame x forward, y left)"),
            "method": "stock pad box (hand-mesh AABB) re-oriented about its centre to the nearest cube-aligned attitude"}


def flushpad_design(upstream: Path, cache_dir: Path, p: CarryParams) -> dict:
    """`flush_pad_frames` on the same MJCF copy `build_carry` uses (recorded in every flush-pad output)."""
    scene = prepare_mjcf(upstream, cache_dir, "humanoid")
    return flush_pad_frames(scene.parent / "berkeley_humanoid_lite.xml", p)


def set_flushpad_contact(spec, frames: dict, p, prefix: str = "") -> None:
    """Flush-pad contact on the pads `add_hand_pads` just added: condim 4 and torsional friction."""
    for side in frames:
        body = spec.body(f"{prefix}arm_{side}_hand_link")
        pads = [g for g in body.geoms if g.name == f"{prefix}arm_{side}_hand_pad"]
        if len(pads) != 1:
            raise RuntimeError(f"expected one {prefix}arm_{side}_hand_pad, got {len(pads)}")
        pads[0].condim = int(p.pad_condim)
        pads[0].friction = [p.pad_friction, float(p.pad_torsional_friction), 0.0001]


def set_flushpad_harness_options(spec, p) -> None:
    """HARNESS CHANGE of the flush-pad variant: global friction cone and impratio (also foot-floor contact)."""
    cones = {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL, "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}
    spec.option.cone = cones[p.solver_cone]
    spec.option.impratio = float(p.solver_impratio)


def flushpad_model_check(model) -> dict:
    """What the compiled model contains, read back from it: every hand pad and the solver options."""
    pads = []
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name.endswith("_hand_pad"):
            pads.append({"name": name, "condim": int(model.geom_condim[g]),
                         "friction": [round(float(v), 6) for v in model.geom_friction[g]],
                         "quat": [round(float(v), 6) for v in model.geom_quat[g]],
                         "size": [round(float(v), 6) for v in model.geom_size[g]]})
    return {"pads": pads,
            "cone": "elliptic" if int(model.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC) else "pyramidal",
            "impratio": float(model.opt.impratio), "noslip_iterations": int(model.opt.noslip_iterations)}


def flushpad_model_ok(check: dict, n_robots: int | None = None) -> bool:
    """True iff the model check shows the frozen flush-pad contact and harness options (two pads per robot,
    condim 4, sliding 1.0, torsional friction 0.04 m, elliptic cone, impratio 10)."""
    pads = (check or {}).get("pads") or []
    return (bool(pads) and (n_robots is None or len(pads) == 2 * int(n_robots))
            and all(pd["condim"] == 4 and abs(pd["friction"][0] - 1.0) < 1e-9
                    and abs(pd["friction"][1] - 0.04) < 1e-9 for pd in pads)
            and check.get("cone") == "elliptic" and abs(float(check.get("impratio", 0.0)) - 10.0) < 1e-9)


def flushpad_design_matches_model(design: dict, check: dict) -> bool:
    """True iff every compiled hand pad has its side's designed flush frame (quat and size, to 1e-6)."""
    frames = (design or {}).get("frames") or {}
    pads = (check or {}).get("pads") or []
    if not pads or set(frames) != {"left", "right"}:
        return False
    for pd in pads:
        side = "left" if pd["name"].endswith("arm_left_hand_pad") else "right"
        f = frames[side]
        if not (np.allclose(pd["quat"], f["quat"], atol=2e-6) and np.allclose(pd["size"], f["size"], atol=2e-6)):
            return False
    return True


#: MuJoCo's bad-number warnings: with mjDSBL_AUTORESET clear (the default; this harness never sets it) `mj_step`
#: calls `mj_resetData` (qpos0, qvel 0, time 0) when qpos, qvel or qacc holds a NaN or |x| > mjMAXVAL = 1e10, then
#: keeps stepping. The warning counter survives that reset (MuJoCo 3.3.5, checked 2026-10-02: number 1 after it).
FLUSHPAD_BAD_NUMBER_WARNINGS = ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC")


def sim_reset_signal(m, d, steps_done: int, substeps: int) -> dict | None:
    """Evidence that MuJoCo auto-reset the simulation since the episode's own reset, read after `steps_done`
    policy steps of `substeps` physics steps each; None when there is none. Two signals: the simulated time is not
    steps_done * substeps * timestep (the reset restarts it at 0; tolerance half a physics step), or a bad-qpos/qvel/
    qacc warning was counted (the episode's own `mj_resetData` zeroes the counters)."""
    ts = float(m.opt.timestep)
    expected = int(steps_done) * int(substeps) * ts
    warnings = {n: int(d.warning[int(getattr(mujoco.mjtWarning, n))].number) for n in FLUSHPAD_BAD_NUMBER_WARNINGS}
    time_mismatch = abs(float(d.time) - expected) > 0.5 * ts
    if not time_mismatch and not any(warnings.values()):
        return None
    return {"sim_time_s": float(d.time), "expected_sim_time_s": expected, "time_mismatch": bool(time_mismatch),
            "bad_number_warnings": warnings}


class CubeTiltRecorder:
    """frame_hook: each pair's cube tilt (`CarryRunner.cube_tilt`) after every policy step. `series(k)` is pair
    k's tilt at every policy-step state: the pre-step state of each step i (time i * dt, aligned with the lift and
    robot-tilt series the rules score) followed by the final state. `chain` is called after recording.
    (The harness trace's `cube_tilt_rad` at trace step i is the post-step value, i.e. `series(k)[i + 1]`.)
    It also checks after every policy step whether MuJoCo auto-reset the simulation (`sim_reset_signal`) and keeps
    the first evidence (`sim_reset_record`)."""

    def __init__(self, model, pairs, chain=None):
        self.post, self.chain = [], chain
        self.reset_tilt = []
        self.sim_reset = None                     # first auto-reset evidence, with the policy step it happened in
        for pr in pairs:                          # the reset state: the cube at qpos0 (runner.reset keeps it)
            a = model.jnt_qposadr[model.body_jntadr[pr.cube_body]]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, model.qpos0[a + 3:a + 7])
            self.reset_tilt.append(float(np.arccos(np.clip(R[8], -1.0, 1.0))))

    def __call__(self, **kw):
        runner, pairs = kw["runner"], kw["pairs"]
        self.post.append([runner.cube_tilt(pr) for pr in pairs])
        if self.sim_reset is None:
            sig = sim_reset_signal(runner.m, runner.d, int(kw["step"]) + 1, runner.substeps)
            if sig is not None:
                self.sim_reset = {"first_step": int(kw["step"]), **sig}
        if self.chain is not None:
            self.chain(**kw)

    def series(self, k: int) -> list:
        return [self.reset_tilt[k]] + [row[k] for row in self.post]

    def sim_reset_record(self) -> dict:
        r = self.sim_reset
        return {"detected": r is not None, "first_step": None if r is None else r["first_step"], "evidence": r,
                "policy_steps": len(self.post),
                "note": ("MuJoCo auto-reset (qpos0, time 0) on a NaN or |x| > 1e10 in qpos/qvel/qacc, checked after "
                         "every policy step (simulated time and bad-number warning counters); first_step = the "
                         "policy step it happened in; detected = the episode fails (no_sim_reset)")}


def _json_safe(x) -> tuple:
    """(`x` as plain JSON types with every non-finite float replaced by None, how many were replaced). A non-finite
    stop leaves NaN in the stock rows and trace (lift_place's final state; the post-step trace entry), which the
    score JSON (json.dumps(allow_nan=False)) cannot hold: nulled, the episode is written and scored as a failure."""
    nulled = 0

    def walk(v):
        nonlocal nulled
        if isinstance(v, dict):
            return {kk: walk(u) for kk, u in v.items()}
        if isinstance(v, (list, tuple)):
            return [walk(u) for u in v]
        if isinstance(v, np.ndarray):
            return walk(v.tolist())
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            if np.isfinite(v):
                return float(v)
            nulled += 1
            return None
        return v

    return walk(x), nulled


def _json_tilt(v) -> float | None:
    """Full float precision (never rounded: a rounded value could pass the 0.35 rad clause it should fail)."""
    v = float(v)
    return v if np.isfinite(v) else None


def run_flushpad_episode(model, slots, pairs, cfg, policy, seed: int, p, rule: dict,
                         q: PlaceParams | None = None, frame_hook=None) -> dict:
    """One flush-pad episode: the stock lift_hold (`run_episode`) or lift_place (`run_place_episode`) protocol on
    a flush-pad model, unchanged, plus each pair's per-step cube tilt series (`cube_tilt_series_rad`), the MuJoCo
    auto-reset record (`sim_reset`), and non-finite numbers written as None (`nonfinite_values_nulled` counts them)."""
    if not is_flushpad(p):
        raise ValueError("run_flushpad_episode needs FlushPadParams")
    if rule.get("variant") != FLUSHPAD_VARIANT:
        raise ValueError("rule is not a flush-pad rule")
    rec = CubeTiltRecorder(model, pairs, chain=frame_hook)
    if rule.get("protocol") == "lift_place":
        q = FlushPadPlaceParams() if q is None else q
        if q.lower_to is not None:
            raise ValueError("the flush-pad lowering is FROZEN: reverse keyframe (lower_to=None)")
        ep = run_place_episode(model, slots, pairs, cfg, policy, seed, p, q=q, frame_hook=rec, rule=rule)
    else:
        ep = run_episode(model, slots, pairs, cfg, policy, seed, p, frame_hook=rec, rule=rule, protocol="lift_hold")
    for k, row in enumerate(ep["pairs"]):
        row["cube_tilt_series_rad"] = [_json_tilt(v) for v in rec.series(k)]
    ep["variant"] = FLUSHPAD_VARIANT
    ep["cube_tilt_series_note"] = ("cube_tilt_series_rad[i] = cube tilt (rad, angle of the cube's z axis from world z) "
                                   "at the pre-step state of policy step i (t = i * dt); the last entry is the "
                                   "final state; null = non-finite")
    ep["sim_reset"] = rec.sim_reset_record()
    ep, nulled = _json_safe(ep)
    ep["nonfinite_values_nulled"] = nulled
    return ep


def _tilt_ok(values, limit: float) -> bool:
    vals = list(values)
    return bool(vals) and all(v is not None and np.isfinite(v) and float(v) <= limit for v in vals)


def flushpad_release_window(script_times_s: dict, dt: float) -> tuple[int, int]:
    """(first, last) policy-step state indices of the place 'at release' window, both inclusive."""
    import math
    k0 = math.ceil(float(script_times_s["unload"]) / dt - 1e-9)
    k1 = math.ceil(float(script_times_s["release"]) / dt - 1e-9)
    return k0, k1


def _sim_reset_logged(ep: dict) -> tuple[bool, bool]:
    """(the episode carries a well-formed MuJoCo auto-reset record, that record reads detected)."""
    rec = ep.get("sim_reset")
    logged = isinstance(rec, dict) and isinstance(rec.get("detected"), bool)
    return bool(logged), bool(logged and rec["detected"])


def flushpad_clauses(rule: dict, ep: dict, k: int, dt: float) -> dict:
    """The flush-pad clauses of pair `k` in one stored episode, from the JSON alone (pure). Every clause set
    includes `no_sim_reset`: no MuJoCo auto-reset in the episode (an absent record is not a pass)."""
    row = ep["pairs"][k]
    series = row.get("cube_tilt_series_rad")
    logged = isinstance(series, list) and len(series) == int(ep.get("steps", -1)) + 1
    reset_logged, reset = _sim_reset_logged(ep)
    no_reset = bool(reset_logged and not reset)
    lim = float(rule["cube_tilt_max_rad"])
    kind = rule["clauses"]
    out = {"pair": k, "tilt_logged": bool(logged), "sim_reset_logged": reset_logged, "sim_reset_detected": reset}
    if kind == "probe":
        c = row["checks"]
        checks = {"cube_tilt": bool(logged and _tilt_ok(series, lim)), "no_fall": bool(c["no_fall"]),
                  "finite": bool(c["finite"]), "no_sim_reset": no_reset, "full_episode": bool(c["full_episode"])}
        vals = [v for v in series if v is not None] if logged else []
        out["cube_tilt_max_rad"] = max(vals) if vals else None
    elif kind == "hold":
        hs, he = row.get("hold_start_s"), row.get("hold_end_s")
        win = []
        if logged and hs is not None and he is not None:
            win = series[int(round(hs / dt)):int(round(he / dt))]
        checks = {"rule": bool(row["success"]), "cube_tilt_hold": bool(win) and _tilt_ok(win, lim),
                  "no_sim_reset": no_reset}
        vals = [v for v in win if v is not None]
        out["cube_tilt_hold_max_rad"] = max(vals) if vals else None
    elif kind == "place":
        ts = ep.get("script_times_s") or {}
        win, seated = [], None
        if logged and "unload" in ts and "release" in ts:
            k0, k1 = flushpad_release_window(ts, dt)
            if len(series) > k1 + 1:              # the release window ends before the final state
                win = series[k0:k1 + 1]
                out["release_window_states"] = [k0, k1]
            seated = series[-1]
        checks = {"rule": bool(row["success"]), "cube_tilt_release": bool(win) and _tilt_ok(win, lim),
                  "cube_tilt_seated": seated is not None and _tilt_ok([seated], lim), "no_sim_reset": no_reset}
        vals = [v for v in win if v is not None]
        out["cube_tilt_release_max_rad"] = max(vals) if vals else None
        out["cube_tilt_seated_rad"] = seated
    else:
        raise ValueError(f"unknown flush-pad clause set {kind!r}")
    success = all(checks.values())
    out.update({"success": bool(success), "checks": checks,
                "first_failed_check": None if success else next(n for n, v in checks.items() if not v)})
    return out


def flushpad_stage_rule(stage: str, protocol: str = "lift_hold", seeds=None) -> dict:
    """The frozen rule of a stage; "smoke" = the probe (lift_hold) or place (lift_place) clauses on tuning seeds."""
    if stage == "probe":
        return FLUSHPAD_PROBE_RULE
    if stage == "hold":
        return FLUSHPAD_HOLD_RULE
    if stage == "place":
        return FLUSHPAD_PLACE_RULE
    if stage == "smoke":
        base = FLUSHPAD_PLACE_RULE if protocol == "lift_place" else FLUSHPAD_PROBE_RULE
        return {**base, "stage": "smoke", "seeds": sorted(int(s) for s in (seeds or [])), "crews": [2, 4],
                "pass_min": None, "smoke": "pipeline check on declared tuning seeds 100-119; never scored"}
    raise ValueError(f"unknown flush-pad stage {stage!r}")


def _jsonable(x):
    import json
    return json.loads(json.dumps(x))


def _flushpad_payload_problem(payload: dict, stage: str, rule: dict, crew: int) -> str | None:
    """Why a score JSON cannot be read under `rule` (None when it can)."""
    if not isinstance(payload, dict):
        return "no score JSON"
    if payload.get("variant") != FLUSHPAD_VARIANT:
        return "not a flush-pad score JSON"
    if payload.get("stage") != stage:
        return f"stage {payload.get('stage')!r}, expected {stage!r}"
    if payload.get("rule") != _jsonable(rule):
        return "rule differs from the frozen flush-pad rule"
    if payload.get("params") != _jsonable(params_dict(FlushPadParams())):
        return "params differ from the frozen FlushPadParams"
    if payload.get("keyframes_left") != _jsonable(KEYFRAMES_LEFT):
        return "arm keyframes differ"
    if rule.get("protocol") == "lift_place":
        frozen = _jsonable(place_params_dict(FlushPadPlaceParams()))
        if payload.get("place_params") != frozen or any(e.get("place_params") != frozen
                                                        for e in payload.get("episodes", [])):
            return "place params differ from the frozen FlushPadPlaceParams (reverse-keyframe lowering)"
    if int(payload.get("crew", -1)) != int(crew):
        return f"crew {payload.get('crew')}, expected {crew}"
    if not payload.get("scored_run"):
        return "not a scored run"
    if not flushpad_model_ok(payload.get("model_check") or {}, int(crew)):
        return "model check does not show the flush-pad contact and harness options"
    if not flushpad_design_matches_model(payload.get("flush_pad_design") or {}, payload.get("model_check") or {}):
        return "compiled hand pads differ from the flush-pad design"
    if "policy_dt" not in payload:
        return "no policy_dt"
    return None


def _flushpad_rows(payload: dict, rule: dict, k: int) -> tuple[list, list]:
    """(per-seed clause rows, seeds that are missing, short without a recorded failure, or without a tilt log or an
    auto-reset record)."""
    dt = float(payload["policy_dt"])
    n_full = int(round(float(rule["episode_s"]) / dt))
    eps = {}
    for e in payload.get("episodes", []):
        eps.setdefault(int(e["seed"]), e)
    rows, missing = [], []
    for s in rule["seeds"]:
        e = eps.get(int(s))
        if e is None or len(e.get("pairs", [])) <= k:
            missing.append(s)
            continue
        if e.get("failed") is None and int(e.get("steps", -1)) != n_full:
            missing.append(s)
            continue
        r = flushpad_clauses(rule, e, k, dt)
        if not (r["tilt_logged"] and r["sim_reset_logged"]):
            missing.append(s)
            continue
        base = e["pairs"][k]
        r.update({"seed": int(s), "base_rule_success": bool(base["success"]),
                  "base_rule_first_failed_check": base["first_failed_check"],
                  "lift_peak_m": base["lift_peak_m"], "lift_hold_s": base["lift_hold_s"],
                  "max_robot_tilt_rad": base["max_tilt_rad"], "cube_floor_contact": base["cube_floor_contact"]})
        rows.append(r)
    return rows, missing


def _flushpad_head(stage: str) -> dict:
    return {"variant": FLUSHPAD_VARIANT, "stage": stage, "label": FLUSHPAD_LABEL,
            "label_detail": FLUSHPAD_LABEL_DETAIL, "end_effector": FLUSHPAD_NOTE,
            "harness_change": FLUSHPAD_HARNESS_NOTE, "clauses_as_applied": list(FLUSHPAD_CLAUSES_AS_APPLIED),
            "seed_evidence": list(FLUSHPAD_SEED_EVIDENCE)}


def flushpad_probe_verdict(payload: dict | None) -> dict:
    """Stage-1 verdict from the probe score JSON (pure): PROCEED | NEGATIVE | INCOMPLETE | INVALID."""
    rule = FLUSHPAD_PROBE_RULE
    out = {**_flushpad_head("probe"), "rule_text": FLUSHPAD_PROBE_RULE_TEXT, "rule": rule}
    if payload is None:
        return {**out, "verdict": "INCOMPLETE", "reason": "no probe score JSON", "successes": 0,
                "of": len(rule["seeds"]), "missing_seeds": list(rule["seeds"]), "per_seed": [],
                "reading": "no probe score JSON: no decision"}
    bad = _flushpad_payload_problem(payload, "probe", rule, 2)
    if bad:
        return {**out, "verdict": "INVALID", "reason": bad, "successes": 0, "of": len(rule["seeds"]),
                "missing_seeds": [], "per_seed": [], "reading": f"INVALID ({bad}): no decision"}
    rows, missing = _flushpad_rows(payload, rule, 0)
    n_ok = sum(1 for r in rows if r["success"])
    verdict = "INCOMPLETE" if missing else ("PROCEED" if n_ok >= rule["pass_min"] else "NEGATIVE")
    reading = {"PROCEED": "the flush pad kept the cube within 0.35 rad on >= 4/5 probe seeds: run the scored stage",
               "NEGATIVE": "STOP: NEGATIVE, the flush pad does not stop the roll",
               "INCOMPLETE": ("an episode is missing, short, or has no tilt log or auto-reset record: no "
                              "decision")}[verdict]
    return {**out, "verdict": verdict, "successes": n_ok, "of": len(rule["seeds"]), "missing_seeds": missing,
            "per_seed": rows, "reading": reading,
            "diagnostic_not_gated": {
                "lift_hold_rule_successes": sum(1 for r in rows if r["base_rule_success"]),
                "lifted_ge_0.10_m": sum(1 for r in rows if r["lift_peak_m"] >= LIFT_HOLD_RULE["lift_peak_m"]),
                "note": "LIFT_HOLD_RULE's lift/hold/floor clauses are not part of the probe rule"}}


def flushpad_group_summary(payload: dict, stage: str) -> dict:
    """Per-pair counts of one scored score JSON (hold or place, one crew), from the JSON alone."""
    rule = FLUSHPAD_HOLD_RULE if stage == "hold" else FLUSHPAD_PLACE_RULE
    crew = int(payload.get("crew", -1)) if isinstance(payload, dict) else -1
    bad = _flushpad_payload_problem(payload, stage, rule, crew) if crew in rule["crews"] else f"crew {crew}"
    if bad:
        return {"stage": stage, "crew": crew, "invalid": bad, "complete": False, "groups": []}
    groups, complete = [], True
    for k in range(crew // 2):
        rows, missing = _flushpad_rows(payload, rule, k)
        n_ok = sum(1 for r in rows if r["success"])
        complete = complete and not missing
        groups.append({"stage": stage, "crew": crew, "pair": k, "episodes": len(rows), "successes": n_ok,
                       "missing_seeds": missing, "pass": bool(not missing and n_ok >= rule["pass_min"]),
                       "base_rule_successes": sum(1 for r in rows if r["base_rule_success"]),
                       "first_failed_check_counts": {
                           c: sum(1 for r in rows if r["first_failed_check"] == c)
                           for c in sorted({r["first_failed_check"] for r in rows} - {None})},
                       "per_seed": rows})
    return {"stage": stage, "crew": crew, "invalid": None, "complete": complete, "groups": groups}


FLUSHPAD_SCORED_KEYS = ("hold_crew2", "hold_crew4", "place_crew2", "place_crew4")


def flushpad_scored_verdict(payloads: dict) -> dict:
    """Stage-2 verdict (pure): `payloads` maps "hold_crew2", "hold_crew4", "place_crew2", "place_crew4" to a
    score JSON or None. PASS iff every crew/pair of both protocols has >= 8/10; INCOMPLETE if any episode is
    missing; INVALID if a JSON is not under the frozen rule."""
    out = {**_flushpad_head("scored"), "rule_text": FLUSHPAD_SCORED_RULE_TEXT,
           "rules": {"hold": FLUSHPAD_HOLD_RULE, "place": FLUSHPAD_PLACE_RULE}}
    groups, invalid, complete = [], [], True
    for stage, rule in (("hold", FLUSHPAD_HOLD_RULE), ("place", FLUSHPAD_PLACE_RULE)):
        for crew in rule["crews"]:
            key = f"{stage}_crew{crew}"
            pl = payloads.get(key)
            if pl is None:
                complete = False
                groups.append({"stage": stage, "crew": crew, "pair": None, "missing_json": True, "pass": False})
                continue
            s = flushpad_group_summary(pl, stage)
            if s["invalid"] or int(s["crew"]) != crew:
                invalid.append(f"{key}: {s['invalid'] or 'crew mismatch'}")
                continue
            complete = complete and s["complete"]
            groups += s["groups"]
    if invalid:
        verdict = "INVALID"
    elif not complete:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS" if len(groups) == 6 and all(g["pass"] for g in groups) else "NEGATIVE"
    reading = {"PASS": "every crew/pair met its unchanged rule plus the 0.35 rad tilt clause on >= 8/10 seeds",
               "NEGATIVE": "at least one crew/pair fell short of 8/10 under its unchanged rule plus the tilt clause",
               "INCOMPLETE": "a score JSON or an episode is missing: no decision",
               "INVALID": "a score JSON is not under the frozen rule: no decision"}[verdict]
    return {**out, "verdict": verdict, "invalid": invalid, "reading": reading,
            "groups": [{k: v for k, v in g.items() if k != "per_seed"} for g in groups],
            "per_seed": {f"{g['stage']}_crew{g['crew']}_pair{g['pair']}": g.get("per_seed", []) for g in groups}}


def flushpad_not_run_verdict(probe_verdict: dict | None) -> dict:
    """Stage-2 verdict when the probe did not PROCEED: NOT_RUN, and seeds 20-39 stay unused."""
    pv = (probe_verdict or {}).get("verdict", "MISSING")
    return {**_flushpad_head("scored"), "rule_text": FLUSHPAD_SCORED_RULE_TEXT, "verdict": "NOT_RUN",
            "probe_verdict": pv, "groups": [],
            "reading": (f"stage 2 not run: the probe verdict is {pv}, not PROCEED (the frozen rule runs the scored "
                        "stage only after PROCEED); seeds 20-39 stay unused")}


def flushpad_smoke_check(payload: dict | None) -> dict:
    """Pipeline checks of one flush-pad SMOKE score JSON (pure; never a verdict on the variant): the variant built
    as designed, each episode ran the full 20 s on a declared tuning seed with no MuJoCo auto-reset (its record
    present and clear), and the cube tilt is logged at every policy-step state, finite, flat at reset, and agrees
    with the harness's own trace. Outcomes (tilt, lift, falls) are reported as diagnostics, never gated."""
    problems, rows = [], []
    if not isinstance(payload, dict):
        return {**_flushpad_head("smoke"), "verdict": "SMOKE_FAIL", "problems": ["no smoke score JSON"],
                "episodes": []}
    crew = int(payload.get("crew", -1))
    protocol = payload.get("protocol")
    if payload.get("variant") != FLUSHPAD_VARIANT or payload.get("stage") != "smoke":
        problems.append("not a flush-pad smoke score JSON")
    if payload.get("scored_run"):
        problems.append("a smoke JSON must not be marked scored_run")
    if crew not in (2, 4) or protocol not in ("lift_hold", "lift_place"):
        problems.append(f"crew {crew} / protocol {protocol!r}")
    if payload.get("params") != _jsonable(params_dict(FlushPadParams())):
        problems.append("params differ from the frozen FlushPadParams")
    if protocol == "lift_place" and payload.get("place_params") != _jsonable(place_params_dict(FlushPadPlaceParams())):
        problems.append("place params differ from the frozen FlushPadPlaceParams")
    mc = payload.get("model_check") or {}
    if not flushpad_model_ok(mc, crew):
        problems.append("model check does not show the flush-pad contact and harness options")
    if not flushpad_design_matches_model(payload.get("flush_pad_design") or {}, mc):
        problems.append("compiled hand pads differ from the flush-pad design")
    eps = payload.get("episodes") or []
    seeds = [int(e["seed"]) for e in eps]
    if not eps:
        problems.append("no episode")
    if any(s not in FLUSHPAD_SMOKE_SEEDS for s in seeds):
        problems.append("a seed outside the declared tuning seeds 100-119")
    rule = payload.get("rule") or {}
    if protocol in ("lift_hold", "lift_place") and rule != _jsonable(flushpad_stage_rule("smoke", protocol, seeds)):
        problems.append("rule differs from the smoke rule (full-length smoke episodes only)")
    dt = float(payload.get("policy_dt") or 0.0)
    n_full = int(round(20.0 / dt)) if dt > 0 else -1
    for e in eps:
        seed, steps = int(e["seed"]), int(e.get("steps", -1))
        if e.get("failed") is not None:
            problems.append(f"seed {seed}: episode failed ({e['failed']})")
        if steps != n_full:
            problems.append(f"seed {seed}: {steps} steps, expected {n_full}")
        reset_logged, reset = _sim_reset_logged(e)
        if not reset_logged:
            problems.append(f"seed {seed}: no MuJoCo auto-reset record (sim_reset)")
        elif reset:                               # a blown-up episode proves nothing about valid probe data
            problems.append(f"seed {seed}: MuJoCo auto-reset the simulation (numerical blow-up) in policy step "
                            f"{(e.get('sim_reset') or {}).get('first_step')}")
        if len(e.get("pairs", [])) != max(crew // 2, 0):
            problems.append(f"seed {seed}: {len(e.get('pairs', []))} pairs for crew {crew}")
        for k, row in enumerate(e.get("pairs", [])):
            ser = row.get("cube_tilt_series_rad")
            if not isinstance(ser, list) or len(ser) != steps + 1:
                problems.append(f"seed {seed} pair {k}: tilt series missing or not steps + 1 long")
                continue
            if any(v is None or not np.isfinite(v) for v in ser):
                problems.append(f"seed {seed} pair {k}: non-finite tilt in the series")
                continue
            if abs(float(ser[0])) > 1e-9:
                problems.append(f"seed {seed} pair {k}: tilt at reset {ser[0]} (the cube spawns flat)")
            trace = e.get("trace") or []
            matched = 0
            for tr in trace:
                i = int(round(float(tr["t"]) / dt)) if dt > 0 else -1
                if not (0 <= i < steps):
                    continue
                tv = tr["cube_tilt_rad"][k]
                if tv is None or abs(float(ser[i + 1]) - float(tv)) > 0.0005 + 1e-9:
                    problems.append(f"seed {seed} pair {k}: tilt series disagrees with the trace at t={tr['t']}")
                    break
                matched += 1
            if matched == 0:
                problems.append(f"seed {seed} pair {k}: no trace entry to cross-check the tilt series")
            try:
                c = flushpad_clauses(rule, e, k, dt)
                # lift_hold smokes also run the scored hold stage's clause code on this real episode
                ch = flushpad_clauses(FLUSHPAD_HOLD_RULE, e, k, dt) if protocol == "lift_hold" else None
            except Exception as exc:  # noqa: BLE001
                problems.append(f"seed {seed} pair {k}: clauses do not compute ({exc!r})")
                continue
            if not c["tilt_logged"] or (ch is not None and not ch["tilt_logged"]):
                problems.append(f"seed {seed} pair {k}: clauses report no tilt log")
            rows.append({"seed": seed, "pair": k, "trace_points_checked": matched,
                         "sim_reset_detected": reset,
                         "diagnostic_cube_tilt_max_rad": max(ser), "diagnostic_flushpad_clauses": c,
                         "diagnostic_hold_stage_clauses": ch,
                         "diagnostic_base_rule_success": bool(row.get("success")),
                         "diagnostic_base_first_failed_check": row.get("first_failed_check"),
                         "diagnostic_lift_peak_m": row.get("lift_peak_m"),
                         "diagnostic_max_robot_tilt_rad": row.get("max_tilt_rad")})
    return {**_flushpad_head("smoke"), "verdict": "SMOKE_FAIL" if problems else "SMOKE_PASS",
            "crew": crew, "protocol": protocol, "seeds": seeds, "problems": problems, "episodes": rows,
            "note": "pipeline check on a declared tuning seed; outcomes are diagnostics, never a verdict"}


#: The smoke launcher's steps, whose exit statuses it records in step_status.json (unit tests first).
FLUSHPAD_SMOKE_STEPS = ("pytest", "smoke_hold_crew2", "smoke_place_crew4")


def flushpad_smoke_step_problems(status) -> list:
    """Problems with the smoke's recorded step exit statuses (pure): every declared step, the unit tests first,
    recorded as integer 0, and nothing recorded non-zero. The smoke verdict reads SMOKE_PASS only without any, so a
    smoke id whose unit tests failed can never let a run through."""
    if not isinstance(status, dict):
        return ["no step-status record (step_status.json): the unit-test result is unknown"]
    problems = []
    for name in FLUSHPAD_SMOKE_STEPS:
        if name not in status:
            problems.append(f"step {name}: no recorded exit status")
    for name, v in status.items():
        if isinstance(v, bool) or not isinstance(v, int) or v != 0:
            problems.append(f"step {name}: exit status {v!r}, not 0")
    return problems


def flushpad_file_summary(payload: dict) -> dict:
    """Summary written into a flush-pad score JSON after every episode (the verdict JSONs are written by the
    launcher from these files). Smoke runs report the pipeline checks and never a verdict on the variant."""
    stage = payload.get("stage")
    if stage == "probe":
        return flushpad_probe_verdict(payload)
    if stage in ("hold", "place"):
        return flushpad_group_summary(payload, stage)
    return flushpad_smoke_check(payload)


def flushpad_verdict_line(v: dict) -> str:
    """One line for the log, from a verdict / summary dict."""
    stage = v.get("stage")
    if stage == "probe":
        body = (f"{v.get('successes')}/{v.get('of')} seeds kept cube tilt <= {FLUSHPAD_TILT_MAX_RAD} rad with no fall "
                f"(need >= {FLUSHPAD_PROBE_RULE['pass_min']}); missing {v.get('missing_seeds')}; "
                + " ".join(f"s{r['seed']}:max_tilt={r.get('cube_tilt_max_rad')},fail={r['first_failed_check']}"
                           for r in v.get("per_seed", [])))
        if v.get("reason"):
            body += f" ({v['reason']})"
    elif stage == "scored":
        body = " ".join(f"{g['stage']}/crew{g['crew']}/pair{g['pair']}="
                        + ("missing" if g.get("missing_json") else f"{g['successes']}/{g['episodes']}")
                        for g in v.get("groups", [])) or v.get("reading", "")
        if v.get("invalid"):
            body += f" invalid: {v['invalid']}"
    elif stage in ("hold", "place"):
        body = " ".join(f"crew{g['crew']}/pair{g['pair']}={g['successes']}/{g['episodes']}"
                        for g in v.get("groups", [])) or f"invalid: {v.get('invalid')}"
    else:
        body = (" ".join(f"seed{r['seed']}/pair{r['pair']}:max_tilt={r['diagnostic_cube_tilt_max_rad']:.3f},"
                         f"base={r['diagnostic_base_first_failed_check'] or 'ok'}" for r in v.get("episodes", []))
                + (f" problems: {v['problems']}" if v.get("problems") else ""))
    verdict = v.get("verdict") or ("INVALID" if v.get("invalid") else
                                   "COMPLETE" if v.get("complete") else "INCOMPLETE")
    return f"COOP-FLUSHPAD {stage}: {verdict} | {body} | {FLUSHPAD_NOTE} | {FLUSHPAD_HARNESS_NOTE} | {FLUSHPAD_LABEL}"


# --- coop-wristhold --- (W, opt-in; 2026-10-03) ----------------------------------------------------------------------
#
# OPT-IN, 2026-10-03: design (W) of the user approval recorded in SLURM_JOBS.md on 2026-10-03 09:55 ("cooperative
# scripted lift that holds wrist orientation"); launcher slurm/repo20260923/cpu_coop_wristhold.sbatch; CLI entry
# `wristhold_main` in scripts/bench/coop_scripted_carry.py (its own entry point: that script's `main()` is unchanged).
# Nothing in this section runs unless the caller passes a `WristHoldParams` (or asks for the smoke's measure-only
# baseline); no stock or flush-pad line is changed (tests/test_coop_wristhold.py: this file's diff to 3a67bc2 is this
# section only, appended; the stock and flush-pad models are byte-identical to 3a67bc2's and short stock and flush-pad
# episodes are identical step for step).
#
# What runs: the flush-pad variant EXACTLY as predeclared on 2026-10-02 04:59 (`WristHoldParams` subclasses
# `FlushPadParams` with every field and default unchanged: MODIFIED END-EFFECTOR = flush pads, condim 4, torsional
# friction 0.04 m; HARNESS CHANGE = elliptic cone + impratio 10; lift-place lowering = the reverse keyframe,
# `FlushPadPlaceParams`), PLUS a kinematic wrist-orientation hold:
# * The wrist. The BHL arm has no wrist joint: its chain is shoulder pitch -> shoulder roll -> shoulder yaw -> elbow
#   pitch -> elbow roll -> hand link, and the hand link is welded to the elbow-roll body (upstream MJCF: 'Joint is
#   "fixed"'), so the elbow roll is the joint the hand hangs from. The hold servoes it (`WristHoldParams.wrist_joints`).
#   Shoulder yaw is a shoulder joint whose rotation swings the bent forearm and translates the pad (12 cm at its range
#   limit), which a wrist does not do; it is not used. Frozen from kinematics alone, before any episode.
# * What is held: each grasping hand's rotation about the pinch axis -- world x, a constant of the scripted layout (the
#   two robots of a pair stand at x = c -/+ side_off facing +y); never read from the cube -- relative to the hand's
#   orientation at grasp time, in the world frame: the twist angle of the swing-twist decomposition of R R_ref^T.
# * Grasp time: the first policy-step state at or after the end of the squeeze segment (t_settle + t_reach + t_squeeze
#   = 4.5 s): state 113, t = 4.52 s (`wristhold_grasp_state`). The reference is that state's hand orientation.
# * When: every control step (the harness recomputes every arm target once per policy step, 0.04 s) from the grasp
#   state on: lift_hold to the end of the episode (lift + hold); lift_place through the end of the release (lift,
#   hold, lower, release: t < PlaceScript t_end['release'] = 11.7 s). After the release the hold stops and its last
#   correction is withdrawn over the retract segment (11.7 -> 12.7 s) with the script's own smooth profile, so the wrist
#   target never jumps (the stock script never jumps a target).
# * How (`wrist_twist_ik`): from the robot's own measured state (base attitude and arm joint angles read from the
#   simulator, i.e. exact IMU + encoders; never the cube), the wrist angle that brings the hand's twist back to its
#   grasp value, or as close as the joint range (+/-0.785 rad) allows: damped Gauss-Newton on the 1-D task, clamped to
#   the range, with the exact derivative of the twist (`wrist_twist_derivative`: the hand's MuJoCo rotational Jacobian
#   column of the wrist, mj_jacBody, chained through the swing-twist decomposition), a step accepted only if it
#   strictly reduces |twist| from the measured angle projected onto the range (so the target is never worse than the
#   measured state and always within the range), until no step is accepted or a
#   step moves < 1e-10 rad (at most 50 iterations; damping 0.01, a numerical safeguard). The wrist target becomes that
#   solution; every other target is the script's or the gait's.
#   DISCLOSED FIX (2026-10-03, before any smoke, probe or scored episode): the first implementation used the projected
#   Jacobian (pinch axis . wrist axis) as the derivative and accepted every step. A dry run on throw-away seed 140
#   showed that with a large swing this is not the twist's derivative: at the release, as the elbow-roll axis passes
#   perpendicular to the pinch axis, the iteration walked to the opposite range bound and made the twist worse (one hand:
#   -0.003 -> -0.15 rad; target 0 -> -0.785 rad, a 23.6 Nm PD demand), which contradicts the frozen design. Fixed to the
#   exact derivative + strictly decreasing |twist|; no frozen choice (wrist, pinch axis, grasp state, window, gains,
#   cap, keyframes, rules) changed.
# * Unchanged: the arm keyframes and their timeline, grasping-arm PD kp 30 and kd 2, the 4 Nm arm effort cap (the stock
#   PD loop computes and clips every torque; the hold only replaces the wrist entry of each grasping arm's target), the
#   gait, cube, plinth, layout, pads, harness options and scoring.
# * Measured, never gated (`WristHold`): per grasping hand and policy-step state, its twist and swing relative to grasp
#   time; per servoed step, the wrist angle against its range, whether the target was clamped, the unclipped PD demand
#   at the step start against the 4 Nm cap, and the clipped torque applied at the step's last physics substep.
# * Mechanics: `run_wristhold_episode` calls the stock `run_episode` / `run_place_episode` unchanged with a frame_hook
#   chain (CubeTiltRecorder -> WristHold -> the caller's hook). On its first call (after policy step 0) `WristHold`
#   wraps that runner's `step`, so each later call first replaces the grasping arms' wrist targets, then runs the
#   unchanged stock PD loop. The hold engages at step 113, long after the wrap. The wrapper never emits a non-finite
#   target (it falls back to the script's target and counts the event).
# * KINEMATIC PREDICTION (pre-episode; `wristhold_design`, recorded in every output): the cube stops the shoulder roll
#   near 0 (0.01-0.07 rad), so at the contact pose the arm hangs and the forearm points forward. Joint axis . pinch
#   axis at the squeeze / lift contact poses: shoulder pitch 1.00 / 1.00, elbow pitch 0.96 / 0.95, shoulder yaw 0.27 /
#   0.31, elbow roll 0.24 / 0.17, shoulder roll 0.00. The lift keyframe turns the hand ~0.98 rad about the pinch axis,
#   almost all through the two lift joints. Clamped at its range the elbow roll removes ~0.17 rad of it (~18 %) and
#   swings the pad ~0.9 rad off the cube face; no admissible wrist set holds the twist (shoulder yaw + elbow roll:
#   ~31 %). The hold is therefore expected NOT to stop the roll: the probe is expected to read NEGATIVE.

WRISTHOLD_VARIANT = "wristhold_v1"
WRISTHOLD_TILT_MAX_RAD = 0.35
WRISTHOLD_PROBE_SEEDS = (125, 126, 127, 128, 129)
#: Throw-away seeds, the only ones a wrist-hold smoke may run (seed-context grep, 2026-10-03: no coop or scripted-carry
#: run ever used them; outside 0-39, 100-129 and 300-319). The smoke runs seed 140.
WRISTHOLD_SMOKE_SEEDS = (140, 141, 142, 143, 144)
#: The candidate wrist sets the kinematic design record evaluates (the arm joints no keyframe moves); the frozen one is
#: `WristHoldParams.wrist_joints`.
WRISTHOLD_CANDIDATE_SETS = (("elbow_roll",), ("shoulder_yaw",), ("shoulder_yaw", "elbow_roll"))
_WRISTHOLD_ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_roll")  # = ARM_JOINTS_L

WRISTHOLD_HOLD_NOTE = (
    "WRIST-ORIENTATION HOLD: each grasping arm's wrist (elbow roll, the joint the hand link hangs from) is servoed every "
    "control step (0.04 s) by kinematics so the hand's rotation about the pinch axis (world x) stays at its grasp-time "
    "value; arm PD kp 30, the 4 Nm cap and the keyframe timeline unchanged")
WRISTHOLD_LABEL = ("LEARNED gait (frozen arms-dr1.0-s0) + SCRIPTED arms (with a kinematic wrist-orientation hold) + "
                   "ORACLE cube pose (scoring only) | " + FLUSHPAD_NOTE)
WRISTHOLD_LABEL_DETAIL = (
    "legs + outer arm: frozen learned PPO gait arms-dr1.0-s0, unmodified observation, zero velocity command all "
    "episode | grasping arm (one per robot): scripted joint keyframes (unchanged) except the wrist (elbow roll), which a "
    "kinematic hold servoes every control step to keep the hand's rotation about the pinch axis (world x) at its "
    "grasp-time value from the robot's own state (never the cube); PD kp 30 (deploy 10), 4 Nm cap | cube pose: "
    "simulator oracle, scoring only | contact: MuJoCo, no welds | MODIFIED END-EFFECTOR: hand pad = hand-mesh AABB box "
    "re-oriented flush to the cube faces at the squeeze pose, condim 4, torsional friction 0.04 m | HARNESS CHANGE: "
    "elliptic friction cone, impratio 10 (global; also foot-floor contact)")
WRISTHOLD_NOHOLD_LABEL = (FLUSHPAD_LABEL + " | NO-HOLD BASELINE of the wrist-hold smoke: the flush-pad variant "
                          "unchanged, the hand's twist measured only")

#: The W PREDECLARED RULES (user approval 2026-10-03 09:55): the flush-pad probe rule verbatim except the seeds
#: (120-124 -> 125-129) and the flush-pad scored rule verbatim. The launcher header repeats them word for word.
WRISTHOLD_PROBE_RULE_TEXT = (
    "Probe on exploration seeds 125-129 (state the grep evidence that they were never used). PROCEED iff the cube "
    "tilt stays <= 0.35 rad throughout lift-hold on >= 4/5 seeds, with LIFT_HOLD_RULE's robot-fall clauses "
    "unchanged. Otherwise STOP: NEGATIVE, the flush pad does not stop the roll.")
WRISTHOLD_SCORED_RULE_TEXT = (
    "Scored, only after PROCEED (state the evidence that seeds 20-39 were never used): lift-hold on seeds 20-29 and "
    "lift-place on seeds 30-39, each with its UNCHANGED rule (LIFT_HOLD_RULE / LIFT_PLACE_RULE, pass_min 8) PLUS "
    "cube tilt <= 0.35 rad (hold: throughout the hold; place: at release and when seated), >= 8/10 per crew or pair "
    "as the existing rules define them. PASS iff every crew/pair meets it; otherwise NEGATIVE; INCOMPLETE if any "
    "episode is missing.")

#: How the frozen rules are applied: the flush-pad clauses as applied (2026-10-02), with the W seeds, plus the hold's
#: record. Decided 2026-10-03 before any episode of this variant; each reading is stricter than or equal to the text.
WRISTHOLD_CLAUSES_AS_APPLIED = (
    "cube tilt = the angle between the cube's own z axis and world z (CarryRunner.cube_tilt; a cube rolled 90 deg "
    "reads 1.571 rad), logged at every policy-step state (t = i * 0.04 s, i = 0..500) in full float precision; "
    "<= 0.35 rad passes, anything larger or non-finite fails",
    "probe 'throughout lift-hold' = every policy-step state of the 20 s lift_hold episode, t = 0 to 20 s inclusive "
    "(stricter than the hold alone). A probe seed passes iff that tilt clause, LIFT_HOLD_RULE's robot-fall clause "
    "unchanged (no robot of the pair tilts > 0.78 rad at any policy step) and the validity clauses (finite state, "
    "no MuJoCo auto-reset, full 20 s) all hold; crew 2 (one pair), seeds 125-129, PROCEED iff >= 4 of 5",
    "probe disclosure: the rule text names only the robot-fall clauses of LIFT_HOLD_RULE, so its lift, hold and "
    "floor clauses are not gated in the probe (they are reported per seed as diagnostics). A probe episode that "
    "never lifts the cube therefore passes the probe's tilt clause (so would a cube the hold drops that lands flat); "
    "the scored stage keeps every LIFT_HOLD_RULE clause",
    "hold 'throughout the hold' = every policy-step state of the hold LIFT_HOLD_RULE scores (its longest contiguous "
    ">= 0.05 m run, hold_start_s inclusive to hold_end_s exclusive); no hold = the clause fails",
    "place 'at release' = every policy-step state from the first at or after the end of the lowering "
    "(PlaceScript t_end['unload'] = 10.7 s: state 268, t = 10.72 s) through the first at or after the hands are "
    "fully open (t_end['release'] = 11.7 s: state 293, t = 11.72 s), inclusive; 'when seated' = the final state "
    "(t = 20 s), where LIFT_PLACE_RULE checks the seated cube",
    "crews as the existing rules and launchers define them: crew 2 (one pair) and crew 4 (two pairs, two cubes, one "
    "world), >= 8/10 per pair; PASS iff all six pairs (lift-hold crew 2, crew 4 pair 0, crew 4 pair 1; lift-place "
    "the same) meet it",
    "an episode stopped by a non-finite state is a scored failure; its non-finite numbers are written as null so the "
    "score JSON holds it. A numerical blow-up that MuJoCo catches itself is a scored failure too, in every stage "
    "(check no_sim_reset): MuJoCo 3.3.5 resets the state automatically (qpos0, time 0) when qpos, qvel or qacc holds a "
    "NaN or a value beyond 1e10, so the harness's own non-finite check never sees it; it is detected after every "
    "policy step from the simulated time and MuJoCo's bad-qpos/qvel/qacc warning counters. An episode that is absent, "
    "shorter than 20 s without a recorded failure, or has no tilt log or no auto-reset record is MISSING (INCOMPLETE)",
    "the wrist-orientation hold adds measurements (each hand's twist and swing, the wrist angle against its range, the "
    "PD demand against the 4 Nm cap), never a gate clause. An episode without a well-formed hold record (hold installed "
    "after policy step 0, engaged at the grasp state 113, the twist logged at every state) is MISSING (INCOMPLETE)",
    "INVALID (never PROCEED, never PASS): a score JSON that is not this variant's or its stage's, not under the frozen "
    "rule, parameters (WristHoldParams), keyframes or lowering, not a scored run, whose compiled model does not show "
    "the designed flush pads with condim 4, torsional friction 0.04 m, elliptic cone and impratio 10, or whose "
    "wrist-hold design record does not name the frozen wrist (elbow roll)",
)

#: Seed-use evidence (grep of results, Slurm logs, the ledger and the investigators' files, 2026-10-03), recorded in
#: every wrist-hold verdict JSON.
WRISTHOLD_SEED_EVIDENCE = (
    "grep -rhoE '\"seed\": *[0-9]+' over results/scripted-carry-20260926, results/scripted-carry-20260927, "
    "results/coop-flushpad-20261002 and every logs/coop-flushpad-*-smoke directory finds seeds 0-19, 119 and 120-124 "
    "only: no episode of this harness ran any seed 20-39 or 125-129",
    "Slurm logs of this harness: scripted-carry-21434982 and coop-lift-hold-21435079 ran seeds 0-9, "
    "coop-lift-place-21443282 seeds 10-19, the flush-pad smokes seed 119 and the flush-pad probe 21507881 seeds "
    "120-124; its scored stage did not run (results/coop-flushpad-20261002/scored/verdict.json: NOT_RUN, seeds 20-39 "
    "stay unused)",
    "exploration: the harness docstring and lift-place notes name seeds 100-119 only; the 2026-09-30 investigators' "
    "files (solutions-20260930/coop, verify-coop) hold seeds 100-104 only; the flush-pad workstream's files 119-124",
    "a seed-context scan (2026-10-03) of the 255 coop / carry / crew / cube / flush files under results/ and logs/, "
    "the investigators' and flush-pad workstream's files and SLURM_JOBS.md finds seed-like integers <= 1000 only in "
    "0-39, 99 (a MARL smoke), 100-129 and 300-319; 125-129 appear only in the ledger's 2026-10-03 design (W) and 20-39 "
    "only as reserved, unused scored seeds (ledger text and the rule text inside the flush-pad JSONs)",
    "smoke seeds 140-144: the same scan and a targeted grep find them in no seed context; the smoke runs seed 140 only "
    "(the CLI refuses any seed outside 140-144 in smoke mode, and every probe or scored seed outside its stage)",
)

#: Probe (stage 1): the flush-pad probe rule with the W seeds; crew 2, lift_hold protocol (20 s), seeds 125-129.
WRISTHOLD_PROBE_RULE = {**FLUSHPAD_PROBE_RULE, "variant": WRISTHOLD_VARIANT, "seeds": list(WRISTHOLD_PROBE_SEEDS)}
#: Scored lift-hold and lift-place (stage 2): the flush-pad scored rules verbatim (seeds 20-29 and 30-39, crews 2, 4).
WRISTHOLD_HOLD_RULE = {**FLUSHPAD_HOLD_RULE, "variant": WRISTHOLD_VARIANT}
WRISTHOLD_PLACE_RULE = {**FLUSHPAD_PLACE_RULE, "variant": WRISTHOLD_VARIANT}
WRISTHOLD_STAGES = ("probe", "hold", "place", "smoke", "smoke_nohold")


@dataclass
class WristHoldParams(FlushPadParams):
    """OPT-IN W variant: `FlushPadParams` (every field and default unchanged: flush pads, condim 4, torsional 0.04 m,
    elliptic cone, impratio 10) plus the kinematic wrist-orientation hold. Frozen 2026-10-03, before any episode."""
    wrist_hold: str = "pinch_axis_twist"        # the hand's rotation about the pinch axis (world), held at grasp time
    wrist_joints: tuple = ("elbow_roll",)       # the BHL arm's wrist: the joint the hand link hangs from
    pinch_axis_world: tuple = (1.0, 0.0, 0.0)   # world x: the layout's pinch axis (never read from the cube)
    wrist_ik_damping: float = 0.01              # numerical safeguard; the converged solution does not depend on it
    wrist_ik_max_iter: int = 50
    wrist_ik_tol: float = 1e-10                 # rad
    wrist_release_fade: str = "retract"         # lift_place: the last correction is withdrawn over the retract segment


def is_wristhold(p) -> bool:
    """True only for the opt-in W variant (a stock `CarryParams` or a `FlushPadParams` has no `wrist_hold`)."""
    return is_flushpad(p) and getattr(p, "wrist_hold", None) == "pinch_axis_twist"


def wristhold_grasp_state(dt: float, p=None) -> int:
    """Index of the grasp state: the first policy-step state at or after the end of the squeeze segment."""
    import math
    p = WristHoldParams() if p is None else p
    return int(math.ceil((p.t_settle + p.t_reach + p.t_squeeze) / float(dt) - 1e-9))


def _wristhold_rel_quat(R_ref, R) -> np.ndarray:
    """Unit quaternion (w >= 0) of the world-frame rotation taking attitude `R_ref` to `R` (R R_ref^T)."""
    rel = np.asarray(R, dtype=float).reshape(3, 3) @ np.asarray(R_ref, dtype=float).reshape(3, 3).T
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, rel.flatten())
    return -q if q[0] < 0.0 else q


def hand_twist_swing(R_ref, R, axis) -> tuple:
    """(twist, swing) of the world-frame rotation taking attitude `R_ref` to `R` (R R_ref^T): the twist is its signed
    angle about `axis` (swing-twist decomposition, in [-pi, pi]); the swing is the angle of the rest, a rotation about
    an axis normal to `axis`. With q = (w, v) the unit quaternion (w >= 0) and p = v . axis: twist = 2 atan2(p, w),
    swing = 2 acos(sqrt(w^2 + p^2))."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    q = _wristhold_rel_quat(R_ref, R)
    p = float(q[1:] @ a)
    return float(2.0 * np.arctan2(p, q[0])), float(2.0 * np.arccos(min(1.0, float(np.sqrt(q[0] ** 2 + p ** 2)))))


def wrist_twist_derivative(R_ref, R, axis, omegas) -> np.ndarray:
    """Exact derivative of `hand_twist_swing`'s twist with respect to turning the attitude `R` about each world axis in
    `omegas` (for a hinge joint: its MuJoCo rotational Jacobian column, i.e. its world axis). With q = (w, v) as in
    `hand_twist_swing` and p = v . a, turning by d about omega gives dq = (-omega . v, w omega + omega x v) d / 2, so
    d twist / d = (w (w omega . a + (omega x v) . a) + p omega . v) / (w^2 + p^2). It equals omega . a only when the
    rotation is a pure twist (no swing); 0 where the twist is undefined (w^2 + p^2 = 0, a half-turn swing)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    q = _wristhold_rel_quat(R_ref, R)
    w, v = float(q[0]), q[1:]
    p = float(v @ a)
    n2 = w * w + p * p
    om = np.asarray(omegas, dtype=float).reshape(-1, 3)
    if n2 < 1e-12:
        return np.zeros(len(om))
    return np.array([(w * (w * float(o @ a) + float(np.cross(o, v) @ a)) + p * float(o @ v)) / n2 for o in om])


def wrist_twist_ik(model, data, body: int, joints, R_ref, axis, *, damping: float, max_iter: int, tol: float) -> dict:
    """The hold's small IK: the values of the hinge `joints` [(joint id, lo, hi), ...] that bring `body`'s twist about
    `axis` (relative to `R_ref`, world frame) back to 0, or as close as their ranges allow, starting from the
    configuration in `data` (the measured state; its kinematics need not be current). Damped Gauss-Newton on the 1-D
    task with each iterate clamped to [lo, hi]; the derivative is exact (`wrist_twist_derivative` of the body's MuJoCo
    rotational Jacobian columns, mj_jacBody, at those joints' dofs); a step is accepted only if it strictly reduces
    |twist| (halved up to 40 times otherwise) from the measured angle projected onto the range, so the result is
    never worse than the measured state (up to that projection: a joint can sit a hair past its soft limit). Stops when no
    step is accepted, a step moves less than `tol`, or after `max_iter`. `data`'s joint values are restored (and its
    kinematics recomputed) before returning. Returns q (the solution), the twist before and after, the derivative at
    the start, the number of iterations, whether a range bound holds the solution, and the unclamped one-step (linear)
    solution."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    jids = [int(j) for j, _, _ in joints]
    adr = np.array([model.jnt_qposadr[j] for j in jids])
    dof = np.array([model.jnt_dofadr[j] for j in jids])
    lo = np.array([float(v) for _, v, _ in joints])
    hi = np.array([float(v) for _, _, v in joints])
    q0 = data.qpos[adr].copy()
    jacr = np.zeros((3, model.nv))

    def evaluate(q):
        data.qpos[adr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_jacBody(model, data, None, jacr, body)
        R = data.xmat[body]
        return (hand_twist_swing(R_ref, R, a)[0],
                wrist_twist_derivative(R_ref, R, a, [jacr[:, k] for k in dof]))

    e0, J0 = evaluate(q0)                         # the hand's twist in the measured state
    q = np.clip(q0, lo, hi)                       # start: the measured angle, projected onto the range (a joint may
    e, J = evaluate(q) if np.any(q != q0) else (e0, J0.copy())   # sit a hair past its soft limit)
    iters = 0
    for iters in range(1, int(max_iter) + 1):
        step = -J * e / (float(J @ J) + damping ** 2)
        alpha, accepted, moved = 1.0, False, 0.0
        for _ in range(40):
            qn = np.clip(q + alpha * step, lo, hi)
            moved = float(np.max(np.abs(qn - q)))
            if moved < tol:
                break
            en, Jn = evaluate(qn)
            if abs(en) < abs(e):
                q, e, J, accepted = qn, en, Jn, True
                break
            alpha *= 0.5
        if not accepted or moved < tol:
            break
    linear = q0 - J0 * e0 / (float(J0 @ J0) + damping ** 2)
    data.qpos[adr] = q0
    mujoco.mj_kinematics(model, data)
    return {"q": q, "twist_before": float(e0), "twist_after": float(e), "J": J0, "iters": int(iters),
            "clamped": bool(np.any((q <= lo + 1e-12) | (q >= hi - 1e-12))), "q_linear": linear}


def wrist_hold_design(robot_xml: Path, p, keyframes: dict = KEYFRAMES_LEFT, n_sweep: int = 11) -> dict:
    """Kinematics only (no episode, no seed): the arm chain, each joint axis' projection on the pinch axis at the squeeze
    and lift contact poses, the hand's twist along the scripted squeeze -> lift sweep without a hold, and what a clamped
    hold on each candidate wrist set (`WRISTHOLD_CANDIDATE_SETS`) leaves of it. Robot alone, base upright at the origin
    (robot frame: x forward, y left = the pinch axis, z up), flush pads; along the sweep the shoulder pitch and elbow
    pitch go linearly from the squeeze to the lift keyframe while the shoulder roll stops where the pad's outer face
    reaches the cube face plane (the cube stops the squeeze and the lift short of their keyframes, as in
    `flush_pad_frames`); the reference is the hand at the squeeze contact pose."""
    if not is_wristhold(p):
        raise ValueError("wrist_hold_design needs WristHoldParams")
    flush = flush_pad_frames(robot_xml, p, keyframes)
    fk = PadFK(robot_xml, flush["frames"])
    m, d = fk.m, fk.d
    face = float(p.side_off - CUBE_HALF)
    a = np.array([0.0, 1.0, 0.0])
    sq = np.asarray(keyframes["squeeze"], dtype=float)
    lf = np.asarray(keyframes["lift"], dtype=float)
    us = [float(u) for u in np.linspace(0.0, 1.0, int(n_sweep))]
    sides = {}
    for side in ("left", "right"):
        sgn = 1.0 if side == "left" else -1.0
        names = ARM_JOINTS_L if side == "left" else ARM_JOINTS_R
        jid = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for s, n in zip(_WRISTHOLD_ARM, names)}
        hand, pad = fk.body[side], fk.geom[side]

        def pose_at(u):
            ql = sq + u * (lf - sq)
            ql[2] = ql[4] = 0.0
            ql[1] = _bisect_increasing(
                lambda r: fk.lateral_extent(side, sgn * np.array([ql[0], r, 0.0, ql[3], 0.0])) - face, -0.6, 1.3)
            fk.pose(side, sgn * ql)
            return ql

        ql0 = pose_at(0.0)
        R_ref = d.xmat[hand].reshape(3, 3).copy()
        proj, sweep = {}, []
        for tag, u in (("squeeze_contact", 0.0), ("lift_contact", 1.0)):
            ql = pose_at(u)
            proj[tag] = {"joints_left_order": [round(float(v), 6) for v in ql],
                         "axis_dot_pinch": {s: round(float(a @ d.xaxis[jid[s]]), 4) for s in _WRISTHOLD_ARM}}
        cands = {"+".join(c): [] for c in WRISTHOLD_CANDIDATE_SETS}
        for u in us:
            ql = pose_at(u)
            c0 = d.geom_xpos[pad].copy()
            tw0, sw0 = hand_twist_swing(R_ref, d.xmat[hand], a)
            sweep.append({"u": round(u, 4), "shoulder_roll_at_contact": round(float(ql[1]), 4),
                          "twist_no_hold_rad": round(tw0, 4), "swing_no_hold_rad": round(sw0, 4)})
            for c in WRISTHOLD_CANDIDATE_SETS:
                js = [(jid[s], *m.jnt_range[jid[s]]) for s in c]
                sol = wrist_twist_ik(m, d, hand, js, R_ref, a, damping=p.wrist_ik_damping,
                                     max_iter=p.wrist_ik_max_iter, tol=p.wrist_ik_tol)
                adr = [m.jnt_qposadr[j] for j, _, _ in js]
                d.qpos[adr] = sol["q"]
                mujoco.mj_kinematics(m, d)
                tw1, sw1 = hand_twist_swing(R_ref, d.xmat[hand], a)
                cands["+".join(c)].append({
                    "u": round(u, 4), "twist_hold_rad": round(tw1, 4), "swing_hold_rad": round(sw1, 4),
                    "pad_moved_m": round(float(np.linalg.norm(d.geom_xpos[pad] - c0)), 4),
                    "wrist_q_actual": [round(float(v), 4) for v in sol["q"]], "clamped": sol["clamped"]})
                fk.pose(side, sgn * ql)
        summary = {}
        tw_end = sweep[-1]["twist_no_hold_rad"]
        for name, rows in cands.items():
            end = rows[-1]
            sat = next((r["u"] for r in rows if r["clamped"]), None)
            summary[name] = {
                "twist_at_lift_contact_no_hold_rad": tw_end, "twist_at_lift_contact_hold_rad": end["twist_hold_rad"],
                "removed_fraction": round(1.0 - abs(end["twist_hold_rad"]) / abs(tw_end), 4) if tw_end else None,
                "swing_max_hold_rad": max(r["swing_hold_rad"] for r in rows),
                "pad_moved_max_m": max(r["pad_moved_m"] for r in rows), "range_binds_from_u": sat}
        sides[side] = {"chain": [n for n in names] + [f"arm_{side}_hand_link (welded to the elbow-roll body)"],
                       "ranges_rad": {s: [round(float(v), 6) for v in m.jnt_range[jid[s]]] for s in _WRISTHOLD_ARM},
                       "projections": proj, "sweep_no_hold": sweep, "candidates": cands, "candidate_summary": summary}
    chosen = "+".join(p.wrist_joints)
    worst = max(abs(sides[s]["candidate_summary"][chosen]["twist_at_lift_contact_hold_rad"]) for s in sides)
    best_any = min(abs(sides[s]["candidate_summary"][c]["twist_at_lift_contact_hold_rad"])
                   for s in sides for c in sides[s]["candidate_summary"])
    prediction = (
        f"chosen wrist {chosen}: with the clamped hold the hand's twist at the lift contact pose is "
        + ", ".join(f"{s} {sides[s]['candidate_summary'][chosen]['twist_at_lift_contact_hold_rad']:+.3f} rad (no hold "
                    f"{sides[s]['candidate_summary'][chosen]['twist_at_lift_contact_no_hold_rad']:+.3f}, removed "
                    f"{100 * sides[s]['candidate_summary'][chosen]['removed_fraction']:.0f} %)" for s in sides)
        + f"; the smallest |twist| any candidate set leaves is {best_any:.3f} rad. "
        + ("Every candidate leaves more than 0.35 rad, so a cube that rolls with the hands would still tilt past the "
           "probe's 0.35 rad limit: the hold is expected not to stop the roll (probe expected NEGATIVE)."
           if best_any > WRISTHOLD_TILT_MAX_RAD else
           f"The chosen set leaves {worst:.3f} rad."))
    return {"method": "kinematics only (PadFK, flush pads, base upright at the origin); clamped wrist_twist_ik",
            "pinch_axis_robot_frame": [0.0, 1.0, 0.0], "pinch_axis_world": [float(v) for v in p.pinch_axis_world],
            "chosen": list(p.wrist_joints),
            "selection_rule": ("the wrist = the joint the hand link hangs from: the elbow roll (the hand link is welded "
                               "to the elbow-roll body). Shoulder yaw is a shoulder joint whose rotation swings the bent "
                               "forearm and translates the pad (see pad_moved_max_m), which a wrist does not do."),
            "sides": sides, "prediction": prediction}


def wristhold_design(upstream: Path, cache_dir: Path, p) -> dict:
    """`wrist_hold_design` on the same MJCF copy `build_carry` uses (recorded in every wrist-hold output)."""
    scene = prepare_mjcf(upstream, cache_dir, "humanoid")
    return wrist_hold_design(scene.parent / "berkeley_humanoid_lite.xml", p)


class WristHold:
    """frame_hook of the W variant and, with `servo`, the wrist-orientation hold itself.

    As a frame_hook (the stock episode loop calls it after every policy step) it measures, for each pair's grasping
    hands (robot b's right, robot a's left), the hand's twist and swing relative to its grasp-time orientation at every
    policy-step state, recording the reference at the grasp state. With `servo` its first call (after policy step 0)
    wraps that runner's `step`: from the grasp step on, each call replaces the wrist entries of the grasping arms'
    targets with the hold's (`wrist_twist_ik`, clamped to the joint range) before the unchanged stock PD loop runs, and
    logs the wrist angle, the clamp, the unclipped PD demand at the step start and the clipped torque applied at the
    step's last physics substep. `servo=False` only measures (the smoke's no-hold baseline)."""

    def __init__(self, model, slots, pairs, dt: float, hp=None, servo: bool = True, chain=None):
        self.hp = WristHoldParams() if hp is None else hp
        if not is_wristhold(self.hp):
            raise ValueError("WristHold needs WristHoldParams")
        self.m, self.slots, self.pairs, self.dt = model, slots, pairs, float(dt)
        self.servo, self.chain = bool(servo), chain
        a = np.asarray(self.hp.pinch_axis_world, dtype=float)
        self.axis = a / np.linalg.norm(a)
        self.k_grasp = wristhold_grasp_state(self.dt, self.hp)
        if self.k_grasp < 1:
            raise ValueError("the grasp state must come after policy step 0")
        self.scratch = mujoco.MjData(model)               # kinematics only; never stepped
        self.arms = []
        for k, pr in enumerate(pairs):
            for i, side in ((pr.robot_b, "right"), (pr.robot_a, "left")):
                s = slots[i]
                names = actuator_joint_names(model, s)
                joints = []
                for w in self.hp.wrist_joints:
                    jn = f"arm_{side}_{w}_joint"
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, s.prefix + jn)
                    if jid < 0 or jn not in names:
                        raise RuntimeError(f"no {s.prefix}{jn}")
                    joints.append({"name": jn, "jid": int(jid), "idx": names.index(jn),
                                   "qadr": int(model.jnt_qposadr[jid]), "lo": float(model.jnt_range[jid][0]),
                                   "hi": float(model.jnt_range[jid][1])})
                hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s.prefix}arm_{side}_hand_link")
                if hand < 0:
                    raise RuntimeError(f"no {s.prefix}arm_{side}_hand_link")
                self.arms.append({"robot": int(i), "pair": int(k), "side": side, "hand": int(hand), "joints": joints,
                                  "ref": None, "twist": [None], "swing": [None], "delta": None,
                                  "log": {n: [] for n in ("step", "mode", "q", "target", "clamped", "twist_before_ik",
                                                          "twist_after_ik", "J", "iters", "q_linear",
                                                          "pd_demand_nm", "ctrl_after_nm")}})
        self.runner = self.script = None
        self.next_step = None
        self.installed_after_step = None
        self.window_end_t = self.fade_end_t = None
        self.active = []
        self.nonfinite = 0

    # ---- frame_hook: called by the stock loop after every policy step
    def __call__(self, **kw):
        runner, step = kw["runner"], int(kw["step"])
        if self.runner is None:
            self._install(runner, kw["script"], step)
        self.next_step = step + 1
        self._measure(step + 1)
        if self.chain is not None:
            self.chain(**kw)

    def _install(self, runner, script, step: int) -> None:
        if step != 0:
            raise RuntimeError("WristHold must be chained into the episode from policy step 0")
        lift_start = float(script.segs[3][0])
        if abs(lift_start - (self.hp.t_settle + self.hp.t_reach + self.hp.t_squeeze)) > 1e-9:
            raise RuntimeError("the script's lift segment does not start at the end of the squeeze")
        self.runner, self.script = runner, script
        if isinstance(script, PlaceScript):
            self.window_end_t, self.fade_end_t = float(script.t_end["release"]), float(script.t_end["retract"])
        else:
            self.window_end_t = self.fade_end_t = float("inf")
        if self.servo:
            original = runner.step

            def step_with_wrist_hold(targets_per_robot):
                held = self._filter(targets_per_robot)
                out = original(held)
                self._after_step()
                return out

            runner.step = step_with_wrist_hold
        self.installed_after_step = int(step)

    def _kinematics(self):
        self.scratch.qpos[:] = self.runner.d.qpos
        mujoco.mj_kinematics(self.m, self.scratch)
        return self.scratch

    def _measure(self, state: int) -> None:
        sd = self._kinematics()
        for arm in self.arms:
            R = sd.xmat[arm["hand"]].reshape(3, 3).copy()
            if state == self.k_grasp:
                arm["ref"] = R
            if arm["ref"] is None:
                arm["twist"].append(None)
                arm["swing"].append(None)
            else:
                tw, sw = hand_twist_swing(arm["ref"], R, self.axis)
                arm["twist"].append(tw)
                arm["swing"].append(sw)

    # ---- the hold: wraps runner.step (servo only)
    def _filter(self, targets_per_robot):
        k = int(self.next_step)
        t = k * self.dt
        self.active = []
        if k < self.k_grasp:
            return targets_per_robot
        if t < self.window_end_t - 1e-9:
            mode = "hold"
        elif t < self.fade_end_t - 1e-9:
            mode = "fade"
        else:
            return targets_per_robot
        out = list(targets_per_robot)
        sd = self._kinematics()
        hp, r = self.hp, self.runner
        for arm in self.arms:
            if arm["ref"] is None:
                raise RuntimeError("no grasp reference at a servoed step")
            i = arm["robot"]
            new = np.array(targets_per_robot[i], dtype=float)
            idx = [j["idx"] for j in arm["joints"]]
            incoming = new[idx].copy()
            sol = None
            if mode == "hold":
                sol = wrist_twist_ik(self.m, sd, arm["hand"], [(j["jid"], j["lo"], j["hi"]) for j in arm["joints"]],
                                     arm["ref"], self.axis, damping=hp.wrist_ik_damping, max_iter=hp.wrist_ik_max_iter,
                                     tol=hp.wrist_ik_tol)
                q_new = np.asarray(sol["q"], dtype=float)
                if not np.isfinite(q_new).all():
                    self.nonfinite += 1
                    q_new = incoming
                arm["delta"] = q_new - incoming
            else:
                u = (t - self.window_end_t) / (self.fade_end_t - self.window_end_t)
                q_new = incoming + (arm["delta"] if arm["delta"] is not None else 0.0) * (1.0 - _smooth(u))
            new[idx] = q_new
            out[i] = new
            s = self.slots[i]
            lg = arm["log"]
            lg["step"].append(k)
            lg["mode"].append(mode)
            lg["q"].append([float(r.d.qpos[j["qadr"]]) for j in arm["joints"]])
            lg["target"].append([float(v) for v in q_new])
            lg["clamped"].append(bool(sol["clamped"]) if sol else None)
            lg["twist_before_ik"].append(sol["twist_before"] if sol else None)
            lg["twist_after_ik"].append(sol["twist_after"] if sol else None)
            lg["J"].append([float(v) for v in sol["J"]] if sol else None)
            lg["iters"].append(sol["iters"] if sol else None)
            lg["q_linear"].append([float(v) for v in sol["q_linear"]] if sol else None)
            # the first physics substep's PD torque on the wrist, before the stock clip (CarryRunner.step's own formula)
            lg["pd_demand_nm"].append([float(r.kp_r[i][x] * (q_new[n] - r.d.sensordata[s.jpos_adr[x]])
                                             - r.kd[x] * r.d.sensordata[s.jvel_adr[x]]) for n, x in enumerate(idx)])
            self.active.append(arm)
        return out

    def _after_step(self) -> None:
        for arm in self.active:
            s = self.slots[arm["robot"]]
            arm["log"]["ctrl_after_nm"].append(
                [float(self.runner.d.ctrl[s.ctrl[j["idx"]]]) for j in arm["joints"]])
        self.active = []

    # ---- the episode's record
    def record(self, steps: int) -> dict:
        """The hold's record for the episode JSON (`steps` = policy steps run)."""
        r = self.runner
        i_lift = int(round((self.hp.t_settle + self.hp.t_reach + self.hp.t_squeeze + self.hp.t_lift) / self.dt))
        arms = []
        for arm in self.arms:
            lg = arm["log"]
            tw = [None if v is None else float(v) for v in arm["twist"]]
            sw = [None if v is None else float(v) for v in arm["swing"]]
            held = [v for v in tw if v is not None]
            q = np.array(lg["q"], dtype=float).reshape(-1, len(arm["joints"]))
            dem = np.array(lg["pd_demand_nm"], dtype=float).reshape(-1, len(arm["joints"]))
            ctrl = np.array(lg["ctrl_after_nm"], dtype=float).reshape(-1, len(arm["joints"]))
            lo = np.array([j["lo"] for j in arm["joints"]])
            hi = np.array([j["hi"] for j in arm["joints"]])
            cap = (np.array([float(r.eff[j["idx"]]) for j in arm["joints"]]) if r is not None
                   else np.full(len(arm["joints"]), np.nan))
            hold_rows = [n for n, md in enumerate(lg["mode"]) if md == "hold"]
            clamped = [lg["step"][n] for n in hold_rows if lg["clamped"][n]]
            above = [lg["step"][n] for n in range(len(lg["step"])) if np.any(np.abs(dem[n]) > cap + 1e-9)]
            summary = {
                "twist_at_grasp_rad": tw[self.k_grasp] if len(tw) > self.k_grasp else None,
                "twist_at_lift_end_rad": tw[i_lift] if len(tw) > i_lift else None,
                "lift_end_state": i_lift,
                "twist_final_rad": tw[-1] if held else None,
                "twist_max_abs_rad": max(abs(v) for v in held) if held else None,
                "swing_max_rad": max(v for v in sw if v is not None) if held else None,
                "servoed_steps": len(lg["step"]), "held_steps": len(hold_rows),
                "wrist_q_min": q.min(axis=0).tolist() if q.size else None,
                "wrist_q_max": q.max(axis=0).tolist() if q.size else None,
                "wrist_range": [[j["lo"], j["hi"]] for j in arm["joints"]],
                "steps_within_0.01_rad_of_a_limit": int(np.sum(np.any((q - lo < 0.01) | (hi - q < 0.01), axis=1)))
                if q.size else 0,
                "beyond_range_max_rad": float(np.max(np.maximum(lo - q, q - hi).clip(min=0.0))) if q.size else 0.0,
                "target_clamped_steps": len(clamped),
                "first_clamped_t_s": round(clamped[0] * self.dt, 3) if clamped else None,
                "pd_demand_max_abs_nm": float(np.max(np.abs(dem))) if dem.size else None,
                "pd_demand_above_cap_steps": len(above),
                "first_above_cap_t_s": round(above[0] * self.dt, 3) if above else None,
                "cap_nm": cap.tolist(),
                "ctrl_after_max_abs_nm": float(np.max(np.abs(ctrl))) if ctrl.size else None,
                "J_abs_min": float(min(abs(v) for row in lg["J"] if row for v in row)) if hold_rows else None,
                "J_abs_max": float(max(abs(v) for row in lg["J"] if row for v in row)) if hold_rows else None,
            }
            arms.append({
                "robot": arm["robot"], "pair": arm["pair"], "side": arm["side"],
                "wrist_joints": [j["name"] for j in arm["joints"]],
                "reference_quat": (None if arm["ref"] is None else
                                   [float(v) for v in _mat2quat(arm["ref"])]),
                "twist_series_rad": tw, "swing_series_rad": sw,
                "servo_log": {n: v for n, v in lg.items()}, "summary": summary})
        return {"variant": WRISTHOLD_VARIANT, "servo": self.servo, "installed": self.runner is not None,
                "installed_after_step": self.installed_after_step, "grasp_state": self.k_grasp,
                "grasp_t_s": round(self.k_grasp * self.dt, 6), "pinch_axis_world": self.axis.tolist(),
                "wrist_joints": list(self.hp.wrist_joints),
                "window": {"from_state": self.k_grasp,
                           "hold_until_t_s": None if self.window_end_t in (None, float("inf")) else self.window_end_t,
                           "fade_until_t_s": None if self.fade_end_t in (None, float("inf")) else self.fade_end_t},
                "ik": {"damping": self.hp.wrist_ik_damping, "max_iter": self.hp.wrist_ik_max_iter,
                       "tol": self.hp.wrist_ik_tol, "nonfinite_fallbacks": self.nonfinite},
                "policy_steps": int(steps), "arms": arms,
                "note": ("twist/swing_series_rad[i] = the hand's twist about the pinch axis / swing relative to the grasp "
                         "state, at policy-step state i (null before the grasp state); servo_log rows are the servoed "
                         "policy steps (pre-step state): wrist angle q, target sent, clamped (hold) or null (fade), the "
                         "IK's twist before/after, the twist derivative J (rad/rad), the unclipped PD demand at the step start "
                         "and the clipped torque applied at the step's last physics substep (N m)")}


def _mat2quat(R) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).flatten())
    return q


def run_wristhold_episode(model, slots, pairs, cfg, policy, seed: int, p, rule: dict,
                          q: PlaceParams | None = None, frame_hook=None) -> dict:
    """One W episode: the stock lift_hold (`run_episode`) or lift_place (`run_place_episode`) protocol on the flush-pad
    model, unchanged, with the wrist-orientation hold (`WristHold`), plus the flush-pad records (per-step cube tilt
    `cube_tilt_series_rad`, MuJoCo auto-reset `sim_reset`, non-finite numbers as null) and the hold's `wrist_hold`
    record. A rule of stage "smoke_nohold" runs the flush-pad variant itself (`FlushPadParams`) with the hold's
    measurement only: the smoke's no-hold baseline."""
    if rule.get("variant") != WRISTHOLD_VARIANT:
        raise ValueError("rule is not a wrist-hold rule")
    nohold = rule.get("stage") == "smoke_nohold"
    if nohold:
        if not is_flushpad(p) or is_wristhold(p) or rule.get("protocol") != "lift_hold":
            raise ValueError("the no-hold baseline runs FlushPadParams under the lift_hold protocol")
    elif not is_wristhold(p):
        raise ValueError("run_wristhold_episode needs WristHoldParams")
    hold = WristHold(model, slots, pairs, float(cfg.policy_dt), None if nohold else p, servo=not nohold,
                     chain=frame_hook)
    rec = CubeTiltRecorder(model, pairs, chain=hold)
    if rule.get("protocol") == "lift_place":
        q = FlushPadPlaceParams() if q is None else q
        if q.lower_to is not None:
            raise ValueError("the flush-pad lowering is FROZEN: reverse keyframe (lower_to=None)")
        ep = run_place_episode(model, slots, pairs, cfg, policy, seed, p, q=q, frame_hook=rec, rule=rule)
    else:
        ep = run_episode(model, slots, pairs, cfg, policy, seed, p, frame_hook=rec, rule=rule, protocol="lift_hold")
    for k, row in enumerate(ep["pairs"]):
        row["cube_tilt_series_rad"] = [_json_tilt(v) for v in rec.series(k)]
    ep["variant"] = WRISTHOLD_VARIANT
    ep["cube_tilt_series_note"] = ("cube_tilt_series_rad[i] = cube tilt (rad, angle of the cube's z axis from world z) "
                                   "at the pre-step state of policy step i (t = i * dt); the last entry is the "
                                   "final state; null = non-finite")
    ep["sim_reset"] = rec.sim_reset_record()
    ep["wrist_hold"] = hold.record(int(ep["steps"]))
    ep, nulled = _json_safe(ep)
    ep["nonfinite_values_nulled"] = nulled
    return ep


def wristhold_stage_rule(stage: str, protocol: str = "lift_hold", seeds=None) -> dict:
    """The frozen rule of a stage; "smoke" = the probe (lift_hold) or place (lift_place) clauses on throw-away seeds;
    "smoke_nohold" = the probe clauses on the no-hold baseline (crew 2, lift_hold)."""
    if stage == "probe":
        return WRISTHOLD_PROBE_RULE
    if stage == "hold":
        return WRISTHOLD_HOLD_RULE
    if stage == "place":
        return WRISTHOLD_PLACE_RULE
    s = sorted(int(x) for x in (seeds or []))
    if stage == "smoke":
        base = WRISTHOLD_PLACE_RULE if protocol == "lift_place" else WRISTHOLD_PROBE_RULE
        return {**base, "stage": "smoke", "seeds": s, "crews": [2, 4], "pass_min": None,
                "smoke": "pipeline check on throw-away seeds 140-144; never scored"}
    if stage == "smoke_nohold":
        return {**WRISTHOLD_PROBE_RULE, "stage": "smoke_nohold", "seeds": s, "crews": [2], "pass_min": None,
                "wrist_hold": "OFF: the flush-pad variant unchanged; the hand's twist is measured only",
                "smoke": "no-hold baseline of the smoke's hand-twist measurement on throw-away seeds 140-144; never scored"}
    raise ValueError(f"unknown wrist-hold stage {stage!r}")


def _wristhold_head(stage: str) -> dict:
    return {"variant": WRISTHOLD_VARIANT, "stage": stage, "label": WRISTHOLD_LABEL,
            "label_detail": WRISTHOLD_LABEL_DETAIL, "end_effector": FLUSHPAD_NOTE, "harness_change": FLUSHPAD_HARNESS_NOTE,
            "wrist_hold": WRISTHOLD_HOLD_NOTE, "clauses_as_applied": list(WRISTHOLD_CLAUSES_AS_APPLIED),
            "seed_evidence": list(WRISTHOLD_SEED_EVIDENCE)}


def _wristhold_record_problem(ep: dict, servo: bool, dt: float) -> str | None:
    """Why an episode's wrist-hold record is not well formed (None when it is)."""
    rec = ep.get("wrist_hold") if isinstance(ep, dict) else None
    if not isinstance(rec, dict):
        return "no wrist-hold record"
    if rec.get("variant") != WRISTHOLD_VARIANT:
        return "wrist-hold record of another variant"
    if rec.get("servo") is not bool(servo):
        return f"wrist-hold servo {rec.get('servo')!r}, expected {bool(servo)}"
    if rec.get("installed") is not True or rec.get("installed_after_step") != 0:
        return "the hold was not installed after policy step 0"
    k_g = wristhold_grasp_state(dt)
    if rec.get("grasp_state") != k_g:
        return f"grasp state {rec.get('grasp_state')!r}, expected {k_g}"
    steps = int(ep.get("steps", -1))
    arms = rec.get("arms")
    if not isinstance(arms, list) or len(arms) != 2 * len(ep.get("pairs", [])):
        return "the record does not hold both grasping hands of every pair"
    stopped = ep.get("failed") is not None        # a stopped episode is a scored failure: its record need only exist
    for arm in arms:
        ser = arm.get("twist_series_rad") if isinstance(arm, dict) else None
        if not isinstance(ser, list) or len(ser) != steps + 1:
            return "twist series missing or not steps + 1 long"
        log = (arm.get("servo_log") or {}).get("step")
        if not servo and log:
            return "a no-hold record with servoed steps"
        if stopped:
            continue
        if steps >= k_g and (ser[k_g] is None or abs(float(ser[k_g])) > 1e-9):
            return "the twist is not 0 at the grasp state"
        if servo and steps > k_g and (not isinstance(log, list) or not log or log[0] != k_g):
            return "the hold did not engage at the grasp state"
    return None


def _wristhold_payload_problem(payload: dict, stage: str, rule: dict, crew: int) -> str | None:
    """Why a score JSON cannot be read under `rule` (None when it can)."""
    if not isinstance(payload, dict):
        return "no score JSON"
    if payload.get("variant") != WRISTHOLD_VARIANT:
        return "not a wrist-hold score JSON"
    if payload.get("stage") != stage:
        return f"stage {payload.get('stage')!r}, expected {stage!r}"
    if payload.get("rule") != _jsonable(rule):
        return "rule differs from the frozen wrist-hold rule"
    if payload.get("params") != _jsonable(params_dict(WristHoldParams())):
        return "params differ from the frozen WristHoldParams"
    if payload.get("keyframes_left") != _jsonable(KEYFRAMES_LEFT):
        return "arm keyframes differ"
    if rule.get("protocol") == "lift_place":
        frozen = _jsonable(place_params_dict(FlushPadPlaceParams()))
        if payload.get("place_params") != frozen or any(e.get("place_params") != frozen
                                                        for e in payload.get("episodes", [])):
            return "place params differ from the frozen FlushPadPlaceParams (reverse-keyframe lowering)"
    if int(payload.get("crew", -1)) != int(crew):
        return f"crew {payload.get('crew')}, expected {crew}"
    if not payload.get("scored_run"):
        return "not a scored run"
    if not flushpad_model_ok(payload.get("model_check") or {}, int(crew)):
        return "model check does not show the flush-pad contact and harness options"
    if not flushpad_design_matches_model(payload.get("flush_pad_design") or {}, payload.get("model_check") or {}):
        return "compiled hand pads differ from the flush-pad design"
    if (payload.get("wrist_hold_design") or {}).get("chosen") != list(WristHoldParams().wrist_joints):
        return "the wrist-hold design record does not name the frozen wrist (elbow roll)"
    if "policy_dt" not in payload:
        return "no policy_dt"
    return None


def wristhold_pair_summary(ep: dict, k: int) -> list:
    """The hold's per-hand summaries of pair `k` in one stored episode (diagnostics, never gated)."""
    rec = ep.get("wrist_hold") or {}
    return [{"robot": a.get("robot"), "side": a.get("side"), **(a.get("summary") or {})}
            for a in rec.get("arms") or [] if a.get("pair") == k]


def _wristhold_rows(payload: dict, rule: dict, k: int, servo: bool = True) -> tuple:
    """`_flushpad_rows` (the flush-pad clauses, unchanged) plus the hold's record: a seed whose episode has no
    well-formed wrist-hold record is missing too."""
    rows, missing = _flushpad_rows(payload, rule, k)
    dt = float(payload["policy_dt"])
    eps = {}
    for e in payload.get("episodes", []):
        eps.setdefault(int(e["seed"]), e)
    keep = []
    for r in rows:
        ep = eps[int(r["seed"])]
        bad = _wristhold_record_problem(ep, servo, dt)
        if bad:
            missing.append(int(r["seed"]))
            continue
        r["wrist_hold_diagnostic"] = wristhold_pair_summary(ep, k)
        keep.append(r)
    return keep, sorted(missing)


def wristhold_probe_verdict(payload: dict | None) -> dict:
    """Stage-1 verdict from the probe score JSON (pure): PROCEED | NEGATIVE | INCOMPLETE | INVALID."""
    rule = WRISTHOLD_PROBE_RULE
    out = {**_wristhold_head("probe"), "rule_text": WRISTHOLD_PROBE_RULE_TEXT, "rule": rule}
    if payload is None:
        return {**out, "verdict": "INCOMPLETE", "reason": "no probe score JSON", "successes": 0,
                "of": len(rule["seeds"]), "missing_seeds": list(rule["seeds"]), "per_seed": [],
                "reading": "no probe score JSON: no decision"}
    bad = _wristhold_payload_problem(payload, "probe", rule, 2)
    if bad:
        return {**out, "verdict": "INVALID", "reason": bad, "successes": 0, "of": len(rule["seeds"]),
                "missing_seeds": [], "per_seed": [], "reading": f"INVALID ({bad}): no decision"}
    rows, missing = _wristhold_rows(payload, rule, 0)
    n_ok = sum(1 for r in rows if r["success"])
    verdict = "INCOMPLETE" if missing else ("PROCEED" if n_ok >= rule["pass_min"] else "NEGATIVE")
    reading = {"PROCEED": ("the flush pad with the wrist-orientation hold kept the cube within 0.35 rad on >= 4/5 probe "
                           "seeds: run the scored stage"),
               "NEGATIVE": ("STOP: NEGATIVE, the flush pad does not stop the roll (here: the flush pad with the "
                            "wrist-orientation hold)"),
               "INCOMPLETE": ("an episode is missing, short, or has no tilt log, auto-reset record or wrist-hold "
                              "record: no decision")}[verdict]
    return {**out, "verdict": verdict, "successes": n_ok, "of": len(rule["seeds"]), "missing_seeds": missing,
            "per_seed": rows, "reading": reading,
            "diagnostic_not_gated": {
                "lift_hold_rule_successes": sum(1 for r in rows if r["base_rule_success"]),
                "lifted_ge_0.10_m": sum(1 for r in rows if r["lift_peak_m"] >= LIFT_HOLD_RULE["lift_peak_m"]),
                "note": "LIFT_HOLD_RULE's lift/hold/floor clauses and the hold's measurements are not part of the "
                        "probe rule"}}


def wristhold_group_summary(payload: dict, stage: str) -> dict:
    """Per-pair counts of one scored score JSON (hold or place, one crew), from the JSON alone."""
    rule = WRISTHOLD_HOLD_RULE if stage == "hold" else WRISTHOLD_PLACE_RULE
    crew = int(payload.get("crew", -1)) if isinstance(payload, dict) else -1
    bad = _wristhold_payload_problem(payload, stage, rule, crew) if crew in rule["crews"] else f"crew {crew}"
    if bad:
        return {"stage": stage, "crew": crew, "invalid": bad, "complete": False, "groups": []}
    groups, complete = [], True
    for k in range(crew // 2):
        rows, missing = _wristhold_rows(payload, rule, k)
        n_ok = sum(1 for r in rows if r["success"])
        complete = complete and not missing
        groups.append({"stage": stage, "crew": crew, "pair": k, "episodes": len(rows), "successes": n_ok,
                       "missing_seeds": missing, "pass": bool(not missing and n_ok >= rule["pass_min"]),
                       "base_rule_successes": sum(1 for r in rows if r["base_rule_success"]),
                       "first_failed_check_counts": {
                           c: sum(1 for r in rows if r["first_failed_check"] == c)
                           for c in sorted({r["first_failed_check"] for r in rows} - {None})},
                       "per_seed": rows})
    return {"stage": stage, "crew": crew, "invalid": None, "complete": complete, "groups": groups}


WRISTHOLD_SCORED_KEYS = ("hold_crew2", "hold_crew4", "place_crew2", "place_crew4")


def wristhold_scored_verdict(payloads: dict) -> dict:
    """Stage-2 verdict (pure): `payloads` maps "hold_crew2", "hold_crew4", "place_crew2", "place_crew4" to a score
    JSON or None. PASS iff every crew/pair of both protocols has >= 8/10; INCOMPLETE if any episode is missing; INVALID
    if a JSON is not under the frozen rule."""
    out = {**_wristhold_head("scored"), "rule_text": WRISTHOLD_SCORED_RULE_TEXT,
           "rules": {"hold": WRISTHOLD_HOLD_RULE, "place": WRISTHOLD_PLACE_RULE}}
    groups, invalid, complete = [], [], True
    for stage, rule in (("hold", WRISTHOLD_HOLD_RULE), ("place", WRISTHOLD_PLACE_RULE)):
        for crew in rule["crews"]:
            key = f"{stage}_crew{crew}"
            pl = payloads.get(key)
            if pl is None:
                complete = False
                groups.append({"stage": stage, "crew": crew, "pair": None, "missing_json": True, "pass": False})
                continue
            s = wristhold_group_summary(pl, stage)
            if s["invalid"] or int(s["crew"]) != crew:
                invalid.append(f"{key}: {s['invalid'] or 'crew mismatch'}")
                continue
            complete = complete and s["complete"]
            groups += s["groups"]
    if invalid:
        verdict = "INVALID"
    elif not complete:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS" if len(groups) == 6 and all(g["pass"] for g in groups) else "NEGATIVE"
    reading = {"PASS": "every crew/pair met its unchanged rule plus the 0.35 rad tilt clause on >= 8/10 seeds",
               "NEGATIVE": "at least one crew/pair fell short of 8/10 under its unchanged rule plus the tilt clause",
               "INCOMPLETE": "a score JSON or an episode is missing: no decision",
               "INVALID": "a score JSON is not under the frozen rule: no decision"}[verdict]
    return {**out, "verdict": verdict, "invalid": invalid, "reading": reading,
            "groups": [{k: v for k, v in g.items() if k != "per_seed"} for g in groups],
            "per_seed": {f"{g['stage']}_crew{g['crew']}_pair{g['pair']}": g.get("per_seed", []) for g in groups}}


def wristhold_not_run_verdict(probe_verdict: dict | None) -> dict:
    """Stage-2 verdict when the probe did not PROCEED: NOT_RUN, and seeds 20-39 stay unused."""
    pv = (probe_verdict or {}).get("verdict", "MISSING")
    return {**_wristhold_head("scored"), "rule_text": WRISTHOLD_SCORED_RULE_TEXT, "verdict": "NOT_RUN",
            "probe_verdict": pv, "groups": [],
            "reading": (f"stage 2 not run: the probe verdict is {pv}, not PROCEED (the frozen rule runs the scored "
                        "stage only after PROCEED); seeds 20-39 stay unused")}


def wristhold_smoke_check(payload: dict | None) -> dict:
    """Pipeline checks of one wrist-hold SMOKE score JSON (stage "smoke", or "smoke_nohold" for the no-hold baseline;
    pure; never a verdict on the variant): the variant built as designed, each episode ran the full 20 s on a throw-away
    seed with no MuJoCo auto-reset, the cube tilt is logged at every policy-step state, finite, flat at reset and in
    agreement with the harness's own trace, and the hold's record is well formed: twist logged at every state and 0 at
    the grasp state, and (hold) every wrist target within the joint range and every applied wrist torque within the
    cap. Outcomes (twist, tilt, lift, falls) are diagnostics, never gated."""
    problems, rows = [], []
    if not isinstance(payload, dict):
        return {**_wristhold_head("smoke"), "verdict": "SMOKE_FAIL", "problems": ["no smoke score JSON"],
                "episodes": []}
    stage = payload.get("stage")
    nohold = stage == "smoke_nohold"
    crew = int(payload.get("crew", -1))
    protocol = payload.get("protocol")
    if payload.get("variant") != WRISTHOLD_VARIANT or stage not in ("smoke", "smoke_nohold"):
        problems.append("not a wrist-hold smoke score JSON")
    if payload.get("scored_run"):
        problems.append("a smoke JSON must not be marked scored_run")
    if crew not in (2, 4) or protocol not in ("lift_hold", "lift_place"):
        problems.append(f"crew {crew} / protocol {protocol!r}")
    if nohold and (crew != 2 or protocol != "lift_hold"):
        problems.append("the no-hold baseline is crew 2, lift_hold")
    frozen = FlushPadParams() if nohold else WristHoldParams()
    if payload.get("params") != _jsonable(params_dict(frozen)):
        problems.append(f"params differ from the frozen {type(frozen).__name__}")
    if protocol == "lift_place" and payload.get("place_params") != _jsonable(place_params_dict(FlushPadPlaceParams())):
        problems.append("place params differ from the frozen FlushPadPlaceParams")
    mc = payload.get("model_check") or {}
    if not flushpad_model_ok(mc, crew):
        problems.append("model check does not show the flush-pad contact and harness options")
    if not flushpad_design_matches_model(payload.get("flush_pad_design") or {}, mc):
        problems.append("compiled hand pads differ from the flush-pad design")
    if (payload.get("wrist_hold_design") or {}).get("chosen") != list(WristHoldParams().wrist_joints):
        problems.append("the wrist-hold design record does not name the frozen wrist (elbow roll)")
    eps = payload.get("episodes") or []
    seeds = [int(e["seed"]) for e in eps]
    if not eps:
        problems.append("no episode")
    if any(s not in WRISTHOLD_SMOKE_SEEDS for s in seeds):
        problems.append("a seed outside the throw-away seeds 140-144")
    rule = payload.get("rule") or {}
    if (stage in ("smoke", "smoke_nohold") and protocol in ("lift_hold", "lift_place")
            and rule != _jsonable(wristhold_stage_rule(stage, protocol, seeds))):
        problems.append("rule differs from the smoke rule (full-length smoke episodes only)")
    dt = float(payload.get("policy_dt") or 0.0)
    n_full = int(round(20.0 / dt)) if dt > 0 else -1
    for e in eps:
        seed, steps = int(e["seed"]), int(e.get("steps", -1))
        if e.get("failed") is not None:
            problems.append(f"seed {seed}: episode failed ({e['failed']})")
        if steps != n_full:
            problems.append(f"seed {seed}: {steps} steps, expected {n_full}")
        reset_logged, reset = _sim_reset_logged(e)
        if not reset_logged:
            problems.append(f"seed {seed}: no MuJoCo auto-reset record (sim_reset)")
        elif reset:
            problems.append(f"seed {seed}: MuJoCo auto-reset the simulation (numerical blow-up) in policy step "
                            f"{(e.get('sim_reset') or {}).get('first_step')}")
        bad = _wristhold_record_problem(e, not nohold, dt) if dt > 0 else "no policy_dt"
        if bad:
            problems.append(f"seed {seed}: {bad}")
        rec = e.get("wrist_hold") or {}
        if (rec.get("ik") or {}).get("nonfinite_fallbacks"):
            problems.append(f"seed {seed}: the hold's IK produced a non-finite target")
        for arm in rec.get("arms") or []:
            lg = arm.get("servo_log") or {}
            for tgt in lg.get("target") or []:
                rng = (arm.get("summary") or {}).get("wrist_range") or []
                if any(v is None or not (r[0] - 1e-12 <= v <= r[1] + 1e-12) for v, r in zip(tgt, rng)):
                    problems.append(f"seed {seed} robot {arm.get('robot')}: a wrist target outside the joint range")
                    break
            caps = (arm.get("summary") or {}).get("cap_nm") or []
            for c in lg.get("ctrl_after_nm") or []:
                if any(v is None or abs(v) > cap + 1e-9 for v, cap in zip(c, caps)):
                    problems.append(f"seed {seed} robot {arm.get('robot')}: an applied wrist torque above the cap")
                    break
        if len(e.get("pairs", [])) != max(crew // 2, 0):
            problems.append(f"seed {seed}: {len(e.get('pairs', []))} pairs for crew {crew}")
        for k, row in enumerate(e.get("pairs", [])):
            ser = row.get("cube_tilt_series_rad")
            if not isinstance(ser, list) or len(ser) != steps + 1:
                problems.append(f"seed {seed} pair {k}: tilt series missing or not steps + 1 long")
                continue
            if any(v is None or not np.isfinite(v) for v in ser):
                problems.append(f"seed {seed} pair {k}: non-finite tilt in the series")
                continue
            if abs(float(ser[0])) > 1e-9:
                problems.append(f"seed {seed} pair {k}: tilt at reset {ser[0]} (the cube spawns flat)")
            matched = 0
            for tr in e.get("trace") or []:
                i = int(round(float(tr["t"]) / dt)) if dt > 0 else -1
                if not (0 <= i < steps):
                    continue
                tv = tr["cube_tilt_rad"][k]
                if tv is None or abs(float(ser[i + 1]) - float(tv)) > 0.0005 + 1e-9:
                    problems.append(f"seed {seed} pair {k}: tilt series disagrees with the trace at t={tr['t']}")
                    break
                matched += 1
            if matched == 0:
                problems.append(f"seed {seed} pair {k}: no trace entry to cross-check the tilt series")
            try:
                c = flushpad_clauses(rule, e, k, dt)
                ch = flushpad_clauses(WRISTHOLD_HOLD_RULE, e, k, dt) if protocol == "lift_hold" else None
            except Exception as exc:  # noqa: BLE001
                problems.append(f"seed {seed} pair {k}: clauses do not compute ({exc!r})")
                continue
            if not c["tilt_logged"] or (ch is not None and not ch["tilt_logged"]):
                problems.append(f"seed {seed} pair {k}: clauses report no tilt log")
            rows.append({"seed": seed, "pair": k, "trace_points_checked": matched, "sim_reset_detected": reset,
                         "diagnostic_cube_tilt_max_rad": max(ser), "diagnostic_clauses": c,
                         "diagnostic_hold_stage_clauses": ch,
                         "diagnostic_base_rule_success": bool(row.get("success")),
                         "diagnostic_base_first_failed_check": row.get("first_failed_check"),
                         "diagnostic_lift_peak_m": row.get("lift_peak_m"),
                         "diagnostic_max_robot_tilt_rad": row.get("max_tilt_rad"),
                         "diagnostic_wrist_hold": wristhold_pair_summary(e, k)})
    return {**_wristhold_head(stage if stage in ("smoke", "smoke_nohold") else "smoke"),
            "verdict": "SMOKE_FAIL" if problems else "SMOKE_PASS", "crew": crew, "protocol": protocol,
            "seeds": seeds, "problems": problems, "episodes": rows,
            "note": "pipeline check on a throw-away seed; outcomes are diagnostics, never a verdict"}


#: The smoke launcher's steps, whose exit statuses it records in step_status.json (unit tests first).
WRISTHOLD_SMOKE_STEPS = ("pytest", "smoke_hold_crew2", "smoke_place_crew4", "smoke_nohold_crew2")


def wristhold_smoke_step_problems(status) -> list:
    """Problems with the smoke's recorded step exit statuses (pure): every declared step, the unit tests first,
    recorded as integer 0, and nothing recorded non-zero."""
    if not isinstance(status, dict):
        return ["no step-status record (step_status.json): the unit-test result is unknown"]
    problems = []
    for name in WRISTHOLD_SMOKE_STEPS:
        if name not in status:
            problems.append(f"step {name}: no recorded exit status")
    for name, v in status.items():
        if isinstance(v, bool) or not isinstance(v, int) or v != 0:
            problems.append(f"step {name}: exit status {v!r}, not 0")
    return problems


def wristhold_smoke_measurement(hold: dict | None, nohold: dict | None, place: dict | None = None) -> dict:
    """The smoke's measurement (pure; diagnostics, never a verdict): how much of the hand's rotation about the pinch
    axis the hold removes (the crew-2 lift_hold episode with the hold against the no-hold baseline on the same seed),
    whether the wrist stays within its range and under the 4 Nm cap, and the cube tilt with and without the hold."""
    def first(pl):
        eps = (pl or {}).get("episodes") or []
        return eps[0] if eps else None

    def hands(ep):
        return {a.get("side"): a for a in ((ep or {}).get("wrist_hold") or {}).get("arms") or []}

    eh, en = first(hold), first(nohold)
    out = {"seed_hold": None if eh is None else eh.get("seed"), "seed_nohold": None if en is None else en.get("seed"),
           "same_seed": eh is not None and en is not None and eh.get("seed") == en.get("seed"), "hands": {},
           "note": ("twist = the hand's rotation about the pinch axis relative to the grasp state; removed = 1 - "
                    "|with hold| / |no hold|; diagnostics on a throw-away seed, never a verdict")}
    hh, hn = hands(eh), hands(en)
    for side in ("left", "right"):
        a, b = (hh.get(side) or {}).get("summary") or {}, (hn.get(side) or {}).get("summary") or {}

        def removed(key):
            x, y = a.get(key), b.get(key)
            if x is None or y is None or abs(y) < 1e-12:
                return None
            return round(1.0 - abs(x) / abs(y), 4)
        out["hands"][side] = {
            "twist_at_lift_end_rad": {"hold": a.get("twist_at_lift_end_rad"), "no_hold": b.get("twist_at_lift_end_rad")},
            "twist_max_abs_rad": {"hold": a.get("twist_max_abs_rad"), "no_hold": b.get("twist_max_abs_rad")},
            "twist_final_rad": {"hold": a.get("twist_final_rad"), "no_hold": b.get("twist_final_rad")},
            "swing_max_rad": {"hold": a.get("swing_max_rad"), "no_hold": b.get("swing_max_rad")},
            "removed_fraction_at_lift_end": removed("twist_at_lift_end_rad"),
            "removed_fraction_of_max": removed("twist_max_abs_rad"),
            "wrist": {k: a.get(k) for k in ("wrist_q_min", "wrist_q_max", "wrist_range",
                                            "steps_within_0.01_rad_of_a_limit", "beyond_range_max_rad",
                                            "target_clamped_steps", "first_clamped_t_s", "held_steps")},
            "cap": {k: a.get(k) for k in ("pd_demand_max_abs_nm", "pd_demand_above_cap_steps", "first_above_cap_t_s",
                                          "cap_nm", "ctrl_after_max_abs_nm")}}

    def cube(ep):
        if ep is None:
            return None
        r = ep["pairs"][0]
        ser = [v for v in r.get("cube_tilt_series_rad") or [] if v is not None]
        return {"cube_tilt_max_rad": max(ser) if ser else None, "lift_peak_m": r.get("lift_peak_m"),
                "lift_hold_s": r.get("lift_hold_s"), "base_rule_success": r.get("success"),
                "base_first_failed_check": r.get("first_failed_check"), "max_robot_tilt_rad": r.get("max_tilt_rad"),
                "cube_floor_contact": r.get("cube_floor_contact")}
    out["cube"] = {"hold": cube(eh), "no_hold": cube(en)}
    ep_place = first(place)
    if ep_place is not None:
        out["place_crew4"] = {
            "seed": ep_place.get("seed"),
            "hands": [{"robot": a.get("robot"), "pair": a.get("pair"), "side": a.get("side"),
                       **{k: (a.get("summary") or {}).get(k) for k in (
                           "twist_at_lift_end_rad", "twist_max_abs_rad", "swing_max_rad", "target_clamped_steps",
                           "steps_within_0.01_rad_of_a_limit", "pd_demand_above_cap_steps", "pd_demand_max_abs_nm",
                           "ctrl_after_max_abs_nm")}}
                      for a in (ep_place.get("wrist_hold") or {}).get("arms") or []],
            "pairs": [{"pair": r.get("pair"), "base_rule_success": r.get("success"),
                       "base_first_failed_check": r.get("first_failed_check"), "lift_peak_m": r.get("lift_peak_m"),
                       "cube_tilt_max_rad": max([v for v in r.get("cube_tilt_series_rad") or [] if v is not None],
                                                default=None)} for r in ep_place.get("pairs", [])]}
    return out


def wristhold_file_summary(payload: dict) -> dict:
    """Summary written into a wrist-hold score JSON after every episode (the verdict JSONs are written by the launcher
    from these files). Smoke runs report the pipeline checks and never a verdict on the variant."""
    stage = payload.get("stage")
    if stage == "probe":
        return wristhold_probe_verdict(payload)
    if stage in ("hold", "place"):
        return wristhold_group_summary(payload, stage)
    return wristhold_smoke_check(payload)


def wristhold_verdict_line(v: dict) -> str:
    """One line for the log, from a verdict / summary dict."""
    stage = v.get("stage")
    if stage == "probe":
        body = (f"{v.get('successes')}/{v.get('of')} seeds kept cube tilt <= {WRISTHOLD_TILT_MAX_RAD} rad with no fall "
                f"(need >= {WRISTHOLD_PROBE_RULE['pass_min']}); missing {v.get('missing_seeds')}; "
                + " ".join(f"s{r['seed']}:max_tilt={r.get('cube_tilt_max_rad')},fail={r['first_failed_check']}"
                           for r in v.get("per_seed", [])))
        if v.get("reason"):
            body += f" ({v['reason']})"
    elif stage == "scored":
        body = " ".join(f"{g['stage']}/crew{g['crew']}/pair{g['pair']}="
                        + ("missing" if g.get("missing_json") else f"{g['successes']}/{g['episodes']}")
                        for g in v.get("groups", [])) or v.get("reading", "")
        if v.get("invalid"):
            body += f" invalid: {v['invalid']}"
    elif stage in ("hold", "place"):
        body = " ".join(f"crew{g['crew']}/pair{g['pair']}={g['successes']}/{g['episodes']}"
                        for g in v.get("groups", [])) or f"invalid: {v.get('invalid')}"
    else:
        body = (" ".join(f"seed{r['seed']}/pair{r['pair']}:max_tilt={r['diagnostic_cube_tilt_max_rad']:.3f},"
                         f"base={r['diagnostic_base_first_failed_check'] or 'ok'},twist_max="
                         + "/".join(f"{h.get('side')}:{h.get('twist_max_abs_rad')}"
                                    for h in r.get("diagnostic_wrist_hold") or [])
                         for r in v.get("episodes", []))
                + (f" problems: {v['problems']}" if v.get("problems") else ""))
    verdict = v.get("verdict") or ("INVALID" if v.get("invalid") else
                                   "COMPLETE" if v.get("complete") else "INCOMPLETE")
    return f"COOP-WRISTHOLD {stage}: {verdict} | {body} | {WRISTHOLD_HOLD_NOTE} | {WRISTHOLD_LABEL}"

# --- end coop-wristhold ---
