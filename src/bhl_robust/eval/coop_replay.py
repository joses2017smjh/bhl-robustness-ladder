"""Replay a trained cooperative-lift policy in MuJoCo.

§5 of the README reports thirty-odd coop runs entirely from TensorBoard scalars.
That is the one section of this project with no sim2sim column and no clip of
the thing it describes, and the reason is mechanical rather than principled:
`play.py` exports locomotion policies, the coop task has two robots and 44
actions, and nothing ever wrote a `deploy.yaml` for it. The pinch numbers in §5
are therefore PhysX-side training statistics, which is precisely the kind of
number §1 exists to distrust.

This closes that. It is the same argument as everywhere else in the repo --
train in PhysX, score in MuJoCo, and treat the disagreement as the measurement
-- applied to the lift instead of the gait.

Three things it deliberately does *not* do:

* It does not go through ONNX. `rsl_rl`'s exporter wants a locomotion-shaped
  actor and an `obs_normalizer` that this runner config does not have
  (`empirical_normalization: false`). The actor is four `Linear` layers with ELU
  between them, so the weights are lifted straight out of `model_state_dict`
  and evaluated in numpy. Fewer moving parts than a conversion, and the layer
  shapes are asserted against the observation width rather than trusted.
* It does not rebuild the observation from a specification. The order below is
  read off the `params/env.yaml` that the run itself dumped, term by term, and
  the assembled width is asserted to equal the actor's input width. An
  observation vector that is subtly permuted still runs and still produces
  motion, which is the failure mode worth engineering against.
* It does not add observation noise. Training corrupts with `Unoise`; scoring a
  deployed policy on the clean signal is the same choice the locomotion harness
  makes.

Crew composition is the other half. The policy controls exactly two robots, so
a "crew" is pairs, and the odd robot out is a control rather than a decoration:
its partner slots are fed its own state, i.e. the policy is asked to do the job
believing it has a mirror-image partner doing the same thing. If a single robot
still gets the crate off the floor, the word "cooperative" was not earning its
place in the section title.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from bhl_robust.eval.livery import PAYLOAD_RGBA, apply_livery
from bhl_robust.eval.mjcf_assets import EGO_CAM_NAME, prepare_mjcf

# Joint order the observation and action vectors were built in: arms first, then
# legs. `HUMANOID_LITE_JOINTS = ARM_JOINTS + LEG_JOINTS`, and every coop
# observation term passes `preserve_order=True`, so this is the order and not
# merely one valid order. Duplicated here rather than imported because the asset
# module pulls in `isaaclab.sim`, which needs Isaac Sim booted.
ARM_JOINTS = [
    "arm_left_shoulder_pitch_joint", "arm_left_shoulder_roll_joint",
    "arm_left_shoulder_yaw_joint", "arm_left_elbow_pitch_joint",
    "arm_left_elbow_roll_joint",
    "arm_right_shoulder_pitch_joint", "arm_right_shoulder_roll_joint",
    "arm_right_shoulder_yaw_joint", "arm_right_elbow_pitch_joint",
    "arm_right_elbow_roll_joint",
]
LEG_JOINTS = [
    "leg_left_hip_roll_joint", "leg_left_hip_yaw_joint",
    "leg_left_hip_pitch_joint", "leg_left_knee_pitch_joint",
    "leg_left_ankle_pitch_joint", "leg_left_ankle_roll_joint",
    "leg_right_hip_roll_joint", "leg_right_hip_yaw_joint",
    "leg_right_hip_pitch_joint", "leg_right_knee_pitch_joint",
    "leg_right_ankle_pitch_joint", "leg_right_ankle_roll_joint",
]
JOINTS = ARM_JOINTS + LEG_JOINTS
NJ = len(JOINTS)

# `ImplicitActuatorCfg` gains, straight from the asset config. The arms are a
# weaker group than the legs (4 Nm, kp 10) and the lift lives entirely in them.
_GAINS = {"arm": (10.0, 2.0, 4.0), "leg": (20.0, 2.0, 6.0)}

# The crouch-hold spawn. `_PINCH_JOINT_POS` in `coop_lift_env_cfg`, and the same
# pose the scripted kinematics clip interpolates through.
PINCH_POSE = {
    "arm_left_shoulder_pitch_joint": -0.55, "arm_left_shoulder_roll_joint": -0.26,
    "arm_left_elbow_pitch_joint": 0.90,
    "arm_right_shoulder_pitch_joint": 0.55, "arm_right_shoulder_roll_joint": 0.26,
    "arm_right_elbow_pitch_joint": -0.90,
    "leg_left_hip_pitch_joint": -0.85, "leg_left_knee_pitch_joint": 1.45,
    "leg_left_ankle_pitch_joint": -0.55,
    "leg_right_hip_pitch_joint": -0.85, "leg_right_knee_pitch_joint": 1.45,
    "leg_right_ankle_pitch_joint": -0.55,
}

ACTION_SCALE = 0.25
POLICY_DT = 0.04          # decimation 8 x sim dt 0.005
PHYSICS_DT = 0.005
EPISODE_STEPS = 200       # episode_length_s 8.0
TILT_LIMIT = 0.78

# Cube task: 0.28 m box, side pinch across y, contact points +/-0.16 m from the
# centre, spawned with its centre 0.14 m up (resting on the floor).
CUBE_HALF = 0.14
CUBE_MASS = 0.5
CONTACT_AXIS = np.array([0.0, 1.0, 0.0])
CONTACT_OFFSET = 0.16
SPAWN_Z = 0.14

# Half the pair's y separation, from `CoopLiftSceneCfg`: robots at y = +/-0.48.
PAIR_HALF_Y = 0.48
# Clear air between one pair and the next, so a four-robot shot does not read as
# four robots on one crate.
PAIR_GAP = 0.95

_WORLD = """<mujoco model="bhl-carry">
  <compiler angle="radian"/>
  <statistic center="0 0 0.35" extent="3.0"/>
  <visual>
    <headlight diffuse="0.62 0.62 0.62" ambient="0.34 0.34 0.34" specular="0.05 0.05 0.05"/>
    <rgba haze="0.17 0.19 0.23 1"/>
    <global azimuth="120" elevation="-14" offwidth="1920" offheight="1080"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.30 0.33 0.38" rgb2="0.07 0.08 0.10"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.30 0.31 0.33" rgb2="0.20 0.21 0.23" markrgb="0.55 0.55 0.55"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="6 6" reflectance="0.12"/>
  </asset>
  <worldbody>
    <light pos="0 0 3.2" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"
          friction="1.0 0.005 0.0001"/>
  </worldbody>
</mujoco>
"""


# Egocentric depth, matched to `tasks/coop_depth_env_cfg`: a 64x64 ray-cast
# image average-pooled to 8x8 and divided by the 6 m range.
DEPTH_RES = 64
DEPTH_POOL = 8
DEPTH_N = (DEPTH_RES // DEPTH_POOL) ** 2
DEPTH_RANGE = 6.0
# Depth is rendered against the floor and the payload only, which is the target
# list the Isaac sensor casts against. Upstream puts robot visual meshes in geom
# group 2 and collision primitives in group 3; the world floor and the crates
# added here are group 0. Masking to groups 0 and 1 therefore reproduces the
# sensor's blindness to the robots exactly, rather than approximately.
DEPTH_GEOM_GROUPS = (0, 1)

# Observation widths this replay knows how to assemble, and what each one is.
# Reading the width off the checkpoint and assembling to match is safer than a
# flag: a flag can be wrong, a weight matrix cannot.
#
# Term order is Isaac Lab's field-declaration order, and `@configclass` is a
# dataclass, so an inherited field precedes a field the subclass adds. That is
# why depth lands *after* `last_action` even though it is an observation and
# `last_action` is a proprioceptive echo -- `actions` is declared on the base
# `PolicyCfg` and `depth_a` / `depth_b` on the override.
OBS_FULL = 194
OBS_NOTRACK = 150
OBS_DEPTH_SWAP = 316      # full, minus object-in-root, plus two depth images
OBS_DEPTH_BOTH = 322      # full, plus two depth images
_OBS_LAYOUTS = {
    OBS_FULL: "projected gravity, base ang vel, joint pos, joint vel, "
              "object-in-root, PD tracking residual, previous action",
    OBS_NOTRACK: "as OBS_FULL without the PD tracking residual",
    OBS_DEPTH_SWAP: "as OBS_FULL without object-in-root, then two 8x8 depth "
                    "images appended after previous action",
    OBS_DEPTH_BOTH: "as OBS_FULL, then two 8x8 depth images appended after "
                    "previous action",
}
_DEPTH_LAYOUTS = (OBS_DEPTH_SWAP, OBS_DEPTH_BOTH)
N_ACT = 2 * NJ


class CoopActor:
    """The trained actor, evaluated in numpy.

    `RslRlPpoActorCriticCfg(actor_hidden_dims=[256, 256, 128], activation="elu")`
    builds `Linear -> ELU` three times and a final `Linear`, which lands in the
    state dict as `actor.{0,2,4,6}`. `empirical_normalization` is off for this
    runner, so there is no normaliser to reproduce.

    The observation width is *read* from the first layer rather than asserted
    against a constant, and `n_obs` is then what the runner assembles. Passing
    `expect_obs` still pins it when a caller knows what it wants.
    """

    def __init__(self, checkpoint: Path, expect_obs: int | None = None,
                 expect_act: int = N_ACT):
        import torch

        sd = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        sd = sd["model_state_dict"]
        self.layers: list[tuple[np.ndarray, np.ndarray]] = []
        for i in (0, 2, 4, 6):
            w = sd[f"actor.{i}.weight"].numpy().astype(np.float64)
            b = sd[f"actor.{i}.bias"].numpy().astype(np.float64)
            self.layers.append((w, b))
        n_in = self.layers[0][0].shape[1]
        n_out = self.layers[-1][0].shape[0]
        if n_out != expect_act:
            raise RuntimeError(
                f"{checkpoint.name} emits {n_out} actions; this task drives "
                f"{expect_act}."
            )
        if n_in not in _OBS_LAYOUTS:
            raise RuntimeError(
                f"{checkpoint.name} takes {n_in} observations; this replay knows "
                f"how to assemble {sorted(_OBS_LAYOUTS)} only. A width outside "
                "that set is a different observation config, and guessing which "
                "terms were dropped would produce motion rather than an error."
            )
        if expect_obs is not None and n_in != expect_obs:
            raise RuntimeError(
                f"{checkpoint.name} takes {n_in} observations, caller asked for "
                f"{expect_obs}."
            )
        self.n_obs = n_in
        self.layout = _OBS_LAYOUTS[n_in]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64)
        for k, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if k < len(self.layers) - 1:
                x = np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))   # ELU
        return x


@dataclass
class RobotSlot:
    """Index bundle for one robot inside the shared model."""
    prefix: str
    body_id: int
    qpos_adr: int
    qvel_adr: int
    jnt_qpos: np.ndarray      # per JOINTS, qpos address
    jnt_qvel: np.ndarray      # per JOINTS, dof address
    ctrl: np.ndarray          # per JOINTS, actuator index
    hands: np.ndarray         # body ids of the two hand links
    target: np.ndarray = field(default_factory=lambda: np.zeros(NJ))


@dataclass
class Crate:
    """One payload, and the robots assigned to it."""
    body_id: int
    qpos_adr: int
    qvel_adr: int
    slot_a: int               # robot on the +axis side
    slot_b: int | None        # None for a solo attempt
    spawn: np.ndarray


def _yaw_quat(yaw: float) -> list[float]:
    return [float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0))]


def build_crew(upstream: Path, cache_dir: Path, n_robots: int,
               ego_camera: bool = False):
    """Compose `n_robots` humanoids and one crate per pair into one model.

    Robots are laid out along +y, so a single camera frames the crew, and every
    consecutive pair faces inward across its own crate. An odd robot gets a
    crate of its own with no partner.

    `ego_camera` mounts the base depth camera on every robot, which a
    depth-conditioned policy needs and a blind one does not. It is cheap to
    leave on -- a camera the renderer never points at costs nothing -- but the
    default is off so a blind replay is bit-identical to what it was before
    vision existed.

    Returns (model, slots, crates).
    """
    if n_robots < 2:
        raise ValueError("a crew is two or more robots")

    scene = prepare_mjcf(upstream, cache_dir, "humanoid", ego_camera=ego_camera)
    robot_xml = scene.parent / "berkeley_humanoid_lite.xml"

    world_path = cache_dir / "carry_world.xml"
    world_path.write_text(_WORLD)
    spec = mujoco.MjSpec.from_file(str(world_path))

    n_pairs = (n_robots + 1) // 2
    pitch = 2.0 * PAIR_HALF_Y + PAIR_GAP
    # Centre the crew on y = 0 so the camera does not have to be re-aimed per
    # crew size.
    y_first = -0.5 * (n_pairs - 1) * pitch

    ys, yaws, pair_of = [], [], []
    for p in range(n_pairs):
        centre = y_first + p * pitch
        # Slot b sits at -y facing +y; slot a sits at +y facing -y, matching
        # robot_b / robot_a in the trained scene.
        ys.append(centre - PAIR_HALF_Y); yaws.append(+np.pi / 2); pair_of.append(p)
        if len(ys) < n_robots:
            ys.append(centre + PAIR_HALF_Y); yaws.append(-np.pi / 2); pair_of.append(p)

    for i, (y, yaw) in enumerate(zip(ys, yaws)):
        child = mujoco.MjSpec.from_file(str(robot_xml))
        frame = spec.worldbody.add_frame()
        frame.pos = [0.0, y, 0.0]
        frame.quat = _yaw_quat(yaw)
        frame.attach_body(child.bodies[1], f"r{i}_", "")

    for p in range(n_pairs):
        centre = y_first + p * pitch
        body = spec.worldbody.add_body()
        body.name = f"crate{p}"
        body.pos = [0.0, centre, SPAWN_Z]
        body.add_freejoint(name=f"crate{p}_free")
        g = body.add_geom()
        g.name = f"crate{p}_g"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [CUBE_HALF, CUBE_HALF, CUBE_HALF]
        g.mass = CUBE_MASS
        # Isaac: static 1.4 / dynamic 1.2. MuJoCo has one sliding coefficient,
        # so take the dynamic one -- the lift is a sliding-contact problem and
        # the static value would flatter it.
        g.friction = [1.2, 0.005, 0.0001]
        g.rgba = PAYLOAD_RGBA

    model = spec.compile()

    def jid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    slots = []
    for i in range(n_robots):
        p = f"r{i}_"
        jq, jv, ct = [], [], []
        for jn in JOINTS:
            j = jid(p + jn)
            if j < 0:
                raise RuntimeError(f"joint {p + jn} missing; upstream layout changed")
            jq.append(int(model.jnt_qposadr[j]))
            jv.append(int(model.jnt_dofadr[j]))
            a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, p + jn)
            if a < 0:
                raise RuntimeError(f"actuator {p + jn} missing")
            ct.append(int(a))
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p + "base")
        fj = model.body_jntadr[bid]
        hands = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p + h)
                 for h in ("arm_left_hand_link", "arm_right_hand_link")]
        if min(hands) < 0:
            raise RuntimeError(f"hand links missing for {p}")
        slots.append(RobotSlot(
            prefix=p, body_id=int(bid),
            qpos_adr=int(model.jnt_qposadr[fj]), qvel_adr=int(model.jnt_dofadr[fj]),
            jnt_qpos=np.asarray(jq), jnt_qvel=np.asarray(jv),
            ctrl=np.asarray(ct), hands=np.asarray(hands),
        ))
        apply_livery(model, p)

    crates = []
    for p in range(n_pairs):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"crate{p}")
        j = jid(f"crate{p}_free")
        members = [i for i in range(n_robots) if pair_of[i] == p]
        # `pair_of` appends the -y robot first, so members[0] is slot b.
        slot_b = members[0]
        slot_a = members[1] if len(members) > 1 else members[0]
        crates.append(Crate(
            body_id=int(bid),
            qpos_adr=int(model.jnt_qposadr[j]), qvel_adr=int(model.jnt_dofadr[j]),
            slot_a=slot_a, slot_b=(slot_b if len(members) > 1 else None),
            spawn=np.array([0.0, y_first + p * pitch, SPAWN_Z]),
        ))
    return model, slots, crates


class CrewRunner:
    """Drives every pair in the shared model from one trained actor."""

    def __init__(self, model, slots, crates, actor: CoopActor):
        self.m, self.slots, self.crates, self.actor = model, slots, crates, actor
        self.d = mujoco.MjData(model)
        self.m.opt.timestep = PHYSICS_DT
        self.substeps = int(round(POLICY_DT / PHYSICS_DT))

        self.kp = np.array([_GAINS["arm" if j.startswith("arm") else "leg"][0] for j in JOINTS])
        self.kd = np.array([_GAINS["arm" if j.startswith("arm") else "leg"][1] for j in JOINTS])
        self.eff = np.array([_GAINS["arm" if j.startswith("arm") else "leg"][2] for j in JOINTS])
        self.qdefault = np.array([PINCH_POSE.get(j, 0.0) for j in JOINTS])
        self.n_obs = actor.n_obs
        self.prev_action = [np.zeros(2 * NJ) for _ in crates]
        self.fell = [False] * len(slots)

        # Depth is only rendered when the policy actually reads it. A renderer
        # costs an EGL context and two frames per robot per step, and a blind
        # policy would pay that for a vector it never sees.
        self.wants_depth = self.n_obs in _DEPTH_LAYOUTS
        self._dep = None
        self._dep_opt = None
        self._cams: list[int] = []
        if self.wants_depth:
            self._setup_depth()

    def _setup_depth(self) -> None:
        self._cams = []
        for s in self.slots:
            c = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA,
                                  s.prefix + EGO_CAM_NAME)
            if c < 0:
                raise RuntimeError(
                    f"policy reads depth but {s.prefix}{EGO_CAM_NAME} is absent; "
                    "build_crew needs ego_camera=True"
                )
            self._cams.append(int(c))
        self._dep = mujoco.Renderer(self.m, height=DEPTH_RES, width=DEPTH_RES)
        self._dep.enable_depth_rendering()
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        for g in DEPTH_GEOM_GROUPS:
            opt.geomgroup[g] = 1
        self._dep_opt = opt

    def depth_image(self, i: int) -> np.ndarray:
        """Robot `i`'s raw 64x64 depth image in metres, floor and payload only."""
        self._dep.update_scene(self.d, camera=self._cams[i],
                               scene_option=self._dep_opt)
        return np.asarray(self._dep.render(), dtype=np.float64)

    def depth_obs(self, i: int) -> np.ndarray:
        """The depth term as the policy consumes it: pooled, scaled, clamped.

        Same treatment as `tasks/coop_depth_env_cfg.coop_depth_obs`, restated in
        numpy. Non-finite pixels become the clip range, because that is what
        `depth_clipping_behavior="max"` does on the Isaac side -- a real sensor
        reports its maximum where it sees nothing, not a NaN.
        """
        d = self.depth_image(i)
        d = np.nan_to_num(d, nan=DEPTH_RANGE, posinf=DEPTH_RANGE,
                          neginf=DEPTH_RANGE)
        d = np.minimum(d, DEPTH_RANGE)
        k = DEPTH_POOL
        pooled = d.reshape(DEPTH_RES // k, k, DEPTH_RES // k, k).mean(axis=(1, 3))
        return np.clip(pooled.ravel() / DEPTH_RANGE, 0.0, 1.0)

    def close(self) -> None:
        if self._dep is not None:
            try:
                self._dep.close()
            except Exception:
                pass
            self._dep = None

    # ---------------------------------------------------------------- reset

    def reset(self, rng: np.random.Generator, jitter: bool = True) -> None:
        """Spawn the crouch-hold and settle the feet onto the floor.

        Isaac places the root at z = -0.07, a pelvis drop derived from the
        sagittal shortening of a 0.12 m thigh and 0.16 m shank at the crouch
        angles. Rather than trust that the two descriptions of the robot put
        their root frames in the same place, the feet are planted: pose the
        joints, measure the lowest collision geom, and translate the base so it
        rests on the plane. Any frame disagreement then shows up as a printed
        offset instead of a robot spawned inside the floor.
        """
        mujoco.mj_resetData(self.m, self.d)
        for s in self.slots:
            self.d.qpos[s.qpos_adr:s.qpos_adr + 3] = self.m.qpos0[s.qpos_adr:s.qpos_adr + 3]
            self.d.qpos[s.qpos_adr + 3:s.qpos_adr + 7] = \
                self.m.qpos0[s.qpos_adr + 3:s.qpos_adr + 7]
            q = self.qdefault.copy()
            if jitter:
                q = q + rng.uniform(-0.08, 0.08, NJ)
            self.d.qpos[s.jnt_qpos] = q
            s.target[:] = self.qdefault
        for c in self.crates:
            self.d.qpos[c.qpos_adr:c.qpos_adr + 3] = c.spawn
            self.d.qpos[c.qpos_adr + 3:c.qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.m, self.d)

        for s in self.slots:
            lo = self._lowest_z(s)
            self.d.qpos[s.qpos_adr + 2] -= lo
        mujoco.mj_forward(self.m, self.d)
        self.prev_action = [np.zeros(2 * NJ) for _ in self.crates]
        self.fell = [False] * len(self.slots)

    def _lowest_z(self, s: RobotSlot) -> float:
        lo = np.inf
        for g in range(self.m.ngeom):
            if int(self.m.geom_contype[g]) == 0:
                continue
            b = int(self.m.geom_bodyid[g])
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if not name.startswith(s.prefix):
                continue
            # Conservative: the geom's lowest point is its centre minus the
            # largest of its half-extents, which never spawns a foot below the
            # plane even for a rotated capsule.
            lo = min(lo, float(self.d.geom_xpos[g][2] - self.m.geom_size[g].max()))
        return 0.0 if not np.isfinite(lo) else lo

    # ------------------------------------------------------------ observe

    def _robot_obs(self, s: RobotSlot, crate: Crate):
        """(projected_gravity, base_ang_vel, joint_pos_rel, joint_vel, object_pos_in_root, track_err)."""
        R = self.d.xmat[s.body_id].reshape(3, 3)
        proj_g = R.T @ np.array([0.0, 0.0, -1.0])
        # A MuJoCo free joint stores its 3 rotational DOFs in the body frame,
        # which is already what `mdp.base_ang_vel` reports.
        ang = self.d.qvel[s.qvel_adr + 3:s.qvel_adr + 6].copy()
        q = self.d.qpos[s.jnt_qpos]
        qd = self.d.qvel[s.jnt_qvel]
        obj = self.d.xpos[crate.body_id]
        obj_in_root = R.T @ (obj - self.d.xpos[s.body_id])
        return proj_g, ang, q - self.qdefault, qd.copy(), obj_in_root, q - s.target

    def observe(self, crate: Crate) -> np.ndarray:
        """Assemble the policy observation for one crate's pair.

        Term order is the `ObservationsCfg.PolicyCfg` field order, which is what
        `concatenate_terms` follows and what `params/env.yaml` records:
        projected gravity (a, b), base angular velocity (a, b), joint position
        (a, b), joint velocity (a, b), object-in-root (a, b), PD tracking
        residual (a, b), previous action. The residual pair is omitted for a
        `notrack` actor, which is the only difference between the two widths.

        For a solo attempt both halves are the same robot: the policy acts as
        though its partner were a mirror image doing exactly what it does.
        """
        i_a = crate.slot_a
        i_b = crate.slot_b if crate.slot_b is not None else crate.slot_a
        a = self._robot_obs(self.slots[i_a], crate)
        b = self._robot_obs(self.slots[i_b], crate)
        p = self.prev_action[self.crates.index(crate)]
        terms = [
            a[0], b[0],           # projected_gravity
            a[1], b[1],           # base_ang_vel
            a[2], b[2],           # joint_pos_rel
            a[3], b[3],           # joint_vel_rel
        ]
        if self.n_obs != OBS_DEPTH_SWAP:
            terms += [a[4], b[4]]  # object_pos_in_root
        if self.n_obs != OBS_NOTRACK:
            terms += [a[5], b[5]]  # track_err
        terms.append(p)            # last_action
        if self.wants_depth:
            terms += [self.depth_obs(i_a), self.depth_obs(i_b)]
        obs = np.concatenate(terms)
        if obs.size != self.n_obs:
            raise RuntimeError(
                f"assembled {obs.size} observations for a {self.n_obs}-wide "
                "actor; the term list and the checkpoint disagree."
            )
        return obs

    # --------------------------------------------------------------- step

    def step(self) -> None:
        for k, c in enumerate(self.crates):
            action = self.actor(self.observe(c))
            self.prev_action[k] = action
            targets = self.qdefault + ACTION_SCALE * action.reshape(2, NJ)
            self.slots[c.slot_a].target[:] = targets[0]
            if c.slot_b is not None:
                self.slots[c.slot_b].target[:] = targets[1]

        for _ in range(self.substeps):
            for s in self.slots:
                q = self.d.qpos[s.jnt_qpos]
                qd = self.d.qvel[s.jnt_qvel]
                tau = self.kp * (s.target - q) - self.kd * qd
                self.d.ctrl[s.ctrl] = np.clip(tau, -self.eff, self.eff)
            mujoco.mj_step(self.m, self.d)

        for i, s in enumerate(self.slots):
            if self.tilt(i) > TILT_LIMIT:
                self.fell[i] = True

    # ------------------------------------------------------------ measure

    def tilt(self, i: int) -> float:
        R = self.d.xmat[self.slots[i].body_id].reshape(3, 3)
        return float(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))

    def hand_mid(self, i: int) -> np.ndarray:
        return self.d.xpos[self.slots[i].hands].mean(axis=0)

    def pinch_distance(self, c: Crate) -> float:
        """The quantity §5's `pinch` column is a kernel of.

        Mean distance from each robot's hand midpoint to its own contact point
        on the crate. `reaching_fine` is `1 - tanh(d / 0.12)`, so this is the
        raw metres behind that number and is directly comparable across arms.
        """
        centre = self.d.xpos[c.body_id]
        c_a = centre + CONTACT_OFFSET * CONTACT_AXIS
        c_b = centre - CONTACT_OFFSET * CONTACT_AXIS
        d_a = np.linalg.norm(self.hand_mid(c.slot_a) - c_a)
        if c.slot_b is None:
            return float(d_a)
        d_b = np.linalg.norm(self.hand_mid(c.slot_b) - c_b)
        return float(0.5 * (d_a + d_b))

    def pinch_kernel(self, c: Crate, std: float = 0.12) -> float:
        return float(1.0 - np.tanh(self.pinch_distance(c) / std))

    def crate_lift(self, c: Crate) -> float:
        """Metres the crate has risen above its spawn height."""
        return float(self.d.xpos[c.body_id][2] - SPAWN_Z)
