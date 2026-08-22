"""Vision in the cooperative-lift loop, and the reason it is a different
problem from vision in the locomotion loop.

§6 put depth in the *locomotion* loop for 1.6% of throughput, using
`RayCasterCamera`: a pinhole ray bundle intersected against the meshes named in
`mesh_prim_paths`. That works for walking because the thing a walking policy
needs to see is the ground, and the ground is static. A warp mesh built once at
startup is a complete description of it.

A lift is not that. The thing the policy has to see is the object it is trying
to pinch, and that object moves — it is the only prim in the scene whose motion
*is* the task. `RayCasterCamera` cannot see it: its meshes are baked at startup
and never re-transformed, so a camera pointed at a cube that has been picked up
still reports the empty floor behind where the cube used to be. That is not a
tuning problem, it is the sensor's documented scope ("Currently, only static
meshes are supported", `ray_caster.py`).

So the naive reading is that manipulation-with-vision needs the RTX renderer,
which is the thing that segfaults here under Isaac Sim 5.1 — and which, measured
on Isaac Sim 6.0.1 in `scripts/bench/rtx_probe.py`, costs 13.7 ms per 64x64
camera. Two cameras per env at 1,024 envs is 2,048 cameras, i.e. ~28 s per
policy step against a physics budget of ~0.19 s. Not a tax; a different
experiment.

`MultiMeshRayCasterCamera` is the way out, and it is already in the pinned
Isaac Lab 2.3.2. It keeps a `RigidBodyView` or `ArticulationView` per tracked
target and re-transforms the ray bundle each update, so a moving cube is a
legitimate raycast target. Geometric depth of a dynamic scene, no Hydra engine,
no RTX plugins.

**What the comparison is.** Not "blind versus sighted" — the blind policy is not
blind. `object_pos_in_root` hands it the exact object position in each robot's
own frame, which is a privileged measurement no camera provides. So adding depth
on top of that would be adding a noisy copy of something the policy already
knows exactly, and would measure nothing. The arms below therefore are:

* `swap` — depth *replaces* `object_pos_in_root`. Can the pair form the pinch
  from an image instead of from a cheat? This is the deployable policy, and it
  is the same teacher/student split §3 draws between the privileged height scan
  and blind proprioception.
* `both` — depth *and* the pose. The control for `swap`: if `both` matches the
  §5 baseline and `swap` does not, the gap is the missing pose rather than the
  presence of depth.

Pooling is 8x8 rather than §6's 16x16. The coop observation is already 194 wide
against locomotion's 45, and two robots each contribute an image; at 16x16 the
depth would still be 512 of 700 inputs. 8x8 puts it at 128 of 316, which is the
same proportion §6's locomotion policy saw.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import MultiMeshRayCasterCameraCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

from bhl_robust.tasks.coop_lift_env_cfg import CoopLiftCubeCfg, ObservationsCfg
from bhl_robust.tasks.depth_env_cfg import (
    CAM_APERTURE,
    CAM_FOCAL,
    CAM_POS,
    CAM_RANGE,
    CAM_ROT,
)

# Same pose, aperture and focal length as the locomotion camera, so the two
# depth experiments are one sensor in two tasks rather than two sensors. The
# MuJoCo replay camera in `eval/mjcf_assets.py` is matched to this as well.
COOP_CAM_RES = 64
COOP_CAM_POOL = 8


def make_coop_depth_camera(robot: str, res: int = COOP_CAM_RES) -> MultiMeshRayCasterCameraCfg:
    """A forward-looking depth camera on `robot` that can see the payload.

    The target list is the whole argument of this module. `/World/ground` is
    shared and static, so its transform is not tracked -- that is the cheap
    path and it is what `RayCasterCamera` does for everything. The object is
    per-environment and tracked, which is the capability `RayCasterCamera`
    lacks and the reason a lift could not be given vision before.

    The partner robot is deliberately *not* a target. Tracking an articulation
    means a transform read per link per env per step, and the partner's
    configuration is already in the observation exactly (its joint positions,
    velocities and projected gravity are six of the terms). Ray-casting it would
    buy a noisy view of something already known, at the cost of the tracked
    target that is not.
    """
    return MultiMeshRayCasterCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{robot}/base",
        mesh_prim_paths=[
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="/World/ground", is_shared=True, track_mesh_transforms=False,
            ),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/object", is_shared=True,
                track_mesh_transforms=True,
            ),
        ],
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=CAM_POS, rot=CAM_ROT, convention="world"),
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",
        max_distance=CAM_RANGE,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=CAM_FOCAL, horizontal_aperture=CAM_APERTURE,
            width=res, height=res,
        ),
        update_period=0.0,
        debug_vis=False,
    )


def coop_depth_obs(env, sensor_cfg: SceneEntityCfg, pool: int = COOP_CAM_POOL,
                   clip: float = CAM_RANGE) -> torch.Tensor:
    """One robot's depth image, average-pooled and scaled to roughly [0, 1].

    Identical treatment to `depth_env_cfg.depth_obs`, restated here only so the
    pool factor differs without touching the locomotion experiment.
    """
    d = env.scene[sensor_cfg.name].data.output["distance_to_image_plane"]
    if d.ndim == 3:
        d = d.unsqueeze(-1)
    d = d.permute(0, 3, 1, 2).nan_to_num(nan=clip, posinf=clip)
    return (F.avg_pool2d(d, pool).flatten(1) / clip).clamp(0.0, 1.0)


def _depth_term(name: str) -> ObsTerm:
    return ObsTerm(
        func=coop_depth_obs,
        params={"sensor_cfg": SceneEntityCfg(name), "pool": COOP_CAM_POOL},
        # Depth is scaled to [0, 1] over 6 m, so 2 cm of range noise is 0.0033.
        noise=GaussianNoiseCfg(mean=0.0, std=0.0033),
    )


@configclass
class CoopDepthObservationsCfg(ObservationsCfg):
    """§5's groups with a depth image per robot appended to the policy.

    The critic is left privileged, as everywhere else in §5: it keeps the object
    pose and twist. That is the point of an asymmetric actor-critic -- the value
    function may cheat, the deployed policy may not.
    """

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        depth_a = _depth_term("depth_cam_a")
        depth_b = _depth_term("depth_cam_b")

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        depth_a = _depth_term("depth_cam_a")
        depth_b = _depth_term("depth_cam_b")

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class CoopLiftDepthCfg(CoopLiftCubeCfg):
    """§5's cube lift with a payload-tracking depth camera on each robot.

    `drop_object_pose` is the arm selector, and it defaults to the interesting
    case. With it set the policy's only channel to the cube is the image, which
    is the configuration a real robot could run; cleared, the policy gets both
    and the run is the control that separates "vision suffices" from "vision
    helps".

    Staging is on by default because it is what §5 concluded: zero the lift
    weights until the batch-mean pinch clears the gate, then turn them on. An
    arm that changes the observation *and* the reward schedule at once would not
    be comparable to anything.
    """

    observations: CoopDepthObservationsCfg = CoopDepthObservationsCfg()

    drop_object_pose: bool = True
    stage_lift_on_pinch: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.scene.depth_cam_a = make_coop_depth_camera("robot_a")
        self.scene.depth_cam_b = make_coop_depth_camera("robot_b")
        apply_depth_flags(self)


# Captured once at import, while the terms still exist. Dropping an observation
# term is destructive -- setting the field to `None` discards the only reference
# to it -- and hydra's override lands *after* `__post_init__` has already done
# the dropping. So `drop_object_pose=false` could set the flag but never get the
# terms back, and the first submission of `58_coop_depth` was about to train
# `depthboth` as a bit-identical copy of `depthswap`: the validator caught it as
# `obs_width: 316 against replay's 322`. This is the same class of bug §5
# documents for `nopriv`/`notrack`, which is why the validator asserts widths.
_PRISTINE_OBJECT_TERMS = {
    "object_pos_a": ObservationsCfg.PolicyCfg().object_pos_a,
    "object_pos_b": ObservationsCfg.PolicyCfg().object_pos_b,
}


def apply_depth_flags(cfg) -> None:
    """Make the policy's object-pose terms match `drop_object_pose`, either way.

    Idempotent and reversible, so it is safe to call in `__post_init__` and
    again after hydra. Restoring puts each term back in its own dataclass field,
    and field order is fixed by declaration, so the rebuilt observation vector
    has the same layout the checkpoint was trained with rather than the same
    contents in a new order.
    """
    drop = getattr(cfg, "drop_object_pose", False)
    policy = cfg.observations.policy
    for name, pristine in _PRISTINE_OBJECT_TERMS.items():
        if drop:
            setattr(policy, name, None)
        elif getattr(policy, name, None) is None:
            setattr(policy, name, copy.deepcopy(pristine))
