"""Harder cooperative-lift variants: randomised payloads, and an unseen payload.

Two of the five follow-ups live here.

**Payload randomisation.** Every cube run so far has lifted the same 0.5 kg,
0.28 m block with the same friction. A policy that only lifts that block has
memorised a mass, and the sim2sim harness cannot tell the difference because it
replays the same block. Mass and surface friction are randomised at startup, on
the object rather than the robots -- the robots already get upstream's
randomisation and this is about the thing being carried.

**The occluded payload, which is the only honest way to ask for vision.** The
blind policy is handed `object_pos_in_root`: the exact pose. Adding a camera to
that measured *worse* -- pinch collapsed from 0.30 to 0.00 -- and the reason is
structural, not a tuning failure. Depth was substituting a pooled image for a
quantity the policy already had exactly.

So the vision arms in this repo were never a fair test of vision. This class is:
the exact object pose is withheld from the actor, and the payload's size varies
per environment so its geometry cannot be memorised either. Now depth is the
only route to knowing where the object is and how big it is, and "does vision
help" becomes answerable. If a sighted policy cannot beat a blind one *here*,
the finding is about the encoder or the task, not about redundant input.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from berkeley_humanoid_lite.tasks.locomotion.velocity import mdp

from .coop_lift_env_cfg import CoopLiftCubeCfg
from .coop_depth_env_cfg import CoopLiftDepthCfg


def _payload_events(events):
    """Attach startup randomisation of payload mass, friction and size."""
    events.object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            # The pair carries 0.5 kg. +/- 60% spans a light box to one that is
            # a real fraction of what two 6 Nm arms can hold up.
            "mass_distribution_params": (-0.3, 0.3),
            "operation": "add",
        },
        mode="startup",
    )
    events.object_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            # A non-prehensile pinch is entirely a friction bet. The low end is
            # a payload that squirts out of a clamp the high end holds.
            "static_friction_range": (0.7, 1.8),
            "dynamic_friction_range": (0.6, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 32,
        },
        mode="startup",
    )
    return events


@configclass
class CoopLiftRandomCfg(CoopLiftCubeCfg):
    """Blind lift, randomised payload mass and friction."""

    def __post_init__(self):
        super().__post_init__()
        _payload_events(self.events)


@configclass
class CoopLiftOccludedCfg(CoopLiftCubeCfg):
    """Blind lift with the object pose withheld -- the control for vision.

    This arm is *expected* to do badly. It exists so the sighted arm below has
    something to beat that is not a policy holding a free exact answer.
    """

    def __post_init__(self):
        super().__post_init__()
        _payload_events(self.events)
        self.events.object_scale = EventTerm(
            func=mdp.randomize_rigid_body_scale,
            params={"asset_cfg": SceneEntityCfg("object"),
                    "scale_range": {"x": (0.8, 1.25), "y": (0.8, 1.25), "z": (0.8, 1.25)}},
            mode="prestartup",
        )
        # The whole point: the actor no longer knows where the object is.
        self.observations.policy.object_pos_a = None
        self.observations.policy.object_pos_b = None


@configclass
class CoopLiftOccludedDepthCfg(CoopLiftDepthCfg):
    """Sighted lift with the object pose withheld. Depth is the only route."""

    def __post_init__(self):
        super().__post_init__()
        _payload_events(self.events)
        self.events.object_scale = EventTerm(
            func=mdp.randomize_rigid_body_scale,
            params={"asset_cfg": SceneEntityCfg("object"),
                    "scale_range": {"x": (0.8, 1.25), "y": (0.8, 1.25), "z": (0.8, 1.25)}},
            mode="prestartup",
        )
        self.observations.policy.object_pos_a = None
        self.observations.policy.object_pos_b = None
