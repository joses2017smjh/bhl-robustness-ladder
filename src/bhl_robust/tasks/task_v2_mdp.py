"""Success criteria for the three redesigned tasks.

Every term here answers "is the task done", which is the thing the original lift
never had. A policy could max every reward and the episode still ended on a
timeout, so there was no state the environment called success and nothing to
report a success *rate* over -- only curves that went up.

Each task gets a predicate, used twice: as a termination so the episode ends on
completion, and as a large one-off reward so completing is worth more than
hovering near completion. `held_for` exists because a payload that passes
through a target is not a payload that was placed in it; every predicate has to
stay true for a stretch before it counts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from bhl_robust.tasks.coop_lift_mdp import _t

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _obj_local(env: "ManagerBasedRLEnv", name: str = "object") -> torch.Tensor:
    """Object position relative to its own env's origin.

    Targets are placed per-env at fixed local offsets, so every comparison has
    to be in local coordinates. Using world coordinates would make the test pass
    only for the env that happens to sit at the world origin -- and would do it
    silently, since env 0 is exactly the one a debug print shows.
    """
    obj: RigidObject = env.scene[name]
    return _t(obj.data.root_pos_w)[:, :3] - env.scene.env_origins


def _held(env: "ManagerBasedRLEnv", key: str, now: torch.Tensor,
          steps: int) -> torch.Tensor:
    """True once `now` has been continuously true for `steps` steps."""
    buf = getattr(env, key, None)
    if buf is None or buf.shape[0] != env.num_envs:
        buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    buf = torch.where(now, buf + 1, torch.zeros_like(buf))
    setattr(env, key, buf)
    return buf >= steps


# --------------------------------------------------------------- cube -> shelf

def cube_in_slot(
    env: "ManagerBasedRLEnv",
    slot: float = 0.34,
    deck_z: float = 0.38,
    shelf_x: float = 1.2,
    depth: float = 0.40,
    speed: float = 0.05,
    hold_steps: int = 12,
) -> torch.Tensor:
    """Cube resting inside the shelf opening.

    The slot is 6 cm wider than the cube, so this cannot be satisfied by
    throwing: a payload arriving with speed either misses or bounces out, and
    the speed test plus `hold_steps` require it to have come to rest.
    """
    p = _obj_local(env, "object")
    obj: RigidObject = env.scene["object"]
    half = slot / 2.0
    inside = (
        (p[:, 0] > shelf_x - depth / 2.0) & (p[:, 0] < shelf_x + depth / 2.0)
        & (p[:, 1].abs() < half)
        & (p[:, 2] > deck_z) & (p[:, 2] < deck_z + slot)
    )
    still = _t(obj.data.root_lin_vel_w).norm(dim=-1) < speed
    return _held(env, "_bhl_slot_hold", inside & still, hold_steps)


# ----------------------------------------------------------------- ball -> net

def ball_in_net(
    env: "ManagerBasedRLEnv",
    mouth: float = 0.70,
    rim_z: float = 0.60,
    net_x: float = 3.5,
    depth: float = 0.45,
    hold_steps: int = 4,
) -> torch.Tensor:
    """Ball inside the net volume, having entered from above.

    The downward-velocity test is what distinguishes a throw from a ball rolled
    in through a wall the collider happened to let through, and it is cheap
    insurance against the volume test alone being satisfied by a tunnelling
    contact at 4,096 envs.
    """
    p = _obj_local(env, "object")
    obj: RigidObject = env.scene["object"]
    half = mouth / 2.0
    inside = (
        ((p[:, 0] - net_x).abs() < half)
        & (p[:, 1].abs() < half)
        & (p[:, 2] < rim_z) & (p[:, 2] > rim_z - depth)
    )
    return _held(env, "_bhl_net_hold", inside, hold_steps)


def ball_toward_net(
    env: "ManagerBasedRLEnv",
    net_x: float = 3.5,
    std: float = 1.5,
) -> torch.Tensor:
    """Shaping: horizontal closing speed on the net, once the ball is airborne.

    Zero while the ball is supported, so it pays for a throw and not for
    shoving the ball along the ground toward the target.
    """
    p = _obj_local(env, "object")
    obj: RigidObject = env.scene["object"]
    airborne = (p[:, 2] > 0.45).float()
    to_net = torch.stack([net_x - p[:, 0], -p[:, 1]], dim=-1)
    to_net = to_net / to_net.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    closing = (_t(obj.data.root_lin_vel_w)[:, :2] * to_net).sum(dim=-1)
    return airborne * torch.tanh(closing.clamp_min(0.0) / std)


# --------------------------------------------------------------- plank -> wall

def plank_leaned(
    env: "ManagerBasedRLEnv",
    wall_x: float = 1.0,
    min_deg: float = 50.0,
    max_deg: float = 80.0,
    foot_gap: float = 0.30,
    contact_z: float = 0.50,
    speed: float = 0.10,
    hold_steps: int = 12,
) -> torch.Tensor:
    """Plank standing against the wall at a leaning angle, and staying there.

    Angle alone is not enough -- a plank stood on end in open floor satisfies
    it. The lower end has to be near the wall base and the upper end has to be
    high, which together can only happen against the wall.
    """
    obj: RigidObject = env.scene["object"]
    p = _obj_local(env, "object")
    q = _t(obj.data.root_quat_w)
    # Long axis of the plank (its local x) in world coordinates.
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    ax = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z),
                      2 * (x * z - w * y)], dim=-1)
    tilt = torch.asin(ax[:, 2].abs().clamp(max=1.0)) * 180.0 / torch.pi
    half = 0.75
    end_hi = p + half * ax
    end_lo = p - half * ax
    swap = end_lo[:, 2] > end_hi[:, 2]
    hi_z = torch.where(swap, end_lo[:, 2], end_hi[:, 2])
    lo_x = torch.where(swap, end_hi[:, 0], end_lo[:, 0])
    # Resting, not merely passing through the right pose. A tumbling plank
    # satisfies angle, foot position and height for a frame or two on its way
    # somewhere else -- with a zero action it did so in 3.1% of environments,
    # which is a measurement of the spawn rather than of the policy. A leaned
    # plank is stationary; that is most of what "leaned" means.
    still = _t(obj.data.root_lin_vel_w).norm(dim=-1) < speed
    ok = (
        (tilt > min_deg) & (tilt < max_deg)
        & ((wall_x - lo_x).abs() < foot_gap + 0.35)
        & (hi_z > contact_z)
        & still
    )
    return _held(env, "_bhl_lean_hold", ok, hold_steps)


# ------------------------------------------------------------------ shared

def carry_progress(
    env: "ManagerBasedRLEnv",
    target_x: float,
    std: float = 0.8,
) -> torch.Tensor:
    """Shaping toward the target in x, gated on the payload being lifted.

    Gated, because an ungated version pays for pushing the payload across the
    floor -- which is the same failure mode as the old lift rewarding a
    collapse: the cheapest way to move a thing is not the way we meant.
    """
    p = _obj_local(env, "object")
    lifted = (p[:, 2] > 0.34).float()
    return lifted * (1.0 - torch.tanh((target_x - p[:, 0]).abs() / std))
