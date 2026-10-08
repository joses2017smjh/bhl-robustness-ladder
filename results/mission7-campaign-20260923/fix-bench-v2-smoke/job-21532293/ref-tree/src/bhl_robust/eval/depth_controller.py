"""An `RlController` that can carry a depth term, for replaying sighted policies.

Upstream's controller assembles the observation itself from raw pieces --
command, angular velocity, projected gravity, joint position, joint velocity,
previous actions -- which comes to 45 for this biped. A depth-conditioned policy
wants 301, and upstream has no concept of depth, so replaying one raised

    ValueError: could not broadcast input array from shape (45,) into shape (1, 301)

before a frame was drawn. That is what blocked the B3 ice clip: the finding with
the strongest numbers in the repo had no picture because the replay could not
feed the network it was replaying.

Appending after `prev_actions` is not a guess. Isaac Lab's term order is
field-declaration order and `@configclass` is a dataclass, so an inherited field
precedes one the subclass adds -- `actions` is declared on the base `PolicyCfg`
and `depth` on the override, so depth lands last. `depth_env_cfg` says the same
thing in its own comment.

`external/` stays pristine by repo rule, so this subclasses rather than patches.
The duplicated assembly is copied deliberately: calling `super().update()` would
write a 45-wide vector into the 301-wide buffer and fail before this code could
append anything.
"""

from __future__ import annotations

import numpy as np
from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController


class DepthRlController(RlController):
    """`RlController` with an optional depth vector appended to each frame."""

    def update(self, robot_observations: np.ndarray,
               depth: np.ndarray | None = None) -> np.ndarray:
        if depth is None:
            return super().update(robot_observations)

        n_act = self.cfg.num_actions
        quat = robot_observations[0:4]
        ang_vel = robot_observations[4:7]
        joint_pos = robot_observations[7:7 + n_act] - self.default_joint_positions
        joint_vel = robot_observations[7 + n_act:7 + n_act * 2]
        command = robot_observations[7 + n_act * 2 + 1:7 + n_act * 2 + 4]

        frame = np.concatenate([
            command,
            ang_vel,
            self.quat_rotate_inverse(quat, self.gravity_vector),
            joint_pos,
            joint_vel,
            self.prev_actions,
            np.asarray(depth, dtype=np.float32).reshape(-1),
        ], axis=0)

        want = self.policy_observations.shape[1]
        if frame.shape[0] != want:
            raise ValueError(
                f"assembled {frame.shape[0]} observations for a {want}-wide "
                f"policy: {frame.shape[0] - depth.size} proprioceptive plus "
                f"{depth.size} depth. The depth term and the checkpoint disagree."
            )

        # History shift, as upstream does it: drop the oldest frame, append the
        # newest. With history_length 1 the leading slice is empty and this is
        # just an assignment.
        self.policy_observations[:] = np.concatenate(
            [self.policy_observations[0, want:], frame], axis=0)

        self.policy_actions[:] = self.policy.forward(self.policy_observations)
        clipped = np.clip(self.policy_actions[0],
                          self.cfg.action_limit_lower, self.cfg.action_limit_upper)
        self.prev_actions[:] = clipped
        return clipped * self.cfg.action_scale + self.default_joint_positions
