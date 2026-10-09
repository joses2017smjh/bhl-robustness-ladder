"""Command-latched relative heading for the H4 R1HO experiment.

The two appended features are [sin(yaw-reference), cos(yaw-reference)-1],
masked by the observable abs(command_wz) < 0.05 rad/s. Reference yaw is
latched at reset and at any exact change of the float32 command vector.
This is not the hidden explicit-mode/resample reference of R1H's reward.
MuJoCo supplies simulated orientation; this experiment does not establish
performance with an estimated IMU yaw or on physical hardware.
"""
from __future__ import annotations

import math
import numpy as np

from bhl_robust.eval.gait_clock import clock_features

HEADING_KEY = "command_latched_heading"
WZ_THRESHOLD = 0.05
FEATURE_CONTRACT = {
    "features": "sin_error_cos_error_minus_one",
    "reference": "reset_or_exact_float32_command_change",
    "command_change_atol": 0.0,
    "mask": "abs_command_wz_lt_threshold",
    "wz_threshold": WZ_THRESHOLD,
    "yaw_source": "simulated_orientation",
    "reward_reference": "unchanged_R1H_explicit_mode_last_resample",
}


def heading_features(yaw: float, reference: float, command_wz: float) -> np.ndarray:
    if abs(float(command_wz)) >= WZ_THRESHOLD:
        return np.zeros(2, dtype=np.float32)
    error = float(yaw) - float(reference)
    return np.array([math.sin(error), math.cos(error) - 1.0], dtype=np.float32)


def torch_latched_features(yaw, command, reference, previous_command, ready, reset):
    """Batched counterpart of the deployment latch; returns features and new state."""
    import torch
    command = command.to(dtype=torch.float32)
    changed = (command != previous_command).any(dim=-1)
    latch = ~ready | reset.bool() | changed
    reference = torch.where(latch, yaw, reference)
    error = yaw - reference
    features = torch.stack((torch.sin(error), torch.cos(error) - 1.0), dim=-1)
    features = torch.where((command[:, 2].abs() < WZ_THRESHOLD)[:, None], features,
                           torch.zeros_like(features))
    return features, reference, command.clone(), torch.ones_like(ready)


class HeadingLatch:
    """A causal state machine, using current yaw and command only."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.reference = None
        self.previous_command = None

    def update(self, yaw: float, command) -> np.ndarray:
        command = np.asarray(command, dtype=np.float32)
        if command.shape != (3,) or not np.isfinite(command).all() or not math.isfinite(yaw):
            raise ValueError("heading latch requires finite yaw and a three-component command")
        if self.reference is None or not np.array_equal(command, self.previous_command):
            self.reference = float(yaw)
        self.previous_command = command.copy()
        return heading_features(yaw, self.reference, command[2])


_CLS = None


def make_controller(cfg):
    """Create the experiment's controller without changing generic factory code."""
    if HEADING_KEY not in cfg or dict(cfg[HEADING_KEY]) != FEATURE_CONTRACT:
        raise ValueError("H4 deploy feature contract does not match the frozen experiment")
    return _controller_class()(cfg)


def _controller_class():
    global _CLS
    if _CLS is not None:
        return _CLS
    from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController

    class CommandLatchedHeadingController(RlController):
        def __init__(self, cfg):
            super().__init__(cfg)
            if (int(cfg.num_actions), int(cfg.num_joints), int(cfg.num_observations),
                    int(cfg.history_length)) != (22, 22, 79, 0):
                raise ValueError("H4 requires the unchanged 22-DoF asset and 79 observations")
            if dict(cfg.gait_clock).get("period_s") != 0.8 or dict(cfg.gait_clock).get("phase_offset") != 0.0:
                raise ValueError("H4 requires the unchanged R1H gait clock")
            self.heading_latch = HeadingLatch()
            self.clock_step = 0

        def load_policy(self):
            # The upstream loader treats a symbolic ONNX batch dimension as a
            # literal numpy shape and then chooses an unrelated legacy input
            # name. The H4 export has an explicit obs input and dynamic batch.
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(self.cfg.policy_checkpoint_path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
            input_meta, output_meta = session.get_inputs()[0], session.get_outputs()[0]
            if input_meta.shape[-1] != 79 or output_meta.shape[-1] != 22:
                raise ValueError("H4 export must map 79 observations to 22 actions")
            key = input_meta.name
            class OnnxHeadingPolicy:
                def forward(self, observations):
                    return session.run(None, {key: np.asarray(observations, dtype=np.float32)})[0]
            self.policy = OnnxHeadingPolicy()
            probe = self.policy.forward(np.zeros((1, 79), dtype=np.float32))
            if probe.shape != (1, 22) or not np.isfinite(probe).all():
                raise ValueError("H4 ONNX startup probe failed")

        def update(self, robot_observations):
            # Existing runners signal reset by clearing this buffer.
            if not self.policy_observations.any():
                self.heading_latch.reset()
                self.clock_step = 0
            n = self.cfg.num_actions
            quat = robot_observations[:4]
            command = np.asarray(robot_observations[8 + 2 * n:11 + 2 * n], dtype=np.float32)
            yaw = math.atan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                             1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
            heading = self.heading_latch.update(yaw, command)
            clock = clock_features(self.clock_step, float(self.cfg.policy_dt), 0.8)
            self.policy_observations[:] = np.concatenate((
                command, robot_observations[4:7],
                self.quat_rotate_inverse(quat, self.gravity_vector),
                robot_observations[7:7 + n] - self.default_joint_positions,
                robot_observations[7 + n:7 + 2 * n], self.prev_actions, clock, heading))
            self.clock_step += 1
            self.policy_actions[:] = self.policy.forward(self.policy_observations)
            clipped = np.clip(self.policy_actions[0], self.cfg.action_limit_lower,
                              self.cfg.action_limit_upper)
            self.prev_actions[:] = clipped
            return clipped * self.cfg.action_scale + self.default_joint_positions

    _CLS = CommandLatchedHeadingController
    return _CLS


def expand_parent_state(parent_state, target_state):
    """Copy all parent tensors, with two zero columns appended to first layers."""
    import torch
    if set(parent_state) != set(target_state):
        raise ValueError("parent and candidate parameter keys differ")
    result = {}
    for key, source in parent_state.items():
        target = target_state[key]
        if key in ("actor.0.weight", "critic.0.weight"):
            expected = 77 if key.startswith("actor") else 80
            if source.shape != (256, expected) or target.shape != (256, expected + 2):
                raise ValueError(f"unexpected first-layer shape: {key}")
            result[key] = torch.cat((source, source.new_zeros((256, 2))), dim=1)
        else:
            if source.shape != target.shape:
                raise ValueError(f"unexpected parent parameter shape: {key}")
            result[key] = source.clone()
    return result
