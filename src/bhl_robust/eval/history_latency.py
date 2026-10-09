"""H3 controller: current joints/commands plus causally delayed IMU packets."""
from __future__ import annotations

import numpy as np
import torch

from bhl_robust.latency_history import ACTOR_WIDTH, CausalImuHistory, POLICY_DT


_CONTROLLER = None


def load_onnx_policy(path):
    """Use the model's declared input name, including dynamic batch exports."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or inputs[0].shape[-1] != 63 or outputs[0].shape[-1] != 12:
        raise ValueError("H3 ONNX interface is not one 63-column input and 12 actions")

    class _Policy:
        def forward(self, observations):
            return session.run(None, {inputs[0].name: observations})[0]

    return _Policy()


def make_history_controller(cfg, *, arm, delay_steps):
    global _CONTROLLER
    if _CONTROLLER is None:
        from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController

        class HistoryLatencyController(RlController):
            def __init__(self, cfg, *, arm, delay_steps):
                if int(cfg.num_actions) != 12 or int(cfg.num_joints) != 12:
                    raise ValueError("H3 requires the unchanged 12-DoF biped")
                if int(cfg.num_observations) != ACTOR_WIDTH or int(cfg.history_length) != 0:
                    raise ValueError("H3 deploy must declare 63 observations and no full-frame history")
                if abs(float(cfg.policy_dt) - POLICY_DT) > 1e-12:
                    raise ValueError("H3 requires 40 ms policy ticks")
                super().__init__(cfg)
                mode = "history" if arm == "history" else "feedforward" if arm in (
                    "feedforward", "repeated_current") else None
                if mode is None:
                    raise ValueError("unknown H3 arm")
                self.packet_queue = CausalImuHistory(1, "cpu", mode=mode,
                                                    max_delay_steps=3, noise=False)
                self.packet_queue.force_delay = delay_steps
                self.tick = 0
                self.last_packet_source_ticks = []
                self.reset_packets()

            def reset_packets(self):
                self.packet_queue.reset()
                self.tick = 0
                self.last_packet_source_ticks = []

            def load_policy(self):
                # Upstream's ONNX bootstrap tries np.zeros on the declared
                # shape; a dynamic batch name makes it choose an incorrect
                # fallback input key. Read the actual model interface here.
                self.policy = load_onnx_policy(self.cfg.policy_checkpoint_path)

            def update(self, robot_observations):
                if not self.policy_observations.any():
                    self.reset_packets()
                n = self.cfg.num_actions
                obs = np.asarray(robot_observations, dtype=np.float32)
                gravity = self.quat_rotate_inverse(obs[:4], self.gravity_vector)
                current = np.concatenate([obs[4:7], gravity]).astype(np.float32)
                self.last_captured_imu = current.tolist()
                packets = self.packet_queue.update(torch.from_numpy(current)[None, :], self.tick)[0].numpy()
                self.last_packet_source_ticks = self.packet_queue.source_ticks()[0].tolist()
                self.policy_observations[0] = np.concatenate([
                    obs[7 + 2 * n + 1:7 + 2 * n + 4], packets[-1],
                    obs[7:7 + n] - self.default_joint_positions,
                    obs[7 + n:7 + 2 * n], self.prev_actions, packets[:-1].reshape(-1),
                ])
                self.tick += 1
                self.policy_actions[:] = self.policy.forward(self.policy_observations)
                clipped = np.clip(self.policy_actions[0], self.cfg.action_limit_lower,
                                  self.cfg.action_limit_upper)
                self.prev_actions[:] = clipped
                return clipped * self.cfg.action_scale + self.default_joint_positions

        _CONTROLLER = HistoryLatencyController
    return _CONTROLLER(cfg, arm=arm, delay_steps=delay_steps)
