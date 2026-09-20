"""Exercise physical-step accounting and reward scale without Isaac Sim."""
from types import SimpleNamespace
import unittest

import torch
from bhl_robust import maze_recovery as m


class Scene(dict):
    env_origins = torch.zeros(2, 3)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.pos = torch.tensor([[2.8, 0., 0.6], [0., 0., 0.6]])
        self.data = SimpleNamespace(root_pos_w=self.pos,
            projected_gravity_b=torch.tensor([[0., 0., -1.], [0., 0., -1.]]),
            root_lin_vel_b=torch.zeros(2, 3), root_ang_vel_b=torch.zeros(2, 3))
        self.env = SimpleNamespace(scene=Scene(robot=SimpleNamespace(data=self.data)),
            common_step_counter=0, episode_length_buf=torch.ones(2, dtype=torch.long),
            step_dt=0.04, termination_manager=SimpleNamespace(terminated=torch.tensor([True, True])))

    def step(self):
        self.env.common_step_counter += 1
        self.env.episode_length_buf += 1

    def test_hold_counts_physical_steps_not_callback_calls(self):
        m.button_reached(self.env)
        for i in range(1, 9):
            self.step()
            for _ in range(3):
                self.assertEqual(bool(m.button_reached(self.env)[0]), i == 8)
                m.progress_rate(self.env)
        self.assertEqual(float(m.success_bonus(self.env)[0] * self.env.step_dt), 1.0)
        self.assertEqual(m.failure_penalty(self.env).tolist(), [0., 1.])

    def test_progress_is_a_rate_and_reset_does_not_pay_teleportation(self):
        m.progress_rate(self.env)
        self.step()
        self.pos[1, 0] += 0.02
        self.assertAlmostEqual(float(m.progress_rate(self.env)[1]), 0.5, places=4)
        self.step()
        self.env.episode_length_buf[1] = 0
        self.pos[1, 0] = 2.8
        self.assertEqual(float(m.progress_rate(self.env)[1]), 0.0)
        self.assertFalse(bool(m.button_reached(self.env)[1]))

    def test_command_brakes_and_does_not_drive_backwards_while_turning(self):
        d = torch.tensor([0.20, 0.50, 2.0, 2.0])
        h = torch.tensor([0.0, 0.0, 0.0, 3.14])
        v = m.command_speed(d, h)
        self.assertEqual(float(v[0]), 0.0)
        self.assertTrue(0 < float(v[1]) < float(v[2]))
        self.assertEqual(float(v[3]), 0.0)

    def test_fall_on_last_dwell_step_is_not_success(self):
        m.button_reached(self.env)
        for _ in range(7):
            self.step()
            m.button_reached(self.env)
        self.data.projected_gravity_b[0] = torch.tensor([1., 0., 0.])
        self.step()
        self.assertFalse(bool(m.button_reached(self.env)[0]))
        self.assertEqual(float(m.failure_penalty(self.env)[0]), 1.0)

    def test_fast_crossing_resets_the_dwell(self):
        m.button_reached(self.env)
        for _ in range(8):
            self.step()
            self.data.root_lin_vel_b[0, 0] = 0.3
            self.assertFalse(bool(m.button_reached(self.env)[0]))
        self.data.root_lin_vel_b[0, 0] = 0.0
        for i in range(1, 9):
            self.step()
            self.assertEqual(bool(m.button_reached(self.env)[0]), i == 8)


if __name__ == "__main__":
    unittest.main()
