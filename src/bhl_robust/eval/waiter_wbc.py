"""Waiter phase 1 deploy side (docs/WAITER_PROGRAM.md): the deploy.yaml stamp for `Velocity-BHL-Waiter-WBC-v0`.

`stamp` appends two blocks to the exported deploy.yaml, derived from the run's own params/env.yaml, after checking
that the export is the frozen layout: the `gait_clock` block (as R1's, so `gait_clock.has_clock` holds) and a
`waiter_wbc` block (upper-body joints and observation layout). The existing text stays byte-identical.

Usage: python -m bhl_robust.eval.waiter_wbc stamp --deploy <deploy.yaml> --env-yaml <params/env.yaml>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bhl_robust import waiter_asset as W
from bhl_robust.eval import gait_clock as G

KEY = "waiter_wbc"
TASK_ID = "Velocity-BHL-Waiter-WBC-v0"
POLICY_TERMS = ["velocity_commands", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions",
                "upper_body", "gait_clock"]
CRITIC_TERMS = POLICY_TERMS[:6] + ["base_lin_vel", "upper_body", "gait_clock"]
OBS_POLICY = 83
OBS_CRITIC = 86
N_JOINTS = 24
N_ACTIONS = 12
LEG_INDICES = list(range(10, 22))


def check_layout(cfg, env: dict) -> list[str]:
    """Problems with an exported deploy config + env.yaml against the frozen phase 1 layout ([] = none)."""
    p = []
    obs = env["observations"]
    if G._terms(obs["policy"]) != POLICY_TERMS:
        p.append(f"actor terms {G._terms(obs['policy'])}")
    if G._terms(obs["critic"]) != CRITIC_TERMS:
        p.append(f"critic terms {G._terms(obs['critic'])}")
    if int(cfg.num_joints) != N_JOINTS or list(cfg.joints) != W.JOINT_ORDER:
        p.append(f"joints {cfg.num_joints} {list(cfg.joints)}")
    if int(cfg.num_actions) != N_ACTIONS or list(cfg.action_indices) != LEG_INDICES:
        p.append(f"actions {cfg.num_actions} {list(cfg.action_indices)}")
    if int(cfg.num_observations) != OBS_POLICY or int(cfg.history_length) != 0:
        p.append(f"observations {cfg.num_observations} history {cfg.history_length}")
    eff = list(cfg.effort_limits)
    if any(abs(eff[i] - W.ARM_EFFORT_NM) > 1e-6 for i in range(10)) or any(abs(eff[i] - W.GRIPPER_EFFORT_NM) > 1e-6
                                                                           for i in (22, 23)):
        p.append(f"effort limits {eff}")
    ub = env["commands"].get("upper_body") or {}
    if list(ub.get("joint_names", [])) != W.UPPER_BODY_JOINTS:
        p.append(f"upper_body joints {ub.get('joint_names')}")
    return p


def stamp(deploy: Path, env_yaml: Path) -> str:
    from omegaconf import OmegaConf
    deploy, env_yaml = Path(deploy), Path(env_yaml)
    env = G.load_env_yaml(env_yaml)
    info = G.check_env(env, "R1")                     # gait_clock last in actor and critic, period 0.8, offset 0
    cfg = OmegaConf.load(deploy)
    problems = check_layout(cfg, env)
    if abs(float(cfg.policy_dt) - info["step_dt"]) > 1e-9:
        problems.append(f"policy_dt {cfg.policy_dt} != env step_dt {info['step_dt']}")
    if problems:
        raise ValueError("not the frozen Waiter WBC layout: " + "; ".join(problems))
    if KEY in cfg:
        return f"already stamped: {deploy}"
    if G.CLOCK_KEY in cfg:
        raise ValueError(f"{deploy} has a gait_clock block but no {KEY} block")
    text = deploy.read_text()
    joints = "".join(f"    - {j}\n" for j in W.UPPER_BODY_JOINTS)
    block = (f"{'' if text.endswith(chr(10)) else chr(10)}{G.CLOCK_KEY}:\n"
             f"  period_s: {info['period_s']}\n"
             f"  phase_offset: {info['phase_offset']}\n"
             f"  layout: sin_cos_appended_last\n"
             f"  source: {env_yaml}\n"
             f"{KEY}:\n"
             f"  task: {TASK_ID}\n"
             f"  upper_body_joints:\n{joints}"
             f"  upper_body_indices: {list(range(10)) + [22, 23]}\n"
             f"  gripper_closed_rad: {W.GRIPPER_CLOSED_RAD}\n"
             f"  policy_terms: {POLICY_TERMS}\n"
             f"  source: {env_yaml}\n")
    deploy.write_text(text + block)
    back = OmegaConf.load(deploy)
    assert list(back[KEY]["upper_body_joints"]) == W.UPPER_BODY_JOINTS and G.has_clock(back)
    return f"stamped {deploy}: gait_clock {info['period_s']} s + {KEY}"


class WaiterWbcController:
    """MuJoCo-side WBC: the 12-action leg policy + commanded arm and gripper targets (deploy.yaml stamped by `stamp`).

    Same interface as upstream's RlController (`policy`, `prev_actions`, `policy_observations`, `update`), so the
    existing runners drive it. `update(robot_observations)` takes [quat 4, ang vel 3, joint pos 24, joint vel 24,
    mode 1, command 3] (all 24 joints in deploy order) and returns 24 joint position targets: legs from the policy
    (action x scale + default), arms and grippers = `upper_body_target` (absolute rad; default pose, grippers open
    unless set). The actor observation is Isaac's term order (POLICY_TERMS), the gait clock as GaitClockRlController.
    """

    def __init__(self, cfg):
        import numpy as np
        self.cfg = cfg
        if KEY not in cfg or not G.has_clock(cfg):
            raise ValueError("not a stamped Waiter WBC deploy config")
        self.n_j = int(cfg.num_joints)
        self.n_a = int(cfg.num_actions)
        self.leg_idx = np.asarray(cfg.action_indices, dtype=int)
        self.ub_idx = np.asarray(list(cfg[KEY]["upper_body_indices"]), dtype=int)
        self.gripper_closed = float(cfg[KEY]["gripper_closed_rad"])
        self.default_joint_positions = np.asarray(cfg.default_joint_positions, dtype=np.float32)
        self.command_velocity = np.asarray(cfg.command_velocity, dtype=np.float32)
        self.gravity_vector = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.policy_observations = np.zeros((1, int(cfg.num_observations)), dtype=np.float32)
        self.policy_actions = np.zeros((1, self.n_a), dtype=np.float32)
        self.prev_actions = np.zeros((self.n_a,), dtype=np.float32)
        self.upper_body_target = self.default_joint_positions[self.ub_idx].copy()
        self.clock_period = float(cfg[G.CLOCK_KEY]["period_s"])
        self.clock_phase_offset = float(cfg[G.CLOCK_KEY]["phase_offset"])
        self.clock_step_dt = float(cfg.policy_dt)
        self.clock_step = 0
        self.policy = None
        if self.n_j != N_JOINTS or self.n_a != N_ACTIONS or int(cfg.num_observations) != OBS_POLICY:
            raise ValueError(f"Waiter WBC deploy config: {self.n_j} joints / {self.n_a} actions / "
                             f"{cfg.num_observations} observations")

    @staticmethod
    def quat_rotate_inverse(q, v):
        """Upstream RlController.quat_rotate_inverse, verbatim (q = [w, x, y, z])."""
        import numpy as np
        q_w, q_vec = q[0], q[1:4]
        a = v * (2.0 * q_w ** 2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * (np.dot(q_vec, v)) * 2.0
        return a - b + c

    def set_upper_body(self, target) -> None:
        """Absolute targets for the 12 upper-body joints (10 arm joints in rad, 2 grippers in rad)."""
        import numpy as np
        t = np.asarray(target, dtype=np.float32)
        if t.shape != (12,):
            raise ValueError(f"upper-body target shape {t.shape}")
        self.upper_body_target[:] = t

    def reset_phase(self) -> None:
        self.clock_step = 0

    def update(self, robot_observations):
        import numpy as np
        if not self.policy_observations.any():
            self.clock_step = 0
        n = self.n_j
        quat = robot_observations[0:4]
        ang_vel = robot_observations[4:7]
        jpos = robot_observations[7:7 + n] - self.default_joint_positions
        jvel = robot_observations[7 + n:7 + 2 * n]
        cmd = robot_observations[7 + 2 * n + 1:7 + 2 * n + 4]
        gravity = self.quat_rotate_inverse(quat, self.gravity_vector)
        enc = (self.upper_body_target - self.default_joint_positions[self.ub_idx]).astype(np.float32)
        enc[10:] = self.upper_body_target[10:] / self.gripper_closed
        clock = G.clock_features(self.clock_step, self.clock_step_dt, self.clock_period, self.clock_phase_offset)
        self.policy_observations[0, :] = np.concatenate([cmd, ang_vel, gravity, jpos, jvel, self.prev_actions,
                                                         enc, clock]).astype(np.float32)
        self.clock_step += 1
        self.policy_actions[:] = self.policy.forward(self.policy_observations)
        a = np.clip(self.policy_actions[0], self.cfg.action_limit_lower, self.cfg.action_limit_upper)
        self.prev_actions[:] = a
        targets = self.default_joint_positions.copy()
        targets[self.leg_idx] = a * float(self.cfg.action_scale) + self.default_joint_positions[self.leg_idx]
        targets[self.ub_idx] = self.upper_body_target
        return targets


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stamp")
    s.add_argument("--deploy", type=Path, required=True)
    s.add_argument("--env-yaml", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        print("WAITER-WBC:", stamp(a.deploy, a.env_yaml))
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"WAITER-WBC: STAMP FAILED ({e})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
