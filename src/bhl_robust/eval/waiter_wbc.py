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
CURRICULUM_TASK_ID = "Velocity-BHL-Waiter-WBC-Curriculum-v0"     # phase 1c (2026-10-07)
STANCE_TASK_ID = "Velocity-BHL-Waiter-WBC-Stance-v0"             # phase 1d (2026-10-08)
STAND_TASK_ID = "Velocity-BHL-Waiter-WBC-Stand-v0"               # phase 1e (2026-10-10)
STANDPUSH_TASK_ID = "Velocity-BHL-Waiter-WBC-StandPush-v0"       # phase 1f (2026-10-11)
STANDBONUS_TASK_ID = "Velocity-BHL-Waiter-WBC-StandBonus-v0"     # phase 1g (2026-10-11)
STANDSTILL_TASK_ID = "Velocity-BHL-Waiter-WBC-StandStill-v0"     # phase 1h (2026-10-12)
TASK_IDS = (TASK_ID, CURRICULUM_TASK_ID, STANCE_TASK_ID, STAND_TASK_ID, STANDPUSH_TASK_ID, STANDBONUS_TASK_ID,
            STANDSTILL_TASK_ID)
STAND_ANCHOR_WEIGHT = -1.0
STAND_TILT_WEIGHT = -0.5
STAND_ANCHOR_BONUS_WEIGHT = 2.0          # phase 1g: positive, bounded (waiter_stand_mdp.bounded_bonus)
STAND_TILT_BONUS_WEIGHT = 1.0
STAND_STILL_WEIGHT = -0.4                # phase 1h: Isaac Lab's Digit stand_still weight
STILL_COMMAND_THRESHOLD = 0.06
STAND_SHARE = (0.02, 0.25)               # phase 1h: rel_standing_envs at the start and the end of its curriculum
STANCE_REL_STANDING_ENVS = 0.25
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


def check_curriculum(env: dict) -> list[str]:
    """Phase 1c only: the disturbance curriculum term is present and the hand payload is a reset event ([] = ok).
    The frozen ramp values are checked by gpu_waiter_wbc.sbatch for full runs (the smoke compresses the ramp)."""
    p = []
    term = (env.get("curriculum") or {}).get("disturbance") or {}
    if "disturbance_curriculum" not in str(term.get("func", "")):
        p.append(f"curriculum term disturbance missing ({term.get('func')})")
    if ((env.get("events") or {}).get("hand_payload") or {}).get("mode") != "reset":
        p.append("hand_payload is not a reset event")
    return p


def check_stance(env: dict) -> list[str]:
    """Phase 1d only: standing envs at 0.25 and the stance-held clock in the actor, critic and feet_gait ([] = ok)."""
    p = []
    rel = ((env.get("commands") or {}).get("base_velocity") or {}).get("rel_standing_envs")
    if rel is None or abs(float(rel) - STANCE_REL_STANDING_ENVS) > 1e-9:
        p.append(f"rel_standing_envs {rel}")
    for group in ("policy", "critic"):
        f = str(((env["observations"].get(group) or {}).get("gait_clock") or {}).get("func", ""))
        if "gait_clock_stance" not in f:
            p.append(f"{group} gait_clock func {f}")
    f = str(((env.get("rewards") or {}).get("feet_gait") or {}).get("func", ""))
    if "feet_gait_stance" not in f:
        p.append(f"feet_gait func {f}")
    return p


def check_stand(env: dict) -> list[str]:
    """Phase 1e only: 1d's stance terms plus no disturbances on standing envs and the two standing penalties."""
    p = check_stance(env)
    ev = env.get("events") or {}
    for name, fn in (("push_robot", "push_moving_only"), ("hand_forces", "hand_forces_moving_only")):
        f = str((ev.get(name) or {}).get("func", ""))
        if fn not in f:
            p.append(f"{name} func {f}")
    rw = env.get("rewards") or {}
    for name, fn, w in (("stand_anchor", "StandAnchor", STAND_ANCHOR_WEIGHT), ("stand_tilt", "stand_tilt", STAND_TILT_WEIGHT)):
        term = rw.get(name) or {}
        if fn not in str(term.get("func", "")):
            p.append(f"{name} func {term.get('func')}")
        if term.get("weight") is None or abs(float(term["weight"]) - w) > 1e-9:
            p.append(f"{name} weight {term.get('weight')}")
    return p


def check_stand_push(env: dict) -> list[str]:
    """Phase 1f only: 1d's stance terms, the two standing penalties, and disturbances on every env (the push event
    is push_and_reanchor, the hand forces are the unrestricted ones)."""
    p = check_stance(env)
    ev = env.get("events") or {}
    f = str((ev.get("push_robot") or {}).get("func", ""))
    if "push_and_reanchor" not in f:
        p.append(f"push_robot func {f}")
    f = str((ev.get("hand_forces") or {}).get("func", ""))
    if not f.endswith("hand_forces"):
        p.append(f"hand_forces func {f}")
    rw = env.get("rewards") or {}
    for name, fn, w in (("stand_anchor", "StandAnchor", STAND_ANCHOR_WEIGHT), ("stand_tilt", "stand_tilt", STAND_TILT_WEIGHT)):
        term = rw.get(name) or {}
        if fn not in str(term.get("func", "")):
            p.append(f"{name} func {term.get('func')}")
        if term.get("weight") is None or abs(float(term["weight"]) - w) > 1e-9:
            p.append(f"{name} weight {term.get('weight')}")
    return p


def check_stand_bonus(env: dict) -> list[str]:
    """Phase 1g only: 1d's stance terms, disturbances on every env as 1f (push_and_reanchor, unrestricted hand
    forces), the two bounded positive standing rewards with their weights, and NO standing penalty term."""
    p = check_stance(env)
    ev = env.get("events") or {}
    f = str((ev.get("push_robot") or {}).get("func", ""))
    if "push_and_reanchor" not in f:
        p.append(f"push_robot func {f}")
    f = str((ev.get("hand_forces") or {}).get("func", ""))
    if not f.endswith("hand_forces"):
        p.append(f"hand_forces func {f}")
    rw = env.get("rewards") or {}
    for name, fn, w in (("stand_anchor_bonus", "StandAnchorBonus", STAND_ANCHOR_BONUS_WEIGHT),
                        ("stand_tilt_bonus", "stand_tilt_bonus", STAND_TILT_BONUS_WEIGHT)):
        term = rw.get(name) or {}
        if not str(term.get("func", "")).endswith(fn):
            p.append(f"{name} func {term.get('func')}")
        if term.get("weight") is None or abs(float(term["weight"]) - w) > 1e-9:
            p.append(f"{name} weight {term.get('weight')}")
    for name in ("stand_anchor", "stand_tilt"):
        if rw.get(name) is not None:
            p.append(f"{name} penalty present")
    return p


def check_stand_still(env: dict, window: bool = True) -> list[str]:
    """Phase 1h only: 1d's held-clock stance terms; the standing share starting at 0.02 and its curriculum to 0.25
    (with window=True also on the frozen 4000-7000 window; a smoke run compresses it, as it does the disturbance ramp);
    disturbances on every env as 1d (upstream push, unrestricted hand forces); the stand-still term on the 12 leg joints
    with its weight and threshold; and none of the 1e-1g standing terms."""
    p = []
    rel = ((env.get("commands") or {}).get("base_velocity") or {}).get("rel_standing_envs")
    if rel is None or abs(float(rel) - STAND_SHARE[0]) > 1e-9:
        p.append(f"rel_standing_envs {rel}")
    for group in ("policy", "critic"):
        f = str(((env["observations"].get(group) or {}).get("gait_clock") or {}).get("func", ""))
        if "gait_clock_stance" not in f:
            p.append(f"{group} gait_clock func {f}")
    rw = env.get("rewards") or {}
    if "feet_gait_stance" not in str((rw.get("feet_gait") or {}).get("func", "")):
        p.append(f"feet_gait func {(rw.get('feet_gait') or {}).get('func')}")
    cur = ((env.get("curriculum") or {}).get("standing_share") or {})
    cp = cur.get("params") or {}
    if not str(cur.get("func", "")).endswith("standing_share_curriculum") or \
            cp.get("steps_per_iter") != 24 or (window and (cp.get("start_iter"), cp.get("end_iter")) != (4000, 7000)) or \
            cp.get("lo") is None or abs(float(cp["lo"]) - STAND_SHARE[0]) > 1e-9 or \
            cp.get("hi") is None or abs(float(cp["hi"]) - STAND_SHARE[1]) > 1e-9:
        p.append(f"standing_share curriculum {cur}")
    ev = env.get("events") or {}
    f = str((ev.get("push_robot") or {}).get("func", ""))
    if "moving_only" in f or "reanchor" in f or "push_by_setting_velocity" not in f:
        p.append(f"push_robot func {f}")
    f = str((ev.get("hand_forces") or {}).get("func", ""))
    if not f.endswith("hand_forces"):
        p.append(f"hand_forces func {f}")
    term = rw.get("stand_still") or {}
    tp = term.get("params") or {}
    if not str(term.get("func", "")).endswith("stand_still_joint_deviation_l1"):
        p.append(f"stand_still func {term.get('func')}")
    if term.get("weight") is None or abs(float(term["weight"]) - STAND_STILL_WEIGHT) > 1e-9:
        p.append(f"stand_still weight {term.get('weight')}")
    if tp.get("command_threshold") is None or abs(float(tp["command_threshold"]) - STILL_COMMAND_THRESHOLD) > 1e-9:
        p.append(f"stand_still command_threshold {tp.get('command_threshold')}")
    if list((tp.get("asset_cfg") or {}).get("joint_names") or []) != list(W.LEG_JOINTS):
        p.append(f"stand_still joints {(tp.get('asset_cfg') or {}).get('joint_names')}")
    for name in ("stand_anchor", "stand_tilt", "stand_anchor_bonus", "stand_tilt_bonus"):
        if rw.get(name) is not None:
            p.append(f"{name} present")
    return p


def stamp(deploy: Path, env_yaml: Path, task: str = TASK_ID) -> str:
    from omegaconf import OmegaConf
    if task not in TASK_IDS:
        raise ValueError(f"unknown Waiter WBC task {task!r}; expected one of {TASK_IDS}")
    deploy, env_yaml = Path(deploy), Path(env_yaml)
    env = G.load_env_yaml(env_yaml)
    funcs = ("gait_clock", "gait_clock_stance") \
        if task in (STANCE_TASK_ID, STAND_TASK_ID, STANDPUSH_TASK_ID, STANDBONUS_TASK_ID, STANDSTILL_TASK_ID) \
        else ("gait_clock",)
    info = G.check_env(env, "R1", clock_funcs=funcs)  # gait_clock last in actor and critic, period 0.8, offset 0
    cfg = OmegaConf.load(deploy)
    problems = check_layout(cfg, env)
    if task in (CURRICULUM_TASK_ID, STANCE_TASK_ID, STAND_TASK_ID, STANDPUSH_TASK_ID, STANDBONUS_TASK_ID,
                STANDSTILL_TASK_ID):
        problems += check_curriculum(env)
    if task == STANCE_TASK_ID:
        problems += check_stance(env)
    if task == STAND_TASK_ID:
        problems += check_stand(env)
    if task == STANDPUSH_TASK_ID:
        problems += check_stand_push(env)
    if task == STANDBONUS_TASK_ID:
        problems += check_stand_bonus(env)
    if task == STANDSTILL_TASK_ID:
        problems += check_stand_still(env, window=False)  # the launcher checks the window on full runs
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
             f"  task: {task}\n"
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

    def load_policy(self) -> None:
        """As upstream RlController.load_policy (run_eval calls it): ONNX or TorchScript by file name."""
        from berkeley_humanoid_lite_lowlevel.policy.rl_controller import OnnxPolicy, TorchPolicy
        path = str(self.cfg.policy_checkpoint_path)
        if ".onnx" in path:
            self.policy = OnnxPolicy(path)
        elif ".pt" in path:
            self.policy = TorchPolicy(path)
        else:
            raise ValueError(f"unrecognized policy format: {path}")

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
    s.add_argument("--task", default=TASK_ID, choices=TASK_IDS)
    a = ap.parse_args(argv)
    try:
        print("WAITER-WBC:", stamp(a.deploy, a.env_yaml, a.task))
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"WAITER-WBC: STAMP FAILED ({e})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
