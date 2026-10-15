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
STANDONLY_TASK_ID = "Velocity-BHL-Waiter-WBC-StandOnly-v0"       # phase 1i (2026-10-13): the standing half of a pair
STANDDRIFT_TASK_ID = "Velocity-BHL-Waiter-WBC-StandDrift-v0"     # phase 1l (2026-10-15): standing half with a drift input
TASK_IDS = (TASK_ID, CURRICULUM_TASK_ID, STANCE_TASK_ID, STAND_TASK_ID, STANDPUSH_TASK_ID, STANDBONUS_TASK_ID,
            STANDSTILL_TASK_ID, STANDONLY_TASK_ID, STANDDRIFT_TASK_ID)
STAND_HALF_TASKS = (STANDONLY_TASK_ID, STANDDRIFT_TASK_ID)
DRIFT_OBS = 3                            # phase 1l: [dx, dy, dyaw] inserted just before the gait clock
DRIFT_NOISE = 0.2                        # waiter_stand_mdp.DRIFT_NOISE
STAND_KEY = "waiter_stand_policy"        # phase 1i: the block a dual deploy.yaml adds (DualWaiterController)
STANDONLY_REL_STANDING_ENVS = 1.0
STANCE_PHASE = 0.5                       # gait_clock_mdp.STANCE_PHASE: the phase standing envs are trained at
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


def check_layout(cfg, env: dict, drift: bool = False) -> list[str]:
    """Problems with an exported deploy config + env.yaml against the frozen phase 1 layout ([] = none)."""
    p = []
    obs = env["observations"]
    pol_terms, cri_terms, n_obs = POLICY_TERMS, CRITIC_TERMS, OBS_POLICY
    if drift:                                    # phase 1l: the drift input sits just before the gait clock
        pol_terms, cri_terms, n_obs = (POLICY_TERMS[:-1] + ["stand_drift", "gait_clock"],
                                       CRITIC_TERMS[:-1] + ["stand_drift", "gait_clock"], OBS_POLICY + DRIFT_OBS)
    if G._terms(obs["policy"]) != pol_terms:
        p.append(f"actor terms {G._terms(obs['policy'])}")
    if G._terms(obs["critic"]) != cri_terms:
        p.append(f"critic terms {G._terms(obs['critic'])}")
    if int(cfg.num_joints) != N_JOINTS or list(cfg.joints) != W.JOINT_ORDER:
        p.append(f"joints {cfg.num_joints} {list(cfg.joints)}")
    if int(cfg.num_actions) != N_ACTIONS or list(cfg.action_indices) != LEG_INDICES:
        p.append(f"actions {cfg.num_actions} {list(cfg.action_indices)}")
    if int(cfg.num_observations) != n_obs or int(cfg.history_length) != 0:
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


def check_stand_only(env: dict) -> list[str]:
    """Phase 1i only: every env standing (rel_standing_envs 1.0) with the held-clock stance terms; pushes with
    re-anchoring and unrestricted hand forces on every env; 1g's two bounded standing rewards and 1h's stand-still
    term with their weights; no standing penalty and no standing-share curriculum."""
    p = []
    rel = ((env.get("commands") or {}).get("base_velocity") or {}).get("rel_standing_envs")
    if rel is None or abs(float(rel) - STANDONLY_REL_STANDING_ENVS) > 1e-9:
        p.append(f"rel_standing_envs {rel}")
    for group in ("policy", "critic"):
        f = str(((env["observations"].get(group) or {}).get("gait_clock") or {}).get("func", ""))
        if "gait_clock_stance" not in f:
            p.append(f"{group} gait_clock func {f}")
    rw = env.get("rewards") or {}
    if "feet_gait_stance" not in str((rw.get("feet_gait") or {}).get("func", "")):
        p.append(f"feet_gait func {(rw.get('feet_gait') or {}).get('func')}")
    ev = env.get("events") or {}
    f = str((ev.get("push_robot") or {}).get("func", ""))
    if "push_and_reanchor" not in f:
        p.append(f"push_robot func {f}")
    f = str((ev.get("hand_forces") or {}).get("func", ""))
    if not f.endswith("hand_forces"):
        p.append(f"hand_forces func {f}")
    for name, fn, w in (("stand_anchor_bonus", "StandAnchorBonus", STAND_ANCHOR_BONUS_WEIGHT),
                        ("stand_tilt_bonus", "stand_tilt_bonus", STAND_TILT_BONUS_WEIGHT),
                        ("stand_still", "stand_still_joint_deviation_l1", STAND_STILL_WEIGHT)):
        term = rw.get(name) or {}
        if not str(term.get("func", "")).endswith(fn):
            p.append(f"{name} func {term.get('func')}")
        if term.get("weight") is None or abs(float(term["weight"]) - w) > 1e-9:
            p.append(f"{name} weight {term.get('weight')}")
    tp = (rw.get("stand_still") or {}).get("params") or {}
    if list((tp.get("asset_cfg") or {}).get("joint_names") or []) != list(W.LEG_JOINTS):
        p.append(f"stand_still joints {(tp.get('asset_cfg') or {}).get('joint_names')}")
    for name in ("stand_anchor", "stand_tilt"):
        if rw.get(name) is not None:
            p.append(f"{name} penalty present")
    if ((env.get("curriculum") or {}).get("standing_share")) is not None:
        p.append("standing_share curriculum present")
    return p


def check_stand_drift(env: dict) -> list[str]:
    """Phase 1l only: 1i's standing-only checks plus the drift input in the actor and the critic (the anchor reward's
    drift, just before the gait clock), with +-DRIFT_NOISE uniform noise on the actor's copy only."""
    p = check_stand_only(env)
    for group, noisy in (("policy", True), ("critic", False)):
        term = (env["observations"].get(group) or {}).get("stand_drift") or {}
        if not str(term.get("func", "")).endswith("stand_drift"):
            p.append(f"{group} stand_drift func {term.get('func')}")
            continue
        if (term.get("params") or {}).get("reward_term") != "stand_anchor_bonus":
            p.append(f"{group} stand_drift reward_term {(term.get('params') or {}).get('reward_term')}")
        nz = term.get("noise") or {}
        ok = (nz.get("n_min") is not None and abs(float(nz["n_min"]) + DRIFT_NOISE) < 1e-9 and
              abs(float(nz.get("n_max", 0.0)) - DRIFT_NOISE) < 1e-9) if noisy else not nz
        if not ok:
            p.append(f"{group} stand_drift noise {nz}")
    return p


DUAL_MATCH_KEYS = ("joints", "num_joints", "num_actions", "action_indices", "default_joint_positions", "action_scale",
                   "num_observations", "history_length", "policy_dt", "action_limit_lower", "action_limit_upper")


def dual_problems(walk_cfg, stand_cfg) -> list[str]:
    """Why two stamped Waiter WBC deploy configs cannot share one controller ([] = they can)."""
    from omegaconf import OmegaConf
    p = []
    for name, c in (("walk", walk_cfg), ("stand", stand_cfg)):
        if KEY not in c or not G.has_clock(c):
            p.append(f"{name}: not a stamped Waiter WBC deploy config")
    if p:
        return p
    if STAND_KEY in walk_cfg or STAND_KEY in stand_cfg:
        p.append("a dual deploy config cannot be one half of another")
    stand_task = str(stand_cfg[KEY].get("task"))
    for k in DUAL_MATCH_KEYS:
        a = OmegaConf.to_container(walk_cfg[k]) if OmegaConf.is_config(walk_cfg.get(k)) else walk_cfg.get(k)
        b = OmegaConf.to_container(stand_cfg[k]) if OmegaConf.is_config(stand_cfg.get(k)) else stand_cfg.get(k)
        if k == "num_observations" and stand_task == STANDDRIFT_TASK_ID:
            a = int(a) + DRIFT_OBS                   # phase 1l: the standing half also reads the drift input
        if a != b:
            p.append(f"{k} differs")
    for k in ("period_s", "phase_offset"):
        if walk_cfg[G.CLOCK_KEY][k] != stand_cfg[G.CLOCK_KEY][k]:
            p.append(f"gait_clock {k} differs")
    for k in ("upper_body_indices", "gripper_closed_rad"):
        a, b = walk_cfg[KEY][k], stand_cfg[KEY][k]
        if k.endswith("indices"):
            a, b = list(a), list(b)
        if a != b:
            p.append(f"{KEY}.{k} differs")
    if stand_task not in STAND_HALF_TASKS:
        p.append(f"the standing half is task {stand_task}, not one of {STAND_HALF_TASKS}")
    return p


def stamp_dual(walk: Path, stand: Path, out: Path) -> str:
    """Write a dual deploy config: the walking config's text unchanged + a STAND_KEY block naming the standing policy.
    Refuses incompatible halves; never overwrites a different file."""
    from omegaconf import OmegaConf
    walk, stand, out = Path(walk), Path(stand), Path(out)
    wc, sc = OmegaConf.load(walk), OmegaConf.load(stand)
    problems = dual_problems(wc, sc)
    if problems:
        raise ValueError("cannot pair these deploy configs: " + "; ".join(problems))
    text = walk.read_text()
    block = (f"{'' if text.endswith(chr(10)) else chr(10)}{STAND_KEY}:\n"
             f"  checkpoint: {sc.policy_checkpoint_path}\n"
             f"  deploy: {stand}\n"
             f"  task: {sc[KEY]['task']}\n"
             f"  switch: clock_held\n"
             f"  stance_phase: {STANCE_PHASE}\n"
             f"  drift_noise: {DRIFT_NOISE}\n"
             f"  walk_deploy: {walk}\n")
    if out.exists():
        if out.read_text() == text + block:
            return f"already written: {out}"
        raise ValueError(f"{out} exists with different content; not overwritten")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + block)
    back = OmegaConf.load(out)
    assert STAND_KEY in back and KEY in back and G.has_clock(back)
    return f"dual deploy written: {out} (walk {walk.name}, stand {stand.name})"


def stamp(deploy: Path, env_yaml: Path, task: str = TASK_ID) -> str:
    from omegaconf import OmegaConf
    if task not in TASK_IDS:
        raise ValueError(f"unknown Waiter WBC task {task!r}; expected one of {TASK_IDS}")
    deploy, env_yaml = Path(deploy), Path(env_yaml)
    env = G.load_env_yaml(env_yaml)
    funcs = ("gait_clock", "gait_clock_stance") \
        if task in (STANCE_TASK_ID, STAND_TASK_ID, STANDPUSH_TASK_ID, STANDBONUS_TASK_ID, STANDSTILL_TASK_ID,
                    STANDONLY_TASK_ID, STANDDRIFT_TASK_ID) else ("gait_clock",)
    info = G.check_env(env, "R1", clock_funcs=funcs)  # gait_clock last in actor and critic, period 0.8, offset 0
    cfg = OmegaConf.load(deploy)
    problems = check_layout(cfg, env, drift=(task == STANDDRIFT_TASK_ID))
    if task in (CURRICULUM_TASK_ID, STANCE_TASK_ID, STAND_TASK_ID, STANDPUSH_TASK_ID, STANDBONUS_TASK_ID,
                STANDSTILL_TASK_ID, STANDONLY_TASK_ID, STANDDRIFT_TASK_ID):
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
    if task == STANDONLY_TASK_ID:
        problems += check_stand_only(env)
    if task == STANDDRIFT_TASK_ID:
        problems += check_stand_drift(env)
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


class _StandOnnx:
    """onnxruntime on CPU, one thread (as the gates' CpuPolicy)."""

    def __init__(self, path):
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = 1
        o.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])
        self.key = self.session.get_inputs()[0].name

    def forward(self, observation):
        return self.session.run(None, {self.key: observation})[0]


class _DriftInput:
    """Phase 1l: the standing policy reads the 83 inputs of the walking layout plus the drift since its anchor, inserted
    just before the gait clock (86); `feat` is set by DualWaiterController before each standing update."""

    def __init__(self, policy):
        self.policy, self.feat = policy, None

    def forward(self, observation):
        import numpy as np
        o = np.asarray(observation, dtype=np.float32)
        return self.policy.forward(np.concatenate([o[:, :-2], np.asarray(self.feat, np.float32).reshape(1, -1),
                                                   o[:, -2:]], axis=1).astype(np.float32))


def _stand_mdp():
    """tasks/waiter_stand_mdp.py loaded by path: importing it through bhl_robust.tasks would run that package's
    __init__, which registers the Isaac tasks and boots the Isaac Sim kernel. The module itself imports only math at
    module scope, so the MuJoCo deploy side can use its pure helpers (drift_features)."""
    import importlib.util
    name = "bhl_robust_waiter_stand_mdp_pure"
    mod = sys.modules.get(name)
    if mod is None:
        path = Path(__file__).resolve().parents[1] / "tasks" / "waiter_stand_mdp.py"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[name] = mod
    return mod


class DualWaiterController(WaiterWbcController):
    """Phase 1i: the walking WBC and a standing-only WBC behind one switch (deploy config stamped by `stamp_dual`).

    `.policy` is the walking policy (cfg.policy_checkpoint_path), assigned by the caller exactly as for
    WaiterWbcController; the standing policy (cfg.waiter_stand_policy.checkpoint) is loaded on first use, or injected
    as `.stand_policy`. The standing policy acts on an update iff the gait clock did not advance since the previous
    update, i.e. the env or a gate is holding it (WaiterEnv's stance hold; the Q3/Q4 gates); the first held update
    still goes to the walking policy (one 0.04 s step of delay). While standing, the policy sees what it was trained on:
    a zero command and the clock at the training stance phase (STANCE_PHASE, 0.5), whatever step is held. The clock
    itself is left as the caller set it. `standing_steps` counts the updates the standing policy made.

    Phase 1l (a standing half of task STANDDRIFT_TASK_ID): the standing policy also reads its drift since the anchor,
    the base pose at its first standing update (waiter_stand_mdp.drift_features, ORACLE: from the simulator's base pose,
    which the runner hands over through `attach_sim`), plus uniform noise of +-drift_noise as in training. The anchor
    is reset whenever standing begins again. `drift_log` keeps the first inputs of each standing period."""

    def __init__(self, cfg):
        import numpy as np
        super().__init__(cfg)
        if STAND_KEY not in cfg:
            raise ValueError("not a dual Waiter WBC deploy config")
        self.stand_checkpoint = str(cfg[STAND_KEY]["checkpoint"])
        self.stance_phase = float(cfg[STAND_KEY].get("stance_phase", STANCE_PHASE))
        self.stance_step = int(round(self.stance_phase * self.clock_period / self.clock_step_dt))
        self.stand_task = str(cfg[STAND_KEY].get("task", STANDONLY_TASK_ID))
        self.drift = self.stand_task == STANDDRIFT_TASK_ID
        self.drift_noise = float(cfg[STAND_KEY].get("drift_noise", DRIFT_NOISE))
        self.stand_policy = None
        self.standing = False
        self.standing_steps = 0
        self._prev_clock = None
        self._sim = None
        self._anchor = None
        self._rng = np.random.default_rng(0)
        self.drift_log = []

    def attach_sim(self, data, slot) -> None:
        """Called by the MuJoCo runner (eval/multi_robot.MultiRunner) with its MjData and this robot's slot."""
        self._sim = (data, int(slot.qpos_adr))

    def reset_phase(self) -> None:
        import numpy as np
        super().reset_phase()
        self._prev_clock = None
        self.standing = False
        self._anchor = None
        self._rng = np.random.default_rng(0)
        self.drift_log = []

    def _base_pose(self):
        import math
        if self._sim is None:
            raise RuntimeError("the drift input needs the simulator's base pose: no runner called attach_sim")
        d, a = self._sim
        w, x, y, z = (float(v) for v in d.qpos[a + 3:a + 7])
        return d.qpos[a:a + 2].copy(), math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def update(self, robot_observations):
        import numpy as np
        if not self.policy_observations.any():                # a harness reset (MultiRunner.reset): a new episode
            self.standing, self._anchor = False, None
        held = self._prev_clock is not None and self.clock_step == self._prev_clock
        starting = held and not self.standing
        if not held:
            self.standing = False
            targets = super().update(robot_observations)
            self._prev_clock = self.clock_step - 1               # the step this update used
            return targets
        if self.stand_policy is None:
            self.stand_policy = _StandOnnx(self.stand_checkpoint)
        policy = self.stand_policy
        if self.drift:
            drift_features = _stand_mdp().drift_features
            xy, yaw = self._base_pose()
            if starting or self._anchor is None:
                self._anchor = (xy.copy(), yaw)
            clean = drift_features(xy, np.float64(yaw), self._anchor[0], np.float64(self._anchor[1]))
            fed = clean + self._rng.uniform(-self.drift_noise, self.drift_noise, size=3)
            if starting or len(self.drift_log) < 5:
                self.drift_log.append({"starting": bool(starting), "clean": [float(v) for v in clean],
                                       "fed": [float(v) for v in fed]})
            policy = _DriftInput(self.stand_policy)
            policy.feat = fed
        self.standing = True                                     # after the drift input: a failed update changes nothing
        obs = np.array(robot_observations, dtype=np.float64, copy=True)
        n = self.n_j
        obs[7 + 2 * n + 1:7 + 2 * n + 4] = 0.0                   # the standing policy is trained on a zero command
        step, walk = self.clock_step, self.policy
        self.clock_step, self.policy = self.stance_step, policy
        try:
            targets = super().update(obs)
        finally:
            self.policy, self.clock_step = walk, step + 1
        self._prev_clock = step
        self.standing_steps += 1
        return targets


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stamp")
    s.add_argument("--deploy", type=Path, required=True)
    s.add_argument("--env-yaml", type=Path, required=True)
    s.add_argument("--task", default=TASK_ID, choices=TASK_IDS)
    d = sub.add_parser("dual", help="phase 1i: pair a walking and a standing-only deploy config")
    d.add_argument("--walk", type=Path, required=True)
    d.add_argument("--stand", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "dual":
            print("WAITER-WBC:", stamp_dual(a.walk, a.stand, a.out))
        else:
            print("WAITER-WBC:", stamp(a.deploy, a.env_yaml, a.task))
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"WAITER-WBC: {'DUAL' if a.cmd == 'dual' else 'STAMP'} FAILED ({e})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
