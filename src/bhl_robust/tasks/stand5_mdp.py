"""Observation, curriculum, rule and smoke terms for CubeToShelfStand5 (`task_v2_env_cfg`).

CubeToShelfStand5 is CubeToShelfStand4 plus exactly two changes, frozen by the
user on 2026-10-03 (SLURM_JOBS.md, "User approval recorded 2026-10-03 09:55",
item (S); `DESIGN_S` below quotes it verbatim). It is a different
configuration from Stand4, reported as Stand5 and judged only by its own
predeclared rule; like Stand4 it is never compared with CubeToShelfStand3 or
CubeToShelf numbers.

Context (not a reason the design states): Stand4 (`21506758`) was NEGATIVE by
its rule (last-200 success 0.0064 / 0.0009; the crews stand but neither lift
the cube clear nor seat it).

What changes from Stand4 -- these two and nothing else
------------------------------------------------------
(i) Observation, ORACLE like `object_pos_a/b`. `object_zaxis_a` /
    `object_zaxis_b`: the cube's own z axis expressed in robot_a's / robot_b's
    root frame, 3 numbers each, z_r = R_robot^T (R_cube e_z), from the
    simulator's root quaternions read through `quat_order.unpack_wxyz` (the
    2976f36 helpers; (x, y, z, w) on v60). Appended LAST to BOTH the actor's
    group (`policy`) and the critic's (`critic`): policy 194 -> 200, critic
    206 -> 212 (Stand4's widths, from its smoke `21506757`). Like
    `object_pos_a/b`: no noise, no scale. Clipped to +/-OBS_CLIP (100) like
    every observation term since Stand3: Stand3's `_clip_obs_terms` runs in its
    `__post_init__`, before these terms exist, so the clip is set on them
    explicitly; every component is in [-1, 1], so the clip never binds.
    Only the z axis, as the design says; it carries the tilt that Stand4's lift
    (<= 15 deg) and success (<= 8 deg) clauses read, not the yaw about the
    cube's own z axis, which no clause reads.
(ii) Lift curriculum. `lift_height` keeps its name, its place in the
    curriculum manager, its parameters (step 0.02, 0.04 -> 0.06 cap, target
    0.35, demotion below half the target, term `lifting_object`) and its write
    into `lifting_object`'s `minimal_height`. Its competence measure -- the
    share of the resetting envs that count -- becomes Stand4's ROLL-PROOF lift
    condition, `stand4_mdp.roll_proof_mask`: the cube's lowest corner >= 0.02 m
    above every support under it (the floor; plinth / deck tops under its xy
    footprint) AND tilt <= 15 deg (own z axis vs world up), INSTEAD of
    centre > spawn + level AND pinch distance < 0.20 m. The pinch no longer
    enters the curriculum (the design replaces "centre height + pinch"); it
    still enters the lift pay. `stand5_lift_height_curriculum` is
    `coop_lift_mdp.lift_height_curriculum` line for line with that one test
    changed; its wall-clock branch (`clock_lift_height`, False in Stand4 and
    Stand5) is delegated to the original unchanged.

NOT changed: rewards (`lifting_object` still pays centre > spawn +
minimal_height x roll-proof gate x pinch kernel x upright gate), success
(Stand4's termination: seated, tilt <= 8 deg, released, held 12 steps; `placed`
reads it), terminations, scene, spawn, hand colliders, actuators, actions,
events, runner (Stand3's), 8000 iterations, 2 seeds, resources.

Labels: LABELS_NOTE5. Rule: Stand4's, verbatim (PREDECLARED_RULE5,
`stand4_mdp.PREDECLARED_RULE`, AS_APPLIED5); the verdict functions are
Stand4's, called (`evaluate_kill`, `evaluate_seed5`, `seed_result5`) or
relabelled (`pair_verdict5`).

Launchers: slurm/repo20260923/gpu_v2_stand5_smoke.sbatch (+
slurm/inner/inner_v2_stand5_smoke.sh) and slurm/repo20260923/
gpu_v2_stand5_train.sbatch. Verdicts come from JSON written by the pure
functions below, never from exit codes.

Pure kernels and rule functions import without Isaac Lab (tests/test_stand5.py
runs off-simulator; the launchers load this file by path). Inside Isaac the
package's own `stand4_mdp` / `stand_mdp` modules are used (the same objects the
cfg classes use); the env-facing terms reach Isaac Lab only lazily.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import torch


def _load_stand4_mdp():
    """`stand4_mdp`: the package's own module inside Isaac (same object the
    Stand4 cfg uses), else loaded by path (tests, launchers)."""
    mod = sys.modules.get("bhl_robust.tasks.stand4_mdp")
    if mod is None and "bhl_robust.tasks" in sys.modules:
        mod = importlib.import_module("bhl_robust.tasks.stand4_mdp")
    if mod is not None:
        return mod
    path = Path(__file__).with_name("stand4_mdp.py")
    spec = importlib.util.spec_from_file_location("_stand4_mdp_for_stand5", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_quat_order():
    """`bhl_robust.quat_order` (import-safe), else loaded by path (launchers)."""
    mod = sys.modules.get("bhl_robust.quat_order")
    if mod is not None:
        return mod
    try:
        return importlib.import_module("bhl_robust.quat_order")
    except ImportError:
        path = Path(__file__).resolve().parents[1] / "quat_order.py"
        spec = importlib.util.spec_from_file_location("_quat_order_for_stand5", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


s4 = _load_stand4_mdp()
sm = s4.sm
qo = _load_quat_order()

# ---------------------------------------------------------------- constants

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK5_ID = "TaskV2-BHL-CubeToShelfStand5-Blind-v0"
TASK4_ID = "TaskV2-BHL-CubeToShelfStand4-Blind-v0"

#: (i) the observation: term name -> robot whose root frame it is expressed in.
OBS_GROUPS = ("policy", "critic")
OBS_TERM_ROBOTS = (("object_zaxis_a", "robot_a"), ("object_zaxis_b", "robot_b"))
OBS_TERMS5 = tuple(n for n, _ in OBS_TERM_ROBOTS)
OBS_TERM_DIM = 3
#: Stand4's groups as its smoke `21506757` built them (term order and widths).
STAND4_OBS_TERMS = {
    "policy": ("projected_gravity_a", "projected_gravity_b", "base_ang_vel_a", "base_ang_vel_b",
               "joint_pos_a", "joint_pos_b", "joint_vel_a", "joint_vel_b", "object_pos_a",
               "object_pos_b", "track_err_a", "track_err_b", "actions"),
}
STAND4_OBS_TERMS["critic"] = STAND4_OBS_TERMS["policy"] + (
    "base_lin_vel_a", "base_lin_vel_b", "object_lin_vel", "object_ang_vel")
STAND4_OBS_WIDTH = {"policy": 194, "critic": 206}
STAND5_OBS_WIDTH = {g: w + OBS_TERM_DIM * len(OBS_TERMS5) for g, w in STAND4_OBS_WIDTH.items()}

#: (ii) the curriculum: Stand4's parameters plus the roll-proof gate's two.
CURR_CLEARANCE = s4.CORNER_CLEARANCE          # 0.02 m
CURR_TILT_MAX_DEG = s4.LIFT_TILT_MAX_DEG      # 15 deg
CURRICULUM4_PARAMS = {"term_name": "lifting_object", "step": 0.02, "min_height": 0.04,
                      "max_height": sm.STAND3_LIFT_MAX, "success_rate_target": 0.35}
CURRICULUM5_PARAMS = dict(CURRICULUM4_PARAMS, clearance=CURR_CLEARANCE,
                          tilt_max_deg=CURR_TILT_MAX_DEG)
#: Stand4's curriculum read pinch distance < this (coop_lift_mdp, hard-coded).
STAND4_PINCH_PROMOTE = 0.20

FUNC_OBS5 = "bhl_robust.tasks.stand5_mdp:object_zaxis_in_root"
FUNC_CURR5 = "bhl_robust.tasks.stand5_mdp:stand5_lift_height_curriculum"
FUNC_CURR4 = "bhl_robust.tasks.coop_lift_mdp:lift_height_curriculum"

# ------------------------------------------------------ predeclared rules
STAND5_MAX_ITER = s4.STAND4_MAX_ITER         # 8000
COMPLETE_ITER = s4.COMPLETE_ITER             # 7999
RESULT_WINDOW = s4.RESULT_WINDOW             # 200
RESULT_MIN_SUCCESS = s4.RESULT_MIN_SUCCESS   # 0.10
STAND5_SEEDS = s4.STAND4_SEEDS               # (0, 1)

#: The frozen design, verbatim from SLURM_JOBS.md ("User approval recorded
#: 2026-10-03 09:55", item (S)); one entry per line, list markers dropped.
DESIGN_S = (
    "**(S) Stand5.** Stand4 + two changes.",
    "(i) The actor and critic observe the cube's orientation in each robot's root frame (its z "
    "axis; ORACLE, like `object_pos_a/b`).",
    "(ii) The lift curriculum promotes on Stand4's roll-proof lift condition (lowest corner "
    "≥ 2 cm above every support and tilt ≤ 15°) instead of centre height + pinch.",
    "Stand4's rule verbatim: the kill rule at model_1000; COMPLETE at 7999; PASS iff ≥ 1 of 2 "
    "seeds is COMPLETE, not killed, with last-200 success ≥ 0.10. 2 seeds × 8000 "
    "iterations, Stand4's resources.",
    "Labels: LEARNED crew policy (blind) + ORACLE privileged observations (now including cube "
    "orientation).",
)
#: The rule line of the frozen design (verbatim).
PREDECLARED_RULE5 = DESIGN_S[3]
#: Stand4's rule, quoted verbatim (it is the rule Stand5 applies).
STAND4_RULE = s4.PREDECLARED_RULE
PRECEDENCE_NOTE = s4.PRECEDENCE_NOTE
COMPLETE_NOTE = s4.COMPLETE_NOTE
SUCCESS_NOTE = s4.SUCCESS4_NOTE
TILT_NOTE = s4.TILT_NOTE
AS_APPLIED5 = (
    "As applied to Stand5 (Stand4's clauses, unchanged; Stand4's functions, called): kill = "
    "Stand3's kill rule at model_1000 (stand_mdp.evaluate_kill3: iterations 901-1000, mean "
    "episode length < 100 OR mean (time_out + success) < 0.10, the length clause skipped when "
    "mean success >= 0.10), stopped by the same watcher as Stand3 and Stand4; " + COMPLETE_NOTE
    + "; success = Stand4's termination, unchanged in Stand5 (" + SUCCESS_NOTE + "); PASS iff "
    ">= 1 of the 2 seeds is COMPLETE, not killed, with last-200-iteration success >= 0.10; "
    "precedence: " + PRECEDENCE_NOTE)
LABELS_NOTE5 = (
    "LEARNED crew policy (one PPO actor drives both robots, blind); ORACLE privileged "
    "observations, now including cube orientation: the actor and the critic both observe "
    "object_pos_a/b (the simulator cube position in each robot's root frame) and object_zaxis_a/b "
    "(NEW: the simulator cube's own z axis in each robot's root frame); the critic also "
    "base_lin_vel_a/b, object_lin_vel, object_ang_vel; MODIFIED hand colliders (the overlay) "
    "relative to Stand3, inherited unchanged from Stand4")
TASK5_NOTE = (
    "CubeToShelfStand5: CubeToShelfStand4 plus two changes (the cube's z axis observed by the "
    "actor and the critic; the lift curriculum promotes on the roll-proof lift condition); a "
    "different configuration from Stand4, judged only by its own predeclared rule; never "
    "compared with CubeToShelfStand3 or CubeToShelf numbers")
PASS_LABEL5 = "PASS: learned roll-proof placement (Stand5)"

# ------------------------------------------------------------ pure kernels


def quat_matrix(w, x, y, z):
    """Rows of the rotation matrix (world-from-body) of (w, x, y, z) columns,
    normalised first (the same normalisation as Stand4's tilt)."""
    w, x, y, z = s4._normalised(w, x, y, z)
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


def body_zaxis_world(w, x, y, z) -> torch.Tensor:
    """(N, 3): a body's own z axis in the world (third column of its matrix)."""
    r = quat_matrix(w, x, y, z)
    return torch.stack((r[0][2], r[1][2], r[2][2]), dim=-1)


def zaxis_in_frame(frame_wxyz, body_wxyz) -> torch.Tensor:
    """(N, 3): the body's own z axis expressed in `frame`, R_frame^T (R_body e_z).

    Both arguments are (w, x, y, z) column tuples (`quat_order.unpack_wxyz`).
    The vector has unit length; for an identity frame it is `body_zaxis_world`.
    """
    v = body_zaxis_world(*body_wxyz)
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    r = quat_matrix(*frame_wxyz)
    return torch.stack((r[0][0] * vx + r[1][0] * vy + r[2][0] * vz,
                        r[0][1] * vx + r[1][1] * vy + r[2][1] * vz,
                        r[0][2] * vx + r[1][2] * vy + r[2][2] * vz), dim=-1)


def quat_axis_angle(axis, deg: float) -> tuple:
    """(w, x, y, z) of a rotation by `deg` about `axis` (world frame). Written to
    the simulator only through `quat_order.native_quat`."""
    n = math.sqrt(sum(a * a for a in axis))
    h = math.radians(deg) / 2.0
    return (math.cos(h),) + tuple(math.sin(h) * a / n for a in axis)


def curriculum_decision(level: float, competence: float, step: float, min_height: float,
                        max_height: float, success_rate_target: float) -> float:
    """`lift_height_curriculum`'s level update (coop_lift_mdp, verbatim
    arithmetic): promote above the target, demote below half of it, else hold.
    A NaN competence holds."""
    if competence > success_rate_target:
        return min(max_height, level + step)
    if competence < 0.5 * success_rate_target:
        return max(min_height, level - step)
    return level


# ------------------------------------------------------- env-facing terms


def _t(v):
    return s4._t(v)


def object_zaxis_in_root(env, robot_cfg, object_name: str = "object") -> torch.Tensor:
    """ORACLE observation: the cube's own z axis in `robot_cfg`'s root frame, (N, 3).

    Same frame as `coop_lift_mdp.object_pos_in_root` (the robot's
    `root_pos_w` / `root_quat_w`); both quaternions are read through
    `quat_order.unpack_wxyz`, so the stack's storage order never matters.
    """
    from bhl_robust.quat_order import unpack_wxyz
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_name]
    return zaxis_in_frame(unpack_wxyz(_t(robot.data.root_quat_w)),
                          unpack_wxyz(_t(obj.data.root_quat_w)))


def stand5_lift_height_curriculum(env, env_ids: Sequence[int], term_name: str = "lifting_object",
                                  step: float = 0.02, min_height: float = 0.04,
                                  max_height: float = 0.22, success_rate_target: float = 0.35,
                                  clearance: float = CURR_CLEARANCE,
                                  tilt_max_deg: float = CURR_TILT_MAX_DEG) -> float:
    """`coop_lift_mdp.lift_height_curriculum` with ONE change: an env counts as
    competent iff its cube meets Stand4's roll-proof lift condition
    (`stand4_mdp.roll_proof_mask`: lowest corner >= `clearance` above every
    support under it AND tilt <= `tilt_max_deg`), instead of centre > spawn +
    level AND pinch distance < 0.20 m. Same defaults, same env_ids handling,
    same update, same write into the reward term's `minimal_height`.
    """
    if getattr(env.cfg, "clock_lift_height", False):
        # the wall-clock ramp ignores competence: the original, unchanged
        return sm._coop().lift_height_curriculum(
            env, env_ids, term_name=term_name, step=step, min_height=min_height,
            max_height=max_height, success_rate_target=success_rate_target)
    height = getattr(env, "_bhl_lift_h", min_height)
    p, _ = sm._obj_state(env)                     # env-local centre (the supports are env-local)
    w, x, y, z = s4._cube_quat_wxyz(env)
    if env_ids is None or len(env_ids) == 0:
        idx = slice(None)
    else:
        idx = env_ids
    roll_ok = s4.roll_proof_mask(p[idx], w[idx], x[idx], y[idx], z[idx],
                                 clearance=clearance, tilt_max_deg=tilt_max_deg)
    success = float(roll_ok.float().mean())
    if success > success_rate_target:
        height = min(max_height, height + step)
    elif success < 0.5 * success_rate_target:
        height = max(min_height, height - step)
    env._bhl_lift_h = height

    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.params["minimal_height"] = height
    env.reward_manager.set_term_cfg(term_name, term_cfg)
    return height


# --------------------------------------------------------- rule evaluation
# Stand4's functions, called: the rule is Stand4's verbatim.


def kill_verdict(ep_len: float, time_out: float, success: float) -> dict:
    """Stand3's `kill_verdict3` (through Stand4's `kill_verdict`)."""
    return s4.kill_verdict(ep_len, time_out, success)


def evaluate_kill(scalars: dict) -> dict:
    """Stand3's `evaluate_kill3`: model_1000, iterations 901-1000."""
    return s4.evaluate_kill(scalars)


def seed_result5(last_iter: int, success: float, killed: bool = False, bad_config: bool = False,
                 window_finite: bool = True, has_value: bool = True) -> dict:
    """Stand4's `seed_result4` (COMPLETE = 7999 + Stand3's completeness check)."""
    return s4.seed_result4(last_iter, success, killed=killed, bad_config=bad_config,
                           window_finite=window_finite, has_value=has_value)


def evaluate_seed5(scalars: dict, killed: bool = False, bad_config: bool = False) -> dict:
    """Stand4's `evaluate_seed4` (same tags, window, completeness and diagnostics)."""
    return s4.evaluate_seed4(scalars, killed=killed, bad_config=bad_config)


def pair_verdict5(seeds: Sequence[dict]) -> str:
    """Stand4's `pair_verdict4` (PRECEDENCE_NOTE), with Stand5's PASS label."""
    v = s4.pair_verdict4(seeds)
    if v.startswith(s4.PASS_LABEL):
        return PASS_LABEL5 + v[len(s4.PASS_LABEL):]
    return v


# ------------------------------------------- Isaac Lab dumps (params/*.yaml)


def load_dump(text: str):
    """An Isaac Lab `dump_yaml` text as plain data. `!!python/...` tags (tuple,
    object/apply:builtins.slice, ...) become {"!python/<suffix>": value}:
    nothing is constructed or executed (a SafeLoader subclass)."""
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    def _python(loader, suffix, node):
        if isinstance(node, yaml.SequenceNode):
            val = loader.construct_sequence(node, deep=True)
        elif isinstance(node, yaml.MappingNode):
            val = loader.construct_mapping(node, deep=True)
        else:
            val = loader.construct_scalar(node)
        return {"!python/" + suffix: val}

    _Loader.add_multi_constructor("tag:yaml.org,2002:python/", _python)
    return yaml.load(text, Loader=_Loader)   # noqa: S506 -- SafeLoader subclass


def flatten_dump(d, pre: str = "") -> dict:
    """Dotted keys -> repr(value); an empty mapping is a leaf so it still shows
    (the smoke probe flattens `cfg.to_dict()` the same way)."""
    out = {}
    if isinstance(d, dict):
        if not d and pre:
            out[pre] = "{}"
        for k, v in d.items():
            out.update(flatten_dump(v, f"{pre}.{k}" if pre else str(k)))
    else:
        out[pre] = repr(d)
    return out


def diff_keys(a: dict, b: dict) -> list:
    """Flattened keys whose values differ (or exist on one side only)."""
    fa, fb = flatten_dump(a), flatten_dump(b)
    return sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))


def diff_check(keys: Sequence[str], allowed: Sequence[str]) -> dict:
    """Every allowed entry differs (an entry covers its subtree) and nothing else."""
    def covers(entry, k):
        return k == entry or k.startswith(entry + ".")
    unexpected = sorted(k for k in keys if not any(covers(e, k) for e in allowed))
    missing = [e for e in allowed if not any(covers(e, k) for k in keys)]
    return {"ok": not unexpected and not missing, "unexpected": unexpected, "missing": missing}


#: Stand4 -> Stand5 differences allowed, in cfg.to_dict() and in the dumped
#: params/env.yaml alike (an entry covers its subtree).
CFG_DIFF_ALLOWED5 = tuple(f"observations.{g}.{n}" for g in OBS_GROUPS for n in OBS_TERMS5) + (
    "curriculum.lift_height.func", "curriculum.lift_height.params.clearance",
    "curriculum.lift_height.params.tilt_max_deg")
#: params/agent.yaml: Stand4's smoke vs Stand5's smoke (same seed and budget).
AGENT_DIFF_ALLOWED5 = ("run_name",)


def dump_diff(env5: str, env4: str, agent5: str, agent4: str) -> dict:
    """Stand5's dumped params vs Stand4's smoke dump: declared keys only."""
    ek = diff_keys(load_dump(env4), load_dump(env5))
    ak = diff_keys(load_dump(agent4), load_dump(agent5))
    env = diff_check(ek, CFG_DIFF_ALLOWED5)
    agent = diff_check(ak, AGENT_DIFF_ALLOWED5)
    return {"ok": env["ok"] and agent["ok"], "env": dict(env, keys=ek), "agent": dict(agent, keys=ak)}


def _group_terms(group: dict) -> list:
    """Observation term names of a dumped group, in order (mappings with a func)."""
    return [k for k, v in group.items() if isinstance(v, dict) and "func" in v]


def _clip_pair(v):
    """A dumped clip ({"!python/tuple": [lo, hi]} or a list) as a tuple of floats."""
    if isinstance(v, dict) and "!python/tuple" in v:
        v = v["!python/tuple"]
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return tuple(float(x) for x in v)
    return None


def config_check5(agent_yaml: str, env_yaml: str) -> dict:
    """Did this run train CubeToShelfStand5 with Stand3's runner? Stand4's
    `config_check` (every clause) plus Stand5's: both observation groups are
    Stand4's terms followed by object_zaxis_a/b (this module's function,
    robot_a/b, no noise, no scale, clip +/-OBS_CLIP), and `lift_height` is this
    module's curriculum with Stand4's parameters plus clearance 0.02 and
    tilt_max_deg 15.0. {"ok", "failed", "checks"}."""
    r4 = s4.config_check(agent_yaml, env_yaml)
    c = dict(r4["checks"])
    try:
        env = load_dump(env_yaml) or {}
    except Exception:                                               # noqa: BLE001
        env = {}
    obs = env.get("observations") or {}
    for g in OBS_GROUPS:
        grp = obs.get(g) or {}
        terms = _group_terms(grp) if isinstance(grp, dict) else []
        ok = terms == list(STAND4_OBS_TERMS[g]) + list(OBS_TERMS5)
        for name, robot in OBS_TERM_ROBOTS:
            t = grp.get(name) if isinstance(grp, dict) else None
            t = t if isinstance(t, dict) else {}
            rc = (t.get("params") or {}).get("robot_cfg") or {}
            ok = ok and (t.get("func") == FUNC_OBS5 and isinstance(rc, dict)
                         and rc.get("name") == robot and t.get("noise") is None
                         and t.get("scale") is None
                         and _clip_pair(t.get("clip")) == (-sm.OBS_CLIP, sm.OBS_CLIP))
        c[f"obs_{g}_stand4_terms_then_zaxis_a_b"] = bool(ok)
    lh = (env.get("curriculum") or {}).get("lift_height") or {}
    c["lift_curriculum_roll_proof"] = bool(
        isinstance(lh, dict) and lh.get("func") == FUNC_CURR5
        and (lh.get("params") or {}) == CURRICULUM5_PARAMS)
    failed = [k for k, v in c.items() if not v]
    return {"ok": not failed, "failed": failed, "checks": c}


# ------------------------------------------------------- training-log tables

_OBS_HEAD = re.compile(r"Active Observation Terms in Group: '(\w+)' \(shape: \(([0-9, ]*)\)\)")
_OBS_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*([A-Za-z0-9_]+)\s*\|\s*\(([0-9, ]*)\)\s*\|$")
_MLP_IN = {k: re.compile(rf"{k} Model: \w+\(.*?\(0\): Linear\(in_features=(\d+)", re.S)
           for k in ("Actor", "Critic")}


def _dims(s: str) -> tuple:
    return tuple(int(v) for v in s.replace(" ", "").split(",") if v)


def obs_tables(log_text: str) -> dict:
    """{group: {"shape": (..), "terms": [(name, (dims))]}} from the observation
    manager's tables in a training log (the first table of each group)."""
    out, cur = {}, None
    for ln in s4.strip_ansi(log_text).splitlines():
        ln = ln.strip()
        m = _OBS_HEAD.search(ln)
        if m:
            cur = None if m.group(1) in out else m.group(1)
            if cur:
                out[cur] = {"shape": _dims(m.group(2)), "terms": []}
            continue
        if cur is None:
            continue
        r = _OBS_ROW.match(ln)
        if r:
            out[cur]["terms"].append((r.group(2), _dims(r.group(3))))
        elif out[cur]["terms"]:          # the border (or anything) after the rows ends it
            cur = None
    return out


def mlp_in_features(log_text: str) -> dict:
    """{"actor": n, "critic": n}: the first Linear layer of each rsl-rl model."""
    text = s4.strip_ansi(log_text)
    out = {}
    for k, rx in _MLP_IN.items():
        m = rx.search(text)
        out[k.lower()] = int(m.group(1)) if m else None
    return out


def expected_stand5_tables(stand4_tables: dict) -> dict:
    """Stand4's tables with object_zaxis_a/b (3,) appended to both groups."""
    out = {}
    for g in OBS_GROUPS:
        t4 = stand4_tables.get(g)
        if not t4:
            return {}
        terms = list(t4["terms"]) + [(n, (OBS_TERM_DIM,)) for n in OBS_TERMS5]
        out[g] = {"shape": (sum(d[0] for _, d in terms),), "terms": terms}
    return out


def tables_check(run_tables: dict, stand4_tables: dict, mlp: dict | None = None) -> dict:
    """The run's tables == Stand4's + the two rows per group; widths 200 / 212;
    actor / critic first layers read those widths (when `mlp` is given)."""
    want = expected_stand5_tables(stand4_tables)
    norm = {g: {"shape": tuple(v["shape"]), "terms": [(n, tuple(d)) for n, d in v["terms"]]}
            for g, v in (run_tables or {}).items() if g in OBS_GROUPS}
    ok = bool(want) and norm == want and all(
        want[g]["shape"] == (STAND5_OBS_WIDTH[g],) for g in OBS_GROUPS)
    if mlp is not None:
        ok = ok and mlp.get("actor") == STAND5_OBS_WIDTH["policy"] \
            and mlp.get("critic") == STAND5_OBS_WIDTH["critic"]
    return {"ok": bool(ok), "want": want, "got": norm, "mlp": mlp}


# -------------------------------------------------------- source identity

STAND4_REF_SMOKE_JOB = "21506757"            # Stand4's smoke of its committed code (4c5af8e)
STAND4_REF_RUN = ("external/Berkeley-Humanoid-Lite/logs/rsl_rl/task_v2/"
                  "2026-10-02_03-31-37_v2-cubetoshelfstand4-blind-s100-j21506757-smoke")
STAND4_REF_DIR = "results/repo-gpu-20260923/stand4_2026-10-01-smoke"
TASK_V2_REL = "src/bhl_robust/tasks/task_v2_env_cfg.py"
STAND5_MARKER = "\n\n# --- stand5 ---"
#: Stand4's sources that Stand5 builds on, byte-identical to what Stand4's smoke
#: hashed (the overlay path comes from stand4_mdp; tasks/__init__.py is shared).
STAND4_REF_FILES = ("src/bhl_robust/tasks/stand4_mdp.py", "src/bhl_robust/tasks/stand_mdp.py",
                    "src/bhl_robust/tasks/coop_lift_mdp.py",
                    "src/bhl_robust/tasks/coop_lift_env_cfg.py",
                    s4.HAND_COLLIDER_USD.relative_to(s4.HAND_COLLIDER_USD.parents[2]).as_posix())


def parse_sha256_list(text: str) -> dict:
    """`sha256sum` output -> {path: hex}."""
    out = {}
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) == 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            out[parts[1].lstrip("*")] = parts[0]
    return out


def task_v2_stand4_prefix(data: bytes) -> bytes | None:
    """task_v2_env_cfg.py up to (not including) the Stand5 block; None if absent."""
    i = data.find(STAND5_MARKER.encode())
    return data[:i] if i > 0 else None


TASKS_INIT_REL = "src/bhl_robust/tasks/__init__.py"


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def stand4_sources_check(repo: str | Path, ref_sha_text: str, start_sha_text: str | None = None,
                         start_exclude: Sequence[str] = (TASKS_INIT_REL,)) -> dict:
    """Stand4's sources are the bytes Stand4's reference smoke validated: each of
    STAND4_REF_FILES matches the reference sha256 list, and task_v2_env_cfg.py
    before the Stand5 marker hashes to the reference's whole-file hash. With
    `start_sha_text` (the smoke's own list, taken when it started), also: every
    file in it but `start_exclude` (tasks/__init__.py is shared with other
    workstreams) still has those bytes, so the smoke judged one set of bytes."""
    repo = Path(repo)
    ref = parse_sha256_list(ref_sha_text)
    checks = {}
    for rel in STAND4_REF_FILES:
        checks[rel] = bool(rel in ref and _sha(repo / rel) == ref[rel])
    p = repo / TASK_V2_REL
    pre = task_v2_stand4_prefix(p.read_bytes()) if p.is_file() else None
    checks[TASK_V2_REL + " (before the stand5 marker)"] = bool(
        pre is not None and TASK_V2_REL in ref and hashlib.sha256(pre).hexdigest() == ref[TASK_V2_REL])
    out = {}
    if start_sha_text is not None:
        start = parse_sha256_list(start_sha_text)
        changed = sorted(rel for rel, h in start.items()
                         if rel not in start_exclude and _sha(repo / rel) != h)
        checks["hashed files unchanged since the smoke started"] = bool(start) and not changed
        out["changed_since_start"] = changed
    failed = [k for k, v in checks.items() if not v]
    return dict(out, ok=not failed, failed=failed, checks=checks)


# ------------------------------------------------------------- smoke rule
# gpu_v2_stand5_smoke.sbatch: (A) an in-Isaac probe of the built task
# (slurm/inner/inner_v2_stand5_smoke.sh), (B) a 20-iteration, 1024-env PPO run
# of the real training path, (C) host checks; then smoke_verdict5. Stand4's
# smoke settings exactly (seed 100, 16-env probe, 1024 x 20), so the dumped
# params compare with Stand4's smoke key for key. SCRIPTED probe + smoke PPO:
# never a result.

SMOKE_SEED = s4.SMOKE_SEED                   # 100
SMOKE_PROBE_ENVS = s4.SMOKE_PROBE_ENVS       # 16
SMOKE_ITERS = s4.SMOKE_ITERS                 # 20
SMOKE_ENVS = s4.SMOKE_ENVS                   # 1024
SMOKE_TAGS = s4.SMOKE_TAGS
FREE_Z = 2.5                                 # m, env-local: free space above the plinth

#: Obs stage: the cube in free space, at rest, in a known orientation per env.
AXIS_CASE_ORDER = ("identity", "x+90", "y+90", "x180", "z+30")
AXIS_CASES = {"identity": quat_axis_angle((0.0, 0.0, 1.0), 0.0),
              "x+90": quat_axis_angle((1.0, 0.0, 0.0), 90.0),
              "y+90": quat_axis_angle((0.0, 1.0, 0.0), 90.0),
              "x180": quat_axis_angle((1.0, 0.0, 0.0), 180.0),
              "z+30": quat_axis_angle((0.0, 0.0, 1.0), 30.0)}
#: The cube's own z axis in the world for each case (+90 deg about x takes
#: +z to -y; about y, to +x).
EXPECTED_WORLD_Z = {"identity": (0.0, 0.0, 1.0), "x+90": (0.0, -1.0, 0.0),
                    "y+90": (1.0, 0.0, 0.0), "x180": (0.0, 0.0, -1.0), "z+30": (0.0, 0.0, 1.0)}
#: The control: cases whose stored numbers, read in the other order, give a z
#: axis far from the right one (x+90 -> +z, x180 -> +z). Not y+90 (read
#: swapped it is a half-turn about (1, 0, 1)/sqrt 2, which also takes +z to +x),
#: identity or z+30 (+z, and 0.5 away).
ORDER_CONTROL_CASES = ("x+90", "x180")


def axis_case(i: int) -> str:
    return AXIS_CASE_ORDER[i % len(AXIS_CASE_ORDER)]


#: Curriculum stage groups (env i -> i % 4) and their poses: in free space flat
#: (roll-proof), in free space tilted 20 deg (centre high, tilt fails), rolled
#: 45 deg about y resting on its edge on the plinth (centre +0.058 m: Stand4's
#: centre clause holds at the 0.04 level; the corner rests on the plinth), flat
#: at rest on the plinth (neither).
CURR_GROUPS = ("clear_flat", "clear_tilt20", "edge_plinth", "flat_plinth")
#: What the roll-proof gate must read after the stage's one step (None: not
#: asserted -- a flat cube on the plinth among the hand hulls is left to the
#: physics; the calls pick their envs by the recomputed flags anyway).
CURR_ROLL_OK = {"clear_flat": True, "clear_tilt20": False, "edge_plinth": False,
                "flat_plinth": None}


def curr_group(i: int) -> str:
    return CURR_GROUPS[i % len(CURR_GROUPS)]


def curr_pose_local(i: int) -> tuple:
    """((x, y, z), (w, x, y, z)): env `i`'s curriculum-stage cube pose, env-local."""
    g = curr_group(i)
    if g == "clear_flat":
        return (0.0, 0.0, FREE_Z), quat_axis_angle((1.0, 0.0, 0.0), 0.0)
    if g == "clear_tilt20":
        return (0.0, 0.0, FREE_Z), quat_axis_angle((1.0, 0.0, 0.0), 20.0)
    if g == "edge_plinth":
        q = quat_axis_angle((0.0, 1.0, 0.0), 45.0)
        hz = float(s4.cube_half_extents(*(torch.tensor([v], dtype=torch.float64) for v in q))[2])
        return (0.0, 0.0, sm.PLINTH_TOP + hz), q
    return (0.0, 0.0, sm.PLINTH_TOP + s4.CUBE_HALF), quat_axis_angle((1.0, 0.0, 0.0), 0.0)


def _cols(q, order: str):
    return qo.unpack_wxyz(torch.as_tensor(q, dtype=torch.float64), order=order)


def _maxabs(a, b) -> float:
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    if a.shape != b.shape or a.numel() == 0:
        return math.inf
    return float((a - b).abs().max())


def obs_stage_eval(rec: dict, order: str, tol_tail: float = 1e-5, tol_ref: float = 1e-4,
                   tol_case: float = 1e-3, control_min: float = 0.5) -> dict:
    """Clauses A4 (tail) and A5 (axis order) from the probe's raw obs records.

    `rec`: per env the case, `valid` (no reset in the step), the cube's and each
    robot's root quaternion as stored (native order `order`), the two terms'
    values, Isaac Lab's own reference R_robot^T z (matrix_from_quat), the
    cube's world z axis by Isaac Lab, and each group's last 6 observation
    columns. Recomputed here (host): the cube's world z axis per case vs the
    fixed numbers; the terms from the raw quaternions with this module's
    kernels; the agreement with Isaac Lab's reference; and the control -- the
    same stored numbers read in the other order must miss the x+90 / y+90
    vectors by > control_min.
    """
    try:
        cases = list(rec["cases"])
        n = len(cases)
        idx = [i for i in range(n) if bool(rec["valid"][i])]
        cube = _cols(rec["cube_quat_native"], order)
        world = body_zaxis_world(*cube)
        want = torch.tensor([EXPECTED_WORLD_Z[c] for c in cases], dtype=torch.float64)
        sel = torch.tensor(idx, dtype=torch.long)
        case_err = _maxabs(world[sel], want[sel])
        isaac_world_err = _maxabs(world[sel], torch.tensor(rec["world_zaxis_isaac"])[sel])
        host_err = ref_err = tail_err = 0.0
        for k, (name, robot) in enumerate(OBS_TERM_ROBOTS):
            got = torch.tensor(rec["term_values"][name], dtype=torch.float64)
            host = zaxis_in_frame(_cols(rec["robot_quat_native"][robot], order), cube)
            host_err = max(host_err, _maxabs(host[sel], got[sel]))
            ref_err = max(ref_err, _maxabs(torch.tensor(rec["isaac_ref_values"][name])[sel], got[sel]))
            for g in OBS_GROUPS:
                tail = torch.tensor(rec["tail"][g], dtype=torch.float64)
                tail_err = max(tail_err, _maxabs(tail[sel, 3 * k:3 * k + 3], got[sel]))
        other = qo.WXYZ if order == qo.XYZW else qo.XYZW
        wrong = body_zaxis_world(*_cols(rec["cube_quat_native"], other))
        ctrl = [i for i in idx if cases[i] in ORDER_CONTROL_CASES]
        control_dev = (min(float((wrong[i] - want[i]).abs().max()) for i in ctrl)
                       if ctrl else 0.0)
        norms_ok = all(abs(float(torch.tensor(rec["term_values"][nm], dtype=torch.float64)[i].norm())
                           - 1.0) < 1e-4 for nm in OBS_TERMS5 for i in idx)
    except (KeyError, TypeError, ValueError, IndexError, RuntimeError) as exc:
        return {"ok_tail": False, "ok_axis": False, "error": repr(exc)}
    covered = sorted({cases[i] for i in idx})
    ok_tail = bool(idx) and tail_err <= tol_tail
    ok_axis = bool(order == qo.XYZW and idx and covered == sorted(AXIS_CASE_ORDER)
                   and case_err <= tol_case and isaac_world_err <= tol_ref
                   and host_err <= tol_ref and ref_err <= tol_ref and norms_ok
                   and ctrl and control_dev > control_min)
    return {"ok_tail": ok_tail, "ok_axis": ok_axis, "order": order, "n_valid": len(idx),
            "cases_covered": covered, "case_err": case_err, "isaac_world_err": isaac_world_err,
            "host_recompute_err": host_err, "isaac_ref_err": ref_err, "tail_err": tail_err,
            "unit_norm": norms_ok, "order_control_min_dev": control_dev}


def curriculum_stage_eval(cur: dict, order: str) -> dict:
    """Clause A6 from the probe's raw curriculum records (host recomputation).

    `cur`: per env the group, `valid`, the env-local cube centre, its stored
    quaternion and world centre height, the probe's roll_ok, the term's params
    and Stand4's, the spawn height; per call the env ids, the level before
    the call, the pinch distance the call saw (all envs), Stand5's return value,
    `lifting_object`'s minimal_height and env._bhl_lift_h after it, and
    Stand4's function's return value on the same state and level.

    PASS iff: roll_ok recomputed here from the raw state equals the probe's,
    and on valid envs equals the group's design (only clear_flat); every call's
    Stand5 value, written minimal_height and _bhl_lift_h equal the level update
    on the roll-proof share of its ids; every call's Stand4 value equals the
    update on Stand4's test (centre > spawn + level AND pinch < 0.20); a call
    promotes under Stand5 and not under Stand4 (roll-proof, unpinched) and
    another promotes under Stand4 and not under Stand5 (centre high, pinched,
    not roll-proof), both with >= 1 env; and one call ran on int32 ids, the
    dtype the env's reset passes.
    """
    try:
        groups = list(cur["groups"])
        n = len(groups)
        valid = [bool(v) for v in cur["valid"]]
        prm = cur["params"]
        p = torch.tensor(cur["p_local"], dtype=torch.float64)
        w, x, y, z = _cols(cur["quat_native"], order)
        roll = s4.roll_proof_mask(p, w, x, y, z, clearance=prm["clearance"],
                                  tilt_max_deg=prm["tilt_max_deg"]).tolist()
        recompute_ok = roll == [bool(v) for v in cur["roll_ok"]]
        design_ok = all(roll[i] == CURR_ROLL_OK[groups[i]] for i in range(n)
                        if valid[i] and CURR_ROLL_OK[groups[i]] is not None)
        design_ok = design_ok and all(any(valid[i] and groups[i] == g for i in range(n))
                                      for g, v in CURR_ROLL_OK.items() if v is not None)
        zw = [float(v) for v in cur["z_world"]]
        spawn = float(cur["spawn_z"])
        upd = dict(step=prm["step"], min_height=prm["min_height"], max_height=prm["max_height"],
                   success_rate_target=prm["success_rate_target"])
        calls, s5_only, s4_only, int32_seen = [], False, False, False
        all_ok = True
        for c in cur["calls"]:
            ids = list(c["ids"]) or list(range(n))
            pre = float(c["pre"])
            pinched = float(c["pinch_d"]) < STAND4_PINCH_PROMOTE
            comp5 = sum(roll[i] for i in ids) / len(ids)
            comp4 = sum((zw[i] > spawn + pre) and pinched for i in ids) / len(ids)
            e5, e4 = curriculum_decision(pre, comp5, **upd), curriculum_decision(pre, comp4, **upd)
            ok = (math.isclose(c["r5"], e5, abs_tol=1e-12)
                  and math.isclose(c["minimal_height_after_r5"], e5, abs_tol=1e-12)
                  and math.isclose(c["lift_h_after_r5"], e5, abs_tol=1e-12)
                  and math.isclose(c["r4"], e4, abs_tol=1e-12))
            all_ok = all_ok and ok
            if c["ids"] and e5 > pre and not e4 > pre:
                s5_only = True
            if c["ids"] and e4 > pre and not e5 > pre:
                s4_only = True
            int32_seen = int32_seen or (c.get("ids_dtype") == "torch.int32" and ok)
            calls.append({"name": c.get("name"), "n": len(c["ids"]), "pre": pre,
                          "competence5": comp5, "competence4": comp4, "want5": e5, "want4": e4,
                          "r5": c["r5"], "r4": c["r4"], "ok": ok})
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError, RuntimeError) as exc:
        return {"ok": False, "error": repr(exc)}
    ok = bool(recompute_ok and design_ok and calls and all_ok and s5_only and s4_only
              and int32_seen)
    return {"ok": ok, "roll_ok_recomputed_equal": recompute_ok, "roll_ok_matches_design": design_ok,
            "all_calls_consistent": all_ok, "promotes_on_roll_proof_only": s5_only,
            "ignores_centre_and_pinch": s4_only, "int32_ids_call": int32_seen, "calls": calls,
            "n_valid": sum(valid)}


SMOKE5_CLAUSES = ("A1_managers", "A2_cfg_diff_declared_only", "A3_obs_widths",
                  "A4_obs_tail_is_the_terms", "A5_axis_order_xyzw", "A6_curriculum_roll_proof",
                  "B1_logged_tags", "B2_training_ran", "B3_no_failure_markers", "B4_config_check",
                  "B5_dump_diff_vs_stand4_smoke", "B6_train_obs_tables",
                  "C1_stand4_sources_identical", "C2_config_recorded")


def _managers_ok(m: dict, exp: dict) -> bool:
    obs = m.get("obs_terms", {})
    new = m.get("new_obs", {})
    lh = m.get("lift_height", {})
    ok = (m.get("reward_terms") == exp.get("reward_terms")
          and m.get("termination_terms") == exp.get("termination_terms")
          and m.get("curriculum_terms") == exp.get("curriculum_terms")
          and all(obs.get(g) == list(exp.get("obs_terms", {}).get(g, [])) + list(OBS_TERMS5)
                  for g in OBS_GROUPS)
          and exp.get("obs_terms", {}).get("policy") == list(STAND4_OBS_TERMS["policy"])
          and exp.get("obs_terms", {}).get("critic") == list(STAND4_OBS_TERMS["critic"])
          and lh.get("func") == FUNC_CURR5 and lh.get("params") == CURRICULUM5_PARAMS
          and exp.get("lift_height_func") == FUNC_CURR4
          and exp.get("lift_height_params") == CURRICULUM4_PARAMS
          and m.get("lifting_object", {}).get("func") == s4.FUNC_LIFT4
          and m.get("lifting_object", {}).get("weight") == 15.0
          and m.get("placed", {}).get("func") == s4.FUNC_PLACED
          and m.get("placed", {}).get("weight") == sm.STAND3_PLACED_WEIGHT
          and m.get("success", {}).get("func") == s4.FUNC_SUCCESS4)
    for g in OBS_GROUPS:
        for name, robot in OBS_TERM_ROBOTS:
            t = new.get(g, {}).get(name, {})
            ok = ok and (t.get("func") == FUNC_OBS5 and t.get("robot") == robot
                         and t.get("noise") == "None" and t.get("scale") == "None"
                         and t.get("clip") == [-sm.OBS_CLIP, sm.OBS_CLIP])
    return bool(ok)


def smoke_verdict5(probe: dict | None, tags: set, train_log: str | None,
                   cfg_check_res: dict | None, dump_diff_res: dict | None,
                   stand4_log: str | None, stand4_bytes: dict | None,
                   recorded: dict | None) -> dict:
    """The smoke rule (gpu_v2_stand5_smoke.sbatch header). PASS iff every clause
    holds; INCOMPLETE if an input is missing; else FAIL."""
    C, missing = {}, []
    ref_tables = obs_tables(stand4_log) if stand4_log is not None else {}
    if stand4_log is None or not expected_stand5_tables(ref_tables):
        missing.append("Stand4 reference smoke log (observation tables) missing or unreadable")
    if probe is None:
        missing.append("probe JSON missing or unreadable")
    elif probe.get("status") != "ok":
        missing.append(f"probe status {probe.get('status')!r}")
    else:
        m = probe.get("managers", {})
        C["A1_managers"] = _managers_ok(m, probe.get("stand4_expected", {}))
        C["A2_cfg_diff_declared_only"] = diff_check(probe.get("cfg_diff_keys", ["?"]),
                                                    CFG_DIFF_ALLOWED5)["ok"]
        if ref_tables:
            probe_tables = {g: {"shape": tuple(m.get("obs_group_dims", {}).get(g, [])),
                                "terms": list(zip(m.get("obs_terms", {}).get(g, []),
                                                  [tuple(d) for d in m.get("obs_term_dims", {})
                                                   .get(g, [])]))}
                            for g in OBS_GROUPS}
            C["A3_obs_widths"] = tables_check(probe_tables, ref_tables)["ok"]
        order = probe.get("quat_order", "")
        ev = obs_stage_eval(probe.get("obs") or {}, order) if order in (qo.XYZW, qo.WXYZ) else {}
        C["A4_obs_tail_is_the_terms"] = bool(ev.get("ok_tail"))
        C["A5_axis_order_xyzw"] = bool(ev.get("ok_axis"))
        cu = (curriculum_stage_eval(probe.get("curriculum") or {}, order)
              if order in (qo.XYZW, qo.WXYZ) else {})
        C["A6_curriculum_roll_proof"] = bool(cu.get("ok"))
    C["B1_logged_tags"] = set(SMOKE_TAGS) <= set(tags)
    if train_log is None:
        missing.append("training log missing")
    else:
        C["B2_training_ran"] = len(s4.logged_iterations(train_log)) >= SMOKE_ITERS
        C["B3_no_failure_markers"] = s4.FAIL_MARKERS.search(s4.strip_ansi(train_log)) is None
        if ref_tables:
            C["B6_train_obs_tables"] = tables_check(obs_tables(train_log), ref_tables,
                                                    mlp_in_features(train_log))["ok"]
    if cfg_check_res is None:
        missing.append("run params (agent.yaml / env.yaml) missing")
    else:
        C["B4_config_check"] = bool(cfg_check_res.get("ok"))
    if dump_diff_res is None:
        missing.append("dumped-params diff against Stand4's smoke not computed")
    else:
        C["B5_dump_diff_vs_stand4_smoke"] = bool(dump_diff_res.get("ok"))
    if stand4_bytes is None:
        missing.append("Stand4 reference sha256 list missing")
    else:
        C["C1_stand4_sources_identical"] = bool(stand4_bytes.get("ok"))
    if recorded is None:
        missing.append("run params not recorded")
    else:
        C["C2_config_recorded"] = bool(recorded.get("ok"))
    if missing or any(k not in C for k in SMOKE5_CLAUSES):
        verdict = "INCOMPLETE (no verdict: " + "; ".join(missing or ["clause not evaluated"]) + ")"
    elif all(C[k] for k in SMOKE5_CLAUSES):
        verdict = "PASS"
    else:
        verdict = "FAIL (" + ", ".join(k for k in SMOKE5_CLAUSES if not C[k]) + ")"
    return {"verdict": verdict, "clauses": C, "missing": missing,
            "tags_missing": sorted(set(SMOKE_TAGS) - set(tags))}
