"""Deploy-side gait clock for the R1 turning arm (Velocity-BHL-Arms-TurnGaitClock-v0).

R1's actor sees 77 observations: upstream's 75 (command, base angular velocity,
projected gravity, joint positions, joint velocities, last action -- Isaac's term
order) followed by [sin(2 pi phase), cos(2 pi phase)], where phase is Isaac's
episode clock ((episode_length_buf * step_dt) mod period) / period
(`bhl_robust.tasks.gait_clock_mdp.gait_clock`).

Isaac computes observation k of an episode (k = 0 right after the reset) with
episode_length_buf = k, so the deployed clock counts policy updates since the
reset: update k appends the clock of step k, one update = one step_dt (the deploy
config's policy_dt). The counter restarts whenever update() finds the controller's
observation buffer all zeros, which is exactly the state RlController.__init__,
MultiRunner.reset and harness.run_episode leave it in -- so the phase resets at
runner.reset without touching the harnesses. (Mid-episode the buffer is never all
zeros: projected gravity is a unit vector.)

A deploy config carries the clock only if it has a `gait_clock` block, written after
export by `stamp` below from the trained run's own params/env.yaml. Without one,
`make_controller` returns upstream's RlController(cfg) itself, so every existing
75-observation deploy runs the unchanged code path.

    python -m bhl_robust.eval.gait_clock stamp --deploy <run>/exported/deploy.yaml --env-yaml <run>/params/env.yaml
    python -m bhl_robust.eval.gait_clock check-env --env-yaml <run>/params/env.yaml --arm R2
    python -m bhl_robust.eval.gait_clock check-recipe --env-yaml <run>/params/env.yaml --arm R1 \
        --ref-env-yaml <arms-turn-turnboth-s0>/params/env.yaml
    python -m bhl_robust.eval.gait_clock smoke --deploy <run>/exported/deploy.yaml --upstream <U> \
        --cache-dir <dir> --expect clock|plain
    python -m bhl_robust.eval.gait_clock run-eval --deploy <run>/exported/deploy.yaml --upstream <U> \
        --cache-dir <dir> --out <new csv> --label <run> [--seed0 100]     # run_eval on exploration seeds only
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

CLOCK_KEY = "gait_clock"
# Frozen with the Isaac terms (gait_clock_mdp.GAIT_PERIOD_S / GAIT_OFFSETS[0]); `stamp`
# refuses a run whose env.yaml says anything else.
EXPECTED_PERIOD_S = 0.8
EXPECTED_PHASE_OFFSET = 0.0
N_BASE_TERMS_FIXED = 9          # command 3 + base angular velocity 3 + projected gravity 3


def clock_phase(step: int, step_dt: float, period: float) -> float:
    """Isaac's ((episode_length_buf * step_dt) % period) / period, in float32 like torch."""
    t = np.float32(np.float32(step) * np.float32(step_dt))
    return float(np.float32(np.remainder(t, np.float32(period)) / np.float32(period)))


def clock_features(step: int, step_dt: float, period: float, phase_offset: float = 0.0) -> np.ndarray:
    """[sin(2 pi phase), cos(2 pi phase)] for policy update `step` after the reset."""
    ph = (clock_phase(step, step_dt, period) + phase_offset) % 1.0
    return np.array([math.sin(2.0 * math.pi * ph), math.cos(2.0 * math.pi * ph)], dtype=np.float32)


def base_obs_width(num_actions: int) -> int:
    """Upstream's per-frame actor width: 9 + 3 * num_actions (75 for 22 actions)."""
    return N_BASE_TERMS_FIXED + 3 * int(num_actions)


def has_clock(cfg) -> bool:
    try:
        return CLOCK_KEY in cfg and cfg[CLOCK_KEY] is not None
    except TypeError:
        return False


def make_controller(cfg):
    """RlController(cfg) for a deploy config without a gait_clock block (the unchanged path);
    GaitClockRlController(cfg) for one with it."""
    from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
    if not has_clock(cfg):
        return RlController(cfg)
    return _clock_controller_class()(cfg)


_CLS = None


def _clock_controller_class():
    global _CLS
    if _CLS is not None:
        return _CLS
    from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController

    class GaitClockRlController(RlController):
        """Upstream RlController with [sin, cos] of Isaac's episode clock appended to each frame."""

        def __init__(self, cfg):
            super().__init__(cfg)
            gc = cfg[CLOCK_KEY]
            self.clock_period = float(gc["period_s"])
            self.clock_phase_offset = float(gc["phase_offset"])
            self.clock_step_dt = float(cfg.policy_dt)
            self.clock_step = 0
            n_base = base_obs_width(cfg.num_actions)
            if int(cfg.num_observations) != n_base + 2:
                raise ValueError(f"gait_clock deploy config: num_observations {cfg.num_observations} != "
                                 f"{n_base} + 2 (upstream terms + sin/cos)")
            if int(cfg.history_length) != 0:
                raise ValueError("gait_clock deploy config: history_length must be 0")
            if not self.clock_period > 0.0:
                raise ValueError(f"gait_clock period_s must be > 0, got {self.clock_period}")

        def reset_phase(self) -> None:
            self.clock_step = 0

        def clock_now(self) -> np.ndarray:
            return clock_features(self.clock_step, self.clock_step_dt, self.clock_period, self.clock_phase_offset)

        def update(self, robot_observations: np.ndarray) -> np.ndarray:
            # A zeroed observation buffer = a fresh controller or a harness reset
            # (MultiRunner.reset, harness.run_episode): Isaac's episode_length_buf is 0 again.
            if not self.policy_observations.any():
                self.clock_step = 0
            n = self.cfg.num_actions
            # --- verbatim upstream RlController.update parsing ---
            robot_base_quat = robot_observations[0:4]
            robot_base_ang_vel = robot_observations[4:7]
            robot_joint_pos = robot_observations[7:7 + n] - self.default_joint_positions
            robot_joint_vel = robot_observations[7 + n:7 + n * 2]
            command_velocity = robot_observations[7 + n * 2 + 1:7 + n * 2 + 4]
            base_ang_vel = robot_base_ang_vel
            projected_gravity = self.quat_rotate_inverse(robot_base_quat, self.gravity_vector)
            joint_pos = robot_joint_pos
            joint_vel = robot_joint_vel
            clock = self.clock_now()
            self.policy_observations[:] = np.concatenate([
                self.policy_observations[0, self.cfg.num_observations:],
                command_velocity,
                base_ang_vel,
                projected_gravity,
                joint_pos,
                joint_vel,
                self.prev_actions,
                clock,                      # Isaac term order: gait_clock is the last actor term
            ], axis=0)
            self.clock_step += 1
            # --- verbatim upstream from here ---
            self.policy_actions[:] = self.policy.forward(self.policy_observations)
            policy_actions_clipped = np.clip(self.policy_actions[0],
                                             self.cfg.action_limit_lower,
                                             self.cfg.action_limit_upper)
            self.prev_actions[:] = policy_actions_clipped
            policy_actions_scaled = policy_actions_clipped * self.cfg.action_scale + self.default_joint_positions
            return policy_actions_scaled

    _CLS = GaitClockRlController
    return _CLS


# ---------------------------------------------------------------- env.yaml / deploy.yaml

def load_env_yaml(path: Path) -> dict:
    """Isaac Lab's params/env.yaml, with its !!python/... tags read as plain data."""
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    def _py(loader, suffix, node):
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        return loader.construct_scalar(node)

    _Loader.add_multi_constructor("tag:yaml.org,2002:python/", _py)
    return yaml.load(Path(path).read_text(), Loader=_Loader)


def _terms(group: dict) -> list[str]:
    return [k for k, v in group.items() if isinstance(v, dict) and "func" in v]


def check_env(env: dict, arm: str) -> dict:
    """Clock layout of a trained run's env.yaml. arm R1: gait_clock last in actor and critic;
    R2: last in the critic, absent from the actor. Returns {period_s, phase_offset, step_dt}; raises otherwise."""
    obs = env["observations"]
    pol, cri = _terms(obs["policy"]), _terms(obs["critic"])
    if arm == "R1":
        if not pol or pol[-1] != "gait_clock":
            raise ValueError(f"R1 env.yaml: actor terms {pol} do not end with gait_clock")
    elif arm == "R2":
        if "gait_clock" in pol:
            raise ValueError(f"R2 env.yaml: gait_clock in the actor terms {pol}")
    else:
        raise ValueError(f"unknown arm {arm!r}")
    if not cri or cri[-1] != "gait_clock":
        raise ValueError(f"{arm} env.yaml: critic terms {cri} do not end with gait_clock")
    clock = (obs["policy"] if arm == "R1" else obs["critic"])["gait_clock"]
    if not str(clock["func"]).endswith("gait_clock_mdp:gait_clock"):
        raise ValueError(f"gait_clock func is {clock['func']!r}")
    period = float(clock["params"]["period"])
    gait = env["rewards"]["feet_gait"]
    if float(gait["params"]["period"]) != period:
        raise ValueError(f"feet_gait period {gait['params']['period']} != clock period {period}")
    offset0 = float(list(gait["params"]["offset"])[0])
    step_dt = float(env["sim"]["dt"]) * int(env["decimation"])
    if period != EXPECTED_PERIOD_S or offset0 != EXPECTED_PHASE_OFFSET:
        raise ValueError(f"clock period {period} / left offset {offset0} differ from the frozen "
                         f"{EXPECTED_PERIOD_S} / {EXPECTED_PHASE_OFFSET}")
    return {"period_s": period, "phase_offset": offset0, "step_dt": step_dt}


def stamp(deploy: Path, env_yaml: Path) -> str:
    """Append the gait_clock block to an R1 deploy.yaml, derived from the run's own env.yaml.
    The existing text is left byte-identical. Idempotent; refuses a conflicting block."""
    from omegaconf import OmegaConf
    deploy, env_yaml = Path(deploy), Path(env_yaml)
    info = check_env(load_env_yaml(env_yaml), "R1")
    cfg = OmegaConf.load(deploy)
    if abs(float(cfg.policy_dt) - info["step_dt"]) > 1e-9:
        raise ValueError(f"deploy policy_dt {cfg.policy_dt} != env step_dt {info['step_dt']}")
    want = base_obs_width(cfg.num_actions) + 2
    if int(cfg.num_observations) != want:
        raise ValueError(f"deploy num_observations {cfg.num_observations} != {want}")
    if has_clock(cfg):
        have = cfg[CLOCK_KEY]
        if float(have["period_s"]) == info["period_s"] and float(have["phase_offset"]) == info["phase_offset"]:
            return f"already stamped: {deploy}"
        raise ValueError(f"{deploy} already carries a different gait_clock block: {dict(have)}")
    text = deploy.read_text()
    block = (f"{'' if text.endswith(chr(10)) else chr(10)}{CLOCK_KEY}:\n"
             f"  period_s: {info['period_s']}\n"
             f"  phase_offset: {info['phase_offset']}\n"
             f"  layout: sin_cos_appended_after_actions\n"
             f"  source: {env_yaml}\n")
    deploy.write_text(text + block)
    back = OmegaConf.load(deploy)
    assert float(back[CLOCK_KEY]["period_s"]) == info["period_s"]
    return f"stamped {deploy}: period {info['period_s']} s, phase offset {info['phase_offset']}"


# ---------------------------------------------------------------- the frozen recipe, from env.yaml

# docs/SOLUTIONS_2026-10-01.md section 2 (frozen; mirrors gait_clock_mdp's constants and
# arms_env_cfg.HumanoidTurnGait*Cfg). A trained run's params/env.yaml must show exactly these
# values, and differ from TurnBoth's env.yaml only where RECIPE_CHANGES says.
RECIPE = {
    "feet_gait": {"func": "bhl_robust.tasks.gait_clock_mdp:feet_gait", "weight": 0.5, "period": 0.8,
                  "offset": [0.0, 0.5], "threshold": 0.55,
                  "body_names": [".*_left_ankle_roll", ".*_right_ankle_roll"]},
    "feet_swing_height": {"func": "bhl_robust.tasks.gait_clock_mdp:feet_swing_height", "weight": -20.0,
                          "target_height": 0.05, "foot_height_offset": 0.06, "force_threshold": 1.0},
    "feet_air_time_weight": 0.0,
    "push_robot": {"func": "isaaclab.envs.mdp.events:push_by_setting_velocity", "mode": "interval",
                   "interval_range_s": [5.0, 9.0], "velocity_range": {"x": [-0.5, 0.5], "y": [-0.5, 0.5]}},
}
# Paths (section, key[, key]) where an R1 / R2 env.yaml may differ from TurnBoth's; everything
# else in RECIPE_SECTIONS must be identical (same rewards, command mix, DR, terminations, actions).
RECIPE_SECTIONS = ("rewards", "events", "commands", "observations", "terminations", "actions", "curriculum")
RECIPE_CHANGES = {
    "R1": {("rewards", "feet_gait"), ("rewards", "feet_swing_height"), ("rewards", "feet_air_time", "weight"),
           ("events", "push_robot"), ("observations", "policy", "gait_clock"), ("observations", "critic", "gait_clock")},
    "R2": {("rewards", "feet_gait"), ("rewards", "feet_swing_height"), ("rewards", "feet_air_time", "weight"),
           ("events", "push_robot"), ("observations", "critic", "gait_clock")},
}


def _num_eq(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) <= 1e-9
    except (TypeError, ValueError):
        return False


def _list_eq(a, b) -> bool:
    try:
        return len(a) == len(b) and all(_num_eq(x, y) for x, y in zip(a, b))
    except TypeError:
        return False


def check_recipe(env: dict, arm: str) -> list[str]:
    """Problems with a trained run's env.yaml against the frozen R1 / R2 recipe; [] = the recipe."""
    p: list[str] = []
    rew = env.get("rewards") or {}
    g, gp = rew.get("feet_gait") or {}, (rew.get("feet_gait") or {}).get("params") or {}
    want = RECIPE["feet_gait"]
    if str(g.get("func")) != want["func"]:
        p.append(f"feet_gait func {g.get('func')!r}")
    if not _num_eq(g.get("weight"), want["weight"]):
        p.append(f"feet_gait weight {g.get('weight')!r} != {want['weight']}")
    for k in ("period", "threshold"):
        if not _num_eq(gp.get(k), want[k]):
            p.append(f"feet_gait {k} {gp.get(k)!r} != {want[k]}")
    if not _list_eq(gp.get("offset") or [], want["offset"]):
        p.append(f"feet_gait offset {gp.get('offset')!r} != {want['offset']}")
    if "command_name" in gp:
        p.append("feet_gait is command-gated (command_name present); it must pay at every command")
    sc = gp.get("sensor_cfg") or {}
    if list(sc.get("body_names") or []) != want["body_names"] or sc.get("preserve_order") is not True:
        p.append(f"feet_gait feet {sc.get('body_names')!r} preserve_order {sc.get('preserve_order')!r}")
    s, sp = rew.get("feet_swing_height") or {}, (rew.get("feet_swing_height") or {}).get("params") or {}
    want = RECIPE["feet_swing_height"]
    if str(s.get("func")) != want["func"]:
        p.append(f"feet_swing_height func {s.get('func')!r}")
    if not _num_eq(s.get("weight"), want["weight"]):
        p.append(f"feet_swing_height weight {s.get('weight')!r} != {want['weight']}")
    for k in ("target_height", "foot_height_offset", "force_threshold"):
        if not _num_eq(sp.get(k), want[k]):
            p.append(f"feet_swing_height {k} {sp.get(k)!r} != {want[k]}")
    if not _num_eq((rew.get("feet_air_time") or {}).get("weight"), RECIPE["feet_air_time_weight"]):
        p.append(f"feet_air_time weight {(rew.get('feet_air_time') or {}).get('weight')!r} != 0.0")
    e = (env.get("events") or {}).get("push_robot") or {}
    want = RECIPE["push_robot"]
    if str(e.get("func")) != want["func"] or e.get("mode") != want["mode"]:
        p.append(f"push_robot func/mode {e.get('func')!r} / {e.get('mode')!r}")
    if not _list_eq(e.get("interval_range_s") or [], want["interval_range_s"]):
        p.append(f"push_robot interval {e.get('interval_range_s')!r} != {want['interval_range_s']}")
    vr = (e.get("params") or {}).get("velocity_range") or {}
    if sorted(vr) != sorted(want["velocity_range"]) or not all(_list_eq(vr[k], v) for k, v in want["velocity_range"].items()):
        p.append(f"push_robot velocity_range {vr!r} != {want['velocity_range']} (fixed magnitude)")
    cur = env.get("curriculum") or {}
    if isinstance(cur, dict) and any(isinstance(v, dict) for v in cur.values()):
        p.append(f"a curriculum is active ({sorted(cur)}); the recipe has none (fixed push magnitude)")
    try:
        check_env(env, arm)
    except (ValueError, KeyError, TypeError) as exc:
        p.append(f"clock layout: {exc}")
    return p


def recipe_diff(env: dict, ref: dict, arm: str) -> list[str]:
    """Paths in RECIPE_SECTIONS where env differs from the reference (TurnBoth) env.yaml, other than the
    recipe's declared changes. [] = TurnBoth's config plus the recipe's changes ONLY."""
    allowed = RECIPE_CHANGES[arm]
    out: list[str] = []

    def walk(a, b, path):
        if path in allowed:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b), key=str):
                if k not in a or k not in b:
                    if path + (k,) not in allowed:
                        out.append(".".join(map(str, path + (k,))) + (" (new)" if k not in b else " (missing)"))
                else:
                    walk(a[k], b[k], path + (k,))
        elif a != b:
            out.append(".".join(map(str, path)) + f": {a!r} != TurnBoth {b!r}")

    for sec in RECIPE_SECTIONS:
        walk(env.get(sec), ref.get(sec), (sec,))
    return out


# ---------------------------------------------------------------- smoke rollout (MuJoCo, CPU)

def smoke(deploy: Path, upstream: Path, cache: Path, expect: str, steps: int = 30, seed: int = 100,
          wz: float = 0.6) -> dict:
    """A short MuJoCo rollout of an exported deploy through turn_test.py's path (build_multi +
    ContactRunner + CpuPolicy + make_controller), two episodes from runner.reset.

    expect "clock" (an R1 export): the controller is GaitClockRlController, the deploy, the ONNX input
    and every frame are 9 + 3 * num_actions + 2 wide, frame k ends with clock_features(k) and the clock
    restarts at 0 after runner.reset. expect "plain" (an R2 or any older export): the controller is
    upstream's RlController itself and everything is 9 + 3 * num_actions wide. Not a gate: it proves the
    deploy path. `steps` should not be a multiple of the 20-step period, so a missed reset shows."""
    from omegaconf import OmegaConf
    from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
    from bhl_robust.eval.multi_robot import build_multi
    bench = Path(__file__).resolve().parents[3] / "scripts" / "bench"
    if str(bench) not in sys.path:
        sys.path.insert(0, str(bench))
    import mujoco
    from team_airlock import ContactRunner, CpuPolicy

    if expect not in ("clock", "plain"):
        raise ValueError(f"expect must be 'clock' or 'plain', got {expect!r}")
    problems: list[str] = []
    cfg = OmegaConf.load(deploy)
    policy = CpuPolicy(cfg.policy_checkpoint_path)
    onnx_width = policy.session.get_inputs()[0].shape[-1]
    width = base_obs_width(cfg.num_actions) + (2 if expect == "clock" else 0)
    rec = {"deploy": str(deploy), "expect": expect, "width": width, "onnx_width": onnx_width,
           "num_observations": int(cfg.num_observations), "has_clock_block": has_clock(cfg), "steps": steps,
           "seed": seed}
    if int(cfg.num_observations) != width:
        problems.append(f"deploy num_observations {cfg.num_observations} != {width}")
    if onnx_width != width:
        problems.append(f"ONNX input width {onnx_width} != {width}")
    if has_clock(cfg) != (expect == "clock"):
        problems.append(f"deploy gait_clock block present={has_clock(cfg)} but expect={expect}")
    ctrl = None
    if not problems:
        ctrl = make_controller(cfg)
        rec["controller"] = type(ctrl).__name__
        if expect == "plain" and type(ctrl) is not RlController:
            problems.append(f"controller {type(ctrl).__name__} is not upstream's RlController")
        if expect == "clock" and (type(ctrl).__name__ != "GaitClockRlController" or not isinstance(ctrl, RlController)):
            problems.append(f"controller {type(ctrl).__name__} is not GaitClockRlController")
    if problems:
        return {**rec, "problems": problems, "verdict": "FAIL"}

    ctrl.policy = policy
    model, slots = build_multi(upstream, cache / "humanoid", 1, ["t"], variant="humanoid", world="flat")
    runner = ContactRunner(model, slots, [cfg], [ctrl])
    slot = slots[0]
    owners = np.array([0 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "")
                       .startswith(slot.prefix) else -1 for g in range(model.ngeom)])
    runner.configure_contacts(owners)
    rng = np.random.default_rng(seed)
    clocks: list[list[float]] = []
    for episode, n in ((0, steps), (1, 3)):
        runner.reset(rng)
        for k in range(n):
            cmd = np.array([0.0, 0.0, wz if k >= n // 2 else 0.0], dtype=np.float32)
            target = ctrl.update(runner.observe(0, cmd))
            frame = ctrl.policy_observations[0]
            if frame.shape[0] != width:
                problems.append(f"episode {episode} update {k}: frame width {frame.shape[0]} != {width}")
                break
            if not np.all(np.isfinite(target)):
                problems.append(f"episode {episode} update {k}: non-finite joint targets")
            if not np.allclose(frame[:3], cmd, atol=1e-6):
                problems.append(f"episode {episode} update {k}: frame starts {frame[:3]} not the command {cmd}")
            if expect == "clock":
                ref = clock_features(k, float(cfg.policy_dt), float(cfg[CLOCK_KEY]["period_s"]),
                                     float(cfg[CLOCK_KEY]["phase_offset"]))
                if not np.array_equal(frame[-2:], ref):
                    problems.append(f"episode {episode} update {k}: clock {frame[-2:].tolist()} != {ref.tolist()}")
                if ctrl.clock_step != k + 1:
                    problems.append(f"episode {episode} update {k}: clock_step {ctrl.clock_step} != {k + 1}")
                if episode == 0:
                    clocks.append([round(float(x), 6) for x in frame[-2:]])
            runner.step([target])
            if runner.tilt(0) >= 0.78 and f"fell_episode{episode}" not in rec:
                rec[f"fell_episode{episode}"] = k        # recorded, not judged: a 3-iteration policy may fall
    if expect == "clock":
        distinct = len({tuple(c) for c in clocks})
        rec["distinct_clock_values_episode0"] = distinct
        if distinct < min(len(clocks), 20) or len(clocks) < 2:
            problems.append(f"the clock does not advance: {distinct} distinct values in {len(clocks)} updates")
    return {**rec, "problems": problems[:20], "n_problems": len(problems), "verdict": "FAIL" if problems else "PASS"}


# ---------------------------------------------------------------- run_eval on exploration seeds (plumbing)

# Plumbing runs use seeds >= 100, disjoint from every scored set of the turning gates: turn_test v2
# (reset seeds 0-2, walk seed 0), v2x (10-14, walk 10-12) and cpu_turn_qualify's push eval seeds 0-9.
EXPLORATION_SEED0 = 100


def run_eval_exploration(argv: list[str], seed0: int = EXPLORATION_SEED0) -> int:
    """`bhl_robust.eval.run_eval.main(argv)` -- the qualify gate's push entry point, run unchanged -- except
    that its eval seeds are seed0 + range(--seeds) instead of range(--seeds), so a plumbing run never
    touches the scored push eval seeds 0-9. run_eval.py is not modified: the EvalConfig name its main()
    looks up is wrapped for the duration of the call and restored afterwards; the CSV's seed column
    shows whether the wrap took effect (`run_eval_smoke` checks it)."""
    if int(seed0) < EXPLORATION_SEED0:
        raise ValueError(f"exploration seeds start at {EXPLORATION_SEED0}; seed0={seed0} could be a scored seed")
    from bhl_robust.eval import run_eval
    original = run_eval.EvalConfig

    def exploration_eval_config(**kw):
        kw["seeds"] = tuple(int(seed0) + int(s) for s in kw["seeds"])
        return original(**kw)

    run_eval.EvalConfig = exploration_eval_config
    try:
        return run_eval.main(argv)
    finally:
        run_eval.EvalConfig = original


def run_eval_smoke(deploy: Path, upstream: Path, cache: Path, out: Path, label: str, episode_s: float = 2.0,
                   push_speed: float = 0.5, n_seeds: int = 1, seed0: int = EXPLORATION_SEED0) -> dict:
    """run_eval's six commands x n_seeds exploration seeds on a deploy (humanoid, flat ground, pushes at
    push_speed), into a NEW CSV. Not a gate: PASS iff run_eval returns 0 and the CSV holds 6 * n_seeds rows
    whose seeds are exactly seed0 .. seed0 + n_seeds - 1."""
    import csv
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    want = list(range(int(seed0), int(seed0) + int(n_seeds)))
    rc = run_eval_exploration(["--deploy-cfg", str(deploy), "--upstream", str(upstream), "--cache-dir", str(cache),
                               "--out", str(out), "--label", label, "--variant", "humanoid",
                               "--episode-s", str(episode_s), "--seeds", str(int(n_seeds)),
                               "--push-speed", str(push_speed), "--terrain-difficulty", "0"], seed0)
    problems: list[str] = []
    rows: list[dict] = []
    seeds: list[int] = []
    if rc != 0:
        problems.append(f"run_eval returned {rc}")
    try:
        with out.open(newline="") as f:
            rows = list(csv.DictReader(f))
        seeds = sorted({int(r["seed"]) for r in rows})
    except Exception as exc:                                        # noqa: BLE001
        problems.append(f"no readable CSV with a seed column: {out} ({exc!r})")
    if len(rows) != 6 * int(n_seeds):
        problems.append(f"CSV has {len(rows)} rows, expected {6 * int(n_seeds)}")
    if seeds != want:
        problems.append(f"CSV seeds {seeds} != the exploration seeds {want}")
    return {"deploy": str(deploy), "csv": str(out), "rc": rc, "rows": len(rows), "seeds": seeds,
            "falls": sum(str(r.get("fell", "")).strip().lower() == "true" for r in rows),
            "problems": problems, "verdict": "FAIL" if problems else "PASS"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stamp")
    s.add_argument("--deploy", type=Path, required=True)
    s.add_argument("--env-yaml", type=Path, required=True)
    c = sub.add_parser("check-env")
    c.add_argument("--env-yaml", type=Path, required=True)
    c.add_argument("--arm", choices=("R1", "R2"), required=True)
    r = sub.add_parser("check-recipe", help="the frozen R1/R2 recipe in a run's env.yaml (+ diff against TurnBoth's)")
    r.add_argument("--env-yaml", type=Path, required=True)
    r.add_argument("--arm", choices=("R1", "R2"), required=True)
    r.add_argument("--ref-env-yaml", type=Path, default=None, help="TurnBoth's params/env.yaml (arms-turn-turnboth-s0)")
    k = sub.add_parser("smoke", help="short MuJoCo rollout through turn_test's path; exit 1 unless PASS")
    k.add_argument("--deploy", type=Path, required=True)
    k.add_argument("--upstream", type=Path, required=True)
    k.add_argument("--cache-dir", type=Path, required=True)
    k.add_argument("--expect", choices=("clock", "plain"), required=True)
    k.add_argument("--steps", type=int, default=30)
    k.add_argument("--seed", type=int, default=100)
    k.add_argument("--out", type=Path, default=None)
    e = sub.add_parser("run-eval", help="run_eval, unchanged, on exploration seeds (>= 100) only, into a new CSV; "
                                        "plumbing, exit 1 unless PASS")
    e.add_argument("--deploy", type=Path, required=True)
    e.add_argument("--upstream", type=Path, required=True)
    e.add_argument("--cache-dir", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--label", required=True)
    e.add_argument("--episode-s", type=float, default=2.0)
    e.add_argument("--push-speed", type=float, default=0.5)
    e.add_argument("--n-seeds", type=int, default=1)
    e.add_argument("--seed0", type=int, default=EXPLORATION_SEED0)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "stamp":
            print(f"GAIT-CLOCK: {stamp(args.deploy, args.env_yaml)}")
        elif args.cmd == "check-env":
            print(f"GAIT-CLOCK: env.yaml OK for {args.arm}: {check_env(load_env_yaml(args.env_yaml), args.arm)}")
        elif args.cmd == "check-recipe":
            env = load_env_yaml(args.env_yaml)
            probs = check_recipe(env, args.arm)
            if args.ref_env_yaml is not None:
                probs += [f"differs from TurnBoth: {d}" for d in recipe_diff(env, load_env_yaml(args.ref_env_yaml), args.arm)]
            if probs:
                print(f"GAIT-CLOCK RECIPE: FAIL {args.arm} {args.env_yaml}: " + "; ".join(probs[:12])
                      + (f" (+{len(probs) - 12} more)" if len(probs) > 12 else ""))
                return 1
            print(f"GAIT-CLOCK RECIPE: OK {args.arm} {args.env_yaml}"
                  + (" (= TurnBoth + the declared changes only)" if args.ref_env_yaml is not None else ""))
        elif args.cmd == "run-eval":
            res = run_eval_smoke(args.deploy, args.upstream, args.cache_dir, args.out, args.label, args.episode_s,
                                 args.push_speed, args.n_seeds, args.seed0)
            print(f"GAIT-CLOCK RUN-EVAL: {res['verdict']} ({res['rows']} rows, seeds {res['seeds']}, rc {res['rc']}, "
                  f"falls {res['falls']}; problems {res['problems'][:3]}) -> {res['csv']}")
            return 0 if res["verdict"] == "PASS" else 1
        else:
            res = smoke(args.deploy, args.upstream, args.cache_dir, args.expect, args.steps, args.seed)
            if args.out is not None:
                import json
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(res, indent=2) + "\n")
            print(f"GAIT-CLOCK SMOKE: {res['verdict']} ({args.expect}; {res.get('controller')}; width {res['width']}, "
                  f"ONNX {res['onnx_width']}; distinct clock values {res.get('distinct_clock_values_episode0', '-')}; "
                  f"problems {res['problems'][:3]})")
            return 0 if res["verdict"] == "PASS" else 1
    except Exception as exc:                                        # noqa: BLE001
        print(f"GAIT-CLOCK: FAIL {exc!r}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
