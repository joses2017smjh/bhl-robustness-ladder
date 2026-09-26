"""Reward, diagnostic and rule terms for CubeToShelfStand2 (`task_v2_env_cfg`).

CubeToShelfStand2 is a DIFFERENT, EASIER task than CubeToShelf (the cube sits
at standing hand height). Its numbers are never compared with CubeToShelf's.

Why this file exists -- the diagnosis of CubeToShelfStand v1
-------------------------------------------------------------
Run `2026-09-24_09-54-29_v2-cubetoshelfstand-blind-s0` (job 21408514, killed at
iteration 1016 by its predeclared rule, fall rate > 0.9). From its TensorBoard
scalars (every `Episode_Reward/*` is `sum(weight * value * dt) / 20 s`, with
dt = 0.04 s, which the logged `still_alive` reproduces exactly):

* It learned to stand first. Mean episode length 35 steps (fall rate 1.00) at
  iterations 0-100, then 334-378 steps with fall rate 0.49-0.56 at 300-325 and
  45% time-outs; at that point the hands hovered 5-6 cm from the pinch points
  (reach kernels 0.84 coarse, 0.58 fine) and the lift bonus was ~0.
* Then the lift bonus arrived and standing went. `lifting_object` went
  0.02 -> 0.07 -> 0.72 -> 0.98 between iterations 325 and 500 while the fall
  rate went 0.49 -> 0.59 -> 0.87 -> 0.96, and stayed there (0.93 at 1000).
  `base_contact_b` grew 7x (-0.014 -> -0.105) and `base_contact_a` 3x across
  the same window: the lift appears to come with a body brace or crouch
  (`Curriculum/base_height` -0.076 -> -0.083). Unverified without a rollout.
* The reward was indifferent to that trade. Window-mean return 32.0
  (iterations 300-400, fall 0.55) -> 37.0 (600-700, fall 0.91) -> 28.1
  (900-1000): falling nine times in ten cost the policy nothing.

The arithmetic says why. Per 25 Hz step, a lifted-and-pinched cube pays
15 x 0.04 = 0.60; standing pays 1.0 x 0.04 = 0.04; a fall costs
10 x 0.04 = 0.40, once. One lifted step is worth 1.5 falls or 15 steps of
standing, so a lift that ends on the floor out-earns standing still.

`Policy/mean_std` also rose monotonically 0.94 -> 1.24 from iteration 150, at a
constant rate that did not change at lift onset: a confound (0.31 rad of
per-step noise per joint by the end) rather than the trigger. The runner is
shared (`_V2_RUNNER`) and is left alone.

The redesign: task income is conditional on staying up
------------------------------------------------------
Two coupled terms, one lever. Each alone has a hole:

1. **Upright gate** -- every shaping term (reach coarse/fine, clamp, lift
   progress, lift bonus, carry) is multiplied by `g(tilt_a) * g(tilt_b)`. A lift
   bought by leaning pays little, and the steps spent tipping over pay nothing.
   Alone it does nothing to an upright lift that is followed by a fall.
2. **Fall price** -- the fall-only penalty rises from 0.4 to 20 units
   (`FALL_PENALTY_WEIGHT` x dt). Alone, a big penalty risks a stand-and-never-
   reach optimum; the gate keeps reaching paid whenever the robots are upright.

`placed` is NOT gated: success terminates whatever the posture, and a gated
success bonus would only make succeeding while leaning worth less than
succeeding at all.

Everything here that can be is pure torch and imports without Isaac Lab, so
`tests/test_stand_mdp.py` runs on the login node. The env-facing wrappers import
`coop_lift_mdp` / `task_v2_mdp` lazily, inside the call.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

# ---------------------------------------------------------------- constants

#: Env step: sim dt 0.005 x decimation 8.
STEP_DT = 0.04
#: 20 s episode at 25 Hz.
EPISODE_STEPS = 500
#: The tilt at which `either_fallen` terminates (rad).
FALL_LIMIT = 0.78

#: Upright gate: flat (= 1) up to `GATE_FREE` rad of tilt, then a Gaussian
#: decay of width `GATE_STD`. Anchored on v1's own data, not taste:
#:   * `flat_a` / `flat_b` in v1's standing window (it 300-400) give
#:     mean sin^2(tilt) = 0.026-0.040, i.e. tilt_rms 0.16-0.20 rad, so ordinary
#:     standing sway must keep g >= 0.8 at 0.20 rad (g(0.20) = 0.85);
#:   * the gate must be ~0 well inside the 0.78 rad termination, so the tipping
#:     phase before a fall pays nothing (g(0.50) = 0.077, g(0.78) = 0.0006).
GATE_FREE = 0.10
GATE_STD = 0.25

#: Fall-only termination penalty weight. x dt = 20 reward units per fall.
#:   lower bound: v1's basins, 32.0 return at fall 0.553 vs 37.0 at 0.906, need
#:     > 5.0 / 0.353 = 14.2 extra units per fall to flip their ranking
#:     (total 14.6, weight <= -365);
#:   anchor: one fall forfeits a full episode of `still_alive`
#:     (1.0 x 0.04 x 500 = 20) -- 37% above break-even, and the price of
#:     33 lifted-and-pinched steps (1.3 s of holding the cube);
#:   upper bound: v1's standing basin still fell ~50% of episodes, so a
#:     penalty P costs it ~0.5 P against ~36 units/episode of shaping income
#:     (it 300-400); P must stay well under 72 (weight -1800) or
#:     stand-and-never-reach beats stand-and-reach.
FALL_PENALTY_WEIGHT = -500.0

#: v1 per-step prices, for the record and the tests.
V1_LIFT_STEP = 15.0 * STEP_DT        # 0.60
V1_ALIVE_STEP = 1.0 * STEP_DT        # 0.04
V1_FALL = 10.0 * STEP_DT             # 0.40
V1_BREAK_EVEN_EXTRA = (37.0 - 32.0) / (0.906 - 0.553)   # 14.2 units per fall
V1_STAND_SHAPING_INCOME = 36.0       # units / episode, it 300-400

# ------------------------------------------------------- predeclared rules
# Decided 2026-09-26 from the v1 diagnosis, before any Stand2 run. Written into
# slurm/repo20260923/gpu_v2_stand2_train.sbatch's header verbatim.

#: Kill checkpoint (model_1000.pt) and its window.
#: `placed` weight for Stand2. Isaac Lab scales every reward by step_dt (0.04), so v1's
#: weight 200 paid 8 once and ended the episode, while a lifted, pinched cube near
#: the shelf pays about 0.94 per step -- worth ~90 discounted (gamma 0.99) over the
#: rest of a 500-step episode. Placing was therefore irrational (review, 2026-09-26).
#: 2500 x 0.04 = 100 > the ~93 a full episode of hovering is worth.
PLACED_WEIGHT = 2500.0

KILL_AT_ITER = 1000
KILL_WINDOW = 100
#: "cannot stand": mean episode length under 4 s ...
KILL_MIN_EP_LEN = 100.0
#: ... or fewer than 1 episode in 10 ends any way other than a fall. v1 at its
#: kill had length 175 but time_out 0.067, so length alone would have let it
#: run to 8000; v1's standing phase (it 300) had time_out 0.447 and passes.
KILL_MIN_NONFALL = 0.10
#: Result: last-200 mean Episode_Termination/success >= 0.10 on >= 1 of 2 seeds.
RESULT_WINDOW = 200
RESULT_MIN_SUCCESS = 0.10
#: Secondary label, so a NEGATIVE says whether standing or placement failed:
#: last-200 mean (time_out + success) >= 0.50 = "stands".
STANDS_MIN_NONFALL = 0.50

POSITIVE_LABEL = "learned placement (standing variant v2)"


# ------------------------------------------------------------ pure kernels

def upright_kernel(tilt: torch.Tensor, free: float = GATE_FREE,
                   std: float = GATE_STD) -> torch.Tensor:
    """1 for tilt <= `free`, then exp(-((tilt - free) / std)^2). In [0, 1]."""
    excess = (tilt - free).clamp_min(0.0)
    return torch.exp(-torch.square(excess / std))


def pair_upright_gate(tilt_a: torch.Tensor, tilt_b: torch.Tensor,
                      free: float = GATE_FREE, std: float = GATE_STD) -> torch.Tensor:
    """Product of both robots' kernels: one robot tipping voids the pair's pay."""
    return upright_kernel(tilt_a, free, std) * upright_kernel(tilt_b, free, std)


def share_over_limit(tilt: torch.Tensor, env_ids, limit: float = FALL_LIMIT) -> float:
    """Fraction of `env_ids` whose tilt exceeds `limit`. 0.0 for an empty set."""
    t = tilt if env_ids is None else tilt[env_ids]
    if t.numel() == 0:
        return 0.0
    return float((t > limit).float().mean())


# ------------------------------------------------------- env-facing terms

def _coop():
    from bhl_robust.tasks import coop_lift_mdp
    return coop_lift_mdp


def _v2():
    from bhl_robust.tasks import task_v2_mdp
    return task_v2_mdp


def _tilts(env) -> tuple[torch.Tensor, torch.Tensor]:
    coop = _coop()
    return (coop._tilt_from_quat(env.scene["robot_a"]),
            coop._tilt_from_quat(env.scene["robot_b"]))


def upright_gate(env, gate_free: float = GATE_FREE, gate_std: float = GATE_STD) -> torch.Tensor:
    """The pair's upright gate, per env.

    Tilt comes from `coop_lift_mdp._tilt_from_quat` -- the definition
    `either_fallen` terminates on -- so the gate and the fall test cannot
    disagree about what "upright" means.
    """
    ta, tb = _tilts(env)
    return pair_upright_gate(ta, tb, gate_free, gate_std)


# Explicit signatures on purpose. The reward manager checks params against the
# signature, and a generic `func=` passthrough would hide the hand-body
# SceneEntityCfgs from the resolver on stacks that do not recurse into nested
# params -- `_hand_midpoint` would then silently average all 27 bodies. No
# SceneEntityCfg defaults either, so this module imports without Isaac Lab.

def gated_constellation_reach(env, std: float, robot_a_cfg, robot_b_cfg,
                              gate_free: float = GATE_FREE,
                              gate_std: float = GATE_STD) -> torch.Tensor:
    """`constellation_reach` x upright gate.

    The inner call caches the UNGATED hand distance in `env._bhl_pinch_d`, which
    the clamp and lift terms read through `_pinch_weight`; gating that cache as
    well would apply the gate twice to them.
    """
    r = _coop().constellation_reach(env, std=std, robot_a_cfg=robot_a_cfg,
                                    robot_b_cfg=robot_b_cfg)
    return r * upright_gate(env, gate_free, gate_std)


def gated_opposing_clamp(env, robot_a_cfg, robot_b_cfg,
                         gate_free: float = GATE_FREE,
                         gate_std: float = GATE_STD) -> torch.Tensor:
    """`opposing_clamp` x upright gate."""
    r = _coop().opposing_clamp(env, robot_a_cfg=robot_a_cfg, robot_b_cfg=robot_b_cfg)
    return r * upright_gate(env, gate_free, gate_std)


def gated_lift_progress(env, gate_free: float = GATE_FREE,
                        gate_std: float = GATE_STD) -> torch.Tensor:
    """`object_lift_progress` (pinch-gated as in v1) x upright gate."""
    return _coop().object_lift_progress(env) * upright_gate(env, gate_free, gate_std)


def gated_object_is_lifted(env, minimal_height: float, gate_free: float = GATE_FREE,
                           gate_std: float = GATE_STD) -> torch.Tensor:
    """`object_is_lifted` x upright gate.

    Keeps the parameter name `minimal_height`: `lift_height_curriculum` writes
    `params["minimal_height"]` on the term named `lifting_object`.
    """
    r = _coop().object_is_lifted(env, minimal_height=minimal_height)
    return r * upright_gate(env, gate_free, gate_std)


def gated_carry_progress(env, target_x: float, std: float = 0.8,
                         gate_free: float = GATE_FREE,
                         gate_std: float = GATE_STD) -> torch.Tensor:
    """`carry_progress` x upright gate."""
    r = _v2().carry_progress(env, target_x=target_x, std=std)
    return r * upright_gate(env, gate_free, gate_std)


def fall_penalty(env, term_name: str = "fallen") -> torch.Tensor:
    """1.0 on the step the `term_name` termination fires, else 0.0.

    Not `mdp.is_terminated`: that reads the union of every non-timeout
    termination, and `success` is one, so v1 charged a placement the same -0.4
    it charged a fall. At this variant's weight that would make a success net
    negative (8 - 20). Terminations are computed before rewards in
    `ManagerBasedRLEnv.step`, so this is the current step's value.
    """
    return env.termination_manager.get_term(term_name).float()


# ----------------------------------------------- diagnostic curriculum terms
# Curriculum terms log their return value as `Curriculum/<name>` and touch no
# gradient (see `coop_lift_mdp.base_height_mean`). They run in `_reset_idx`
# BEFORE the scene resets, so the resetting envs still hold their terminal
# state -- which is what makes "which robot fell" answerable. v1 could not say:
# `either_fallen` ORs the two robots and only the asymmetric `base_contact_b`
# (2.2x `base_contact_a`) hinted at it.

def upright_gate_mean(env, env_ids: Sequence[int], gate_free: float = GATE_FREE,
                      gate_std: float = GATE_STD) -> float:
    """Batch-mean upright gate over all envs (the fraction of task pay kept)."""
    return float(upright_gate(env, gate_free, gate_std).mean())


def fell_share(env, env_ids: Sequence[int], robot_name: str,
               limit_angle: float = FALL_LIMIT) -> float:
    """Of the envs resetting now, the fraction where `robot_name` is past the
    fall limit. Comparable to `Episode_Termination/fallen` (same denominator);
    fell_a + fell_b - fallen is the share where both went down."""
    tilt = _coop()._tilt_from_quat(env.scene[robot_name])
    ids = None if isinstance(env_ids, slice) else env_ids
    return share_over_limit(tilt, ids, limit_angle)


# -------------------------------------------------------- rule evaluation

def kill_verdict(ep_len: float, time_out: float, success: float) -> dict:
    """Predeclared kill rule at model_1000, on last-100 iteration means."""
    nonfall = time_out + success
    reasons = []
    if not (ep_len >= KILL_MIN_EP_LEN):
        reasons.append(f"mean episode length {ep_len:.1f} < {KILL_MIN_EP_LEN:.0f}")
    if not (nonfall >= KILL_MIN_NONFALL):
        reasons.append(f"time_out+success {nonfall:.3f} < {KILL_MIN_NONFALL:.2f}")
    return {
        "verdict": "KILL" if reasons else "CONTINUE",
        "reasons": reasons,
        "ep_len": ep_len, "time_out": time_out, "success": success, "nonfall": nonfall,
    }


def seed_result(success: float, time_out: float, killed: bool = False) -> dict:
    """Per-seed result on last-200 iteration means. A killed seed is NEGATIVE."""
    nonfall = time_out + success
    placed = (not killed) and success >= RESULT_MIN_SUCCESS
    return {
        "placed": placed,
        "stands": (not killed) and nonfall >= STANDS_MIN_NONFALL,
        "killed": killed,
        "success": success, "time_out": time_out, "nonfall": nonfall,
    }


def pair_result(seeds: Sequence[dict]) -> str:
    """Predeclared result over the two seeds."""
    if any(s["placed"] for s in seeds):
        return POSITIVE_LABEL
    stands = sum(1 for s in seeds if s["stands"])
    return f"NEGATIVE (stands on {stands}/{len(seeds)} seeds)"


def window_means(scalars: dict, tags: Sequence[str], last_iter: int, window: int) -> dict:
    """Mean of each tag over iterations (last_iter - window, last_iter].

    `scalars` maps tag -> list of (step, value). Missing tag or empty window
    -> NaN, which every rule above treats as a failure (`not (nan >= x)`).
    """
    out = {}
    for t in tags:
        v = [x for s, x in scalars.get(t, []) if last_iter - window < s <= last_iter]
        out[t] = sum(v) / len(v) if v else math.nan
    return out


def read_scalars(run_dir: str) -> dict:
    """All scalars of an rsl-rl run dir, tag -> [(step, value)]. Needs tensorboard."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    return {t: [(e.step, e.value) for e in ea.Scalars(t)] for t in ea.Tags()["scalars"]}


TAG_LEN = "Train/mean_episode_length"
TAG_TIMEOUT = "Episode_Termination/time_out"
TAG_SUCCESS = "Episode_Termination/success"
TAG_FALLEN = "Episode_Termination/fallen"


def evaluate_kill(scalars: dict, at_iter: int = KILL_AT_ITER) -> dict:
    m = window_means(scalars, (TAG_LEN, TAG_TIMEOUT, TAG_SUCCESS, TAG_FALLEN), at_iter, KILL_WINDOW)
    out = kill_verdict(m[TAG_LEN], m[TAG_TIMEOUT], m[TAG_SUCCESS])
    out.update({"at_iter": at_iter, "window": KILL_WINDOW, "fallen": m[TAG_FALLEN]})
    return out


def evaluate_seed(scalars: dict, killed: bool = False) -> dict:
    steps = [s for s, _ in scalars.get(TAG_SUCCESS, [])]
    last = max(steps) if steps else -1
    m = window_means(scalars, (TAG_LEN, TAG_TIMEOUT, TAG_SUCCESS, TAG_FALLEN), last, RESULT_WINDOW)
    out = seed_result(m[TAG_SUCCESS], m[TAG_TIMEOUT], killed=killed)
    out.update({"last_iter": last, "window": RESULT_WINDOW, "ep_len": m[TAG_LEN],
                "fallen": m[TAG_FALLEN]})
    return out
