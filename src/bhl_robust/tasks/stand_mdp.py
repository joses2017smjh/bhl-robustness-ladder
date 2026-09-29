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


# ===================================================================== Stand3
# CubeToShelfStand3 -- "side deck": a DIFFERENT, EASIER task than CubeToShelf
# AND than CubeToShelfStand2. Its numbers are never compared with either.
#
# Diagnosis of Stand2 (job 21434946; TensorBoard of
# 2026-09-26_16-35-02_v2-cubetoshelfstand2-blind-s{0,1}; the only runs read for
# this design, together with v1 -- the Stand3 seeds are fresh runs):
#
# What s0 learned (8000 iterations). It stands (fall 0.36, time_out 0.64 over
# the last 200) and it lifts: `lifting_object` 2.2-3.0 per episode from
# iteration 2000 = ~3 s of each ~15 s episode with the cube pinched above the
# lift threshold, and the lift curriculum climbed to +0.19 m (cube centre
# ~0.74, above the 0.72 slot ceiling). It never carried: `carry` 0.22 per
# episode (387-step episodes) is a mean kernel of 0.095; divided by the mean
# upright gate 0.935 that is 1 - tanh(|1.2 - x| / 0.8) = 0.101, a mean cube x
# of ~+0.03 m (the gate and kernel means are not exactly separable, so this is
# an estimate). The cube stayed over the plinth; the shelf is 1.2 m away and
# nothing on the way paid enough to walk sideways for (carry's slope at x = 0
# is 0.68 per metre).
#
# At the end the reward is lift income (lifting_object 2.2, reach 0.56 + 0.28,
# lift_progress 0.46, clamp 0.30, carry 0.22, alive 0.78) against action_rate
# -3.0 and fall_penalty -0.38; action_rate grew monotonically -1.4 -> -3.0 with
# the policy std 1.29 -> 1.98 (entropy 73 -> 89). The -500 fall penalty was
# not the trigger of anything: its episode sum was flat (-0.5 to -0.7) through
# every instability below.
#
# What preceded the divergence. Loss/value LEADS; Episode_Reward/action_rate
# spikes, where it spikes at all, 0-14 iterations later. Episode_Reward/* is
# logged when an episode ENDS, while a mid-episode runaway's per-step rewards
# enter the PPO returns (and so Loss/value) at once:
#   s0 4207: value loss 2.8e6 with episode action_rate -63 (median -2.1) and
#          mean reward -1.25e3 in the same iteration;
#   s0 4345-4358: value loss 90 -> 981 -> 7e4 -> ... -> 3.7e32 -> inf, rising
#          ~1e3x per iteration while episode action_rate stayed normal (-2.2
#          to -2.7); only at 4359-4360, when those episodes ended, action_rate
#          -2.9e18 / -2.8e12 and mean reward -3.6e19 / -5.4e13; recovered 4361;
#   s1 4098, 4453: value loss 1.1e3 with NO action_rate outlier; 4439: 2.2e3
#          with a mild one (-5.0, median -2.8);
#   s1 4466: 1.6e16, 4467: 3.5e34, 4468: NaN -> "normal expects all elements
#          of std >= 0.0" (the scalar std parameter went NaN). The run crashed
#          before any runaway episode ended, so no action_rate spike was logged.
# The learning rate sat at the adaptive schedule's 1e-5 floor through every
# runaway, so the schedule was not the trigger. An episode action_rate of
# -2.8e12 needs raw actions ~1e6: the reading most consistent with all of the
# above is an actor-output runaway inside some envs' episodes, fed back through
# the last-action observation (no observation or action clipping anywhere on
# this path; the runner's `clip_actions` is dead code here -- scripts/train.py
# builds RslRlVecEnvWrapper without it). It is consistent, not proven; the
# clipping below bounds that path whatever started it. Policy std rose
# monotonically in both seeds (1.0 -> 2.0 / 2.1) while the surrogate loss went
# to ~0: the entropy bonus, not the task, was driving it.
#
# The redesign, least change:
# 1. Placement within (near) reach. A deck on EACH side of the plinth along x
#    (the pair's shoulder line), its top a 2 cm lip above the plinth top so the
#    cube cannot simply be slid across. The lip does NOT force a lift: a push
#    along x can catch the cube's bottom edge on the deck's face and tip it
#    over onto the deck (~6 N at hand height), landing seated. So a success
#    means "seated on the deck", mechanism not asserted; `Curriculum/
#    cube_tilted` and `Curriculum/tipped_over_deck` make the mechanism readable
#    afterwards (a tipped cube's body z axis is ~90 deg from vertical, a
#    lifted-and-lowered one's is not). The shelf at x = 1.2 is removed. MEASURED (MuJoCo FK, feet planted, 60k arm samples
#    per hand, hands on the pinch plane): arms alone shift the hand midpoint
#    +/-0.044 m; with a pelvis twist <= 0.5 rad, 0.119 m. The required shift is
#    0.19 m, so ~0.07 m must come from lateral lean or a shuffle. It is NOT
#    "within arm reach"; it is 6x shorter than Stand2's 1.2 m walk.
# 2. Placing beats every hover, and the shift costs nothing. Because `success`
#    TERMINATES the episode, the bonus must beat the discounted continuation
#    value of the richest non-terminal state, wherever it is. Stand3 keeps
#    Stand2's lift terms UNGATED by position (a 2026-09-27 review showed that
#    zeroing them over a deck made crossing |x| = DECK_EDGE cost ~0.57/step,
#    exactly where the shift starts, while the plinth hover Stand2 s0 learned
#    stayed fully paid). The richest state is then lifted + pinched over a
#    deck centre: reach 1 + 1 + clamp 1.5 + lift_progress 2 + lifting_object
#    15 + alive 1 + deck_progress 3 = 24.5 weight (every kernel <= 1; the
#    upright gate <= 1) = 0.98/step, <= 98 discounted at gamma 0.99.
#    `placed` = STAND3_PLACED_WEIGHT x dt = 200, 2.04x that bound, from any
#    state. Lowering to seated (centre 0.57, below the 0.59-0.61 lift
#    threshold) forfeits lifting_object (0.60/step) and at least half of lift_progress
#    for the lowering + 12-step hold: ~10-15 units against 200. The bonus is
#    itself discounted over those ~25 steps: 200 x 0.99^25 = 156 against the
#    98 bound, so placing leads by >= ~57 units, not the undiscounted ~85. `object_xy`
#    (which penalised the shift itself) is removed, and the lift curriculum is
#    capped at +0.06 m (Stand2 climbed to +0.19, above anything placement
#    needs). (Stand2: bonus 100 against a ~93 hover.)
#    Scale check: a 200-unit return the critic did not foresee is one sample's
#    ~4e4 squared error averaged over a ~25k-sample rollout, far under the
#    UNSTABLE_VALUE_LOSS = 1000 label; Stand2's spikes (1e3 to 1e34) came with
#    success 0, so the bonus was never their source.
# 3. `placed` reads the `success` termination instead of re-evaluating the
#    predicate: when a reward and a termination both call a `_held` predicate
#    its counter advances twice per step (Stand2's hold_steps=12 was 6 steps).
# 4. Numerical stability, Stand3 only: every policy/critic observation term is
#    clipped (last action to +/-ACTION_CLIP, the rest to +/-OBS_CLIP), the
#    action-rate penalty is computed on clipped actions, the joint position
#    TARGETS are clipped to +/-TARGET_CLIP rad (a no-op inside the joint
#    limits; it also bounds the tracking-error observation, which reads the
#    target), and a new runner class
#    (task_v2_env_cfg.TaskV2Stand3PPORunnerCfg) uses a log-parameterised std
#    and entropy_coef STAND3_ENTROPY_COEF. _V2_RUNNER is not touched.

#: Geometry (m). The cube is the 0.28 m cube of CubeToShelf at Stand's 0.55.
CUBE_HALF = 0.14
STAND3_CUBE_Z = 0.55
PLINTH_TOP = STAND3_CUBE_Z - CUBE_HALF          # 0.41, Stand's plinth, unchanged
DECK_LIP = 0.02
DECK_TOP = PLINTH_TOP + DECK_LIP                # 0.43
DECK_SEATED_Z = DECK_TOP + CUBE_HALF            # 0.57, cube centre when seated
#: Object x reset jitter for Stand3 (Stand/Stand2: +/-0.03). Smaller, so the
#: deck can start closer without touching a spawned cube.
OBJ_X_JITTER = 0.01
DECK_CLEARANCE = 0.02
#: Near edge of each deck, |x|: spawned cube's far face + jitter + clearance.
DECK_EDGE = CUBE_HALF + OBJ_X_JITTER + DECK_CLEARANCE   # 0.17
DECK_LEN = 0.30
DECK_WIDTH = 0.30
DECK_CENTER = DECK_EDGE + DECK_LEN / 2.0        # 0.32
#: Success ("seated"): centre over a deck by SEAT_MARGIN inside either edge,
#: |y| < SEAT_Y_HALF, centre within SEAT_Z_TOL of the seated height, speed <
#: SEAT_SPEED, for SEAT_HOLD_STEPS consecutive steps -- cube_in_slot's test
#: (inside + still + held 12), with a z window that means "seated". Stricter
#: than cube_in_slot's, on purpose. NOT checked: release or tilt. A cube held
#: still by the pair 0-2 cm above the deck for 12 steps counts, i.e. the hands
#: may still be on it (SEATED_NOTE); a tipped cube lies on another face at the
#: same height, which the tilt diagnostics record.
SEAT_MARGIN = 0.02
SEAT_X_LO = DECK_EDGE + SEAT_MARGIN             # 0.19
SEAT_X_HI = DECK_EDGE + DECK_LEN - SEAT_MARGIN  # 0.45
SEAT_Y_HALF = 0.10
SEAT_Z_TOL = 0.02
SEAT_SPEED = 0.05
SEAT_HOLD_STEPS = 12
#: Shift the pair must produce from a centred spawn.
REQUIRED_SHIFT = SEAT_X_LO
#: Measured feet-planted hand-midpoint shift on the pinch plane (MuJoCo FK).
MEASURED_SHIFT_ARMS = 0.044
MEASURED_SHIFT_ARMS_TWIST = 0.119
#: Progress toward the nearer deck's centre; slope at x = 0 is
#: 3.0 x sech^2(1.6) / 0.2 = 2.2 per metre (Stand2's carry: 0.68).
DECK_PROGRESS_STD = 0.20
DECK_PROGRESS_WEIGHT = 3.0
#: Cube counts as "off the floor" above this centre height (v2.carry_progress).
OFF_FLOOR_Z = 0.34
#: Lift curriculum cap: cleared bottom 0.41 + 0.06 = 0.47, 4 cm over the lip.
STAND3_LIFT_MAX = 0.06
#: Non-terminal income weights of Stand3 (every kernel <= 1): the base
#: coop-lift weights of the five task terms and still_alive, plus deck progress.
STAND3_INCOME_WEIGHTS = {"reaching_coarse": 1.0, "reaching_fine": 1.0, "opposing_clamp": 1.5,
                         "lift_progress": 2.0, "lifting_object": 15.0, "still_alive": 1.0,
                         "deck_progress": DECK_PROGRESS_WEIGHT}
STAND3_GAMMA = 0.99
#: Upper bound on the discounted value of never placing (24.5 x 0.04 / 0.01 = 98).
STAND3_HOVER_BOUND = sum(STAND3_INCOME_WEIGHTS.values()) * STEP_DT / (1.0 - STAND3_GAMMA)
#: Stand3's placement bonus weight: x dt = 200 units, 2.04 x STAND3_HOVER_BOUND.
#: (PLACED_WEIGHT, 2500, is Stand2's and unchanged.)
STAND3_PLACED_WEIGHT = 5000.0

#: Stability (Stand3 only). Raw actions are scaled by 0.25 into joint targets,
#: so +/-10 is +/-2.5 rad, past every joint's range: clipping there changes
#: nothing the robot can do. 100 is far above any healthy observation (joint
#: velocity noise alone is +/-2) and far below a runaway.
ACTION_CLIP = 10.0
OBS_CLIP = 100.0
#: Processed joint-position target clip (rad). Largest |joint limit| of the
#: 22-DoF asset is 2.44 (knee; MJCF), so +/-3.0 never binds on a reachable pose.
TARGET_CLIP = 3.0
MAX_JOINT_LIMIT = 2.44
#: Diagnostic: cube body z axis more than this from vertical = tipped.
TILT_TIPPED_DEG = 30.0
#: Stand2: 0.005, with std 1.0 -> 2.0/2.1 monotonically in both seeds.
STAND3_ENTROPY_COEF = 0.001
STAND3_STD_TYPE = "log"

# ------------------------------------------------ Stand3 predeclared rules
# Decided 2026-09-27 from the Stand2 diagnosis above, before any Stand3 run.
# Written into slurm/repo20260923/gpu_v2_stand3_train.sbatch's header.
# Kill rule: Stand2's (KILL_AT_ITER, KILL_MIN_EP_LEN, KILL_MIN_NONFALL), except
# that the episode-length clause is skipped when the last-100 mean success is
# >= RESULT_MIN_SUCCESS (kill_verdict3; amended 2026-09-27 on review, before
# any Stand3 run): success ends the episode and a side-deck placement can end
# well inside KILL_MIN_EP_LEN steps, so a short mean length no longer implies
# falling, and a placing seed must not be killed and recorded as not placed.
# Result: last-200 success >= RESULT_MIN_SUCCESS on >= 1 of 2 seeds, counted
# only on COMPLETE runs.

STAND3_MAX_ITER = 8000
#: Secondary label: last-200 mean `Curriculum/over_deck` >= this = "shifts"
#: (cube centre over a deck, off the floor, 10% of the time).
SHIFTS_MIN_OVER_DECK = 0.10
#: Stability label: iterations with Loss/value > this or non-finite. Stand2
#: had 15 (s0) and 6 (s1); "stable" = 0.
UNSTABLE_VALUE_LOSS = 1000.0
POSITIVE_LABEL3 = "learned placement (side-deck variant v3)"
TASK3_NOTE = ("CubeToShelfStand3 (side deck): a different, easier task than "
              "CubeToShelf and CubeToShelfStand2; never compared with either")
SEATED_NOTE = ("success = cube centre over a deck, within 2 cm of its resting "
               "height and still for 12 consecutive steps; release is not checked "
               "(the hands may still be on it), nor how it got there (lift or tip; "
               "see cube_tilted / tipped_over_deck)")

TAG_VALUE = "Loss/value"
TAG_OVER_DECK = "Curriculum/over_deck"
TAG_ABS_X = "Curriculum/cube_abs_x"
TAG_TILTED = "Curriculum/cube_tilted"
TAG_TIPPED_DECK = "Curriculum/tipped_over_deck"


# ------------------------------------------------------ Stand3 pure kernels

def clipped_action_rate(action: torch.Tensor, prev: torch.Tensor,
                        bound: float = ACTION_CLIP) -> torch.Tensor:
    """sum((clip(a) - clip(prev))^2) over the last dim. <= n x (2 bound)^2."""
    a = action.clamp(-bound, bound)
    p = prev.clamp(-bound, bound)
    return torch.sum(torch.square(a - p), dim=-1)


def deck_progress_kernel(p: torch.Tensor, center: float = DECK_CENTER,
                         std: float = DECK_PROGRESS_STD,
                         off_floor_z: float = OFF_FLOOR_Z) -> torch.Tensor:
    """Off the floor x (1 - tanh(| |x| - center | / std)): toward the nearer deck."""
    up = (p[:, 2] > off_floor_z).float()
    return up * (1.0 - torch.tanh((p[:, 0].abs() - center).abs() / std))


def seated_mask(p: torch.Tensor, speed: torch.Tensor,
                x_lo: float = SEAT_X_LO, x_hi: float = SEAT_X_HI,
                y_half: float = SEAT_Y_HALF, seated_z: float = DECK_SEATED_Z,
                z_tol: float = SEAT_Z_TOL, max_speed: float = SEAT_SPEED) -> torch.Tensor:
    """Cube resting on either deck, this step (bool). `p` is env-local."""
    ax = p[:, 0].abs()
    return ((ax > x_lo) & (ax < x_hi) & (p[:, 1].abs() < y_half)
            & ((p[:, 2] - seated_z).abs() < z_tol) & (speed < max_speed))


def body_z_tilt_deg(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                    z: torch.Tensor) -> torch.Tensor:
    """Angle (deg) between a body's own z axis and world up, from (w, x, y, z)."""
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return torch.rad2deg(torch.acos(up_z.clamp(-1.0, 1.0)))


def over_deck_mask(p: torch.Tensor, edge: float = DECK_EDGE,
                   off_floor_z: float = OFF_FLOOR_Z) -> torch.Tensor:
    """Cube centre over a deck and off the floor (held or resting)."""
    return (p[:, 0].abs() > edge) & (p[:, 2] > off_floor_z)


# -------------------------------------------------- Stand3 env-facing terms

def _obj_state(env) -> tuple[torch.Tensor, torch.Tensor]:
    """(env-local cube position, cube speed)."""
    v2, coop = _v2(), _coop()
    p = v2._obj_local(env, "object")
    speed = coop._t(env.scene["object"].data.root_lin_vel_w).norm(dim=-1)
    return p, speed


def action_rate_clipped_l2(env, bound: float = ACTION_CLIP) -> torch.Tensor:
    """`mdp.action_rate_l2` on actions clipped to +/-bound."""
    am = env.action_manager
    return clipped_action_rate(am.action, am.prev_action, bound)


def gated_deck_progress(env, center: float = DECK_CENTER, std: float = DECK_PROGRESS_STD,
                        gate_free: float = GATE_FREE, gate_std: float = GATE_STD) -> torch.Tensor:
    """Progress toward the nearer deck x upright gate."""
    p, _ = _obj_state(env)
    return deck_progress_kernel(p, center, std) * upright_gate(env, gate_free, gate_std)


def cube_on_side_deck(env, hold_steps: int = SEAT_HOLD_STEPS) -> torch.Tensor:
    """Success: `seated_mask` true for `hold_steps` consecutive steps.

    Call it from the `success` termination ONLY; the `placed` reward reads the
    termination (`success_bonus`), so the hold counter advances once per step.
    """
    p, speed = _obj_state(env)
    return _v2()._held(env, "_bhl_deck_hold", seated_mask(p, speed), hold_steps)


def success_bonus(env, term_name: str = "success") -> torch.Tensor:
    """1.0 on the step the `term_name` termination fires (see `fall_penalty`)."""
    return env.termination_manager.get_term(term_name).float()


def over_deck_share(env, env_ids: Sequence[int]) -> float:
    """Diagnostic: batch share of envs whose cube is over a deck, off the floor."""
    p, _ = _obj_state(env)
    return float(over_deck_mask(p).float().mean())


def cube_abs_x_mean(env, env_ids: Sequence[int]) -> float:
    """Diagnostic: batch-mean |x| of the cube (the shift, in metres)."""
    p, _ = _obj_state(env)
    return float(p[:, 0].abs().mean())


def _cube_tilt_deg(env) -> torch.Tensor:
    from bhl_robust.quat_order import unpack_wxyz
    q = _coop()._t(env.scene["object"].data.root_quat_w)
    return body_z_tilt_deg(*unpack_wxyz(q))


def cube_tilted_share(env, env_ids: Sequence[int],
                      limit_deg: float = TILT_TIPPED_DEG) -> float:
    """Diagnostic: batch share of envs whose cube is tipped past `limit_deg`."""
    return float((_cube_tilt_deg(env) > limit_deg).float().mean())


def tipped_over_deck_share(env, env_ids: Sequence[int],
                           limit_deg: float = TILT_TIPPED_DEG) -> float:
    """Diagnostic: batch share with the cube over a deck AND tipped: placement
    by tipping rather than by lift-shift-lower."""
    p, _ = _obj_state(env)
    return float((over_deck_mask(p) & (_cube_tilt_deg(env) > limit_deg)).float().mean())


# -------------------------------------------------- Stand3 rule evaluation

def value_loss_instability(scalars: dict, limit: float = UNSTABLE_VALUE_LOSS) -> list:
    """Iterations whose Loss/value is non-finite or above `limit`."""
    return [s for s, x in scalars.get(TAG_VALUE, []) if not (math.isfinite(x) and x <= limit)]


def seed_result3(success: float, time_out: float, over_deck: float, complete: bool,
                 killed: bool = False, unstable_iters: int = 0) -> dict:
    """Per-seed Stand3 result. `decided` = complete or killed by the kill rule;
    an undecided (incomplete) seed is never a result."""
    nonfall = time_out + success
    decided = bool(complete or killed)
    ok = complete and not killed
    return {
        "complete": bool(complete), "killed": bool(killed), "decided": decided,
        "placed": bool(ok and success >= RESULT_MIN_SUCCESS),
        "stands": bool(ok and nonfall >= STANDS_MIN_NONFALL),
        "shifts": bool(ok and over_deck >= SHIFTS_MIN_OVER_DECK),
        "stable": unstable_iters == 0,
        "unstable_iters": int(unstable_iters),
        "success": success, "time_out": time_out, "nonfall": nonfall, "over_deck": over_deck,
    }


def pair_result3(seeds: Sequence[dict]) -> str:
    """Predeclared two-seed verdict.

    POSITIVE if any complete seed placed; NEGATIVE only if every seed is decided
    (complete, or killed by rule) and none placed; otherwise INCOMPLETE.
    """
    n = len(seeds)
    if any(s["placed"] for s in seeds):
        k = sum(1 for s in seeds if s["placed"])
        return f"{POSITIVE_LABEL3} ({k}/{n} seeds)"
    if all(s["decided"] for s in seeds):
        st = sum(1 for s in seeds if s["stands"])
        sh = sum(1 for s in seeds if s["shifts"])
        return f"NEGATIVE (stands on {st}/{n}, shifts on {sh}/{n} seeds)"
    bad = [i for i, s in enumerate(seeds) if not s["decided"]]
    return f"INCOMPLETE (no verdict: seed index {bad} did not complete)"


def evaluate_seed3(scalars: dict, killed: bool = False, max_iter: int = STAND3_MAX_ITER) -> dict:
    """Stand3 per-seed evaluation, with the completion check.

    COMPLETE = the run logged iteration max_iter - 1 AND every Loss/value, success
    and time_out value in the last RESULT_WINDOW iterations is finite. Isaac exits
    0 after a crash, so the exit code says nothing; a NaN'd policy can also log
    garbage to the end, which the finite-window clause catches.
    """
    steps = [s for s, _ in scalars.get(TAG_SUCCESS, [])]
    last = max(steps) if steps else -1
    lo = last - RESULT_WINDOW
    finite = all(math.isfinite(x) for t in (TAG_VALUE, TAG_SUCCESS, TAG_TIMEOUT)
                 for s, x in scalars.get(t, []) if lo < s <= last)
    has_value = any(lo < s <= last for s, _ in scalars.get(TAG_VALUE, []))
    complete = last >= max_iter - 1 and finite and has_value
    m = window_means(scalars, (TAG_LEN, TAG_TIMEOUT, TAG_SUCCESS, TAG_FALLEN, TAG_OVER_DECK,
                               TAG_ABS_X, TAG_TILTED, TAG_TIPPED_DECK), last, RESULT_WINDOW)
    unstable = value_loss_instability(scalars)
    out = seed_result3(m[TAG_SUCCESS], m[TAG_TIMEOUT], m[TAG_OVER_DECK], complete,
                       killed=killed, unstable_iters=len(unstable))
    out.update({"last_iter": last, "max_iter": max_iter, "window": RESULT_WINDOW,
                "window_finite": finite, "ep_len": m[TAG_LEN], "fallen": m[TAG_FALLEN],
                "cube_abs_x": m[TAG_ABS_X], "cube_tilted": m[TAG_TILTED],
                "tipped_over_deck": m[TAG_TIPPED_DECK], "unstable_first": unstable[:20],
                "success_def": SEATED_NOTE})
    return out


def kill_verdict3(ep_len: float, time_out: float, success: float) -> dict:
    """Stand3 kill rule at model_1000: Stand2's `kill_verdict`, except that the
    episode-length clause is skipped when the last-100 mean success is
    >= RESULT_MIN_SUCCESS (a quick placement ends the episode early; see the
    Stand3 predeclared rules above). The (time_out + success) clause is kept."""
    out = kill_verdict(ep_len, time_out, success)
    skip = bool(success >= RESULT_MIN_SUCCESS)
    if skip:
        out["reasons"] = [r for r in out["reasons"] if not r.startswith("mean episode length")]
        out["verdict"] = "KILL" if out["reasons"] else "CONTINUE"
    out["length_clause_skipped"] = skip
    return out


def evaluate_kill3(scalars: dict, at_iter: int = KILL_AT_ITER) -> dict:
    m = window_means(scalars, (TAG_LEN, TAG_TIMEOUT, TAG_SUCCESS, TAG_FALLEN), at_iter, KILL_WINDOW)
    out = kill_verdict3(m[TAG_LEN], m[TAG_TIMEOUT], m[TAG_SUCCESS])
    out.update({"at_iter": at_iter, "window": KILL_WINDOW, "fallen": m[TAG_FALLEN]})
    return out
