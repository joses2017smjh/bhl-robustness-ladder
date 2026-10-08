"""Lift, success, diagnostic and rule terms for CubeToShelfStand4 (`task_v2_env_cfg`).

CubeToShelfStand4 ("roll-proof lift") is a DIFFERENT task from CubeToShelfStand3,
CubeToShelfStand2 and CubeToShelf. Its numbers are never compared with theirs.

Why it exists (SLURM_JOBS.md, "Corrections recorded 2026-10-01"; docs/
SOLUTIONS_2026-10-01.md section 4, row C2). Stand3 (`21443327`, NEGATIVE) earned
its 15-weight `lifting_object` income by RE-ORIENTING a cube that rests on its
support, not by lifting it: `coop_lift_mdp.object_is_lifted` tests the cube
centre height only. In the Stand3 replay (`21470828`), on steps with the centre
above 0.61 m the lowest corner sat within 1 cm of the plinth or deck top on
97.4 % (s0) / 79.7 % (s1) of steps, and a cube within 8 deg of flat was above
0.61 m on 2 of 44,990 (s0) and 0 of 40,510 (s1) steps. A cube rolled 45 deg onto
an edge raises its centre by 0.14 (sqrt 2 - 1) = 0.058 m, standing on a corner
by 0.14 (sqrt 3 - 1) = 0.102 m, both above the 0.04-0.06 m curriculum levels.

The user overrode the predeclared Stand4 funding rule on 2026-10-01
(SLURM_JOBS.md, "User authorizations recorded 2026-10-01 15:45").

What changes from Stand3 -- these and nothing else
--------------------------------------------------
0. Hand colliders (label: MODIFIED relative to Stand3). The robot USD that
   every TaskV2 task spawns (`HUMANOID_LITE_CFG`, `berkeley_humanoid_lite.usd`)
   has NO collision geometry on `arm_*_hand_link`, the links every reach /
   clamp / pinch term measures; the only arm colliders are the forearm
   (`elbow_roll`, capsule r 0.03 m reaching 8 cm past the wrist) and upper-arm
   (`shoulder_yaw`) capsules, so the fingers pass through the cube (Stand3
   replay: hand-link contact force 0 on every step). Stand4 spawns both robots
   from the repo's existing overlay
   `assets/cloth/berkeley_humanoid_lite_hand_colliders.usda` (sublayers the
   untouched upstream USD, adds one 64-vertex convex-hull collider per hand
   link; no joint, mass or visual change; it collides on v60: ClothSort
   `21300603`). Stand4 only: no other task's spawn changes.
   Stated, not tuned: with Stand3's reset jitter, numpy FK puts at least one
   hand hull inside the cube on ~61 % of resets (median depth 0.45 cm, 5th
   percentile 4 cm; nominal pose 2.3-2.8 cm clear); PhysX depenetrates it.
1. Lift pay. `lifting_object` keeps its name, its weight (15), its
   `minimal_height` parameter (the lift curriculum writes it) and everything it
   paid on in Stand3 (centre > spawn + minimal_height, x pinch kernel, x upright
   gate), and additionally pays only when the ROLL-PROOF gate holds:
   * the cube's lowest corner is >= CORNER_CLEARANCE (0.02 m) above EVERY
     support surface it could rest on: the floor always, and the plinth top /
     deck tops whose footprint overlaps the cube's world-xy footprint (the AABB
     of its eight corners, which is conservative); and
   * the cube tilt is <= LIFT_TILT_MAX_DEG (15 deg).
   Tilt is the angle between the cube's OWN z axis and world up
   (`stand_mdp.body_z_tilt_deg`, read through `quat_order.unpack_wxyz`: the
   2976f36 helpers, xyzw on v60). A cube rolled 90 deg onto another face
   therefore reads 90 deg, not "flat": that is what makes the gate roll-proof
   (and what keeps a cube tipped over the deck lip from counting as seated).
   The same definition serves BOTH tilt clauses, this 15-deg lift clause and
   the 8-deg success clause (TILT_NOTE; stated for both on review 2026-10-02,
   before any Stand4 run). The Stand3 replay's "face tilt" (the most vertical
   body axis, `scripts/bench/stand3_replay.py` `cube_face_tilt`, which the C2
   thresholds were read from) is NOT used: it reads a 90-deg roll as 0.
   The lowest corner comes from the same quaternion: centre z minus
   half x (|R20| + |R21| + |R22|).
   NOT changed ("changes ONLY"): `lift_progress` (weight 2.0) still pays
   centre-height progress, including progress bought by rolling; it is bounded
   by 2.0 x 0.04 = 0.08 per step against the 15 x 0.04 = 0.60 per step the
   rolled cube used to earn through `lifting_object`. Also NOT changed:
   `lift_height_curriculum` still promotes `minimal_height` (0.04 -> 0.06 cap)
   on Stand3's test (centre above spawn + level AND pinch distance < 0.20 m),
   which a rolled cube can still meet; the level is logged
   (`Curriculum/lift_height`), and only the 15-weight pay is roll-proof.
2. Success (`terminations.success`, still read by `placed` = success_bonus x
   STAND3_PLACED_WEIGHT): Stand3's seated test (`stand_mdp.seated_mask`) AND
   tilt <= SEAT_TILT_MAX_DEG (8 deg) AND released, all three on the same step,
   held SEAT_HOLD_STEPS (12) consecutive steps. Tilt as in 1 (the cube's own z
   axis vs world up): a cube rolled 90 deg onto another face never seats, even
   lying flat on the deck with the hands off. Released = no robot link in
   contact with the cube: a contact sensor on the cube, filtered one-to-many
   against the 13 collider-bearing links of each robot (26 bodies:
   RELEASE_LINKS), reports every link's normal force on the cube, and the cube
   is released iff every one of them is < RELEASE_FORCE_N (1.0 N; the threshold
   the Stand3 replay's hand-contact proxy and `undesired_contacts` use). A NaN
   force counts as contact.
3. Logging. Per step and per env, `env._bhl_s4` (also `env.extras["stand4"]`)
   holds the lift curriculum level (`minimal_height`), the pinch distance, the
   pinch kernel, the upright gate, the centre clause, the cube tilt, the
   lowest-corner clearance, the roll-proof gate, the lift pay, the release
   state, the largest robot-link force on the cube and the seated/hold state
   (CACHE_KEYS), so a replay can recompute the paid share exactly
   (lift_pay = centre_ok x roll_ok x pinch_kernel x upright_gate). Batch means
   go to TensorBoard as `Curriculum/*` (no gradient), beside the inherited
   `Curriculum/lift_height` (= minimal_height) and `Curriculum/upright_gate`.

Labels: LEARNED crew policies; ORACLE privileged observations exactly as Stand3
(LABELS_NOTE says which); MODIFIED hand colliders (the overlay) relative to Stand3.

Launchers: slurm/repo20260923/gpu_v2_stand4_smoke.sbatch (+ inner_v2_stand4_smoke.sh)
and gpu_v2_stand4_train.sbatch. Both read their verdicts from JSON written by the
pure functions at the end of this file (`config_check`, `smoke_verdict`,
`evaluate_kill`, `evaluate_seed4`, `pair_verdict4`), never from exit codes.

Pure kernels and rule functions import without Isaac Lab (`tests/test_stand4.py`
runs off-simulator; the launchers load this file by path). The env-facing terms
reach Isaac Lab only through `stand_mdp`'s lazy `_coop()` / `_v2()` and a lazy
`quat_order` import.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import torch


def _load_stand_mdp():
    """`stand_mdp`: the package's own module inside Isaac (same object the Stand3
    cfg uses), else loaded by path (tests, launchers)."""
    mod = sys.modules.get("bhl_robust.tasks.stand_mdp")
    if mod is not None:
        return mod
    path = Path(__file__).with_name("stand_mdp.py")
    spec = importlib.util.spec_from_file_location("_stand_mdp_for_stand4", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sm = _load_stand_mdp()

# ---------------------------------------------------------------- constants

#: The hand-collider overlay (Stand4 robots only). Repo root = parents[3].
HAND_COLLIDER_USD = (Path(__file__).resolve().parents[3]
                     / "assets" / "cloth" / "berkeley_humanoid_lite_hand_colliders.usda")

#: Roll-proof lift gate.
CORNER_CLEARANCE = 0.02
LIFT_TILT_MAX_DEG = 15.0
#: Success: seated (Stand3) and flat within this, and released.
SEAT_TILT_MAX_DEG = 8.0
RELEASE_FORCE_N = 1.0

#: Geometry (m), Stand3's: cube half 0.14; plinth 0.26 x 0.26, top 0.41
#: (`CubeToShelfStandCfg`: `_box("plinth", (0.26, 0.26, h), ...)`, h = 0.41);
#: decks |x| in [0.17, 0.47], |y| <= 0.15, top 0.43 (`CubeToShelfStand3Cfg`).
CUBE_HALF = sm.CUBE_HALF
FLOOR_Z = 0.0
PLINTH_HALF = 0.13
#: (name, x_lo, x_hi, y_lo, y_hi, top) of every raised support, env-local.
SUPPORTS = (
    ("plinth", -PLINTH_HALF, PLINTH_HALF, -PLINTH_HALF, PLINTH_HALF, sm.PLINTH_TOP),
    ("deck_pos", sm.DECK_EDGE, sm.DECK_EDGE + sm.DECK_LEN,
     -sm.DECK_WIDTH / 2.0, sm.DECK_WIDTH / 2.0, sm.DECK_TOP),
    ("deck_neg", -(sm.DECK_EDGE + sm.DECK_LEN), -sm.DECK_EDGE,
     -sm.DECK_WIDTH / 2.0, sm.DECK_WIDTH / 2.0, sm.DECK_TOP),
)

#: Release sensing. The cube's contact sensor (scene name) and the robot links
#: it is filtered against: every link that carries collision geometry with the
#: overlay (base, upper arm, forearm, hand, thigh, shank, foot; one shape each;
#: campaign-stand4/step0/usd_colliders.json + the overlay's two hand hulls).
#: Links without a collider cannot touch the cube. ONE EXPLICIT NAME PER
#: PATTERN: PhysX `create_rigid_contact_view` pairs each filter pattern's i-th
#: match with the i-th sensor body (one cube per env), so a pattern must match
#: exactly one prim per env -- `arm_.*_hand_link` (two per env) would not.
CUBE_SENSOR = "cube_contact"
RELEASE_LINKS = ("base",
                 "arm_left_shoulder_yaw", "arm_right_shoulder_yaw",
                 "arm_left_elbow_roll", "arm_right_elbow_roll",
                 "arm_left_hand_link", "arm_right_hand_link",
                 "leg_left_hip_pitch", "leg_right_hip_pitch",
                 "leg_left_knee_pitch", "leg_right_knee_pitch",
                 "leg_left_ankle_roll", "leg_right_ankle_roll")
RELEASE_ROBOTS = ("robot_a", "robot_b")
RELEASE_FILTER_EXPRS = tuple(f"{{ENV_REGEX_NS}}/{r}/{link}"
                             for r in RELEASE_ROBOTS for link in RELEASE_LINKS)
#: 2 robots x (base + 2 x 6 limb links).
N_RELEASE_BODIES = 26
assert len(RELEASE_FILTER_EXPRS) == N_RELEASE_BODIES == len(set(RELEASE_FILTER_EXPRS))
assert not any(".*" in e or "*" in e for e in RELEASE_FILTER_EXPRS)

HOLD_KEY = "_bhl_s4_hold"          # Stand3's counter is "_bhl_deck_hold"
CACHE_KEY = "_bhl_s4"
EXTRAS_KEY = "stand4"
#: Per-step cache keys (env._bhl_s4 / env.extras["stand4"]): enough to recompute
#: the lift pay (centre_ok x roll_ok x pinch_kernel x upright_gate) and the
#: success predicate (seated & tilt & released, held) of every env on every step.
LIFT_CACHE_KEYS = ("minimal_height", "pinch_d", "pinch_kernel", "upright_gate", "centre_ok",
                   "tilt_deg", "corner_clearance", "roll_ok", "lift_pay")
SUCCESS_CACHE_KEYS = ("released", "robot_force_max", "seated", "placed_now", "hold")
CACHE_KEYS = LIFT_CACHE_KEYS + SUCCESS_CACHE_KEYS

# ------------------------------------------------------ predeclared rules
#: Training budget and the frozen two-seed rule (gpu_v2_stand4_train.sbatch).
STAND4_MAX_ITER = 8000
COMPLETE_ITER = STAND4_MAX_ITER - 1          # 7999
RESULT_WINDOW = sm.RESULT_WINDOW             # 200
RESULT_MIN_SUCCESS = sm.RESULT_MIN_SUCCESS   # 0.10
STAND4_SEEDS = (0, 1)

#: Verbatim from the workstream task (2026-10-01); also in the launcher header.
PREDECLARED_RULE = (
    "PREDECLARED RULE (frozen; Stand3's structure): the kill rule at model_1000 exactly as "
    "Stand3's launcher applies it (copy its condition and window verbatim); a seed is COMPLETE "
    "iff it reaches iteration 7999; Stand4 PASSES iff >= 1 of 2 seeds has last-200-iteration "
    "success >= 0.10, where success is the Stand4 termination above (seated, tilt <= 8 deg, "
    "released). Otherwise NEGATIVE; INCOMPLETE if a seed is neither complete nor stopped by the "
    "kill rule. Stand4 is a different task from Stand3 and CubeToShelf and is never compared "
    "with their numbers. Labels: LEARNED crew policies; ORACLE privileged observations exactly "
    "as Stand3 uses them (say which); MODIFIED hand colliders (the overlay) relative to Stand3."
)
#: How the clauses combine (Stand3's `pair_result3` precedence, stated up front).
PRECEDENCE_NOTE = (
    "PASS if any seed that is COMPLETE and not killed has last-200 success >= 0.10, whatever the "
    "other seed did; else INCOMPLETE if any seed is neither COMPLETE nor killed by the rule; "
    "else NEGATIVE"
)
#: COMPLETE is Stand3's completeness check, as the rule's "Stand3's structure"
#: requires (decided on review 2026-10-02, before any Stand4 run, so that a
#: Stand4 seed is never easier to call COMPLETE than a Stand3 seed): the events
#: reach iteration 7999 AND every Loss/value, success and time_out value in the
#: last 200 iterations is finite AND that window holds a Loss/value value
#: (`stand_mdp.evaluate_seed3`'s clauses). A seed that logged to 7999 but went
#: non-finite in its last 200 iterations is INCOMPLETE: neither a PASS nor a
#: NEGATIVE.
COMPLETE_NOTE = ("COMPLETE iff the run's events reach iteration 7999 AND every Loss/value, "
                 "success and time_out value in its last 200 iterations is finite AND that window "
                 "holds a Loss/value value (Stand3's completeness check); the Loss/value "
                 "instability count is reported, not gating")
TASK4_NOTE = ("CubeToShelfStand4 (roll-proof lift): a different task from CubeToShelfStand3 and "
              "CubeToShelf; never compared with their numbers")
#: The one tilt definition, for both tilt clauses (stated on review 2026-10-02).
TILT_NOTE = ("tilt = the angle between the cube's own z axis and world up "
             "(stand_mdp.body_z_tilt_deg via quat_order.unpack_wxyz), for both the 15-deg lift "
             "clause and the 8-deg success clause; a cube rolled 90 deg onto another face reads "
             "90 deg, so it never earns the lift pay and never seats (the Stand3 replay's face "
             "tilt, the most vertical body axis, is not used: it reads a 90-deg roll as 0)")
SUCCESS4_NOTE = ("success = Stand3's seated test (centre over a deck by 2 cm, |y| < 0.10, within "
                 "2 cm of the seated height, speed < 0.05) AND cube tilt <= 8 deg, tilt = the "
                 "angle between the cube's own z axis and world up (a 90-deg roll onto another "
                 "face reads 90 deg and never seats) AND released (every robot-link normal force "
                 "on the cube < 1.0 N, filtered contact sensor), all on the same step, for 12 "
                 "consecutive steps")
LABELS_NOTE = ("LEARNED crew policies (one PPO actor drives both robots, blind); ORACLE privileged "
               "observations exactly as Stand3: the actor's object_pos_a/b (simulator cube "
               "position in each robot's root frame) and the critic's base_lin_vel_a/b, "
               "object_lin_vel, object_ang_vel; MODIFIED hand colliders relative to Stand3 (both "
               "robots spawn from assets/cloth/berkeley_humanoid_lite_hand_colliders.usda)")
PASS_LABEL = "PASS: learned roll-proof placement (Stand4)"

#: TensorBoard tags read by the rules (Stand3's names).
TAG_LEN, TAG_TIMEOUT = sm.TAG_LEN, sm.TAG_TIMEOUT
TAG_SUCCESS, TAG_FALLEN, TAG_VALUE = sm.TAG_SUCCESS, sm.TAG_FALLEN, sm.TAG_VALUE
#: The Stand4 diagnostics (CurrTerm names in CubeToShelfStand4Cfg), and the
#: inherited curriculum level and gate.
DIAG_TERMS = ("pinch_dist", "cube_tilt_deg", "corner_clear", "roll_ok", "lift_paid",
              "released", "robot_force_max")
TAG_LIFT_LEVEL = "Curriculum/lift_height"
TAG_GATE = "Curriculum/upright_gate"
DIAG_TAGS = tuple(f"Curriculum/{n}" for n in DIAG_TERMS)

# ------------------------------------------------------------ pure kernels


def _normalised(w, x, y, z):
    n = torch.sqrt(w * w + x * x + y * y + z * z).clamp_min(1e-12)
    return w / n, x / n, y / n, z / n


def cube_half_extents(w, x, y, z, half: float = CUBE_HALF):
    """(hx, hy, hz): half-extents of the rotated cube's world AABB, from the
    (w, x, y, z) quaternion columns. hz is how far the lowest corner sits below
    the centre: half x (|R20| + |R21| + |R22|)."""
    w, x, y, z = _normalised(w, x, y, z)
    r00, r01, r02 = 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)
    r10, r11, r12 = 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)
    r20, r21, r22 = 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)
    hx = half * (r00.abs() + r01.abs() + r02.abs())
    hy = half * (r10.abs() + r11.abs() + r12.abs())
    hz = half * (r20.abs() + r21.abs() + r22.abs())
    return hx, hy, hz


def lowest_corner_z(p: torch.Tensor, w, x, y, z, half: float = CUBE_HALF) -> torch.Tensor:
    """Height of the cube's lowest corner (p: (N, 3) centre)."""
    return p[:, 2] - cube_half_extents(w, x, y, z, half)[2]


def support_clearance(p: torch.Tensor, w, x, y, z, half: float = CUBE_HALF,
                      supports=SUPPORTS, floor_z: float = FLOOR_Z) -> torch.Tensor:
    """Lowest corner minus the highest support surface the cube could rest on.

    Candidates: the floor always; each raised support whose footprint overlaps
    the cube's world-xy footprint (AABB of its corners; touching edges do not
    count). `p` is the env-local centre (N, 3). The roll-proof gate asks this to
    be >= CORNER_CLEARANCE.
    """
    hx, hy, hz = cube_half_extents(w, x, y, z, half)
    low = p[:, 2] - hz
    clear = low - floor_z
    for _name, x0, x1, y0, y1, top in supports:
        under = ((p[:, 0] + hx > x0) & (p[:, 0] - hx < x1)
                 & (p[:, 1] + hy > y0) & (p[:, 1] - hy < y1))
        clear = torch.where(under, torch.minimum(clear, low - top), clear)
    return clear


def cube_tilt_deg(w, x, y, z) -> torch.Tensor:
    """Angle (deg) between the cube's own z axis and world up (Stand3's
    `body_z_tilt_deg`). 90 deg for a cube rolled onto another face."""
    w, x, y, z = _normalised(w, x, y, z)
    return sm.body_z_tilt_deg(w, x, y, z)


def roll_proof_mask(p: torch.Tensor, w, x, y, z, clearance: float = CORNER_CLEARANCE,
                    tilt_max_deg: float = LIFT_TILT_MAX_DEG) -> torch.Tensor:
    """The Stand4 lift gate: lowest corner >= `clearance` above every support it
    could rest on AND tilt <= `tilt_max_deg`. NaN anywhere -> False."""
    return ((support_clearance(p, w, x, y, z) >= clearance)
            & (cube_tilt_deg(w, x, y, z) <= tilt_max_deg))


def centre_clause(z_centre: torch.Tensor, spawn_z: float, minimal_height: float) -> torch.Tensor:
    """Stand3's height clause of `object_is_lifted`: centre > spawn + minimal_height."""
    return z_centre > (spawn_z + minimal_height)


def lift_pay(z_centre, p, w, x, y, z, spawn_z: float, minimal_height: float,
             pinch_kernel, upright_gate, clearance: float = CORNER_CLEARANCE,
             tilt_max_deg: float = LIFT_TILT_MAX_DEG) -> torch.Tensor:
    """The whole Stand4 `lifting_object` value (before its weight): Stand3's
    centre clause x pinch kernel x upright gate, x the roll-proof gate."""
    ok = centre_clause(z_centre, spawn_z, minimal_height) & roll_proof_mask(
        p, w, x, y, z, clearance, tilt_max_deg)
    return ok.float() * pinch_kernel * upright_gate


def robot_force_max(force_matrix: torch.Tensor) -> torch.Tensor:
    """Largest robot-link normal-force magnitude on the cube, per env.

    `force_matrix`: (N, ..., 3), e.g. the sensor's (N, 1, 26, 3). NaN propagates.
    """
    if force_matrix is None:
        raise ValueError("no force matrix: the cube sensor needs filter_prim_paths_expr")
    f = force_matrix.reshape(force_matrix.shape[0], -1, 3)
    if f.shape[1] == 0:
        raise ValueError("force matrix has no filter bodies: nothing would ever count as contact")
    mag = torch.linalg.vector_norm(f, dim=-1)
    nan = torch.isnan(mag).any(dim=-1)
    worst = mag.nan_to_num(nan=0.0).max(dim=-1).values
    return torch.where(nan, torch.full_like(worst, float("nan")), worst)


def released_mask(force_matrix: torch.Tensor, threshold: float = RELEASE_FORCE_N) -> torch.Tensor:
    """True iff every filtered robot link pushes on the cube with < `threshold` N.
    A NaN force is contact (not released)."""
    return robot_force_max(force_matrix) < threshold


def placed_mask(p: torch.Tensor, speed: torch.Tensor, tilt_deg: torch.Tensor,
                released: torch.Tensor, tilt_max_deg: float = SEAT_TILT_MAX_DEG) -> torch.Tensor:
    """Stand4 success, this step: Stand3's seated test AND tilt <= 8 deg AND released."""
    return sm.seated_mask(p, speed) & (tilt_deg <= tilt_max_deg) & released.bool()


# ------------------------------------------------------- env-facing terms


def _t(v):
    return sm._coop()._t(v)


def _cube_quat_wxyz(env):
    from bhl_robust.quat_order import unpack_wxyz
    return unpack_wxyz(_t(env.scene["object"].data.root_quat_w))


def _cache(env) -> dict:
    """Per-step, per-env Stand4 state (one dict object, also in env.extras)."""
    c = getattr(env, CACHE_KEY, None)
    if c is None:
        c = {}
        setattr(env, CACHE_KEY, c)
    extras = getattr(env, "extras", None)
    if isinstance(extras, dict):
        extras[EXTRAS_KEY] = c
    return c


def _nan_like(n: int, device) -> torch.Tensor:
    return torch.full((n,), float("nan"), device=device)


def stand4_object_is_lifted(env, minimal_height: float, clearance: float = CORNER_CLEARANCE,
                            tilt_max_deg: float = LIFT_TILT_MAX_DEG,
                            gate_free: float = sm.GATE_FREE,
                            gate_std: float = sm.GATE_STD) -> torch.Tensor:
    """Stand4 `lifting_object`: Stand3's `gated_object_is_lifted` x the roll-proof gate.

    Keeps the parameter name `minimal_height` (the lift curriculum writes it on
    the term named `lifting_object`). The Stand3 product is computed by Stand3's
    own function, so the only difference is the gate.
    """
    base = sm.gated_object_is_lifted(env, minimal_height=minimal_height,
                                     gate_free=gate_free, gate_std=gate_std)
    p, _ = sm._obj_state(env)
    w, x, y, z = _cube_quat_wxyz(env)
    clear = support_clearance(p, w, x, y, z)
    tilt = cube_tilt_deg(w, x, y, z)
    ok = (clear >= clearance) & (tilt <= tilt_max_deg)
    pay = base * ok.to(base.dtype)
    # diagnostics only: nothing below feeds the returned value
    coop = sm._coop()
    zc = _t(env.scene["object"].data.root_pos_w)[:, 2]
    d = getattr(env, "_bhl_pinch_d", None)
    if getattr(env.cfg, "gate_lift_on_pinch", True):
        kern = coop._pinch_weight(env)          # the factor object_is_lifted used
    else:
        kern = torch.ones_like(zc)
    c = _cache(env)
    c.update({
        "minimal_height": float(minimal_height),
        "pinch_d": d.detach() if d is not None else _nan_like(p.shape[0], p.device),
        "pinch_kernel": kern.detach(),
        "upright_gate": sm.upright_gate(env, gate_free, gate_std).detach(),
        "centre_ok": centre_clause(zc, float(env.cfg.object_spawn_z), float(minimal_height)),
        "tilt_deg": tilt.detach(),
        "corner_clearance": clear.detach(),
        "roll_ok": ok,
        "lift_pay": pay.detach(),
    })
    return pay


def stand4_cube_placed(env, hold_steps: int = sm.SEAT_HOLD_STEPS,
                       tilt_max_deg: float = SEAT_TILT_MAX_DEG,
                       release_force_n: float = RELEASE_FORCE_N,
                       sensor_name: str = CUBE_SENSOR) -> torch.Tensor:
    """Success: `placed_mask` true for `hold_steps` consecutive steps.

    Call it from the `success` termination ONLY (`placed` reads the termination
    through `stand_mdp.success_bonus`), so the hold counter advances once per step.
    """
    p, speed = sm._obj_state(env)
    w, x, y, z = _cube_quat_wxyz(env)
    tilt = cube_tilt_deg(w, x, y, z)
    fm = _t(env.scene[sensor_name].data.force_matrix_w)
    worst = robot_force_max(fm)
    rel = worst < release_force_n
    seated = sm.seated_mask(p, speed)
    now = seated & (tilt <= tilt_max_deg) & rel
    held = sm._v2()._held(env, HOLD_KEY, now, hold_steps)
    _cache(env).update({"released": rel, "robot_force_max": worst.detach(), "seated": seated,
                        "placed_now": now, "hold": getattr(env, HOLD_KEY).detach()})
    return held


# ----------------------------------------------- diagnostic curriculum terms
# Logged as Curriculum/<name>, batch means over all envs, no gradient (Stand3's
# pattern). State-derived where possible; the pinch distance and lift pay come
# from this step's reward pass (NaN before the first step).


def pinch_distance_mean(env, env_ids: Sequence[int]) -> float:
    """Batch-mean pinch distance (m): the reach terms' hand-to-pinch-point mean."""
    d = getattr(env, "_bhl_pinch_d", None)
    return float(d.float().mean()) if d is not None else float("nan")


def cube_tilt_mean(env, env_ids: Sequence[int]) -> float:
    """Batch-mean cube tilt (deg), body z vs world up."""
    return float(cube_tilt_deg(*_cube_quat_wxyz(env)).mean())


def corner_clearance_mean(env, env_ids: Sequence[int]) -> float:
    """Batch-mean lowest-corner clearance over the supports under the cube (m)."""
    p, _ = sm._obj_state(env)
    return float(support_clearance(p, *_cube_quat_wxyz(env)).mean())


def roll_ok_share(env, env_ids: Sequence[int], clearance: float = CORNER_CLEARANCE,
                  tilt_max_deg: float = LIFT_TILT_MAX_DEG) -> float:
    """Batch share with the roll-proof gate open (corner clear AND tilt small)."""
    p, _ = sm._obj_state(env)
    return float(roll_proof_mask(p, *_cube_quat_wxyz(env), clearance=clearance,
                                 tilt_max_deg=tilt_max_deg).float().mean())


def lift_paid_share(env, env_ids: Sequence[int]) -> float:
    """Batch share whose `lifting_object` paid > 0 on the last reward pass."""
    v = _cache(env).get("lift_pay")
    return float((v > 0).float().mean()) if v is not None else float("nan")


def released_share(env, env_ids: Sequence[int], release_force_n: float = RELEASE_FORCE_N,
                   sensor_name: str = CUBE_SENSOR) -> float:
    """Batch share with no robot link pushing on the cube (>= release_force_n)."""
    fm = _t(env.scene[sensor_name].data.force_matrix_w)
    return float(released_mask(fm, release_force_n).float().mean())


def robot_force_max_mean(env, env_ids: Sequence[int], sensor_name: str = CUBE_SENSOR) -> float:
    """Batch-mean largest robot-link normal force on the cube (N)."""
    fm = _t(env.scene[sensor_name].data.force_matrix_w)
    return float(robot_force_max(fm).nan_to_num(nan=0.0).mean())


# --------------------------------------------------------- rule evaluation


def kill_verdict(ep_len: float, time_out: float, success: float) -> dict:
    """The kill rule: Stand3's `kill_verdict3`, called, not copied."""
    return sm.kill_verdict3(ep_len, time_out, success)


def evaluate_kill(scalars: dict) -> dict:
    """Stand3's `evaluate_kill3`: model_1000, iterations 901-1000."""
    return sm.evaluate_kill3(scalars)


def seed_result4(last_iter: int, success: float, killed: bool = False,
                 bad_config: bool = False, window_finite: bool = True,
                 has_value: bool = True) -> dict:
    """Per-seed Stand4 result, the frozen rule's clauses only.

    COMPLETE iff the run reached iteration COMPLETE_ITER (7999) AND its last
    RESULT_WINDOW iterations are finite (`window_finite`: Loss/value, success,
    time_out) AND hold a Loss/value value (`has_value`): Stand3's completeness
    check (COMPLETE_NOTE; `evaluate_seed4` computes both flags). A seed killed
    by the rule is decided and does not pass. `bad_config` (wrong task/runner
    in params/*.yaml) is never a result: not complete, not decided.
    """
    complete = ((not bad_config) and last_iter >= COMPLETE_ITER
                and bool(window_finite) and bool(has_value))
    killed = bool(killed) and not bad_config
    decided = complete or killed
    passes = complete and (not killed) and bool(success >= RESULT_MIN_SUCCESS)
    return {"complete": bool(complete), "killed": bool(killed), "decided": bool(decided),
            "passes": bool(passes), "bad_config": bool(bad_config),
            "window_finite": bool(window_finite), "has_value": bool(has_value),
            "last_iter": int(last_iter), "success": success}


def pair_verdict4(seeds: Sequence[dict]) -> str:
    """The frozen two-seed rule (PREDECLARED_RULE), with PRECEDENCE_NOTE:
    PASS if any complete, unkilled seed has last-200 success >= 0.10; else
    INCOMPLETE if any seed is neither complete nor killed; else NEGATIVE."""
    n = len(seeds)
    k = sum(1 for s in seeds if s["passes"])
    if k:
        return f"{PASS_LABEL} ({k}/{n} seeds)"
    bad = [i for i, s in enumerate(seeds) if not s["decided"]]
    if bad:
        return f"INCOMPLETE (no verdict: seed index {bad} neither complete nor killed by the rule)"
    kl = sum(1 for s in seeds if s["killed"])
    return f"NEGATIVE (0/{n} seeds with last-200 success >= {RESULT_MIN_SUCCESS:.2f}; {kl} killed)"


def evaluate_seed4(scalars: dict, killed: bool = False, bad_config: bool = False) -> dict:
    """Per-seed evaluation from the run's TensorBoard scalars (tag -> [(step, value)]).

    last_iter = the last iteration with an `Episode_Termination/success` value;
    success = its mean over (last_iter - 200, last_iter]. The completeness
    flags are Stand3's (`stand_mdp.evaluate_seed3`, same lines): window_finite
    (every Loss/value, success and time_out value in that window is finite) and
    has_value (the window holds a Loss/value value); both gate COMPLETE.
    Reported beside the verdict, not part of it: the Loss/value instability
    count, time_out, episode length, fall share and the Stand4 diagnostics.
    """
    steps = [s for s, _ in scalars.get(TAG_SUCCESS, [])]
    last = max(steps) if steps else -1
    lo = last - RESULT_WINDOW
    finite = all(math.isfinite(x) for t in (TAG_VALUE, TAG_SUCCESS, TAG_TIMEOUT)
                 for s, x in scalars.get(t, []) if lo < s <= last)
    has_value = any(lo < s <= last for s, _ in scalars.get(TAG_VALUE, []))
    tags = (TAG_LEN, TAG_TIMEOUT, TAG_SUCCESS, TAG_FALLEN, TAG_LIFT_LEVEL, TAG_GATE) + DIAG_TAGS
    m = sm.window_means(scalars, tags, last, RESULT_WINDOW)
    out = seed_result4(last, m[TAG_SUCCESS], killed=killed, bad_config=bad_config,
                       window_finite=finite, has_value=has_value)
    unstable = sm.value_loss_instability(scalars)
    out.update({"window": RESULT_WINDOW, "complete_iter": COMPLETE_ITER,
                "time_out": m[TAG_TIMEOUT], "ep_len": m[TAG_LEN], "fallen": m[TAG_FALLEN],
                "lift_level": m[TAG_LIFT_LEVEL], "upright_gate": m[TAG_GATE],
                "diagnostics_last200": {t: m[t] for t in DIAG_TAGS},
                "reported_not_gated": {"unstable_iters": len(unstable),
                                       "unstable_first": unstable[:20]},
                "complete_def": COMPLETE_NOTE, "success_def": SUCCESS4_NOTE,
                "tilt_def": TILT_NOTE})
    return out


# ------------------------------------------------- run config check (YAML)
# The training launcher's RUNNER-CHECK and the smoke both call `config_check`
# on a run's params/agent.yaml + params/env.yaml, so a pattern that does not
# match Isaac Lab's dump fails the smoke, never a scored seed.

FUNC_LIFT4 = "bhl_robust.tasks.stand4_mdp:stand4_object_is_lifted"
FUNC_SUCCESS4 = "bhl_robust.tasks.stand4_mdp:stand4_cube_placed"
FUNC_PLACED = "bhl_robust.tasks.stand_mdp:success_bonus"
FUNC_ACTION_RATE = "bhl_robust.tasks.stand_mdp:action_rate_clipped_l2"
OVERLAY_NAME = HAND_COLLIDER_USD.name
BASE_USD_NAME = "berkeley_humanoid_lite.usd"
#: What InteractiveScene writes for {ENV_REGEX_NS} (it formats the cfg in place).
ENV_NS_RESOLVED = "/World/envs/env_.*"
CURRICULUM4_TERMS = ("lift_height", "upright_gate") + DIAG_TERMS


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _unq(v: str) -> str:
    v = v.strip()
    return v[1:-1] if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"" else v


def yaml_block(text: str, path: Sequence[str]) -> list[str] | None:
    """The lines of the mapping at `path` in an Isaac Lab `dump_yaml` text (two
    spaces per level; PyYAML writes list items at their key's indentation), key
    line first; None if absent. Text-only on purpose: the dump carries
    `!!python/tuple` tags that `yaml.safe_load` rejects."""
    lines = text.splitlines()
    lo, hi, start = 0, len(lines), None
    for depth, key in enumerate(path):
        ind = 2 * depth
        start = None
        for i in range(lo, hi):
            s = lines[i].strip()
            if _indent(lines[i]) == ind and (s == f"{key}:" or s.startswith(f"{key}: ")):
                start = i
                break
        if start is None:
            return None
        end = start + 1
        while end < hi:
            ln = lines[end]
            if ln.strip():
                k = _indent(ln)
                if k < ind or (k == ind and not ln.lstrip().startswith("- ")):
                    break
            end += 1
        lo, hi = start + 1, end
    return lines[start:hi]


def yaml_value(block: list[str] | None, key: str, depth: int) -> str | None:
    """Scalar `key: value` directly inside `block` (the key at 2 x depth spaces)."""
    if not block:
        return None
    for ln in block[1:]:
        s = ln.strip()
        if _indent(ln) == 2 * depth and s.startswith(f"{key}:"):
            v = s[len(key) + 1:]
            if v == "" or v.startswith(" "):
                return _unq(v)
    return None


def yaml_list(block: list[str] | None, key: str, depth: int) -> list[str] | None:
    """Items of the list `key:` directly inside `block`; None if the key is absent."""
    if not block:
        return None
    for i, ln in enumerate(block):
        s = ln.strip()
        if _indent(ln) != 2 * depth:
            continue
        if s == f"{key}: []":
            return []
        if s == f"{key}:":
            out = []
            for ln2 in block[i + 1:]:
                if _indent(ln2) == 2 * depth and ln2.lstrip().startswith("- "):
                    out.append(_unq(ln2.lstrip()[2:]))
                else:
                    break
            return out
    return None


def _norm_ns(path: str) -> str:
    return path.replace(ENV_NS_RESOLVED, "{ENV_REGEX_NS}")


def config_check(agent_yaml: str, env_yaml: str) -> dict:
    """Did this run train CubeToShelfStand4 with Stand3's runner? Every clause
    must hold ({"ok", "failed", "checks"}). Text checks on Isaac Lab's dump."""
    c = {}
    a = agent_yaml.splitlines()
    c["agent_entropy_coef_0.001"] = any(re.fullmatch(r" *entropy_coef: 0\.001", ln) for ln in a)
    c["agent_std_type_log"] = any(re.fullmatch(r" *std_type: log", ln) for ln in a)
    lift = yaml_block(env_yaml, ("rewards", "lifting_object"))
    lp = yaml_block(env_yaml, ("rewards", "lifting_object", "params"))
    c["lifting_object_stand4_x15"] = (yaml_value(lift, "func", 2) == FUNC_LIFT4
                                      and yaml_value(lift, "weight", 2) == "15.0")
    c["lift_gate_0.02m_15deg"] = (yaml_value(lp, "clearance", 3) == repr(CORNER_CLEARANCE)
                                  and yaml_value(lp, "tilt_max_deg", 3) == repr(LIFT_TILT_MAX_DEG))
    placed = yaml_block(env_yaml, ("rewards", "placed"))
    c["placed_success_bonus_x5000"] = (yaml_value(placed, "func", 2) == FUNC_PLACED
                                       and yaml_value(placed, "weight", 2)
                                       == repr(sm.STAND3_PLACED_WEIGHT))
    c["action_rate_clipped"] = (yaml_value(yaml_block(env_yaml, ("rewards", "action_rate_clipped")),
                                           "func", 2) == FUNC_ACTION_RATE)
    succ = yaml_block(env_yaml, ("terminations", "success"))
    sp = yaml_block(env_yaml, ("terminations", "success", "params"))
    c["success_stand4"] = yaml_value(succ, "func", 2) == FUNC_SUCCESS4
    c["success_12steps_8deg_1N"] = (yaml_value(sp, "hold_steps", 3) == str(sm.SEAT_HOLD_STEPS)
                                    and yaml_value(sp, "tilt_max_deg", 3) == repr(SEAT_TILT_MAX_DEG)
                                    and yaml_value(sp, "release_force_n", 3) == repr(RELEASE_FORCE_N)
                                    and yaml_value(sp, "sensor_name", 3) == CUBE_SENSOR)
    for r in RELEASE_ROBOTS:
        u = yaml_value(yaml_block(env_yaml, ("scene", r, "spawn")), "usd_path", 3) or ""
        c[f"{r}_overlay_usd"] = u.endswith("/" + OVERLAY_NAME)
    c["cube_contact_reporting"] = (yaml_value(yaml_block(env_yaml, ("scene", "object", "spawn")),
                                              "activate_contact_sensors", 3) == "true")
    cc = yaml_block(env_yaml, ("scene", CUBE_SENSOR))
    flt = yaml_list(cc, "filter_prim_paths_expr", 2)
    c["cube_sensor_26_filters"] = (flt is not None
                                   and [_norm_ns(f) for f in flt] == list(RELEASE_FILTER_EXPRS)
                                   and _norm_ns(yaml_value(cc, "prim_path", 2) or "")
                                   == "{ENV_REGEX_NS}/object")
    c["curriculum_terms"] = all(yaml_block(env_yaml, ("curriculum", n)) for n in CURRICULUM4_TERMS)
    lh = yaml_block(env_yaml, ("curriculum", "lift_height", "params"))
    c["lift_curriculum_cap_0.06"] = yaml_value(lh, "max_height", 3) == repr(sm.STAND3_LIFT_MAX)
    failed = [k for k, v in c.items() if not v]
    return {"ok": not failed, "failed": failed, "checks": c}


# ------------------------------------------------------------- smoke rule
# gpu_v2_stand4_smoke.sbatch: (A) an in-Isaac probe of the built task, (B) a
# 20-iteration, 1024-env PPO run of the real training path, (C) this verdict.
# SCRIPTED probe + smoke PPO: never a result.

SMOKE_SEED = 100
SMOKE_PROBE_ENVS = 16
SMOKE_ITERS = 20
SMOKE_ENVS = 1024
#: Logged keys the smoke requires (TensorBoard or the console table).
SMOKE_TAGS = ((TAG_LIFT_LEVEL, TAG_GATE) + DIAG_TAGS
              + (TAG_SUCCESS, "Episode_Reward/lifting_object", "Episode_Reward/placed"))
#: Stand3 -> Stand4 differences allowed in cfg.to_dict() (dotted keys; an entry
#: covers its subtree). Every entry must differ and nothing else may.
CFG_DIFF_ALLOWED = (
    "scene.robot_a.spawn.usd_path", "scene.robot_b.spawn.usd_path",
    "scene.object.spawn.activate_contact_sensors",
    f"scene.{CUBE_SENSOR}",
    "rewards.lifting_object.func", "rewards.lifting_object.params.clearance",
    "rewards.lifting_object.params.tilt_max_deg",
    "terminations.success.func", "terminations.success.params.tilt_max_deg",
    "terminations.success.params.release_force_n", "terminations.success.params.sensor_name",
) + tuple(f"curriculum.{n}" for n in DIAG_TERMS)
SMOKE_CLAUSES = ("A1_terms_in_managers", "A2_cfg_diff_declared_only", "A3_hand_colliders",
                 "A4_sensor_26_filters", "A5_step_cache_keys", "A6_free_cube_released",
                 "A7_hand_contact_sensed", "A8_success_chain_fires", "B1_logged_tags",
                 "B2_training_ran", "B3_no_failure_markers", "B4_config_check")

# Probe stage 8, "seat" (added on review 2026-10-02): the success chain with the
# real sensor. After a fresh reset each env's cube is teleported onto a deck,
# resting (lowest corner on the deck top, zero velocity), zero actions:
#   flat      identity: must seat, stay released, fire `success` on the 12th
#             step with `placed` paying;
#   rolled90  90 deg about x, flat on another face: seated and released but
#             tilt 90 deg under TILT_NOTE -- the control, must never fire (it is
#             tied to the body-z definition: under face tilt it would fire);
#   tilt10    10 deg about x on its lowest edge: falls flat within a few steps,
#             so it is followed for consistency only, never gated (a "must not
#             fire" on it would hinge on how fast it settles).
# Each env is followed until its first reset: success resets the env on the
# step it fires (that step counts); any earlier reset ends the env's evidence.
SMOKE_SEAT_STEPS = 16              # >= SEAT_HOLD_STEPS + 1; a flat cube fires on step 12
SEAT_GROUPS = ("flat", "flat", "flat", "flat", "flat", "rolled90", "rolled90", "tilt10")
SEAT_ROLL_DEG = {"flat": 0.0, "rolled90": 90.0, "tilt10": 10.0}
#: Per-step, per-env record keys (taken after each env.step).
SEAT_RECORD_KEYS = ("success", "reset", "hold", "seated", "released", "placed_now",
                    "tilt_deg", "placed_reward")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
FAIL_MARKERS = re.compile(r"train\.sh: FAILED|Traceback \(most recent call last\)"
                          r"|(?i:physx[^\n]*overflow|overflow[^\n]*(?:buffer|capacity))")
_TAG_LINE = re.compile(r"((?:Curriculum|Episode_Reward|Episode_Termination)/[A-Za-z0-9_]+):\s+\S+")


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def console_tags(log_text: str) -> set:
    """Tags printed in rsl-rl's per-iteration console tables."""
    return set(_TAG_LINE.findall(strip_ansi(log_text)))


def logged_iterations(log_text: str) -> set:
    return {int(i) for i in re.findall(r"Learning iteration (\d+)/", strip_ansi(log_text))}


def cfg_diff_check(diff_keys: Sequence[str]) -> dict:
    """Stand3 vs Stand4 `to_dict()` keys that differ -> undeclared and missing changes."""
    def covers(entry, k):
        return k == entry or k.startswith(entry + ".")
    unexpected = sorted(k for k in diff_keys if not any(covers(e, k) for e in CFG_DIFF_ALLOWED))
    missing = [e for e in CFG_DIFF_ALLOWED if not any(covers(e, k) for k in diff_keys)]
    return {"ok": not unexpected and not missing, "unexpected": unexpected, "missing": missing}


def seat_group(i: int) -> str:
    """Seat-stage group of env `i` (SEAT_GROUPS repeats every 8 envs)."""
    return SEAT_GROUPS[i % len(SEAT_GROUPS)]


def seat_side(i: int) -> float:
    """+1 (deck_pos) for even env ids, -1 (deck_neg) for odd ones."""
    return 1.0 if i % 2 == 0 else -1.0


def roll_x_wxyz(deg: float) -> tuple:
    """(w, x, y, z) of a rotation by `deg` about the x axis. Written to the
    simulator only through `quat_order.native_quat`."""
    h = math.radians(deg) / 2.0
    return (math.cos(h), math.sin(h), 0.0, 0.0)


def seat_pose_local(i: int) -> tuple:
    """((x, y, z), (w, x, y, z)): env `i`'s seat-stage cube pose, env-local, on its
    deck's centre line with the lowest corner on the deck top (z = DECK_TOP +
    the rotated half-height, so flat and rolled90 sit at DECK_SEATED_Z)."""
    q = roll_x_wxyz(SEAT_ROLL_DEG[seat_group(i)])
    hz = float(cube_half_extents(*(torch.tensor([v], dtype=torch.float64) for v in q))[2])
    return (seat_side(i) * sm.DECK_CENTER, 0.0, sm.DECK_TOP + hz), q


def seat_stage_eval(groups: Sequence[str], records: Sequence[dict],
                    hold_steps: int = sm.SEAT_HOLD_STEPS,
                    tilt_max_deg: float = SEAT_TILT_MAX_DEG) -> dict:
    """Clause A8 from the seat stage's records (pure; the probe and the host
    verdict both call it).

    `records[k]` holds, for step k + 1 and every env, what the probe read AFTER
    that env.step: the termination's `success`, `reset_buf`, the cache's hold /
    seated / released / placed_now / tilt_deg, and the `placed` reward of that
    step. An env counts until its first reset (inclusive). PASS iff the stage
    ran >= hold_steps + 1 steps, >= 1 flat env fired, every firing is
    well-formed (seated, released, placed_now, hold >= hold_steps, placed > 0),
    no followed step is inconsistent (success == hold >= hold_steps; placed > 0
    == success; placed_now == seated & released & tilt <= tilt_max_deg; success
    resets its env on the same step; the hold counter is `_held`'s: previous + 1
    when placed_now, else 0), no rolled90 env fired, and >= 1 followed
    rolled90 step was seated and released with tilt > tilt_max_deg (blocked by
    the tilt clause alone).
    """
    n = len(groups)
    shapes_ok = all(len(r.get(k, ())) == n for r in records for k in SEAT_RECORD_KEYS)
    alive = [True] * n
    prev = [None] * n                 # the hold counter one step earlier (unknown at step 1)
    fire_step = [None] * n
    events, inconsistent, control_blocked = [], [], 0
    if shapes_ok:
        for k, r in enumerate(records, start=1):
            for i in range(n):
                if not alive[i]:
                    continue
                succ, hold = bool(r["success"][i]), int(r["hold"][i])
                seated, rel, now = bool(r["seated"][i]), bool(r["released"][i]), bool(r["placed_now"][i])
                tilt, rew = float(r["tilt_deg"][i]), float(r["placed_reward"][i])
                reset = bool(r["reset"][i])
                want_now = seated and rel and tilt <= tilt_max_deg
                counter_ok = (hold == 0) if not now else (
                    hold >= 1 if prev[i] is None else hold == prev[i] + 1)
                prev[i] = hold
                if (succ != (hold >= hold_steps) or (rew > 0.0) != succ or now != want_now
                        or (succ and not reset) or not counter_ok):
                    inconsistent.append({"env": i, "step": k, "success": succ, "hold": hold,
                                         "placed_reward": rew, "placed_now": now,
                                         "want_now": want_now})
                if succ:
                    fire_step[i] = k
                    events.append({"env": i, "group": groups[i], "step": k, "seated": seated,
                                   "released": rel, "placed_now": now, "hold": hold,
                                   "placed_reward": rew})
                if groups[i] == "rolled90" and seated and rel and tilt > tilt_max_deg and not now:
                    control_blocked += 1
                if reset:
                    alive[i] = False
    bad = [e for e in events if not (e["seated"] and e["released"] and e["placed_now"]
                                     and e["hold"] >= hold_steps and e["placed_reward"] > 0.0)]
    flat = [i for i in range(n) if groups[i] == "flat"]
    flat_fired = [i for i in flat if fire_step[i] is not None]
    control_fired = [e for e in events if e["group"] == "rolled90"]
    ok = bool(shapes_ok and len(records) >= hold_steps + 1 and flat_fired and not bad
              and not inconsistent and not control_fired and control_blocked >= 1)
    return {"ok": ok, "shapes_ok": shapes_ok, "n_steps": len(records), "n_flat": len(flat),
            "n_flat_fired": len(flat_fired), "fire_step": fire_step,
            "n_fire_events": len(events), "n_fire_bad": len(bad),
            "n_inconsistent": len(inconsistent), "inconsistent_head": inconsistent[:10],
            "n_control_fired": len(control_fired),
            "control_blocked_by_tilt_steps": control_blocked, "fire_events": events}


def smoke_verdict(probe: dict | None, tags: set, train_log: str | None,
                  cfg_check_res: dict | None) -> dict:
    """The smoke rule (gpu_v2_stand4_smoke.sbatch header). PASS iff every clause
    holds; INCOMPLETE if an input is missing; else FAIL."""
    C, missing = {}, []
    if probe is None:
        missing.append("probe JSON missing or unreadable")
    elif probe.get("status") != "ok":
        missing.append(f"probe status {probe.get('status')!r}")
    else:
        m = probe.get("managers", {})
        lo, pl, su = (m.get(k, {}) for k in ("lifting_object", "placed", "success"))
        C["A1_terms_in_managers"] = bool(
            lo.get("func") == FUNC_LIFT4 and lo.get("weight") == 15.0
            and pl.get("func") == FUNC_PLACED and pl.get("weight") == sm.STAND3_PLACED_WEIGHT
            and su.get("func") == FUNC_SUCCESS4
            and {"lifting_object", "placed"} <= set(m.get("reward_terms", []))
            and "success" in m.get("termination_terms", [])
            and set(CURRICULUM4_TERMS) <= set(m.get("curriculum_terms", [])))
        C["A2_cfg_diff_declared_only"] = cfg_diff_check(probe.get("cfg_diff_keys", ["?"]))["ok"]
        hc = probe.get("hand_colliders", {})
        up = probe.get("usd_paths", {})
        C["A3_hand_colliders"] = bool(
            len(hc) == 4 and all(v.get("valid") and v.get("collision_api") for v in hc.values())
            and all(str(up.get("stand4", {}).get(r, "")).endswith("/" + OVERLAY_NAME)
                    for r in RELEASE_ROBOTS)
            and all(str(up.get("stand3", {}).get(r, "")).endswith("/" + BASE_USD_NAME)
                    for r in RELEASE_ROBOTS))
        s = probe.get("sensor", {})
        n = probe.get("num_envs")
        C["A4_sensor_26_filters"] = bool(
            s.get("force_matrix_shape") == [n, 1, N_RELEASE_BODIES, 3]
            and s.get("filter_count") == N_RELEASE_BODIES)
        ch = probe.get("cache", {})
        C["A5_step_cache_keys"] = bool(ch.get("n_steps", 0) > 0 and ch.get("keys_missing") == []
                                       and ch.get("in_extras") and ch.get("shapes_ok"))
        fs = probe.get("free_space", {})
        C["A6_free_cube_released"] = bool(fs.get("n_valid", 0) >= 1
                                          and fs.get("max_force_valid", math.inf) < RELEASE_FORCE_N
                                          and fs.get("all_released_valid"))
        ov = probe.get("overlap", {})
        C["A7_hand_contact_sensed"] = bool(ov.get("n_hand_contact", 0) >= 1
                                           and ov.get("inconsistent", 1) == 0
                                           and ov.get("cache_mismatch", 1) == 0)
        st = probe.get("seat") or {}
        try:            # recomputed here from the raw records, not read from the probe
            C["A8_success_chain_fires"] = bool(
                st.get("steps") == SMOKE_SEAT_STEPS
                and seat_stage_eval(st.get("groups", []), st.get("records", []))["ok"])
        except (KeyError, TypeError, ValueError, AttributeError):
            C["A8_success_chain_fires"] = False
    C["B1_logged_tags"] = set(SMOKE_TAGS) <= set(tags)
    if train_log is None:
        missing.append("training log missing")
    else:
        C["B2_training_ran"] = len(logged_iterations(train_log)) >= SMOKE_ITERS
        C["B3_no_failure_markers"] = FAIL_MARKERS.search(strip_ansi(train_log)) is None
    if cfg_check_res is None:
        missing.append("run params (agent.yaml / env.yaml) missing")
    else:
        C["B4_config_check"] = bool(cfg_check_res.get("ok"))
    if missing or any(k not in C for k in SMOKE_CLAUSES):
        verdict = "INCOMPLETE (no verdict: " + "; ".join(missing or ["clause not evaluated"]) + ")"
    elif all(C[k] for k in SMOKE_CLAUSES):
        verdict = "PASS"
    else:
        verdict = "FAIL (" + ", ".join(k for k in SMOKE_CLAUSES if not C[k]) + ")"
    return {"verdict": verdict, "clauses": C, "missing": missing,
            "tags_missing": sorted(set(SMOKE_TAGS) - set(tags))}
