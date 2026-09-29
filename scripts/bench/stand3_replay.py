"""CubeToShelfStand3 replay diagnostic: where does the cube go, and why is it never seated?

REPORT-ONLY. No gate; changes no recorded verdict. Stand3 (job 21443327) is
NEGATIVE by its predeclared training rule (last-200 success 0.0020 / 0.0012,
over_deck 0.095 / 0.043; stands 2/2, shifts 0/2, stable 2/2). This replays the
two FINAL checkpoints and classifies what the cube does whenever it is over a
deck, so that a Stand4 is funded (or not) on a measured failure mode.

Predeclared 2026-09-28, before any replay ran (thresholds, classes and the
funding reading below are fixed here; slurm/repo20260923/gpu_stand3_replay.sbatch
carries the same text).

WHAT IS REPLAYED
  task_v2/2026-09-27_17-56-20_v2-cubetoshelfstand3-blind-s0/model_7999.pt
  task_v2/2026-09-27_20-03-28_v2-cubetoshelfstand3-blind-s1/model_7999.pt
  under their OWN training config, TaskV2-BHL-CubeToShelfStand3-Blind-v0, built
  from the source each run recorded in <run>/git/bhl-robustness-ladder.diff
  (commit 493c123 / 9a89001 -- identical under src/ -- plus the uncommitted src/
  diff captured at training time). That is the config BEFORE the 2026-09-28
  quaternion fix: the cube spawns with the raw literal rot=(1, 0, 0, 0), which
  Isaac Lab 3.0 reads as (x, y, z, w) = 180 deg about x, and object_tilt_l2
  indexes the quaternion as wxyz. The policy observation carries no cube
  orientation (object_pos_in_root uses the ROBOT quaternion; the critic's
  object_ang_vel is the only cube-rotation term), so the fix would not change
  the policy's inputs -- the pin is kept because this is a diagnostic of the
  policy as trained. Checked in the job: SOURCE-CHECK (snapshot files
  byte-identical to `git show <commit>:`, raw literal present) and CONFIG-CHECK
  (the live env cfg dumped after construction against the run's
  params/env.yaml; allowlisted: seed, scene.num_envs, sim.device, sim.log_dir;
  rewards / curriculum / UI / recording fields are reported, not gated).

PROTOCOL
  v60 (Isaac Sim 6.0 / Lab 3.0.0b2), --enable_cameras for stack parity with
  training (the blind task has no camera). 32 envs, reset seed 1000 (never used:
  training seeded 0 / 1), 3 episodes per env = 96 episodes per checkpoint
  (<= 3 x 500 + 50 steps). Deterministic actions: rsl-rl 5 MLPModel.forward with
  stochastic_output=False (the distribution mean), checked by evaluating the
  first observation twice. Observation noise ON and reset randomisation as
  trained. State is logged after physics and before the auto-reset (a hook on
  the termination manager's compute), so every episode's terminal step is in
  the trace.
  Label: learned (PPO checkpoint, deterministic mean action); cube and robot
  state read from the simulator (oracle) only to classify; nothing scripted.

THRESHOLDS (literals below; equal to src/bhl_robust/tasks/stand_mdp.py's,
asserted by tests/test_stand3_replay.py and again in the job)
  over_deck    |x| > DECK_EDGE 0.17 and z > OFF_FLOOR_Z 0.34 (stand_mdp.over_deck_mask)
  seated       0.19 < |x| < 0.45, |y| < 0.10, |z - DECK_SEATED_Z 0.57| < SEAT_Z_TOL 0.02,
               speed < SEAT_SPEED 0.05 m/s (stand_mdp.seated_mask); success = seated for
               SEAT_HOLD_STEPS 12 consecutive steps
  reach        REQUIRED_SHIFT 0.19 (= SEAT_X_LO), MEASURED_SHIFT_ARMS_TWIST 0.119 (feet planted,
               arms + <= 0.5 rad pelvis twist, MuJoCo FK), MEASURED_SHIFT_ARMS 0.044
  contact      PROXY, not a filtered hand-cube sensor: a hand touches the cube when its hand-link
               origin is within HAND_NEAR_M 0.05 m of the cube surface (box distance in the cube
               frame) AND the net contact force on that hand link is >= HAND_FORCE_N 1.0 N
  All comparisons in float32 with float32 thresholds, as the env's torch terms do.

CLASSES -- every over_deck step gets exactly one, tested in this order:
  MOVING              speed >= 0.05
  HELD_ABOVE          still and z - 0.57 >= 0.02 (at or above the top of the seat window:
                      held up -- the contact proxy is reported alongside)
  SEATED              stand_mdp.seated_mask
  RESTING_NOT_SEATED  still, not above, not seated; reason bits LOW (0.57 - z >= 0.02),
                      X_BAND (|x| outside (0.19, 0.45)), Y_BAND (|y| >= 0.10)
  Per episode: max |cube x| while z > 0.34 ("carried", primary) and over all steps,
  against 0.19 and 0.119; how it ended; longest SEATED run.

FUNDING READING (report-only, per checkpoint; not a gate)
  S = share of the 96 episodes that end in the success termination
  R = share whose carried max |x| >= 0.19;  M = median carried max |x|
  D = among the R episodes, share with any SEATED or RESTING_NOT_SEATED step (set down)
  R1 PLACES     S >= 0.10: the deterministic policy places; no Stand4 on a
                placement-failure argument (Stand3's recorded NEGATIVE, predeclared on
                training metrics, stands).
  R2 REACH-SHORT   S < 0.10, R < 0.10, M >= 0.044 (MEASURED_SHIFT_ARMS): the cube is
                shifted but stops short of the 0.19 m seat band -- a reach problem -> a
                Stand4 that closes the reach gap (seat band within the demonstrated shift,
                or a lean / shuffle incentive). Reach-limit sub-reading (reported, not a
                separate code): M < 0.119 = the shift stays inside the feet-planted
                arm+twist envelope (consistent with the reach-limit hypothesis as
                stand_mdp states it); M >= 0.119 = partly beyond it (leans or shuffles,
                not far enough).
  R3 NO-SHIFT      S < 0.10, R < 0.10, M < 0.044: the shift itself is not learned (the
                cube stays within the arms-only envelope of the plinth centre); deck
                distance is not shown to be the binding constraint -> no Stand4 on this
                evidence.
  R4 LOWERING / HOLD   S < 0.10, R >= 0.10, D < 0.50: carried over the seat band but not
                set down -> a Stand4 aimed at lowering / release.
  R5 SETTLE        S < 0.10, R >= 0.10, D >= 0.50: set down but never seated still for 12
                steps -> a Stand4 aimed at the settle (sub-report: SEATED run lengths and
                the LOW / X_BAND / Y_BAND reasons).
  (R < 0.10 forces M < 0.19, so R1-R5 are exhaustive for a complete replay.)
  Also reported, not in any reading: the share of MOVING over-deck steps at or
  above the top of the seat window (a swaying held cube lands in MOVING).
  A Stand4 is justified by this diagnostic only if BOTH checkpoints give the same
  R2, R4 or R5 reading; R1, R3, a disagreement or an INCOMPLETE replay -> no Stand4
  on this evidence.

Subcommands
  isaac    the replay (inside the v60 container; --args-file written by the launcher)
  dry-run  every check possible without Isaac (run dir, checkpoint, params, pinned source)
  synth    a synthetic trace + meta (plumbing test; labelled SYNTHETIC everywhere)
  classify trace + meta -> replay JSON (numpy only; runs on the login node)
  report   two replay JSONs -> per-checkpoint and pooled funding reading
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------ predeclared literals
# Geometry and seat test: stand_mdp.py's values, written out so the rule cannot
# move with that file (tests/test_stand3_replay.py asserts equality).
CUBE_HALF = 0.14
DECK_EDGE = 0.17
OFF_FLOOR_Z = 0.34
DECK_SEATED_Z = 0.57
SEAT_Z_TOL = 0.02
SEAT_SPEED = 0.05
SEAT_HOLD_STEPS = 12
SEAT_X_LO = 0.19
SEAT_X_HI = 0.45
SEAT_Y_HALF = 0.10
REQUIRED_SHIFT = 0.19
MEASURED_SHIFT_ARMS = 0.044
MEASURED_SHIFT_ARMS_TWIST = 0.119
STEP_DT = 0.04
EPISODE_STEPS = 500
#: stand_mdp name -> literal here, for the equality checks.
STAND_MDP_CONSTANTS = {
    "CUBE_HALF": CUBE_HALF, "DECK_EDGE": DECK_EDGE, "OFF_FLOOR_Z": OFF_FLOOR_Z,
    "DECK_SEATED_Z": DECK_SEATED_Z, "SEAT_Z_TOL": SEAT_Z_TOL, "SEAT_SPEED": SEAT_SPEED,
    "SEAT_HOLD_STEPS": SEAT_HOLD_STEPS, "SEAT_X_LO": SEAT_X_LO, "SEAT_X_HI": SEAT_X_HI,
    "SEAT_Y_HALF": SEAT_Y_HALF, "REQUIRED_SHIFT": REQUIRED_SHIFT,
    "MEASURED_SHIFT_ARMS": MEASURED_SHIFT_ARMS,
    "MEASURED_SHIFT_ARMS_TWIST": MEASURED_SHIFT_ARMS_TWIST,
    "STEP_DT": STEP_DT, "EPISODE_STEPS": EPISODE_STEPS,
}
#: Contact proxy (declared here; not stand_mdp quantities).
HAND_NEAR_M = 0.05
HAND_FORCE_N = 1.0
HANDS = ("arm_left_hand_link", "arm_right_hand_link")
HAND_COLUMNS = ("a_left", "a_right", "b_left", "b_right")

# Protocol.
TASK_ID = "TaskV2-BHL-CubeToShelfStand3-Blind-v0"
TRAIN_JOB = "21443327"
CHECKPOINT = "model_7999.pt"
NUM_ENVS = 32
EPISODES_PER_ENV = 3
REPLAY_SEED = 1000
TRAIN_SEEDS = (0, 1)
RUNS = {
    "s0": "2026-09-27_17-56-20_v2-cubetoshelfstand3-blind-s0",
    "s1": "2026-09-27_20-03-28_v2-cubetoshelfstand3-blind-s1",
}
#: Stand3's task files; the launcher requires them byte-identical to the run's commit.
PINNED_FILES = ("bhl_robust/tasks/coop_lift_env_cfg.py", "bhl_robust/tasks/coop_lift_mdp.py",
                "bhl_robust/tasks/stand_mdp.py", "bhl_robust/tasks/task_v2_env_cfg.py",
                "bhl_robust/tasks/task_v2_mdp.py", "bhl_robust/quat_order.py")
#: The pre-fix spawn literal, exactly as committed before 2026-09-28.
RAW_OBJECT_LITERAL = "init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z), rot=(1.0, 0.0, 0.0, 0.0))"

# Funding reading (report-only).
READ_MIN_SUCCESS = 0.10      # S: the training rule's RESULT_MIN_SUCCESS, reused as a reading cut
READ_MIN_REACH = 0.10        # R
READ_MIN_SETDOWN = 0.50      # D
STAND4_READINGS = ("R2", "R4", "R5")

LABEL = ("learned (PPO checkpoint model_7999.pt, deterministic mean action, observation noise as "
         "trained); cube and robot state read from the simulator (oracle) only to classify; nothing scripted")
SYNTH_LABEL = "SYNTHETIC (no Isaac step ran; plumbing test only -- not a result)"
AS_TRAINED_NOTE = ("config as trained, BEFORE the 2026-09-28 quaternion fix: cube spawn literal "
                   "rot=(1,0,0,0) read as xyzw = 180 deg about x; object_tilt_l2 indexes wxyz. "
                   "The policy observation has no cube orientation, so the fix would not change its inputs.")
REPORT_ONLY_NOTE = "REPORT-ONLY diagnostic: no gate; changes no recorded verdict (Stand3 NEGATIVE stands)"

# Classes.
NOT_OVER, MOVING, HELD_ABOVE, SEATED, RESTING = 0, 1, 2, 3, 4
CLASS_NAMES = {NOT_OVER: "NOT_OVER_DECK", MOVING: "MOVING", HELD_ABOVE: "HELD_ABOVE",
               SEATED: "SEATED", RESTING: "RESTING_NOT_SEATED"}
REASON_LOW, REASON_X_BAND, REASON_Y_BAND = 1, 2, 4
REASON_NAMES = {REASON_LOW: "LOW", REASON_X_BAND: "X_BAND", REASON_Y_BAND: "Y_BAND"}

# Config check.
CONFIG_ALLOWLIST = ("seed", "scene.num_envs", "sim.device", "sim.log_dir")
CONFIG_REPORT_ONLY_TOP = ("rewards", "curriculum", "viewer", "video_recorder", "recorders",
                          "ui_window_class_type", "teleop_devices", "isaac_teleop", "xr",
                          "export_io_descriptors", "log_dir", "wait_for_textures",
                          "rerender_on_reset", "num_rerenders_on_reset")
AGENT_ALLOWLIST = ("seed", "run_name", "device", "max_iterations", "resume", "load_run",
                   "load_checkpoint")

#: Trace arrays: name -> (trailing shape, dtype). Dense (T, N, ...) per checkpoint.
TRACE_SPEC = {
    "valid": ((), np.bool_), "ep_idx": ((), np.int8), "done": ((), np.bool_),
    "ep_step": ((), np.int16),
    "cube_pos": ((3,), np.float32), "cube_vel": ((3,), np.float32), "cube_speed": ((), np.float32),
    "cube_quat_raw": ((4,), np.float32), "cube_face_tilt_deg": ((), np.float32),
    "cube_up_axis": ((), np.int8),
    "over_deck_env": ((), np.bool_), "seated_env": ((), np.bool_), "hold_counter": ((), np.int16),
    "term_success": ((), np.bool_), "term_fallen": ((), np.bool_), "term_time_out": ((), np.bool_),
    "base_pos_a": ((3,), np.float32), "base_pos_b": ((3,), np.float32),
    "tilt_a": ((), np.float32), "tilt_b": ((), np.float32),
    "up_a": ((3,), np.float32), "up_b": ((3,), np.float32),
    "fwd_a": ((3,), np.float32), "fwd_b": ((3,), np.float32),
    "hand_sdf": ((4,), np.float32), "hand_force": ((4,), np.float32),
    "action_absmax": ((), np.float32),
}


# ======================================================================= pure numpy
def f32(v) -> np.float32:
    return np.float32(v)


def over_deck_np(p: np.ndarray) -> np.ndarray:
    """stand_mdp.over_deck_mask in float32: |x| > DECK_EDGE and z > OFF_FLOOR_Z."""
    p = np.asarray(p, dtype=np.float32)
    return (np.abs(p[..., 0]) > f32(DECK_EDGE)) & (p[..., 2] > f32(OFF_FLOOR_Z))


def seated_np(p: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """stand_mdp.seated_mask in float32 (same operations, same order)."""
    p = np.asarray(p, dtype=np.float32)
    speed = np.asarray(speed, dtype=np.float32)
    ax = np.abs(p[..., 0])
    return ((ax > f32(SEAT_X_LO)) & (ax < f32(SEAT_X_HI)) & (np.abs(p[..., 1]) < f32(SEAT_Y_HALF))
            & (np.abs(p[..., 2] - f32(DECK_SEATED_Z)) < f32(SEAT_Z_TOL)) & (speed < f32(SEAT_SPEED)))


def classify_steps(p: np.ndarray, speed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(class code, reason bits) per step; see the module docstring.

    One float32 difference d = z - DECK_SEATED_Z decides the height tests:
    HELD_ABOVE d >= tol, seat window |d| < tol, LOW d <= -tol. They partition
    every finite d exactly, so the classes are exhaustive and exclusive.
    """
    p = np.asarray(p, dtype=np.float32)
    speed = np.asarray(speed, dtype=np.float32)
    over = over_deck_np(p)
    moving = speed >= f32(SEAT_SPEED)
    d = p[..., 2] - f32(DECK_SEATED_Z)
    above = d >= f32(SEAT_Z_TOL)
    seated = seated_np(p, speed)
    cls = np.full(over.shape, NOT_OVER, dtype=np.int8)
    cls[over & moving] = MOVING
    cls[over & ~moving & above] = HELD_ABOVE
    cls[over & ~moving & ~above & seated] = SEATED
    rest = over & ~moving & ~above & ~seated
    cls[rest] = RESTING
    ax = np.abs(p[..., 0])
    reason = np.zeros(over.shape, dtype=np.int8)
    reason |= np.where(rest & (d <= -f32(SEAT_Z_TOL)), REASON_LOW, 0).astype(np.int8)
    reason |= np.where(rest & ~((ax > f32(SEAT_X_LO)) & (ax < f32(SEAT_X_HI))), REASON_X_BAND, 0).astype(np.int8)
    reason |= np.where(rest & ~(np.abs(p[..., 1]) < f32(SEAT_Y_HALF)), REASON_Y_BAND, 0).astype(np.int8)
    return cls, reason


def run_lengths(mask: np.ndarray) -> list[int]:
    """Lengths of the runs of True in a 1-D bool array."""
    out, n = [], 0
    for v in np.asarray(mask, dtype=bool):
        if v:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def consecutive_count(mask: np.ndarray) -> np.ndarray:
    """The env's hold counter recomputed: consecutive True steps ending at each step."""
    out = np.zeros(len(mask), dtype=np.int64)
    n = 0
    for i, v in enumerate(np.asarray(mask, dtype=bool)):
        n = n + 1 if v else 0
        out[i] = n
    return out


def episode_segments(valid: np.ndarray, ep_idx: np.ndarray, episodes: int) -> list[tuple[int, int, np.ndarray]]:
    """[(env, episode, step indices)] from the dense (T, N) masks; indices are contiguous."""
    segs = []
    T, N = valid.shape
    for n in range(N):
        for e in range(episodes):
            idx = np.nonzero(valid[:, n] & (ep_idx[:, n] == e))[0]
            if idx.size == 0:
                continue
            if idx[-1] - idx[0] + 1 != idx.size:
                raise ValueError(f"env {n} episode {e}: steps are not contiguous")
            segs.append((n, e, idx))
    return segs


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _fmax(a) -> float | None:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.max()) if a.size else None


def _robot_mechanics(tr: dict, r: str, idx: np.ndarray, n: int, k: int, side: float) -> dict:
    """Base shuffle, lean and pelvis twist of robot r over one episode.

    lean_x: signed angle of the body up axis toward world +x (the shoulder line,
    toward the +x deck), from the stack's own rotation matrix -- i.e. the
    pelvis roll for a robot facing +/-y. `*_toward_deck` is signed by the side
    the cube is on at its carried-max step k (positive = toward that deck).
    """
    base = np.asarray(tr[f"base_pos_{r}"][idx, n], dtype=np.float64)
    disp = base[:, :2] - base[0, :2]
    up = np.asarray(tr[f"up_{r}"][idx, n], dtype=np.float64)
    fwd = np.asarray(tr[f"fwd_{r}"][idx, n], dtype=np.float64)
    lean_x = np.degrees(np.arctan2(up[:, 0], up[:, 2]))
    lean_y = np.degrees(np.arctan2(up[:, 1], up[:, 2]))
    yaw = np.arctan2(fwd[:, 1], fwd[:, 0])
    yaw_rel = np.degrees(_wrap(yaw - yaw[0]))
    tilt = np.asarray(tr[f"tilt_{r}"][idx, n], dtype=np.float64)
    return {
        "max_base_disp_m": _fmax(np.linalg.norm(disp, axis=1)),
        "max_abs_base_dx_m": _fmax(np.abs(disp[:, 0])),
        "max_abs_lean_x_deg": _fmax(np.abs(lean_x)),
        "max_abs_lean_y_deg": _fmax(np.abs(lean_y)),
        "max_abs_yaw_rel_deg": _fmax(np.abs(yaw_rel)),
        "max_tilt_rad": _fmax(tilt),
        "at_carried_max": {
            "base_dx_toward_deck_m": float(disp[k, 0] * side),
            "lean_x_toward_deck_deg": float(lean_x[k] * side),
            "yaw_rel_deg": float(yaw_rel[k]),
        },
    }


def episode_summary(tr: dict, n: int, e: int, idx: np.ndarray) -> dict:
    """Everything the reading needs about one (env, episode)."""
    p = np.asarray(tr["cube_pos"][idx, n], dtype=np.float32)
    sp = np.asarray(tr["cube_speed"][idx, n], dtype=np.float32)
    vz = np.asarray(tr["cube_vel"][idx, n, 2], dtype=np.float32)
    cls, reason = classify_steps(p, sp)
    over = cls != NOT_OVER
    seated = cls == SEATED
    runs = run_lengths(seated)
    last = int(idx[-1])
    done = bool(tr["done"][last, n])
    succ = bool(tr["term_success"][last, n])
    fell = bool(tr["term_fallen"][last, n])
    tout = bool(tr["term_time_out"][last, n])
    ended_by = ("success" if succ else "fallen" if fell else "time_out" if tout else
                "done_other" if done else "unfinished")
    ax = np.abs(p[:, 0])
    carried = p[:, 2] > f32(OFF_FLOOR_Z)
    if carried.any():
        k = int(np.argmax(np.where(carried, ax, np.float32(-1.0))))
        mx_c = float(ax[k])
    else:                                   # cube off the plinth and on the floor from step 1
        k, mx_c = 0, 0.0
    side = 1.0 if p[k, 0] >= 0 else -1.0

    # contact proxy
    sdf = np.asarray(tr["hand_sdf"][idx, n], dtype=np.float32)
    force = np.asarray(tr["hand_force"][idx, n], dtype=np.float32)
    near = sdf <= f32(HAND_NEAR_M)
    with np.errstate(invalid="ignore"):
        touch = near & (force >= f32(HAND_FORCE_N))          # NaN force -> False
    pair_touch = touch[:, :2].any(axis=1) & touch[:, 2:].any(axis=1)
    any_near = near.any(axis=1)

    counts = {CLASS_NAMES[c]: int((cls == c).sum()) for c in (MOVING, HELD_ABOVE, SEATED, RESTING)}
    rest = cls == RESTING
    reasons = {REASON_NAMES[b]: int(((reason & b) != 0).sum()) for b in (REASON_LOW, REASON_X_BAND, REASON_Y_BAND)}
    held = cls == HELD_ABOVE
    moving = cls == MOVING

    # consistency with the env's own terms (same float32 predicates -> expect 0)
    hold_env = np.asarray(tr["hold_counter"][idx, n], dtype=np.int64)
    hold_np = consecutive_count(seated)
    face_axis = np.asarray(tr["cube_up_axis"][idx, n])
    out = {
        "env": int(n), "episode": int(e), "steps": int(idx.size), "finished": done,
        "ended_by": ended_by, "success_env": succ, "fallen": fell, "time_out": tout,
        "success_recount": bool(hold_np[-1] >= SEAT_HOLD_STEPS),
        "max_abs_x_carried": mx_c,
        "max_abs_x_all": float(ax.max()),
        "carried_max_step": int(k), "carried_max_z": float(p[k, 2]),
        "carried_max_class": CLASS_NAMES[int(cls[k])],
        "reached_required": bool(f32(mx_c) >= f32(REQUIRED_SHIFT)),
        "beyond_twist_envelope": bool(f32(mx_c) >= f32(MEASURED_SHIFT_ARMS_TWIST)),
        "beyond_arms_envelope": bool(f32(mx_c) >= f32(MEASURED_SHIFT_ARMS)),
        "dropped_to_floor": bool((~carried).any()),
        "over_deck_steps": int(over.sum()),
        "class_steps": counts,
        "resting_reason_steps": reasons,
        "set_down": bool((seated | rest).any()),
        "seated_runs": runs,
        "longest_seated_run": int(max(runs, default=0)),
        "held_above_max_z": _fmax(p[held, 2]) if held.any() else None,
        "held_above_pair_touch_steps": int((held & pair_touch).sum()),
        "held_above_any_hand_near_steps": int((held & any_near).sum()),
        "over_deck_lowering_steps": int((moving & (vz <= -f32(SEAT_SPEED))).sum()),
        "moving_above_seat_steps": int((moving & ((p[:, 2] - f32(DECK_SEATED_Z)) >= f32(SEAT_Z_TOL))).sum()),
        "pair_touch_steps": int(pair_touch.sum()),
        "force_available": bool(np.isfinite(force).all()),
        "cube_face_changed_steps": int((face_axis != face_axis[0]).sum()),
        "robot_a": _robot_mechanics(tr, "a", idx, n, k, side),
        "robot_b": _robot_mechanics(tr, "b", idx, n, k, side),
        "mismatch": {
            "over_deck": int((np.asarray(tr["over_deck_env"][idx, n], dtype=bool) != over).sum()),
            "seated": int((np.asarray(tr["seated_env"][idx, n], dtype=bool) != seated).sum()),
            "hold_counter": int((hold_env != hold_np).sum()),
            "success": int(succ != bool(hold_np[-1] >= SEAT_HOLD_STEPS)),
        },
    }
    return out


def _median(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def _share(k: int, n: int) -> float | None:
    return (k / n) if n else None


def summarize(tr: dict, num_envs: int, episodes: int) -> dict:
    """Per-episode summaries and the per-checkpoint aggregate."""
    segs = episode_segments(np.asarray(tr["valid"], dtype=bool), np.asarray(tr["ep_idx"]), episodes)
    eps = [episode_summary(tr, n, e, idx) for n, e, idx in segs]
    fin = [x for x in eps if x["finished"]]
    n_exp = num_envs * episodes
    nf = len(fin)
    reach = [x for x in fin if x["reached_required"]]
    mx = [x["max_abs_x_carried"] for x in fin]
    edges = (0.0, MEASURED_SHIFT_ARMS, MEASURED_SHIFT_ARMS_TWIST, REQUIRED_SHIFT)
    bins = {"[0, 0.044)": 0, "[0.044, 0.119)": 0, "[0.119, 0.19)": 0, "[0.19, inf)": 0}
    for v in mx:
        v32 = f32(v)
        i = int(sum(v32 >= f32(b) for b in edges[1:]))
        bins[list(bins)[i]] += 1
    tot = {c: sum(x["class_steps"][c] for x in fin) for c in ("MOVING", "HELD_ABOVE", "SEATED", "RESTING_NOT_SEATED")}
    over_steps = sum(x["over_deck_steps"] for x in fin)
    all_steps = sum(x["steps"] for x in fin)
    held_steps = tot["HELD_ABOVE"]
    mism = {k: sum(x["mismatch"][k] for x in eps) for k in ("over_deck", "seated", "hold_counter", "success")}

    def mech(group, r, key):
        return _median([x[f"robot_{r}"][key] for x in group])

    def mech_at(group, r, key):
        return _median([x[f"robot_{r}"]["at_carried_max"][key] for x in group])

    not_reach = [x for x in fin if not x["reached_required"]]
    summary = {
        "episodes_expected": n_exp, "episodes_finished": nf, "complete": nf == n_exp,
        "S_success_share": _share(sum(x["success_env"] for x in fin), nf),
        "success_recount_share": _share(sum(x["success_recount"] for x in fin), nf),
        "fallen_share": _share(sum(x["ended_by"] == "fallen" for x in fin), nf),
        "time_out_share": _share(sum(x["ended_by"] == "time_out" for x in fin), nf),
        "R_reach_share": _share(len(reach), nf),
        "M_median_carried_max_abs_x": _median(mx),
        "carried_max_abs_x_p10": float(np.percentile(mx, 10)) if mx else None,
        "carried_max_abs_x_p90": float(np.percentile(mx, 90)) if mx else None,
        "carried_max_abs_x_max": _fmax(mx) if mx else None,
        "share_beyond_twist_envelope": _share(sum(x["beyond_twist_envelope"] for x in fin), nf),
        "carried_max_abs_x_bins": bins,
        "D_setdown_share_of_reaching": _share(sum(x["set_down"] for x in reach), len(reach)),
        "reaching_with_held_above_share": _share(sum(x["class_steps"]["HELD_ABOVE"] > 0 for x in reach), len(reach)),
        "reaching_with_seated_share": _share(sum(x["class_steps"]["SEATED"] > 0 for x in reach), len(reach)),
        "dropped_to_floor_share": _share(sum(x["dropped_to_floor"] for x in fin), nf),
        "over_deck_step_share": _share(over_steps, all_steps),
        "over_deck_steps": over_steps,
        "class_step_share_of_over_deck": {c: _share(v, over_steps) for c, v in tot.items()},
        "class_steps": tot,
        "resting_reason_steps": {r: sum(x["resting_reason_steps"][r] for x in fin) for r in ("LOW", "X_BAND", "Y_BAND")},
        "episodes_with_class": {c: sum(x["class_steps"][c] > 0 for x in fin) for c in tot},
        "longest_seated_run_max": max((x["longest_seated_run"] for x in fin), default=0),
        "seated_run_lengths_hist": _hist([r for x in fin for r in x["seated_runs"]]),
        "held_above_pair_touch_share": _share(sum(x["held_above_pair_touch_steps"] for x in fin), held_steps),
        "held_above_any_hand_near_share": _share(sum(x["held_above_any_hand_near_steps"] for x in fin), held_steps),
        "over_deck_lowering_steps": sum(x["over_deck_lowering_steps"] for x in fin),
        "moving_above_seat_share_of_moving": _share(sum(x["moving_above_seat_steps"] for x in fin), tot["MOVING"]),
        "contact_force_available": all(x["force_available"] for x in fin) if fin else False,
        "reach_mechanics_median": {
            grp: {r: {"max_base_disp_m": mech(g, r, "max_base_disp_m"),
                      "max_abs_lean_x_deg": mech(g, r, "max_abs_lean_x_deg"),
                      "max_abs_yaw_rel_deg": mech(g, r, "max_abs_yaw_rel_deg"),
                      "base_dx_toward_deck_at_max_m": mech_at(g, r, "base_dx_toward_deck_m"),
                      "lean_x_toward_deck_at_max_deg": mech_at(g, r, "lean_x_toward_deck_deg")}
                  for r in ("a", "b")}
            for grp, g in (("reaching", reach), ("not_reaching", not_reach))
        },
        "mismatch_totals": mism,
        "consistent": all(v == 0 for v in mism.values()),
    }
    return {"episodes": eps, "summary": summary}


def _hist(xs: list[int]) -> dict:
    out: dict[str, int] = {}
    for v in xs:
        key = str(v) if v < SEAT_HOLD_STEPS else f">={SEAT_HOLD_STEPS}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (len(kv[0]), kv[0])))


def funding_reading(summary: dict) -> dict:
    """The predeclared per-checkpoint reading (R1-R5, or INCOMPLETE)."""
    if summary.get("as_trained") is False:
        return {"code": "INCOMPLETE", "text": "not the config as trained (CONFIG-CHECK not PASS, or the "
                "source is not the pre-quaternion-fix tree); numbers reported, no reading"}
    if summary.get("consistent") is False:                 # review 2026-09-28, before any replay: stricter
        return {"code": "INCOMPLETE", "text": "the trace's recount disagrees with the env's own buffers "
                f"({summary.get('mismatch_totals')}); numbers reported, no reading"}
    if not summary.get("complete"):
        return {"code": "INCOMPLETE", "text": f"{summary.get('episodes_finished')}/"
                f"{summary.get('episodes_expected')} episodes finished; no reading"}
    S, R = summary["S_success_share"], summary["R_reach_share"]
    M, D = summary["M_median_carried_max_abs_x"], summary["D_setdown_share_of_reaching"]
    nums = f"S={S:.3f} R={R:.3f} M={M:.3f} m D={'n/a' if D is None else f'{D:.3f}'}"
    if S >= READ_MIN_SUCCESS:
        return {"code": "R1", "kind": "places",
                "text": f"PLACES ({nums}): the deterministic policy places in >= 10% of episodes; "
                        "no Stand4 on a placement-failure argument"}
    if R < READ_MIN_REACH:
        if f32(M) >= f32(MEASURED_SHIFT_ARMS):
            inside = bool(f32(M) < f32(MEASURED_SHIFT_ARMS_TWIST))
            sub = ("inside the 0.119 m feet-planted arm+twist envelope: consistent with the reach-limit "
                   "hypothesis" if inside else
                   "partly beyond the 0.119 m feet-planted envelope: leans or shuffles, not far enough")
            return {"code": "R2", "kind": "reach gap", "within_twist_envelope": inside,
                    "text": f"REACH-SHORT ({nums}): shifted but short of the 0.19 m seat band, median "
                            f"carry {sub} -- a Stand4 that closes the reach gap is justified by this checkpoint"}
        return {"code": "R3", "kind": "no shift",
                "text": f"NO-SHIFT ({nums}): the median carry stays inside the 0.044 m arms-only envelope; "
                        "the shift itself is not learned and deck distance is not shown to bind -- "
                        "no Stand4 on this evidence"}
    if D is None or D < READ_MIN_SETDOWN:
        return {"code": "R4", "kind": "lowering/hold",
                "text": f"LOWERING/HOLD ({nums}): carried over the seat band but not set down in most "
                        "such episodes -- a Stand4 aimed at lowering/release is justified by this checkpoint"}
    return {"code": "R5", "kind": "settle",
            "text": f"SETTLE ({nums}): set down over the deck but never seated still for 12 steps -- "
                    "a Stand4 aimed at the settle is justified by this checkpoint"}


def pair_reading(readings: dict[str, dict]) -> dict:
    """Both checkpoints must give the same R2 / R4 / R5 for a Stand4 funding reading."""
    codes = {k: v.get("code") for k, v in readings.items()}
    vals = list(codes.values())
    if len(vals) == 2 and vals[0] == vals[1] and vals[0] in STAND4_READINGS:
        kind = readings[next(iter(readings))].get("kind")
        return {"funding": "STAND4_JUSTIFIED", "kind": kind, "codes": codes,
                "text": f"a Stand4 aimed at {kind} is justified by this diagnostic (both checkpoints {vals[0]})"}
    if len(vals) == 2 and vals[0] == vals[1] and vals[0] == "R1":
        why = "both checkpoints place when deterministic (R1)"
    elif len(vals) == 2 and vals[0] == vals[1] and vals[0] == "R3":
        why = "both checkpoints R3: the shift itself is not learned"
    elif "INCOMPLETE" in vals or len(vals) != 2:
        why = "a replay is INCOMPLETE"
    else:
        why = f"the checkpoints disagree ({codes})"
    return {"funding": "NO_STAND4_ON_THIS_EVIDENCE", "kind": None, "codes": codes,
            "text": f"no Stand4 on this evidence: {why}"}


# ================================================================ synthetic traces
SCENARIOS = ("no_shift", "short_shift", "reach_hold", "setdown_jitter", "place")


def _scenario_path(s: str, L: int, rng: np.random.Generator, sign: float):
    """Cube (p, v) for one synthetic episode of at most L steps; ends early on success."""
    t = np.arange(L, dtype=np.float64)
    ramp = np.clip(t / 20.0, 0.0, 1.0)
    target = {"no_shift": 0.04, "short_shift": 0.15, "reach_hold": 0.25,
              "setdown_jitter": 0.25, "place": 0.30}[s]
    x = sign * (0.005 + (target - 0.005) * ramp) + rng.normal(0, 1e-4, L)
    y = rng.normal(0, 1e-4, L)
    z = 0.55 + 0.07 * ramp                     # lifted to 0.62 while shifting
    if s in ("setdown_jitter", "place"):
        down = np.clip((t - 25.0) / 10.0, 0.0, 1.0)
        z = z - (0.62 - DECK_SEATED_Z) * down  # lowered to the seated height
    p = np.stack([x, y, z], axis=1).astype(np.float32)
    v = np.zeros_like(p)
    v[1:] = (p[1:] - p[:-1]) / STEP_DT
    speed = np.linalg.norm(v, axis=1).astype(np.float32)
    if s == "setdown_jitter":                  # never still for 12 steps once set down
        jitter = (t >= 35) & ((t.astype(int) // 5) % 2 == 1)
        speed = np.where(jitter, np.float32(0.08), speed).astype(np.float32)
    return p, v, speed


def synthetic_trace(scenario: str, num_envs: int = 4, episodes: int = EPISODES_PER_ENV,
                    ep_len: int = 60, seed: int = 0) -> dict:
    """A dense trace with the Isaac trace's keys, env flags derived by the same predicates."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; one of {SCENARIOS}")
    rng = np.random.default_rng(seed)
    per_env = []
    for n in range(num_envs):
        rows = []
        for e in range(episodes):
            sign = 1.0 if (n + e) % 2 == 0 else -1.0
            p, v, speed = _scenario_path(scenario, ep_len, rng, sign)
            seated = seated_np(p, speed)
            hold = consecutive_count(seated)
            L = ep_len
            hit = np.nonzero(hold >= SEAT_HOLD_STEPS)[0]
            if hit.size:
                L = int(hit[0]) + 1
            rows.append((e, p[:L], v[:L], speed[:L], hold[:L], L, bool(hit.size)))
        per_env.append(rows)
    T = max(sum(r[5] for r in rows) for rows in per_env) + 5
    tr = {k: np.zeros((T, num_envs, *shape), dtype=dt) for k, (shape, dt) in TRACE_SPEC.items()}
    tr["hand_force"][:] = np.float32(3.0)
    tr["hand_sdf"][:] = np.float32(0.02)
    for n, rows in enumerate(per_env):
        t0 = 0
        for e, p, v, speed, hold, L, success in rows:
            sl = slice(t0, t0 + L)
            tr["valid"][sl, n] = True
            tr["ep_idx"][sl, n] = e
            tr["ep_step"][sl, n] = np.arange(1, L + 1)
            tr["cube_pos"][sl, n] = p
            tr["cube_vel"][sl, n] = v
            tr["cube_speed"][sl, n] = speed
            tr["cube_quat_raw"][sl, n] = (1.0, 0.0, 0.0, 0.0)
            tr["cube_face_tilt_deg"][sl, n] = 0.0
            tr["cube_up_axis"][sl, n] = 2
            tr["over_deck_env"][sl, n] = over_deck_np(p)
            tr["seated_env"][sl, n] = seated_np(p, speed)
            tr["hold_counter"][sl, n] = hold
            tr["term_success"][t0 + L - 1, n] = success
            tr["term_time_out"][t0 + L - 1, n] = not success
            tr["done"][t0 + L - 1, n] = True
            for r, ysgn in (("a", 1.0), ("b", -1.0)):
                tr[f"base_pos_{r}"][sl, n] = (0.0, 0.48 * ysgn, 0.0)
                tr[f"up_{r}"][sl, n] = (0.0, 0.0, 1.0)
                tr[f"fwd_{r}"][sl, n] = (0.0, -ysgn, 0.0)
            t0 += L
        tr["ep_idx"][t0:, n] = episodes
    return tr


def synthetic_meta(scenario: str, num_envs: int, episodes: int) -> dict:
    return {"synthetic": True, "scenario": scenario, "label": SYNTH_LABEL, "num_envs": num_envs,
            "episodes_per_env": episodes, "task": TASK_ID, "seed": None,
            "config_check": {"verdict": "N/A (synthetic)"}, "source": {"pre_quat_fix": None},
            "determinism": {"repeat_equal": None}, "trace_complete": True}


# ===================================================== torch helpers (Isaac path; tested on CPU)
def hand_box_distance(rc, hands, cube_w, half: float = CUBE_HALF):
    """Signed distance of hand points to the cube's surface, in the cube frame.

    rc (N, 3, 3): cube rotation matrices (columns = body axes in world, from the
    stack's own matrix_from_quat); hands (N, K, 3) and cube_w (N, 3) in world.
    Negative inside the box. A cube is symmetric, so the as-trained 180-deg
    spawn flip does not change the distance.
    """
    import torch
    d_c = torch.einsum("nji,nkj->nki", rc, hands - cube_w[:, None, :])       # R^T d
    qd = d_c.abs() - half
    return qd.clamp(min=0.0).norm(dim=-1) + qd.max(dim=-1).values.clamp(max=0.0)


def cube_face_tilt(rc):
    """(deg, axis): angle from vertical of the cube's most vertical body axis, and
    which axis (0 x, 1 y, 2 z). Invariant to the 180-deg spawn flip; a 90-deg
    roll in the hands changes the axis, not the angle."""
    import torch
    mz, axis = rc[:, 2, :].abs().max(dim=-1)
    return torch.rad2deg(torch.acos(mz.clamp(-1.0, 1.0))), axis


# ============================================================ config / params checks
def load_params_yaml(path: Path) -> dict:
    """Isaac Lab dumps configs with python/tuple and python/object tags."""
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    def _py(loader, _suffix, node):
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        return loader.construct_scalar(node)

    _Loader.add_multi_constructor("tag:yaml.org,2002:python/", _py)
    with open(path) as fh:
        return yaml.load(fh, Loader=_Loader)


def _same_leaf(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b and type(a) is type(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            return True
        return math.isclose(fa, fb, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def config_diff(a, b, path: str = "") -> list[tuple[str, object, object]]:
    """Recursive differences; tuples and lists compare as sequences."""
    out: list[tuple[str, object, object]] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in list(a) + [k for k in b if k not in a]:
            sub = f"{path}.{k}" if path else str(k)
            if k not in a:
                out.append((sub, "<missing>", b[k]))
            elif k not in b:
                out.append((sub, a[k], "<missing>"))
            else:
                out.extend(config_diff(a[k], b[k], sub))
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [(path, list(a), list(b))]
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(config_diff(x, y, f"{path}[{i}]"))
        return out
    return [] if _same_leaf(a, b) else [(path, a, b)]


def _short(v, n: int = 120):
    s = repr(v)
    return s if len(s) <= n else s[: n - 3] + "..."


def config_check(trained: dict, live: dict) -> dict:
    """CONFIG-CHECK: gated on everything but the allowlist and the report-only top-level keys."""
    gated, reported, allowed = [], [], []
    for path, a, b in config_diff(trained, live):
        top = re.split(r"[.\[]", path, maxsplit=1)[0]
        row = {"path": path, "trained": _short(a), "live": _short(b)}
        if path in CONFIG_ALLOWLIST:
            allowed.append(row)
        elif top in CONFIG_REPORT_ONLY_TOP:
            reported.append(row)
        else:
            gated.append(row)
    return {"verdict": "PASS" if not gated else "FAIL", "gated_diffs": gated[:200],
            "n_gated": len(gated), "reported_diffs": reported[:200], "n_reported": len(reported),
            "allowlisted": allowed, "allowlist": list(CONFIG_ALLOWLIST),
            "report_only_top": list(CONFIG_REPORT_ONLY_TOP)}


def agent_check(trained: dict, live: dict) -> dict:
    diffs = [{"path": p, "trained": _short(a), "live": _short(b)}
             for p, a, b in config_diff(trained, live) if p not in AGENT_ALLOWLIST]
    return {"n_diffs": len(diffs), "diffs": diffs[:100], "allowlist": list(AGENT_ALLOWLIST),
            "note": "report-only; runner.load(strict=True) is the architecture guard"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_stand_mdp(src_root: Path):
    """The pinned stand_mdp.py by path (pure torch; no simulator)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "stand_mdp_pinned", Path(src_root) / "bhl_robust" / "tasks" / "stand_mdp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def constants_mismatch(sm) -> dict:
    """{name: (stand_mdp value, literal)} for every predeclared literal that differs."""
    bad = {}
    for name, lit in STAND_MDP_CONSTANTS.items():
        v = getattr(sm, name, None)
        if v is None or not _same_leaf(v, lit):
            bad[name] = (v, lit)
    return bad


def preflight(a: dict, check_constants: bool = True) -> dict:
    """Every check possible without Isaac. Raises SystemExit on a failure.

    `check_constants=False` skips loading the pinned stand_mdp (it imports torch);
    the Isaac path re-checks the constants on the imported module after launch.
    """
    run = Path(a["run_dir"])
    label = a["label"]
    if label not in RUNS or run.name != RUNS[label]:
        raise SystemExit(f"{label}: run dir {run.name!r} is not the predeclared {RUNS.get(label)!r}")
    ckpt = run / a["checkpoint"]
    if a["checkpoint"] != CHECKPOINT or not ckpt.is_file():
        raise SystemExit(f"{label}: checkpoint {ckpt} missing or not {CHECKPOINT}")
    env = load_params_yaml(run / "params" / "env.yaml")
    agent = load_params_yaml(run / "params" / "agent.yaml")
    why = []
    if env.get("vision") != "blind":
        why.append(f"vision={env.get('vision')!r}")
    if (env.get("terminations", {}).get("success") or {}).get("func") != "bhl_robust.tasks.stand_mdp:cube_on_side_deck":
        why.append("terminations.success is not stand_mdp:cube_on_side_deck")
    placed = (env.get("rewards", {}).get("placed") or {})
    if placed.get("func") != "bhl_robust.tasks.stand_mdp:success_bonus" or float(placed.get("weight", 0)) != 5000.0:
        why.append("rewards.placed is not success_bonus x 5000")
    scene = env.get("scene", {})
    if "deck_pos" not in scene or "deck_neg" not in scene:
        why.append("no deck_pos / deck_neg in scene")
    rot = [float(v) for v in scene.get("object", {}).get("init_state", {}).get("rot", [])]
    if rot != [1.0, 0.0, 0.0, 0.0]:
        why.append(f"trained object rot {rot} is not the raw (1,0,0,0) literal")
    if agent.get("run_name") != f"v2-cubetoshelfstand3-blind-{label}" or agent.get("experiment_name") != "task_v2":
        why.append(f"agent run_name/experiment {agent.get('run_name')!r}/{agent.get('experiment_name')!r}")
    alg, act = agent.get("algorithm", {}), agent.get("actor", {})
    if float(alg.get("entropy_coef", -1)) != 0.001 or (act.get("distribution_cfg") or {}).get("std_type") != "log":
        why.append("agent.yaml is not Stand3's runner (entropy 0.001, std_type log)")
    if int(env.get("seed", -1)) not in TRAIN_SEEDS or int(a["seed"]) in TRAIN_SEEDS:
        why.append(f"seeds: trained {env.get('seed')}, replay {a['seed']} (must differ from {TRAIN_SEEDS})")
    src = Path(a["src_root"])
    for f in PINNED_FILES:
        if not (src / f).is_file():
            why.append(f"pinned source missing {f}")
    cfg_txt = (src / "bhl_robust/tasks/coop_lift_env_cfg.py").read_text() if (src / "bhl_robust/tasks/coop_lift_env_cfg.py").is_file() else ""
    if RAW_OBJECT_LITERAL not in cfg_txt or "native_quat(_OBJECT_ROT_WXYZ)" in cfg_txt:
        why.append("pinned coop_lift_env_cfg.py is not the pre-fix source (raw object literal absent)")
    bad = {}
    if check_constants:
        try:
            bad = constants_mismatch(load_stand_mdp(src))
        except Exception as exc:                                 # noqa: BLE001
            bad = {"load": repr(exc)}
    if bad:
        why.append(f"pinned stand_mdp constants differ from the predeclared literals: {bad}")
    if int(a["num_envs"]) != NUM_ENVS or int(a["episodes"]) != EPISODES_PER_ENV or int(a["seed"]) != REPLAY_SEED:
        why.append(f"protocol {a['num_envs']} envs / {a['episodes']} episodes / seed {a['seed']} is not "
                   f"the predeclared {NUM_ENVS} / {EPISODES_PER_ENV} / {REPLAY_SEED}")
    for suffix in (".trace.npz", ".meta.json"):
        if Path(a["out_prefix"] + suffix).exists():
            why.append(f"refusing to overwrite {a['out_prefix'] + suffix}")
    if why:
        raise SystemExit(f"PREFLIGHT {label}: FAIL -- " + "; ".join(why))
    return {"label": label, "run_dir": str(run), "checkpoint": str(ckpt),
            "checkpoint_sha256": sha256_file(ckpt), "task": TASK_ID, "train_job": TRAIN_JOB,
            "trained_seed": int(env["seed"]), "replay_seed": int(a["seed"]),
            "num_envs": int(a["num_envs"]), "episodes_per_env": int(a["episodes"]),
            "src_root": str(src), "trained_object_rot": rot}


def read_args_file(path: str) -> dict:
    a = json.loads(Path(path).read_text())
    need = ("label", "run_dir", "checkpoint", "src_root", "out_prefix", "seed", "num_envs",
            "episodes", "enable_cameras", "config_check_mode")
    missing = [k for k in need if k not in a]
    if missing:
        raise SystemExit(f"args file {path}: missing {missing}")
    if a["config_check_mode"] not in ("gate", "report"):
        raise SystemExit(f"args file {path}: config_check_mode must be gate|report")
    return a


def _write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    os.replace(tmp, path)


# ======================================================================= Isaac side
def cmd_isaac(args) -> int:
    a = read_args_file(args.args_file)
    plan = preflight(a, check_constants=False)
    src_root = Path(a["src_root"]).resolve()
    sys.path.insert(0, str(src_root))       # the pinned tree, never $REPO/src
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True, enable_cameras=bool(a["enable_cameras"]))
    app = launcher.app
    try:
        _isaac_body(a, plan, launcher, src_root)
    except BaseException:                                        # noqa: BLE001
        tb = traceback.format_exc()
        print("STAND3-REPLAY-ERROR\n" + tb, flush=True)           # stdout: Kit swallows stderr
        Path(a["out_prefix"] + ".error.txt").write_text(tb)
        raise
    finally:
        app.close()
    return 0


def _isaac_body(a: dict, plan: dict, launcher, src_root: Path) -> None:
    import gymnasium as gym
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True              # as scripts/train.py
    torch.backends.cudnn.allow_tf32 = True
    import bhl_robust
    bfile = Path(bhl_robust.__file__).resolve()
    if src_root not in bfile.parents:
        raise RuntimeError(f"bhl_robust imported from {bfile}, not from the pinned {src_root}")
    from bhl_robust import compat as _compat
    _compat.apply()
    import berkeley_humanoid_lite.tasks  # noqa: F401
    import bhl_robust.tasks  # noqa: F401
    from bhl_robust.quat_order import quat_order, unpack_wxyz
    from bhl_robust.tasks import coop_lift_env_cfg as clc
    from bhl_robust.tasks import coop_lift_mdp as coop
    from bhl_robust.tasks import stand_mdp as sm
    from bhl_robust.tasks.coop_depth_env_cfg import apply_depth_flags
    from isaaclab.utils.io import dump_yaml
    from isaaclab.utils.math import matrix_from_quat

    bad = constants_mismatch(sm)
    if bad:
        raise RuntimeError(f"pinned stand_mdp constants differ from the predeclared literals: {bad}")
    run = Path(a["run_dir"])
    out_prefix = a["out_prefix"]
    N, E, seed = int(a["num_envs"]), int(a["episodes"]), int(a["seed"])
    # The configs are built the way scripts/train.py built them: its
    # hydra_task_config with no overrides (the Stand3 OVERRIDE_FILE was empty) is
    # register_task (registry load + preset resolution) and no Hydra round trip.
    # resolve_task_config is that path; it reads sys.argv, so it gets a bare one.
    cfg_path = "isaaclab_tasks.utils.hydra.resolve_task_config (no overrides), as scripts/train.py"
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        from isaaclab_tasks.utils.hydra import resolve_task_config
        cfg, agent = resolve_task_config(TASK_ID, "rsl_rl_cfg_entry_point")
    except Exception as exc:                                     # noqa: BLE001
        # Never run before this job: a failure here must not cost the replay.
        # CONFIG-CHECK verifies whichever path built the cfg.
        print(f"[cfg] resolve_task_config failed ({exc!r}); falling back to the gym-spec entry points",
              flush=True)
        spec = gym.spec(TASK_ID)
        cfg = spec.kwargs["env_cfg_entry_point"]()
        agent = spec.kwargs["rsl_rl_cfg_entry_point"]()
        cfg_path = f"gym.spec entry points (resolve_task_config failed: {exc!r})"
    finally:
        sys.argv = saved_argv
    cfg.scene.num_envs = N
    cfg.seed = seed
    cfg.sim.device = launcher.device
    if hasattr(cfg.sim, "log_dir"):          # as scripts/train.py: Isaac Lab's own logs off /tmp (allowlisted)
        cfg.sim.log_dir = os.path.join(os.environ.get("TMPDIR") or os.path.dirname(out_prefix),
                                       f"stand3-replay-isaaclab-{os.getpid()}")
        os.makedirs(cfg.sim.log_dir, exist_ok=True)
    clc.apply_strategy_flags(cfg)            # as scripts/train.py after hydra
    apply_depth_flags(cfg)
    env = gym.make(TASK_ID, cfg=cfg)
    u = env.unwrapped

    live_yaml = out_prefix + ".live_env.yaml"
    dump_yaml(live_yaml, u.cfg)
    cc = config_check(load_params_yaml(run / "params" / "env.yaml"), load_params_yaml(Path(live_yaml)))
    cc["mode"] = a["config_check_mode"]
    rot_live = [float(v) for v in u.cfg.scene.object.init_state.rot]
    order = quat_order()
    source = {"src_root": str(src_root), "bhl_robust_file": str(bfile), "quat_order": order,
              "object_init_rot_live": rot_live,
              "pre_quat_fix": bool(rot_live == [1.0, 0.0, 0.0, 0.0] and order == "xyzw"),
              "note": AS_TRAINED_NOTE}
    meta = {"label": LABEL, "synthetic": False, "report_only": REPORT_ONLY_NOTE, **plan,
            "args": a, "source": source, "config_check": cc, "cfg_path": cfg_path,
            "stack": os.environ.get("BHL_STACK"),
            "cube_size": [float(v) for v in u.cfg.scene.object.spawn.size],
            "step_dt": float(u.step_dt), "max_episode_length": int(u.max_episode_length),
            "trace_complete": False}
    meta_path = Path(out_prefix + ".meta.json")
    print(f"CONFIG-CHECK {a['label']}: {cc['verdict']} (gated diffs {cc['n_gated']}, reported {cc['n_reported']}, "
          f"mode {cc['mode']}); object rot live {rot_live}, quat order {order}", flush=True)
    if cc["verdict"] != "PASS" and cc["mode"] == "gate":
        meta["stopped"] = "CONFIG-CHECK FAIL in gate mode: not the config as trained; no replay"
        _write_json_atomic(meta_path, meta)
        env.close()
        return

    from importlib.metadata import version
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    try:
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
        agent = handle_deprecated_rsl_rl_cfg(agent, version("rsl-rl-lib"))
    except ImportError:
        pass
    agent_yaml = out_prefix + ".live_agent.yaml"
    dump_yaml(agent_yaml, agent)
    meta["agent_check"] = agent_check(load_params_yaml(run / "params" / "agent.yaml"),
                                      load_params_yaml(Path(agent_yaml)))
    wrapped = RslRlVecEnvWrapper(env)        # no clip_actions: scripts/train.py builds it without
    runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=u.device)
    runner.load(str(run / a["checkpoint"]))
    policy = runner.get_inference_policy(device=u.device)

    ra, rb = u.scene["robot_a"], u.scene["robot_b"]
    hand_ids = {}
    for r, robot in (("a", ra), ("b", rb)):
        ids, names = robot.find_bodies(list(HANDS), preserve_order=True)
        if list(names) != list(HANDS):
            raise RuntimeError(f"robot_{r} hands resolved to {names}")
        hand_ids[r] = list(ids)
    contact = {}
    contact_err = None
    try:
        for r in ("a", "b"):
            sensor = u.scene[f"contact_{r}"]
            ids, names = sensor.find_bodies(list(HANDS), preserve_order=True)
            if list(names) != list(HANDS):
                raise RuntimeError(f"contact_{r} hands resolved to {names}")
            contact[r] = (sensor, list(ids))
    except Exception as exc:                                     # noqa: BLE001
        contact, contact_err = {}, repr(exc)
    meta["robot_root_body"] = {"a": ra.body_names[0], "b": rb.body_names[0]}
    meta["contact_proxy"] = {"available": bool(contact), "error": contact_err,
                             "hand_near_m": HAND_NEAR_M, "hand_force_n": HAND_FORCE_N,
                             "columns": list(HAND_COLUMNS)}

    api = SimpleNamespace(obj_state=sm._obj_state, over_deck_mask=sm.over_deck_mask,
                          seated_mask=sm.seated_mask, t=coop._t, tilt_from_quat=coop._tilt_from_quat,
                          matrix_from_quat=matrix_from_quat, unpack_wxyz=unpack_wxyz)
    T = int(u.max_episode_length) * E + 50
    buf = make_buffers(T, N, u.device)
    state = {"t": 0, "conv_dev": torch.zeros((), device=u.device)}
    record = make_recorder(u, buf, api, hand_ids, contact, state)
    roll = run_episodes(u, wrapped, policy, buf, record, E, seed, state)
    meta["determinism"] = roll["determinism"]
    t_end = roll["t_end"]
    trace = trace_from_buffers(buf, t_end)
    trace_path = Path(out_prefix + ".trace.npz")
    tmp = trace_path.with_name(f".{trace_path.stem}.tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, **trace)
    os.replace(tmp, trace_path)
    meta.update({
        "trace_file": trace_path.name, "num_steps": t_end,
        "episodes_done_min": roll["episodes_done_min"], "trace_complete": roll["all_done"],
        "convention_check": {"max_abs_R22_minus_quat_upz": float(state["conv_dev"]),
                             "ok": bool(float(state["conv_dev"]) < 1e-4),
                             "note": "matrix_from_quat vs unpack_wxyz (the fall check's) up-z; lean/yaw valid iff ok"},
    })
    _write_json_atomic(meta_path, meta)
    print(f"STAND3-REPLAY-TRACE {a['label']}: {trace_path} ({t_end} steps, episodes done min "
          f"{meta['episodes_done_min']}/{E}, deterministic repeat {meta['determinism']['repeat_equal']}, "
          f"convention dev {meta['convention_check']['max_abs_R22_minus_quat_upz']:.2e})", flush=True)
    env.close()


# ---------------------------------------------------- rollout pieces (fake-env tested)
def make_buffers(T: int, N: int, device) -> dict:
    """Dense (T, N, ...) torch buffers for every TRACE_SPEC array."""
    import torch
    torch_dt = {np.bool_: torch.bool, np.int8: torch.int8, np.int16: torch.int16, np.float32: torch.float32}
    buf = {k: torch.zeros((T, N, *shape), dtype=torch_dt[dt], device=device) for k, (shape, dt) in TRACE_SPEC.items()}
    buf["hand_force"].fill_(float("nan"))
    return buf


def make_recorder(u, buf: dict, api, hand_ids: dict, contact: dict, state: dict):
    """record(t): the post-physics, pre-reset state of every env into row t.

    `api` carries the env-facing functions: obj_state (stand_mdp._obj_state),
    over_deck_mask / seated_mask (stand_mdp), t (coop_lift_mdp._t),
    tilt_from_quat (coop_lift_mdp._tilt_from_quat, the fall check's),
    matrix_from_quat (Isaac Lab's, the stack's own convention) and unpack_wxyz.
    """
    import torch
    ra, rb, obj = u.scene["robot_a"], u.scene["robot_b"], u.scene["object"]
    half = float(CUBE_HALF)

    def record(t: int) -> None:
        p, speed = api.obj_state(u)
        buf["cube_pos"][t] = p
        buf["cube_speed"][t] = speed
        buf["cube_vel"][t] = api.t(obj.data.root_lin_vel_w)[:, :3]
        q = api.t(obj.data.root_quat_w)
        buf["cube_quat_raw"][t] = q
        rc = api.matrix_from_quat(q)
        tilt_deg, up_axis = cube_face_tilt(rc)
        buf["cube_face_tilt_deg"][t] = tilt_deg
        buf["cube_up_axis"][t] = up_axis.to(torch.int8)
        buf["over_deck_env"][t] = api.over_deck_mask(p)
        buf["seated_env"][t] = api.seated_mask(p, speed)
        hold = getattr(u, "_bhl_deck_hold", None)
        if hold is not None:
            buf["hold_counter"][t] = hold.to(torch.int16)
        tm = u.termination_manager
        buf["term_success"][t] = tm.get_term("success")
        buf["term_fallen"][t] = tm.get_term("fallen")
        buf["term_time_out"][t] = tm.time_outs
        buf["ep_step"][t] = u.episode_length_buf.to(torch.int16)
        cube_w = api.t(obj.data.root_pos_w)[:, :3]
        sdf_cols, force_cols = [], []
        for r, robot in (("a", ra), ("b", rb)):
            rq = api.t(robot.data.root_quat_w)
            rm = api.matrix_from_quat(rq)
            buf[f"up_{r}"][t] = rm[:, :, 2]
            buf[f"fwd_{r}"][t] = rm[:, :, 0]
            buf[f"tilt_{r}"][t] = api.tilt_from_quat(robot)
            w, x, y, z = api.unpack_wxyz(rq)
            dev_ = (rm[:, 2, 2] - (1.0 - 2.0 * (x * x + y * y))).abs().max()
            state["conv_dev"] = torch.maximum(state["conv_dev"], dev_)
            buf[f"base_pos_{r}"][t] = api.t(robot.data.root_pos_w)[:, :3] - u.scene.env_origins
            hands = api.t(robot.data.body_pos_w)[:, hand_ids[r], :]            # (N, 2, 3)
            sdf_cols.append(hand_box_distance(rc, hands, cube_w, half))
            if contact:
                sensor, cids = contact[r]
                force_cols.append(api.t(sensor.data.net_forces_w)[:, cids, :].norm(dim=-1))
        buf["hand_sdf"][t] = torch.cat(sdf_cols, dim=1)
        if force_cols:
            buf["hand_force"][t] = torch.cat(force_cols, dim=1)

    return record


def run_episodes(u, wrapped, policy, buf: dict, record, episodes: int, seed: int, state: dict) -> dict:
    """Step until every env has finished `episodes` episodes (or the buffer is full).

    The termination manager's compute is hooked on the instance: it runs after
    physics and before the auto-reset inside env.step, so row t holds the state
    that step's terminations saw, including each episode's terminal step. Row
    t's valid / ep_idx are set before the step: the episode the step belongs to.
    """
    import torch
    T, N = buf["valid"].shape
    tm = u.termination_manager
    orig_compute = tm.compute

    def hooked_compute(*args, **kwargs):
        out = orig_compute(*args, **kwargs)
        if state["t"] < T:
            record(state["t"])
        return out

    t_end = 0
    with torch.inference_mode():
        u.reset(seed=seed)
        obs = wrapped.get_observations()
        a1, a2 = policy(obs), policy(obs)
        det = {"action_mode": "rsl-rl 5 MLPModel.forward(stochastic_output=False) -> distribution mean",
               "repeat_equal": bool(torch.equal(a1, a2)), "action_dim": int(a1.shape[-1])}
        tm.compute = hooked_compute
        ep_count = torch.zeros(N, dtype=torch.long, device=buf["valid"].device)
        try:
            for t in range(T):
                state["t"] = t
                actions = policy(obs)
                buf["action_absmax"][t] = actions.abs().max(dim=-1).values
                buf["valid"][t] = ep_count < episodes
                buf["ep_idx"][t] = ep_count.clamp(max=episodes).to(torch.int8)
                obs, _, dones, _ = wrapped.step(actions)
                d = dones.bool()
                buf["done"][t] = d
                ep_count += (d & (ep_count < episodes)).long()
                t_end = t + 1
                if bool((ep_count >= episodes).all()):
                    break
        finally:
            tm.compute = orig_compute
    return {"t_end": t_end, "determinism": det, "episodes_done_min": int(ep_count.min()),
            "all_done": bool((ep_count >= episodes).all())}


def trace_from_buffers(buf: dict, t_end: int) -> dict:
    return {k: v[:t_end].cpu().numpy().astype(TRACE_SPEC[k][1]) for k, v in buf.items()}


# ====================================================================== host side
def cmd_make_args(args) -> int:
    """The launcher's per-checkpoint args file, with the predeclared protocol."""
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    a = {"label": args.label, "run_dir": args.run_dir, "checkpoint": CHECKPOINT,
         "src_root": args.src_root, "out_prefix": args.out_prefix, "seed": REPLAY_SEED,
         "num_envs": NUM_ENVS, "episodes": EPISODES_PER_ENV, "enable_cameras": True,
         "config_check_mode": args.config_check}
    _write_json_atomic(out, a)
    read_args_file(str(out))
    return 0


def cmd_dry_run(args) -> int:
    a = read_args_file(args.args_file)
    plan = preflight(a)
    print(json.dumps(plan, indent=1))
    print(f"DRY-RUN {a['label']}: OK (run dir, {CHECKPOINT} sha256 {plan['checkpoint_sha256'][:12]}, "
          f"params = Stand3 blind seed {plan['trained_seed']}, pinned source pre-quat-fix, "
          f"constants = predeclared literals; replay seed {plan['replay_seed']}, {plan['num_envs']} envs x "
          f"{plan['episodes_per_env']} episodes). No Isaac step ran.")
    return 0


def cmd_synth(args) -> int:
    for suffix in (".trace.npz", ".meta.json"):
        if Path(args.out_prefix + suffix).exists():
            raise SystemExit(f"refusing to overwrite {args.out_prefix + suffix}")
    tr = synthetic_trace(args.scenario, args.num_envs, args.episodes, args.ep_len, args.seed)
    np.savez_compressed(args.out_prefix + ".trace.npz", **tr)
    meta = synthetic_meta(args.scenario, args.num_envs, args.episodes)
    meta["trace_file"] = Path(args.out_prefix + ".trace.npz").name
    _write_json_atomic(Path(args.out_prefix + ".meta.json"), meta)
    print(f"SYNTH {args.scenario}: {args.out_prefix}.trace.npz ({tr['valid'].shape[0]} steps x "
          f"{args.num_envs} envs) -- {SYNTH_LABEL}")
    return 0


def classify_files(trace_path: Path, meta_path: Path) -> dict:
    meta = json.loads(Path(meta_path).read_text())
    with np.load(trace_path) as z:
        tr = {k: z[k] for k in z.files}
    missing = [k for k in TRACE_SPEC if k not in tr]
    if missing:
        raise SystemExit(f"{trace_path}: trace lacks {missing}")
    res = summarize(tr, int(meta["num_envs"]), int(meta.get("episodes_per_env", EPISODES_PER_ENV)))
    s = res["summary"]
    synthetic = bool(meta.get("synthetic"))
    cc = (meta.get("config_check") or {}).get("verdict")
    as_trained = bool(synthetic or (cc == "PASS" and (meta.get("source") or {}).get("pre_quat_fix")))
    s["as_trained"] = as_trained
    s["complete"] = bool(s["complete"] and meta.get("trace_complete", False))
    reading = funding_reading(s)
    return {"label": meta.get("label"), "synthetic": synthetic, "report_only": REPORT_ONLY_NOTE,
            "as_trained": as_trained, "as_trained_note": AS_TRAINED_NOTE,
            "thresholds": {**STAND_MDP_CONSTANTS, "HAND_NEAR_M": HAND_NEAR_M, "HAND_FORCE_N": HAND_FORCE_N,
                           "READ_MIN_SUCCESS": READ_MIN_SUCCESS, "READ_MIN_REACH": READ_MIN_REACH,
                           "READ_MIN_SETDOWN": READ_MIN_SETDOWN},
            "meta": meta, "summary": s, "reading": reading, "episodes": res["episodes"]}


def _fmt(v, nd=3):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def cmd_classify(args) -> int:
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    res = classify_files(Path(args.trace), Path(args.meta))
    _write_json_atomic(out, res)
    s, r, tag = res["summary"], res["reading"], args.tag
    pre = "SYNTHETIC " if res["synthetic"] else ""
    cs = s["class_step_share_of_over_deck"]
    print(f"{pre}REPLAY {tag}: {'COMPLETE' if s['complete'] and s['as_trained'] else 'INCOMPLETE'} "
          f"({s['episodes_finished']}/{s['episodes_expected']} episodes, as_trained={res['as_trained']}, "
          f"consistent={s['consistent']} {s['mismatch_totals']}) [{res['label']}]")
    print(f"{pre}REPLAY {tag}: success {_fmt(s['S_success_share'])} fallen {_fmt(s['fallen_share'])} "
          f"time_out {_fmt(s['time_out_share'])}; carried max|x| median {_fmt(s['M_median_carried_max_abs_x'])} "
          f"p90 {_fmt(s['carried_max_abs_x_p90'])} max {_fmt(s['carried_max_abs_x_max'])} m; "
          f">= 0.19 in {_fmt(s['R_reach_share'])}, >= 0.119 in {_fmt(s['share_beyond_twist_envelope'])}; "
          f"bins {s['carried_max_abs_x_bins']}")
    print(f"{pre}REPLAY {tag}: over_deck {_fmt(s['over_deck_step_share'])} of steps ({s['over_deck_steps']}); "
          f"of those MOVING {_fmt(cs['MOVING'])} HELD_ABOVE {_fmt(cs['HELD_ABOVE'])} SEATED {_fmt(cs['SEATED'])} "
          f"RESTING_NOT_SEATED {_fmt(cs['RESTING_NOT_SEATED'])} (reasons {s['resting_reason_steps']}); "
          f"longest seated run {s['longest_seated_run_max']}; set down in {_fmt(s['D_setdown_share_of_reaching'])} "
          f"of reaching episodes; HELD_ABOVE with both robots touching {_fmt(s['held_above_pair_touch_share'])}; "
          f"MOVING at/above the seat window {_fmt(s['moving_above_seat_share_of_moving'])} of MOVING")
    print(f"{pre}READING {tag}: {r['code']} -- {r['text']} ({REPORT_ONLY_NOTE})")
    return 0


def cmd_report(args) -> int:
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    readings, files = {}, {}
    synthetic = False
    for tag, path in zip(args.tags, args.replays):
        try:
            d = json.loads(Path(path).read_text())
            readings[tag] = d["reading"]
            synthetic = synthetic or bool(d.get("synthetic"))
        except Exception as exc:                                 # noqa: BLE001
            readings[tag] = {"code": "INCOMPLETE", "text": f"no readable replay JSON ({exc!r})"}
        files[tag] = str(path)
    pr = pair_reading(readings)
    label = SYNTH_LABEL if synthetic else LABEL
    res = {"report_only": REPORT_ONLY_NOTE, "synthetic": synthetic, "label": label, "task": TASK_ID,
           "train_job": TRAIN_JOB, "readings": readings, "pair": pr, "files": files,
           "rule": "Stand4 justified only if both checkpoints give the same R2, R4 or R5 reading"}
    _write_json_atomic(out, res)
    pre = "SYNTHETIC " if synthetic else ""
    for tag, r in readings.items():
        print(f"{pre}READING {tag}: {r.get('code')} -- {r.get('text')}")
    print(f"{pre}FUNDING (report-only, not a gate): {pr['funding']} -- {pr['text']} [{label}]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("isaac", help="the replay (v60 container)")
    p.add_argument("--args-file", required=True)
    p = sub.add_parser("dry-run", help="checks without Isaac")
    p.add_argument("--args-file", required=True)
    p = sub.add_parser("make-args", help="write a per-checkpoint args file (predeclared protocol)")
    p.add_argument("--label", choices=tuple(RUNS), required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--src-root", required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--config-check", choices=("gate", "report"), default="gate")
    p.add_argument("--out", required=True)
    p = sub.add_parser("synth", help="synthetic trace + meta (plumbing test)")
    p.add_argument("--scenario", choices=SCENARIOS, required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--episodes", type=int, default=EPISODES_PER_ENV)
    p.add_argument("--ep-len", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("classify", help="trace + meta -> replay JSON")
    p.add_argument("--trace", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default="?")
    p = sub.add_parser("report", help="two replay JSONs -> funding reading")
    p.add_argument("--replays", nargs=2, required=True)
    p.add_argument("--tags", nargs=2, default=["s0", "s1"])
    p.add_argument("--out", required=True)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {"isaac": cmd_isaac, "dry-run": cmd_dry_run, "make-args": cmd_make_args, "synth": cmd_synth,
            "classify": cmd_classify, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
