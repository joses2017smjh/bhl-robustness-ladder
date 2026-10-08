"""Staged plate-crossing intervention on the exact ten Mission7 replays.

The unchanged replay and its recorded gate-opening times remain the baseline.
This intervention changes only the command stream near each unopened plate:
approach a pre-plate pose, settle with zero translation, then make a short
straight crossing with yaw correction frozen. Correct-side staging is retained;
wrong-side staging is allowed only when the correct plate is more than 1.0 m
away. Geometry, activation schedule, fall predicate, and deterministic reset
stream are unchanged.

``--stage-gait turnboth`` (opt-in; M2 of docs/SOLUTIONS_2026-10-01.md) swaps
the gait for the stage only: at the stage's start env.controller.policy becomes
TurnBoth-s0's (arms-turn-turnboth-s0, the one qualified turning checkpoint; its
deploy.yaml differs from the shipped gait's only in the policy path and the
unused command_velocity, checked at load) and prev_actions is zeroed.  After the
unchanged approach and settle the stage turns in place to the door direction,
crosses straight forward (no lateral command) until the base is past the plate,
turns back to the heading it had at takeover, and at hand-back restores the
shipped policy and zeroes prev_actions again.  LEARNED gaits, SCRIPTED stage,
ORACLE layout and plate pose.  The default ``shipped`` path is unchanged.

``--stage-gait m3`` (opt-in; M3 of docs/SOLUTIONS_2026-10-01.md, conditional on
M2's bench FAIL) keeps the SHIPPED gait (no policy swap).  After the unchanged
approach and settle it turns WHILE STEPPING toward the door direction (forward
0.30 m/s plus a 0.40 rad/s yaw command: the shipped gait does not turn in place),
crosses straight forward with a heading hold once aligned, then turns back to
the takeover heading the same way; a stall watchdog (turn and cross) recovers a
bounded number of times and then hands back.  Constants and their sources are
declared at M3_* below.  LEARNED gait, SCRIPTED stage, ORACLE layout and plate pose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from bhl_robust.mission.approach_debug import DebugEnv, command_action, physical_sample, wrap
from mission7_overnight import write


# ---- --stage-gait turnboth (M2) ----------------------------------------------------------------
# Declared 2026-10-02 before any bench, replay or route episode ran with it; nothing below is tuned
# on a result.  The laws are TurnBoth-s0's scored maze controller (random_maze.TurnWalkController:
# turn in place, then walk forward with a heading hold, never sideways), except the turn rate.
STAGE_GAITS = ("shipped", "turnboth", "m3")
SHIPPED_EXPORT = "logs/rsl_rl/humanoid/2026-08-18_20-57-50_arms-dr1.0-s0/exported"   # = MissionEnv's gait
TURNBOTH_EXPORT = "logs/rsl_rl/humanoid/2026-09-25_18-54-44_arms-turn-turnboth-s0/exported"
TURNBOTH_POLICY_SHA256 = "562ceed71e04df2bc317edd352c9ebf59fb13fee5d0cc6708fa82aff33b55c95"
TURNBOTH_FREE_DEPLOY_KEYS = ("policy_checkpoint_path", "command_velocity")   # the only keys allowed to differ
# 0.40 rad/s is the largest yaw rate Mission 7's command interface admits (tanh x 0.4 in
# MissionEnv.step); TurnWalkController's 0.6 cannot be commanded through it.  TurnBoth-s0 was
# qualified at +-0.6 rad/s; its turn at 0.4 rad/s was unmeasured before this protocol was declared;
# smoke showed 2/3 turns timing out (cpu_m7_plate_bench.sbatch header); nothing was changed in response.
TURNBOTH_TURN_RATE = .40
TURNBOTH_TURN_EXIT = .15          # rad, TurnWalkController.turn_exit
TURNBOTH_TURN_MAX_S = 2 * np.pi / TURNBOTH_TURN_RATE   # a full revolution; no turn needs more than half
TURNBOTH_CRUISE = .30             # m/s, TurnWalkController.cruise = the stage's crossing speed
TURNBOTH_K_YAW = 1.2              # TurnWalkController.k_yaw: heading hold while crossing
TURNBOTH_WZ_WALK = .40            # TurnWalkController.wz_walk
TURNBOTH_CROSS_CLEAR_M = .35      # = mission7_gates.CLEAR_ALONG_M, used when --cross-clear is not given
_TURNBOTH_CACHE = {}

# ---- --stage-gait m3 (M3) ----------------------------------------------------------------------
# Declared 2026-10-02 before any bench-v2, replay or route episode ran with it; nothing is tuned on a
# bench, replay or route result.  The gait is NEVER swapped: env.controller.policy stays the shipped
# arms-dr1.0-s0 policy, and prev_actions is touched only by the watchdog's recovery.  Every constant comes
# from kinematics, the existing code, or ONE pre-registered sweep on open floor built from EXPLORATION train
# layout 255 (walls, doors and plates disabled as in the 2026-09-21 gait study; selection rule written before
# the result was read; srun job 21508399): candidates .15/.20/.25/.30 m/s forward with +-0.40 rad/s for 2 s
# from a 3 s standstill; .15 and .20 never stepped (< 0.08 rad), .25 stepped at +0.40 only, .30 stepped both
# ways (+0.70 rad / 0.55 m, -0.89 rad / 0.52 m, no fall) -> the smallest candidate turning both ways.
M3_TURN_VX = .30        # m/s while turning; = the existing "measured 0.30 m/s onset threshold" (PulseApproachController)
M3_TURN_WZ = .40        # rad/s: the largest yaw rate Mission 7's command interface admits (tanh x 0.4)
M3_ALIGN_EXIT = .25     # rad: BearingController.heading_tolerance (the shipped gait's turn-then-walk oracle)
M3_TURN_MAX_S = 2 * np.pi / M3_TURN_WZ   # a full revolution at the commanded rate bounds a turn (as TURNBOTH_TURN_MAX_S)
M3_CROSS_VX = .30       # m/s: the stage's crossing speed (_world_command) and the route's near-plate cap
M3_K_YAW = 1.2          # heading hold while crossing: MeasuredRouteController's clip(1.2 x error, +-0.35)
M3_WZ_HOLD = .35
M3_CROSS_CLEAR_M = .35  # = mission7_gates.CLEAR_ALONG_M, used when --cross-clear is not given
# Turn direction: the short way; when the error is within M3_ALIGN_EXIT of +-pi (no short way), the way whose
# forward arc ends toward the door centreline (kinematics: a forward U-turn at wz > 0 ends 2 vx/wz to the
# right of the heading it ends on), i.e. wz sign = +side for the turn and -side for the turn back.
# Stall watchdog (turn and cross only): RecoveryRouteController's predicate, the base moved < 0.05 m over the
# last 1.2 s; recovery = prev_actions := 0 (the stage's cross-kick, matched evidence of un-sticking this gait,
# 2/2 genuine stalls) and a 0.40 s pulse at 0.30 m/s (the route's recovery pulse length and speed) BACKWARD,
# off the plate edge the stalls stand on; then the interrupted phase resumes with a fresh window.  After
# M3_MAX_RECOVERIES recoveries a further stall hands back to the route (no turn back).
# Timing as polled (disclosed, not changed): the inherited settle (`now < self.phase_until`) and the m3 cross bound
# (`now < self.cross_start_s + self.cross_max_s`) compare float-accumulated d.time without an epsilon, so at the
# bench's 0.2 s polling the declared 0.40 s settle lasts 0.40 or 0.60 s and the 4.0 s cross bound 4.0 or 4.2 s.
M3_STALL_WINDOW_S = 1.2
M3_STALL_MIN_M = .05
M3_RECOVER_VX = .30
M3_RECOVER_S = .40
# Pooled per crossing: m3_recoveries resets only at takeover (_start), so either guarded phase may spend both.
# Sized, not enforced per phase: 2 = the two phases the watchdog guards (turn, cross) x RecoveryRouteController's
# one recovery per stalled (waypoint, phase).
M3_MAX_RECOVERIES = 2
M3_PHASES = ("turn", "cross", "turn_back", "recover")


def m3_constants():
    """The declared M3 constants (provenance and preflight)."""
    return {"policy": "shipped arms-dr1.0-s0 (env.controller.policy is never swapped)",
            "turn_vx_mps": M3_TURN_VX, "turn_wz_rps": M3_TURN_WZ, "align_exit_rad": M3_ALIGN_EXIT,
            "turn_max_s": M3_TURN_MAX_S, "cross_vx_mps": M3_CROSS_VX, "k_yaw": M3_K_YAW, "wz_hold_rps": M3_WZ_HOLD,
            "cross_clear_m": M3_CROSS_CLEAR_M, "stall_window_s": M3_STALL_WINDOW_S, "stall_min_m": M3_STALL_MIN_M,
            "recover_vx_mps": -M3_RECOVER_VX, "recover_s": M3_RECOVER_S, "max_recoveries": M3_MAX_RECOVERIES}


def turnboth_check(upstream, shipped_cfg=None):
    """Verify the TurnBoth-s0 export before it is swapped in; return its provenance.

    The policy file must have the pinned sha256, and its deploy.yaml must equal
    the shipped gait's in every key but TURNBOTH_FREE_DEPLOY_KEYS, so swapping
    env.controller.policy alone is functionally the same as running TurnBoth-s0's
    own RlController (action scale, defaults, limits, gains, 75 observations,
    history 0).
    """
    from omegaconf import OmegaConf
    upstream = Path(upstream)
    export = upstream / TURNBOTH_EXPORT
    policy = export / "policy.onnx"
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    if digest != TURNBOTH_POLICY_SHA256:
        raise RuntimeError(f"TurnBoth-s0 policy {policy} has sha256 {digest}, expected {TURNBOTH_POLICY_SHA256}")
    if shipped_cfg is None:
        shipped_cfg = OmegaConf.load(upstream / SHIPPED_EXPORT / "deploy.yaml")
    turnboth = OmegaConf.to_container(OmegaConf.load(export / "deploy.yaml"))
    shipped = OmegaConf.to_container(shipped_cfg)
    differing = sorted(key for key in set(turnboth) | set(shipped)
                       if turnboth.get(key) != shipped.get(key))
    unexpected = [key for key in differing if key not in TURNBOTH_FREE_DEPLOY_KEYS]
    if unexpected:
        raise RuntimeError(f"TurnBoth-s0 deploy.yaml differs from the shipped gait's in {unexpected}; "
                           "a policy swap would not be equivalent")
    if (turnboth["num_actions"], turnboth["num_observations"], turnboth["history_length"]) != (22, 75, 0):
        raise RuntimeError("TurnBoth-s0 is not a 22-action, 75-observation, history-0 export")
    return {"export": str(export), "policy": str(policy), "policy_sha256": digest,
            "deploy_keys_differing": differing}


def load_turnboth_policy(env):
    """TurnBoth-s0 as a CPU ONNX policy for env's RlController, checked once per export."""
    key = str(Path(env.upstream) / TURNBOTH_EXPORT)
    if key not in _TURNBOTH_CACHE:
        from bhl_robust.mission.env import CpuPolicy   # the class MissionEnv runs the shipped gait with
        info = turnboth_check(env.upstream, env.cfg)
        _TURNBOTH_CACHE[key] = (CpuPolicy(info["policy"]), info)
    return _TURNBOTH_CACHE[key]


def _source(campaign):
    data = json.loads((campaign / "fullroute/legacy-doors.json").read_text())
    if not data.get("complete"):
        raise RuntimeError("legacy door route is not complete")
    return data


def _world_command(env, vector, speed=0.30):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.zeros(3)
    yaw = float(np.arctan2(
        2 * (env.runner.d.qpos[env.slot.qpos_adr + 3] * env.runner.d.qpos[env.slot.qpos_adr + 6]
           + env.runner.d.qpos[env.slot.qpos_adr + 4] * env.runner.d.qpos[env.slot.qpos_adr + 5]),
        1 - 2 * (env.runner.d.qpos[env.slot.qpos_adr + 5] ** 2
                 + env.runner.d.qpos[env.slot.qpos_adr + 6] ** 2)))
    world = vector / norm * float(speed)
    body = np.array([[np.cos(yaw), np.sin(yaw)],
                     [-np.sin(yaw), np.cos(yaw)]]) @ world
    return np.array([np.clip(body[0], -.4, .4), np.clip(body[1], -.35, .35), 0.])


def _yaw(env):
    q = env.runner.d.qpos[env.slot.qpos_adr + 3:env.slot.qpos_adr + 7]
    return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


def _plate_state(env, door, side):
    center, direction = env.layout.door(door)
    plate = env.layout.plate(door, side)
    direction = np.asarray(direction, dtype=float)
    return np.asarray(plate), direction


class PlateStage:
    """One bounded staged maneuver per unopened correct plate."""

    def __init__(self, env, approach_radius=.78, settle_s=.40, cross_s=1.20,
                 stage_lateral_m=None, wait_open_s=0., press_hold=False,
                 creep_mps=.15, creep_max_m=.45, pre_point_m=.30, cross_clear_m=None,
                 cross_max_s=4.0, cross_kick=False, align_yaw=False, align_tol_rad=.20,
                 align_max_s=1.5, stage_gait="shipped", turnboth_policy=None):
        self.env = env
        self.approach_radius = float(approach_radius)
        self.settle_s = float(settle_s)
        self.cross_s = float(cross_s)
        # Where the body centres for the crossing, as a lateral offset from the
        # door centreline toward the plate side.  None keeps the exact replay
        # behaviour: the plate centre itself, 0.42 m off the centreline, which
        # puts the body 0.29-0.39 m from the corridor wall against a 0.316 m
        # half-body.  The plate is 0.24 m in radius and is pressed by a foot,
        # not by the body centre, so a smaller offset keeps the body clear of
        # the wall while the wall-side foot still lands on the plate.  Capture
        # (approach_radius from the plate) and the fall predicate are unchanged.
        self.stage_lateral_m = None if stage_lateral_m is None else float(stage_lateral_m)
        # After the fixed settle, optionally keep standing on the plate for up
        # to wait_open_s more until the door is open, instead of crossing into
        # a closed door.  0 keeps the exact replay behaviour.
        self.wait_open_s = float(wait_open_s)
        self.wait_open_until = None
        # Press-and-hold: after the settle, creep along the door direction at
        # creep_mps until the plate registers a press (door open) or creep_max_m
        # has been covered, hold still with activate asserted for up to
        # wait_open_s, then cross.  The default stage waits at the pre-point
        # 0.30 m before the plate, where nothing presses it: 8 of 9 F1+F2+F3
        # in-stage terminations were crossings into a still-closed door.
        self.press_hold = bool(press_hold)
        self.creep_mps = float(creep_mps)
        self.creep_max_m = float(creep_max_m)
        self.creep_origin = None
        # Where the settle point sits before the plate along the door direction.
        # 0.30 m is the replay value; the baseline trace shows the feet already
        # on the plate edge there, and approach-phase trips (doors/13, 5, 7)
        # walking onto the raised plate from the side.
        self.pre_point_m = float(pre_point_m)
        # Cross until the body is this far PAST the plate centre along the door
        # direction (None = the replay's fixed 1.20 s).  From the 0.40 s
        # standstill the gait covers only ~0.13 m in 1.20 s, so the replay
        # crossing ends with the base 3 cm short of the plate edge and the feet
        # on it -- which is where the route resumes with full lateral and every
        # one of the eight post-exit falls in Campaign A tripped on the plate.
        self.cross_clear_m = None if cross_clear_m is None else float(cross_clear_m)
        self.cross_max_s = float(cross_max_s)
        self.cross_start_s = None
        # After the settle standstill the gait restarts for ~0.2 m at the
        # 0.30 m/s command and then stands on the plate edge: the feedback fixed
        # point.  A stage-owned one-shot prev_actions := 0 at the start of the
        # crossing is the intervention with matched evidence of un-sticking it
        # (2/2 genuine stalls).  Recorded in history; the replay gate must pass.
        self.cross_kick = bool(cross_kick)
        self.kick_events = []
        # The crossing is a world-frame vector with no yaw correction, so a
        # robot that reached the pre-point sideways crosses sideways: in the
        # three V2 replay falls the body command was [0.01, 0.30, 0.0] for two
        # seconds before the plate-edge trip.  Turn in place toward the door
        # direction during the settle (bounded), so the crossing is forward.
        self.align_yaw = bool(align_yaw)
        self.align_tol_rad = float(align_tol_rad)
        self.align_max_s = float(align_max_s)
        self.align_until = None
        self.align_events = []
        # A wrong-side plate is only an intervention target when the route is
        # clearly not entering the intended plate.  The threshold separates
        # the layout-13 wrong-side trace from layout-4's earlier near miss.
        self.wrong_side_min_correct_distance = 1.0
        self.door = None
        self.side = None
        self.phase = "recorded"
        self.phase_until = 0.
        self.done = set()
        self.history = []
        # Opt-in stage gait (module header).  "shipped" leaves every line above and below as it was.
        if stage_gait not in STAGE_GAITS:
            raise ValueError(f"stage_gait must be one of {STAGE_GAITS}, got {stage_gait!r}")
        self.stage_gait = stage_gait
        if stage_gait == "turnboth":
            if self.align_yaw or self.press_hold:
                raise ValueError("--stage-gait turnboth turns in place itself; --align-yaw and --press-hold "
                                 "do not compose with it")
            if turnboth_policy is None:
                turnboth_policy, self.turnboth_provenance = load_turnboth_policy(env)
            else:
                self.turnboth_provenance = {"policy": "supplied by the caller"}
            self.turnboth_policy = turnboth_policy
            self.turnboth_clear_m = (TURNBOTH_CROSS_CLEAR_M if self.cross_clear_m is None
                                     else self.cross_clear_m)
            self.shipped_policy = None
            self.takeover_yaw = None
            self.turn_start_s = None
            self.gait_events = []
        elif stage_gait == "m3":
            if self.align_yaw or self.press_hold:
                raise ValueError("--stage-gait m3 turns while stepping itself; --align-yaw and --press-hold "
                                 "do not compose with it")
            self.m3_clear_m = M3_CROSS_CLEAR_M if self.cross_clear_m is None else self.cross_clear_m
            self.takeover_yaw = None
            self.turn_start_s = None
            self.gait_events = []        # M3 never swaps the policy: this stays empty
            self.m3_events = []          # takeover, watchdog recoveries, resumes, watchdog hand-backs
            self.m3_recoveries = 0
            self.m3_resume = None
            self.m3_recover_until = None
            self.m3_track = []

    # ---- M3 (stage_gait == "m3") ---------------------------------------------------------------
    def _m3_segment(self, now, xy):
        """Start a fresh watchdog window (phase entry or resume after a recovery)."""
        self.m3_track = [(float(now), np.asarray(xy, dtype=float).copy())]

    def _m3_stalled(self, now, xy):
        """True when the base moved < M3_STALL_MIN_M since its position M3_STALL_WINDOW_S ago."""
        xy = np.asarray(xy, dtype=float)
        self.m3_track.append((float(now), xy.copy()))
        if now - self.m3_track[0][0] < M3_STALL_WINDOW_S - 1e-9:
            return False
        ref = max(i for i, (t, _) in enumerate(self.m3_track) if t <= now - M3_STALL_WINDOW_S + 1e-9)
        self.m3_track = self.m3_track[ref:]
        return float(np.linalg.norm(xy - self.m3_track[0][1])) < M3_STALL_MIN_M

    def _m3_turn_sign(self, err, centre_sign):
        """Short way; within M3_ALIGN_EXIT of +-pi, the way whose arc ends toward the door centreline."""
        if abs(err) > np.pi - M3_ALIGN_EXIT:
            return float(centre_sign)
        return float(np.copysign(1., err))

    def _m3_handback(self, recorded, now, **fields):
        self.done.add(self.door)
        self.history.append({"time_s": now, "door": int(self.door), "side": int(self.side),
                             "phase": "recorded", **fields})
        self.door = None
        self.side = None
        self.phase = "recorded"
        return np.asarray(recorded, dtype=float), "recorded"

    def _m3_stall(self, recorded, now):
        """The watchdog fired in self.phase (turn or cross): recover, or hand back once recoveries are spent."""
        if self.m3_recoveries < M3_MAX_RECOVERIES:
            self.m3_recoveries += 1
            controller = self.env.controller
            before = float(np.linalg.norm(controller.prev_actions))
            controller.prev_actions[:] = 0.
            entry = {"time_s": now, "door": int(self.door), "phase": "recover", "resume": self.phase,
                     "recovery": self.m3_recoveries, "prev_actions_before_norm": before}
            self.history.append(dict(entry))
            self.m3_events.append(dict(entry, event="stall_recover"))
            self.m3_resume = self.phase
            self.m3_recover_until = now + M3_RECOVER_S
            self.phase = "recover"
            return np.array([-M3_RECOVER_VX, 0., 0.]), self.phase
        self.m3_events.append({"time_s": now, "door": int(self.door), "event": "stall_handback",
                               "phase": self.phase, "recoveries": self.m3_recoveries})
        return self._m3_handback(recorded, now, watchdog_handback=True, stalled_in=self.phase)

    def _m3_command(self, recorded, now, xy, direction):
        """turn (while stepping) -> cross -> turn_back -> hand-back on the shipped gait, after approach and settle."""
        env = self.env
        heading = float(np.arctan2(direction[1], direction[0]))
        if self.phase == "recover":
            if now < self.m3_recover_until - 1e-9:
                return np.array([-M3_RECOVER_VX, 0., 0.]), self.phase
            self.phase, self.m3_resume = self.m3_resume, None
            self.m3_events.append({"time_s": now, "door": int(self.door), "event": "resume", "phase": self.phase})
            self._m3_segment(now, xy)
        if self.phase == "turn":
            err = wrap(heading - _yaw(env))
            if abs(err) >= M3_ALIGN_EXIT and now < self.turn_start_s + M3_TURN_MAX_S:
                if self._m3_stalled(now, xy):
                    return self._m3_stall(recorded, now)
                return np.array([M3_TURN_VX, 0., self._m3_turn_sign(err, self.side) * M3_TURN_WZ]), self.phase
            # Every crossing starts with a "cross" history entry: mission7_gates scores it from there.
            self.phase = "cross"
            self.cross_start_s = now
            self.history.append({"time_s": now, "door": int(self.door), "phase": self.phase,
                                 "yaw_error_rad": float(err), "turn_timed_out": bool(abs(err) >= M3_ALIGN_EXIT)})
            self._kick(now)
            self._m3_segment(now, xy)
        if self.phase == "cross":
            err = wrap(heading - _yaw(env))
            along = float((xy - _plate_state(env, self.door, self.side)[0]) @ direction)
            if along < self.m3_clear_m and now < self.cross_start_s + self.cross_max_s:
                if self._m3_stalled(now, xy):
                    return self._m3_stall(recorded, now)
                # Straight forward with the route's heading hold: no lateral command.
                return np.array([M3_CROSS_VX, 0., float(np.clip(M3_K_YAW * err, -M3_WZ_HOLD, M3_WZ_HOLD))]), self.phase
            self.phase = "turn_back"
            self.turn_start_s = now
            self.history.append({"time_s": now, "door": int(self.door), "phase": self.phase,
                                 "along_m": along, "cleared": bool(along >= self.m3_clear_m)})
        if self.phase == "turn_back":
            err = wrap(self.takeover_yaw - _yaw(env))
            if abs(err) >= M3_ALIGN_EXIT and now < self.turn_start_s + M3_TURN_MAX_S:
                return np.array([M3_TURN_VX, 0., self._m3_turn_sign(err, -self.side) * M3_TURN_WZ]), self.phase
            # Hand-back: the "recorded" entry stays the last one (the route probe reads history[-1]["door"]).
            return self._m3_handback(recorded, now, yaw_error_rad=float(err),
                                     turn_back_timed_out=bool(abs(err) >= M3_ALIGN_EXIT))
        raise AssertionError(self.phase)

    def _swap_gait(self, now, to):
        """Swap env.controller.policy for the stage and zero prev_actions (turnboth only)."""
        controller = self.env.controller
        before = float(np.linalg.norm(controller.prev_actions))
        if to == "turnboth":
            self.shipped_policy = controller.policy
            controller.policy = self.turnboth_policy
        else:
            controller.policy = self.shipped_policy
            self.shipped_policy = None
        controller.prev_actions[:] = 0.
        self.gait_events.append({"time_s": float(now), "door": int(self.door), "event": f"swap_to_{to}",
                                 "prev_actions_before_norm": before,
                                 "prev_actions_after_norm": float(np.linalg.norm(controller.prev_actions))})

    def _turnboth_command(self, recorded, now, xy, direction):
        """turn -> cross -> turn_back -> hand-back, after the unchanged approach and settle."""
        env = self.env
        heading = float(np.arctan2(direction[1], direction[0]))
        if self.phase == "turn":
            err = wrap(heading - _yaw(env))
            if abs(err) >= TURNBOTH_TURN_EXIT and now < self.turn_start_s + TURNBOTH_TURN_MAX_S:
                return np.array([0., 0., float(np.copysign(TURNBOTH_TURN_RATE, err))]), self.phase
            # Every crossing starts with a "cross" history entry: mission7_gates scores it from there.
            self.phase = "cross"
            self.cross_start_s = now
            self.history.append({"time_s": now, "door": int(self.door), "phase": self.phase,
                                 "yaw_error_rad": float(err),
                                 "turn_timed_out": bool(abs(err) >= TURNBOTH_TURN_EXIT)})
            self._kick(now)
        if self.phase == "cross":
            err = wrap(heading - _yaw(env))
            along = float((xy - _plate_state(env, self.door, self.side)[0]) @ direction)
            if along < self.turnboth_clear_m and now < self.cross_start_s + self.cross_max_s:
                # Straight forward with TurnWalkController's walk law: no lateral command.
                return np.array([TURNBOTH_CRUISE * max(0., float(np.cos(err))), 0.,
                                 float(np.clip(TURNBOTH_K_YAW * err, -TURNBOTH_WZ_WALK, TURNBOTH_WZ_WALK))]), self.phase
            self.phase = "turn_back"
            self.turn_start_s = now
            self.history.append({"time_s": now, "door": int(self.door), "phase": self.phase,
                                 "along_m": along, "cleared": bool(along >= self.turnboth_clear_m)})
        if self.phase == "turn_back":
            err = wrap(self.takeover_yaw - _yaw(env))
            if abs(err) >= TURNBOTH_TURN_EXIT and now < self.turn_start_s + TURNBOTH_TURN_MAX_S:
                return np.array([0., 0., float(np.copysign(TURNBOTH_TURN_RATE, err))]), self.phase
            # Hand-back: the shipped policy and a zeroed prev_actions, recorded before the
            # "recorded" entry, which stays the last one (the route probe reads history[-1]["door"]).
            self._swap_gait(now, "shipped")
            self.done.add(self.door)
            self.history.append({"time_s": now, "door": int(self.door), "side": int(self.side),
                                 "phase": "recorded", "yaw_error_rad": float(err),
                                 "turn_back_timed_out": bool(abs(err) >= TURNBOTH_TURN_EXIT)})
            self.door = None
            self.side = None
            self.phase = "recorded"
            return np.asarray(recorded, dtype=float), "recorded"
        raise AssertionError(self.phase)

    def _with_yaw(self, env, command, direction):
        """Moving yaw correction toward the door direction (the frozen gait does
        not turn in place: 1.5 s of a pure yaw command from standstill produced
        no rotation in the V3 replays of layouts 7 and 14)."""
        if not self.align_yaw:
            return command
        err = wrap(np.arctan2(direction[1], direction[0]) - _yaw(env))
        out = np.asarray(command, dtype=float).copy()
        out[2] = float(np.clip(1.2 * err, -.35, .35))
        return out

    def _kick(self, now):
        if not self.cross_kick:
            return
        controller = self.env.controller
        before = float(np.linalg.norm(controller.prev_actions))
        controller.prev_actions[:] = 0.
        self.kick_events.append({"time_s": float(now), "door": int(self.door),
                                 "prev_actions_before_norm": before, "event": "cross_kick"})
        self.history.append({"time_s": float(now), "door": int(self.door), "phase": "cross_kick",
                             "prev_actions_before_norm": before})

    def _start(self, door, side, now):
        self.door = door
        self.side = side
        self.align_until = None
        self.phase = "approach"
        self.phase_until = float(now)
        self.history.append({"time_s": float(now), "door": int(door),
                             "side": int(side), "phase": self.phase})
        if self.stage_gait == "turnboth":
            self.takeover_yaw = _yaw(self.env)
            self._swap_gait(now, "turnboth")
            self.gait_events[-1]["takeover_yaw_rad"] = self.takeover_yaw
        elif self.stage_gait == "m3":   # no swap: the shipped policy keeps running
            self.takeover_yaw = _yaw(self.env)
            self.m3_recoveries = 0
            self.m3_events.append({"time_s": float(now), "door": int(door), "event": "takeover",
                                   "takeover_yaw_rad": self.takeover_yaw})

    def command(self, recorded):
        env, runner, slot = self.env, self.env.runner, self.env.slot
        now = float(runner.d.time)
        xy = runner.d.xpos[slot.body_id, :2].copy()
        if self.phase == "recorded":
            candidates = []
            for door, opened in enumerate(env.state.open):
                if opened or door in self.done:
                    continue
                correct = env.layout.correct_sides[door]
                correct_distance = float(np.linalg.norm(
                    xy - _plate_state(env, door, correct)[0]))
                candidates.append((door, correct))
                wrong = -correct
                if correct_distance > self.wrong_side_min_correct_distance:
                    candidates.append((door, wrong))
            nearest = min(
                ((float(np.linalg.norm(xy - _plate_state(env, door, side)[0])),
                  int(side != env.layout.correct_sides[door]), door, side)
                 for door, side in candidates),
                default=(float("inf"), 1, None, None),
            )
            if nearest[0] <= self.approach_radius:
                self._start(nearest[2], nearest[3], now)
            else:
                return np.asarray(recorded, dtype=float), "recorded"
        plate, direction = _plate_state(env, self.door, self.side)
        if self.stage_lateral_m is not None:
            # Anchor on the same cell the plate is anchored on.  layout.plate()
            # is xy(route[k]) + lateral*side*.42 - direction*.12; the first
            # version of this offset used layout.door()'s centre, half a cell
            # further along, and staged the robot ~0.75 m past the plate.
            k = env.layout.door_indices[self.door]
            anchor = np.asarray(env.layout.xy(env.layout.route[k]), dtype=float)
            lateral = np.array([-direction[1], direction[0]])
            plate = anchor + lateral * self.side * self.stage_lateral_m - direction * .12
        if self.phase == "approach":
            pre = plate - direction * self.pre_point_m
            if np.linalg.norm(xy - pre) <= .13:
                self.phase = "settle"
                self.phase_until = now + self.settle_s
                self.history.append({"time_s": now, "door": int(self.door),
                                     "side": int(self.side), "phase": self.phase})
                return np.zeros(3), self.phase
            return self._with_yaw(env, _world_command(env, pre - xy), direction), self.phase
        if self.phase == "settle":
            if self.align_yaw and (not self.align_events or self.align_events[-1]["door"] != int(self.door)):
                err = wrap(np.arctan2(direction[1], direction[0]) - _yaw(env))
                self.align_events.append({"time_s": now, "door": int(self.door), "yaw_error_rad": float(err),
                                          "aligned": bool(abs(err) <= self.align_tol_rad)})
            if now < self.phase_until:
                return np.zeros(3), self.phase
            if self.press_hold and not env.state.open[self.door]:
                self.phase = "creep"
                self.creep_origin = xy.copy()
                self.history.append({"time_s": now, "door": int(self.door),
                                     "side": int(self.side), "phase": self.phase})
            elif self.wait_open_s > 0. and not env.state.open[self.door]:
                if self.wait_open_until is None:
                    self.wait_open_until = now + self.wait_open_s
                    self.history.append({"time_s": now, "door": int(self.door),
                                         "side": int(self.side), "phase": "wait_open"})
                if now < self.wait_open_until:
                    return np.zeros(3), self.phase
            self.wait_open_until = None
            if self.stage_gait == "turnboth":
                self.phase = "turn"
                self.turn_start_s = now
                self.history.append({"time_s": now, "door": int(self.door), "phase": self.phase,
                                     "yaw_error_rad": wrap(float(np.arctan2(direction[1], direction[0])) - _yaw(env))})
                return self._turnboth_command(recorded, now, xy, direction)
            if self.stage_gait == "m3":
                self.phase = "turn"
                self.turn_start_s = now
                self.history.append({"time_s": now, "door": int(self.door), "phase": self.phase,
                                     "yaw_error_rad": wrap(float(np.arctan2(direction[1], direction[0])) - _yaw(env))})
                self._m3_segment(now, xy)
                return self._m3_command(recorded, now, xy, direction)
            self.phase = "cross"
            self.phase_until = now + self.cross_s
            self.cross_start_s = now
            self.history.append({"time_s": now, "door": int(self.door), "phase": self.phase})
            self._kick(now)
        if self.stage_gait == "turnboth" and self.phase in ("turn", "cross", "turn_back"):
            return self._turnboth_command(recorded, now, xy, direction)
        if self.stage_gait == "m3" and self.phase in M3_PHASES:
            return self._m3_command(recorded, now, xy, direction)
        if self.phase == "creep":
            travelled = float(np.linalg.norm(xy - self.creep_origin))
            if env.state.open[self.door] or travelled >= self.creep_max_m:
                self.phase = "hold"
                self.phase_until = now + self.wait_open_s
                self.history.append({"time_s": now, "door": int(self.door),
                                     "side": int(self.side), "phase": self.phase,
                                     "creep_m": travelled, "open": bool(env.state.open[self.door])})
                return np.zeros(3), self.phase
            return _world_command(env, direction, speed=self.creep_mps), self.phase
        if self.phase == "hold":
            if not env.state.open[self.door] and now < self.phase_until:
                return np.zeros(3), self.phase
            self.phase = "cross"
            self.phase_until = now + self.cross_s
            self.cross_start_s = now
            self.history.append({"time_s": now, "door": int(self.door), "phase": self.phase,
                                 "open": bool(env.state.open[self.door])})
            self._kick(now)
        if self.phase == "cross":
            # The short crossing deliberately carries no yaw correction.
            if self.cross_clear_m is not None:
                along = float((xy - _plate_state(env, self.door, self.side)[0]) @ direction)
                keep = along < self.cross_clear_m and now < self.cross_start_s + self.cross_max_s
            else:
                keep = now < self.phase_until
            if keep:
                return self._with_yaw(env, _world_command(env, direction), direction), self.phase
            self.done.add(self.door)
            self.history.append({"time_s": now, "door": int(self.door),
                                 "side": int(self.side), "phase": "recorded"})
            self.door = None
            self.side = None
            self.phase = "recorded"
            return np.asarray(recorded, dtype=float), "recorded"
        raise AssertionError(self.phase)


def _episode(env, index, source_episode, baseline, stage_lateral_m=None, wait_open_s=0., press_hold=False,
             pre_point_m=.30, cross_clear_m=None, settle_s=.40, cross_kick=False, align_yaw=False,
             stage_gait="shipped"):
    runner = env.runner
    stage = PlateStage(env, stage_lateral_m=stage_lateral_m, wait_open_s=wait_open_s, press_hold=press_hold,
                       pre_point_m=pre_point_m, cross_clear_m=cross_clear_m, settle_s=settle_s, cross_kick=cross_kick,
                       align_yaw=align_yaw, stage_gait=stage_gait)
    samples = []
    maximum_recorded_pose_difference = 0.
    for reference in source_episode["diagnostic_trace"]:
        env.state.open = [t is not None and runner.d.time + 1e-8 >= t
                          for t in source_episode["activation_s"]]
        env._gates()
        recorded = np.asarray(reference["command"], dtype=float)
        effective, phase = stage.command(recorded)
        runner.contact_trace = []
        runner.step([env.controller.update(runner.observe(0, effective))])
        sample = physical_sample(env, effective, phase)
        sample["recorded_command"] = recorded.tolist()
        sample["effective_command"] = effective.tolist()
        sample["contact_events"] = list(runner.contact_trace)
        samples.append(sample)
        maximum_recorded_pose_difference = max(
            maximum_recorded_pose_difference,
            float(np.linalg.norm(np.asarray(sample["xy"]) - np.asarray(reference["xy"]))),
            abs(sample["tilt"] - reference["tilt"]),
            abs(wrap(sample["yaw"] - reference["yaw"])),
        )
        if abs(sample["time_s"] - reference["time_s"]) >= 1e-7:
            raise AssertionError("intervention replay time diverged")
        if not np.isfinite(runner.d.qpos).all() or runner.d.warning.number.sum():
            raise FloatingPointError("plate-stage replay physics warning")
    falls = [sample for sample in samples if sample["tilt"] >= .78]
    contacts = [event for sample in samples for event in sample["contact_events"]
                if event["world_geom"].startswith("plate_")]
    row = {
        "layout_index": index,
        "layout_seed": source_episode["layout"]["seed"],
        "original_elapsed_s": source_episode["elapsed_s"],
        "replay_elapsed_s": samples[-1]["time_s"],
        "fall": bool(falls),
        "first_fall_s": None if not falls else falls[0]["time_s"],
        "maximum_tilt": max(sample["tilt"] for sample in samples),
        "maximum_recorded_pose_difference": maximum_recorded_pose_difference,
        "plate_contacts": len(contacts),
        "peak_plate_normal_force_N": max((event["normal_force_N"] for event in contacts), default=0.),
        "peak_plate_tangent_force_N": max((event["tangent_force_N"] for event in contacts), default=0.),
        "minimum_plate_distance_m": min((event["distance_m"] for event in contacts), default=None),
        "stage_history": stage.history,
        "baseline": {key: baseline.get(key) for key in ("fall_time_s", "fall_phase", "minimum_clearances_m",
                                                         "failure_stage_relative_to_plate")},
    }
    if stage_gait != "shipped":   # the default row keeps its keys and their order
        row["stage_gait"] = stage_gait
        row["gait_events"] = stage.gait_events
        if stage_gait == "m3":   # the turnboth row keeps its keys too
            row["m3_events"] = stage.m3_events
    row["samples"] = samples
    return row


def run(args, out):
    source = _source(args.campaign)
    baseline_path = args.baseline / "result.json"
    baseline = json.loads(baseline_path.read_text())
    if not baseline.get("unchanged_replay_matches"):
        raise RuntimeError("authoritative baseline is not exact; intervention is not interpretable")
    baseline_rows = {row["layout_index"]: row for row in baseline["episodes_detail"]}
    selected = [(index, episode) for index, episode in enumerate(source["episodes"]) if episode.get("fall")]
    if args.smoke:
        selected = selected[:1]
    only = getattr(args, "only", "")
    if only:
        keep = {int(x) for x in only.split(",") if x.strip()}
        selected = [(i, e) for i, e in selected if i in keep]
    env = DebugEnv(args.repo, out / "cache", stage="doors", split="validation", seed=1000)
    rows, next_reset = [], 0
    for index, episode in selected:
        for reset_index in range(next_reset, index + 1):
            env.reset(reset_index)
        next_reset = index + 1
        rows.append(_episode(env, index, episode, baseline_rows[index],
                             stage_lateral_m=getattr(args, 'stage_lateral', None),
                             wait_open_s=getattr(args, 'wait_open', 0.),
                             press_hold=getattr(args, 'press_hold', False),
                             pre_point_m=getattr(args, 'pre_point', .30),
                             cross_clear_m=getattr(args, 'cross_clear', None),
                             settle_s=getattr(args, 'settle_s', .40),
                             cross_kick=getattr(args, 'cross_kick', False),
                             align_yaw=getattr(args, 'align_yaw', False),
                             stage_gait=getattr(args, 'stage_gait', 'shipped')))
        write(out / "episodes.json", {"complete": False, "episodes": rows})
    report = {
        "complete": True,
        "status": "COMPLETED_INTERVENTION" if not args.smoke else "COMPLETED_SMOKE",
        "intervention": "staged plate maneuver: pre-plate approach, 0.40 s settle, 1.20 s straight crossing with yaw correction frozen; wrong-side staging only when the correct plate is over 1.0 m away",
        "geometry_unchanged": True,
        "activation_schedule_unchanged": True,
        "fall_predicate_unchanged": True,
        "baseline_result": str(baseline_path),
        "episodes": len(rows),
        "falls": sum(row["fall"] for row in rows),
        "upright": sum(not row["fall"] for row in rows),
        "exact_replay_gate_passed": len(rows) == 10 and sum(not row["fall"] for row in rows) == 10,
        "episode_comparison": [{key: value for key, value in row.items() if key != "samples"} for row in rows],
    }
    if getattr(args, 'stage_gait', 'shipped') != "shipped":   # the default report is unchanged
        report["intervention"] = stage_gait_description(args.stage_gait)
        report["stage_gait"] = args.stage_gait
        report["stage_gait_provenance"] = (load_turnboth_policy(env)[1] if args.stage_gait == "turnboth"
                                           else m3_constants())
    write(out / "episodes.json", {"complete": True, "episodes": rows})
    write(out / "result.json", report)


def stage_gait_description(stage_gait):
    """The intervention text of a non-default stage gait (turnboth's is unchanged)."""
    return {"turnboth": turnboth_description, "m3": m3_description}[stage_gait]()


def m3_description():
    return ("staged plate maneuver with M3 on the SHIPPED gait (LEARNED gait, SCRIPTED stage, ORACLE layout and plate "
            "pose; no policy swap): pre-plate approach and settle unchanged; turn while stepping toward the door "
            f"direction at {M3_TURN_VX:.2f} m/s forward and {M3_TURN_WZ:.2f} rad/s yaw until within "
            f"{M3_ALIGN_EXIT:.2f} rad (bounded {M3_TURN_MAX_S:.2f} s; the short way, toward the door centreline when "
            f"there is none); straight-forward crossing at {M3_CROSS_VX:.2f} m/s with heading hold clip({M3_K_YAW:.1f} "
            f"x error, +-{M3_WZ_HOLD:.2f}) and no lateral command, until the base is --cross-clear (default "
            f"{M3_CROSS_CLEAR_M:.2f} m) past the plate centre or cross_max_s pass; turn back to the takeover heading "
            f"the same way; stall watchdog in the turn and the crossing: base moved < {M3_STALL_MIN_M:.2f} m over "
            f"{M3_STALL_WINDOW_S:.1f} s -> prev_actions := 0 and {M3_RECOVER_S:.2f} s backward at "
            f"{M3_RECOVER_VX:.2f} m/s, then the phase resumes, at most {M3_MAX_RECOVERIES} times, then hand-back; "
            "wrong-side staging only when the correct plate is over 1.0 m away")


def turnboth_description():
    return ("staged plate maneuver with the TurnBoth-s0 stage gait (LEARNED gaits, SCRIPTED stage, ORACLE layout "
            "and plate pose): at takeover env.controller.policy := TurnBoth-s0 and prev_actions := 0; pre-plate "
            "approach and settle unchanged; turn in place to the door direction at "
            f"{TURNBOTH_TURN_RATE:.2f} rad/s until within {TURNBOTH_TURN_EXIT:.2f} rad (bounded "
            f"{TURNBOTH_TURN_MAX_S:.2f} s); straight-forward crossing at {TURNBOTH_CRUISE:.2f} m/s x max(0, cos "
            f"heading error) with heading hold clip({TURNBOTH_K_YAW:.1f} x error, +-{TURNBOTH_WZ_WALK:.2f}) and no "
            "lateral command, until the base is --cross-clear (default "
            f"{TURNBOTH_CROSS_CLEAR_M:.2f} m) past the plate centre or cross_max_s pass; turn back to the takeover "
            "heading the same way; at hand-back the shipped policy and prev_actions := 0; wrong-side staging only "
            "when the correct plate is over 1.0 m away")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage-lateral", type=float, default=None,
                        help="PlateStage body-centre lateral offset (m); default None = plate "
                             "centre, the exact 10/10 replay behaviour")
    parser.add_argument("--wait-open", type=float, default=0.,
                        help="PlateStage bounded wait on the plate for the door before crossing (s); 0 = replay behaviour")
    parser.add_argument("--press-hold", action="store_true",
                        help="creep onto the plate after the settle and hold until the door opens")
    parser.add_argument("--pre-point", type=float, default=.30,
                        help="settle point distance before the plate along the door direction (m)")
    parser.add_argument("--cross-clear", type=float, default=None,
                        help="cross until the body is this far past the plate centre (m); default fixed 1.20 s")
    parser.add_argument("--settle-s", type=float, default=.40, help="settle standstill before the crossing (s); replay 0.40")
    parser.add_argument("--cross-kick", action="store_true", help="zero prev_actions once at the start of the crossing")
    parser.add_argument("--align-yaw", action="store_true", help="turn in place toward the door direction during the settle")
    parser.add_argument("--only", default="", help="comma-separated layout indices to replay (local checks); default all ten")
    parser.add_argument("--stage-gait", choices=STAGE_GAITS, default="shipped",
                        help="gait the stage runs on: shipped (default, the exact replay behaviour), turnboth "
                             "(TurnBoth-s0 swapped in for the stage: turn in place, straight crossing, turn back) or "
                             "m3 (shipped gait: turn while stepping, straight crossing, turn back, stall watchdog)")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.stage_gait == "turnboth" and (args.align_yaw or args.press_hold):
        parser.error("--stage-gait turnboth does not compose with --align-yaw or --press-hold")
    if args.stage_gait == "m3" and (args.align_yaw or args.press_hold):
        parser.error("--stage-gait m3 does not compose with --align-yaw or --press-hold")
    args.repo = args.repo.resolve()
    args.campaign = args.campaign.resolve()
    args.baseline = args.baseline.resolve()
    args.out = args.out.resolve()
    if args.preflight:
        import sys
        preflight = {"status": "PREFLIGHT_OK", "python": sys.executable, "mujoco": mujoco.__version__,
                     "stage_lateral_m": args.stage_lateral, "wait_open_s": args.wait_open,
                     "pre_point_m": args.pre_point, "cross_clear_m": args.cross_clear,
                     "settle_s": args.settle_s, "cross_kick": args.cross_kick, "align_yaw": args.align_yaw, "campaign": str(args.campaign),
                     "baseline": str(args.baseline), "out": str(args.out)}
        if args.stage_gait != "shipped":   # the default preflight line is unchanged
            preflight["stage_gait"] = args.stage_gait
            preflight["stage_gait_check"] = (turnboth_check(args.repo / "external/Berkeley-Humanoid-Lite")
                                             if args.stage_gait == "turnboth" else m3_constants())
        print(json.dumps(preflight, sort_keys=True), flush=True)
        raise SystemExit(0)
    args.out.mkdir(parents=True, exist_ok=True)
    run(args, args.out)
    print("MISSION7_PLATE_STAGE_COMPLETE", flush=True)
