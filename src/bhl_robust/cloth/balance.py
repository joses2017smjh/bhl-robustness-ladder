"""A stance the free-base robot can hold, and the leg controller that holds it.

The pinch squat the cloth layout was built on cannot stand: its knees need
5.5-6.0 N m against a 6 N m limit, sag up to 0.3 rad and let the robot fall
backward within a second, arm moving or not (21317172; ``scripts/cloth/
stance_mujoco.py``, whose MuJoCo robot reproduces that fall to a few degrees).

This is the candidate found in that MuJoCo model, driven by Isaac's actuator
model (PD 20 N m/rad, 2 N m s/rad, 6 N m legs):

* **stance** -- hip -0.5, knee 1.0, ankle -0.5 on both legs: base level, soles
  flat, peak static leg torque about 3.1 N m, root 5.95 cm above the squat's;
* **feedforward** -- the settled leg torques of that stance over Kp, added to
  the leg position targets so the PD holds the pose instead of sagging to it;
* **ankle feedback** -- base pitch and roll, and their rates, in the robot's
  heading frame, onto the ankle pitch and roll targets.

In that model it stood 10 of 10 noisy resets for 10 s with the arm still (worst
tilt 3.4 deg) and 10 of 10 for 6 s with the arm playing the scripted sweep
(11.3 deg). Its neighbours in gain space mostly did not survive the arm, so it
is narrow. The gains, the leg gains and the limits are unchanged from the
upstream actuator; this is a controller on the targets, not a stiffer robot.
Isaac-free: the numbers are pinned to ``results/cloth/stance/search.json`` by a
test, and ``cloth_sort_mdp.SweepAction`` applies them.
"""

from __future__ import annotations

import math

STANCE: dict[str, float] = {
    "leg_left_hip_pitch_joint": -0.50, "leg_right_hip_pitch_joint": -0.50,
    "leg_left_knee_pitch_joint": 1.00, "leg_right_knee_pitch_joint": 1.00,
    "leg_left_ankle_pitch_joint": -0.50, "leg_right_ankle_pitch_joint": -0.50,
}
#: Root height with the soles flat on the floor in this stance (MuJoCo crew
#: model), plus 1.5 mm so a reset never starts the feet inside the floor.
ROOT_Z = -0.0765
#: Where the root settles once standing: measured in Isaac, 21329076 (arm still,
#: 36 s; the soles seat 2.2 mm). The layout's heights are built on this, not on
#: the spawn height -- with 3 mm of contact clearance, 2 mm matters.
SETTLED_ROOT_Z = -0.0787
#: Settled leg torque / Kp, rad (``search.json`` ``feedforward_rad``).
FEEDFORWARD: dict[str, float] = {
    "leg_left_hip_roll_joint": 0.0034, "leg_left_hip_yaw_joint": 0.01323,
    "leg_left_hip_pitch_joint": 0.08841, "leg_left_knee_pitch_joint": -0.14822,
    "leg_left_ankle_pitch_joint": 0.16106, "leg_left_ankle_roll_joint": -0.02055,
    "leg_right_hip_roll_joint": 0.00351, "leg_right_hip_yaw_joint": -0.01185,
    "leg_right_hip_pitch_joint": 0.08219, "leg_right_knee_pitch_joint": -0.13758,
    "leg_right_ankle_pitch_joint": 0.15638, "leg_right_ankle_roll_joint": -0.00346,
}
#: (pitch kp, pitch kd, roll kp, roll kd): rad of ankle target per rad of tilt,
#: per rad/s of tilt rate.
GAINS = (1.5, 0.3, 2.0, 0.05)
#: How much higher the root stands than in the pinch squat the layout was built on.
ROOT_RISE = ROOT_Z - (-0.137)


def ankle_offsets(up_x: float, up_y: float, up_z: float, w_x: float, w_y: float,
                  heading: float, gains=GAINS):
    """Ankle (pitch, roll) target offsets from the body up axis and angular velocity (world frame).

    Rotated into the robot's heading frame first, where pitch is lean toward the
    robot's front and roll is lean toward its right -- the MuJoCo convention the
    gains were found in (robot facing +x). Plain arithmetic, so torch tensors
    work element-wise through ``math``-free callers; see ``ankle_offsets_torch``
    in the Isaac term.
    """
    kpp, kdp, kpr, kdr = gains
    c, s = math.cos(-heading), math.sin(-heading)
    ux, uy = c * up_x - s * up_y, s * up_x + c * up_y
    wx, wy = c * w_x - s * w_y, s * w_x + c * w_y
    pitch = math.atan2(ux, up_z)
    roll = math.atan2(-uy, up_z)
    return kpp * pitch + kdp * wy, kpr * roll - kdr * wx
