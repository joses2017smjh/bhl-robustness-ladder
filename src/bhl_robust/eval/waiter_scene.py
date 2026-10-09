"""Waiter phase 2 scene (docs/WAITER_INTERFACES.md): the 24-DoF Waiter robot, a pick table with two handled
tumblers and a handled tray, two delivery tables, and a SIMULATED head RGB camera.

Everything here is MuJoCo-side. The model is compiled ONCE (`build_scene`); a layout (`sample_layout(seed)`, a pure
function of the seed) is applied at reset by writing object poses, table positions and colours into the compiled
model/data (`apply_layout`), so a reset costs no recompilation.

Frames. The robot starts at the world origin facing +x, base origin at ground level (as `build_multi`). Table A (the
pick table) is in front of it; the delivery tables are to its left (+y) and right (-y), reached by backing off,
turning about +-90 deg and walking about 2 m.

Design (numbers, and why; all MODIFIABLE only before a scored run):

* Gripper geometry (from `mjcf_assets`): one finger, a 0.02 x 0.05 x 0.07 m box hinged at the hand tip (hand frame
  z = -0.11 m), closing 0-1.2 rad toward the palm side, 2 Nm. In the Waiter model the palm side is LATERAL (the
  left finger closes toward +y, the right toward -y, measured by FK). With the forearm level and pointing forward
  the hinge axis is vertical, so the finger is a vertical plate (0.05 tall, 0.02 thick) pointing forward. The grasp
  used by the expert is therefore: insert the OPEN finger forward through a vertical handle loop whose opening
  faces the robot, then close; the finger swings toward the palm side and wedges/clamps inside the loop.
* Reach (fixed-base IK over the 5 arm joints, `dev/probe_ik*.py`): with the hinge within ~10 deg of vertical the
  finger pivot reaches z ~ 0.57-0.68 m above the BASE ORIGIN (forearm level; it rises as the upper arm pitches
  forward), x ~ 0.20-0.36 m, and |y| >= 0.05 m on the hand's own side only (the right hand cannot cross the
  midline). Standing, the base origin is at z ~ -0.038 m (feet 0.039 m above it at the default leg pose), so the
  world band is ~0.53-0.64 m.
* Table heights: the finger enters a handle about 0.07 m above the table top (handle opening 0.015-0.105 m) and the
  lift adds ~0.05 m, so the hand works at table + 0.07 ... table + 0.125. Putting that band inside the world reach
  band gives TABLE_TOP = 0.49 m (pivot 0.56-0.615 m). The delivery tables have the same height, so setting the tray
  down is the lift in reverse.
* Tumbler: cylinder r 0.038 m, h 0.12 m, 0.20 kg, with a big D-handle on its local +y side: clear opening 0.050 m
  radial x 0.090 m vertical (bars 0.012 m square), opening axis = local x. The finger (0.02 thick, 0.05 tall) fits
  with 15 mm radial and 20 mm vertical clearance. The layout yaws each tumbler so its opening axis is near the
  right forearm's direction at that place (handles "set out for a right-handed server"), +-8 deg jitter.
* Tray: floor 0.20 (x) x 0.34 (y) x 0.01 m, rim 0.02 m, 0.25 kg in all, with a big ear loop at each end (+-y):
  clear opening 0.055 m (y) x 0.098 m (z), opening axis x, at the tray's mid-depth. Each hand's finger goes
  forward through its ear and closes outward (the palm side), so the two fingers clamp the tray in tension.
* Contacts: friction 1.0 sliding (rubber-on-plastic order, the robot's own default), torsional 0.005, rolling
  1e-4, condim 3 everywhere (the robot's default; torsional friction is not needed because the grasps are geometric).
  solref (0.005, 1) on the objects and tables: stiffer than MuJoCo's 0.02 default so a 0.2 kg cup does not sink
  into the tray, and still >= 2 x physics_dt (0.0005 s) as MuJoCo requires for stability. `multiccd` is enabled so
  cylinder-on-box contacts (tumbler on tray/table) get several contact points instead of one (a single CCD point
  lets a standing cylinder rock).
* Head camera `head_rgb`: SIMULATED (the hardware has no head camera), fixed to the torso top 0.06 m forward of the
  base axis at 0.84 m, pitched 40 deg down, fovy 80 deg, 224 x 224: it sees both hands, table A and, while walking,
  the delivery table from ~2 m.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from bhl_robust.eval.mjcf_assets import prepare_mjcf
from bhl_robust.eval.multi_robot import Slot, colorize_robots

PREFIX = "r0_"
HEAD_CAM = "head_rgb"
IMAGE_SIZE = 224
HEAD_CAM_POS = (0.06, 0.0, 0.84)
HEAD_CAM_PITCH_DEG = 40.0
HEAD_CAM_FOVY = 80.0

TABLE_TOP = 0.46        # coordinator 2026-10-05: lowered from 0.49 for the lid-handle tumbler (see _add_tumbler)
# Measured: with the legs at the deploy default pose the feet's lowest point is 0.039 m above the base origin, so the
# standing robot's base origin sits at z ~ -0.038 (not 0). Every hand height in the design notes is world z.
STAND_BASE_Z = -0.038
TABLE_THICK = 0.03
# Table A (pick table), world frame, fixed: x in [0.23, 0.71], y in [-0.62, 0.38]. The front edge is at 0.23 m because
# the arm cannot raise the hand above table height without sweeping it forward: with the gripper closed the hand's
# points below table height stay at x <= 0.21 m on the tuck path (FK sweep, dev/t_path.py), so 0.23 m clears it.
TABLE_A_CENTER = (0.42, 0.0)          # coordinator: centred, near edge at x 0.18 (tumblers in front of the tray)
TABLE_A_HALF = (0.24, 0.50)
# Delivery tables: depth (along the approach) x width; centre jittered by the layout.
DELIVERY_HALF = (0.22, 0.40)          # (along the approach, across it)
DELIVERY_NOMINAL = {"left": (-0.30, 2.45), "right": (-0.30, -2.45)}
DELIVERY_YAW = {"left": math.pi / 2, "right": -math.pi / 2}   # table frame x = direction the robot walks
# Target zone (visual mat on each delivery table), in the table frame: centre and half sizes.
ZONE_CENTER = (-0.07, 0.0)            # 0.15 m in from the near edge (near edge at -0.22)
ZONE_HALF = (0.11, 0.15)

# Tumbler (local frame: origin at the bottom centre, z up, handle on +y, opening axis x).
CUP_R = 0.038
CUP_H = 0.10
CUP_MASS = 0.17
CUP_BASE_MASS = 0.10                  # weighted base disc (2026-10-09: lowers the CoM so the tumbler rides out the
                                      # carried tray's rocking; 0.12 kg made the old lift too slow: dev2 env8)
CUP_BASE_T = 0.01
HANDLE_T = 0.012                      # bar thickness (square section)
HANDLE_GAP = 0.050                    # clear radial opening
HANDLE_Z0, HANDLE_Z1 = 0.015, 0.105   # (unused since the lid handle)
BAIL_GAP = 0.065                      # lid handle: clear height between the lid and the handle's top bar
BAIL_HALF_SPAN = 0.050                # lid handle: post centres at +-0.050 (outside the 0.038 body, on brackets)
HANDLE_BAR_MASS = 0.01                # each of 3 bars
# Tray (local frame: origin at the floor's bottom centre, x = depth, y = length).
TRAY_HALF = (0.13, 0.13)              # 2026-10-08: 0.17 -> 0.13 with the handle tunnels (worst-case clearance between a
                                      # tunnel wall and the other tumbler 1.6 cm over the layout jitter)
TRAY_FLOOR_T = 0.01
TRAY_RIM_H = 0.012                    # coordinator: lowered from 0.02 (the lifted tumbler clears it by ~2 cm)
#
TRAY_RIM_T = 0.008
TRAY_FRONT_MASS = 0.13                # front rim ballast (kg; see _add_tray)
# Tray handles = two HANDLE LOOPS (2026-10-08, after loops, tunnels and pegs: dev2 env7-close2): at each end of the tray,
# 2 cm ahead of its mid-depth, two posts EAR_GAP apart joined by a top bar along y, EAR_TOP above the tray bottom,
# open below, on a base bar outside the floor's end. The OPEN finger slides in from behind under the bar, between the
# posts, and lifts: the mechanism of the tumbler's lid handle. (A closing finger on a peg shoved the light tray 5 cm
# and tilted it 13 deg instead of wrapping the stem: dev2 close2.)
EAR_DX = 0.02                         # loop plane x offset from the tray centre (tray frame)
EAR_OUT = 0.07                        # loop centre this far outside the floor's end (|y| 0.20)
EAR_GAP = 0.09                        # clear width between the posts (y)
EAR_TOP = 0.20                        # top bar's underside above the tray bottom (hook point ~0.625 m: in reach)
EAR_T = 0.012                         # post / bar thickness
EAR_DEPTH = EAR_T
STEM_HALF = EAR_T / 2.0               # (kept for older tools)
# The loop's top bar is a FLAT bar, 4 cm deep (along x), set 0.8 cm ahead of the loop plane (2026-10-09): on a 1.2 cm
# square bar the hooked tray pivoted freely on the two fingers and swung like a ~1.1 Hz pendulum, near the gait's
# 1.25 Hz stride (23-32 deg while walking; the tumbler tipped over in 2/15 walks, dev2 P2); the finger's 7 cm top
# face now carries the bar over 4 cm, which resists pitching. Its back face stays >= 1 cm ahead of the carry path.
CAP_DX = 0.008                        # top bar centre ahead of the loop plane (tray x)
CAP_HALF = (0.02, EAR_GAP / 2.0 + EAR_T, EAR_T / 2.0)   # the top bar's half sizes (older tools read CAP_HALF)

COLOURS = {"red": (0.85, 0.12, 0.10, 1.0), "blue": (0.12, 0.30, 0.85, 1.0)}
SIDES = ("left", "right")

OBJ_FRICTION = (1.0, 0.005, 0.0001)
OBJ_SOLREF = (0.005, 1.0)
OBJ_CONDIM = 3


def ear_center_y() -> float:
    """|y| of the tray handle loop's centre, tray frame (its x is EAR_DX)."""
    return TRAY_HALF[1] + EAR_OUT


def handle_center_local() -> np.ndarray:
    """Centre of the tumbler LID handle's clear opening (under its top bar), tumbler frame. The finger enters along
    the tumbler's local x (the layout yaws it to the right forearm's heading) and the bar runs along local y."""
    return np.array([0.0, 0.0, CUP_H + 0.5 * BAIL_GAP])


# ----------------------------------------------------------------------------------------------------------- layout

@dataclass(frozen=True)
class Layout:
    """Everything that varies between episodes; `sample_layout(seed)` is the only constructor used by the env."""
    seed: int
    target_colour: str
    target_side: str
    instruction_template: int
    tumblers: dict = field(default_factory=dict)     # colour -> (x, y, yaw) on table A
    tray: tuple = (0.30, 0.0, 0.0)                   # (x, y, yaw) on table A
    delivery: dict = field(default_factory=dict)     # side -> (x, y) table centre

    def as_dict(self) -> dict:
        return {"seed": self.seed, "target_colour": self.target_colour, "target_side": self.target_side,
                "instruction_template": self.instruction_template,
                "tumblers": {k: list(v) for k, v in self.tumblers.items()}, "tray": list(self.tray),
                "delivery": {k: list(v) for k, v in self.delivery.items()}}


# Tumbler slots on table A (right of the tray, reachable by the right hand). Near slot: the handle opening at
# about y -0.21; far slot: about y -0.36. Yaw = right-forearm heading there (fit from the IK map: ~0 deg at pivot
# y -0.20, about -2.4 deg per cm further right).
# coordinator 2026-10-05: one tumbler on EACH side of the tray (the right hand picks the right one, the left hand
# the left one). With both on the right, carrying the far one to the tray dragged it through the near one
# (dev t_bail.py: 4/5 failures); a hand cannot lift high enough to clear the other tumbler's handle.
# Revised again: beside the tray they spawned INSIDE its 14 cm end handles (ears reach |y| 0.249) and a carried
# tumbler cannot clear them. In FRONT of the tray (nearer the robot), one per side: each is carried forward
# over the tray's low front rim onto its own half, a path that never crosses an ear.
# v4 (coordinator, after dev t_bail.py renders): BESIDE the tray, in front of its ears. The ears are thin loops at
# the tray's mid-depth (x extent +-0.006), so a tumbler carried sideways at x 0.31 passes 3.6 cm in front of
# them; x 0.31 keeps the approach start (0.11 m behind the tumbler) inside the reach band (pivot x >= 0.20).
CUP_SLOTS = ({"x": 0.30, "y": -0.33}, {"x": 0.30, "y": 0.33})   # pick at x 0.30; at |y| 0.37 (layout v5, 2026-10-08)
                                      # the forearm meets the handle ~40 deg off its axis and pushed the tumbler
TRAY_NOMINAL = (0.36, 0.0)   # pegs at (0.38, +-0.175): behind them (x <= 0.343) the tumblers are carried sideways


def forearm_heading(pivot_y: float) -> float:
    """Approximate right-forearm heading (rad) with the hinge vertical at pivot y (IK-map fit, x ~ 0.24 m)."""
    return math.radians(-2.4 * (abs(pivot_y) - 0.20) * 100.0)


def sample_layout(seed: int, n_templates: int = 6) -> Layout:
    """Pure function of `seed` (numpy PCG64 stream seeded by the integer)."""
    rng = np.random.default_rng(int(seed))
    target_colour = ("red", "blue")[int(rng.integers(2))]
    target_side = SIDES[int(rng.integers(2))]
    template = int(rng.integers(n_templates))
    swap = bool(rng.integers(2))                      # which colour stands in the near slot
    tumblers = {}
    for k, colour in enumerate(("red", "blue") if not swap else ("blue", "red")):
        s = CUP_SLOTS[k]
        x = s["x"] + rng.uniform(-0.005, 0.005)
        y = s["y"] + rng.uniform(-0.008, 0.008)
        # lid handle: the finger passes under the bar through the tumbler's axis, so the pivot y is the tumbler's y;
        # the left-side tumbler takes the mirrored (left-forearm) heading
        # the IK (hinge vertical) points the forearm almost straight forward at both slots (measured: heading
        # within ~6 deg of +x, dev t_bail.py VERB), so the handle opening faces +x with +-6 deg jitter
        yaw = math.radians(rng.uniform(-6.0, 6.0))
        tumblers[colour] = (float(x), float(y), float(yaw))
    tray = (TRAY_NOMINAL[0] + rng.uniform(-0.005, 0.005), TRAY_NOMINAL[1] + rng.uniform(-0.005, 0.005),
            math.radians(rng.uniform(-2.0, 2.0)))
    delivery = {}
    for side in SIDES:
        nx, ny = DELIVERY_NOMINAL[side]
        delivery[side] = (float(nx + rng.uniform(-0.06, 0.06)), float(ny + math.copysign(rng.uniform(-0.15, 0.15), ny)))
    return Layout(seed=int(seed), target_colour=target_colour, target_side=target_side,
                  instruction_template=template, tumblers=tumblers, tray=tuple(float(v) for v in tray),
                  delivery=delivery)


# ------------------------------------------------------------------------------------------------------------ build

def _quat_z(yaw: float) -> list[float]:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def _box(body, name, pos, half, rgba, mass=None, contact=True):
    g = body.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = list(half)
    g.pos = list(pos)
    g.rgba = list(rgba)
    if mass is not None:
        g.mass = mass
    if contact:
        g.friction = list(OBJ_FRICTION)
        g.solref = list(OBJ_SOLREF)
        g.condim = OBJ_CONDIM
    else:
        g.contype = 0
        g.conaffinity = 0
    return g


def _add_table(spec, name, half_xy, rgba=(0.62, 0.48, 0.34, 1.0)):
    b = spec.worldbody.add_body()
    b.name = name
    hx, hy = half_xy
    _box(b, f"{name}_top", (0.0, 0.0, TABLE_TOP - TABLE_THICK / 2.0), (hx, hy, TABLE_THICK / 2.0), rgba)
    leg_h = (TABLE_TOP - TABLE_THICK) / 2.0
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        _box(b, f"{name}_leg{i}", (sx * (hx - 0.03), sy * (hy - 0.03), leg_h), (0.02, 0.02, leg_h),
             (0.30, 0.24, 0.18, 1.0))
    return b


def _add_tumbler(spec, colour):
    b = spec.worldbody.add_body()
    b.name = f"tumbler_{colour}"
    b.add_freejoint().name = f"tumbler_{colour}_free"
    rgba = COLOURS[colour]
    g = b.add_geom()
    g.name = f"tumbler_{colour}_body"
    g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    g.size = [CUP_R, CUP_H / 2.0, 0.0]
    g.pos = [0.0, 0.0, CUP_H / 2.0]
    g.rgba = list(rgba)
    g.mass = CUP_MASS
    g.friction = list(OBJ_FRICTION)
    g.solref = list(OBJ_SOLREF)
    g.condim = OBJ_CONDIM
    if CUP_BASE_MASS > 0:                      # weighted base (inside the wall cylinder; no contact of its own)
        gb = b.add_geom()
        gb.name = f"tumbler_{colour}_base"
        gb.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        gb.size = [CUP_R - 0.003, CUP_BASE_T / 2.0, 0.0]
        gb.pos = [0.0, 0.0, CUP_BASE_T / 2.0 + 0.0005]
        gb.rgba = list(rgba)
        gb.mass = CUP_BASE_MASS
        gb.contype = 0
        gb.conaffinity = 0
    # LID CARRY HANDLE (coordinator, 2026-10-05; replaces the agent's side D-handle). The modelled gripper is a
    # one-finger L-hook (FK: closing swings the finger sideways, it never presses on the palm), so a side handle
    # hangs the tumbler off-centre: it tilted 40-50 deg and slid off the finger in 4/4 dev carries
    # (campaign-waiter/phase2/dev/t_place.py). A handle arching over the lid hangs it from above its axis: two posts
    # at local +-y and a top bar along y; the finger passes under the bar along local x.
    t = HANDLE_T
    dark = tuple(0.7 * c for c in rgba[:3]) + (1.0,)
    # WIDE handle (coordinator, 2026-10-06): the posts stand OUTSIDE the tumbler body on short brackets (inner faces
    # at +-(BAIL_HALF_SPAN - t/2) = +-0.044 m instead of +-0.032). The hanging tumbler spins ~60 deg about vertical
    # during the carry, so the finger leaves at an angle and grazed a post with the narrow handle (dev seeds 1002,
    # 1016); the reach band caps the withdrawal at ~6 cm, so the clearance has to come from the handle.
    post_y = BAIL_HALF_SPAN
    post_half_z = (BAIL_GAP + t) / 2.0
    for k, sy in enumerate((1.0, -1.0)):
        _box(b, f"tumbler_{colour}_post{k}", (0.0, sy * post_y, CUP_H + post_half_z), (t / 2.0, t / 2.0, post_half_z),
             dark, mass=HANDLE_BAR_MASS)
        y0, y1 = CUP_R - 0.006, post_y + t / 2.0
        _box(b, f"tumbler_{colour}_bracket{k}", (0.0, sy * 0.5 * (y0 + y1), CUP_H - t / 2.0),
             (t / 2.0, 0.5 * (y1 - y0), t / 2.0), dark, mass=HANDLE_BAR_MASS)
    _box(b, f"tumbler_{colour}_bar_top", (0.0, 0.0, CUP_H + BAIL_GAP + t / 2.0), (t / 2.0, post_y + t / 2.0, t / 2.0),
         dark, mass=HANDLE_BAR_MASS)
    return b


def _add_tray(spec):
    b = spec.worldbody.add_body()
    b.name = "tray"
    b.add_freejoint().name = "tray_free"
    hx, hy = TRAY_HALF
    col = (0.80, 0.80, 0.78, 1.0)
    rim_col = (0.55, 0.55, 0.55, 1.0)
    _box(b, "tray_floor", (0.0, 0.0, TRAY_FLOOR_T / 2.0), (hx, hy, TRAY_FLOOR_T / 2.0), col, mass=0.13)
    zr = TRAY_FLOOR_T + TRAY_RIM_H / 2.0
    rt = TRAY_RIM_T / 2.0
    # the front rim is a ballast (TRAY_FRONT_MASS): the tray hangs from its handle bars 2 cm ahead of its centre and the
    # tumbler sits ~7 cm behind them; with a light rim the lifted tray pitched 14-19 deg (dev2 fixT3/fix2)
    _box(b, "tray_rim_front", (hx - rt, 0.0, zr), (rt, hy, TRAY_RIM_H / 2.0), rim_col, mass=TRAY_FRONT_MASS)
    _box(b, "tray_rim_back", (-hx + rt, 0.0, zr), (rt, hy, TRAY_RIM_H / 2.0), rim_col, mass=0.015)
    _box(b, "tray_rim_left", (0.0, hy - rt, zr), (hx, rt, TRAY_RIM_H / 2.0), rim_col, mass=0.01)
    _box(b, "tray_rim_right", (0.0, -hy + rt, zr), (hx, rt, TRAY_RIM_H / 2.0), rim_col, mass=0.01)
    t = EAR_T
    for side, s in (("left", 1.0), ("right", -1.0)):
        yc = s * (hy + EAR_OUT)
        y_in, y_out = yc - s * (EAR_GAP / 2.0 + t / 2.0), yc + s * (EAR_GAP / 2.0 + t / 2.0)
        span = abs(y_out - (s * hy)) / 2.0
        _box(b, f"tray_ear_{side}_base", (EAR_DX, s * hy + s * span, TRAY_FLOOR_T / 2.0),
             (t / 2.0, span + t / 2.0, TRAY_FLOOR_T / 2.0), rim_col, mass=0.008)
        for k, yp in enumerate((y_in, y_out)):
            _box(b, f"tray_ear_{side}_post{k}", (EAR_DX, yp, (EAR_TOP + t) / 2.0), (t / 2.0, t / 2.0, (EAR_TOP + t) / 2.0),
                 rim_col, mass=0.008)
        _box(b, f"tray_ear_{side}_bar", (EAR_DX + CAP_DX, yc, EAR_TOP + t / 2.0), CAP_HALF, rim_col, mass=0.008)
    return b


def _add_zone(body, name):
    """Visual-only target mat on a delivery table (no contact)."""
    _box(body, name, (ZONE_CENTER[0], ZONE_CENTER[1], TABLE_TOP + 0.0006), (ZONE_HALF[0], ZONE_HALF[1], 0.0005),
         (0.20, 0.70, 0.30, 1.0), contact=False)


_WORLD = """<mujoco model="waiter-phase2">
  <compiler angle="radian"/>
  <option timestep="0.0005"><flag multiccd="enable"/></option>
  <statistic center="0.3 0 0.5" extent="3.0"/>
  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global azimuth="150" elevation="-25" offwidth="1280" offheight="960"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.60 0.66" rgb2="0.15 0.16 0.18" width="512" height="3072"/>
    <texture type="2d" name="floortex" builtin="checker" mark="edge" rgb1="0.74 0.74 0.72" rgb2="0.62 0.62 0.60"
             markrgb="0.45 0.45 0.44" width="300" height="300"/>
    <material name="floormat" texture="floortex" texuniform="true" texrepeat="6 6" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0.5 0 3.0" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="floormat"/>
  </worldbody>
</mujoco>
"""


@dataclass
class SceneIds:
    """Resolved ids of everything the env, the expert and the tests read."""
    slot: Slot
    base_body: int
    cam: int
    weld: int
    anchor_mocap: int
    tumbler_body: dict
    tumbler_qadr: dict
    tumbler_vadr: dict
    tumbler_geoms: dict
    tray_body: int
    tray_qadr: int
    tray_vadr: int
    tray_geoms: list
    tray_floor_geom: int
    table_body: dict
    table_top_geom: dict
    robot_geoms: np.ndarray
    finger_geom: dict
    palm_geom: dict
    finger_body: dict
    hand_body: dict
    arm_qadr: dict
    arm_dadr: dict
    arm_range: dict
    gripper_qadr: dict


def build_scene(upstream: Path, cache_dir: Path):
    """Compile the scene once. Returns (model, SceneIds). The base weld (`base_weld`, robot base to the mocap body
    `base_anchor`, identity relative pose) is compiled in and inactive; the env activates it for `fixed_base=True`
    and puts the anchor where the base should be held."""
    scene = prepare_mjcf(Path(upstream), Path(cache_dir) / "waiter", "waiter")
    robot_xml = scene.parent / "berkeley_humanoid_lite.xml"
    spec = mujoco.MjSpec.from_string(_WORLD)
    child = mujoco.MjSpec.from_file(str(robot_xml))
    frame = spec.worldbody.add_frame()
    frame.pos = [0.0, 0.0, 0.0]
    frame.attach_body(child.bodies[1], PREFIX, "")
    base = spec.body(f"{PREFIX}base")
    cam = base.add_camera()
    cam.name = HEAD_CAM
    cam.pos = list(HEAD_CAM_POS)
    th = math.radians(HEAD_CAM_PITCH_DEG)
    # MuJoCo cameras look down their -z with +y up: x = robot -y (image right), y = up tilted forward.
    xa = np.array([0.0, -1.0, 0.0])
    ya = np.array([math.sin(th), 0.0, math.cos(th)])
    za = np.cross(xa, ya)
    R = np.stack([xa, ya, za], axis=1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    cam.quat = list(q)
    cam.fovy = HEAD_CAM_FOVY
    _add_table(spec, "table_A", TABLE_A_HALF)
    for side in SIDES:
        t = _add_table(spec, f"table_{side}", DELIVERY_HALF)
        _add_zone(t, f"zone_{side}")
    for colour in COLOURS:
        _add_tumbler(spec, colour)
    _add_tray(spec)
    anchor = spec.worldbody.add_body()      # mocap target of the fixed-base weld (pose set by the env)
    anchor.name = "base_anchor"
    anchor.mocap = True
    eq = spec.add_equality()
    eq.name = "base_weld"
    eq.type = mujoco.mjtEq.mjEQ_WELD
    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq.name1 = f"{PREFIX}base"
    eq.name2 = "base_anchor"
    eq.solref = [0.002, 1.0]
    eq.active = False
    model = spec.compile()
    colorize_robots(model, 1, hero=0)
    return model, resolve_ids(model)


def resolve_ids(model) -> SceneIds:
    def oid(kind, name):
        i = mujoco.mj_name2id(model, kind, name)
        if i < 0:
            raise KeyError(name)
        return i

    def bid(name):
        return oid(mujoco.mjtObj.mjOBJ_BODY, name)

    def gid(name):
        return oid(mujoco.mjtObj.mjOBJ_GEOM, name)

    def sid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)

    p = PREFIX
    act = [j for j in range(model.nu)
           if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) or "").startswith(p)]
    jpos, jvel = [], []
    for j in act:
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
        base = jn[len(p):].replace("_joint", "")
        a_p, a_v = sid(f"{p}{base}_pos"), sid(f"{p}{base}_vel")
        if a_p < 0 or a_v < 0:
            raise RuntimeError(f"cannot resolve sensors for {jn}")
        jpos.append(model.sensor_adr[a_p])
        jvel.append(model.sensor_adr[a_v])
    base_id = bid(f"{p}base")
    jid = model.body_jntadr[base_id]
    slot = Slot(prefix=p, label="waiter", ctrl=np.asarray(act, dtype=int), jpos_adr=np.asarray(jpos, dtype=int),
                jvel_adr=np.asarray(jvel, dtype=int), quat_adr=int(model.sensor_adr[sid(f"{p}imu_quat")]),
                gyro_adr=int(model.sensor_adr[sid(f"{p}imu_gyro")]), qpos_adr=int(model.jnt_qposadr[jid]),
                qvel_adr=int(model.jnt_dofadr[jid]), body_id=int(base_id))
    tb, tq, tv, tg = {}, {}, {}, {}
    for colour in COLOURS:
        b = bid(f"tumbler_{colour}")
        tb[colour] = b
        tq[colour] = int(model.jnt_qposadr[model.body_jntadr[b]])
        tv[colour] = int(model.jnt_dofadr[model.body_jntadr[b]])
        tg[colour] = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b]
    tray = bid("tray")
    robot_geoms = np.array([(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "")
                            .startswith(p) for g in range(model.ngeom)])
    arm_names = {"left": ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_roll"],
                 "right": ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_roll"]}
    aq, ad, ar, gq = {}, {}, {}, {}
    for side, names in arm_names.items():
        js = [oid(mujoco.mjtObj.mjOBJ_JOINT, f"{p}arm_{side}_{n}_joint") for n in names]
        aq[side] = np.array([model.jnt_qposadr[j] for j in js])
        ad[side] = np.array([model.jnt_dofadr[j] for j in js])
        ar[side] = np.array([model.jnt_range[j] for j in js])
        gq[side] = int(model.jnt_qposadr[oid(mujoco.mjtObj.mjOBJ_JOINT, f"{p}arm_{side}_gripper_joint")])
    return SceneIds(
        slot=slot, base_body=base_id, cam=oid(mujoco.mjtObj.mjOBJ_CAMERA, HEAD_CAM),
        weld=oid(mujoco.mjtObj.mjOBJ_EQUALITY, "base_weld"),
        anchor_mocap=int(model.body_mocapid[bid("base_anchor")]),
        tumbler_body=tb, tumbler_qadr=tq, tumbler_vadr=tv, tumbler_geoms=tg,
        tray_body=tray, tray_qadr=int(model.jnt_qposadr[model.body_jntadr[tray]]),
        tray_vadr=int(model.jnt_dofadr[model.body_jntadr[tray]]),
        tray_geoms=[g for g in range(model.ngeom) if model.geom_bodyid[g] == tray], tray_floor_geom=gid("tray_floor"),
        table_body={k: bid(f"table_{k}") for k in ("A", "left", "right")},
        table_top_geom={k: gid(f"table_{k}_top") for k in ("A", "left", "right")},
        robot_geoms=robot_geoms,
        finger_geom={s: gid(f"{p}arm_{s}_finger_geom") for s in SIDES},
        palm_geom={s: gid(f"{p}arm_{s}_palm_collision") for s in SIDES},
        finger_body={s: bid(f"{p}arm_{s}_finger_link") for s in SIDES},
        hand_body={s: bid(f"{p}arm_{s}_hand_link") for s in SIDES},
        arm_qadr=aq, arm_dadr=ad, arm_range=ar, gripper_qadr=gq)


def apply_layout(model, data, ids: SceneIds, layout: Layout) -> None:
    """Write the layout's static poses into `model` (tables) and object poses into `data` (free joints).
    Call before mj_forward. Robot state is the caller's."""
    model.body_pos[ids.table_body["A"]] = [TABLE_A_CENTER[0], TABLE_A_CENTER[1], 0.0]
    model.body_quat[ids.table_body["A"]] = [1.0, 0.0, 0.0, 0.0]
    for side in SIDES:
        x, y = layout.delivery[side]
        model.body_pos[ids.table_body[side]] = [x, y, 0.0]
        model.body_quat[ids.table_body[side]] = _quat_z(DELIVERY_YAW[side])
    for colour, (x, y, yaw) in layout.tumblers.items():
        a = ids.tumbler_qadr[colour]
        data.qpos[a:a + 3] = [x, y, TABLE_TOP + 0.0005]
        data.qpos[a + 3:a + 7] = _quat_z(yaw)
        v = ids.tumbler_vadr[colour]
        data.qvel[v:v + 6] = 0.0
    x, y, yaw = layout.tray
    a = ids.tray_qadr
    data.qpos[a:a + 3] = [x, y, TABLE_TOP + 0.0005]
    data.qpos[a + 3:a + 7] = _quat_z(yaw)
    data.qvel[ids.tray_vadr:ids.tray_vadr + 6] = 0.0


# --------------------------------------------------------------------------------------------------- geometry reads

def body_pose(data, body: int):
    """(position, 3x3 rotation) of a body in the world."""
    return data.xpos[body].copy(), data.xmat[body].reshape(3, 3).copy()


def tilt_of(R: np.ndarray) -> float:
    """Angle between a body's +z and world +z (rad)."""
    return float(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))


def yaw_of_quat(q) -> float:
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def zone_world(data, ids: SceneIds, side: str):
    """(centre xy, yaw) of a delivery table's target zone in the world."""
    p, R = body_pose(data, ids.table_body[side])
    c = p[:2] + R[:2, :2] @ np.asarray(ZONE_CENTER)
    return c, math.atan2(R[1, 0], R[0, 0])
