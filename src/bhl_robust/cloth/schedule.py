"""One sweep as a joint schedule -- the Isaac action term's plan, checkable offline.

The Isaac sweep action used to latch a hand-tuned 3-joint mapping every 40 ms,
never read where the garment was, and advanced its timer by the env step inside
a physics-substep loop, so its "1.5 s sweep" re-started before it began. This
builds the whole primitive once per RL step instead:

    pinch hold -> over an anchor -> descend to contact
      -> glide at contact height, routed around the garment, to the sweep start
      -> sweep -> glide, routed around where the garment should now be, to an
      anchor -> lift -> pinch hold

**Anchors** are the cells where the fingertip table has the *same* arm branch at
contact and hover height. The arm cannot lift the hand above most of the table
-- over its interior the elbow is already fully bent -- so a sweep cannot simply
descend at its start (``scripts/cloth/build_tip_table.py``: 78 of 219 contact
cells can hover).

**Routes** run over contact cells and keep the hand's footprint
(``layout.HAND_RADIUS``) ``KEEP_OUT`` clear of the garment's rectangle: A* on
the 2 cm grid, then shortened wherever a straight glide stays clear.

**Every candidate is replayed through forward kinematics of the hand hull**
(``arm_fk``) at ``CHECK_DT`` over the joint-space interpolation the arm will
actually be commanded. It is refused if the hull enters the garment before the
sweep, enters where the garment is expected after it, or goes into the table.
The first version checked cells and missed what happens between them; a MuJoCo
replay of 21300603's schedule had the fingertip swing 7 cm into the garment
during a 0.2 s "vertical" descent.

**Timing and feedforward.** The arm is Isaac's implicit PD at 10 N m/rad,
2 N m s/rad, 4 N m (``ARM_*``). Played as bare position targets, a schedule lags
by Kd/Kp = 0.2 s: 21307211's trace has the fingertip 65-72 mm behind its target,
and the hand cut through the shirt on its way to the sweep. So each phase is
played with a quintic time scaling (at rest at both ends, peak speed as the
constants say), and the schedule carries what the drive needs to follow it: a
velocity target, and a position target offset by the inverse-dynamics torque
over Kp (``arm_fk.inverse_dynamics``). The drive's torque is then that
feedforward plus PD on the residual, still clipped at 4 N m by the simulator. A
phase whose feedforward would need more than ``TORQUE_BUDGET`` of the limit is
slowed until it does not. Isaac-free on purpose: the login-node tests check it,
and a MuJoCo arm that reproduces Isaac's tracking to 1e-3 rad checks the rest.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from bhl_robust.cloth.arm_fk import ArmChain, hull_points, inverse_dynamics, load_chain
from bhl_robust.cloth.controller import PINCH_JOINT_POS, is_plan_valid
from bhl_robust.cloth.garments import GarmentSpec
from bhl_robust.cloth.kinematics import is_valid_joints
from bhl_robust.cloth.layout import HAND_RADIUS, ROBOT_XY, ROBOT_YAW, TABLE_TOP_Z, table_top_rect
from bhl_robust.cloth.reach import ContactTable, load_contact
from bhl_robust.cloth.sweep import SweepConfig, decode_action, plan_sweep

#: One RL step is one sweep, start to finish, inside this many seconds. Was 4.0;
#: phases that start and stop at rest take 1.875x as long at the same peak speed.
MACRO_STEP_S = 6.0
SETTLE_S = 0.30          # left at the end of the macro step for the arm to arrive
MOVE_S = 0.40            # least time for pinch hold <-> over an anchor, before smoothing
DESCEND_S = 0.25         # least time for hover <-> contact at an anchor, before smoothing
MAX_JOINT_SPEED = 3.0    # rad/s, peak, any joint outside the sweep
GLIDE_SPEED = 0.25       # m/s, peak, at contact height to and from the sweep
MAX_SWEEP_S = 1.50       # a slow sweep is sped up to fit the macro step
KNOT_SPACING = 0.01      # m between table lookups along a path
CHECK_DT = 0.01          # s between forward-kinematics checks
KEEP_OUT = 0.006         # m between the hand's footprint and the garment on a glide
TABLE_TOL = 0.002        # m the hull may sit below the top (blend error is < 1 mm)
N_ANCHORS = 12
#: The arm actuator schedules are played through: the upstream HUMANOID_LITE_CFG
#: "arms" group, an implicit PD. The feedforward is computed against these, and
#: ``cloth_sort_mdp.SweepAction`` refuses to start if the live cfg differs.
ARM_KP, ARM_KD, ARM_EFFORT, ARM_ARMATURE = 10.0, 2.0, 4.0, 0.002
TORQUE_BUDGET = 0.85     # fraction of ARM_EFFORT a schedule's feedforward may ask for
QUINTIC_PEAK = 1.875     # peak over mean speed of 10t^3 - 15t^4 + 6t^5
ACC_SMOOTH_S = 0.015     # s, Gaussian smoothing of the numerical acceleration


@dataclass
class Schedule:
    q: np.ndarray                  # (n_steps, 5) right-arm joint targets
    joints: tuple[str, ...]
    valid: bool
    reason: str
    start_xy: np.ndarray | None = None
    end_xy: np.ndarray | None = None
    duration: float = 0.0
    anchor_in: np.ndarray | None = None
    anchor_out: np.ndarray | None = None
    route_in: list = field(default_factory=list)
    route_out: list = field(default_factory=list)
    sweep_window: tuple[float, float] | None = None
    #: What the drive is sent: position target with the feedforward folded in,
    #: and velocity target. ``q`` is where the hand should be; checks use ``q``.
    q_cmd: np.ndarray | None = None
    qd: np.ndarray | None = None
    #: Largest feedforward torque the schedule asks of any arm joint (N m).
    tau_peak: float = 0.0


def pinch_arm(joints: tuple[str, ...]) -> np.ndarray:
    return np.array([PINCH_JOINT_POS.get(j, 0.0) for j in joints], dtype=float)


# ------------------------------------------------------------------ geometry

def _robot_xy(p_world: np.ndarray) -> np.ndarray:
    """World (N, 2) -> robot-frame (N, 2), for the layout's robot pose."""
    c, s = np.cos(-ROBOT_YAW), np.sin(-ROBOT_YAW)
    d = np.atleast_2d(p_world) - np.asarray(ROBOT_XY)
    return np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], axis=1)


def _world_xyz(p_robot: np.ndarray) -> np.ndarray:
    """Robot-frame (..., 3) -> world (..., 3); z unchanged."""
    c, s = np.cos(ROBOT_YAW), np.sin(ROBOT_YAW)
    out = np.array(p_robot, dtype=float, copy=True)
    out[..., 0] = ROBOT_XY[0] + c * p_robot[..., 0] - s * p_robot[..., 1]
    out[..., 1] = ROBOT_XY[1] + s * p_robot[..., 0] + c * p_robot[..., 1]
    return out


@dataclass(frozen=True)
class Rect:
    """A garment's footprint: centre, yaw and half extents, world frame."""

    centre: np.ndarray
    yaw: float
    half: np.ndarray

    def local(self, p: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        d = np.atleast_2d(p)[..., :2] - self.centre
        return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], axis=-1)

    def distance(self, p: np.ndarray) -> np.ndarray:
        """Distance from points to the rectangle; negative inside."""
        q = np.abs(self.local(p)) - self.half
        return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)

    def extent_along(self, d: np.ndarray) -> float:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        u, v = np.array([c, s]), np.array([-s, c])
        return float(abs(d @ u) * self.half[0] + abs(d @ v) * self.half[1])


def _on_contact(pts_world: np.ndarray, t: ContactTable) -> np.ndarray:
    """Can each world point be blended from valid contact cells only? ``(N,)``.

    The ``q_at(strict=True)`` test, vectorised: every grid corner that carries
    bilinear weight must be a contact cell, which keeps the fingertip within
    1.3 mm of the point instead of up to 1.3 cm at the region's edge.
    """
    r = _robot_xy(pts_world)
    fx, fy = (r[:, 0] - t.x[0]) / t.step, (r[:, 1] - t.y[0]) / t.step
    i0, j0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    ax, ay = fx - i0, fy - j0
    ok = np.ones(len(r), dtype=bool)
    nx, ny = t.contact_mask.shape
    for di, wx in ((0, 1.0 - ax), (1, ax)):
        for dj, wy in ((0, 1.0 - ay), (1, ay)):
            i, j = i0 + di, j0 + dj
            inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
            valid = np.zeros(len(r), dtype=bool)
            valid[inside] = t.contact_mask[i[inside], j[inside]]
            ok &= (wx * wy <= 1e-9) | valid
    return ok


def _glide_ok(a, b, t: ContactTable, keep: Rect | None, radius: float) -> bool:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / 0.005)) + 1)
    pts = a[None] + (b - a)[None] * np.linspace(0.0, 1.0, n)[:, None]
    if not _on_contact(pts, t).all():
        return False
    return keep is None or bool((keep.distance(pts) >= radius).all())


_CELLS: dict[int, tuple[ContactTable, np.ndarray]] = {}


def _cell_world(t: ContactTable) -> np.ndarray:
    """World xy of every grid cell, ``(nx, ny, 2)``. Cached per table object."""
    hit = _CELLS.get(id(t))
    if hit is None or hit[0] is not t:
        X, Y = np.meshgrid(t.x, t.y, indexing="ij")
        hit = (t, _world_xyz(np.stack([X, Y, np.zeros_like(X)], axis=-1))[..., :2])
        _CELLS[id(t)] = hit
    return hit[1]


def _route(src, dst, t: ContactTable, keep: Rect | None, radius: float) -> list[np.ndarray] | None:
    """Shortest glide from ``src`` to ``dst`` over contact cells, clear of ``keep``."""
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    if _glide_ok(src, dst, t, keep, radius):
        return [src, dst]
    cells = _cell_world(t)
    free = t.contact_mask.copy()
    if keep is not None:
        free &= keep.distance(cells.reshape(-1, 2)).reshape(free.shape) >= radius
    nx, ny = free.shape

    def attach(p) -> list[tuple[int, int]]:
        d = np.linalg.norm(cells - p, axis=-1)
        d[~free] = np.inf
        order = np.argsort(d, axis=None)[:16]
        return [(int(k // ny), int(k % ny)) for k in order
                if np.isfinite(d.flat[k]) and _glide_ok(p, cells.reshape(-1, 2)[k], t, keep, radius)]

    starts, goals = attach(src), set(attach(dst))
    if not starts or not goals:
        return None
    dist = {c: float(np.linalg.norm(cells[c] - src)) for c in starts}
    prev: dict = {}
    heap = [(dist[c] + float(np.linalg.norm(cells[c] - dst)), c) for c in starts]
    heapq.heapify(heap)
    done: set = set()
    goal = None
    while heap:
        _, c = heapq.heappop(heap)
        if c in done:
            continue
        done.add(c)
        if c in goals:
            goal = c
            break
        i, j = c
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                a, b = i + di, j + dj
                if not (di or dj) or not (0 <= a < nx and 0 <= b < ny) or not free[a, b]:
                    continue
                if di and dj and not (free[a, j] and free[i, b]):
                    continue  # no corner cutting past a blocked cell
                nd = dist[c] + float(np.hypot(di, dj)) * t.step
                if nd < dist.get((a, b), np.inf):
                    dist[(a, b)] = nd
                    prev[(a, b)] = c
                    heapq.heappush(heap, (nd + float(np.linalg.norm(cells[a, b] - dst)), (a, b)))
    if goal is None:
        return None
    chain = [goal]
    while chain[-1] in prev:
        chain.append(prev[chain[-1]])
    path = [src] + [cells[c] for c in reversed(chain)] + [dst]
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _glide_ok(path[i], path[j], t, keep, radius):
            j -= 1
        out.append(path[j])
        i = j
    return out


def _anchors(t: ContactTable) -> np.ndarray:
    cells = _cell_world(t)
    return cells[t.contact_mask & t.hover_mask]


# ------------------------------------------------------------------- knots

class _Knots:
    """Joint-space knots with times; the arm is commanded their linear interpolation."""

    def __init__(self, q0: np.ndarray):
        self.times, self.qs = [0.0], [np.asarray(q0, dtype=float)]
        self.marks: list[int] = []      # knot index that ends each phase

    def mark(self) -> bool:
        if len(self.qs) - 1 > (self.marks[-1] if self.marks else 0):
            self.marks.append(len(self.qs) - 1)
        return True

    def move(self, q: np.ndarray | None, least: float) -> bool:
        if q is None:
            return False
        dq = float(np.abs(np.asarray(q) - self.qs[-1]).max())
        self.times.append(self.times[-1] + max(least, dq / MAX_JOINT_SPEED, 1e-4))
        self.qs.append(np.asarray(q, dtype=float))
        return True

    def glide(self, pts: list[np.ndarray], speed: float, t: ContactTable, cap_joint_speed: bool = True) -> bool:
        for a, b in zip(pts[:-1], pts[1:]):
            a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            length = float(np.linalg.norm(b - a))
            n = max(1, int(np.ceil(length / KNOT_SPACING)))
            for k in range(1, n + 1):
                p = a + (b - a) * (k / n)
                q = t.q_at(_robot_xy(p)[0], "contact", strict=True)
                if q is None:
                    return False
                least = (length / n) / speed
                if not cap_joint_speed:
                    self.times.append(self.times[-1] + max(least, 1e-4))
                    self.qs.append(q)
                elif not self.move(q, least):
                    return False
        return True

    def sample(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        tk, qk = np.array(self.times), np.stack(self.qs)
        grid = np.unique(np.concatenate([np.arange(0.0, tk[-1], dt), tk]))
        return grid, np.stack([np.interp(grid, tk, qk[:, j]) for j in range(qk.shape[1])], axis=1)


def _quintic(tau: np.ndarray) -> np.ndarray:
    return tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau ** 2)


def _retime(k: _Knots, stretch: np.ndarray, dt: float, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample the knots at ``dt`` with each phase on a quintic time scaling.

    Phase ``i`` keeps its knots and their spacing in time; its duration becomes
    ``stretch[i]`` times the constant-speed one. Returns ``(q, phase_ends)``,
    ``q`` of shape ``(n_steps, 5)`` holding the last knot after the end.
    """
    tk, qk = np.array(k.times), np.stack(k.qs)
    grid = np.arange(n_steps) * dt
    q = np.repeat(qk[-1][None], n_steps, axis=0)
    t0, starts = 0.0, [0] + k.marks[:-1]
    ends = []
    for i, (a, b) in enumerate(zip(starts, k.marks)):
        span = float(tk[b] - tk[a])
        dur = span * float(stretch[i])
        sel = (grid >= t0) & (grid < t0 + dur)
        if dur > 0 and sel.any():
            u = (tk[a:b + 1] - tk[a]) / span
            sp = _quintic((grid[sel] - t0) / dur)
            q[sel] = np.stack([np.interp(sp, u, qk[a:b + 1, j]) for j in range(qk.shape[1])], axis=1)
        t0 += dur
        ends.append(t0)
    q[grid >= t0] = qk[-1]
    return q, np.array(ends)


def _smooth(x: np.ndarray, sigma_steps: float) -> np.ndarray:
    if sigma_steps < 0.5:
        return x
    r = int(np.ceil(3 * sigma_steps))
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_steps) ** 2)
    w /= w.sum()
    pad = np.pad(x, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, j], w, mode="valid") for j in range(x.shape[1])], axis=1)


def feedforward(q: np.ndarray, dt: float, chain: ArmChain) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(q_cmd, qd, tau)`` that make Isaac's arm PD follow the joint path ``q`` sampled at ``dt``."""
    qd = np.gradient(q, dt, axis=0)
    qdd = _smooth(np.gradient(qd, dt, axis=0), ACC_SMOOTH_S / dt)
    tau = inverse_dynamics(chain, q, qd, qdd, armature=ARM_ARMATURE)
    return q + tau / ARM_KP, qd, tau


def hold_command(joints: tuple[str, ...], n_steps: int, chain: ArmChain | None = None):
    """``(q, q_cmd, qd)`` holding the pinch pose, gravity compensated."""
    arm = chain or load_chain()
    pinch = pinch_arm(joints)
    q = np.repeat(pinch[None], n_steps, axis=0)
    tau = inverse_dynamics(arm, pinch[None], np.zeros((1, 5)), np.zeros((1, 5)))[0]
    return q, q + tau / ARM_KP, np.zeros_like(q)


def _hull_hits(chain: ArmChain, q: np.ndarray, rect: Rect | None, garment_top: float) -> str | None:
    """Why this joint path is unsafe, or None. ``q`` is ``(N, 5)`` samples."""
    pts = _world_xyz(hull_points(chain, q))            # (N, P, 3)
    lo, hi = table_top_rect()
    over_table = ((pts[..., 0] >= lo[0]) & (pts[..., 0] <= hi[0])
                  & (pts[..., 1] >= lo[1]) & (pts[..., 1] <= hi[1]))
    if bool((over_table & (pts[..., 2] < TABLE_TOP_Z - TABLE_TOL)).any()):
        return "hand into the table"
    if rect is not None:
        near = rect.distance(pts[..., :2].reshape(-1, 2)).reshape(pts.shape[:2]) < 0.5 * KEEP_OUT
        if bool((near & (pts[..., 2] <= garment_top + 0.5 * KEEP_OUT)).any()):
            return "hand hits the garment"
    return None


def _expected_after(rect: Rect, start: np.ndarray, end: np.ndarray, radius: float) -> Rect | None:
    """Where a pushed garment should rest: just ahead of the hand, or None if off the table."""
    d = end - start
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        return rect
    n = max(2, int(np.ceil(length / 0.005)) + 1)
    pts = start[None] + d[None] * np.linspace(0.0, 1.0, n)[:, None]
    if not bool((rect.distance(pts) < radius).any()):
        return rect                                           # the sweep misses it
    u = d / length
    centre = end + u * (radius + rect.extent_along(u) + 0.005)
    ahead = float((rect.centre - end) @ u)
    if ahead > radius + rect.extent_along(u) + 0.005:
        centre = rect.centre                                  # already beyond the sweep's end
    lo, hi = table_top_rect()
    if not (lo[0] <= centre[0] <= hi[0] and lo[1] <= centre[1] <= hi[1]):
        return None
    return Rect(centre, rect.yaw, rect.half)


# ------------------------------------------------------------------ builder

def build_schedule(garment_xy, spec: GarmentSpec, action, dt: float, n_steps: int,
                   cfg: SweepConfig | None = None, table: ContactTable | None = None,
                   garment_yaw: float = 0.0, chain: ArmChain | None = None) -> Schedule:
    """Joint schedule for one sweep of ``spec`` at ``garment_xy`` (layout world frame)."""
    t = table or load_contact()
    arm = chain or load_chain()
    pinch = pinch_arm(t.joints)
    hold, hold_cmd, hold_qd = hold_command(t.joints, n_steps, arm)

    def refuse(reason: str) -> Schedule:
        return Schedule(hold, t.joints, False, reason, q_cmd=hold_cmd, qd=hold_qd)

    g = np.asarray(garment_xy, dtype=float)[:2]
    plan = plan_sweep(g, decode_action(action), cfg or SweepConfig())
    if not is_plan_valid(plan):
        return refuse("plan refused")
    rect = Rect(g, float(garment_yaw), 0.5 * np.asarray(spec.proxy_size[:2], dtype=float))
    top = TABLE_TOP_Z + float(spec.proxy_size[2])
    radius = HAND_RADIUS + KEEP_OUT
    start, end = np.asarray(plan.start_xy, dtype=float), np.asarray(plan.end_xy, dtype=float)
    if float(rect.distance(start)[0]) < radius:
        return refuse("sweep starts on the garment")
    anchors = _anchors(t)
    budget = min(MACRO_STEP_S - SETTLE_S, n_steps * dt)

    # ---- approach: first anchor whose routed approach the hull replay clears
    why = "no route to the sweep start"
    approach = None
    for a in anchors[np.argsort(np.linalg.norm(anchors - start, axis=1))][:N_ANCHORS]:
        if float(rect.distance(a)[0]) < radius:
            continue
        route = _route(a, start, t, rect, radius)
        if route is None:
            continue
        k = _Knots(pinch)
        ra = _robot_xy(a)[0]
        if not (k.move(t.q_at(ra, "hover"), MOVE_S) and k.mark()
                and k.move(t.q_at(ra, "contact"), DESCEND_S) and k.mark()
                and k.glide(route, GLIDE_SPEED, t) and k.mark()):
            why = "approach leaves the contact region"
            continue
        if QUINTIC_PEAK * k.times[-1] > budget:
            why = "approach does not fit the macro step"
            continue
        _, qs = k.sample(CHECK_DT)
        hit = _hull_hits(arm, qs, rect, top)
        if hit is not None:
            why = f"approach: {hit}"
            continue
        approach = (a, route, k)
        break
    if approach is None:
        return refuse(why)
    a_in, route_in, k = approach

    # ---- sweep
    t_sweep0 = k.times[-1]
    sweep_speed = max(plan.speed, QUINTIC_PEAK * plan.distance / MAX_SWEEP_S)
    n_before = len(k.qs)
    if not (k.glide([start, end], sweep_speed, t, cap_joint_speed=False) and k.mark()):
        return refuse("sweep leaves the contact region")
    i_sweep = len(k.marks) - 1
    t_sweep1 = k.times[-1]
    sweep_q = np.stack(k.qs[n_before - 1:])
    tk = np.array(k.times[n_before - 1:])
    grid = np.arange(tk[0], tk[-1] + 1e-9, CHECK_DT)
    hit = _hull_hits(arm, np.stack([np.interp(grid, tk, sweep_q[:, j]) for j in range(5)], axis=1), None, top)
    if hit is not None:
        return refuse(f"sweep: {hit}")

    # ---- retreat: first anchor reachable around where the garment should be
    after = _expected_after(rect, start, end, HAND_RADIUS)
    why = "no route back to an anchor"
    retreat = None
    base_times, base_qs, base_marks = list(k.times), list(k.qs), list(k.marks)
    for a in anchors[np.argsort(np.linalg.norm(anchors - end, axis=1))][:N_ANCHORS]:
        if after is not None and float(after.distance(a)[0]) < radius:
            continue
        route = _route(end, a, t, after, radius)
        if route is None:
            continue
        k.times, k.qs, k.marks = list(base_times), list(base_qs), list(base_marks)
        ra = _robot_xy(a)[0]
        if not (k.glide(route, GLIDE_SPEED, t) and k.mark()
                and k.move(t.q_at(ra, "hover"), DESCEND_S) and k.mark()
                and k.move(pinch, MOVE_S) and k.mark()):
            why = "retreat leaves the contact region"
            continue
        if QUINTIC_PEAK * k.times[-1] > budget:
            why = f"needs {QUINTIC_PEAK * k.times[-1]:.2f} s, macro step leaves {budget:.2f} s"
            continue
        tk_all = np.array(k.times)
        grid = np.arange(t_sweep1, tk_all[-1] + 1e-9, CHECK_DT)
        qk = np.stack(k.qs)
        qs = np.stack([np.interp(grid, tk_all, qk[:, j]) for j in range(5)], axis=1)
        hit = _hull_hits(arm, qs, after, top)
        if hit is not None:
            why = f"retreat: {hit}"
            continue
        retreat = (a, route)
        break
    if retreat is None:
        return refuse(why)
    a_out, route_out = retreat

    # ---- timing: quintic phases, slowed where the feedforward exceeds the arm
    # The re-timed path is the knot path the hull replay above checked; only
    # the time along it changes.
    stretch = np.full(len(k.marks), QUINTIC_PEAK)
    limit = TORQUE_BUDGET * ARM_EFFORT
    for _ in range(8):
        q, ends = _retime(k, stretch, dt, n_steps)
        if ends[-1] > budget + 1e-9:
            return refuse(f"needs {ends[-1]:.2f} s, macro step leaves {budget:.2f} s")
        q_cmd, qd, tau = feedforward(q, dt, arm)
        peak = np.abs(tau).max(axis=1)
        times = np.arange(n_steps) * dt
        over = [i for i, (a, b) in enumerate(zip(np.r_[0.0, ends[:-1]], ends))
                if peak[(times >= a) & (times < b)].max(initial=0.0) > limit]
        if not over:
            break
        stretch[over] *= 1.3
    else:
        return refuse(f"needs {float(peak.max()):.2f} N m; the arm has {ARM_EFFORT:.1f}")
    hi_pose = dict(zip(t.joints, q.max(axis=0)))
    lo_pose = dict(zip(t.joints, q.min(axis=0)))
    if not (is_valid_joints({**PINCH_JOINT_POS, **hi_pose}) and is_valid_joints({**PINCH_JOINT_POS, **lo_pose})):
        return refuse("schedule leaves the joint walls")
    window = (float(ends[i_sweep - 1]) if i_sweep > 0 else 0.0, float(ends[i_sweep]))
    return Schedule(q, t.joints, True, "ok", start, end, float(ends[-1]), a_in, a_out,
                    route_in, route_out, window, q_cmd=q_cmd, qd=qd, tau_peak=float(np.abs(tau).max()))
