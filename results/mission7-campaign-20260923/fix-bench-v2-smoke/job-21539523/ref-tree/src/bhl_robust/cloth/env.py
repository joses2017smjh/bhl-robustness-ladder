"""Kinematic rigid-proxy cloth-sort environment.

One macro-step is one sweep. Garments are sliding rectangles on the table;
there is no Newton cloth and no 22-DoF balance. That is intentional: C0 and
C1 must be runnable on a login node, and a planner that cannot sort rectangles
will not sort cloth.

Isaac rigid-body and deformable stages share the same action, observation,
success, and reward code. They do not share this integrator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bhl_robust.cloth.controller import default_pose
from bhl_robust.cloth.garments import (
    BASKET_IDS,
    GARMENTS,
    GarmentSpec,
    class_index,
    one_garment,
)
from bhl_robust.cloth.layout import (
    CONTACT_HEIGHT,
    HAND_RADIUS,
    ROBOT_XY,
    TABLE_TOP_Z,
    basket_aabb,
    basket_center,
    garment_spawn_range,
    parking_xy,
    table_aabb,
)
from bhl_robust.cloth.observations import ObservationSpec, pack_oracle
from bhl_robust.cloth.randomization import IDENTITY, DomainRandomization
from bhl_robust.cloth.rewards import RewardBreakdown, RewardWeights, compute_reward
from bhl_robust.cloth.schedule import CHECK_DT, MACRO_STEP_S, build_schedule
from bhl_robust.cloth.scripted import pick_unsorted
from bhl_robust.cloth.success import rigid_outcome
from bhl_robust.cloth.sweep import SweepConfig, decode_action, plan_sweep, segment_hits_disk
from bhl_robust.limb_partition import JOINTS_22


@dataclass
class _GarmentState:
    spec: GarmentSpec
    xy: np.ndarray
    z: float
    yaw: float
    mass: float
    size: np.ndarray
    friction: float
    sorted_correct: bool = False
    sorted_wrong: bool = False
    #: Waiting off the table for its turn (five-garment scene).
    parked: bool = False
    #: Where it goes on the table when its turn comes.
    home: np.ndarray | None = None


@dataclass
class StepInfo:
    reward: RewardBreakdown
    selected: int
    invalid: bool
    hit: bool
    displacement: float
    outcomes: list
    all_correct: bool
    any_wrong: bool


class KinematicClothSortEnv:
    """Oracle-state, 5-D sweep-action environment."""

    def __init__(
        self,
        garments: tuple[GarmentSpec, ...] | None = None,
        *,
        domain_rand: DomainRandomization | None = None,
        sweep_cfg: SweepConfig | None = None,
        reward_weights: RewardWeights | None = None,
        max_sweeps: int | None = None,
        success_threshold: float = 1.0,
        seed: int = 0,
        active_garment_mode: bool = False,
    ) -> None:
        self.specs = garments if garments is not None else one_garment()
        self.n_garments = len(self.specs)
        self.domain_rand = domain_rand if domain_rand is not None else IDENTITY
        self.sweep_cfg = sweep_cfg or SweepConfig()
        self.reward_weights = reward_weights or RewardWeights()
        self.max_sweeps = max_sweeps if max_sweeps is not None else max(6, 2 * self.n_garments + 2)
        self.success_threshold = success_threshold
        self.active_garment_mode = active_garment_mode
        self.spec_obs = ObservationSpec(n_garments=self.n_garments, privileged=True)
        self._rng = np.random.default_rng(seed)
        self._states: list[_GarmentState] = []
        self._n_sweeps = 0
        self._selected = 0
        self._done = False
        self.action_space_n = 5
        self.observation_space_n = self.spec_obs.size

    # ------------------------------------------------------------------ reset

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        spawn_lo, spawn_hi = garment_spawn_range()
        centre = 0.5 * (spawn_lo + spawn_hi)
        self._states = []
        # The redesigned table holds one garment at a time: it is 0.20 x 0.18 m,
        # the size of the robot's fingertip workspace. A five-garment scene
        # therefore presents garments in turn -- the first on the table, the
        # rest parked well clear -- and places the next when one is sorted.
        # That is Mode B, sequential; it is not five garments on a table at once.
        for i, spec in enumerate(self.specs):
            dr = self.domain_rand.sample(self._rng)
            home = np.array([centre[0] + dr["spawn_dx"], centre[1] + dr["spawn_dy"]])
            home[0] = float(np.clip(home[0], spawn_lo[0], spawn_hi[0]))
            home[1] = float(np.clip(home[1], spawn_lo[1], spawn_hi[1]))
            parked = i > 0
            size = np.array(spec.proxy_size) * dr["size_scale"]
            self._states.append(_GarmentState(
                spec=spec,
                xy=np.array(parking_xy(i)) if parked else home.copy(),
                z=TABLE_TOP_Z + 0.5 * size[2],
                yaw=dr["yaw"],
                mass=spec.mass * dr["mass_scale"],
                size=size,
                friction=dr["friction"],
                parked=parked,
                home=home,
            ))
        self._n_sweeps = 0
        self._done = False
        self._selected = self._pick()
        return self.observation()

    # ------------------------------------------------------------------- step

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, StepInfo]:
        if self._done:
            raise RuntimeError("step() called after the episode ended; reset() first")
        # Hold the selection chosen at reset / after the previous sweep. Re-
        # picking here would apply an action the policy computed for a
        # different garment — that is what made C5 score 0.
        # Mode B: only the selected garment is "active". Others are frozen
        # proxies and cannot be displaced this sweep. Documented as such.
        active = {i for i, st in enumerate(self._states) if not st.parked}
        if self.active_garment_mode or self.n_garments > 1:
            active = {self._selected}

        prev_dist = self._distance_to_target(self._selected)
        params = decode_action(action)
        sel = self._states[self._selected]
        plan = plan_sweep(sel.xy, params, self.sweep_cfg)
        # Refused exactly when the Isaac action term would refuse it: the same
        # schedule, with its routes and hand-hull replay, not a cell check alone.
        invalid = not build_schedule(
            sel.xy, sel.spec, action, CHECK_DT, int(round(MACRO_STEP_S / CHECK_DT)),
            cfg=self.sweep_cfg, garment_yaw=sel.yaw,
        ).valid
        hit = False
        total_disp = 0.0
        # A trajectory the controller refuses does not execute, so it moves
        # nothing. Before this, an invalid plan was logged as invalid and then
        # swept the garment anyway -- which is how a hand that cannot reach the
        # table still sorted every garment on it.
        for i in (() if invalid else active):
            st = self._states[i]
            if st.sorted_correct or st.sorted_wrong:
                continue
            radius = HAND_RADIUS + 0.5 * float(np.hypot(st.size[0], st.size[1]))
            if not segment_hits_disk(plan.start_xy, plan.end_xy, st.xy, radius):
                continue
            hit = True
            before = st.xy.copy()
            coupling = float(np.clip(0.55 + st.friction, 0.70, 1.00))
            # Light garments slide a little further; do not let that
            # overshoot launch them past the basket.
            coupling *= float(np.clip(0.12 / max(st.mass, 1e-3), 0.85, 1.05))
            travel = min(plan.distance * coupling, float(np.linalg.norm(
                basket_center(st.spec.target_basket)[:2] - st.xy
            )) + 0.08)
            push = plan.direction * travel
            st.xy = st.xy + push
            self._resolve_table_or_basket(st)
            total_disp += float(np.linalg.norm(st.xy - before))

        newly_correct = False
        newly_wrong = False
        outcomes = []
        for st in self._states:
            was_c, was_w = st.sorted_correct, st.sorted_wrong
            out = rigid_outcome(
                np.array([st.xy[0], st.xy[1], st.z]),
                st.spec,
                threshold=self.success_threshold,
            )
            outcomes.append(out)
            st.sorted_correct = out.correct
            st.sorted_wrong = out.wrong
            if out.correct and not was_c:
                newly_correct = True
            if out.wrong and not was_w:
                newly_wrong = True

        curr_dist = self._distance_to_target(self._selected)
        useful = float(np.linalg.norm(
            basket_center(self._states[self._selected].spec.target_basket)[:2]
            - self._states[self._selected].xy
        ))
        # ``useful`` is *after* the sweep; the efficiency penalty uses the
        # pre-sweep distance so a necessary long sweep is not punished.
        rb = compute_reward(
            prev_dist=prev_dist,
            curr_dist=curr_dist,
            newly_correct=newly_correct,
            newly_wrong=newly_wrong,
            sweep_distance=plan.distance,
            min_useful_distance=prev_dist + 0.08,
            fell=False,
            invalid=invalid,
            weights=self.reward_weights,
        )
        self._n_sweeps += 1
        all_correct = all(s.sorted_correct for s in self._states)
        done = all_correct or self._n_sweeps >= self.max_sweeps
        self._done = done
        if not done:
            self._selected = self._pick()
        info = StepInfo(
            reward=rb,
            selected=self._selected,
            invalid=invalid,
            hit=hit,
            displacement=total_disp,
            outcomes=outcomes,
            all_correct=all_correct,
            any_wrong=any(s.sorted_wrong for s in self._states),
        )
        return self.observation(), float(rb.total), done, info

    # --------------------------------------------------------------- helpers

    def _pick(self) -> int:
        done = np.array(
            [s.sorted_correct or s.sorted_wrong or s.parked for s in self._states], dtype=bool
        )
        if done.all():
            # Nothing unsorted on the table: bring the next parked garment on.
            for st in self._states:
                if st.parked and not (st.sorted_correct or st.sorted_wrong):
                    st.parked = False
                    st.xy = st.home.copy()
                    st.z = TABLE_TOP_Z + 0.5 * st.size[2]
                    break
            done = np.array(
                [s.sorted_correct or s.sorted_wrong or s.parked for s in self._states], dtype=bool
            )
        return pick_unsorted(done, np.stack([s.xy for s in self._states]))

    def _distance_to_target(self, idx: int) -> float:
        st = self._states[idx]
        return float(np.linalg.norm(st.xy - basket_center(st.spec.target_basket)[:2]))

    def _resolve_table_or_basket(self, st: _GarmentState) -> None:
        table = table_aabb()
        on_table = table.low[0] <= st.xy[0] <= table.high[0] and table.low[1] <= st.xy[1] <= table.high[1]
        if on_table:
            st.z = TABLE_TOP_Z + 0.5 * st.size[2]
            return
        # Off the table: drop. If the CoM is over a basket's opening, it lands in
        # that basket; otherwise it rests on the floor. The opening is the
        # basket's own box -- this used a hard-coded 0.24 x 0.28 footprint that
        # described no basket in either layout.
        st.z = 0.5 * st.size[2]
        for bid in BASKET_IDS:
            box = basket_aabb(bid)
            if box.contains_xy(st.xy):
                st.z = 0.5 * box.high[2] + 0.5 * st.size[2]
                break

    def observation(self) -> np.ndarray:
        joints = default_pose()
        jp = np.array([joints[n] for n in JOINTS_22], dtype=float)
        jv = np.zeros(len(JOINTS_22), dtype=float)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        ang = np.zeros(3)
        hands = np.array([
            ROBOT_XY[0] + 0.15, 0.12, CONTACT_HEIGHT,
            ROBOT_XY[0] + 0.15, -0.12, CONTACT_HEIGHT,
        ])
        gxy = np.stack([s.xy for s in self._states])
        gyaw = np.array([s.yaw for s in self._states])
        cls = np.array([class_index(s.spec.semantic_class) for s in self._states])
        rel = np.stack([
            basket_center(s.spec.target_basket)[:2] - s.xy for s in self._states
        ])
        sorted_mask = np.array([s.sorted_correct for s in self._states], dtype=float)
        basket_xy = np.stack([basket_center(b)[:2] for b in BASKET_IDS])
        progress = float(sorted_mask.mean())
        return pack_oracle(
            joint_pos=jp, joint_vel=jv, base_quat=quat, base_ang_vel=ang,
            hand_pos=hands, garment_xy=gxy, garment_yaw=gyaw, class_index=cls,
            rel_xy=rel, selected=self._selected, sorted_mask=sorted_mask,
            basket_xy=basket_xy, progress=progress, spec=self.spec_obs,
        )

    # ------------------------------------------------------- episode helpers

    @property
    def all_correct(self) -> bool:
        return bool(self._states) and all(s.sorted_correct for s in self._states)

    @property
    def n_sweeps(self) -> int:
        return self._n_sweeps

    def garment_xy(self, idx: int = 0) -> np.ndarray:
        return self._states[idx].xy.copy()

    def garment_com(self, idx: int = 0) -> np.ndarray:
        st = self._states[idx]
        return np.array([st.xy[0], st.xy[1], st.z])

    def selected_spec(self) -> GarmentSpec:
        return self._states[self._selected].spec

    def garment_yaw(self, idx: int = 0) -> float:
        return float(self._states[idx].yaw)

    def snapshot(self) -> list[dict]:
        return [
            {
                "name": s.spec.name,
                "xy": s.xy.tolist(),
                "z": s.z,
                "correct": s.sorted_correct,
                "wrong": s.sorted_wrong,
            }
            for s in self._states
        ]


def make_env(rung: str = "C0", seed: int = 0, **kwargs) -> KinematicClothSortEnv:
    """Factory matching the C0 / C1 / C5 kinematic stand-ins."""
    if rung in ("C0", "C1", "C2", "C3", "C4"):
        garments = one_garment(kwargs.pop("garment", "shirt_a"))
        dr = IDENTITY if rung == "C0" else DomainRandomization()
        if "domain_rand" in kwargs:
            dr = kwargs.pop("domain_rand")
        return KinematicClothSortEnv(garments, domain_rand=dr, seed=seed, **kwargs)
    if rung == "C5":
        return KinematicClothSortEnv(
            GARMENTS,
            domain_rand=kwargs.pop("domain_rand", IDENTITY),
            seed=seed,
            active_garment_mode=kwargs.pop("active_garment_mode", True),
            **kwargs,
        )
    raise ValueError(f"unknown rung {rung!r}")
