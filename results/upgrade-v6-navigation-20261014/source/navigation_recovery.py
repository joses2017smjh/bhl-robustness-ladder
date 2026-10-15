"""Navigation recovery from observed spatial progress, with no maze topology.

Global presets: 0.30 m anchored movement in 8 s, or no new 0.50 m visited
cell for 15 s after travelling at least 2 m, with 12 s initial grace.
Recovery latches for the rest of the episode and invokes the existing
observed-occupancy A* / turn-walk controller. Distance to the goal never
triggers recovery: necessary maze detours can increase that distance.
V6 budget_watchdog also hands off by one third of the episode's declared
time limit, reserving the remaining two thirds for the observed-map planner.
This is an explicit global time allocation, not a route-time prediction.
"""
from __future__ import annotations
import math
import numpy as np

MODES = ('none', 'watchdog', 'budget_watchdog', 'planner')
SETTINGS = dict(stall_radius_m=.30, stall_s=8., spatial_cell_m=.50,
                novelty_timeout_s=15., loop_path_m=2., grace_s=12.)
BUDGET_FRACTION = 1. / 3.


class NavigationRecovery:
    def __init__(self, mode='none', time_limit_s=180.):
        if mode not in MODES:
            raise ValueError(f'unknown navigation recovery mode: {mode}')
        if not np.isfinite(time_limit_s) or time_limit_s <= 0:
            raise ValueError('positive finite episode time limit required')
        self.mode = mode
        self.time_limit_s = float(time_limit_s)
        self.actor_deadline_s = self.time_limit_s * BUDGET_FRACTION
        self.active = mode == 'planner'
        self.reason = 'planner_from_start' if self.active else None
        self.trigger_s = 0. if self.active else None
        self.anchor = self.previous = None
        self.last_escape = self.last_novel = 0.
        self.path_since_novel = 0.
        self.visited = set()
        self.steps = self.actor_steps = self.recovery_steps = 0
        self.plans = self.plan_failures = 0
        self.last_info = {}

    def update(self, now, xy, settle_s=1.):
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(now):
            raise ValueError('finite estimated planar pose and time required')
        if self.previous is None or now < settle_s:
            self.anchor = xy.copy()
            self.last_escape = self.last_novel = now
        else:
            self.path_since_novel += float(np.linalg.norm(xy-self.previous))
        self.previous = xy.copy()
        cell = tuple(math.floor(float(c)/SETTINGS['spatial_cell_m']) for c in xy)
        if cell not in self.visited:
            self.visited.add(cell)
            self.last_novel = now
            self.path_since_novel = 0.
        if np.linalg.norm(xy-self.anchor) > SETTINGS['stall_radius_m']:
            self.anchor = xy.copy()
            self.last_escape = now
        stall = now-self.last_escape
        no_novel = now-self.last_novel
        if self.mode in ('watchdog', 'budget_watchdog') and not self.active:
            if self.mode == 'budget_watchdog' and now >= max(settle_s, self.actor_deadline_s):
                self.reason = 'actor_time_budget_exhausted'
            elif now < settle_s+SETTINGS['grace_s']:
                pass
            elif stall >= SETTINGS['stall_s']:
                self.reason = 'sustained_spatial_stall'
            elif no_novel >= SETTINGS['novelty_timeout_s'] and self.path_since_novel >= SETTINGS['loop_path_m']:
                self.reason = 'repeated_visited_region'
            if self.reason:
                self.active = True
                self.trigger_s = float(now)
        self.steps += 1
        self.recovery_steps += int(self.active)
        self.actor_steps += int(not self.active)
        self.last_info = dict(active=self.active, reason=self.reason, trigger_s=self.trigger_s,
                              stall_s=float(stall), no_novel_cell_s=float(no_novel),
                              path_since_novel_m=self.path_since_novel, visited_cells=len(self.visited))
        if self.mode == 'budget_watchdog':
            self.last_info['actor_deadline_s'] = self.actor_deadline_s
        return self.active

    def record_plan(self, found):
        self.plans += 1
        self.plan_failures += int(not found)

    def summary(self):
        out = dict(mode=self.mode, active=self.active, trigger_s=self.trigger_s, reason=self.reason,
                    settings=SETTINGS, steps=self.steps, actor_steps=self.actor_steps,
                    recovery_steps=self.recovery_steps, plans=self.plans, plan_failures=self.plan_failures,
                    visited_cells=len(self.visited),
                    source='Current estimated pose and previously visited positions only; A* uses observed lidar occupancy, estimated pose and existing goal.',
                    authority='Recovery is latched. Existing sensor brakes remain active. Model authority is disabled during recovery; adaptive reference applies to both model and reference arms.')
        if self.mode == 'budget_watchdog':
            out['time_budget'] = dict(episode_time_limit_s=self.time_limit_s,
                                      actor_fraction=BUDGET_FRACTION,
                                      actor_deadline_s=self.actor_deadline_s,
                                      allocation='Fixed global elapsed-time cap; no route or success prediction.')
        return out


class RecoveryGovernor:
    """Suppress model choices during planner control while retaining filter state."""
    def __init__(self, governor, recovery, model_methods):
        self.governor = governor
        self.recovery = recovery
        self.model_methods = tuple(model_methods)

    def filter(self, *args, **kwargs):
        original_kind = self.governor.kind
        if self.recovery.active and original_kind in self.model_methods:
            self.governor.kind = 'adaptive'
        try:
            output = self.governor.filter(*args, **kwargs)
        finally:
            self.governor.kind = original_kind
        self.governor.last_info['navigation_phase'] = 'planner_recovery' if self.recovery.active else 'actor'
        return output

    @property
    def last_info(self):
        return self.governor.last_info

    def observe_transition(self, **kwargs):
        return self.governor.observe_transition(**kwargs)

    def summary(self):
        out = self.governor.summary()
        out['model_scope'] = 'Model authority and all influence counters exclude planner recovery.'
        return out
