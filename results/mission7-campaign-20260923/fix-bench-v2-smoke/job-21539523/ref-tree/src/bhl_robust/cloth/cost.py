"""Wall-clock cost guard for deformable (and large rigid) training jobs.

G-C1 measured Newton VBD cloth at 182 env-steps/s (8 envs), 177 (32), 71 (128)
on a *lighter* scene than this task needs. A standard 8,000-iteration arm at
2,048 envs × 48 steps is 786 million env-steps: about 50 days at that peak
rate. This module makes that arithmetic unavoidable before anyone submits.
"""

from __future__ import annotations

from dataclasses import dataclass

# G-C1, job 21185969, Isaac-Lift-Cloth-Franka-v0, 961-vertex cloth.
GC1_ENV_STEPS_PER_S = {
    8: 182.0,
    32: 177.0,
    128: 71.0,
}

STANDARD_ITERATIONS = 8000
STANDARD_STEPS_PER_ITER = 48
STANDARD_NUM_ENVS = 2048


@dataclass(frozen=True)
class CostReport:
    num_envs: int
    iterations: int
    steps_per_iter: int
    env_steps: int
    measured_env_steps_per_s: float
    estimated_seconds: float
    source: str
    accepted: bool
    reason: str

    @property
    def estimated_hours(self) -> float:
        return self.estimated_seconds / 3600.0

    @property
    def estimated_days(self) -> float:
        return self.estimated_seconds / 86400.0

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "num_envs": self.num_envs,
            "iterations": self.iterations,
            "steps_per_iter": self.steps_per_iter,
            "env_steps": self.env_steps,
            "measured_env_steps_per_s": self.measured_env_steps_per_s,
            "estimated_seconds": self.estimated_seconds,
            "estimated_hours": self.estimated_hours,
            "estimated_days": self.estimated_days,
            "source": self.source,
            "accepted": self.accepted,
            "reason": self.reason,
        }


def estimate_env_steps(iterations: int, num_envs: int, steps_per_iter: int) -> int:
    return int(iterations) * int(num_envs) * int(steps_per_iter)


def nearest_gc1_rate(num_envs: int) -> tuple[float, str]:
    """Use the closest measured G-C1 point; do not interpolate a speedup.

    Throughput *fell* with env count. Interpolating between 8 and 128 as if
    the curve were flat would invent capacity that was not measured.
    """
    keys = sorted(GC1_ENV_STEPS_PER_S)
    chosen = min(keys, key=lambda k: abs(k - num_envs))
    rate = GC1_ENV_STEPS_PER_S[chosen]
    return rate, f"G-C1 measured {rate:g} env-steps/s at {chosen} envs (job 21185969)"


def report_cost(
    *,
    num_envs: int,
    iterations: int = STANDARD_ITERATIONS,
    steps_per_iter: int = STANDARD_STEPS_PER_ITER,
    measured_env_steps_per_s: float | None = None,
    max_hours: float = 12.0,
    i_accept_the_cost: bool = False,
    physics: str = "deformable",
) -> CostReport:
    """Build a cost report and decide whether a job may be submitted.

    Rigid-proxy jobs are cheap and pass at the default budget. Deformable
    jobs use a measured (or G-C1) rate and refuse unless the estimate fits
    ``max_hours`` or the caller set ``i_accept_the_cost``.
    """
    env_steps = estimate_env_steps(iterations, num_envs, steps_per_iter)
    if measured_env_steps_per_s is not None:
        rate = float(measured_env_steps_per_s)
        source = "caller-measured"
    elif physics == "deformable":
        rate, source = nearest_gc1_rate(num_envs)
    else:
        # Rigid-body locomotion on this cluster is ~2e5 env-steps/s at 2048.
        # Use a conservative 1e4 so a mis-set NUM_ENVS still looks cheap.
        rate, source = 1.0e4, "conservative rigid-body floor (not a cloth measurement)"

    seconds = env_steps / max(rate, 1e-9)
    hours = seconds / 3600.0
    if physics != "deformable":
        accepted, reason = True, "rigid-proxy / kinematic job; cloth cost gate does not apply"
    elif hours <= max_hours:
        accepted, reason = True, f"estimated {hours:.2f} h <= budget {max_hours:.2f} h"
    elif i_accept_the_cost:
        accepted, reason = True, (
            f"estimated {hours:.2f} h exceeds budget {max_hours:.2f} h; "
            "accepted explicitly via i_accept_the_cost"
        )
    else:
        accepted, reason = False, (
            f"estimated {hours:.2f} h exceeds budget {max_hours:.2f} h. "
            "Measure a real env-steps/s and pass it in, shrink the job, or "
            "set i_accept_the_cost=True after reading the number."
        )
    return CostReport(
        num_envs=num_envs,
        iterations=iterations,
        steps_per_iter=steps_per_iter,
        env_steps=env_steps,
        measured_env_steps_per_s=rate,
        estimated_seconds=seconds,
        source=source,
        accepted=accepted,
        reason=reason,
    )


def guard_or_raise(**kwargs) -> CostReport:
    """Raise ``RuntimeError`` unless the job is accepted."""
    rep = report_cost(**kwargs)
    if not rep.accepted:
        raise RuntimeError(
            "refusing to submit a deformable training job: "
            f"{rep.reason} ({rep.env_steps} env-steps at "
            f"{rep.measured_env_steps_per_s:g} env-steps/s → "
            f"{rep.estimated_days:.1f} days)"
        )
    return rep
