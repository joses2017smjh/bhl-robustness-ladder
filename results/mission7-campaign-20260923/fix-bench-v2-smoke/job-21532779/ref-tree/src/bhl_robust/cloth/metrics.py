"""Evaluation metrics for one cloth-sort run, plus the transfer gap."""

from __future__ import annotations

from dataclasses import dataclass, field, fields


def transfer_gap(rigid_success_rate: float, deformable_success_rate: float) -> float:
    """``rigid_success_rate - deformable_success_rate``.

    Positive means the rigid proxy overestimates what the cloth will do.
    """
    return float(rigid_success_rate) - float(deformable_success_rate)


@dataclass
class EpisodeMetrics:
    success: bool = False
    correct_count: int = 0
    wrong_count: int = 0
    n_garments: int = 1
    n_sweeps: int = 0
    steps_to_success: int | None = None
    displacement_per_sweep: list[float] = field(default_factory=list)
    final_distance: float = 0.0
    fell: bool = False
    invalid_trajectory: int = 0
    env_steps: int = 0

    @property
    def correct_rate(self) -> float:
        return self.correct_count / max(self.n_garments, 1)

    @property
    def wrong_rate(self) -> float:
        return self.wrong_count / max(self.n_garments, 1)

    @property
    def mean_displacement(self) -> float:
        if not self.displacement_per_sweep:
            return 0.0
        return float(np_mean(self.displacement_per_sweep))


def np_mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


@dataclass
class RunMetrics:
    episodes: list[EpisodeMetrics] = field(default_factory=list)
    env_steps_per_s: float | None = None
    real_time_factor: float | None = None
    gpu_mem_mb: float | None = None
    physics: str = "kinematic_rigid"
    policy: str = "scripted"
    num_envs: int = 1
    cloth_resolution: str = "n/a"
    n_deformables: int = 0
    n_vertices: int = 0

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return float("nan")
        return sum(e.success for e in self.episodes) / len(self.episodes)

    @property
    def fall_rate(self) -> float:
        if not self.episodes:
            return float("nan")
        return sum(e.fell for e in self.episodes) / len(self.episodes)

    @property
    def mean_sweeps(self) -> float:
        if not self.episodes:
            return float("nan")
        return np_mean([float(e.n_sweeps) for e in self.episodes])

    @property
    def mean_steps_to_success(self) -> float:
        vals = [e.steps_to_success for e in self.episodes if e.steps_to_success is not None]
        return np_mean([float(v) for v in vals]) if vals else float("nan")

    @property
    def wrong_basket_rate(self) -> float:
        if not self.episodes:
            return float("nan")
        return np_mean([e.wrong_rate for e in self.episodes])

    @property
    def invalid_rate(self) -> float:
        """Fraction of sweeps the controller refused, in [0, 1].

        This used to be the mean per-episode *count*, published under the key
        ``invalid_trajectory_rate`` -- which read 6.0 the first time every sweep
        was refused. A rate cannot exceed 1. The count is still reported, as
        ``invalid_per_episode``.
        """
        total = sum(e.n_sweeps for e in self.episodes)
        if total == 0:
            return float("nan")
        return sum(e.invalid_trajectory for e in self.episodes) / total

    @property
    def invalid_per_episode(self) -> float:
        if not self.episodes:
            return float("nan")
        return np_mean([float(e.invalid_trajectory) for e in self.episodes])

    def as_dict(self) -> dict[str, float | str | int | None]:
        return {
            "success_rate": self.success_rate,
            "wrong_basket_rate": self.wrong_basket_rate,
            "mean_sweeps": self.mean_sweeps,
            "mean_steps_to_success": self.mean_steps_to_success,
            "fall_rate": self.fall_rate,
            "invalid_trajectory_rate": self.invalid_rate,
            "invalid_per_episode": self.invalid_per_episode,
            "env_steps_per_s": self.env_steps_per_s,
            "real_time_factor": self.real_time_factor,
            "gpu_mem_mb": self.gpu_mem_mb,
            "n_episodes": len(self.episodes),
            "physics": self.physics,
            "policy": self.policy,
            "num_envs": self.num_envs,
            "cloth_resolution": self.cloth_resolution,
            "n_deformables": self.n_deformables,
            "n_vertices": self.n_vertices,
        }


def summarize_transfer(
    rigid: RunMetrics,
    deformable: RunMetrics,
    adapted: RunMetrics | None = None,
) -> dict[str, float]:
    out = {
        "rigid_success": rigid.success_rate,
        "zero_shot_deformable_success": deformable.success_rate,
        "transfer_gap": transfer_gap(rigid.success_rate, deformable.success_rate),
    }
    if adapted is not None:
        out["deformable_finetuned_success"] = adapted.success_rate
    return out


# Keep dataclass field iteration available for CSV writers.
METRIC_FIELDS = [f.name for f in fields(RunMetrics) if f.name != "episodes"]


def append_rows_csv(path, rows) -> list[str]:
    """Append ``rows`` to a CSV, unioning the schema instead of misaligning it.

    ``csv.DictWriter`` in append mode writes *this* run's field order under
    whatever header is already on disk. When the two differ — the kinematic
    bench and the Isaac bench do not share columns — every appended value
    lands under the wrong heading. `results/cloth_sort_bench.csv` after job
    21234167 had `env_steps_per_s` sitting in the `cloth_resolution` column.
    The markdown summary looked right, which is what made it easy to miss.

    So: read what is there, union the field names, and rewrite the file with
    one header that covers old and new rows. Existing rows are preserved —
    benchmark results are not deleted to fix a schema.

    Returns the final field list.
    """
    import csv
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        with path.open(newline="") as f:
            existing = [dict(r) for r in csv.DictReader(f)]
    new_rows = [dict(r) for r in rows]
    fields: list[str] = []
    for r in (*existing, *new_rows):
        for k in r:
            if k and k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in (*existing, *new_rows):
            w.writerow({k: r.get(k, "") for k in fields})
    return fields
