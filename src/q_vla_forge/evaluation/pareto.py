from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParetoPoint:
    """One compression/error operating point."""

    method: str
    configuration: str
    family: str

    compression_ratio_mean: float
    compression_ratio_std: float

    mse_change_mean: float
    mse_change_std: float

    mae_change_mean: float
    mae_change_std: float

    pilot_feasible_runs: int
    all_runs_pilot_feasible: bool

    is_reference: bool = False


@dataclass(frozen=True)
class DomainParetoAnalysis:
    """Pareto analysis for one pilot domain."""

    domain: str

    points: tuple[ParetoPoint, ...]
    pareto_frontier: tuple[str, ...]
    dominated: tuple[str, ...]

    minimum_compression_ratio: float
    maximum_mse_increase_percent: float


def _validate_point(
    point: ParetoPoint,
) -> None:
    """Validate one Pareto operating point."""
    values = (
        point.compression_ratio_mean,
        point.compression_ratio_std,
        point.mse_change_mean,
        point.mse_change_std,
        point.mae_change_mean,
        point.mae_change_std,
    )

    if not all(math.isfinite(value) for value in values):
        raise ValueError("Pareto point values must be finite")

    if point.compression_ratio_mean <= 0.0:
        raise ValueError("compression ratio must be positive")

    if point.compression_ratio_std < 0.0:
        raise ValueError("compression ratio SD must not be negative")

    if point.mse_change_std < 0.0:
        raise ValueError("MSE SD must not be negative")

    if point.mae_change_std < 0.0:
        raise ValueError("MAE SD must not be negative")


def dominates(
    candidate: ParetoPoint,
    other: ParetoPoint,
) -> bool:
    """
    Return True when candidate Pareto-dominates other.

    Compression is maximized.
    Relative MSE change is minimized.
    """
    _validate_point(candidate)

    _validate_point(other)

    no_worse_compression = (
        candidate.compression_ratio_mean >= other.compression_ratio_mean
    )

    no_worse_error = candidate.mse_change_mean <= other.mse_change_mean

    strictly_better = (
        candidate.compression_ratio_mean > other.compression_ratio_mean
        or candidate.mse_change_mean < other.mse_change_mean
    )

    return no_worse_compression and no_worse_error and strictly_better


def pareto_frontier(
    points: Sequence[ParetoPoint],
    *,
    include_reference: bool = False,
) -> tuple[ParetoPoint, ...]:
    """Return non-dominated compression points."""
    if not points:
        raise ValueError("at least one Pareto point is required")

    for point in points:
        _validate_point(point)

    candidates = [
        point for point in points if (include_reference or not point.is_reference)
    ]

    if not candidates:
        raise ValueError("no eligible Pareto candidates")

    frontier: list[ParetoPoint] = []

    for point in candidates:
        is_dominated = any(
            dominates(
                other,
                point,
            )
            for other in candidates
            if other is not point
        )

        if not is_dominated:
            frontier.append(point)

    frontier.sort(
        key=lambda point: (
            point.compression_ratio_mean,
            point.mse_change_mean,
            point.configuration,
        )
    )

    return tuple(frontier)


def build_domain_pareto_analysis(
    *,
    domain: str,
    points: Sequence[ParetoPoint],
    minimum_compression_ratio: float = 2.0,
    maximum_mse_increase_percent: float = 5.0,
) -> DomainParetoAnalysis:
    """Build Pareto evidence for one domain."""
    if not domain:
        raise ValueError("domain must not be empty")

    if minimum_compression_ratio <= 0.0:
        raise ValueError("minimum compression ratio must be positive")

    if not points:
        raise ValueError("points must not be empty")

    for point in points:
        _validate_point(point)

    frontier = pareto_frontier(
        points,
        include_reference=False,
    )

    frontier_labels = tuple(point.configuration for point in frontier)

    compression_points = [point for point in points if not point.is_reference]

    dominated_labels = tuple(
        point.configuration
        for point in compression_points
        if point.configuration not in frontier_labels
    )

    return DomainParetoAnalysis(
        domain=domain,
        points=tuple(points),
        pareto_frontier=(frontier_labels),
        dominated=(dominated_labels),
        minimum_compression_ratio=(minimum_compression_ratio),
        maximum_mse_increase_percent=(maximum_mse_increase_percent),
    )


def save_domain_pareto_analysis(
    analysis: DomainParetoAnalysis,
    path: Path,
) -> None:
    """Save a domain Pareto report."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            asdict(analysis),
            indent=2,
        ),
        encoding="utf-8",
    )
