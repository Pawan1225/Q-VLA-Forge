"""Training-efficiency analysis utilities for Sprint 3."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class EfficiencyQualityPoint:
    """One domain/method point in the training-efficiency analysis."""

    domain: str
    method: str

    target_reach_count: int
    target_reach_total: int

    trainable_parameters: float
    parameter_reduction_percent: float

    step_reduction_percent_mean: float | None
    step_reduction_percent_std: float | None

    test_mse_mean: float
    test_mse_std: float

    test_mae_mean: float
    test_mae_std: float

    robust_ten_percent_efficiency: bool

    @property
    def target_reach_rate_percent(self) -> float:
        """Return target-reaching rate in percent."""
        return (self.target_reach_count / self.target_reach_total) * 100.0


def parameter_reduction_percent(
    reference_parameters: float,
    candidate_parameters: float,
) -> float:
    """Return candidate trainable-parameter reduction vs reference."""
    if reference_parameters <= 0:
        raise ValueError("reference_parameters must be positive")

    if candidate_parameters <= 0:
        raise ValueError("candidate_parameters must be positive")

    return (1.0 - (candidate_parameters / reference_parameters)) * 100.0


def percentage_improvement(
    reference: float,
    candidate: float,
) -> float:
    """Return positive percentage when candidate is lower/better."""
    if reference <= 0:
        raise ValueError("reference must be positive")

    return ((reference - candidate) / reference) * 100.0


def mean_curve(
    curves: Iterable[Iterable[float]],
) -> tuple[float, ...]:
    """Return an elementwise arithmetic-mean curve."""
    rows = [tuple(float(value) for value in curve) for curve in curves]

    if not rows:
        raise ValueError("curves must not be empty")

    length = len(rows[0])

    if length == 0:
        raise ValueError("curves must not be empty")

    if any(len(row) != length for row in rows):
        raise ValueError("all curves must have equal length")

    return tuple(sum(row[index] for row in rows) / len(rows) for index in range(length))
