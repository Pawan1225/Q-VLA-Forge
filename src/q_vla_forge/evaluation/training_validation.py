"""Three-seed training-efficiency validation utilities."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class MeanStd:
    """Arithmetic mean and sample standard deviation."""

    mean: float
    std: float
    n: int


@dataclass(frozen=True)
class ReachSummary:
    """Target-reaching statistics across deterministic seeds."""

    reached: int
    total: int

    @property
    def rate(self) -> float:
        return self.reached / self.total

    @property
    def rate_percent(self) -> float:
        return self.rate * 100.0

    @property
    def all_reached(self) -> bool:
        return self.reached == self.total


@dataclass(frozen=True)
class MethodValidationSummary:
    """Aggregated multi-seed evidence for one domain/method."""

    domain: str
    method: str
    seeds: tuple[int, ...]

    target_reach: ReachSummary

    trainable_parameters: MeanStd
    effective_parameters: MeanStd

    best_validation_loss: MeanStd
    final_validation_loss: MeanStd

    test_mse: MeanStd
    test_mae: MeanStd

    epoch_to_target: MeanStd | None
    steps_to_target: MeanStd | None
    samples_to_target: MeanStd | None
    seconds_to_target: MeanStd | None

    epoch_reduction_percent: MeanStd | None
    step_reduction_percent: MeanStd | None
    sample_reduction_percent: MeanStd | None
    wall_time_reduction_percent: MeanStd | None

    robust_ten_percent_step_efficiency: bool


def mean_std(
    values: Iterable[float | int],
) -> MeanStd:
    """Return arithmetic mean and sample standard deviation."""
    numeric = [float(value) for value in values]

    if not numeric:
        raise ValueError("values must not be empty")

    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("values must be finite")

    if len(numeric) == 1:
        std = 0.0
    else:
        std = statistics.stdev(numeric)

    return MeanStd(
        mean=float(statistics.mean(numeric)),
        std=float(std),
        n=len(numeric),
    )


def optional_mean_std(
    values: Iterable[float | int | None],
) -> MeanStd | None:
    """
    Aggregate only available measurements.

    Missing target-reaching measurements remain absent rather
    than being converted to the final training budget.
    """
    available = [float(value) for value in values if value is not None]

    if not available:
        return None

    return mean_std(available)


def build_reach_summary(
    reached_values: Iterable[bool],
) -> ReachSummary:
    """Aggregate target-reaching outcomes."""
    values = list(reached_values)

    if not values:
        raise ValueError("reached_values must not be empty")

    return ReachSummary(
        reached=sum(1 for value in values if value),
        total=len(values),
    )


def robust_ten_percent_efficiency(
    *,
    reach_summary: ReachSummary,
    step_reduction: MeanStd | None,
    threshold_percent: float = 10.0,
) -> bool:
    """
    Return whether a method qualifies for the pilot-scale
    robust training-efficiency observation.

    All seeds must reach the paired FP32 target.
    """
    if threshold_percent < 0.0:
        raise ValueError("threshold_percent must not be negative")

    if not reach_summary.all_reached:
        return False

    if step_reduction is None:
        return False

    return step_reduction.mean >= threshold_percent
