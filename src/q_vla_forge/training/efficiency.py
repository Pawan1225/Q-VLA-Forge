"""Contracts and metrics for Sprint 3 training-efficiency experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EpochEfficiencyRecord:
    """Training and validation measurements for one completed epoch."""

    epoch: int
    train_loss: float
    validation_loss: float
    learning_rate: float
    epoch_seconds: float
    cumulative_seconds: float
    optimizer_steps: int
    samples_processed: int

    def __post_init__(self) -> None:
        if self.epoch < 1:
            raise ValueError("epoch must be >= 1")

        if self.optimizer_steps < 0:
            raise ValueError("optimizer_steps must be >= 0")

        if self.samples_processed < 0:
            raise ValueError("samples_processed must be >= 0")

        numeric_values = (
            self.train_loss,
            self.validation_loss,
            self.learning_rate,
            self.epoch_seconds,
            self.cumulative_seconds,
        )

        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("epoch metrics must be finite")

        if self.train_loss < 0.0:
            raise ValueError("train_loss must be >= 0")

        if self.validation_loss < 0.0:
            raise ValueError("validation_loss must be >= 0")

        if self.learning_rate < 0.0:
            raise ValueError("learning_rate must be >= 0")

        if self.epoch_seconds < 0.0:
            raise ValueError("epoch_seconds must be >= 0")

        if self.cumulative_seconds < 0.0:
            raise ValueError("cumulative_seconds must be >= 0")


@dataclass(frozen=True)
class TargetDefinition:
    """Paired FP32-derived convergence target."""

    reference_best_validation_loss: float
    tolerance_fraction: float
    target_validation_loss: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.reference_best_validation_loss):
            raise ValueError("reference_best_validation_loss must be finite")

        if self.reference_best_validation_loss < 0.0:
            raise ValueError("reference_best_validation_loss must be >= 0")

        if not math.isfinite(self.tolerance_fraction):
            raise ValueError("tolerance_fraction must be finite")

        if self.tolerance_fraction < 0.0:
            raise ValueError("tolerance_fraction must be >= 0")

        if not math.isfinite(self.target_validation_loss):
            raise ValueError("target_validation_loss must be finite")

        if self.target_validation_loss < 0.0:
            raise ValueError("target_validation_loss must be >= 0")


@dataclass(frozen=True)
class TargetReachResult:
    """First point at which a training run reaches its target."""

    reached_target: bool
    epoch_to_target: int | None
    steps_to_target: int | None
    samples_to_target: int | None
    seconds_to_target: float | None


@dataclass(frozen=True)
class TrainingEfficiencySummary:
    """Summary of one complete training-efficiency run."""

    domain: str
    method: str
    seed: int

    trainable_parameters: int
    effective_parameters: int

    total_training_seconds: float
    mean_epoch_seconds: float

    best_validation_loss: float
    best_epoch: int
    final_validation_loss: float

    target: TargetDefinition
    target_reach: TargetReachResult

    history: tuple[EpochEfficiencyRecord, ...]


@dataclass(frozen=True)
class EfficiencyComparison:
    """Paired FP32-versus-candidate efficiency comparison."""

    domain: str
    seed: int
    reference_method: str
    candidate_method: str

    reference_reached_target: bool
    candidate_reached_target: bool

    epoch_reduction_percent: float | None
    step_reduction_percent: float | None
    sample_reduction_percent: float | None
    wall_time_reduction_percent: float | None


def build_target_definition(
    reference_best_validation_loss: float,
    *,
    tolerance_fraction: float = 0.05,
) -> TargetDefinition:
    """Create a convergence target from paired FP32 validation quality."""
    if not math.isfinite(reference_best_validation_loss):
        raise ValueError("reference_best_validation_loss must be finite")

    if reference_best_validation_loss < 0.0:
        raise ValueError("reference_best_validation_loss must be >= 0")

    if not math.isfinite(tolerance_fraction):
        raise ValueError("tolerance_fraction must be finite")

    if tolerance_fraction < 0.0:
        raise ValueError("tolerance_fraction must be >= 0")

    target = reference_best_validation_loss * (1.0 + tolerance_fraction)

    return TargetDefinition(
        reference_best_validation_loss=reference_best_validation_loss,
        tolerance_fraction=tolerance_fraction,
        target_validation_loss=target,
    )


def find_target_reach(
    history: Sequence[EpochEfficiencyRecord],
    target_validation_loss: float,
) -> TargetReachResult:
    """Return the first epoch that reaches the validation target."""
    if not math.isfinite(target_validation_loss):
        raise ValueError("target_validation_loss must be finite")

    if target_validation_loss < 0.0:
        raise ValueError("target_validation_loss must be >= 0")

    if not history:
        raise ValueError("history must not be empty")

    for record in history:
        if record.validation_loss <= target_validation_loss:
            return TargetReachResult(
                reached_target=True,
                epoch_to_target=record.epoch,
                steps_to_target=record.optimizer_steps,
                samples_to_target=record.samples_processed,
                seconds_to_target=record.cumulative_seconds,
            )

    return TargetReachResult(
        reached_target=False,
        epoch_to_target=None,
        steps_to_target=None,
        samples_to_target=None,
        seconds_to_target=None,
    )


def percentage_reduction(
    reference: float,
    candidate: float,
) -> float:
    """Return positive percentage when candidate uses less than reference."""
    if not math.isfinite(reference) or not math.isfinite(candidate):
        raise ValueError("comparison values must be finite")

    if reference <= 0.0:
        raise ValueError("reference must be > 0")

    return (reference - candidate) / reference * 100.0


def compare_target_efficiency(
    *,
    domain: str,
    seed: int,
    reference_method: str,
    candidate_method: str,
    reference: TargetReachResult,
    candidate: TargetReachResult,
) -> EfficiencyComparison:
    """Compare target-reaching efficiency against the FP32 reference."""
    if not domain:
        raise ValueError("domain must not be empty")

    if not reference_method:
        raise ValueError("reference_method must not be empty")

    if not candidate_method:
        raise ValueError("candidate_method must not be empty")

    if not reference.reached_target or not candidate.reached_target:
        return EfficiencyComparison(
            domain=domain,
            seed=seed,
            reference_method=reference_method,
            candidate_method=candidate_method,
            reference_reached_target=reference.reached_target,
            candidate_reached_target=candidate.reached_target,
            epoch_reduction_percent=None,
            step_reduction_percent=None,
            sample_reduction_percent=None,
            wall_time_reduction_percent=None,
        )

    if (
        reference.epoch_to_target is None
        or reference.steps_to_target is None
        or reference.samples_to_target is None
        or reference.seconds_to_target is None
        or candidate.epoch_to_target is None
        or candidate.steps_to_target is None
        or candidate.samples_to_target is None
        or candidate.seconds_to_target is None
    ):
        raise ValueError("reached-target results must contain target metrics")

    return EfficiencyComparison(
        domain=domain,
        seed=seed,
        reference_method=reference_method,
        candidate_method=candidate_method,
        reference_reached_target=True,
        candidate_reached_target=True,
        epoch_reduction_percent=percentage_reduction(
            float(reference.epoch_to_target),
            float(candidate.epoch_to_target),
        ),
        step_reduction_percent=percentage_reduction(
            float(reference.steps_to_target),
            float(candidate.steps_to_target),
        ),
        sample_reduction_percent=percentage_reduction(
            float(reference.samples_to_target),
            float(candidate.samples_to_target),
        ),
        wall_time_reduction_percent=percentage_reduction(
            reference.seconds_to_target,
            candidate.seconds_to_target,
        ),
    )


def training_efficiency_summary_to_dict(
    summary: TrainingEfficiencySummary,
) -> dict[str, Any]:
    """Convert a summary into a JSON-serializable dictionary."""
    return asdict(summary)
