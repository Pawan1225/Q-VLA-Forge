"""Matched-budget classical-vs-QML ablation utilities."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PairedAblationRecord:
    """One seed-level matched-classical versus QML comparison."""

    domain: str
    seed: int

    matched_actor_parameters: int
    qml_actor_parameters: int

    matched_target_reached: bool
    qml_target_reached: bool

    matched_normalized_auc: float
    qml_normalized_auc: float
    normalized_auc_delta: float

    matched_best_progress: float
    qml_best_progress: float
    best_progress_delta: float

    matched_final_progress: float
    qml_final_progress: float
    final_progress_delta: float


def paired_delta(
    *,
    matched_value: float,
    qml_value: float,
) -> float:
    """Return matched-classical minus QML."""

    return float(matched_value - qml_value)


def paired_win_counts(
    deltas: Sequence[float],
) -> dict[str, int]:
    """Count matched wins, QML wins, and exact ties."""

    return {
        "matched_classical_wins": sum(value > 0.0 for value in deltas),
        "qml_wins": sum(value < 0.0 for value in deltas),
        "ties": sum(value == 0.0 for value in deltas),
    }


def sample_mean_sd(
    values: Sequence[float],
) -> tuple[float, float]:
    """Return arithmetic mean and sample standard deviation."""

    if not values:
        raise ValueError("values cannot be empty")

    return (
        float(statistics.mean(values)),
        (float(statistics.stdev(values)) if len(values) > 1 else 0.0),
    )


def validate_parameter_match(
    *,
    matched_actor_parameters: int,
    qml_actor_parameters: int,
) -> None:
    """Fail if the matched-budget ablation is not actor-parameter matched."""

    if matched_actor_parameters != qml_actor_parameters:
        raise RuntimeError("Ablation is not parameter matched")


def record_to_dict(
    record: PairedAblationRecord,
) -> dict[str, Any]:
    """Convert one paired ablation record to a serializable dictionary."""

    return asdict(record)
