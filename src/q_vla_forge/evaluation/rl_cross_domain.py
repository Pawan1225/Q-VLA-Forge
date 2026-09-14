"""Cross-domain RL comparison utilities for Sprint 4.13."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CrossDomainMethodRecord:
    """One domain-level aggregate method record."""

    domain: str
    method: str

    actor_parameters: int

    target_reach_count: int
    target_total: int

    normalized_auc_mean: float
    normalized_auc_sd: float

    best_progress_mean: float
    best_progress_sd: float

    final_progress_mean: float
    final_progress_sd: float


@dataclass(frozen=True)
class MatchedRepresentationComparison:
    """One domain-level matched classical versus QML comparison."""

    domain: str

    matched_actor_parameters: int
    qml_actor_parameters: int

    matched_auc_mean: float
    qml_auc_mean: float
    matched_minus_qml_auc: float

    direction: str


def representation_direction(
    *,
    matched_value: float,
    qml_value: float,
    tolerance: float = 1e-12,
) -> str:
    """Return the direction of a matched classical-vs-QML comparison."""

    if tolerance < 0.0:
        raise ValueError("tolerance cannot be negative")

    difference = matched_value - qml_value

    if math.isclose(
        difference,
        0.0,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        return "tie"

    if difference > 0.0:
        return "matched_classical"

    return "hybrid_qml"


def parameter_reduction_percent(
    *,
    full_actor_parameters: int,
    compact_actor_parameters: int,
) -> float:
    """Return actor-parameter reduction relative to the full actor."""

    if full_actor_parameters <= 0:
        raise ValueError("full actor parameters must be positive")

    if compact_actor_parameters <= 0:
        raise ValueError("compact actor parameters must be positive")

    if compact_actor_parameters > full_actor_parameters:
        raise ValueError("compact actor cannot exceed full actor size")

    return float(
        100.0
        * (full_actor_parameters - compact_actor_parameters)
        / full_actor_parameters
    )


def record_to_dict(
    record: CrossDomainMethodRecord,
) -> dict[str, Any]:
    """Convert one method record to a serializable dictionary."""

    return asdict(record)


def matched_comparison_to_dict(
    comparison: MatchedRepresentationComparison,
) -> dict[str, Any]:
    """Convert one matched representation comparison to a dictionary."""

    return asdict(comparison)
