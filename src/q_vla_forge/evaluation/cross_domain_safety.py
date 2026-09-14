"""Cross-domain safety analysis contracts for Sprint 5.14."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CrossDomainClaimStatus(str, Enum):
    SUPPORTED = "supported"
    SUPPORTED_WITH_LIMITATION = "supported_with_limitation"
    NOT_SUPPORTED = "not_supported"


class ComparisonDirection(str, Enum):
    IMPROVED = "improved"
    DEGRADED = "degraded"
    TIE = "tie"


@dataclass(frozen=True)
class SafetyArchitectureReuse:
    shared_safety_interface: bool
    shared_action_dimension: bool
    shared_decision_contract: bool
    shared_metric_schema: bool
    shared_seed_protocol: bool
    shared_robustness_harness: bool

    domain_specific_constraints: bool
    domain_specific_clipping: bool
    domain_specific_predictor: bool
    domain_specific_lyapunov: bool

    same_policy_weights: bool
    transfer_tested: bool
    universal_controller_supported: bool


@dataclass(frozen=True)
class DomainSafetySummary:
    domain: str
    method: str
    regime: str

    violation_metric: float | None
    violation_delta_from_reference: float | None

    reward_delta_from_reference: float | None
    success_delta_from_reference: float | None

    intervention_rate: float | None
    recovery_rate: float | None


@dataclass(frozen=True)
class CrossDomainSafetyComparison:
    metric: str

    driving_value: float | None
    robotics_value: float | None

    driving_direction: ComparisonDirection | None
    robotics_direction: ComparisonDirection | None

    direction_consistent: bool | None


@dataclass(frozen=True)
class CrossDomainConclusion:
    conclusion_id: str
    supported: bool
    status: CrossDomainClaimStatus
    driving_evidence: str
    robotics_evidence: str
    limitation: str


def relative_violation_reduction(
    baseline: float,
    filtered: float,
) -> float | None:
    """Relative reduction against a positive domain-local baseline."""
    if baseline == 0.0:
        return None

    return (baseline - filtered) / baseline


def comparison_direction(
    delta: float,
    *,
    tolerance: float = 1e-12,
) -> ComparisonDirection:
    """Classify signed delta while preserving exact ties."""
    if abs(delta) <= tolerance:
        return ComparisonDirection.TIE

    if delta < 0.0:
        return ComparisonDirection.IMPROVED

    return ComparisonDirection.DEGRADED


def directions_consistent(
    driving: ComparisonDirection | None,
    robotics: ComparisonDirection | None,
) -> bool | None:
    """Compare domain directions without hiding reversals."""
    if driving is None or robotics is None:
        return None

    return driving == robotics


def exact_count_from_rate(
    rate: float,
    total_steps: int,
    *,
    tolerance: float = 1e-9,
) -> int:
    """Recover an integer event count from a frozen rate and denominator."""
    if total_steps < 0:
        raise ValueError("total_steps must be non-negative")

    if rate < 0.0:
        raise ValueError("rate must be non-negative")

    raw_count = rate * total_steps

    count = round(raw_count)

    if abs(raw_count - count) > tolerance:
        raise ValueError(
            "rate does not reconstruct an exact integer count: "
            f"rate={rate}, total_steps={total_steps}, "
            f"raw_count={raw_count}"
        )

    return int(count)
