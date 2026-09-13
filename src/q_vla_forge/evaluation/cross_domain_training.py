"""Cross-domain training-efficiency comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainMethodEvidence:
    """Training evidence for one domain and structured method."""

    domain: str
    method: str

    target_reached: int
    target_total: int

    parameter_reduction_percent: float

    step_reduction_percent_mean: float | None
    step_reduction_percent_std: float | None

    test_mse_mean: float
    test_mse_std: float

    robust_ten_percent_efficiency: bool

    @property
    def all_targets_reached(self) -> bool:
        """Return whether all deterministic seeds reached target."""
        return self.target_reached == self.target_total


@dataclass(frozen=True)
class CrossDomainMethodComparison:
    """Cross-domain evidence for one structured method."""

    method: str

    driving: DomainMethodEvidence
    robotics: DomainMethodEvidence

    target_consistency: str
    efficiency_consistency: str

    parameter_reduction_difference_percent: float

    robust_cross_domain_efficiency: bool


def target_consistency_label(
    driving: DomainMethodEvidence,
    robotics: DomainMethodEvidence,
) -> str:
    """Classify target-reaching consistency across domains."""
    driving_all = driving.all_targets_reached

    robotics_all = robotics.all_targets_reached

    if driving_all and robotics_all:
        return "strong"

    if driving_all != robotics_all:
        return "mixed"

    return "weak"


def efficiency_consistency_label(
    driving: DomainMethodEvidence,
    robotics: DomainMethodEvidence,
) -> str:
    """Classify robust >=10% efficiency consistency."""
    driving_pass = driving.robust_ten_percent_efficiency

    robotics_pass = robotics.robust_ten_percent_efficiency

    if driving_pass and robotics_pass:
        return "strong"

    if driving_pass != robotics_pass:
        return "mixed"

    return "none"


def build_cross_domain_comparison(
    driving: DomainMethodEvidence,
    robotics: DomainMethodEvidence,
) -> CrossDomainMethodComparison:
    """Create one cross-domain comparison."""
    if driving.method != robotics.method:
        raise ValueError("driving and robotics methods must match")

    if driving.domain != "autonomous_driving":
        raise ValueError("invalid driving domain")

    if robotics.domain != "robotics":
        raise ValueError("invalid robotics domain")

    parameter_difference = abs(
        driving.parameter_reduction_percent - robotics.parameter_reduction_percent
    )

    robust_cross_domain = (
        driving.robust_ten_percent_efficiency and robotics.robust_ten_percent_efficiency
    )

    return CrossDomainMethodComparison(
        method=driving.method,
        driving=driving,
        robotics=robotics,
        target_consistency=(
            target_consistency_label(
                driving,
                robotics,
            )
        ),
        efficiency_consistency=(
            efficiency_consistency_label(
                driving,
                robotics,
            )
        ),
        parameter_reduction_difference_percent=(parameter_difference),
        robust_cross_domain_efficiency=(robust_cross_domain),
    )
