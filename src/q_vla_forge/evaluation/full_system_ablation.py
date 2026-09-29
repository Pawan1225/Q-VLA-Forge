"""Sprint 7.6 full-system ablation contracts and evidence controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

FACTORS = (
    "compression",
    "qml",
    "safety",
)

CONFIGURATIONS = {
    "baseline": (
        False,
        False,
        False,
    ),
    "a": (
        True,
        False,
        False,
    ),
    "b": (
        False,
        True,
        False,
    ),
    "c": (
        False,
        False,
        True,
    ),
    "d": (
        True,
        True,
        False,
    ),
    "e": (
        True,
        False,
        True,
    ),
    "f": (
        False,
        True,
        True,
    ),
    "full": (
        True,
        True,
        True,
    ),
}

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

EVIDENCE_STATUSES = (
    "direct",
    "component_only",
    "not_evaluated",
)

COMPONENT_EVIDENCE = {
    "compression": "Sprint 7.3",
    "qml": "Sprint 7.4",
    "safety": "Sprint 7.5",
}

COMPONENT_SOURCE_ARTIFACTS = {
    "compression": (
        "results/final-validation/" "compression-ablation/" "compression-ablation.json"
    ),
    "qml": ("results/final-validation/" "qml-ablation/" "qml-ablation.json"),
    "safety": ("results/final-validation/" "safety-ablation/" "safety-ablation.json"),
}

INTEGRATION_BOUNDARY_SOURCES = (
    ("results/final-validation/claims/" "final-claim-registry.json"),
    ("results/safety/final/" "sprint7-handoff.json"),
)

PHASE1_STATUS = "evaluated_component_evidence"

PHASE2_STATUS = "candidate_integrated_validation"

ALLOW_SYNTHETIC_METRIC_COMPOSITION = False

INTERACTION_EFFECTS_ESTIMABLE_WITH_COMPONENT_ONLY = False

FULL_SYSTEM_DIRECT_EXECUTION_FOUND = False

FULL_SYSTEM_PHASE1_CLASSIFICATION = "component_only"

FULL_SYSTEM_PHASE1_NOTE = (
    "Phase 1 independently evaluated the relevant component "
    "pathways, but no matched end-to-end Compression x QML x "
    "Safety factorial pipeline was executed. Sprint 4 RL/QML "
    "policies consumed compact environment state rather than "
    "the Sprint 1 shared VLA latent representation. Larger "
    "end-to-end integration remains a Phase 2 objective."
)

CLAIM_CONTROLS = {
    "quantum_advantage_claim_allowed": False,
    "full_system_superiority_claim_allowed": False,
    "cross_domain_zero_shot_claim_allowed": False,
    "production_readiness_claim_allowed": False,
    "integrated_factorial_validation_claim_allowed": False,
    "interaction_effect_claim_allowed_without_direct_evidence": False,
}

END_TO_END_METRIC_NAMES = (
    "reward",
    "mse",
    "latency",
    "success",
    "success_rate",
    "violation_rate",
)


@dataclass(frozen=True)
class FullSystemConfiguration:
    """One factorial configuration/domain evidence classification."""

    name: str

    compression: bool
    qml: bool
    safety: bool

    domain: str
    evidence_status: str

    source_artifacts: tuple[str, ...]

    notes: str | None = None

    def __post_init__(self) -> None:
        validate_configuration_name(self.name)
        validate_domain(self.domain)
        validate_evidence_status(self.evidence_status)

        expected = CONFIGURATIONS[self.name]

        observed = (
            self.compression,
            self.qml,
            self.safety,
        )

        if observed != expected:
            raise ValueError(
                "Factor assignment does not match "
                f"configuration {self.name!r}: "
                f"expected {expected}, got {observed}."
            )

        validate_provenance_for_status(
            evidence_status=(self.evidence_status),
            source_artifacts=(self.source_artifacts),
        )

    @property
    def can_report_end_to_end_metrics(
        self,
    ) -> bool:
        """Return whether measured integrated metrics may be reported."""

        return can_report_end_to_end_metrics(self.evidence_status)

    @property
    def phase_status(
        self,
    ) -> str:
        """Return Phase 1/Phase 2 interpretation."""

        if self.evidence_status == "direct":
            return PHASE1_STATUS

        return PHASE2_STATUS

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Serialize the configuration."""

        payload = asdict(self)

        payload["can_report_end_to_end_metrics"] = self.can_report_end_to_end_metrics

        payload["phase_status"] = self.phase_status

        return payload


def validate_configuration_name(
    name: str,
) -> None:
    """Validate the frozen factorial configuration name."""

    if name not in CONFIGURATIONS:
        raise ValueError("Unknown full-system configuration: " f"{name!r}.")


def validate_domain(
    domain: str,
) -> None:
    """Validate the frozen Sprint 7.6 domain."""

    if domain not in DOMAINS:
        raise ValueError("Unsupported full-system domain: " f"{domain!r}.")


def validate_evidence_status(
    evidence_status: str,
) -> None:
    """Validate an evidence classification."""

    if evidence_status not in EVIDENCE_STATUSES:
        raise ValueError("Unsupported evidence status: " f"{evidence_status!r}.")


def validate_provenance_for_status(
    *,
    evidence_status: str,
    source_artifacts: tuple[str, ...],
) -> None:
    """Require provenance whenever evidence is claimed."""

    validate_evidence_status(evidence_status)

    if (
        evidence_status
        in (
            "direct",
            "component_only",
        )
        and not source_artifacts
    ):
        raise ValueError(
            "DIRECT and COMPONENT_ONLY evidence " "must retain source provenance."
        )

    for source in source_artifacts:
        if not source.strip():
            raise ValueError("Source artifact paths cannot be blank.")


def configuration_factors(
    name: str,
) -> tuple[
    bool,
    bool,
    bool,
]:
    """Return the frozen factor tuple for a configuration."""

    validate_configuration_name(name)

    return CONFIGURATIONS[name]


def can_report_end_to_end_metrics(
    evidence_status: str,
) -> bool:
    """Allow integrated metrics only for direct evidence."""

    validate_evidence_status(evidence_status)

    return evidence_status == "direct"


def synthetic_metric_composition_allowed() -> bool:
    """Return the frozen synthetic-composition control."""

    return ALLOW_SYNTHETIC_METRIC_COMPOSITION


def interaction_effects_estimable(
    evidence_status: str,
) -> bool:
    """Return whether factorial interaction effects may be estimated."""

    validate_evidence_status(evidence_status)

    return evidence_status == "direct"


def interaction_effects_estimable_from_component_only() -> bool:
    """Explicitly block interaction inference from component evidence."""

    return INTERACTION_EFFECTS_ESTIMABLE_WITH_COMPONENT_ONLY


def validate_component_evidence_registry() -> None:
    """Validate the frozen component-evidence registry."""

    if set(COMPONENT_EVIDENCE) != set(FACTORS):
        raise ValueError(
            "Component evidence registry must cover "
            "exactly compression, qml, and safety."
        )

    for sprint in COMPONENT_EVIDENCE.values():
        if not sprint.strip():
            raise ValueError(
                "Component evidence registry contains " "an empty sprint reference."
            )


def phase1_component_evidence_is_integrated_validation() -> bool:
    """Prevent component evidence from becoming integrated validation."""

    return False


def phase2_requires_integrated_validation() -> bool:
    """Return whether Phase 2 requires matched integrated execution."""

    return True


def full_configuration_validated(
    evidence_status: str,
) -> bool:
    """Return whether the full 111 configuration has direct validation."""

    return can_report_end_to_end_metrics(evidence_status)


def end_to_end_metrics_allowed_for_record(
    evidence_status: str,
    metrics: dict[
        str,
        Any,
    ],
) -> bool:
    """Validate that integrated metrics respect evidence status."""

    validate_evidence_status(evidence_status)

    reported_metrics = {
        key
        for key, value in metrics.items()
        if (key in END_TO_END_METRIC_NAMES and value is not None)
    }

    if not reported_metrics:
        return True

    return can_report_end_to_end_metrics(evidence_status)


def validate_end_to_end_metrics(
    evidence_status: str,
    metrics: dict[
        str,
        Any,
    ],
) -> None:
    """Reject synthetic end-to-end metrics for non-direct evidence."""

    if not end_to_end_metrics_allowed_for_record(
        evidence_status,
        metrics,
    ):
        raise ValueError(
            "End-to-end metrics cannot be reported "
            "for COMPONENT_ONLY or NOT_EVALUATED evidence."
        )


def claim_controls() -> dict[str, bool]:
    """Return a copy of the Sprint 7.6 claim controls."""

    return dict(CLAIM_CONTROLS)


def component_sources_for_configuration(
    name: str,
) -> tuple[str, ...]:
    """Return component and integration-boundary provenance."""

    validate_configuration_name(name)

    compression, qml, safety = CONFIGURATIONS[name]

    sources: list[str] = list(INTEGRATION_BOUNDARY_SOURCES)

    if compression:
        sources.append(COMPONENT_SOURCE_ARTIFACTS["compression"])

    if qml:
        sources.append(COMPONENT_SOURCE_ARTIFACTS["qml"])

    if safety:
        sources.append(COMPONENT_SOURCE_ARTIFACTS["safety"])

    if name == "baseline":
        sources.extend(
            [
                COMPONENT_SOURCE_ARTIFACTS["compression"],
                COMPONENT_SOURCE_ARTIFACTS["qml"],
                COMPONENT_SOURCE_ARTIFACTS["safety"],
            ]
        )

    return tuple(dict.fromkeys(sources))


def phase1_evidence_status_for_configuration(
    name: str,
    domain: str,
) -> str:
    """Return the frozen Phase 1 full-system evidence status."""

    validate_configuration_name(name)
    validate_domain(domain)

    if FULL_SYSTEM_DIRECT_EXECUTION_FOUND:
        return "direct"

    return FULL_SYSTEM_PHASE1_CLASSIFICATION


def build_phase1_evidence_matrix() -> tuple[
    FullSystemConfiguration,
    ...,
]:
    """Build the frozen 8 x 2 Phase 1 evidence matrix."""

    records: list[FullSystemConfiguration] = []

    for domain in DOMAINS:
        for name, factors in CONFIGURATIONS.items():
            (
                compression,
                qml,
                safety,
            ) = factors

            evidence_status = phase1_evidence_status_for_configuration(
                name,
                domain,
            )

            records.append(
                FullSystemConfiguration(
                    name=name,
                    compression=compression,
                    qml=qml,
                    safety=safety,
                    domain=domain,
                    evidence_status=(evidence_status),
                    source_artifacts=(component_sources_for_configuration(name)),
                    notes=(FULL_SYSTEM_PHASE1_NOTE),
                )
            )

    if len(records) != 16:
        raise ValueError("Expected exactly 16 full-system " "evidence classifications.")

    observed = {
        (
            record.domain,
            record.name,
        )
        for record in records
    }

    expected = {
        (
            domain,
            name,
        )
        for domain in DOMAINS
        for name in CONFIGURATIONS
    }

    if observed != expected:
        raise ValueError("Full-system evidence matrix is " "incomplete or duplicated.")

    return tuple(records)


def direct_configuration_count() -> int:
    """Count directly executed factorial configurations."""

    return sum(
        record.evidence_status == "direct" for record in build_phase1_evidence_matrix()
    )


def component_only_configuration_count() -> int:
    """Count component-only factorial classifications."""

    return sum(
        record.evidence_status == "component_only"
        for record in build_phase1_evidence_matrix()
    )


def not_evaluated_configuration_count() -> int:
    """Count configurations lacking relevant Phase 1 evidence."""

    return sum(
        record.evidence_status == "not_evaluated"
        for record in build_phase1_evidence_matrix()
    )
