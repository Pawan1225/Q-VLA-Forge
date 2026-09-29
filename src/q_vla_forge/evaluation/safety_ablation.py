"""Sprint 7.5 safety-ablation contract and validation helpers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REQUIRED_SEEDS = (
    42,
    123,
    456,
)

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

SAFETY_METHODS = (
    "none",
    "clipping",
    "lyapunov",
)

REFERENCE_METHOD = "none"

METHOD_ROLES = {
    "none": "REFERENCE",
    "clipping": "EVALUATED",
    "lyapunov": "EVALUATED",
}

METHOD_DESCRIPTIONS = {
    "none": ("Raw policy action under the frozen " "safety evaluation."),
    "clipping": ("Heuristic constraint/action clipping."),
    "lyapunov": ("Classical Lyapunov-guided safety filter."),
}

LYAPUNOV_METHOD_CLASSIFICATION = "classical"

PRIVILEGED_STATE_LIMITATION = (
    "The safety layer operated on true simulator state "
    "while the policy could receive perturbed observations. "
    "The results therefore apply to the frozen Phase 1 "
    "simulator contract and do not establish estimated-state "
    "or real-world sensing robustness."
)

CLAIM_CONTROLS = {
    "formal_stability_guarantee_allowed": False,
    "formal_lyapunov_stability_proof_allowed": False,
    "forward_invariance_guarantee_allowed": False,
    "worst_case_safety_guarantee_allowed": False,
    "iso_26262_certification_claim_allowed": False,
    "production_safety_claim_allowed": False,
    "real_world_safety_claim_allowed": False,
    "universal_safety_controller_claim_allowed": False,
    "quantum_safety_advantage_claim_allowed": False,
    "guaranteed_zero_violations_claim_allowed": False,
}

BASELINE_SUMMARY_SOURCE = (
    "results/safety/baseline/" "sprint5-no-filter-safety-summary.json"
)

CLIPPING_SUMMARY_SOURCE = (
    "results/safety/clipping/" "sprint5-clipping-safety-summary.json"
)

LYAPUNOV_SUMMARY_SOURCES = {
    "autonomous_driving": (
        "results/safety/lyapunov-driving/" "sprint5-driving-lyapunov-summary.json"
    ),
    "robotics": (
        "results/safety/lyapunov-robotics/" "sprint5-robotics-lyapunov-summary.json"
    ),
}

FORBIDDEN_PRIMARY_SOURCE_FRAGMENTS = (
    "gaussian-robustness",
    "structured-state-robustness",
    "action-robustness",
)


@dataclass(frozen=True)
class SafetyAblationRecord:
    """Canonical clean-safety ablation record."""

    domain: str
    method: str

    violation_rate_mean: float
    violation_rate_sample_std: float

    reward_mean: float
    reward_sample_std: float

    success_rate_mean: float
    success_rate_sample_std: float

    seeds: tuple[int, ...]
    source_artifacts: tuple[str, ...]

    intervention_rate_mean: float | None = None
    intervention_rate_sample_std: float | None = None

    safe_episode_rate_mean: float | None = None
    safe_episode_rate_sample_std: float | None = None

    def __post_init__(self) -> None:
        validate_domain(self.domain)
        validate_method(self.method)
        validate_required_seeds(self.seeds)

        validate_nonnegative_finite(
            self.violation_rate_mean,
            name="violation_rate_mean",
        )
        validate_nonnegative_finite(
            self.violation_rate_sample_std,
            name="violation_rate_sample_std",
        )

        validate_finite(
            self.reward_mean,
            name="reward_mean",
        )
        validate_nonnegative_finite(
            self.reward_sample_std,
            name="reward_sample_std",
        )

        validate_rate(
            self.success_rate_mean,
            name="success_rate_mean",
        )
        validate_nonnegative_finite(
            self.success_rate_sample_std,
            name="success_rate_sample_std",
        )

        validate_optional_rate(
            self.intervention_rate_mean,
            name="intervention_rate_mean",
        )
        validate_optional_nonnegative_finite(
            self.intervention_rate_sample_std,
            name="intervention_rate_sample_std",
        )

        validate_optional_rate(
            self.safe_episode_rate_mean,
            name="safe_episode_rate_mean",
        )
        validate_optional_nonnegative_finite(
            self.safe_episode_rate_sample_std,
            name="safe_episode_rate_sample_std",
        )

        validate_source_provenance(self.source_artifacts)

    @property
    def role(self) -> str:
        """Return descriptive comparison role."""

        return method_role(self.method)

    @property
    def lyapunov_is_classical(
        self,
    ) -> bool:
        """Identify the Sprint 5 Lyapunov method correctly."""

        return (
            self.method != "lyapunov" or LYAPUNOV_METHOD_CLASSIFICATION == "classical"
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert the frozen record to a dictionary."""

        payload = asdict(self)

        payload["role"] = self.role

        payload["lyapunov_method_classification"] = (
            LYAPUNOV_METHOD_CLASSIFICATION if self.method == "lyapunov" else None
        )

        return payload


@dataclass(frozen=True)
class LyapunovMechanismObservation:
    """Domain-specific clean-evaluation Lyapunov mechanism observation."""

    domain: str
    lyapunov_decrease_interventions_observed: bool
    interpretation: str
    source_artifacts: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_domain(self.domain)

        if not self.interpretation.strip():
            raise ValueError("Lyapunov mechanism interpretation is required.")

        validate_source_provenance(self.source_artifacts)


@dataclass(frozen=True)
class CanonicalSafetyAblation:
    """Canonical Sprint 7.5 clean-safety evidence package."""

    records: tuple[SafetyAblationRecord, ...]

    lyapunov_mechanism_observations: tuple[LyapunovMechanismObservation, ...]


def validate_domain(
    domain: str,
) -> None:
    """Validate the frozen Sprint 7.5 domain."""

    if domain not in DOMAINS:
        raise ValueError(f"Unsupported safety-ablation domain: " f"{domain!r}.")


def validate_method(
    method: str,
) -> None:
    """Validate a frozen clean-safety method."""

    if method not in SAFETY_METHODS:
        raise ValueError(f"Unsupported safety method: " f"{method!r}.")


def validate_required_seeds(
    seeds: tuple[int, ...],
) -> None:
    """Require exactly the three frozen Phase 1 seeds."""

    if seeds != REQUIRED_SEEDS:
        raise ValueError(
            "Safety-ablation seeds must be exactly " f"{REQUIRED_SEEDS}; got {seeds}."
        )


def validate_finite(
    value: float,
    *,
    name: str,
) -> None:
    """Require a finite numeric value."""

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def validate_nonnegative_finite(
    value: float,
    *,
    name: str,
) -> None:
    """Require a finite non-negative numeric value."""

    validate_finite(
        value,
        name=name,
    )

    if value < 0.0:
        raise ValueError(f"{name} cannot be negative.")


def validate_rate(
    value: float,
    *,
    name: str,
) -> None:
    """Require a finite rate in [0, 1]."""

    validate_finite(
        value,
        name=name,
    )

    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must lie in [0, 1].")


def validate_optional_rate(
    value: float | None,
    *,
    name: str,
) -> None:
    """Validate an optional finite rate."""

    if value is None:
        return

    validate_rate(
        value,
        name=name,
    )


def validate_optional_nonnegative_finite(
    value: float | None,
    *,
    name: str,
) -> None:
    """Validate an optional non-negative quantity."""

    if value is None:
        return

    validate_nonnegative_finite(
        value,
        name=name,
    )


def validate_source_provenance(
    source_artifacts: tuple[str, ...],
) -> None:
    """Require explicit Sprint 5 source provenance."""

    if not source_artifacts:
        raise ValueError("At least one source artifact is required.")

    for source in source_artifacts:
        if not source.strip():
            raise ValueError("Source artifact paths cannot be blank.")


def method_role(
    method: str,
) -> str:
    """Return REFERENCE/EVALUATED classification."""

    validate_method(method)

    return METHOD_ROLES[method]


def method_description(
    method: str,
) -> str:
    """Return the locked safety-method semantics."""

    validate_method(method)

    return METHOD_DESCRIPTIONS[method]


def relative_reduction_percent(
    reference: float,
    candidate: float,
) -> float | None:
    """Return relative reduction, or None for zero reference."""

    validate_nonnegative_finite(
        reference,
        name="reference",
    )
    validate_nonnegative_finite(
        candidate,
        name="candidate",
    )

    if reference == 0.0:
        return None

    return 100.0 * (reference - candidate) / reference


def absolute_difference(
    reference: float,
    candidate: float,
) -> float:
    """Return candidate minus reference."""

    validate_nonnegative_finite(
        reference,
        name="reference",
    )
    validate_nonnegative_finite(
        candidate,
        name="candidate",
    )

    return candidate - reference


def zero_observed_violations(
    violation_rate: float,
) -> bool:
    """Return whether zero violations were empirically observed."""

    validate_nonnegative_finite(
        violation_rate,
        name="violation_rate",
    )

    return violation_rate == 0.0


def zero_observed_is_formal_guarantee() -> bool:
    """Explicitly prevent empirical zero from becoming a guarantee."""

    return False


def lyapunov_method_is_classical() -> bool:
    """Return the frozen Sprint 5 Lyapunov classification."""

    return LYAPUNOV_METHOD_CLASSIFICATION == "classical"


def privileged_state_limitation() -> str:
    """Return the frozen privileged-state limitation."""

    return PRIVILEGED_STATE_LIMITATION


def claim_controls() -> dict[str, bool]:
    """Return a copy of the safety claim-control registry."""

    return dict(CLAIM_CONTROLS)


def _load_json(
    root: Path,
    relative_path: str,
) -> dict[str, Any]:
    """Load a canonical JSON artifact."""

    path = root / relative_path

    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(f"Expected JSON object: " f"{relative_path}")

    return payload


def _assert_close(
    actual: float,
    expected: float,
    *,
    label: str,
) -> None:
    """Require two canonical numeric representations to agree."""

    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Canonical evidence disagreement for " f"{label}: {actual} != {expected}."
        )


def _validate_clean_source_path(
    source: str,
) -> None:
    """Keep robustness evidence outside Sprint 7.5."""

    if not source.startswith("results/safety/"):
        raise ValueError(f"Unexpected safety source: " f"{source}")

    for fragment in FORBIDDEN_PRIMARY_SOURCE_FRAGMENTS:
        if fragment in source:
            raise ValueError(
                "Robustness evidence cannot enter "
                "Sprint 7.5 primary ablation: "
                f"{source}"
            )


def _validate_clean_summary(
    payload: dict[str, Any],
    *,
    source: str,
) -> None:
    """Validate common frozen Sprint 5 summary metadata."""

    _validate_clean_source_path(source)

    if payload.get("condition") != "clean":
        raise ValueError("Expected clean safety condition: " f"{source}")

    if payload.get("new_training_performed") is not False:
        raise ValueError("Unexpected training in safety evidence: " f"{source}")

    seeds = tuple(
        int(seed)
        for seed in payload.get(
            "principal_seeds",
            (),
        )
    )

    validate_required_seeds(seeds)


def _build_none_record(
    baseline: dict[str, Any],
    *,
    domain: str,
) -> SafetyAblationRecord:
    """Build the no-safety reference record."""

    domain_payload = baseline["domains"][domain]

    violation = domain_payload["violation_step_rate"]

    reward = domain_payload["reward"]

    success = domain_payload["success"]

    return SafetyAblationRecord(
        domain=domain,
        method="none",
        violation_rate_mean=float(violation["mean"]),
        violation_rate_sample_std=float(violation["sample_sd"]),
        reward_mean=float(reward["mean_of_seed_means"]),
        reward_sample_std=float(reward["sample_sd_of_seed_means"]),
        success_rate_mean=float(success["mean_of_seed_rates"]),
        success_rate_sample_std=float(success["sample_sd_of_seed_rates"]),
        seeds=REQUIRED_SEEDS,
        source_artifacts=(BASELINE_SUMMARY_SOURCE,),
        intervention_rate_mean=None,
        intervention_rate_sample_std=None,
    )


def _build_clipping_record(
    clipping: dict[str, Any],
    *,
    domain: str,
) -> SafetyAblationRecord:
    """Build the heuristic clipping record."""

    domain_payload = clipping["domains"][domain]

    violation = domain_payload["clipping_violation_step_rate"]

    reward = domain_payload["clipping_reward"]

    success = domain_payload["clipping_success_rate"]

    intervention = domain_payload["intervention_rate"]

    return SafetyAblationRecord(
        domain=domain,
        method="clipping",
        violation_rate_mean=float(violation["mean"]),
        violation_rate_sample_std=float(violation["sample_sd"]),
        reward_mean=float(reward["mean_of_seed_means"]),
        reward_sample_std=float(reward["sample_sd_of_seed_means"]),
        success_rate_mean=float(success["mean"]),
        success_rate_sample_std=float(success["sample_sd"]),
        seeds=REQUIRED_SEEDS,
        source_artifacts=(CLIPPING_SUMMARY_SOURCE,),
        intervention_rate_mean=float(intervention["mean"]),
        intervention_rate_sample_std=float(intervention["sample_sd"]),
    )


def _build_lyapunov_record(
    summary: dict[str, Any],
    *,
    domain: str,
) -> SafetyAblationRecord:
    """Build the classical Lyapunov-guided filter record."""

    method_payload = summary["three_method_summary"]["lyapunov"]

    violation = method_payload["violation_step_rate"]

    reward = method_payload["reward"]

    success = method_payload["success"]

    intervention = method_payload["intervention_rate"]

    return SafetyAblationRecord(
        domain=domain,
        method="lyapunov",
        violation_rate_mean=float(violation["mean"]),
        violation_rate_sample_std=float(violation["sample_sd"]),
        reward_mean=float(reward["mean_of_seed_means"]),
        reward_sample_std=float(reward["sample_sd_of_seed_means"]),
        success_rate_mean=float(success["mean_of_seed_rates"]),
        success_rate_sample_std=float(success["sample_sd_of_seed_rates"]),
        seeds=REQUIRED_SEEDS,
        source_artifacts=(LYAPUNOV_SUMMARY_SOURCES[domain],),
        intervention_rate_mean=float(intervention["mean"]),
        intervention_rate_sample_std=float(intervention["sample_sd"]),
    )


def _cross_check_none_evidence(
    *,
    domain: str,
    baseline: dict[str, Any],
    clipping: dict[str, Any],
    lyapunov: dict[str, Any],
) -> None:
    """Cross-check repeated no-safety representations."""

    baseline_domain = baseline["domains"][domain]

    clipping_domain = clipping["domains"][domain]

    lyapunov_none = lyapunov["three_method_summary"]["none"]

    baseline_violation = baseline_domain["violation_step_rate"]

    baseline_reward = baseline_domain["reward"]

    baseline_success = baseline_domain["success"]

    _assert_close(
        clipping_domain["none_violation_step_rate"]["mean"],
        baseline_violation["mean"],
        label=(f"{domain}/none violation " "baseline-vs-clipping"),
    )

    _assert_close(
        clipping_domain["none_violation_step_rate"]["sample_sd"],
        baseline_violation["sample_sd"],
        label=(f"{domain}/none violation SD " "baseline-vs-clipping"),
    )

    _assert_close(
        lyapunov_none["violation_step_rate"]["mean"],
        baseline_violation["mean"],
        label=(f"{domain}/none violation " "baseline-vs-lyapunov"),
    )

    _assert_close(
        lyapunov_none["violation_step_rate"]["sample_sd"],
        baseline_violation["sample_sd"],
        label=(f"{domain}/none violation SD " "baseline-vs-lyapunov"),
    )

    _assert_close(
        clipping_domain["none_reward"]["mean_of_seed_means"],
        baseline_reward["mean_of_seed_means"],
        label=(f"{domain}/none reward " "baseline-vs-clipping"),
    )

    _assert_close(
        clipping_domain["none_reward"]["sample_sd_of_seed_means"],
        baseline_reward["sample_sd_of_seed_means"],
        label=(f"{domain}/none reward SD " "baseline-vs-clipping"),
    )

    _assert_close(
        lyapunov_none["reward"]["mean_of_seed_means"],
        baseline_reward["mean_of_seed_means"],
        label=(f"{domain}/none reward " "baseline-vs-lyapunov"),
    )

    _assert_close(
        lyapunov_none["reward"]["sample_sd_of_seed_means"],
        baseline_reward["sample_sd_of_seed_means"],
        label=(f"{domain}/none reward SD " "baseline-vs-lyapunov"),
    )

    _assert_close(
        clipping_domain["none_success_rate"]["mean"],
        baseline_success["mean_of_seed_rates"],
        label=(f"{domain}/none success " "baseline-vs-clipping"),
    )

    _assert_close(
        clipping_domain["none_success_rate"]["sample_sd"],
        baseline_success["sample_sd_of_seed_rates"],
        label=(f"{domain}/none success SD " "baseline-vs-clipping"),
    )

    _assert_close(
        lyapunov_none["success"]["mean_of_seed_rates"],
        baseline_success["mean_of_seed_rates"],
        label=(f"{domain}/none success " "baseline-vs-lyapunov"),
    )

    _assert_close(
        lyapunov_none["success"]["sample_sd_of_seed_rates"],
        baseline_success["sample_sd_of_seed_rates"],
        label=(f"{domain}/none success SD " "baseline-vs-lyapunov"),
    )


def _cross_check_clipping_evidence(
    *,
    domain: str,
    clipping: dict[str, Any],
    lyapunov: dict[str, Any],
) -> None:
    """Cross-check repeated clipping representations."""

    clipping_domain = clipping["domains"][domain]

    lyapunov_clipping = lyapunov["three_method_summary"]["clipping"]

    _assert_close(
        lyapunov_clipping["violation_step_rate"]["mean"],
        clipping_domain["clipping_violation_step_rate"]["mean"],
        label=(f"{domain}/clipping violation " "cross-check"),
    )

    _assert_close(
        lyapunov_clipping["violation_step_rate"]["sample_sd"],
        clipping_domain["clipping_violation_step_rate"]["sample_sd"],
        label=(f"{domain}/clipping violation SD " "cross-check"),
    )

    _assert_close(
        lyapunov_clipping["reward"]["mean_of_seed_means"],
        clipping_domain["clipping_reward"]["mean_of_seed_means"],
        label=(f"{domain}/clipping reward " "cross-check"),
    )

    _assert_close(
        lyapunov_clipping["reward"]["sample_sd_of_seed_means"],
        clipping_domain["clipping_reward"]["sample_sd_of_seed_means"],
        label=(f"{domain}/clipping reward SD " "cross-check"),
    )

    _assert_close(
        lyapunov_clipping["success"]["mean_of_seed_rates"],
        clipping_domain["clipping_success_rate"]["mean"],
        label=(f"{domain}/clipping success " "cross-check"),
    )

    _assert_close(
        lyapunov_clipping["success"]["sample_sd_of_seed_rates"],
        clipping_domain["clipping_success_rate"]["sample_sd"],
        label=(f"{domain}/clipping success SD " "cross-check"),
    )

    _assert_close(
        lyapunov_clipping["intervention_rate"]["mean"],
        clipping_domain["intervention_rate"]["mean"],
        label=(f"{domain}/clipping intervention " "cross-check"),
    )

    _assert_close(
        lyapunov_clipping["intervention_rate"]["sample_sd"],
        clipping_domain["intervention_rate"]["sample_sd"],
        label=(f"{domain}/clipping intervention SD " "cross-check"),
    )


def _build_mechanism_observation(
    summary: dict[str, Any],
    *,
    domain: str,
) -> LyapunovMechanismObservation:
    """Build the frozen clean Lyapunov mechanism observation."""

    mechanism = summary["mechanism_interpretation"]

    observed = bool(mechanism["principal_lyapunov_decrease_interventions_observed"])

    return LyapunovMechanismObservation(
        domain=domain,
        lyapunov_decrease_interventions_observed=(observed),
        interpretation=str(mechanism["interpretation"]),
        source_artifacts=(LYAPUNOV_SUMMARY_SOURCES[domain],),
    )


def load_canonical_safety_ablation(
    root: Path,
) -> CanonicalSafetyAblation:
    """Load and validate the frozen Sprint 5 clean-safety evidence."""

    baseline = _load_json(
        root,
        BASELINE_SUMMARY_SOURCE,
    )

    clipping = _load_json(
        root,
        CLIPPING_SUMMARY_SOURCE,
    )

    _validate_clean_summary(
        baseline,
        source=BASELINE_SUMMARY_SOURCE,
    )

    _validate_clean_summary(
        clipping,
        source=CLIPPING_SUMMARY_SOURCE,
    )

    if baseline.get("method") != "none":
        raise ValueError("Baseline summary method must be 'none'.")

    if clipping.get("method") != "clipping":
        raise ValueError("Clipping summary method must be 'clipping'.")

    records: list[SafetyAblationRecord] = []

    mechanism_observations: list[LyapunovMechanismObservation] = []

    for domain in DOMAINS:
        lyapunov_source = LYAPUNOV_SUMMARY_SOURCES[domain]

        lyapunov = _load_json(
            root,
            lyapunov_source,
        )

        _validate_clean_summary(
            lyapunov,
            source=lyapunov_source,
        )

        if lyapunov.get("domain") != domain:
            raise ValueError(f"Lyapunov domain mismatch " f"for {domain}.")

        _cross_check_none_evidence(
            domain=domain,
            baseline=baseline,
            clipping=clipping,
            lyapunov=lyapunov,
        )

        _cross_check_clipping_evidence(
            domain=domain,
            clipping=clipping,
            lyapunov=lyapunov,
        )

        records.extend(
            [
                _build_none_record(
                    baseline,
                    domain=domain,
                ),
                _build_clipping_record(
                    clipping,
                    domain=domain,
                ),
                _build_lyapunov_record(
                    lyapunov,
                    domain=domain,
                ),
            ]
        )

        mechanism_observations.append(
            _build_mechanism_observation(
                lyapunov,
                domain=domain,
            )
        )

    expected_matrix = {
        (
            domain,
            method,
        )
        for domain in DOMAINS
        for method in SAFETY_METHODS
    }

    observed_matrix = {
        (
            record.domain,
            record.method,
        )
        for record in records
    }

    if observed_matrix != expected_matrix:
        raise ValueError(
            "Canonical safety-ablation matrix " "is incomplete or duplicated."
        )

    if len(records) != 6:
        raise ValueError("Expected exactly six canonical " "safety-ablation records.")

    if len(mechanism_observations) != len(DOMAINS):
        raise ValueError("Expected one Lyapunov mechanism " "observation per domain.")

    for record in records:
        for source in record.source_artifacts:
            _validate_clean_source_path(source)

    for observation in mechanism_observations:
        for source in observation.source_artifacts:
            _validate_clean_source_path(source)

    return CanonicalSafetyAblation(
        records=tuple(records),
        lyapunov_mechanism_observations=tuple(mechanism_observations),
    )
