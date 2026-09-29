"""Focused tests for Sprint 7.5 safety-ablation contract."""

import math
from pathlib import Path

import pytest

from q_vla_forge.evaluation.safety_ablation import (
    BASELINE_SUMMARY_SOURCE,
    CLAIM_CONTROLS,
    CLIPPING_SUMMARY_SOURCE,
    DOMAINS,
    LYAPUNOV_METHOD_CLASSIFICATION,
    LYAPUNOV_SUMMARY_SOURCES,
    METHOD_DESCRIPTIONS,
    REQUIRED_SEEDS,
    SAFETY_METHODS,
    SafetyAblationRecord,
    absolute_difference,
    claim_controls,
    load_canonical_safety_ablation,
    lyapunov_method_is_classical,
    method_description,
    method_role,
    privileged_state_limitation,
    relative_reduction_percent,
    validate_domain,
    validate_method,
    validate_required_seeds,
    zero_observed_is_formal_guarantee,
    zero_observed_violations,
)

ROOT = Path(__file__).resolve().parents[1]

SOURCE = "results/safety/baseline/" "placeholder-canonical-source.json"


def _record(
    *,
    domain: str = "autonomous_driving",
    method: str = "none",
    violation_rate_mean: float = 0.2,
    violation_rate_sample_std: float = 0.1,
    reward_mean: float = -1.0,
    reward_sample_std: float = 0.2,
    success_rate_mean: float = 0.5,
    success_rate_sample_std: float = 0.1,
    intervention_rate_mean: float | None = None,
    intervention_rate_sample_std: float | None = None,
) -> SafetyAblationRecord:
    return SafetyAblationRecord(
        domain=domain,
        method=method,
        violation_rate_mean=violation_rate_mean,
        violation_rate_sample_std=(violation_rate_sample_std),
        reward_mean=reward_mean,
        reward_sample_std=reward_sample_std,
        success_rate_mean=success_rate_mean,
        success_rate_sample_std=(success_rate_sample_std),
        seeds=REQUIRED_SEEDS,
        source_artifacts=(SOURCE,),
        intervention_rate_mean=(intervention_rate_mean),
        intervention_rate_sample_std=(intervention_rate_sample_std),
    )


def test_required_seeds_are_frozen() -> None:
    assert REQUIRED_SEEDS == (
        42,
        123,
        456,
    )


def test_domains_are_frozen() -> None:
    assert DOMAINS == (
        "autonomous_driving",
        "robotics",
    )


def test_methods_are_frozen() -> None:
    assert SAFETY_METHODS == (
        "none",
        "clipping",
        "lyapunov",
    )


@pytest.mark.parametrize(
    (
        "method",
        "expected",
    ),
    [
        (
            "none",
            "REFERENCE",
        ),
        (
            "clipping",
            "EVALUATED",
        ),
        (
            "lyapunov",
            "EVALUATED",
        ),
    ],
)
def test_method_roles(
    method: str,
    expected: str,
) -> None:
    assert method_role(method) == expected


def test_none_is_reference() -> None:
    assert _record(method="none").role == "REFERENCE"


def test_clipping_is_evaluated() -> None:
    assert _record(method="clipping").role == "EVALUATED"


def test_lyapunov_is_evaluated() -> None:
    assert _record(method="lyapunov").role == "EVALUATED"


def test_invalid_domain_rejected() -> None:
    with pytest.raises(ValueError):
        validate_domain("invalid")


def test_invalid_method_rejected() -> None:
    with pytest.raises(ValueError):
        validate_method("invalid")


def test_wrong_seed_order_rejected() -> None:
    with pytest.raises(ValueError):
        validate_required_seeds(
            (
                456,
                123,
                42,
            )
        )


def test_missing_seed_rejected() -> None:
    with pytest.raises(ValueError):
        validate_required_seeds(
            (
                42,
                123,
            )
        )


def test_negative_violation_rate_rejected() -> None:
    with pytest.raises(ValueError):
        _record(violation_rate_mean=-0.01)


def test_zero_violation_rate_accepted() -> None:
    record = _record(violation_rate_mean=0.0)

    assert record.violation_rate_mean == 0.0


def test_negative_violation_sample_sd_rejected() -> None:
    with pytest.raises(ValueError):
        _record(violation_rate_sample_std=-0.1)


def test_negative_success_rate_rejected() -> None:
    with pytest.raises(ValueError):
        _record(success_rate_mean=-0.01)


def test_success_rate_above_one_rejected() -> None:
    with pytest.raises(ValueError):
        _record(success_rate_mean=1.01)


def test_success_rate_one_accepted() -> None:
    record = _record(success_rate_mean=1.0)

    assert record.success_rate_mean == 1.0


def test_reward_may_be_negative() -> None:
    record = _record(reward_mean=-50.0)

    assert record.reward_mean == -50.0


def test_nonfinite_reward_rejected() -> None:
    with pytest.raises(ValueError):
        _record(reward_mean=math.inf)


def test_negative_reward_sample_sd_rejected() -> None:
    with pytest.raises(ValueError):
        _record(reward_sample_std=-0.01)


def test_missing_optional_metric_remains_none() -> None:
    record = _record(
        intervention_rate_mean=None,
        intervention_rate_sample_std=None,
    )

    assert record.intervention_rate_mean is None

    assert record.intervention_rate_sample_std is None


def test_missing_optional_metric_is_not_zero() -> None:
    record = _record(intervention_rate_mean=None)

    assert record.intervention_rate_mean != 0.0


def test_invalid_optional_rate_rejected() -> None:
    with pytest.raises(ValueError):
        _record(intervention_rate_mean=1.1)


def test_source_provenance_required() -> None:
    with pytest.raises(ValueError):
        SafetyAblationRecord(
            domain="autonomous_driving",
            method="none",
            violation_rate_mean=0.1,
            violation_rate_sample_std=0.0,
            reward_mean=0.0,
            reward_sample_std=0.0,
            success_rate_mean=0.0,
            success_rate_sample_std=0.0,
            seeds=REQUIRED_SEEDS,
            source_artifacts=(),
        )


def test_blank_source_provenance_rejected() -> None:
    with pytest.raises(ValueError):
        SafetyAblationRecord(
            domain="autonomous_driving",
            method="none",
            violation_rate_mean=0.1,
            violation_rate_sample_std=0.0,
            reward_mean=0.0,
            reward_sample_std=0.0,
            success_rate_mean=0.0,
            success_rate_sample_std=0.0,
            seeds=REQUIRED_SEEDS,
            source_artifacts=("",),
        )


def test_relative_reduction_formula() -> None:
    assert relative_reduction_percent(
        reference=0.4,
        candidate=0.2,
    ) == pytest.approx(50.0)


def test_complete_reduction_from_nonzero_reference() -> None:
    assert relative_reduction_percent(
        reference=0.4,
        candidate=0.0,
    ) == pytest.approx(100.0)


def test_zero_reference_relative_reduction_is_undefined() -> None:
    assert (
        relative_reduction_percent(
            reference=0.0,
            candidate=0.0,
        )
        is None
    )


def test_zero_reference_nonzero_candidate_is_undefined() -> None:
    assert (
        relative_reduction_percent(
            reference=0.0,
            candidate=0.2,
        )
        is None
    )


def test_absolute_difference_remains_defined() -> None:
    assert absolute_difference(
        reference=0.0,
        candidate=0.2,
    ) == pytest.approx(0.2)


def test_absolute_difference_uses_candidate_minus_reference() -> None:
    assert absolute_difference(
        reference=0.4,
        candidate=0.1,
    ) == pytest.approx(-0.3)


def test_zero_observed_violation_detected() -> None:
    assert zero_observed_violations(0.0)


def test_nonzero_observed_violation_detected() -> None:
    assert not zero_observed_violations(0.01)


def test_zero_observed_is_not_formal_guarantee() -> None:
    assert zero_observed_is_formal_guarantee() is False


def test_lyapunov_method_is_classical() -> None:
    assert LYAPUNOV_METHOD_CLASSIFICATION == "classical"

    assert lyapunov_method_is_classical() is True


def test_lyapunov_description_says_classical() -> None:
    description = method_description("lyapunov")

    assert "Classical" in description


def test_method_descriptions_are_locked() -> None:
    assert set(METHOD_DESCRIPTIONS) == set(SAFETY_METHODS)


def test_quantum_safety_advantage_blocked() -> None:
    assert CLAIM_CONTROLS["quantum_safety_advantage_claim_allowed"] is False


def test_formal_stability_guarantee_blocked() -> None:
    assert CLAIM_CONTROLS["formal_stability_guarantee_allowed"] is False


def test_formal_lyapunov_proof_blocked() -> None:
    assert CLAIM_CONTROLS["formal_lyapunov_stability_proof_allowed"] is False


def test_forward_invariance_guarantee_blocked() -> None:
    assert CLAIM_CONTROLS["forward_invariance_guarantee_allowed"] is False


def test_certification_claim_blocked() -> None:
    assert CLAIM_CONTROLS["iso_26262_certification_claim_allowed"] is False


def test_production_safety_claim_blocked() -> None:
    assert CLAIM_CONTROLS["production_safety_claim_allowed"] is False


def test_real_world_safety_claim_blocked() -> None:
    assert CLAIM_CONTROLS["real_world_safety_claim_allowed"] is False


def test_guaranteed_zero_violations_claim_blocked() -> None:
    assert CLAIM_CONTROLS["guaranteed_zero_violations_claim_allowed"] is False


def test_all_claim_controls_remain_blocked() -> None:
    controls = claim_controls()

    assert controls

    assert all(value is False for value in controls.values())


def test_claim_controls_returns_copy() -> None:
    controls = claim_controls()

    controls["formal_stability_guarantee_allowed"] = True

    assert CLAIM_CONTROLS["formal_stability_guarantee_allowed"] is False


def test_privileged_state_limitation_retained() -> None:
    limitation = privileged_state_limitation()

    lowered = limitation.lower()

    assert "true simulator state" in lowered

    assert "perturbed observations" in lowered


def test_privileged_state_does_not_claim_real_world_robustness() -> None:
    limitation = (privileged_state_limitation()).lower()

    assert "do not establish" in limitation

    assert "real-world sensing robustness" in limitation


def test_record_serialization_retains_role() -> None:
    payload = _record(method="none").to_dict()

    assert payload["role"] == "REFERENCE"


def test_lyapunov_serialization_retains_classical_label() -> None:
    payload = _record(method="lyapunov").to_dict()

    assert payload["lyapunov_method_classification"] == "classical"


def test_non_lyapunov_record_has_no_lyapunov_classification() -> None:
    payload = _record(method="clipping").to_dict()

    assert payload["lyapunov_method_classification"] is None


def test_canonical_safety_ablation_has_six_records() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    assert len(evidence.records) == 6


def test_canonical_safety_ablation_matrix_complete() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    observed = {
        (
            record.domain,
            record.method,
        )
        for record in evidence.records
    }

    expected = {
        (
            domain,
            method,
        )
        for domain in DOMAINS
        for method in SAFETY_METHODS
    }

    assert observed == expected


def test_canonical_records_use_required_seeds() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    assert all(record.seeds == REQUIRED_SEEDS for record in evidence.records)


def test_none_record_uses_baseline_provenance() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    none_records = [record for record in evidence.records if record.method == "none"]

    assert len(none_records) == 2

    assert all(
        record.source_artifacts == (BASELINE_SUMMARY_SOURCE,) for record in none_records
    )


def test_clipping_record_uses_clipping_provenance() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    records = [record for record in evidence.records if record.method == "clipping"]

    assert len(records) == 2

    assert all(
        record.source_artifacts == (CLIPPING_SUMMARY_SOURCE,) for record in records
    )


def test_lyapunov_records_use_domain_provenance() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    records = [record for record in evidence.records if record.method == "lyapunov"]

    assert len(records) == 2

    for record in records:
        assert record.source_artifacts == (LYAPUNOV_SUMMARY_SOURCES[record.domain],)


def test_primary_sources_exclude_robustness() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    sources = {
        source for record in evidence.records for source in record.source_artifacts
    }

    assert all("gaussian-robustness" not in source for source in sources)

    assert all("structured-state-robustness" not in source for source in sources)

    assert all("action-robustness" not in source for source in sources)


def test_driving_none_sanity_values() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    record = next(
        record
        for record in evidence.records
        if (record.domain == "autonomous_driving" and record.method == "none")
    )

    assert record.violation_rate_mean == pytest.approx(0.39084090909090907)

    assert record.violation_rate_sample_std == pytest.approx(0.42242665691834197)

    assert record.reward_mean == pytest.approx(-9.200229677202655)

    assert record.reward_sample_std == pytest.approx(1.935700075744146)

    assert record.success_rate_mean == pytest.approx(0.05)

    assert record.success_rate_sample_std == pytest.approx(0.08660254037844387)


def test_robotics_none_sanity_values() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    record = next(
        record
        for record in evidence.records
        if (record.domain == "robotics" and record.method == "none")
    )

    assert record.violation_rate_mean == pytest.approx(0.014666666666666666)

    assert record.violation_rate_sample_std == pytest.approx(0.018536001007049316)

    assert record.reward_mean == pytest.approx(0.6328459382840873)

    assert record.reward_sample_std == pytest.approx(0.02881796527171248)

    assert record.success_rate_mean == pytest.approx(0.0)


@pytest.mark.parametrize(
    "domain",
    DOMAINS,
)
@pytest.mark.parametrize(
    "method",
    (
        "clipping",
        "lyapunov",
    ),
)
def test_evaluated_safety_methods_have_zero_observed_violation_rate(
    domain: str,
    method: str,
) -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    record = next(
        record
        for record in evidence.records
        if (record.domain == domain and record.method == method)
    )

    assert record.violation_rate_mean == pytest.approx(0.0)

    assert record.violation_rate_sample_std == pytest.approx(0.0)


def test_driving_clipping_sanity_values() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    record = next(
        record
        for record in evidence.records
        if (record.domain == "autonomous_driving" and record.method == "clipping")
    )

    assert record.reward_mean == pytest.approx(-5.212558982989345)

    assert record.success_rate_mean == pytest.approx(0.3333333333333333)

    assert record.intervention_rate_mean == pytest.approx(0.4175)


def test_robotics_clipping_sanity_values() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    record = next(
        record
        for record in evidence.records
        if (record.domain == "robotics" and record.method == "clipping")
    )

    assert record.reward_mean == pytest.approx(0.6778999451491963)

    assert record.success_rate_mean == pytest.approx(0.0)

    assert record.intervention_rate_mean == pytest.approx(0.014666666666666666)


def test_driving_lyapunov_sanity_values() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    record = next(
        record
        for record in evidence.records
        if (record.domain == "autonomous_driving" and record.method == "lyapunov")
    )

    assert record.reward_mean == pytest.approx(-5.212558982989345)

    assert record.success_rate_mean == pytest.approx(0.3333333333333333)

    assert record.intervention_rate_mean == pytest.approx(0.4175)


def test_robotics_lyapunov_sanity_values() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    record = next(
        record
        for record in evidence.records
        if (record.domain == "robotics" and record.method == "lyapunov")
    )

    assert record.reward_mean == pytest.approx(0.6779026320391401)

    assert record.success_rate_mean == pytest.approx(0.0)

    assert record.intervention_rate_mean == pytest.approx(0.014666666666666666)


def test_none_intervention_metric_is_not_manufactured() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    for record in evidence.records:
        if record.method == "none":
            assert record.intervention_rate_mean is None

            assert record.intervention_rate_sample_std is None


def test_clipping_and_lyapunov_intervention_metrics_exist() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    for record in evidence.records:
        if record.method in (
            "clipping",
            "lyapunov",
        ):
            assert record.intervention_rate_mean is not None

            assert record.intervention_rate_sample_std is not None


def test_lyapunov_mechanism_observations_cover_both_domains() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    observed = {item.domain for item in evidence.lyapunov_mechanism_observations}

    assert observed == set(DOMAINS)


def test_clean_principal_runs_show_no_lyapunov_decrease_intervention() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    assert all(
        (item.lyapunov_decrease_interventions_observed is False)
        for item in evidence.lyapunov_mechanism_observations
    )


def test_lyapunov_mechanism_interpretations_are_retained() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    assert all(
        item.interpretation.strip() for item in evidence.lyapunov_mechanism_observations
    )


def test_mechanism_provenance_matches_domain() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    for observation in evidence.lyapunov_mechanism_observations:
        assert observation.source_artifacts == (
            LYAPUNOV_SUMMARY_SOURCES[observation.domain],
        )


def test_canonical_none_remains_reference() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    for record in evidence.records:
        if record.method == "none":
            assert record.role == "REFERENCE"


def test_canonical_evaluated_methods_remain_evaluated() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    for record in evidence.records:
        if record.method in (
            "clipping",
            "lyapunov",
        ):
            assert record.role == "EVALUATED"


def test_canonical_lyapunov_records_remain_classical() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    for record in evidence.records:
        if record.method == "lyapunov":
            assert record.lyapunov_is_classical is True

            assert record.to_dict()["lyapunov_method_classification"] == "classical"


def test_canonical_zero_observed_does_not_create_guarantee() -> None:
    evidence = load_canonical_safety_ablation(ROOT)

    evaluated = [
        record
        for record in evidence.records
        if record.method
        in (
            "clipping",
            "lyapunov",
        )
    ]

    assert all(record.violation_rate_mean == 0.0 for record in evaluated)

    assert zero_observed_is_formal_guarantee() is False
