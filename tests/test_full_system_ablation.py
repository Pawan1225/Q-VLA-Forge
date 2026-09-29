"""Focused tests for Sprint 7.6 full-system ablation."""

import pytest

from q_vla_forge.evaluation.full_system_ablation import (
    ALLOW_SYNTHETIC_METRIC_COMPOSITION,
    CLAIM_CONTROLS,
    COMPONENT_EVIDENCE,
    COMPONENT_SOURCE_ARTIFACTS,
    CONFIGURATIONS,
    DOMAINS,
    EVIDENCE_STATUSES,
    FACTORS,
    FULL_SYSTEM_DIRECT_EXECUTION_FOUND,
    FULL_SYSTEM_PHASE1_CLASSIFICATION,
    INTEGRATION_BOUNDARY_SOURCES,
    INTERACTION_EFFECTS_ESTIMABLE_WITH_COMPONENT_ONLY,
    PHASE1_STATUS,
    PHASE2_STATUS,
    FullSystemConfiguration,
    build_phase1_evidence_matrix,
    can_report_end_to_end_metrics,
    claim_controls,
    component_only_configuration_count,
    component_sources_for_configuration,
    configuration_factors,
    direct_configuration_count,
    end_to_end_metrics_allowed_for_record,
    full_configuration_validated,
    interaction_effects_estimable,
    interaction_effects_estimable_from_component_only,
    not_evaluated_configuration_count,
    phase1_component_evidence_is_integrated_validation,
    phase1_evidence_status_for_configuration,
    phase2_requires_integrated_validation,
    synthetic_metric_composition_allowed,
    validate_component_evidence_registry,
    validate_configuration_name,
    validate_domain,
    validate_end_to_end_metrics,
    validate_evidence_status,
)

SOURCE = "results/final-validation/" "compression-ablation/" "compression-ablation.json"


def _configuration(
    *,
    name: str = "baseline",
    domain: str = "autonomous_driving",
    evidence_status: str = "component_only",
    source_artifacts: tuple[str, ...] = (SOURCE,),
) -> FullSystemConfiguration:
    compression, qml, safety = CONFIGURATIONS[name]

    return FullSystemConfiguration(
        name=name,
        compression=compression,
        qml=qml,
        safety=safety,
        domain=domain,
        evidence_status=(evidence_status),
        source_artifacts=(source_artifacts),
    )


def test_exactly_three_factors() -> None:
    assert FACTORS == (
        "compression",
        "qml",
        "safety",
    )


def test_exactly_eight_configurations() -> None:
    assert len(CONFIGURATIONS) == 8


@pytest.mark.parametrize(
    (
        "name",
        "expected",
    ),
    [
        (
            "baseline",
            (
                False,
                False,
                False,
            ),
        ),
        (
            "a",
            (
                True,
                False,
                False,
            ),
        ),
        (
            "b",
            (
                False,
                True,
                False,
            ),
        ),
        (
            "c",
            (
                False,
                False,
                True,
            ),
        ),
        (
            "d",
            (
                True,
                True,
                False,
            ),
        ),
        (
            "e",
            (
                True,
                False,
                True,
            ),
        ),
        (
            "f",
            (
                False,
                True,
                True,
            ),
        ),
        (
            "full",
            (
                True,
                True,
                True,
            ),
        ),
    ],
)
def test_factorial_configuration_map(
    name: str,
    expected: tuple[
        bool,
        bool,
        bool,
    ],
) -> None:
    assert configuration_factors(name) == expected


def test_baseline_is_000() -> None:
    assert CONFIGURATIONS["baseline"] == (
        False,
        False,
        False,
    )


def test_a_is_100() -> None:
    assert CONFIGURATIONS["a"] == (
        True,
        False,
        False,
    )


def test_b_is_010() -> None:
    assert CONFIGURATIONS["b"] == (
        False,
        True,
        False,
    )


def test_c_is_001() -> None:
    assert CONFIGURATIONS["c"] == (
        False,
        False,
        True,
    )


def test_d_is_110() -> None:
    assert CONFIGURATIONS["d"] == (
        True,
        True,
        False,
    )


def test_e_is_101() -> None:
    assert CONFIGURATIONS["e"] == (
        True,
        False,
        True,
    )


def test_f_is_011() -> None:
    assert CONFIGURATIONS["f"] == (
        False,
        True,
        True,
    )


def test_full_is_111() -> None:
    assert CONFIGURATIONS["full"] == (
        True,
        True,
        True,
    )


def test_two_domains_locked() -> None:
    assert DOMAINS == (
        "autonomous_driving",
        "robotics",
    )


def test_evidence_statuses_locked() -> None:
    assert EVIDENCE_STATUSES == (
        "direct",
        "component_only",
        "not_evaluated",
    )


def test_direct_permits_end_to_end_metrics() -> None:
    assert can_report_end_to_end_metrics("direct") is True


def test_component_only_blocks_end_to_end_metrics() -> None:
    assert can_report_end_to_end_metrics("component_only") is False


def test_not_evaluated_blocks_end_to_end_metrics() -> None:
    assert can_report_end_to_end_metrics("not_evaluated") is False


def test_synthetic_metric_composition_disabled() -> None:
    assert ALLOW_SYNTHETIC_METRIC_COMPOSITION is False

    assert synthetic_metric_composition_allowed() is False


def test_interaction_estimation_blocked_for_component_only() -> None:
    assert INTERACTION_EFFECTS_ESTIMABLE_WITH_COMPONENT_ONLY is False

    assert interaction_effects_estimable_from_component_only() is False

    assert interaction_effects_estimable("component_only") is False


def test_interaction_estimation_blocked_for_not_evaluated() -> None:
    assert interaction_effects_estimable("not_evaluated") is False


def test_direct_evidence_can_support_interaction_analysis_in_principle() -> None:
    assert interaction_effects_estimable("direct") is True


def test_direct_requires_provenance() -> None:
    with pytest.raises(ValueError):
        _configuration(
            evidence_status="direct",
            source_artifacts=(),
        )


def test_component_only_requires_provenance() -> None:
    with pytest.raises(ValueError):
        _configuration(
            evidence_status="component_only",
            source_artifacts=(),
        )


def test_not_evaluated_may_have_no_provenance() -> None:
    record = _configuration(
        evidence_status="not_evaluated",
        source_artifacts=(),
    )

    assert record.source_artifacts == ()


def test_component_evidence_registry_valid() -> None:
    validate_component_evidence_registry()

    assert set(COMPONENT_EVIDENCE) == set(FACTORS)


def test_component_registry_points_to_completed_sprints() -> None:
    assert COMPONENT_EVIDENCE == {
        "compression": "Sprint 7.3",
        "qml": "Sprint 7.4",
        "safety": "Sprint 7.5",
    }


def test_phase1_component_evidence_not_integrated_validation() -> None:
    assert PHASE1_STATUS == "evaluated_component_evidence"

    assert phase1_component_evidence_is_integrated_validation() is False


def test_phase2_integrated_validation_identified() -> None:
    assert PHASE2_STATUS == "candidate_integrated_validation"

    assert phase2_requires_integrated_validation() is True


def test_full_configuration_not_validated_without_direct_evidence() -> None:
    assert full_configuration_validated("component_only") is False

    assert full_configuration_validated("not_evaluated") is False


def test_full_configuration_validated_only_with_direct_evidence() -> None:
    assert full_configuration_validated("direct") is True


def test_component_only_record_blocks_reward() -> None:
    assert (
        end_to_end_metrics_allowed_for_record(
            "component_only",
            {
                "reward": 1.0,
            },
        )
        is False
    )


def test_component_only_record_blocks_mse() -> None:
    assert (
        end_to_end_metrics_allowed_for_record(
            "component_only",
            {
                "mse": 0.01,
            },
        )
        is False
    )


def test_component_only_record_blocks_latency() -> None:
    assert (
        end_to_end_metrics_allowed_for_record(
            "component_only",
            {
                "latency": 2.0,
            },
        )
        is False
    )


def test_component_only_record_blocks_success() -> None:
    assert (
        end_to_end_metrics_allowed_for_record(
            "component_only",
            {
                "success_rate": 0.5,
            },
        )
        is False
    )


def test_component_only_record_blocks_violation_rate() -> None:
    assert (
        end_to_end_metrics_allowed_for_record(
            "component_only",
            {
                "violation_rate": 0.0,
            },
        )
        is False
    )


def test_not_evaluated_record_blocks_metrics() -> None:
    assert (
        end_to_end_metrics_allowed_for_record(
            "not_evaluated",
            {
                "reward": 1.0,
            },
        )
        is False
    )


def test_direct_record_may_report_metrics() -> None:
    assert (
        end_to_end_metrics_allowed_for_record(
            "direct",
            {
                "reward": 1.0,
                "latency": 2.0,
            },
        )
        is True
    )


def test_missing_metrics_allowed_for_component_only() -> None:
    assert (
        end_to_end_metrics_allowed_for_record(
            "component_only",
            {
                "reward": None,
                "latency": None,
            },
        )
        is True
    )


def test_validate_metrics_rejects_synthetic_component_result() -> None:
    with pytest.raises(ValueError):
        validate_end_to_end_metrics(
            "component_only",
            {
                "reward": 1.0,
            },
        )


def test_validate_metrics_accepts_empty_component_result() -> None:
    validate_end_to_end_metrics(
        "component_only",
        {
            "reward": None,
            "mse": None,
            "latency": None,
        },
    )


def test_configuration_factor_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        FullSystemConfiguration(
            name="a",
            compression=False,
            qml=False,
            safety=False,
            domain="autonomous_driving",
            evidence_status="component_only",
            source_artifacts=(SOURCE,),
        )


def test_invalid_configuration_name_rejected() -> None:
    with pytest.raises(ValueError):
        validate_configuration_name("invalid")


def test_invalid_domain_rejected() -> None:
    with pytest.raises(ValueError):
        validate_domain("invalid")


def test_invalid_evidence_status_rejected() -> None:
    with pytest.raises(ValueError):
        validate_evidence_status("invalid")


def test_blank_provenance_rejected() -> None:
    with pytest.raises(ValueError):
        _configuration(source_artifacts=("",))


def test_record_reports_metric_permission() -> None:
    direct = _configuration(evidence_status="direct")

    component = _configuration(evidence_status="component_only")

    assert direct.can_report_end_to_end_metrics is True

    assert component.can_report_end_to_end_metrics is False


def test_record_serialization_retains_controls() -> None:
    record = _configuration(
        name="full",
        evidence_status="component_only",
    )

    payload = record.to_dict()

    assert payload["compression"] is True

    assert payload["qml"] is True

    assert payload["safety"] is True

    assert payload["can_report_end_to_end_metrics"] is False

    assert payload["phase_status"] == PHASE2_STATUS


def test_quantum_advantage_claim_blocked() -> None:
    assert CLAIM_CONTROLS["quantum_advantage_claim_allowed"] is False


def test_full_system_superiority_claim_blocked() -> None:
    assert CLAIM_CONTROLS["full_system_superiority_claim_allowed"] is False


def test_cross_domain_zero_shot_claim_blocked() -> None:
    assert CLAIM_CONTROLS["cross_domain_zero_shot_claim_allowed"] is False


def test_production_readiness_claim_blocked() -> None:
    assert CLAIM_CONTROLS["production_readiness_claim_allowed"] is False


def test_integrated_factorial_validation_claim_blocked() -> None:
    assert CLAIM_CONTROLS["integrated_factorial_validation_claim_allowed"] is False


def test_interaction_claim_without_direct_evidence_blocked() -> None:
    assert (
        CLAIM_CONTROLS["interaction_effect_claim_allowed_without_direct_evidence"]
        is False
    )


def test_all_claim_controls_blocked() -> None:
    controls = claim_controls()

    assert controls

    assert all(value is False for value in controls.values())


def test_claim_controls_returns_copy() -> None:
    controls = claim_controls()

    controls["full_system_superiority_claim_allowed"] = True

    assert CLAIM_CONTROLS["full_system_superiority_claim_allowed"] is False


def test_phase1_matrix_has_sixteen_classifications() -> None:
    matrix = build_phase1_evidence_matrix()

    assert len(matrix) == 16


def test_phase1_matrix_covers_all_domain_configuration_pairs() -> None:
    matrix = build_phase1_evidence_matrix()

    observed = {
        (
            record.domain,
            record.name,
        )
        for record in matrix
    }

    expected = {
        (
            domain,
            name,
        )
        for domain in DOMAINS
        for name in CONFIGURATIONS
    }

    assert observed == expected


def test_no_direct_full_system_execution_found() -> None:
    assert FULL_SYSTEM_DIRECT_EXECUTION_FOUND is False


def test_frozen_phase1_classification_is_component_only() -> None:
    assert FULL_SYSTEM_PHASE1_CLASSIFICATION == "component_only"


@pytest.mark.parametrize(
    "domain",
    DOMAINS,
)
@pytest.mark.parametrize(
    "name",
    CONFIGURATIONS,
)
def test_all_phase1_factorial_cells_are_component_only(
    domain: str,
    name: str,
) -> None:
    assert (
        phase1_evidence_status_for_configuration(
            name,
            domain,
        )
        == "component_only"
    )


def test_matrix_contains_zero_direct_cells() -> None:
    assert direct_configuration_count() == 0


def test_matrix_contains_sixteen_component_only_cells() -> None:
    assert component_only_configuration_count() == 16


def test_matrix_contains_zero_not_evaluated_cells() -> None:
    assert not_evaluated_configuration_count() == 0


def test_full_configuration_is_not_direct_in_either_domain() -> None:
    matrix = build_phase1_evidence_matrix()

    full_records = [record for record in matrix if record.name == "full"]

    assert len(full_records) == 2

    assert all(record.evidence_status == "component_only" for record in full_records)

    assert all(
        (record.can_report_end_to_end_metrics is False) for record in full_records
    )


def test_baseline_is_not_promoted_to_direct_integrated_baseline() -> None:
    matrix = build_phase1_evidence_matrix()

    baseline_records = [record for record in matrix if record.name == "baseline"]

    assert len(baseline_records) == 2

    assert all(
        record.evidence_status == "component_only" for record in baseline_records
    )


def test_component_sources_registry_covers_all_factors() -> None:
    assert set(COMPONENT_SOURCE_ARTIFACTS) == set(FACTORS)


def test_full_configuration_retains_all_component_sources() -> None:
    sources = component_sources_for_configuration("full")

    assert COMPONENT_SOURCE_ARTIFACTS["compression"] in sources

    assert COMPONENT_SOURCE_ARTIFACTS["qml"] in sources

    assert COMPONENT_SOURCE_ARTIFACTS["safety"] in sources


def test_integration_boundary_provenance_retained() -> None:
    matrix = build_phase1_evidence_matrix()

    for record in matrix:
        for source in INTEGRATION_BOUNDARY_SOURCES:
            assert source in record.source_artifacts


def test_component_only_matrix_has_no_end_to_end_metric_permission() -> None:
    matrix = build_phase1_evidence_matrix()

    assert all((record.can_report_end_to_end_metrics is False) for record in matrix)


def test_component_only_matrix_is_phase2_candidate_validation() -> None:
    matrix = build_phase1_evidence_matrix()

    assert all(record.phase_status == PHASE2_STATUS for record in matrix)
