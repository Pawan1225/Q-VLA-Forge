"""Focused tests for Sprint 7.8 final-result-table contracts."""

import pytest

from q_vla_forge.evaluation.final_result_tables import (
    CLAIM_CONTROLS,
    COMPRESSION_CRITERION,
    COMPRESSION_METHODS,
    DOMAINS,
    FULL_SYSTEM_COUNTS,
    MISSING_VALUE_SEMANTICS,
    NEW_EXPERIMENTS,
    NEW_SCIENTIFIC_RESULTS,
    NEW_TRAINING,
    NOT_APPLICABLE,
    NOT_MEASURED,
    NOT_REACHED,
    QML_PARAMETER_REDUCTION_PERCENT,
    QML_TARGET_REACH_COUNTS,
    RL_POLICIES,
    SAFETY_METHODS,
    SEEDS,
    STATISTICAL_PROTOCOL,
    SYNTHETIC_METRIC_COMPOSITION_ALLOWED,
    TABLE_COLUMNS,
    TABLE_FILENAMES,
    TABLE_IDS,
    TABLE_SOURCES,
    FinalResultTable,
    build_all_table_contracts,
    build_table_contract,
    claim_controls,
    missing_value_semantics,
    not_applicable_value,
    qml_nonattainment_value,
    synthetic_metric_composition_allowed,
    table_output_path,
    unmeasured_value,
    validate_table_id,
)


def test_exactly_five_canonical_tables() -> None:
    assert TABLE_IDS == (
        "compression",
        "training",
        "rl",
        "safety",
        "cross_domain",
    )


def test_output_filenames_match_table_ids() -> None:
    assert set(TABLE_FILENAMES) == set(TABLE_IDS)


@pytest.mark.parametrize(
    "table_id",
    TABLE_IDS,
)
def test_every_table_has_provenance(
    table_id: str,
) -> None:
    contract = build_table_contract(table_id)

    assert contract.source_artifacts
    assert all(source.strip() for source in contract.source_artifacts)


@pytest.mark.parametrize(
    "table_id",
    TABLE_IDS,
)
def test_every_table_has_columns(
    table_id: str,
) -> None:
    contract = build_table_contract(table_id)

    assert contract.columns


@pytest.mark.parametrize(
    "table_id",
    TABLE_IDS,
)
def test_every_table_has_metrics(
    table_id: str,
) -> None:
    contract = build_table_contract(table_id)

    assert contract.metrics


@pytest.mark.parametrize(
    "table_id",
    TABLE_IDS,
)
def test_every_table_has_limitations(
    table_id: str,
) -> None:
    contract = build_table_contract(table_id)

    assert contract.limitations


@pytest.mark.parametrize(
    "table_id",
    TABLE_IDS,
)
def test_all_tables_use_frozen_evidence(
    table_id: str,
) -> None:
    contract = build_table_contract(table_id)

    assert contract.generated_from_frozen_evidence is True


def test_locked_seed_protocol() -> None:
    assert SEEDS == (
        42,
        123,
        456,
    )

    assert "sample SD" in (STATISTICAL_PROTOCOL)

    assert "n=3" in (STATISTICAL_PROTOCOL)


def test_no_new_training() -> None:
    assert NEW_TRAINING is False


def test_no_new_experiments() -> None:
    assert NEW_EXPERIMENTS is False


def test_no_new_scientific_results() -> None:
    assert NEW_SCIENTIFIC_RESULTS is False


def test_no_synthetic_metric_composition() -> None:
    assert SYNTHETIC_METRIC_COMPOSITION_ALLOWED is False

    assert synthetic_metric_composition_allowed() is False


def test_compression_contains_four_methods() -> None:
    assert COMPRESSION_METHODS == (
        "fp32",
        "int8",
        "svd",
        "tt_mps",
    )


def test_compression_contains_both_domains() -> None:
    assert DOMAINS == (
        "autonomous_driving",
        "robotics",
    )


def test_compression_int8_pass_retained() -> None:
    assert COMPRESSION_CRITERION["int8"] == "PASS"


def test_compression_svd_fail_retained() -> None:
    assert COMPRESSION_CRITERION["svd"] == "FAIL"


def test_compression_tt_mps_fail_retained() -> None:
    assert COMPRESSION_CRITERION["tt_mps"] == "FAIL"


def test_fp32_is_reference() -> None:
    assert COMPRESSION_CRITERION["fp32"] == "REFERENCE"


def test_compression_table_uses_mse_not_accuracy() -> None:
    columns = TABLE_COLUMNS["compression"]

    assert "MSE" in columns
    assert "Accuracy" not in columns


def test_training_table_uses_real_metric_name() -> None:
    columns = TABLE_COLUMNS["training"]

    assert "Final Validation Loss / MSE" in columns

    assert "Final Accuracy" not in columns


def test_training_negative_result_retained() -> None:
    contract = build_table_contract("training")

    text = " ".join(contract.limitations).lower()

    assert "did not demonstrate" in text

    assert ">=10%" in text


def test_missing_memory_is_not_zero() -> None:
    assert NOT_MEASURED == ("Not measured")

    assert NOT_MEASURED != "0"
    assert NOT_MEASURED != "0 MB"


def test_rl_contains_three_policy_families() -> None:
    assert RL_POLICIES == (
        "ppo_mlp",
        "matched_classical",
        "ppo_pqc",
    )


def test_rl_target_reach_counts_retained() -> None:
    assert QML_TARGET_REACH_COUNTS["ppo_mlp"] == {
        "reached": 6,
        "total": 6,
    }

    assert QML_TARGET_REACH_COUNTS["matched_classical"] == {
        "reached": 1,
        "total": 6,
    }

    assert QML_TARGET_REACH_COUNTS["ppo_pqc"] == {
        "reached": 0,
        "total": 6,
    }


def test_qml_nonattainment_is_not_budget_censored() -> None:
    assert qml_nonattainment_value() == "Not reached"

    assert qml_nonattainment_value() != "20000"


def test_qml_actor_compactness_is_explicit() -> None:
    assert QML_PARAMETER_REDUCTION_PERCENT["autonomous_driving"] == 95.90

    assert QML_PARAMETER_REDUCTION_PERCENT["robotics"] == 95.51


def test_rl_limitations_separate_compactness() -> None:
    contract = build_table_contract("rl")

    text = " ".join(contract.limitations).lower()

    assert "compactness" in text
    assert "sample efficiency" in text


def test_safety_contains_three_methods() -> None:
    assert SAFETY_METHODS == (
        "none",
        "clipping",
        "lyapunov",
    )


def test_safety_uses_observed_not_guaranteed_language() -> None:
    contract = build_table_contract("safety")

    text = " ".join(contract.limitations).lower()

    assert "zero observed" in text
    assert "formal stability proof" in text


def test_lyapunov_remains_classical() -> None:
    contract = build_table_contract("safety")

    text = " ".join(contract.limitations).lower()

    assert "classical" in text


def test_robustness_recovery_not_mixed_with_clean_rows() -> None:
    contract = build_table_contract("safety")

    text = " ".join(contract.limitations).lower()

    assert "must not be mixed" in text


def test_cross_domain_shared_architecture_not_shared_weights() -> None:
    contract = build_table_contract("cross_domain")

    text = " ".join(contract.limitations).lower()

    assert "framework" in text
    assert "separate trained policy weights" in text


def test_full_system_direct_remains_zero() -> None:
    assert FULL_SYSTEM_COUNTS["direct"] == 0


def test_full_system_component_only_remains_sixteen() -> None:
    assert FULL_SYSTEM_COUNTS["component_only"] == 16


def test_full_system_not_evaluated_remains_zero() -> None:
    assert FULL_SYSTEM_COUNTS["not_evaluated"] == 0


def test_missing_value_semantics_are_distinct() -> None:
    assert NOT_REACHED == ("Not reached")

    assert NOT_MEASURED == ("Not measured")

    assert NOT_APPLICABLE == ("Not applicable")

    assert (
        len(
            {
                NOT_REACHED,
                NOT_MEASURED,
                NOT_APPLICABLE,
            }
        )
        == 3
    )


def test_missing_value_helpers() -> None:
    assert qml_nonattainment_value() == NOT_REACHED

    assert unmeasured_value() == NOT_MEASURED

    assert not_applicable_value() == NOT_APPLICABLE


def test_missing_value_semantics_returns_copy() -> None:
    values = missing_value_semantics()

    values["not_reached"] = "wrong"

    assert MISSING_VALUE_SEMANTICS["not_reached"] == "Not reached"


def test_claim_controls_all_blocked() -> None:
    assert CLAIM_CONTROLS

    assert all(value is False for value in CLAIM_CONTROLS.values())


def test_claim_controls_returns_copy() -> None:
    controls = claim_controls()

    controls["quantum_advantage_claim_allowed"] = True

    assert CLAIM_CONTROLS["quantum_advantage_claim_allowed"] is False


@pytest.mark.parametrize(
    "claim_name",
    (
        "tt_mps_superiority_claim_allowed",
        ("robust_training_efficiency_" "improvement_claim_allowed"),
        ("qml_sample_efficiency_" "advantage_claim_allowed"),
        ("qml_performance_" "superiority_claim_allowed"),
        "quantum_advantage_claim_allowed",
        "quantum_speedup_claim_allowed",
        ("formal_lyapunov_stability_" "claim_allowed"),
        ("guaranteed_zero_violations_" "claim_allowed"),
        ("iso_26262_certification_" "claim_allowed"),
        "production_readiness_claim_allowed",
        ("zero_shot_cross_domain_" "transfer_claim_allowed"),
        ("full_system_superiority_" "claim_allowed"),
    ),
)
def test_required_claim_control_exists(
    claim_name: str,
) -> None:
    assert claim_name in CLAIM_CONTROLS

    assert CLAIM_CONTROLS[claim_name] is False


def test_every_table_has_output_path() -> None:
    for table_id in TABLE_IDS:
        path = table_output_path(table_id)

        assert path.endswith(".csv")


def test_build_all_contracts_returns_five() -> None:
    contracts = build_all_table_contracts()

    assert len(contracts) == 5

    assert {contract.table_id for contract in contracts} == set(TABLE_IDS)


def test_invalid_table_id_rejected() -> None:
    with pytest.raises(ValueError):
        validate_table_id("invalid")


def test_empty_provenance_rejected() -> None:
    with pytest.raises(ValueError):
        FinalResultTable(
            table_id="compression",
            title="Compression",
            columns=("Method",),
            rows=(),
            source_artifacts=(),
            metrics=("mse",),
            statistical_protocol=(STATISTICAL_PROTOCOL),
            limitations=("test",),
        )


def test_table_sources_are_not_figures() -> None:
    for sources in TABLE_SOURCES.values():
        for source in sources:
            assert not source.endswith(".png")


def test_tables_consume_evidence_not_figures() -> None:
    all_sources = " ".join(
        source for sources in TABLE_SOURCES.values() for source in sources
    ).lower()

    assert "figure-01" not in all_sources

    assert "figure-02" not in all_sources


def test_cross_domain_no_synthetic_score() -> None:
    contract = build_table_contract("cross_domain")

    text = " ".join(contract.limitations).lower()

    assert "no synthetic aggregate score" in text
