"""Focused tests for Sprint 7.7 final-figure contracts."""

import pytest

from q_vla_forge.evaluation.final_figures import (
    ABLATION_COMPONENT_ONLY_COUNT,
    ABLATION_DIRECT_COUNT,
    ABLATION_NOT_EVALUATED_COUNT,
    ALLOW_FABRICATED_CONVERGENCE,
    ALLOW_FULL_SYSTEM_PERFORMANCE_FABRICATION,
    ALLOW_QML_NONATTAINMENT_CENSORING,
    ALLOW_SYNTHETIC_CROSS_DOMAIN_SCORE,
    ALLOW_SYNTHETIC_METRICS,
    CLAIM_CONTROLS,
    COMPRESSION_MSE_DEGRADATION_THRESHOLD_PERCENT,
    COMPRESSION_RATIO_THRESHOLD,
    DOMAINS,
    FIGURE_FILENAMES,
    FIGURE_IDS,
    FIGURE_SOURCES,
    GENERATED_FROM_FROZEN_EVIDENCE,
    NEW_EXPERIMENTS,
    NEW_SCIENTIFIC_RESULTS,
    NEW_TRAINING,
    QML_TARGET_REACH_COUNTS,
    FigureEvidence,
    build_all_figure_contracts,
    build_figure_contract,
    claim_controls,
    cross_domain_synthetic_score_allowed,
    figure_output_path,
    full_system_performance_fabrication_allowed,
    qml_nonattainment_may_be_censored_to_budget,
    training_target_nonattainment_may_be_fabricated,
    validate_figure_id,
)


def test_exactly_seven_required_figure_ids() -> None:
    assert FIGURE_IDS == (
        "architecture",
        "compression_pareto",
        "training_convergence",
        "rl_sample_efficiency",
        "safety_violations",
        "cross_domain",
        "ablation_summary",
    )


def test_exactly_seven_output_filenames() -> None:
    assert set(FIGURE_FILENAMES) == set(FIGURE_IDS)


@pytest.mark.parametrize(
    "figure_id",
    FIGURE_IDS,
)
def test_every_figure_has_provenance(
    figure_id: str,
) -> None:
    contract = build_figure_contract(figure_id)

    assert contract.source_artifacts

    assert all(source.strip() for source in contract.source_artifacts)


@pytest.mark.parametrize(
    "figure_id",
    FIGURE_IDS,
)
def test_every_figure_has_output_path(
    figure_id: str,
) -> None:
    contract = build_figure_contract(figure_id)

    assert contract.output_path == figure_output_path(figure_id)

    assert contract.output_path.endswith(".png")


@pytest.mark.parametrize(
    "figure_id",
    FIGURE_IDS,
)
def test_every_figure_has_scientific_scope(
    figure_id: str,
) -> None:
    contract = build_figure_contract(figure_id)

    assert contract.scientific_scope


@pytest.mark.parametrize(
    "figure_id",
    FIGURE_IDS,
)
def test_every_figure_retains_limitations(
    figure_id: str,
) -> None:
    contract = build_figure_contract(figure_id)

    assert contract.limitations


@pytest.mark.parametrize(
    "figure_id",
    FIGURE_IDS,
)
def test_every_figure_uses_frozen_evidence(
    figure_id: str,
) -> None:
    contract = build_figure_contract(figure_id)

    assert contract.generated_from_frozen_evidence is True


def test_global_frozen_evidence_flag() -> None:
    assert GENERATED_FROM_FROZEN_EVIDENCE is True


def test_no_new_training() -> None:
    assert NEW_TRAINING is False


def test_no_new_experiments() -> None:
    assert NEW_EXPERIMENTS is False


def test_no_new_scientific_results() -> None:
    assert NEW_SCIENTIFIC_RESULTS is False


def test_synthetic_metrics_blocked() -> None:
    assert ALLOW_SYNTHETIC_METRICS is False


def test_compression_thresholds_frozen() -> None:
    assert COMPRESSION_RATIO_THRESHOLD == 2.0

    assert COMPRESSION_MSE_DEGRADATION_THRESHOLD_PERCENT == 5.0


def test_qml_target_reach_counts_frozen() -> None:
    assert QML_TARGET_REACH_COUNTS["ppo_mlp"] == {
        "reached": 6,
        "total": 6,
    }

    assert QML_TARGET_REACH_COUNTS["ppo_pqc"] == {
        "reached": 0,
        "total": 6,
    }

    assert QML_TARGET_REACH_COUNTS["matched_classical"] == {
        "reached": 1,
        "total": 6,
    }


def test_qml_nonattainment_not_censored_to_budget() -> None:
    assert ALLOW_QML_NONATTAINMENT_CENSORING is False

    assert qml_nonattainment_may_be_censored_to_budget() is False


def test_training_nonattainment_not_fabricated() -> None:
    assert ALLOW_FABRICATED_CONVERGENCE is False

    assert training_target_nonattainment_may_be_fabricated() is False


def test_cross_domain_synthetic_score_blocked() -> None:
    assert ALLOW_SYNTHETIC_CROSS_DOMAIN_SCORE is False

    assert cross_domain_synthetic_score_allowed() is False


def test_full_system_performance_fabrication_blocked() -> None:
    assert ALLOW_FULL_SYSTEM_PERFORMANCE_FABRICATION is False

    assert full_system_performance_fabrication_allowed() is False


def test_sprint7_6_counts_preserved() -> None:
    assert ABLATION_DIRECT_COUNT == 0

    assert ABLATION_COMPONENT_ONLY_COUNT == 16

    assert ABLATION_NOT_EVALUATED_COUNT == 0


def test_architecture_has_integration_boundary_source() -> None:
    assert (
        "results/final-validation/" "full-system-ablation/" "full-system-ablation.json"
    ) in FIGURE_SOURCES["architecture"]


def test_training_figure_uses_frozen_training_summary() -> None:
    assert (
        "results/final-validation/"
        "training-efficiency/"
        "final-training-efficiency-summary.json"
    ) in FIGURE_SOURCES["training_convergence"]


def test_rl_figure_uses_qml_plot_data() -> None:
    assert (
        "results/final-validation/" "qml-ablation/" "qml-ablation-plot-data.csv"
    ) in FIGURE_SOURCES["rl_sample_efficiency"]


def test_safety_figure_uses_safety_plot_data() -> None:
    assert (
        "results/final-validation/" "safety-ablation/" "safety-ablation-plot-data.csv"
    ) in FIGURE_SOURCES["safety_violations"]


def test_ablation_figure_uses_sprint7_6_source() -> None:
    assert (
        "results/final-validation/" "full-system-ablation/" "full-system-ablation.json"
    ) in FIGURE_SOURCES["ablation_summary"]


@pytest.mark.parametrize(
    "figure_id",
    FIGURE_IDS,
)
def test_all_contracts_use_both_domains(
    figure_id: str,
) -> None:
    contract = build_figure_contract(figure_id)

    assert contract.domains == DOMAINS


def test_build_all_contracts_returns_seven() -> None:
    contracts = build_all_figure_contracts()

    assert len(contracts) == 7

    assert {contract.figure_id for contract in contracts} == set(FIGURE_IDS)


def test_invalid_figure_id_rejected() -> None:
    with pytest.raises(ValueError):
        validate_figure_id("invalid")


def test_empty_provenance_rejected() -> None:
    with pytest.raises(ValueError):
        FigureEvidence(
            figure_id="architecture",
            title=("Q-VLA Forge Phase 1 " "Architecture and " "Integration Boundary"),
            source_artifacts=(),
            output_path=("results/final-validation/" "figures/test.png"),
            domains=DOMAINS,
            metrics=("architecture",),
            statistical_protocol=("architecture"),
            scientific_scope="test",
            limitations=("test limitation",),
        )


def test_quantum_advantage_claim_blocked() -> None:
    assert CLAIM_CONTROLS["quantum_advantage_claim_allowed"] is False


def test_quantum_speedup_claim_blocked() -> None:
    assert CLAIM_CONTROLS["quantum_speedup_claim_allowed"] is False


def test_qml_sample_efficiency_superiority_blocked() -> None:
    assert CLAIM_CONTROLS["qml_sample_efficiency_superiority_claim_allowed"] is False


def test_formal_safety_guarantee_blocked() -> None:
    assert CLAIM_CONTROLS["formal_safety_guarantee_claim_allowed"] is False


def test_production_safety_claim_blocked() -> None:
    assert CLAIM_CONTROLS["production_safety_claim_allowed"] is False


def test_full_system_superiority_claim_blocked() -> None:
    assert CLAIM_CONTROLS["full_system_superiority_claim_allowed"] is False


def test_integrated_phase1_pipeline_claim_blocked() -> None:
    assert CLAIM_CONTROLS["integrated_phase1_pipeline_claim_allowed"] is False


def test_all_claim_controls_blocked() -> None:
    controls = claim_controls()

    assert controls

    assert all(value is False for value in controls.values())


def test_claim_controls_returns_copy() -> None:
    controls = claim_controls()

    controls["quantum_advantage_claim_allowed"] = True

    assert CLAIM_CONTROLS["quantum_advantage_claim_allowed"] is False


def test_architecture_scope_does_not_claim_integrated_execution() -> None:
    contract = build_figure_contract("architecture")

    text = " ".join(
        (
            contract.scientific_scope,
            *contract.limitations,
        )
    ).lower()

    assert "does not imply" in text or "integration boundary" in text


def test_safety_scope_uses_observed_language() -> None:
    contract = build_figure_contract("safety_violations")

    text = " ".join(
        (
            contract.scientific_scope,
            *contract.limitations,
        )
    ).lower()

    assert "observed" in text

    assert "formal closed-loop stability proof" in text


def test_cross_domain_scope_has_no_aggregate_score() -> None:
    contract = build_figure_contract("cross_domain")

    text = " ".join(
        (
            contract.scientific_scope,
            *contract.limitations,
        )
    ).lower()

    assert "synthetic aggregate score" in text


def test_ablation_scope_is_evidence_not_performance() -> None:
    contract = build_figure_contract("ablation_summary")

    text = " ".join(
        (
            contract.scientific_scope,
            *contract.limitations,
        )
    ).lower()

    assert "evidence-status" in text or "evidence status" in text

    assert "fabricated" in text
