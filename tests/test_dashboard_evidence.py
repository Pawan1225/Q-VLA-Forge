from __future__ import annotations

from pathlib import Path

import pytest

from q_vla_forge.evaluation.dashboard_evidence import (
    ABLATION_AREAS,
    ALGORITHMS_BY_BOTTLENECK,
    CHALLENGE_BOTTLENECKS,
    DOMAINS,
    MISSING_CANONICAL_EVIDENCE,
    REQUIRED_FIGURE_IDS,
    REQUIRED_SEEDS,
    REQUIRED_TABLE_IDS,
    MissingCanonicalEvidenceError,
    load_dashboard_evidence,
    load_figure_manifest,
    load_full_system_ablation,
    load_table_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evidence() -> dict:
    return load_dashboard_evidence(ROOT)


def test_four_challenge_bottlenecks() -> None:
    assert CHALLENGE_BOTTLENECKS == (
        "compression",
        "training_efficiency",
        "rl_alignment",
        "safety",
    )


def test_two_domains() -> None:
    assert DOMAINS == (
        "autonomous_driving",
        "robotics",
    )


def test_locked_seeds() -> None:
    assert REQUIRED_SEEDS == (
        42,
        123,
        456,
    )


def test_algorithm_families_are_complete() -> None:
    assert ALGORITHMS_BY_BOTTLENECK["compression"] == (
        "FP32",
        "INT8",
        "SVD",
        "TT/MPS",
    )

    assert ALGORITHMS_BY_BOTTLENECK["training_efficiency"] == (
        "baseline",
        "SVD",
        "TT/MPS",
    )

    assert ALGORITHMS_BY_BOTTLENECK["rl_alignment"] == (
        "PPO / MLP",
        "matched classical",
        "QML / PQC",
    )

    assert ALGORITHMS_BY_BOTTLENECK["safety"] == (
        "NONE",
        "CLIPPING",
        "LYAPUNOV",
    )


def test_four_ablation_areas() -> None:
    assert ABLATION_AREAS == (
        "compression",
        "qml",
        "safety",
        "full_system",
    )


def test_dashboard_evidence_is_frozen(
    evidence: dict,
) -> None:
    assert evidence["status"] == "FROZEN"

    assert evidence["phase"] == "Phase 1"

    assert evidence["generated_from_frozen_evidence"] is True


def test_dashboard_introduces_no_new_science(
    evidence: dict,
) -> None:
    assert evidence["new_training"] is False

    assert evidence["new_experiments"] is False

    assert evidence["new_scientific_results"] is False

    assert evidence["synthetic_metric_composition"] is False


def test_figure_manifest_is_frozen() -> None:
    manifest = load_figure_manifest(ROOT)

    assert manifest["status"] == "FROZEN"

    assert manifest["figure_count"] == 7


def test_seven_final_figures_registered(
    evidence: dict,
) -> None:
    figures = evidence["figures"]

    assert len(figures) == 7

    assert tuple(item["figure_id"] for item in figures) == REQUIRED_FIGURE_IDS


def test_every_final_figure_exists(
    evidence: dict,
) -> None:
    for figure in evidence["figures"]:
        path = ROOT / str(figure["output_path"]).replace(
            "\\",
            "/",
        )

        assert path.is_file()
        assert path.stat().st_size > 0


def test_table_manifest_is_frozen() -> None:
    manifest = load_table_manifest(ROOT)

    assert manifest["status"] == "FROZEN"

    assert manifest["table_count"] == 5


def test_five_final_tables_registered(
    evidence: dict,
) -> None:
    assert tuple(evidence["tables"]) == REQUIRED_TABLE_IDS


def test_final_table_row_counts(
    evidence: dict,
) -> None:
    assert len(evidence["tables"]["compression"]) == 8

    assert len(evidence["tables"]["training"]) == 4

    assert len(evidence["tables"]["rl"]) == 6

    assert len(evidence["tables"]["safety"]) == 6

    assert len(evidence["tables"]["cross_domain"]) == 9


def test_compression_outcomes_are_correct(
    evidence: dict,
) -> None:
    outcomes = evidence["criterion_outcomes"]["compression"]

    assert outcomes["INT8"] == {
        "Driving": "PASS",
        "Robotics": "PASS",
    }

    assert outcomes["SVD"] == {
        "Driving": "FAIL",
        "Robotics": "FAIL",
    }

    assert outcomes["TT/MPS"] == {
        "Driving": "FAIL",
        "Robotics": "FAIL",
    }


def test_training_negative_result_retained(
    evidence: dict,
) -> None:
    training = evidence["criterion_outcomes"]["training_efficiency"]

    assert training["phase1_result"] == "NOT DEMONSTRATED"

    assert training["threshold"] == ">=10% robust efficiency improvement"


def test_ppo_target_reaches_are_six_of_six(
    evidence: dict,
) -> None:
    reaches = evidence["criterion_outcomes"]["rl_alignment"]["target_reaches"]

    assert reaches["Classical PPO / MLP"] == {
        "reached": 6,
        "total": 6,
    }


def test_qml_target_reaches_are_zero_of_six(
    evidence: dict,
) -> None:
    reaches = evidence["criterion_outcomes"]["rl_alignment"]["target_reaches"]

    assert reaches["QML / PQC"] == {
        "reached": 0,
        "total": 6,
    }


def test_matched_classical_reaches_one_of_six(
    evidence: dict,
) -> None:
    reaches = evidence["criterion_outcomes"]["rl_alignment"]["target_reaches"]

    assert reaches["Matched classical control"] == {
        "reached": 1,
        "total": 6,
    }


def test_pqc_compactness_is_separate(
    evidence: dict,
) -> None:
    rl = evidence["criterion_outcomes"]["rl_alignment"]

    assert rl["pqc_compactness"] == {
        "Driving": "95.90%",
        "Robotics": "95.51%",
    }

    assert rl["target_reaches"]["QML / PQC"]["reached"] == 0


def test_safety_zero_observed_semantics(
    evidence: dict,
) -> None:
    safety = evidence["criterion_outcomes"]["safety"]

    for domain in (
        "Driving",
        "Robotics",
    ):
        assert safety[domain]["CLIPPING"].startswith("0.000")

        assert safety[domain]["LYAPUNOV"].startswith("0.000")


def test_full_system_ablation_is_frozen() -> None:
    full_system = load_full_system_ablation(ROOT)

    assert full_system["status"] == "FROZEN"

    assert full_system["phase1"]["direct_full_system_execution_found"] is False


def test_direct_count_remains_zero(
    evidence: dict,
) -> None:
    assert evidence["full_system_status"]["DIRECT"] == 0


def test_component_only_count_remains_sixteen(
    evidence: dict,
) -> None:
    assert evidence["full_system_status"]["COMPONENT_ONLY"] == 16


def test_not_evaluated_count_remains_zero(
    evidence: dict,
) -> None:
    assert evidence["full_system_status"]["NOT_EVALUATED"] == 0


def test_no_synthetic_factorial_metrics(
    evidence: dict,
) -> None:
    controls = evidence["full_system"]["scientific_controls"]

    assert controls["synthetic_metric_composition_allowed"] is False

    assert controls["interaction_effects_estimable_with_component_only"] is False

    assert controls["full_system_end_to_end_metrics_reported"] is False


def test_final_claim_registry_is_present(
    evidence: dict,
) -> None:
    claims = evidence["claim_registry"]

    assert len(claims) >= 9

    ids = {row["claim_id"] for row in claims}

    assert {
        "S7-C02",
        "S7-C03",
        "S7-C04",
        "S7-C05",
        "S7-C06",
        "S7-C09",
    }.issubset(ids)


def test_bottleneck_scorecard_has_four_rows(
    evidence: dict,
) -> None:
    scorecard = evidence["bottleneck_scorecard"]

    assert len(scorecard) == 4

    assert {row["bottleneck"] for row in scorecard} == {
        "model_footprint",
        "training_efficiency",
        "rl_alignment_sample_efficiency",
        "safety",
    }


def test_training_scorecard_is_not_demonstrated(
    evidence: dict,
) -> None:
    scorecard = {row["bottleneck"]: row for row in evidence["bottleneck_scorecard"]}

    assert scorecard["training_efficiency"]["phase1_status"] == "not_demonstrated"


def test_claim_controls_all_remain_blocked(
    evidence: dict,
) -> None:
    controls = evidence["claim_controls"]

    assert controls
    assert all(value is False for value in controls.values())


def test_quantum_advantage_is_blocked(
    evidence: dict,
) -> None:
    controls = evidence["claim_controls"]

    assert controls["quantum_advantage_claim_allowed"] is False


def test_quantum_speedup_is_blocked(
    evidence: dict,
) -> None:
    controls = evidence["claim_controls"]

    assert controls["quantum_speedup_claim_allowed"] is False


def test_full_system_superiority_is_blocked(
    evidence: dict,
) -> None:
    controls = evidence["claim_controls"]

    assert controls["full_system_superiority_claim_allowed"] is False


def test_production_claim_is_blocked(
    evidence: dict,
) -> None:
    controls = evidence["claim_controls"]

    production_controls = [
        value for key, value in controls.items() if ("production" in key)
    ]

    assert production_controls
    assert all(value is False for value in production_controls)


def test_provenance_is_available(
    evidence: dict,
) -> None:
    provenance = evidence["provenance"]

    assert {
        "figure_manifest",
        "table_manifest",
        "claim_registry",
        "bottleneck_scorecard",
        "component_ablation",
        "full_system_ablation",
    } == set(provenance)


def test_component_ablation_is_available(
    evidence: dict,
) -> None:
    rows = evidence["component_ablation"]

    areas = {row["area"] for row in rows}

    assert "compression" in areas
    assert "training_efficiency" in areas
    assert "rl_qml" in areas
    assert "safety" in areas
    assert "cross_domain" in areas


def test_missing_canonical_evidence_fails_loudly(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        MissingCanonicalEvidenceError,
        match=MISSING_CANONICAL_EVIDENCE,
    ):
        load_dashboard_evidence(tmp_path)
