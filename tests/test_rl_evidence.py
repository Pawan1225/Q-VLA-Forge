from __future__ import annotations

from q_vla_forge.evaluation.rl_evidence import (
    ProposalCompactnessSummary,
    ProposalMethodSummary,
    ProposalRepresentationSummary,
    compactness_summary_to_dict,
    method_summary_to_dict,
    proposal_status,
    representation_summary_to_dict,
)


def test_supported_status() -> None:
    assert (
        proposal_status(
            supported=True,
        )
        == "supported"
    )


def test_supported_with_limitation() -> None:
    assert (
        proposal_status(
            supported=True,
            limitation=True,
        )
        == "supported_with_limitation"
    )


def test_not_supported_status() -> None:
    assert (
        proposal_status(
            supported=False,
        )
        == "not_supported"
    )


def test_method_summary_to_dict() -> None:
    summary = ProposalMethodSummary(
        domain="autonomous_driving",
        method="hybrid_qml",
        actor_parameters=54,
        target_reach_count=0,
        target_total=3,
        normalized_auc_mean=0.15991195170764086,
        normalized_auc_sd=0.4341965431725851,
        best_progress_mean=0.44182728158311746,
        best_progress_sd=0.1,
        final_progress_mean=0.3272891488880778,
        final_progress_sd=0.1,
    )

    result = method_summary_to_dict(summary)

    assert result["domain"] == "autonomous_driving"
    assert result["method"] == "hybrid_qml"
    assert result["actor_parameters"] == 54
    assert result["target_reach_count"] == 0


def test_compactness_summary_to_dict() -> None:
    summary = ProposalCompactnessSummary(
        domain="robotics",
        full_actor_parameters=1382,
        compact_actor_parameters=62,
        parameter_reduction_percent=95.5137481910275,
    )

    result = compactness_summary_to_dict(summary)

    assert result["domain"] == "robotics"
    assert result["full_actor_parameters"] == 1382
    assert result["compact_actor_parameters"] == 62


def test_representation_summary_to_dict() -> None:
    summary = ProposalRepresentationSummary(
        domain="robotics",
        matched_actor_parameters=62,
        qml_actor_parameters=62,
        matched_auc_mean=0.653060017016295,
        qml_auc_mean=0.5739091206532141,
        matched_minus_qml_auc=0.07915089636308092,
        direction="matched_classical",
    )

    result = representation_summary_to_dict(summary)

    assert result["domain"] == "robotics"
    assert result["matched_actor_parameters"] == 62
    assert result["qml_actor_parameters"] == 62
    assert result["direction"] == "matched_classical"
