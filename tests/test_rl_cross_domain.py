from __future__ import annotations

import pytest

from q_vla_forge.evaluation.rl_cross_domain import (
    CrossDomainMethodRecord,
    MatchedRepresentationComparison,
    matched_comparison_to_dict,
    parameter_reduction_percent,
    record_to_dict,
    representation_direction,
)


def test_qml_direction() -> None:
    result = representation_direction(
        matched_value=0.1,
        qml_value=0.2,
    )

    assert result == "hybrid_qml"


def test_matched_direction() -> None:
    result = representation_direction(
        matched_value=0.7,
        qml_value=0.6,
    )

    assert result == "matched_classical"


def test_tie_direction() -> None:
    result = representation_direction(
        matched_value=0.5,
        qml_value=0.5,
    )

    assert result == "tie"


def test_tie_direction_respects_tolerance() -> None:
    result = representation_direction(
        matched_value=0.5000000000001,
        qml_value=0.5,
        tolerance=1e-12,
    )

    assert result == "tie"


def test_negative_tolerance_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="tolerance cannot be negative",
    ):
        representation_direction(
            matched_value=0.5,
            qml_value=0.5,
            tolerance=-1.0,
        )


def test_driving_compactness() -> None:
    result = parameter_reduction_percent(
        full_actor_parameters=1318,
        compact_actor_parameters=54,
    )

    assert result == pytest.approx(95.90288315629742)


def test_robotics_compactness() -> None:
    result = parameter_reduction_percent(
        full_actor_parameters=1382,
        compact_actor_parameters=62,
    )

    assert result == pytest.approx(95.5137481910275)


def test_nonpositive_full_actor_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="full actor parameters must be positive",
    ):
        parameter_reduction_percent(
            full_actor_parameters=0,
            compact_actor_parameters=10,
        )


def test_nonpositive_compact_actor_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="compact actor parameters must be positive",
    ):
        parameter_reduction_percent(
            full_actor_parameters=100,
            compact_actor_parameters=0,
        )


def test_compact_actor_cannot_exceed_full_actor() -> None:
    with pytest.raises(
        ValueError,
        match="compact actor cannot exceed full actor size",
    ):
        parameter_reduction_percent(
            full_actor_parameters=100,
            compact_actor_parameters=101,
        )


def test_method_record_to_dict() -> None:
    record = CrossDomainMethodRecord(
        domain="autonomous_driving",
        method="full_ppo",
        actor_parameters=1318,
        target_reach_count=3,
        target_total=3,
        normalized_auc_mean=0.554811,
        normalized_auc_sd=0.264606,
        best_progress_mean=1.052632,
        best_progress_sd=0.0,
        final_progress_mean=0.995282,
        final_progress_sd=0.053598,
    )

    payload = record_to_dict(record)

    assert payload["domain"] == "autonomous_driving"
    assert payload["method"] == "full_ppo"
    assert payload["actor_parameters"] == 1318
    assert payload["target_reach_count"] == 3


def test_matched_comparison_to_dict() -> None:
    comparison = MatchedRepresentationComparison(
        domain="robotics",
        matched_actor_parameters=62,
        qml_actor_parameters=62,
        matched_auc_mean=0.653060,
        qml_auc_mean=0.573909,
        matched_minus_qml_auc=0.079151,
        direction="matched_classical",
    )

    payload = matched_comparison_to_dict(comparison)

    assert payload["domain"] == "robotics"
    assert payload["matched_actor_parameters"] == 62
    assert payload["qml_actor_parameters"] == 62
    assert payload["direction"] == "matched_classical"
