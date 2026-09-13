from __future__ import annotations

import pytest

from q_vla_forge.evaluation.rl_ablation import (
    PairedAblationRecord,
    paired_delta,
    paired_win_counts,
    record_to_dict,
    sample_mean_sd,
    validate_parameter_match,
)


def test_paired_delta_positive_when_matched_is_better() -> None:
    assert paired_delta(
        matched_value=0.8,
        qml_value=0.5,
    ) == pytest.approx(0.3)


def test_paired_delta_negative_when_qml_is_better() -> None:
    assert paired_delta(
        matched_value=0.4,
        qml_value=0.6,
    ) == pytest.approx(-0.2)


def test_paired_win_counts() -> None:
    counts = paired_win_counts(
        (
            0.5,
            -0.2,
            0.0,
            0.1,
        )
    )

    assert counts["matched_classical_wins"] == 2

    assert counts["qml_wins"] == 1

    assert counts["ties"] == 1


def test_sample_mean_sd() -> None:
    mean, sd = sample_mean_sd(
        (
            1.0,
            2.0,
            3.0,
        )
    )

    assert mean == pytest.approx(2.0)

    assert sd == pytest.approx(1.0)


def test_sample_mean_sd_single_value() -> None:
    mean, sd = sample_mean_sd((2.5,))

    assert mean == pytest.approx(2.5)

    assert sd == pytest.approx(0.0)


def test_sample_mean_sd_rejects_empty_values() -> None:
    with pytest.raises(
        ValueError,
        match="values cannot be empty",
    ):
        sample_mean_sd(())


def test_parameter_match_accepts_equal_counts() -> None:
    validate_parameter_match(
        matched_actor_parameters=54,
        qml_actor_parameters=54,
    )


def test_parameter_match_rejects_unequal_counts() -> None:
    with pytest.raises(
        RuntimeError,
        match="Ablation is not parameter matched",
    ):
        validate_parameter_match(
            matched_actor_parameters=54,
            qml_actor_parameters=55,
        )


def test_record_to_dict() -> None:
    record = PairedAblationRecord(
        domain="autonomous_driving",
        seed=42,
        matched_actor_parameters=54,
        qml_actor_parameters=54,
        matched_target_reached=False,
        qml_target_reached=False,
        matched_normalized_auc=0.4,
        qml_normalized_auc=0.2,
        normalized_auc_delta=0.2,
        matched_best_progress=0.9,
        qml_best_progress=0.7,
        best_progress_delta=0.2,
        matched_final_progress=0.8,
        qml_final_progress=0.6,
        final_progress_delta=0.2,
    )

    payload = record_to_dict(record)

    assert payload["domain"] == "autonomous_driving"

    assert payload["seed"] == 42

    assert payload["matched_actor_parameters"] == 54

    assert payload["qml_actor_parameters"] == 54
