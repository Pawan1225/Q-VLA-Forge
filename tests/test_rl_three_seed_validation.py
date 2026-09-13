from __future__ import annotations

import pytest

from q_vla_forge.evaluation.rl_three_seed_validation import (
    EXPECTED_EVALUATION_STEPS,
    sample_efficiency_improvement,
    sample_mean_sd,
)


def test_expected_evaluation_schedule() -> None:
    assert len(EXPECTED_EVALUATION_STEPS) == 21

    assert EXPECTED_EVALUATION_STEPS[0] == 0

    assert EXPECTED_EVALUATION_STEPS[-1] == 20_000


def test_sample_mean_sd() -> None:
    mean, sd = sample_mean_sd(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    assert mean == pytest.approx(2.0)

    assert sd == pytest.approx(1.0)


def test_positive_efficiency() -> None:
    result = sample_efficiency_improvement(
        classical_steps=10_000,
        qml_steps=8_000,
    )

    assert result == pytest.approx(20.0)


def test_negative_efficiency() -> None:
    result = sample_efficiency_improvement(
        classical_steps=10_000,
        qml_steps=12_000,
    )

    assert result == pytest.approx(-20.0)


def test_failed_target_preserves_none() -> None:
    result = sample_efficiency_improvement(
        classical_steps=10_000,
        qml_steps=None,
    )

    assert result is None


def test_invalid_classical_steps_rejected() -> None:
    with pytest.raises(ValueError):
        sample_efficiency_improvement(
            classical_steps=0,
            qml_steps=5_000,
        )


def test_empty_statistics_rejected() -> None:
    with pytest.raises(ValueError):
        sample_mean_sd([])
