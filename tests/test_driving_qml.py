from __future__ import annotations

import pytest

from q_vla_forge.evaluation.driving_qml import (
    sample_efficiency_improvement_percent,
)


def test_positive_sample_efficiency_improvement() -> None:
    result = sample_efficiency_improvement_percent(
        classical_steps=12_000,
        qml_steps=9_000,
    )

    assert result == pytest.approx(25.0)


def test_equal_steps_give_zero_improvement() -> None:
    result = sample_efficiency_improvement_percent(
        classical_steps=12_000,
        qml_steps=12_000,
    )

    assert result == pytest.approx(0.0)


def test_more_qml_steps_give_negative_improvement() -> None:
    result = sample_efficiency_improvement_percent(
        classical_steps=12_000,
        qml_steps=15_000,
    )

    assert result == pytest.approx(-25.0)


def test_failed_qml_target_returns_none() -> None:
    result = sample_efficiency_improvement_percent(
        classical_steps=12_000,
        qml_steps=None,
    )

    assert result is None
