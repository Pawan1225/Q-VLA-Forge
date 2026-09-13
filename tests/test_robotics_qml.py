from __future__ import annotations

import pytest

from q_vla_forge.evaluation.robotics_qml import (
    sample_efficiency_improvement_percent,
)


def test_positive_robotics_qml_improvement() -> None:
    result = sample_efficiency_improvement_percent(
        classical_steps=9_000,
        qml_steps=6_000,
    )

    assert result == pytest.approx(33.33333333333333)


def test_equal_robotics_steps_give_zero() -> None:
    result = sample_efficiency_improvement_percent(
        classical_steps=9_000,
        qml_steps=9_000,
    )

    assert result == pytest.approx(0.0)


def test_slower_robotics_qml_is_negative() -> None:
    result = sample_efficiency_improvement_percent(
        classical_steps=9_000,
        qml_steps=12_000,
    )

    assert result == pytest.approx(-33.33333333333333)


def test_failed_robotics_target_is_none() -> None:
    result = sample_efficiency_improvement_percent(
        classical_steps=9_000,
        qml_steps=None,
    )

    assert result is None


def test_classical_steps_must_be_positive() -> None:
    with pytest.raises(ValueError):
        sample_efficiency_improvement_percent(
            classical_steps=0,
            qml_steps=5_000,
        )
