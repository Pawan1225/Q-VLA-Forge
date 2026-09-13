from __future__ import annotations

import pytest

from q_vla_forge.evaluation.rl_sample_efficiency import (
    actor_parameter_reduction_percent,
    normalized_learning_auc,
    normalized_target_progress,
    trapezoidal_auc,
)


def test_random_reference_maps_to_zero() -> None:
    result = normalized_target_progress(
        reward=-10.0,
        random_reference=-10.0,
        target_reward=0.0,
    )

    assert result == pytest.approx(0.0)


def test_target_maps_to_one() -> None:
    result = normalized_target_progress(
        reward=0.0,
        random_reference=-10.0,
        target_reward=0.0,
    )

    assert result == pytest.approx(1.0)


def test_midpoint_maps_to_half() -> None:
    result = normalized_target_progress(
        reward=-5.0,
        random_reference=-10.0,
        target_reward=0.0,
    )

    assert result == pytest.approx(0.5)


def test_above_target_exceeds_one() -> None:
    result = normalized_target_progress(
        reward=5.0,
        random_reference=-10.0,
        target_reward=0.0,
    )

    assert result == pytest.approx(1.5)


def test_invalid_target_reference_rejected() -> None:
    with pytest.raises(ValueError):
        normalized_target_progress(
            reward=0.0,
            random_reference=1.0,
            target_reward=1.0,
        )


def test_trapezoidal_auc() -> None:
    result = trapezoidal_auc(
        steps=[
            0,
            10,
            20,
        ],
        values=[
            0.0,
            1.0,
            1.0,
        ],
    )

    assert result == pytest.approx(15.0)


def test_normalized_auc() -> None:
    result = normalized_learning_auc(
        steps=[
            0,
            10,
            20,
        ],
        normalized_progress=[
            0.0,
            1.0,
            1.0,
        ],
        total_environment_steps=20,
    )

    assert result == pytest.approx(0.75)


def test_driving_parameter_reduction() -> None:
    result = actor_parameter_reduction_percent(
        classical_parameters=1318,
        hybrid_parameters=54,
    )

    assert result == pytest.approx(95.90288315629742)


def test_robotics_parameter_reduction() -> None:
    result = actor_parameter_reduction_percent(
        classical_parameters=1382,
        hybrid_parameters=62,
    )

    assert result == pytest.approx(95.5137481910275)
