"""Tests for Sprint 5.9 robotics Lyapunov evaluation."""

from __future__ import annotations

import pytest

from q_vla_forge.evaluation.robotics_lyapunov_safety import (
    MAXIMUM_REWARD_DEGRADATION_FRACTION,
    MAXIMUM_SUCCESS_RATE_DROP,
    MINIMUM_MEAN_VIOLATION_REDUCTION,
    ROBOTICS_CANDIDATE_SOURCES,
    domain_mean_violation_reduction,
    empirical_effectiveness_supported,
    relative_violation_reduction,
    reward_degradation_fraction,
    seed_safety_requirement_passes,
)


def test_robotics_candidate_sources_are_frozen() -> None:
    assert ROBOTICS_CANDIDATE_SOURCES == (
        "bounded_proposed",
        "hold_gripper",
        "x_only",
        "y_only",
        "stop_preserve_gripper",
        "full_stop",
        "inward_motion",
        "inward_motion_hold_gripper",
        "inward_x_hold_gripper",
    )


@pytest.mark.parametrize(
    (
        "none_rate",
        "lyapunov_rate",
        "expected",
    ),
    [
        (
            0.0355,
            0.0,
            True,
        ),
        (
            0.0085,
            0.0,
            True,
        ),
        (
            0.0,
            0.0,
            True,
        ),
        (
            0.0355,
            0.0355,
            False,
        ),
        (
            0.0,
            0.001,
            False,
        ),
    ],
)
def test_seed_safety_requirement(
    none_rate: float,
    lyapunov_rate: float,
    expected: bool,
) -> None:
    assert (
        seed_safety_requirement_passes(
            none_rate=none_rate,
            lyapunov_rate=lyapunov_rate,
        )
        is expected
    )


def test_relative_violation_reduction() -> None:
    assert relative_violation_reduction(
        baseline_rate=0.02,
        candidate_rate=0.01,
    ) == pytest.approx(0.5)


def test_zero_baseline_no_worsening_reduction() -> None:
    assert relative_violation_reduction(
        baseline_rate=0.0,
        candidate_rate=0.0,
    ) == pytest.approx(0.0)


def test_positive_reward_no_degradation() -> None:
    assert reward_degradation_fraction(
        baseline_reward=0.60,
        candidate_reward=0.70,
    ) == pytest.approx(0.0)


def test_positive_reward_degradation() -> None:
    assert reward_degradation_fraction(
        baseline_reward=1.0,
        candidate_reward=0.95,
    ) == pytest.approx(0.05)


def test_domain_reduction() -> None:
    assert domain_mean_violation_reduction(
        none_mean_rate=0.014666666666666666,
        lyapunov_mean_rate=0.0,
    ) == pytest.approx(1.0)


def test_effectiveness_supported() -> None:
    assert empirical_effectiveness_supported(
        seed_requirements_pass=True,
        mean_violation_reduction=(MINIMUM_MEAN_VIOLATION_REDUCTION),
        reward_degradation=(MAXIMUM_REWARD_DEGRADATION_FRACTION),
        success_rate_drop=(MAXIMUM_SUCCESS_RATE_DROP),
    )


def test_effectiveness_rejects_seed_failure() -> None:
    assert not empirical_effectiveness_supported(
        seed_requirements_pass=False,
        mean_violation_reduction=1.0,
        reward_degradation=0.0,
        success_rate_drop=0.0,
    )


def test_effectiveness_rejects_low_reduction() -> None:
    assert not empirical_effectiveness_supported(
        seed_requirements_pass=True,
        mean_violation_reduction=0.19,
        reward_degradation=0.0,
        success_rate_drop=0.0,
    )


def test_effectiveness_rejects_reward_degradation() -> None:
    assert not empirical_effectiveness_supported(
        seed_requirements_pass=True,
        mean_violation_reduction=1.0,
        reward_degradation=0.100001,
        success_rate_drop=0.0,
    )


def test_effectiveness_rejects_success_drop() -> None:
    assert not empirical_effectiveness_supported(
        seed_requirements_pass=True,
        mean_violation_reduction=1.0,
        reward_degradation=0.0,
        success_rate_drop=0.100001,
    )
