"""Tests for Sprint 5.12 action robustness contracts."""

from __future__ import annotations

import numpy as np
import pytest

from q_vla_forge.evaluation.action_robustness import (
    ActionRobustnessEpisodeEvidence,
    action_safety_recovery_rate,
    environment_effective_action,
    gripper_semantic,
    summarize_action_robustness_seed,
    within_filter_violation_reduction,
)


def _episode(
    *,
    unsafe: int,
    recovered: int,
    reward: float = 1.0,
) -> ActionRobustnessEpisodeEvidence:
    unresolved = unsafe - recovered

    return ActionRobustnessEpisodeEvidence(
        domain="autonomous_driving",
        method="clipping",
        perturbation_name="steering_plus_0p25",
        perturbation_family="steering",
        principal_seed=42,
        evaluation_seed=20_000,
        reward=reward,
        success=False,
        episode_length=100,
        perturbed_violation_step_count=unsafe,
        executed_violation_step_count=unresolved,
        perturbed_constraint_violation_count=unsafe,
        executed_constraint_violation_count=unresolved,
        critical_violation_step_count=0,
        unsafe_perturbed_steps=unsafe,
        recovered_unsafe_steps=recovered,
        unresolved_unsafe_steps=unresolved,
        intervention_count=recovered,
        action_perturbation_l2_sum=25.0,
        action_perturbation_linf_max=0.25,
        safety_correction_l2_sum=5.0,
        safety_correction_l2_values=(
            0.0,
            0.1,
            0.2,
            0.3,
        ),
        safety_correction_l2_max=0.3,
        total_action_displacement_l2_sum=20.0,
    )


def test_recovery_rate() -> None:
    assert action_safety_recovery_rate(
        unsafe_perturbed_steps=10,
        recovered_unsafe_steps=7,
    ) == pytest.approx(0.7)


def test_zero_recovery_denominator_is_none() -> None:
    assert (
        action_safety_recovery_rate(
            unsafe_perturbed_steps=0,
            recovered_unsafe_steps=0,
        )
        is None
    )


def test_recovered_cannot_exceed_unsafe() -> None:
    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        action_safety_recovery_rate(
            unsafe_perturbed_steps=2,
            recovered_unsafe_steps=3,
        )


def test_within_filter_reduction() -> None:
    assert within_filter_violation_reduction(
        perturbed_violation_rate=0.5,
        executed_violation_rate=0.1,
    ) == pytest.approx(0.8)


def test_zero_perturbed_rate_returns_none() -> None:
    assert (
        within_filter_violation_reduction(
            perturbed_violation_rate=0.0,
            executed_violation_rate=0.0,
        )
        is None
    )


def test_episode_recovery_accounting_validation() -> None:
    with pytest.raises(
        ValueError,
        match="recovery accounting",
    ):
        ActionRobustnessEpisodeEvidence(
            domain="robotics",
            method="none",
            perturbation_name="delta_x_plus_0p10",
            perturbation_family="delta_x",
            principal_seed=42,
            evaluation_seed=20_000,
            reward=0.0,
            success=False,
            episode_length=10,
            perturbed_violation_step_count=2,
            executed_violation_step_count=2,
            perturbed_constraint_violation_count=2,
            executed_constraint_violation_count=2,
            critical_violation_step_count=0,
            unsafe_perturbed_steps=2,
            recovered_unsafe_steps=2,
            unresolved_unsafe_steps=1,
            intervention_count=0,
            action_perturbation_l2_sum=1.0,
            action_perturbation_linf_max=0.10,
        )


def test_seed_summary_recovery() -> None:
    summary = summarize_action_robustness_seed(
        [
            _episode(
                unsafe=10,
                recovered=7,
                reward=1.0,
            ),
            _episode(
                unsafe=10,
                recovered=3,
                reward=3.0,
            ),
        ]
    )

    assert summary.mean_reward == pytest.approx(2.0)

    assert summary.unsafe_perturbed_steps == 20
    assert summary.recovered_unsafe_steps == 10
    assert summary.unresolved_unsafe_steps == 10

    assert summary.recovery_rate == pytest.approx(0.5)

    assert summary.perturbed_violation_step_rate == pytest.approx(0.10)

    assert summary.executed_violation_step_rate == pytest.approx(0.05)

    assert summary.within_filter_violation_reduction == pytest.approx(0.5)


@pytest.mark.parametrize(
    (
        "value",
        "expected",
    ),
    [
        (
            0.5000,
            "hold",
        ),
        (
            0.5001,
            "close",
        ),
        (
            -0.5000,
            "hold",
        ),
        (
            -0.5001,
            "open",
        ),
        (
            0.0,
            "hold",
        ),
    ],
)
def test_gripper_semantics(
    value: float,
    expected: str,
) -> None:
    assert gripper_semantic(value) == expected


def test_none_style_no_recovery() -> None:
    summary = summarize_action_robustness_seed(
        [
            _episode(
                unsafe=4,
                recovered=0,
            )
        ]
    )

    assert summary.recovery_rate == 0.0

    assert summary.perturbed_violation_step_rate == summary.executed_violation_step_rate


def test_environment_effective_action_driving() -> None:
    executed = np.asarray(
        [
            1.25,
            0.40,
            -0.25,
        ],
        dtype=np.float32,
    )

    effective = environment_effective_action(
        executed_action=executed,
        lower_bounds=np.asarray(
            [
                -1.0,
                -1.0,
                0.0,
            ],
            dtype=np.float32,
        ),
        upper_bounds=np.asarray(
            [
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        effective,
        np.asarray(
            [
                1.0,
                0.40,
                0.0,
            ],
            dtype=np.float32,
        ),
    )

    assert executed[0] == pytest.approx(1.25)

    assert executed[2] == pytest.approx(-0.25)


def test_environment_effective_action_robotics() -> None:
    executed = np.asarray(
        [
            -1.25,
            0.0,
            1.50,
        ],
        dtype=np.float32,
    )

    effective = environment_effective_action(
        executed_action=executed,
        lower_bounds=np.full(
            3,
            -1.0,
            dtype=np.float32,
        ),
        upper_bounds=np.full(
            3,
            1.0,
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        effective,
        np.asarray(
            [
                -1.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        ),
    )


def test_environment_interface_is_not_filter_recovery() -> None:
    assert (
        action_safety_recovery_rate(
            unsafe_perturbed_steps=4,
            recovered_unsafe_steps=0,
        )
        == 0.0
    )
