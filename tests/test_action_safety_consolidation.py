"""Tests for Sprint 5.13E action robustness consolidation."""

import pytest

from q_vla_forge.evaluation.action_safety_consolidation import (
    ActionSeedResult,
    summarize_rows,
    summarize_seed_values,
    worst_condition,
)


def _row(
    seed: int,
    perturbed: float,
    executed: float,
) -> ActionSeedResult:
    return ActionSeedResult(
        domain="autonomous_driving",
        method="lyapunov",
        perturbation_family="steering",
        perturbation_name="steering_plus_0p25",
        principal_seed=seed,
        perturbed_violation_step_rate=perturbed,
        executed_violation_step_rate=executed,
        executed_constraint_violation_rate=executed,
        critical_violation_step_rate=0.0,
        mean_reward=1.0,
        success_rate=0.0,
        intervention_rate=0.1,
        mean_safety_correction_l2=0.01,
        p95_safety_correction_l2=0.02,
        recovery_rate=0.5,
        within_filter_violation_reduction=0.5,
        executed_violation_delta_from_clean=executed,
        reward_delta_from_clean=0.0,
        success_delta_from_clean=0.0,
        selected_lower_than_perturbed_rate=0.0,
        strict_lyapunov_decrease_rate=0.0,
        lyapunov_nonincrease_rate=1.0,
        environment_interface_adjustment_rate=0.0,
        proposed_to_perturbed_gripper_semantic_change_rate=0.0,
        perturbed_to_executed_gripper_semantic_change_rate=0.0,
        unsafe_perturbed_steps=10,
        recovered_unsafe_steps=5,
        unresolved_unsafe_steps=5,
        steps_object_grasped=0,
        interventions_while_grasped=0,
        perturbed_unsafe_steps_while_grasped=0,
    )


def test_seed_summary() -> None:
    summary = summarize_seed_values(
        {
            42: 1.0,
            123: 2.0,
            456: 3.0,
        }
    )

    assert summary.mean == pytest.approx(2.0)

    assert summary.sample_sd == pytest.approx(1.0)


def test_action_summary() -> None:
    summary = summarize_rows(
        [
            _row(
                42,
                0.3,
                0.1,
            ),
            _row(
                123,
                0.6,
                0.2,
            ),
            _row(
                456,
                0.9,
                0.3,
            ),
        ]
    )

    assert summary["perturbed_violation_step_rate"].mean == pytest.approx(0.6)

    assert summary["executed_violation_step_rate"].mean == pytest.approx(0.2)


def test_missing_seed_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="three principal seeds",
    ):
        summarize_rows(
            [
                _row(
                    42,
                    0.1,
                    0.0,
                ),
                _row(
                    123,
                    0.2,
                    0.0,
                ),
            ]
        )


def test_worst_condition_max() -> None:
    rows = [
        {
            "perturbation_name": "a",
            "executed_violation_delta_from_clean": {"mean": 0.1},
        },
        {
            "perturbation_name": "b",
            "executed_violation_delta_from_clean": {"mean": 0.4},
        },
    ]

    result = worst_condition(
        rows,
        metric="executed_violation_delta_from_clean",
        mode="max",
    )

    assert result["perturbation_name"] == "b"


def test_worst_condition_min() -> None:
    rows = [
        {
            "perturbation_name": "a",
            "reward_delta_from_clean": {"mean": -0.1},
        },
        {
            "perturbation_name": "b",
            "reward_delta_from_clean": {"mean": -0.5},
        },
    ]

    result = worst_condition(
        rows,
        metric="reward_delta_from_clean",
        mode="min",
    )

    assert result["perturbation_name"] == "b"
