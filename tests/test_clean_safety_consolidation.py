"""Tests for Sprint 5.13B clean safety consolidation."""

import pytest

from q_vla_forge.evaluation.clean_safety_consolidation import (
    SeedCleanResult,
    effectiveness_gate,
    relative_reduction,
    reward_degradation_fraction,
    summarize_results,
)


def test_relative_reduction() -> None:
    assert relative_reduction(
        0.5,
        0.0,
    ) == pytest.approx(1.0)


def test_zero_baseline_reduction() -> None:
    assert (
        relative_reduction(
            0.0,
            0.0,
        )
        is None
    )


def test_reward_improvement_has_zero_degradation() -> None:
    assert (
        reward_degradation_fraction(
            -10.0,
            -5.0,
        )
        == 0.0
    )


def test_clean_summary() -> None:
    rows = [
        SeedCleanResult(
            domain="robotics",
            method="none",
            principal_seed=42,
            violation_step_rate=0.03,
            reward=1.0,
            success_rate=0.0,
            intervention_rate=0.0,
            mean_correction_l2=0.0,
        ),
        SeedCleanResult(
            domain="robotics",
            method="none",
            principal_seed=123,
            violation_step_rate=0.01,
            reward=2.0,
            success_rate=0.0,
            intervention_rate=0.0,
            mean_correction_l2=0.0,
        ),
        SeedCleanResult(
            domain="robotics",
            method="none",
            principal_seed=456,
            violation_step_rate=0.0,
            reward=3.0,
            success_rate=0.0,
            intervention_rate=0.0,
            mean_correction_l2=0.0,
        ),
    ]

    summary = summarize_results(rows)

    assert summary["reward"].mean == pytest.approx(2.0)

    assert summary["violation_step_rate"].mean == pytest.approx(0.013333333333333334)


def test_effectiveness_all_positive_and_zero_seed() -> None:
    gate = effectiveness_gate(
        none_violation_by_seed={
            42: 0.1,
            123: 0.2,
            456: 0.0,
        },
        comparison_violation_by_seed={
            42: 0.0,
            123: 0.0,
            456: 0.0,
        },
        none_reward_mean=1.0,
        comparison_reward_mean=1.0,
        none_success_mean=0.5,
        comparison_success_mean=0.5,
    )

    assert gate["overall_pass"] is True


def test_effectiveness_zero_baseline_cannot_worsen() -> None:
    gate = effectiveness_gate(
        none_violation_by_seed={
            42: 0.1,
            123: 0.2,
            456: 0.0,
        },
        comparison_violation_by_seed={
            42: 0.0,
            123: 0.0,
            456: 0.01,
        },
        none_reward_mean=1.0,
        comparison_reward_mean=1.0,
        none_success_mean=0.5,
        comparison_success_mean=0.5,
    )

    assert gate["seed_safety_requirements_pass"] is False

    assert gate["overall_pass"] is False
