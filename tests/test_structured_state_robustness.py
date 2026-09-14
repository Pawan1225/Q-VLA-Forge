"""Tests for Sprint 5.11 structured-state robustness contracts."""

from __future__ import annotations

import math

import pytest

from q_vla_forge.evaluation.structured_state_robustness import (
    StructuredStateEpisodeEvidence,
    aggregate_structured_metric,
    compute_structured_clean_delta,
    summarize_structured_state_seed,
)


def _episode(
    *,
    reward: float,
    success: bool,
    violation_steps: int,
    intervention_count: int,
) -> StructuredStateEpisodeEvidence:
    return StructuredStateEpisodeEvidence(
        domain="autonomous_driving",
        method="lyapunov",
        perturbation_name="lane_offset_plus_0p05",
        perturbation_family="lane_offset",
        principal_seed=42,
        evaluation_seed=20_000,
        reward=reward,
        success=success,
        episode_length=100,
        proposed_violation_step_count=(violation_steps + 1),
        executed_violation_step_count=(violation_steps),
        proposed_constraint_violation_count=(violation_steps + 2),
        executed_constraint_violation_count=(violation_steps),
        critical_violation_step_count=1,
        intervention_count=(intervention_count),
        perturbation_l1_sum=5.0,
        perturbation_l2_sum=5.0,
        perturbation_linf_max=0.05,
        action_correction_l2_sum=2.0,
        action_correction_l2_max=0.4,
        category_violation_counts={"lane_boundary": (violation_steps)},
        intervention_reason_counts={"domain_constraint": (intervention_count)},
        selected_candidate_source_counts={
            "bounded_proposed": 90,
            "neutral_steering": 10,
        },
        intervened_candidate_source_counts={"neutral_steering": (intervention_count)},
        strict_lyapunov_decrease_count=2,
        lyapunov_nonincrease_count=100,
        selected_lower_than_proposed_count=3,
        emergency_fallback_count=0,
    )


def test_summary_arithmetic() -> None:
    episodes = [
        _episode(
            reward=1.0,
            success=False,
            violation_steps=4,
            intervention_count=5,
        ),
        _episode(
            reward=3.0,
            success=True,
            violation_steps=6,
            intervention_count=7,
        ),
    ]

    summary = summarize_structured_state_seed(episodes)

    assert summary.mean_reward == pytest.approx(2.0)

    assert summary.reward_sample_sd == pytest.approx(math.sqrt(2.0))

    assert summary.success_rate == pytest.approx(0.5)

    assert summary.executed_violation_step_rate == pytest.approx(10 / 200)

    assert summary.critical_violation_step_rate == pytest.approx(2 / 200)

    assert summary.intervention_rate == pytest.approx(12 / 200)

    assert summary.mean_action_correction_l2 == pytest.approx(4 / 200)

    assert summary.mean_perturbation_l1 == pytest.approx(10 / 200)

    assert summary.mean_perturbation_l2 == pytest.approx(10 / 200)

    assert summary.max_perturbation_linf == pytest.approx(0.05)

    assert summary.strict_lyapunov_decrease_rate == pytest.approx(4 / 200)

    assert summary.lyapunov_nonincrease_rate == pytest.approx(200 / 200)


def test_clean_delta() -> None:
    clean = summarize_structured_state_seed(
        [
            _episode(
                reward=1.0,
                success=False,
                violation_steps=2,
                intervention_count=0,
            )
        ]
    )

    structured = summarize_structured_state_seed(
        [
            _episode(
                reward=0.5,
                success=False,
                violation_steps=4,
                intervention_count=2,
            )
        ]
    )

    delta = compute_structured_clean_delta(
        structured=structured,
        clean=clean,
    )

    assert delta.violation_delta == pytest.approx(0.02)

    assert delta.reward_delta == pytest.approx(-0.5)

    assert delta.relative_violation_degradation == pytest.approx(1.0)


def test_zero_clean_violation_has_no_relative_delta() -> None:
    clean_episode = _episode(
        reward=1.0,
        success=False,
        violation_steps=0,
        intervention_count=0,
    )

    structured_episode = _episode(
        reward=1.0,
        success=False,
        violation_steps=1,
        intervention_count=0,
    )

    clean = summarize_structured_state_seed([clean_episode])

    structured = summarize_structured_state_seed([structured_episode])

    delta = compute_structured_clean_delta(
        structured=structured,
        clean=clean,
    )

    assert delta.relative_violation_degradation is None


def test_mismatched_seed_rejected() -> None:
    clean = summarize_structured_state_seed(
        [
            _episode(
                reward=1.0,
                success=False,
                violation_steps=1,
                intervention_count=0,
            )
        ]
    )

    other = StructuredStateEpisodeEvidence(
        **{
            **_episode(
                reward=1.0,
                success=False,
                violation_steps=1,
                intervention_count=0,
            ).__dict__,
            "principal_seed": 123,
        }
    )

    structured = summarize_structured_state_seed([other])

    with pytest.raises(
        ValueError,
        match="principal seed mismatch",
    ):
        compute_structured_clean_delta(
            structured=structured,
            clean=clean,
        )


def test_aggregate_metric_sample_sd() -> None:
    metric = aggregate_structured_metric(
        [
            0.1,
            0.2,
            0.3,
        ]
    )

    assert metric.mean == pytest.approx(0.2)

    assert metric.sample_sd == pytest.approx(0.1)


def test_constraint_rate_may_exceed_one() -> None:
    episode = StructuredStateEpisodeEvidence(
        domain="robotics",
        method="none",
        perturbation_name="robot_x_minus_0p05",
        perturbation_family="robot_position",
        principal_seed=42,
        evaluation_seed=20_000,
        reward=0.0,
        success=False,
        episode_length=10,
        proposed_violation_step_count=10,
        executed_violation_step_count=10,
        proposed_constraint_violation_count=25,
        executed_constraint_violation_count=25,
        critical_violation_step_count=0,
        intervention_count=0,
        perturbation_l1_sum=0.5,
        perturbation_l2_sum=0.5,
        perturbation_linf_max=0.05,
        steps_object_grasped=0,
        steps_object_not_grasped=10,
    )

    summary = summarize_structured_state_seed([episode])

    assert summary.executed_constraint_violation_rate == pytest.approx(2.5)


def test_empty_episode_list_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        summarize_structured_state_seed([])
