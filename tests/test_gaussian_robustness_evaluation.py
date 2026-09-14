"""Tests for Sprint 5.10 Gaussian robustness aggregation."""

from __future__ import annotations

import math

import pytest

from q_vla_forge.evaluation.gaussian_robustness import (
    GaussianEpisodeEvidence,
    aggregate_across_principal_seeds,
    compute_clean_delta,
    summarize_gaussian_seed,
)


def _episode(
    *,
    evaluation_seed: int,
    reward: float = 1.0,
    success: bool = False,
    violations: int = 1,
    critical: int = 0,
    interventions: int = 0,
    sigma: float = 0.10,
) -> GaussianEpisodeEvidence:
    return GaussianEpisodeEvidence(
        domain="robotics",
        method="lyapunov",
        principal_seed=42,
        evaluation_seed=evaluation_seed,
        sigma=sigma,
        noise_seed=1234 + evaluation_seed,
        reward=reward,
        success=success,
        episode_length=100,
        proposed_violation_step_count=violations,
        executed_violation_step_count=violations,
        proposed_constraint_violation_count=violations,
        executed_constraint_violation_count=violations,
        critical_violation_step_count=critical,
        intervention_count=interventions,
        noise_sample_count=100,
        noise_l2_sum=10.0,
        noise_l2_max=0.5,
        action_correction_l2_sum=2.0,
        action_correction_l2_max=0.3,
        intervention_reason_counts={"DOMAIN_CONSTRAINT": interventions},
        category_violation_counts={"unsafe_motion": violations},
        strict_lyapunov_decrease_count=0,
        lyapunov_nonincrease_count=100,
        selected_lower_than_proposed_count=2,
        emergency_fallback_count=0,
        steps_object_grasped=0,
        steps_object_not_grasped=100,
        interventions_while_grasped=0,
    )


def test_seed_summary_arithmetic() -> None:
    summary = summarize_gaussian_seed(
        [
            _episode(
                evaluation_seed=20000,
                reward=1.0,
                success=False,
                violations=2,
                critical=1,
                interventions=3,
            ),
            _episode(
                evaluation_seed=20001,
                reward=3.0,
                success=True,
                violations=4,
                critical=1,
                interventions=1,
            ),
        ]
    )

    assert summary.episode_count == 2
    assert summary.total_environment_steps == 200

    assert summary.mean_reward == 2.0

    assert math.isclose(
        summary.reward_sample_sd,
        math.sqrt(2.0),
    )

    assert summary.success_rate == 0.5

    assert summary.executed_violation_step_rate == 6 / 200

    assert summary.critical_violation_step_rate == 2 / 200

    assert summary.intervention_rate == 4 / 200

    assert summary.mean_action_correction_l2 == 4.0 / 200

    assert summary.mean_noise_l2 == 20.0 / 200

    assert summary.category_violation_rates["unsafe_motion"] == 6 / 200

    assert summary.intervention_reason_rates["DOMAIN_CONSTRAINT"] == 4 / 200

    assert summary.lyapunov_nonincrease_rate == 1.0


def test_clean_delta() -> None:
    clean = summarize_gaussian_seed(
        [
            _episode(
                evaluation_seed=20000,
                reward=1.0,
                violations=2,
                sigma=0.0,
            )
        ]
    )

    noisy = summarize_gaussian_seed(
        [
            _episode(
                evaluation_seed=20000,
                reward=0.5,
                violations=5,
                sigma=0.10,
            )
        ]
    )

    delta = compute_clean_delta(
        noisy=noisy,
        clean=clean,
    )

    assert math.isclose(
        delta.violation_step_delta,
        0.03,
    )

    assert delta.reward_delta == -0.5

    assert math.isclose(
        delta.relative_violation_degradation or 0.0,
        1.5,
    )


def test_zero_clean_relative_delta_is_none() -> None:
    clean = summarize_gaussian_seed(
        [
            _episode(
                evaluation_seed=20000,
                violations=0,
                sigma=0.0,
            )
        ]
    )

    noisy = summarize_gaussian_seed(
        [
            _episode(
                evaluation_seed=20000,
                violations=1,
                sigma=0.10,
            )
        ]
    )

    delta = compute_clean_delta(
        noisy=noisy,
        clean=clean,
    )

    assert delta.relative_violation_degradation is None


def test_seed_mismatch_rejected() -> None:
    clean_episode = _episode(
        evaluation_seed=20000,
        sigma=0.0,
    )

    noisy_episode = GaussianEpisodeEvidence(
        **{
            **clean_episode.__dict__,
            "principal_seed": 123,
            "sigma": 0.10,
        }
    )

    clean = summarize_gaussian_seed([clean_episode])

    noisy = summarize_gaussian_seed([noisy_episode])

    with pytest.raises(ValueError):
        compute_clean_delta(
            noisy=noisy,
            clean=clean,
        )


def test_principal_seed_aggregation() -> None:
    result = aggregate_across_principal_seeds(
        {
            42: 0.1,
            123: 0.2,
            456: 0.3,
        }
    )

    assert math.isclose(
        result.mean,
        0.2,
    )

    assert math.isclose(
        result.sample_sd,
        0.1,
    )


def test_constraint_rate_can_exceed_one() -> None:
    episode = GaussianEpisodeEvidence(
        domain="autonomous_driving",
        method="none",
        principal_seed=42,
        evaluation_seed=20000,
        sigma=0.10,
        noise_seed=1,
        reward=0.0,
        success=False,
        episode_length=10,
        proposed_violation_step_count=10,
        executed_violation_step_count=10,
        proposed_constraint_violation_count=15,
        executed_constraint_violation_count=15,
        critical_violation_step_count=0,
        intervention_count=0,
        noise_sample_count=10,
        noise_l2_sum=1.0,
        noise_l2_max=0.2,
    )

    summary = summarize_gaussian_seed([episode])

    assert summary.executed_constraint_violation_rate == 1.5


def test_none_can_have_zero_intervention_rate() -> None:
    episode = GaussianEpisodeEvidence(
        domain="autonomous_driving",
        method="none",
        principal_seed=42,
        evaluation_seed=20000,
        sigma=0.10,
        noise_seed=1,
        reward=0.0,
        success=False,
        episode_length=100,
        proposed_violation_step_count=0,
        executed_violation_step_count=0,
        proposed_constraint_violation_count=0,
        executed_constraint_violation_count=0,
        critical_violation_step_count=0,
        intervention_count=0,
        noise_sample_count=100,
        noise_l2_sum=10.0,
        noise_l2_max=0.4,
    )

    summary = summarize_gaussian_seed([episode])

    assert summary.intervention_rate == 0.0
