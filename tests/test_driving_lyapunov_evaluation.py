from __future__ import annotations

import math

import pytest

from q_vla_forge.evaluation.driving_lyapunov_safety import (
    DRIVING_DOMAIN,
    MAXIMUM_REWARD_DEGRADATION_FRACTION,
    MAXIMUM_SUCCESS_RATE_DROP,
    MINIMUM_MEAN_VIOLATION_REDUCTION,
    DrivingLyapunovEpisodeEvidence,
    domain_mean_violation_reduction,
    empirical_effectiveness_supported,
    relative_violation_reduction,
    reward_degradation_fraction,
    seed_safety_requirement_passes,
    summarize_driving_lyapunov_episodes,
)
from q_vla_forge.safety.contracts import (
    RobustnessCondition,
    SafetyMethod,
)


def _episode(
    *,
    evaluation_seed: int = 20000,
    reward: float = 1.0,
    success: bool = True,
    corrections: tuple[float, ...] = (
        0.0,
        0.2,
        0.0,
        0.4,
    ),
) -> DrivingLyapunovEpisodeEvidence:
    current_v = (
        0.0,
        0.3,
        0.2,
        0.1,
    )

    proposed_next_v = (
        0.0,
        0.4,
        0.3,
        0.2,
    )

    selected_next_v = (
        0.0,
        0.2,
        0.2,
        0.05,
    )

    proposed_delta_v = tuple(
        proposed - current
        for current, proposed in zip(
            current_v,
            proposed_next_v,
            strict=True,
        )
    )

    selected_delta_v = tuple(
        selected - current
        for current, selected in zip(
            current_v,
            selected_next_v,
            strict=True,
        )
    )

    selection_delta = tuple(
        selected - proposed
        for proposed, selected in zip(
            proposed_next_v,
            selected_next_v,
            strict=True,
        )
    )

    intervention_count = sum(value > 1.0e-8 for value in corrections)

    return DrivingLyapunovEpisodeEvidence(
        domain=DRIVING_DOMAIN,
        principal_seed=42,
        evaluation_seed=evaluation_seed,
        method=SafetyMethod.LYAPUNOV,
        robustness_condition=RobustnessCondition.CLEAN,
        reward=reward,
        success=success,
        episode_length=4,
        proposed_violation_step_count=2,
        executed_violation_step_count=1,
        proposed_constraint_violation_count=3,
        executed_constraint_violation_count=1,
        critical_violation_step_count=1,
        intervention_count=intervention_count,
        action_corrections_l2=corrections,
        policy_action_out_of_bounds_count=0,
        executed_action_out_of_bounds_count=0,
        proposed_category_violation_counts={
            "steering_risk": 1,
            "acceleration_braking_conflict": 2,
        },
        executed_category_violation_counts={
            "steering_risk": 1,
        },
        intervention_reason_counts={
            "none": 2,
            "domain_constraint": 1,
            "lyapunov_decrease": 1,
        },
        selected_candidate_source_counts={
            "bounded_proposed": 2,
            "neutral_steering": 1,
            "inward_steer": 1,
        },
        candidate_counts=(
            5,
            6,
            5,
            8,
        ),
        eligible_candidate_counts=(
            5,
            4,
            5,
            6,
        ),
        proposed_hard_guard_failure_count=1,
        selected_hard_guard_failure_count=0,
        emergency_fallback_count=0,
        current_v_values=current_v,
        proposed_next_v_values=proposed_next_v,
        selected_next_v_values=selected_next_v,
        proposed_delta_v_values=proposed_delta_v,
        selected_delta_v_values=selected_delta_v,
        selection_v_delta_values=selection_delta,
        strict_lyapunov_decrease_count=2,
        lyapunov_nonincrease_count=4,
        selected_lower_than_proposed_count=3,
        zero_risk_state_step_count=1,
        zero_risk_preserved_step_count=1,
        zero_risk_recovery_count=1,
        zero_risk_recovery_steps=(2,),
        filter_latency_ms=(
            1.0,
            2.0,
            3.0,
            4.0,
        ),
    )


def test_summary_primary_safety_rates() -> None:
    summary = summarize_driving_lyapunov_episodes(
        [
            _episode(),
        ]
    )

    assert summary.proposed_violation_step_rate == 0.5

    assert summary.executed_violation_step_rate == 0.25

    assert summary.proposed_constraint_violation_rate == 0.75

    assert summary.executed_constraint_violation_rate == 0.25

    assert summary.critical_violation_step_rate == 0.25


def test_multiple_violations_on_one_step_do_not_change_step_rate() -> None:
    summary = summarize_driving_lyapunov_episodes(
        [
            _episode(),
        ]
    )

    assert summary.proposed_constraint_violation_count == 3

    assert summary.proposed_violation_step_count == 2


def test_intervention_statistics() -> None:
    summary = summarize_driving_lyapunov_episodes(
        [
            _episode(),
        ]
    )

    assert summary.intervention_count == 2

    assert summary.intervention_rate == 0.5

    assert summary.mean_action_correction_l2 == pytest.approx(0.15)

    assert summary.max_action_correction_l2 == 0.4

    assert summary.mean_intervention_correction_l2 == pytest.approx(0.3)


def test_reason_aggregation() -> None:
    summary = summarize_driving_lyapunov_episodes(
        [
            _episode(),
        ]
    )

    assert summary.intervention_reason_counts["none"] == 2

    assert summary.intervention_reason_counts["domain_constraint"] == 1

    assert summary.intervention_reason_counts["lyapunov_decrease"] == 1

    assert summary.intervention_reason_step_rates["domain_constraint"] == 0.25

    assert (
        summary.intervention_reason_intervention_fractions["domain_constraint"] == 0.5
    )


def test_candidate_source_aggregation() -> None:
    summary = summarize_driving_lyapunov_episodes(
        [
            _episode(),
        ]
    )

    assert summary.selected_candidate_source_counts["bounded_proposed"] == 2

    assert summary.selected_candidate_source_rates["inward_steer"] == 0.25


def test_candidate_statistics() -> None:
    summary = summarize_driving_lyapunov_episodes(
        [
            _episode(),
        ]
    )

    assert summary.mean_candidate_count == 6.0

    assert summary.max_candidate_count == 8

    assert summary.mean_eligible_candidate_count == 5.0


def test_fallback_rate() -> None:
    summary = summarize_driving_lyapunov_episodes(
        [
            _episode(),
        ]
    )

    assert summary.emergency_fallback_count == 0

    assert summary.emergency_fallback_rate == 0.0


def test_lyapunov_rates() -> None:
    summary = summarize_driving_lyapunov_episodes(
        [
            _episode(),
        ]
    )

    assert summary.strict_lyapunov_decrease_rate == 0.5

    assert summary.lyapunov_nonincrease_rate == 1.0

    assert summary.selected_lower_than_proposed_rate == 0.75


def test_selection_delta_is_negative_when_selected_is_better() -> None:
    summary = summarize_driving_lyapunov_episodes(
        [
            _episode(),
        ]
    )

    assert summary.mean_selection_v_delta < 0.0


def test_zero_risk_preservation() -> None:
    summary = summarize_driving_lyapunov_episodes(
        [
            _episode(),
        ]
    )

    assert summary.zero_risk_state_step_count == 1

    assert summary.zero_risk_preserved_step_count == 1

    assert summary.zero_risk_preservation_rate == 1.0


def test_recovery_metric() -> None:
    summary = summarize_driving_lyapunov_episodes(
        [
            _episode(),
        ]
    )

    assert summary.zero_risk_recovery_count == 1

    assert summary.mean_steps_to_zero_risk_recovery == 2.0


def test_latency_statistics() -> None:
    summary = summarize_driving_lyapunov_episodes(
        [
            _episode(),
        ]
    )

    assert summary.mean_filter_latency_ms == 2.5

    assert summary.max_filter_latency_ms == 4.0

    assert summary.filter_latency_sample_sd_ms > 0.0


def test_category_rates_include_frozen_zero_categories() -> None:
    summary = summarize_driving_lyapunov_episodes(
        [
            _episode(),
        ]
    )

    assert "lane_boundary" in summary.executed_category_violation_rates

    assert summary.executed_category_violation_rates["lane_boundary"] == 0.0

    assert summary.executed_category_violation_rates["steering_risk"] == 0.25


def test_duplicate_evaluation_seed_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="duplicate evaluation seed",
    ):
        summarize_driving_lyapunov_episodes(
            [
                _episode(),
                _episode(),
            ]
        )


def test_step_evidence_length_mismatch_rejected() -> None:
    episode = _episode(
        corrections=(
            0.0,
            0.2,
        )
    )

    with pytest.raises(
        ValueError,
        match="action corrections",
    ):
        summarize_driving_lyapunov_episodes(
            [
                episode,
            ]
        )


def test_candidate_ceiling_rejected() -> None:
    episode = _episode()

    invalid = DrivingLyapunovEpisodeEvidence(
        **{
            **episode.__dict__,
            "candidate_counts": (
                5,
                6,
                11,
                8,
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match="candidate count",
    ):
        summarize_driving_lyapunov_episodes(
            [
                invalid,
            ]
        )


def test_strict_decrease_arithmetic_is_reconstructed() -> None:
    episode = _episode()

    invalid = DrivingLyapunovEpisodeEvidence(
        **{
            **episode.__dict__,
            "strict_lyapunov_decrease_count": 3,
        }
    )

    with pytest.raises(
        ValueError,
        match="strict Lyapunov decrease",
    ):
        summarize_driving_lyapunov_episodes(
            [
                invalid,
            ]
        )


def test_relative_violation_reduction_positive_baseline() -> None:
    assert relative_violation_reduction(
        none_rate=0.5,
        method_rate=0.1,
    ) == pytest.approx(0.8)


def test_relative_violation_reduction_zero_baseline_is_none() -> None:
    assert (
        relative_violation_reduction(
            none_rate=0.0,
            method_rate=0.0,
        )
        is None
    )


def test_seed_safety_positive_baseline_requires_strict_improvement() -> None:
    assert seed_safety_requirement_passes(
        none_rate=0.5,
        method_rate=0.4,
    )

    assert not seed_safety_requirement_passes(
        none_rate=0.5,
        method_rate=0.5,
    )


def test_seed_safety_zero_baseline_requires_no_worsening() -> None:
    assert seed_safety_requirement_passes(
        none_rate=0.0,
        method_rate=0.0,
    )

    assert not seed_safety_requirement_passes(
        none_rate=0.0,
        method_rate=0.01,
    )


def test_reward_degradation_matches_frozen_definition() -> None:
    assert reward_degradation_fraction(
        none_reward=-10.0,
        method_reward=-11.0,
    ) == pytest.approx(0.1)

    assert (
        reward_degradation_fraction(
            none_reward=-10.0,
            method_reward=-8.0,
        )
        == 0.0
    )


def test_domain_mean_reduction() -> None:
    assert domain_mean_violation_reduction(
        none_mean_rate=0.4,
        method_mean_rate=0.1,
    ) == pytest.approx(0.75)


def test_effectiveness_supported_when_all_frozen_conditions_pass() -> None:
    assert empirical_effectiveness_supported(
        seed_safety_passes=(
            True,
            True,
            True,
        ),
        domain_violation_reduction=(MINIMUM_MEAN_VIOLATION_REDUCTION),
        reward_degradation_fraction_value=(MAXIMUM_REWARD_DEGRADATION_FRACTION),
        success_rate_drop=(MAXIMUM_SUCCESS_RATE_DROP),
    )


def test_effectiveness_fails_when_one_seed_safety_requirement_fails() -> None:
    assert not empirical_effectiveness_supported(
        seed_safety_passes=(
            True,
            False,
            True,
        ),
        domain_violation_reduction=1.0,
        reward_degradation_fraction_value=0.0,
        success_rate_drop=0.0,
    )


def test_effectiveness_fails_on_reward_degradation() -> None:
    assert not empirical_effectiveness_supported(
        seed_safety_passes=(
            True,
            True,
            True,
        ),
        domain_violation_reduction=1.0,
        reward_degradation_fraction_value=0.11,
        success_rate_drop=0.0,
    )


def test_nonfinite_reward_rejected() -> None:
    episode = _episode(reward=math.nan)

    with pytest.raises(
        ValueError,
        match="reward",
    ):
        summarize_driving_lyapunov_episodes(
            [
                episode,
            ]
        )
