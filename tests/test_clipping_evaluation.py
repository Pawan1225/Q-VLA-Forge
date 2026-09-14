from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from q_vla_forge.evaluation.clipping_safety import (
    ClippingEpisodeEvidence,
    clipping_episode_to_dict,
    clipping_summary_to_dict,
    run_clipping_episode,
    summarize_clipping_episodes,
)
from q_vla_forge.evaluation.safety_baseline import (
    load_frozen_ppo_policy,
)
from q_vla_forge.safety.contracts import (
    RobustnessCondition,
    SafetyMethod,
)

ROOT = Path(__file__).resolve().parents[1]


def _episode(
    *,
    evaluation_seed: int,
    episode_length: int = 4,
    reward: float = 1.0,
    success: bool = False,
    proposed_violation_steps: int = 2,
    executed_violation_steps: int = 1,
    proposed_constraints: int = 3,
    executed_constraints: int = 1,
    critical_steps: int = 0,
    corrections: tuple[
        float,
        ...,
    ] = (
        0.0,
        0.2,
        0.0,
        0.4,
    ),
    proposed_categories: (
        dict[
            str,
            int,
        ]
        | None
    ) = None,
    executed_categories: (
        dict[
            str,
            int,
        ]
        | None
    ) = None,
    rule_counts: (
        dict[
            str,
            int,
        ]
        | None
    ) = None,
) -> ClippingEpisodeEvidence:
    intervention_count = sum(value > 1e-8 for value in corrections)

    return ClippingEpisodeEvidence(
        domain="autonomous_driving",
        principal_seed=42,
        evaluation_seed=evaluation_seed,
        method=SafetyMethod.CLIPPING,
        robustness_condition=(RobustnessCondition.CLEAN),
        reward=reward,
        success=success,
        episode_length=episode_length,
        proposed_violation_step_count=(proposed_violation_steps),
        executed_violation_step_count=(executed_violation_steps),
        proposed_constraint_violation_count=(proposed_constraints),
        executed_constraint_violation_count=(executed_constraints),
        critical_violation_step_count=(critical_steps),
        intervention_count=(intervention_count),
        total_action_correction_l2=float(sum(corrections)),
        max_action_correction_l2=float(
            max(
                corrections,
                default=0.0,
            )
        ),
        action_corrections_l2=(corrections),
        policy_action_out_of_bounds_count=0,
        executed_action_out_of_bounds_count=0,
        proposed_category_violation_counts=(
            proposed_categories
            or {
                "steering_risk": 3,
            }
        ),
        executed_category_violation_counts=(
            executed_categories
            or {
                "steering_risk": 1,
            }
        ),
        triggered_rule_counts=(
            rule_counts
            or {
                "contextual_steering_cap": 2,
            }
        ),
    )


def test_summary_uses_executed_steps_as_primary_safety_evidence() -> None:
    summary = summarize_clipping_episodes(
        [
            _episode(
                evaluation_seed=20_000,
            )
        ]
    )

    assert summary.proposed_violation_step_rate == pytest.approx(2 / 4)

    assert summary.executed_violation_step_rate == pytest.approx(1 / 4)


def test_constraint_rates_use_total_steps() -> None:
    summary = summarize_clipping_episodes(
        [
            _episode(
                evaluation_seed=20_000,
            )
        ]
    )

    assert summary.proposed_constraint_violation_rate == pytest.approx(3 / 4)

    assert summary.executed_constraint_violation_rate == pytest.approx(1 / 4)


def test_exact_step_correction_mean() -> None:
    summary = summarize_clipping_episodes(
        [
            _episode(
                evaluation_seed=20_000,
                corrections=(
                    0.0,
                    0.2,
                    0.0,
                    0.4,
                ),
            )
        ]
    )

    assert summary.mean_action_correction_l2 == pytest.approx(0.15)


def test_exact_step_correction_p95() -> None:
    corrections = (
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    )

    episode = _episode(
        evaluation_seed=20_000,
        episode_length=10,
        corrections=corrections,
    )

    summary = summarize_clipping_episodes([episode])

    expected = float(
        np.percentile(
            np.asarray(
                corrections,
                dtype=np.float64,
            ),
            95,
        )
    )

    assert summary.p95_action_correction_l2 == pytest.approx(expected)


def test_exact_max_correction() -> None:
    summary = summarize_clipping_episodes(
        [
            _episode(
                evaluation_seed=20_000,
                corrections=(
                    0.0,
                    0.2,
                    0.9,
                    0.4,
                ),
            )
        ]
    )

    assert summary.max_action_correction_l2 == pytest.approx(0.9)


def test_mean_intervention_correction_excludes_zero_steps() -> None:
    summary = summarize_clipping_episodes(
        [
            _episode(
                evaluation_seed=20_000,
                corrections=(
                    0.0,
                    0.2,
                    0.0,
                    0.4,
                ),
            )
        ]
    )

    assert summary.mean_intervention_correction_l2 == pytest.approx(0.3)


def test_intervention_rate_from_exact_step_evidence() -> None:
    summary = summarize_clipping_episodes(
        [
            _episode(
                evaluation_seed=20_000,
                corrections=(
                    0.0,
                    0.2,
                    0.0,
                    0.4,
                ),
            )
        ]
    )

    assert summary.intervention_count == 2

    assert summary.intervention_rate == pytest.approx(0.5)


def test_category_counts_aggregate() -> None:
    summary = summarize_clipping_episodes(
        [
            _episode(
                evaluation_seed=20_000,
                proposed_categories={
                    "steering_risk": 2,
                },
                executed_categories={
                    "steering_risk": 1,
                },
            ),
            _episode(
                evaluation_seed=20_001,
                proposed_categories={
                    "steering_risk": 3,
                },
                executed_categories={
                    "steering_risk": 1,
                },
            ),
        ]
    )

    assert summary.proposed_category_violation_counts["steering_risk"] == 5

    assert summary.executed_category_violation_counts["steering_risk"] == 2


def test_rule_counts_aggregate() -> None:
    summary = summarize_clipping_episodes(
        [
            _episode(
                evaluation_seed=20_000,
                rule_counts={
                    "contextual_steering_cap": 2,
                },
            ),
            _episode(
                evaluation_seed=20_001,
                rule_counts={
                    "contextual_steering_cap": 1,
                    "speed_distance_braking": 2,
                },
            ),
        ]
    )

    assert summary.triggered_rule_counts["contextual_steering_cap"] == 3

    assert summary.triggered_rule_counts["speed_distance_braking"] == 2


def test_duplicate_eval_seed_rejected() -> None:
    with pytest.raises(ValueError):
        summarize_clipping_episodes(
            [
                _episode(
                    evaluation_seed=20_000,
                ),
                _episode(
                    evaluation_seed=20_000,
                ),
            ]
        )


def test_correction_length_mismatch_rejected() -> None:
    episode = ClippingEpisodeEvidence(
        domain="autonomous_driving",
        principal_seed=42,
        evaluation_seed=20_000,
        method=SafetyMethod.CLIPPING,
        robustness_condition=(RobustnessCondition.CLEAN),
        reward=0.0,
        success=False,
        episode_length=4,
        proposed_violation_step_count=0,
        executed_violation_step_count=0,
        proposed_constraint_violation_count=0,
        executed_constraint_violation_count=0,
        critical_violation_step_count=0,
        intervention_count=1,
        total_action_correction_l2=0.2,
        max_action_correction_l2=0.2,
        action_corrections_l2=(
            0.0,
            0.2,
        ),
        policy_action_out_of_bounds_count=0,
        executed_action_out_of_bounds_count=0,
        proposed_category_violation_counts={},
        executed_category_violation_counts={},
        triggered_rule_counts={},
    )

    with pytest.raises(ValueError):
        summarize_clipping_episodes([episode])


def test_intervention_count_mismatch_rejected() -> None:
    episode = ClippingEpisodeEvidence(
        domain="autonomous_driving",
        principal_seed=42,
        evaluation_seed=20_000,
        method=SafetyMethod.CLIPPING,
        robustness_condition=(RobustnessCondition.CLEAN),
        reward=0.0,
        success=False,
        episode_length=2,
        proposed_violation_step_count=0,
        executed_violation_step_count=0,
        proposed_constraint_violation_count=0,
        executed_constraint_violation_count=0,
        critical_violation_step_count=0,
        intervention_count=0,
        total_action_correction_l2=0.2,
        max_action_correction_l2=0.2,
        action_corrections_l2=(
            0.0,
            0.2,
        ),
        policy_action_out_of_bounds_count=0,
        executed_action_out_of_bounds_count=0,
        proposed_category_violation_counts={},
        executed_category_violation_counts={},
        triggered_rule_counts={},
    )

    with pytest.raises(ValueError):
        summarize_clipping_episodes([episode])


def test_total_correction_mismatch_rejected() -> None:
    episode = ClippingEpisodeEvidence(
        domain="autonomous_driving",
        principal_seed=42,
        evaluation_seed=20_000,
        method=SafetyMethod.CLIPPING,
        robustness_condition=(RobustnessCondition.CLEAN),
        reward=0.0,
        success=False,
        episode_length=2,
        proposed_violation_step_count=0,
        executed_violation_step_count=0,
        proposed_constraint_violation_count=0,
        executed_constraint_violation_count=0,
        critical_violation_step_count=0,
        intervention_count=1,
        total_action_correction_l2=0.5,
        max_action_correction_l2=0.2,
        action_corrections_l2=(
            0.0,
            0.2,
        ),
        policy_action_out_of_bounds_count=0,
        executed_action_out_of_bounds_count=0,
        proposed_category_violation_counts={},
        executed_category_violation_counts={},
        triggered_rule_counts={},
    )

    with pytest.raises(ValueError):
        summarize_clipping_episodes([episode])


def test_episode_serialization() -> None:
    episode = _episode(
        evaluation_seed=20_000,
    )

    payload = clipping_episode_to_dict(episode)

    assert payload["method"] == "clipping"

    assert payload["robustness_condition"] == "clean"

    assert isinstance(
        payload["action_corrections_l2"],
        list,
    )


def test_summary_serialization() -> None:
    summary = summarize_clipping_episodes(
        [
            _episode(
                evaluation_seed=20_000,
            )
        ]
    )

    payload = clipping_summary_to_dict(summary)

    assert payload["method"] == "clipping"

    assert payload["robustness_condition"] == "clean"


def test_driving_clipping_smoke_episode() -> None:
    policy = load_frozen_ppo_policy(
        root=ROOT,
        domain="autonomous_driving",
        principal_seed=42,
    )

    episode = run_clipping_episode(
        policy=policy,
        evaluation_seed=20_000,
    )

    assert episode.domain == "autonomous_driving"

    assert episode.principal_seed == 42

    assert episode.evaluation_seed == 20_000

    assert episode.method == SafetyMethod.CLIPPING

    assert episode.robustness_condition == RobustnessCondition.CLEAN

    assert episode.episode_length > 0

    assert len(episode.action_corrections_l2) == episode.episode_length

    assert episode.policy_action_out_of_bounds_count == 0

    assert episode.executed_action_out_of_bounds_count == 0


def test_robotics_clipping_smoke_episode() -> None:
    policy = load_frozen_ppo_policy(
        root=ROOT,
        domain="robotics",
        principal_seed=42,
    )

    episode = run_clipping_episode(
        policy=policy,
        evaluation_seed=20_000,
    )

    assert episode.domain == "robotics"

    assert episode.principal_seed == 42

    assert episode.method == SafetyMethod.CLIPPING

    assert episode.episode_length > 0

    assert len(episode.action_corrections_l2) == episode.episode_length

    assert episode.policy_action_out_of_bounds_count == 0

    assert episode.executed_action_out_of_bounds_count == 0


def test_non_frozen_evaluation_seed_rejected() -> None:
    policy = load_frozen_ppo_policy(
        root=ROOT,
        domain="autonomous_driving",
        principal_seed=42,
    )

    with pytest.raises(ValueError):
        run_clipping_episode(
            policy=policy,
            evaluation_seed=999,
        )
