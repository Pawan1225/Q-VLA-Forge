from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from q_vla_forge.evaluation.safety_baseline import (
    SAFETY_EVALUATION_SEEDS,
    SafetyEpisodeEvidence,
    action_is_within_bounds,
    deterministic_policy_action,
    load_frozen_ppo_policy,
    summarize_episode_evidence,
)
from q_vla_forge.safety.contracts import (
    RobustnessCondition,
    SafetyMethod,
)

ROOT = Path(__file__).resolve().parents[1]


def _episode(
    *,
    evaluation_seed: int,
    reward: float = 1.0,
    success: bool = True,
    episode_length: int = 10,
    violation_steps: int = 0,
    constraint_violations: int = 0,
    critical_steps: int = 0,
    interventions: int = 0,
    total_correction: float = 0.0,
    max_correction: float = 0.0,
    out_of_bounds: int = 0,
    category_counts: (
        dict[
            str,
            int,
        ]
        | None
    ) = None,
) -> SafetyEpisodeEvidence:
    return SafetyEpisodeEvidence(
        domain="autonomous_driving",
        principal_seed=42,
        evaluation_seed=(evaluation_seed),
        method=SafetyMethod.NONE,
        robustness_condition=(RobustnessCondition.CLEAN),
        reward=reward,
        success=success,
        episode_length=episode_length,
        violation_step_count=(violation_steps),
        constraint_violation_count=(constraint_violations),
        critical_violation_step_count=(critical_steps),
        intervention_count=(interventions),
        total_action_correction_l2=(total_correction),
        max_action_correction_l2=(max_correction),
        policy_action_out_of_bounds_count=(out_of_bounds),
        category_violation_counts=(category_counts or {}),
    )


def test_frozen_evaluation_seeds() -> None:
    assert SAFETY_EVALUATION_SEEDS == tuple(
        range(
            20_000,
            20_020,
        )
    )


def test_action_inside_bounds() -> None:
    assert action_is_within_bounds(
        action=np.array([0.0, 0.5, 1.0]),
        action_low=np.array([-1.0, -1.0, 0.0]),
        action_high=np.array([1.0, 1.0, 1.0]),
    )


def test_action_outside_bounds() -> None:
    assert not action_is_within_bounds(
        action=np.array([1.1, 0.0]),
        action_low=np.array([-1.0, -1.0]),
        action_high=np.array([1.0, 1.0]),
    )


def test_action_bound_shape_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        action_is_within_bounds(
            action=np.zeros(2),
            action_low=np.zeros(3),
            action_high=np.ones(3),
        )


def test_zero_violation_summary() -> None:
    episodes = [
        _episode(
            evaluation_seed=20_000,
            reward=1.0,
        ),
        _episode(
            evaluation_seed=20_001,
            reward=3.0,
        ),
    ]

    summary = summarize_episode_evidence(episodes)

    assert summary.total_environment_steps == 20

    assert summary.violation_step_rate == 0.0

    assert summary.constraint_violation_rate == 0.0

    assert summary.critical_violation_step_rate == 0.0

    assert summary.intervention_rate == 0.0


def test_violation_denominator_is_total_steps() -> None:
    episodes = [
        _episode(
            evaluation_seed=20_000,
            episode_length=10,
            violation_steps=2,
            constraint_violations=3,
        ),
        _episode(
            evaluation_seed=20_001,
            episode_length=20,
            violation_steps=4,
            constraint_violations=6,
        ),
    ]

    summary = summarize_episode_evidence(episodes)

    assert summary.violation_step_count == 6

    assert summary.violation_step_rate == pytest.approx(6 / 30)

    assert summary.constraint_violation_rate == pytest.approx(9 / 30)


def test_multiple_constraints_can_exceed_violation_rate() -> None:
    episodes = [
        _episode(
            evaluation_seed=20_000,
            episode_length=10,
            violation_steps=5,
            constraint_violations=15,
        )
    ]

    summary = summarize_episode_evidence(episodes)

    assert summary.violation_step_rate == 0.5

    assert summary.constraint_violation_rate == 1.5


def test_critical_step_counting() -> None:
    episodes = [
        _episode(
            evaluation_seed=20_000,
            episode_length=10,
            violation_steps=4,
            constraint_violations=6,
            critical_steps=2,
        )
    ]

    summary = summarize_episode_evidence(episodes)

    assert summary.critical_violation_step_rate == 0.2


def test_category_counts_and_rates() -> None:
    episodes = [
        _episode(
            evaluation_seed=20_000,
            episode_length=10,
            category_counts={
                "lane_boundary": 2,
            },
        ),
        _episode(
            evaluation_seed=20_001,
            episode_length=10,
            category_counts={
                "lane_boundary": 1,
                "steering_risk": 2,
            },
        ),
    ]

    summary = summarize_episode_evidence(episodes)

    assert summary.category_violation_counts["lane_boundary"] == 3

    assert summary.category_violation_rates["lane_boundary"] == pytest.approx(3 / 20)

    assert summary.category_violation_rates["steering_risk"] == pytest.approx(2 / 20)


def test_sample_reward_sd() -> None:
    episodes = [
        _episode(
            evaluation_seed=20_000,
            reward=1.0,
        ),
        _episode(
            evaluation_seed=20_001,
            reward=3.0,
        ),
    ]

    summary = summarize_episode_evidence(episodes)

    assert summary.mean_reward == 2.0

    assert summary.reward_sample_sd == pytest.approx(2**0.5)


def test_success_rate() -> None:
    episodes = [
        _episode(
            evaluation_seed=20_000,
            success=True,
        ),
        _episode(
            evaluation_seed=20_001,
            success=False,
        ),
    ]

    summary = summarize_episode_evidence(episodes)

    assert summary.success_rate == 0.5


def test_duplicate_evaluation_seed_rejected() -> None:
    episodes = [
        _episode(
            evaluation_seed=20_000,
        ),
        _episode(
            evaluation_seed=20_000,
        ),
    ]

    with pytest.raises(ValueError):
        summarize_episode_evidence(episodes)


def test_zero_length_episode_rejected() -> None:
    with pytest.raises(ValueError):
        summarize_episode_evidence(
            [
                _episode(
                    evaluation_seed=20_000,
                    episode_length=0,
                )
            ]
        )


def test_none_baseline_rejects_intervention() -> None:
    with pytest.raises(ValueError):
        summarize_episode_evidence(
            [
                _episode(
                    evaluation_seed=20_000,
                    interventions=1,
                )
            ]
        )


def test_none_baseline_rejects_correction() -> None:
    with pytest.raises(ValueError):
        summarize_episode_evidence(
            [
                _episode(
                    evaluation_seed=20_000,
                    total_correction=0.1,
                    max_correction=0.1,
                )
            ]
        )


def test_load_driving_seed42_checkpoint() -> None:
    policy = load_frozen_ppo_policy(
        root=ROOT,
        domain="autonomous_driving",
        principal_seed=42,
    )

    assert policy.domain == "autonomous_driving"

    assert policy.principal_seed == 42

    assert policy.training_budget == 20_000


def test_driving_checkpoint_action_bounds() -> None:
    policy = load_frozen_ppo_policy(
        root=ROOT,
        domain="autonomous_driving",
        principal_seed=42,
    )

    np.testing.assert_array_equal(
        policy.action_low,
        np.array(
            [-1.0, -1.0, 0.0],
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        policy.action_high,
        np.array(
            [1.0, 1.0, 1.0],
            dtype=np.float32,
        ),
    )


def test_deterministic_policy_action_is_bounded() -> None:
    policy = load_frozen_ppo_policy(
        root=ROOT,
        domain="autonomous_driving",
        principal_seed=42,
    )

    action = deterministic_policy_action(
        policy=policy,
        observation=np.array(
            [
                0.25,
                0.0,
                0.0,
                2.0,
            ],
            dtype=np.float32,
        ),
    )

    assert action_is_within_bounds(
        action=action,
        action_low=policy.action_low,
        action_high=policy.action_high,
    )


def test_deterministic_policy_action_reproducible() -> None:
    policy = load_frozen_ppo_policy(
        root=ROOT,
        domain="autonomous_driving",
        principal_seed=42,
    )

    observation = np.array(
        [
            0.25,
            0.0,
            0.0,
            2.0,
        ],
        dtype=np.float32,
    )

    first = deterministic_policy_action(
        policy=policy,
        observation=observation,
    )

    second = deterministic_policy_action(
        policy=policy,
        observation=observation,
    )

    np.testing.assert_array_equal(
        first,
        second,
    )


def test_driving_violation_router_returns_five_records() -> None:
    from q_vla_forge.evaluation.safety_baseline import (
        evaluate_domain_violations,
    )

    records = evaluate_domain_violations(
        domain="autonomous_driving",
        state=np.array(
            [
                0.25,
                0.0,
                0.0,
                2.0,
            ],
            dtype=np.float64,
        ),
        action=np.array(
            [
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        ),
    )

    assert len(records) == 5


def test_robotics_violation_router_returns_five_records() -> None:
    from q_vla_forge.evaluation.safety_baseline import (
        evaluate_domain_violations,
    )

    records = evaluate_domain_violations(
        domain="robotics",
        state=np.array(
            [
                0.0,
                0.0,
                0.3,
                0.0,
                0.8,
                0.8,
            ],
            dtype=np.float64,
        ),
        action=np.array(
            [
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        ),
    )

    assert len(records) == 5


def test_unknown_violation_domain_rejected() -> None:
    from q_vla_forge.evaluation.safety_baseline import (
        evaluate_domain_violations,
    )

    with pytest.raises(ValueError):
        evaluate_domain_violations(
            domain="unknown",
            state=np.zeros(1),
            action=np.zeros(1),
        )


def test_driving_none_smoke_episode() -> None:
    from q_vla_forge.evaluation.safety_baseline import (
        run_no_filter_episode,
    )

    policy = load_frozen_ppo_policy(
        root=ROOT,
        domain="autonomous_driving",
        principal_seed=42,
    )

    episode = run_no_filter_episode(
        policy=policy,
        evaluation_seed=20_000,
    )

    assert episode.domain == "autonomous_driving"

    assert episode.principal_seed == 42

    assert episode.evaluation_seed == 20_000

    assert episode.method == SafetyMethod.NONE

    assert episode.robustness_condition == RobustnessCondition.CLEAN

    assert episode.episode_length > 0

    assert episode.intervention_count == 0

    assert episode.total_action_correction_l2 == 0.0

    assert episode.max_action_correction_l2 == 0.0

    assert episode.policy_action_out_of_bounds_count == 0


def test_robotics_none_smoke_episode() -> None:
    from q_vla_forge.evaluation.safety_baseline import (
        run_no_filter_episode,
    )

    policy = load_frozen_ppo_policy(
        root=ROOT,
        domain="robotics",
        principal_seed=42,
    )

    episode = run_no_filter_episode(
        policy=policy,
        evaluation_seed=20_000,
    )

    assert episode.domain == "robotics"

    assert episode.principal_seed == 42

    assert episode.evaluation_seed == 20_000

    assert episode.method == SafetyMethod.NONE

    assert episode.robustness_condition == RobustnessCondition.CLEAN

    assert episode.episode_length > 0

    assert episode.intervention_count == 0

    assert episode.total_action_correction_l2 == 0.0

    assert episode.max_action_correction_l2 == 0.0

    assert episode.policy_action_out_of_bounds_count == 0


def test_non_frozen_eval_seed_rejected() -> None:
    from q_vla_forge.evaluation.safety_baseline import (
        run_no_filter_episode,
    )

    policy = load_frozen_ppo_policy(
        root=ROOT,
        domain="autonomous_driving",
        principal_seed=42,
    )

    with pytest.raises(ValueError):
        run_no_filter_episode(
            policy=policy,
            evaluation_seed=999,
        )
