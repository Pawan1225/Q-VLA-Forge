from __future__ import annotations

import pytest

from q_vla_forge.evaluation.ppo_baseline import (
    PPO_EVALUATION_SEEDS,
    derive_reward_target,
)


def test_evaluation_seed_bank() -> None:
    assert len(PPO_EVALUATION_SEEDS) == 20

    assert PPO_EVALUATION_SEEDS[0] == 20_000

    assert PPO_EVALUATION_SEEDS[-1] == 20_019


def test_positive_reward_target() -> None:
    target = derive_reward_target(
        random_reference_reward=-50.0,
        ppo_best_reward=10.0,
    )

    assert target == pytest.approx(7.0)


def test_negative_reward_target() -> None:
    target = derive_reward_target(
        random_reference_reward=-100.0,
        ppo_best_reward=-20.0,
    )

    assert target == pytest.approx(-24.0)


def test_target_requires_ppo_improvement() -> None:
    with pytest.raises(ValueError):
        derive_reward_target(
            random_reference_reward=5.0,
            ppo_best_reward=4.0,
        )


def test_target_rejects_equal_performance() -> None:
    with pytest.raises(ValueError):
        derive_reward_target(
            random_reference_reward=5.0,
            ppo_best_reward=5.0,
        )


def test_target_fraction_must_be_positive() -> None:
    with pytest.raises(ValueError):
        derive_reward_target(
            random_reference_reward=-10.0,
            ppo_best_reward=10.0,
            target_fraction=0.0,
        )


def test_target_fraction_cannot_exceed_one() -> None:
    with pytest.raises(ValueError):
        derive_reward_target(
            random_reference_reward=-10.0,
            ppo_best_reward=10.0,
            target_fraction=1.01,
        )


def test_non_finite_random_reference_is_rejected() -> None:
    with pytest.raises(ValueError):
        derive_reward_target(
            random_reference_reward=float("nan"),
            ppo_best_reward=10.0,
        )


def test_non_finite_ppo_best_is_rejected() -> None:
    with pytest.raises(ValueError):
        derive_reward_target(
            random_reference_reward=-10.0,
            ppo_best_reward=float("inf"),
        )
