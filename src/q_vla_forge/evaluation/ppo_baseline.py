"""Evidence utilities for Sprint 4.5 PPO baselines."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from q_vla_forge.rl.ppo import PPOTrainingResult

PPO_EVALUATION_SEEDS = tuple(
    range(
        20_000,
        20_020,
    )
)


def training_result_to_dict(
    result: PPOTrainingResult,
) -> dict[str, Any]:
    """Convert one PPO result into JSON-safe evidence."""
    return {
        "domain": result.domain,
        "seed": result.seed,
        "total_environment_steps": (result.total_environment_steps),
        "completed_training_episodes": (result.completed_training_episodes),
        "actor_parameters": (result.actor_parameters),
        "critic_parameters": (result.critic_parameters),
        "total_parameters": (result.total_parameters),
        "training_seconds": (result.training_seconds),
        "evaluations": [asdict(evaluation) for evaluation in result.evaluations],
    }


def best_evaluation_reward(
    result: PPOTrainingResult,
) -> float:
    """Return the best completed evaluation reward."""
    rewards = [evaluation.mean_reward for evaluation in result.evaluations]

    if not rewards:
        raise ValueError("PPO result contains no evaluations")

    best = max(rewards)

    if not math.isfinite(best):
        raise ValueError("best PPO reward must be finite")

    return float(best)


def derive_reward_target(
    *,
    random_reference_reward: float,
    ppo_best_reward: float,
    target_fraction: float = 0.95,
) -> float:
    """Derive paired target relative to random performance."""
    if not math.isfinite(random_reference_reward):
        raise ValueError("random reference must be finite")

    if not math.isfinite(ppo_best_reward):
        raise ValueError("PPO best reward must be finite")

    if ppo_best_reward <= random_reference_reward:
        raise ValueError(
            "PPO must improve over the random reference "
            "before a paired efficiency target is valid"
        )

    if not 0.0 < target_fraction <= 1.0:
        raise ValueError("target_fraction must lie in (0, 1]")

    return float(
        random_reference_reward
        + target_fraction * (ppo_best_reward - random_reference_reward)
    )


def first_target_reach(
    *,
    result: PPOTrainingResult,
    target_reward: float,
) -> tuple[
    int | None,
    int | None,
]:
    """Find the first held-out evaluation reaching target."""
    for evaluation in result.evaluations:
        if evaluation.mean_reward >= target_reward:
            return (
                evaluation.environment_steps,
                evaluation.completed_training_episodes,
            )

    return (
        None,
        None,
    )
