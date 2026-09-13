"""Sprint 4.8 autonomous-driving QML evidence utilities."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from q_vla_forge.rl.hybrid_ppo import (
    HybridPPOTrainingResult,
)


def hybrid_training_result_to_dict(
    result: HybridPPOTrainingResult,
) -> dict[str, Any]:
    """Serialize one hybrid PPO training result."""
    return {
        "domain": result.domain,
        "seed": result.seed,
        "total_environment_steps": (result.total_environment_steps),
        "completed_training_episodes": (result.completed_training_episodes),
        "projection_parameters": (result.projection_parameters),
        "quantum_parameters": (result.quantum_parameters),
        "action_head_parameters": (result.action_head_parameters),
        "actor_parameters": (result.actor_parameters),
        "critic_parameters": (result.critic_parameters),
        "total_parameters": (result.total_parameters),
        "training_seconds": (result.training_seconds),
        "evaluations": [asdict(evaluation) for evaluation in result.evaluations],
    }


def first_qml_target_reach(
    *,
    result: HybridPPOTrainingResult,
    target_reward: float,
) -> tuple[
    int | None,
    int | None,
]:
    """Return the first QML target crossing in steps and episodes."""
    if not math.isfinite(target_reward):
        raise ValueError("target reward must be finite")

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


def sample_efficiency_improvement_percent(
    *,
    classical_steps: int,
    qml_steps: int | None,
) -> float | None:
    """Return paired interaction reduction; positive favors QML."""
    if classical_steps <= 0:
        raise ValueError("classical steps must be positive")

    if qml_steps is None:
        return None

    if qml_steps < 0:
        raise ValueError("QML steps cannot be negative")

    return float(100.0 * (classical_steps - qml_steps) / classical_steps)
