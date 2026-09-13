"""Sprint 4.9 robotics QML evidence utilities."""

from __future__ import annotations

import math

from q_vla_forge.rl.hybrid_ppo import (
    HybridPPOTrainingResult,
)


def first_qml_target_reach(
    *,
    result: HybridPPOTrainingResult,
    target_reward: float,
) -> tuple[int | None, int | None]:
    """Return first held-out evaluation reaching the frozen target."""

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
    """Positive means QML required fewer environment interactions."""

    if classical_steps <= 0:
        raise ValueError("classical steps must be positive")

    if qml_steps is None:
        return None

    if qml_steps < 0:
        raise ValueError("QML steps cannot be negative")

    return float(100.0 * (classical_steps - qml_steps) / classical_steps)
