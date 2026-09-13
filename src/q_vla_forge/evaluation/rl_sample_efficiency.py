"""Sample-efficiency analysis utilities for Sprint 4.11."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass

TOTAL_ENVIRONMENT_STEPS = 20_000


@dataclass(frozen=True)
class LearningCurveSummary:
    """Secondary descriptive metrics for one RL run."""

    normalized_auc: float
    best_normalized_progress: float
    final_normalized_progress: float
    best_reward: float
    final_reward: float


def normalized_target_progress(
    *,
    reward: float,
    random_reference: float,
    target_reward: float,
) -> float:
    """Normalize reward between random reference and paired target."""

    values = (
        reward,
        random_reference,
        target_reward,
    )

    if not all(math.isfinite(value) for value in values):
        raise ValueError("all rewards must be finite")

    denominator = target_reward - random_reference

    if denominator <= 0.0:
        raise ValueError("target must exceed random reference")

    return float((reward - random_reference) / denominator)


def trapezoidal_auc(
    *,
    steps: Sequence[int],
    values: Sequence[float],
) -> float:
    """Calculate trapezoidal area under a learning curve."""

    if len(steps) != len(values):
        raise ValueError("steps and values must have equal length")

    if len(steps) < 2:
        raise ValueError("at least two points are required")

    for value in values:
        if not math.isfinite(value):
            raise ValueError("curve values must be finite")

    area = 0.0

    for index in range(
        1,
        len(steps),
    ):
        left_step = int(steps[index - 1])

        right_step = int(steps[index])

        if right_step <= left_step:
            raise ValueError("steps must be strictly increasing")

        width = right_step - left_step

        left_value = float(values[index - 1])

        right_value = float(values[index])

        area += width * (left_value + right_value) / 2.0

    return float(area)


def normalized_learning_auc(
    *,
    steps: Sequence[int],
    normalized_progress: Sequence[float],
    total_environment_steps: int = TOTAL_ENVIRONMENT_STEPS,
) -> float:
    """Return budget-normalized target-progress AUC."""

    if total_environment_steps <= 0:
        raise ValueError("total_environment_steps must be positive")

    area = trapezoidal_auc(
        steps=steps,
        values=normalized_progress,
    )

    return float(area / total_environment_steps)


def summarize_learning_curve(
    *,
    steps: Sequence[int],
    rewards: Sequence[float],
    random_reference: float,
    target_reward: float,
) -> LearningCurveSummary:
    """Summarize one frozen learning trajectory."""

    if len(steps) != len(rewards):
        raise ValueError("steps and rewards must have equal length")

    if not rewards:
        raise ValueError("learning curve cannot be empty")

    progress = [
        normalized_target_progress(
            reward=float(reward),
            random_reference=random_reference,
            target_reward=target_reward,
        )
        for reward in rewards
    ]

    auc = normalized_learning_auc(
        steps=steps,
        normalized_progress=progress,
    )

    return LearningCurveSummary(
        normalized_auc=auc,
        best_normalized_progress=float(max(progress)),
        final_normalized_progress=float(progress[-1]),
        best_reward=float(max(rewards)),
        final_reward=float(rewards[-1]),
    )


def sample_mean_sd(
    values: Sequence[float],
) -> tuple[float, float]:
    """Arithmetic mean and sample standard deviation."""

    if not values:
        raise ValueError("values cannot be empty")

    if not all(math.isfinite(value) for value in values):
        raise ValueError("values must be finite")

    mean = float(statistics.mean(values))

    sd = float(statistics.stdev(values)) if len(values) > 1 else 0.0

    return (
        mean,
        sd,
    )


def actor_parameter_reduction_percent(
    *,
    classical_parameters: int,
    hybrid_parameters: int,
) -> float:
    """Calculate actor parameter reduction."""

    if classical_parameters <= 0:
        raise ValueError("classical parameter count must be positive")

    if hybrid_parameters <= 0:
        raise ValueError("hybrid parameter count must be positive")

    if hybrid_parameters > classical_parameters:
        raise ValueError("hybrid parameters exceed classical parameters")

    return float(
        100.0 * (classical_parameters - hybrid_parameters) / classical_parameters
    )


def summary_to_dict(
    summary: LearningCurveSummary,
) -> dict[str, float]:
    """Convert one learning-curve summary to a dictionary."""

    return asdict(summary)
