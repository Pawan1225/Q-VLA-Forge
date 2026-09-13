"""Three-seed RL validation utilities for Sprint 4.10."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

PRINCIPAL_SEEDS = (
    42,
    123,
    456,
)

EXPECTED_EVALUATION_STEPS = tuple(
    range(
        0,
        20_001,
        1_000,
    )
)


@dataclass(frozen=True)
class SeedValidationRecord:
    """One paired classical-vs-QML validation record."""

    domain: str
    seed: int

    target_reward: float

    classical_best_reward: float
    classical_final_reward: float
    classical_final_success_rate: float
    classical_steps_to_target: int

    qml_best_reward: float
    qml_final_reward: float
    qml_final_success_rate: float
    qml_steps_to_target: int | None

    qml_target_reached: bool

    best_reward_target_gap: float
    sample_efficiency_improvement_percent: float | None


@dataclass(frozen=True)
class DomainValidationSummary:
    """Three-seed validation summary for one domain."""

    domain: str
    seeds: tuple[int, ...]

    classical_target_reach_count: int
    qml_target_reach_count: int

    qml_target_reach_rate: float

    classical_best_reward_mean: float
    classical_best_reward_sd: float

    qml_best_reward_mean: float
    qml_best_reward_sd: float

    qml_final_reward_mean: float
    qml_final_reward_sd: float

    qml_final_success_mean: float
    qml_final_success_sd: float

    qml_best_reward_target_gap_mean: float
    qml_best_reward_target_gap_sd: float

    valid_qml_efficiency_count: int

    robust_10_percent_criterion: bool


def sample_mean_sd(
    values: Sequence[float],
) -> tuple[float, float]:
    """Return arithmetic mean and sample standard deviation."""

    if not values:
        raise ValueError("at least one value is required")

    for value in values:
        if not math.isfinite(value):
            raise ValueError("all values must be finite")

    mean = float(statistics.mean(values))

    sd = float(statistics.stdev(values)) if len(values) > 1 else 0.0

    return (
        mean,
        sd,
    )


def validate_evaluation_steps(
    evaluations: Sequence[dict[str, Any]],
) -> None:
    """Validate the frozen 21-point evaluation schedule."""

    steps = tuple(int(evaluation["environment_steps"]) for evaluation in evaluations)

    if steps != EXPECTED_EVALUATION_STEPS:
        raise ValueError(
            "evaluation schedule does not match " "0..20000 in 1000-step increments"
        )


def validate_evaluations(
    evaluations: Sequence[dict[str, Any]],
) -> None:
    """Validate one full held-out evaluation trajectory."""

    if len(evaluations) != 21:
        raise ValueError("principal RL run must contain 21 evaluations")

    validate_evaluation_steps(evaluations)

    for evaluation in evaluations:
        reward = float(evaluation["mean_reward"])

        reward_sd = float(evaluation["reward_standard_deviation"])

        success_rate = float(evaluation["success_rate"])

        if not math.isfinite(reward):
            raise ValueError("mean reward must be finite")

        if not math.isfinite(reward_sd):
            raise ValueError("reward SD must be finite")

        if reward_sd < 0.0:
            raise ValueError("reward SD cannot be negative")

        if not 0.0 <= success_rate <= 1.0:
            raise ValueError("success rate must lie in [0, 1]")


def best_evaluation_reward(
    evaluations: Sequence[dict[str, Any]],
) -> float:
    """Return the best held-out mean reward."""

    validate_evaluations(evaluations)

    return float(max(evaluation["mean_reward"] for evaluation in evaluations))


def final_evaluation(
    evaluations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Return the final held-out evaluation."""

    validate_evaluations(evaluations)

    return evaluations[-1]


def first_target_crossing(
    *,
    evaluations: Sequence[dict[str, Any]],
    target_reward: float,
) -> tuple[int | None, int | None]:
    """Reconstruct first held-out target crossing."""

    if not math.isfinite(target_reward):
        raise ValueError("target_reward must be finite")

    validate_evaluations(evaluations)

    for evaluation in evaluations:
        if float(evaluation["mean_reward"]) >= target_reward:
            return (
                int(evaluation["environment_steps"]),
                int(evaluation["completed_training_episodes"]),
            )

    return (
        None,
        None,
    )


def sample_efficiency_improvement(
    *,
    classical_steps: int,
    qml_steps: int | None,
) -> float | None:
    """Positive values mean fewer QML interactions."""

    if classical_steps <= 0:
        raise ValueError("classical_steps must be positive")

    if qml_steps is None:
        return None

    return float(100.0 * (classical_steps - qml_steps) / classical_steps)


def build_seed_validation_record(
    *,
    domain: str,
    seed: int,
    target_reward: float,
    classical_run: dict[str, Any],
    qml_run: dict[str, Any],
) -> SeedValidationRecord:
    """Build one independently reconstructed paired record."""

    if seed not in PRINCIPAL_SEEDS:
        raise ValueError("unexpected principal seed")

    if classical_run["domain"] != domain:
        raise ValueError("classical run domain mismatch")

    if qml_run["domain"] != domain:
        raise ValueError("QML run domain mismatch")

    if int(classical_run["seed"]) != seed:
        raise ValueError("classical seed mismatch")

    if int(qml_run["seed"]) != seed:
        raise ValueError("QML seed mismatch")

    if int(classical_run["total_environment_steps"]) != 20_000:
        raise ValueError("classical run budget mismatch")

    if int(qml_run["total_environment_steps"]) != 20_000:
        raise ValueError("QML run budget mismatch")

    classical_evaluations = classical_run["evaluations"]

    qml_evaluations = qml_run["evaluations"]

    validate_evaluations(classical_evaluations)

    validate_evaluations(qml_evaluations)

    classical_best = best_evaluation_reward(classical_evaluations)

    qml_best = best_evaluation_reward(qml_evaluations)

    classical_final = final_evaluation(classical_evaluations)

    qml_final = final_evaluation(qml_evaluations)

    (
        classical_steps,
        _,
    ) = first_target_crossing(
        evaluations=classical_evaluations,
        target_reward=target_reward,
    )

    if classical_steps is None:
        raise ValueError("classical PPO must reach its frozen paired target")

    (
        qml_steps,
        _,
    ) = first_target_crossing(
        evaluations=qml_evaluations,
        target_reward=target_reward,
    )

    improvement = sample_efficiency_improvement(
        classical_steps=classical_steps,
        qml_steps=qml_steps,
    )

    return SeedValidationRecord(
        domain=domain,
        seed=seed,
        target_reward=float(target_reward),
        classical_best_reward=(classical_best),
        classical_final_reward=float(classical_final["mean_reward"]),
        classical_final_success_rate=float(classical_final["success_rate"]),
        classical_steps_to_target=(classical_steps),
        qml_best_reward=qml_best,
        qml_final_reward=float(qml_final["mean_reward"]),
        qml_final_success_rate=float(qml_final["success_rate"]),
        qml_steps_to_target=qml_steps,
        qml_target_reached=(qml_steps is not None),
        best_reward_target_gap=float(qml_best - target_reward),
        sample_efficiency_improvement_percent=(improvement),
    )


def summarize_domain(
    *,
    domain: str,
    records: Sequence[SeedValidationRecord],
) -> DomainValidationSummary:
    """Aggregate exactly three paired seed records."""

    if len(records) != 3:
        raise ValueError("domain validation requires exactly three seeds")

    seeds = tuple(record.seed for record in records)

    if set(seeds) != set(PRINCIPAL_SEEDS):
        raise ValueError("domain seed set mismatch")

    if any(record.domain != domain for record in records):
        raise ValueError("mixed domains in validation summary")

    classical_best_mean, classical_best_sd = sample_mean_sd(
        [record.classical_best_reward for record in records]
    )

    qml_best_mean, qml_best_sd = sample_mean_sd(
        [record.qml_best_reward for record in records]
    )

    qml_final_mean, qml_final_sd = sample_mean_sd(
        [record.qml_final_reward for record in records]
    )

    qml_success_mean, qml_success_sd = sample_mean_sd(
        [record.qml_final_success_rate for record in records]
    )

    gap_mean, gap_sd = sample_mean_sd(
        [record.best_reward_target_gap for record in records]
    )

    qml_reach_count = sum(record.qml_target_reached for record in records)

    valid_improvements = [
        record.sample_efficiency_improvement_percent
        for record in records
        if (record.sample_efficiency_improvement_percent is not None)
    ]

    robust_criterion = bool(
        qml_reach_count == 3
        and len(valid_improvements) == 3
        and statistics.mean(valid_improvements) >= 10.0
    )

    return DomainValidationSummary(
        domain=domain,
        seeds=tuple(sorted(seeds)),
        classical_target_reach_count=3,
        qml_target_reach_count=(qml_reach_count),
        qml_target_reach_rate=(qml_reach_count / 3.0),
        classical_best_reward_mean=(classical_best_mean),
        classical_best_reward_sd=(classical_best_sd),
        qml_best_reward_mean=(qml_best_mean),
        qml_best_reward_sd=(qml_best_sd),
        qml_final_reward_mean=(qml_final_mean),
        qml_final_reward_sd=(qml_final_sd),
        qml_final_success_mean=(qml_success_mean),
        qml_final_success_sd=(qml_success_sd),
        qml_best_reward_target_gap_mean=(gap_mean),
        qml_best_reward_target_gap_sd=(gap_sd),
        valid_qml_efficiency_count=len(valid_improvements),
        robust_10_percent_criterion=(robust_criterion),
    )


def seed_record_to_dict(
    record: SeedValidationRecord,
) -> dict[str, Any]:
    """Convert one seed record to a serializable dictionary."""

    return asdict(record)


def domain_summary_to_dict(
    summary: DomainValidationSummary,
) -> dict[str, Any]:
    """Convert one domain summary to a serializable dictionary."""

    result = asdict(summary)

    result["seeds"] = list(summary.seeds)

    return result
