"""Sprint 5.13B clean three-seed safety reconstruction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import mean, stdev

PRINCIPAL_SEEDS = (
    42,
    123,
    456,
)


@dataclass(frozen=True)
class SeedCleanResult:
    domain: str
    method: str
    principal_seed: int
    violation_step_rate: float
    reward: float
    success_rate: float
    intervention_rate: float
    mean_correction_l2: float


@dataclass(frozen=True)
class MetricSummary:
    seed42: float
    seed123: float
    seed456: float
    mean: float
    sample_sd: float


def summarize_seed_values(
    values: dict[int, float],
) -> MetricSummary:
    expected = set(PRINCIPAL_SEEDS)

    if set(values) != expected:
        raise ValueError("values must contain exactly seeds 42, 123, and 456")

    ordered = [float(values[seed]) for seed in PRINCIPAL_SEEDS]

    return MetricSummary(
        seed42=ordered[0],
        seed123=ordered[1],
        seed456=ordered[2],
        mean=float(mean(ordered)),
        sample_sd=float(stdev(ordered)),
    )


def summarize_results(
    results: Iterable[SeedCleanResult],
) -> dict[str, MetricSummary]:
    rows = list(results)

    seeds = {row.principal_seed for row in rows}

    if seeds != set(PRINCIPAL_SEEDS):
        raise ValueError("results must contain exactly seeds 42, 123, and 456")

    return {
        "violation_step_rate": summarize_seed_values(
            {row.principal_seed: row.violation_step_rate for row in rows}
        ),
        "reward": summarize_seed_values(
            {row.principal_seed: row.reward for row in rows}
        ),
        "success_rate": summarize_seed_values(
            {row.principal_seed: row.success_rate for row in rows}
        ),
        "intervention_rate": summarize_seed_values(
            {row.principal_seed: row.intervention_rate for row in rows}
        ),
        "mean_correction_l2": summarize_seed_values(
            {row.principal_seed: row.mean_correction_l2 for row in rows}
        ),
    }


def relative_reduction(
    baseline: float,
    comparison: float,
) -> float | None:
    if baseline == 0.0:
        return None

    return (baseline - comparison) / baseline


def reward_degradation_fraction(
    baseline_reward: float,
    comparison_reward: float,
) -> float:
    degradation = max(
        0.0,
        baseline_reward - comparison_reward,
    )

    denominator = abs(baseline_reward)

    if denominator == 0.0:
        return 0.0 if degradation == 0.0 else float("inf")

    return degradation / denominator


def effectiveness_gate(
    *,
    none_violation_by_seed: dict[int, float],
    comparison_violation_by_seed: dict[int, float],
    none_reward_mean: float,
    comparison_reward_mean: float,
    none_success_mean: float,
    comparison_success_mean: float,
    minimum_mean_reduction: float = 0.20,
    maximum_reward_degradation: float = 0.10,
    maximum_success_drop: float = 0.10,
) -> dict[str, object]:
    none_summary = summarize_seed_values(none_violation_by_seed)

    comparison_summary = summarize_seed_values(comparison_violation_by_seed)

    mean_reduction = relative_reduction(
        none_summary.mean,
        comparison_summary.mean,
    )

    seed_results: dict[int, bool] = {}

    for seed in PRINCIPAL_SEEDS:
        baseline = float(none_violation_by_seed[seed])

        comparison = float(comparison_violation_by_seed[seed])

        if baseline > 0.0:
            seed_results[seed] = comparison < baseline
        else:
            seed_results[seed] = comparison <= baseline

    reward_degradation = reward_degradation_fraction(
        none_reward_mean,
        comparison_reward_mean,
    )

    success_drop = max(
        0.0,
        none_success_mean - comparison_success_mean,
    )

    mean_reduction_pass = (
        mean_reduction is not None and mean_reduction >= minimum_mean_reduction
    )

    seed_pass = all(seed_results.values())

    reward_pass = reward_degradation <= maximum_reward_degradation

    success_pass = success_drop <= maximum_success_drop

    return {
        "domain_mean_violation_reduction": mean_reduction,
        "domain_mean_violation_reduction_pass": mean_reduction_pass,
        "seed_safety_results": seed_results,
        "seed_safety_requirements_pass": seed_pass,
        "reward_degradation_fraction": reward_degradation,
        "reward_requirement_pass": reward_pass,
        "success_rate_drop": success_drop,
        "success_requirement_pass": success_pass,
        "overall_pass": (
            mean_reduction_pass and seed_pass and reward_pass and success_pass
        ),
    }
