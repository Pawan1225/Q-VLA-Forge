"""Sprint 5.13C Gaussian robustness consolidation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev

PRINCIPAL_SEEDS = (
    42,
    123,
    456,
)

SIGMAS = (
    0.0,
    0.01,
    0.05,
    0.10,
)


@dataclass(frozen=True)
class GaussianSeedResult:
    domain: str
    method: str
    sigma: float
    principal_seed: int

    violation_step_rate: float
    critical_violation_step_rate: float
    constraint_violation_rate: float

    reward: float
    success_rate: float

    intervention_rate: float
    mean_correction_l2: float

    violation_delta_from_clean: float
    critical_delta_from_clean: float
    constraint_delta_from_clean: float
    reward_delta_from_clean: float
    success_delta_from_clean: float

    strict_lyapunov_decrease_rate: float
    lyapunov_nonincrease_rate: float

    steps_object_grasped: int


@dataclass(frozen=True)
class ThreeSeedSummary:
    seed42: float
    seed123: float
    seed456: float
    mean: float
    sample_sd: float


def summarize_three_seeds(
    values: dict[int, float],
) -> ThreeSeedSummary:
    expected = set(PRINCIPAL_SEEDS)

    if set(values) != expected:
        raise ValueError("values must contain exactly seeds 42, 123, and 456")

    ordered = [float(values[seed]) for seed in PRINCIPAL_SEEDS]

    return ThreeSeedSummary(
        seed42=ordered[0],
        seed123=ordered[1],
        seed456=ordered[2],
        mean=float(mean(ordered)),
        sample_sd=float(stdev(ordered)),
    )


def summarize_gaussian_rows(
    rows: list[GaussianSeedResult],
) -> dict[str, ThreeSeedSummary]:
    if {row.principal_seed for row in rows} != set(PRINCIPAL_SEEDS):
        raise ValueError("Gaussian group must contain exactly three principal seeds")

    return {
        "violation_step_rate": summarize_three_seeds(
            {row.principal_seed: row.violation_step_rate for row in rows}
        ),
        "critical_violation_step_rate": summarize_three_seeds(
            {row.principal_seed: row.critical_violation_step_rate for row in rows}
        ),
        "constraint_violation_rate": summarize_three_seeds(
            {row.principal_seed: row.constraint_violation_rate for row in rows}
        ),
        "reward": summarize_three_seeds(
            {row.principal_seed: row.reward for row in rows}
        ),
        "success_rate": summarize_three_seeds(
            {row.principal_seed: row.success_rate for row in rows}
        ),
        "intervention_rate": summarize_three_seeds(
            {row.principal_seed: row.intervention_rate for row in rows}
        ),
        "mean_correction_l2": summarize_three_seeds(
            {row.principal_seed: row.mean_correction_l2 for row in rows}
        ),
        "violation_delta_from_clean": summarize_three_seeds(
            {row.principal_seed: row.violation_delta_from_clean for row in rows}
        ),
        "critical_delta_from_clean": summarize_three_seeds(
            {row.principal_seed: row.critical_delta_from_clean for row in rows}
        ),
        "constraint_delta_from_clean": summarize_three_seeds(
            {row.principal_seed: row.constraint_delta_from_clean for row in rows}
        ),
        "reward_delta_from_clean": summarize_three_seeds(
            {row.principal_seed: row.reward_delta_from_clean for row in rows}
        ),
        "success_delta_from_clean": summarize_three_seeds(
            {row.principal_seed: row.success_delta_from_clean for row in rows}
        ),
        "strict_lyapunov_decrease_rate": summarize_three_seeds(
            {row.principal_seed: row.strict_lyapunov_decrease_rate for row in rows}
        ),
        "lyapunov_nonincrease_rate": summarize_three_seeds(
            {row.principal_seed: row.lyapunov_nonincrease_rate for row in rows}
        ),
    }


def worst_condition(
    aggregates: list[dict[str, object]],
    *,
    metric: str,
    mode: str,
) -> dict[str, object]:
    if not aggregates:
        raise ValueError("aggregates must be non-empty")

    if mode not in {
        "max",
        "min",
    }:
        raise ValueError("mode must be max or min")

    def value(
        row: dict[str, object],
    ) -> float:
        metric_value = row[metric]

        if not isinstance(
            metric_value,
            dict,
        ):
            raise TypeError(f"{metric} must be an aggregate dict")

        return float(metric_value["mean"])

    selected = (
        max(
            aggregates,
            key=value,
        )
        if mode == "max"
        else min(
            aggregates,
            key=value,
        )
    )

    return selected
