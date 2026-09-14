"""Sprint 5.13D structured-state robustness consolidation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev

PRINCIPAL_SEEDS = (
    42,
    123,
    456,
)


@dataclass(frozen=True)
class StructuredStateSeedResult:
    domain: str
    method: str
    perturbation_family: str
    perturbation_name: str
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
class MetricSummary:
    seed42: float
    seed123: float
    seed456: float
    mean: float
    sample_sd: float


def summarize_seed_values(
    values: dict[int, float],
) -> MetricSummary:
    if set(values) != set(PRINCIPAL_SEEDS):
        raise ValueError("values must contain seeds 42, 123, 456")

    ordered = [float(values[seed]) for seed in PRINCIPAL_SEEDS]

    return MetricSummary(
        seed42=ordered[0],
        seed123=ordered[1],
        seed456=ordered[2],
        mean=float(mean(ordered)),
        sample_sd=float(stdev(ordered)),
    )


def summarize_rows(
    rows: list[StructuredStateSeedResult],
) -> dict[str, MetricSummary]:
    if {row.principal_seed for row in rows} != set(PRINCIPAL_SEEDS):
        raise ValueError("structured-state group must contain three seeds")

    metric_names = (
        "violation_step_rate",
        "critical_violation_step_rate",
        "constraint_violation_rate",
        "reward",
        "success_rate",
        "intervention_rate",
        "mean_correction_l2",
        "violation_delta_from_clean",
        "critical_delta_from_clean",
        "constraint_delta_from_clean",
        "reward_delta_from_clean",
        "success_delta_from_clean",
        "strict_lyapunov_decrease_rate",
        "lyapunov_nonincrease_rate",
    )

    result: dict[str, MetricSummary] = {}

    for metric in metric_names:
        result[metric] = summarize_seed_values(
            {
                row.principal_seed: float(
                    getattr(
                        row,
                        metric,
                    )
                )
                for row in rows
            }
        )

    return result


def worst_condition(
    rows: list[dict[str, object]],
    *,
    metric: str,
    mode: str,
) -> dict[str, object]:
    if not rows:
        raise ValueError("rows must be non-empty")

    if mode not in {
        "max",
        "min",
    }:
        raise ValueError("mode must be max or min")

    def metric_value(
        row: dict[str, object],
    ) -> float:
        value = row[metric]

        if not isinstance(
            value,
            dict,
        ):
            raise TypeError(f"{metric} must contain aggregate data")

        return float(value["mean"])

    if mode == "max":
        return max(
            rows,
            key=metric_value,
        )

    return min(
        rows,
        key=metric_value,
    )
