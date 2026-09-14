"""Sprint 5.13E action-robustness consolidation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev

PRINCIPAL_SEEDS = (
    42,
    123,
    456,
)


@dataclass(frozen=True)
class ActionSeedResult:
    domain: str
    method: str
    perturbation_family: str
    perturbation_name: str
    principal_seed: int

    perturbed_violation_step_rate: float
    executed_violation_step_rate: float
    executed_constraint_violation_rate: float
    critical_violation_step_rate: float

    mean_reward: float
    success_rate: float

    intervention_rate: float
    mean_safety_correction_l2: float
    p95_safety_correction_l2: float

    # Undefined when the relevant denominator is zero.
    recovery_rate: float | None
    within_filter_violation_reduction: float | None

    executed_violation_delta_from_clean: float
    reward_delta_from_clean: float
    success_delta_from_clean: float

    selected_lower_than_perturbed_rate: float
    strict_lyapunov_decrease_rate: float
    lyapunov_nonincrease_rate: float

    environment_interface_adjustment_rate: float

    proposed_to_perturbed_gripper_semantic_change_rate: float
    perturbed_to_executed_gripper_semantic_change_rate: float

    unsafe_perturbed_steps: int
    recovered_unsafe_steps: int
    unresolved_unsafe_steps: int

    steps_object_grasped: int
    interventions_while_grasped: int
    perturbed_unsafe_steps_while_grasped: int


@dataclass(frozen=True)
class MetricSummary:
    seed42: float
    seed123: float
    seed456: float
    mean: float
    sample_sd: float


@dataclass(frozen=True)
class NullableMetricSummary:
    seed42: float | None
    seed123: float | None
    seed456: float | None
    mean: float | None
    sample_sd: float | None
    defined_seed_count: int


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


def summarize_nullable_seed_values(
    values: dict[int, float | None],
) -> NullableMetricSummary:
    if set(values) != set(PRINCIPAL_SEEDS):
        raise ValueError("values must contain seeds 42, 123, 456")

    ordered = [values[seed] for seed in PRINCIPAL_SEEDS]

    defined = [float(value) for value in ordered if value is not None]

    if not defined:
        aggregate_mean: float | None = None
        aggregate_sd: float | None = None

    elif len(defined) == 1:
        aggregate_mean = defined[0]
        aggregate_sd = 0.0

    else:
        aggregate_mean = float(mean(defined))
        aggregate_sd = float(stdev(defined))

    return NullableMetricSummary(
        seed42=ordered[0],
        seed123=ordered[1],
        seed456=ordered[2],
        mean=aggregate_mean,
        sample_sd=aggregate_sd,
        defined_seed_count=len(defined),
    )


def summarize_rows(
    rows: list[ActionSeedResult],
) -> dict[
    str,
    MetricSummary | NullableMetricSummary,
]:
    if {row.principal_seed for row in rows} != set(PRINCIPAL_SEEDS):
        raise ValueError("action group must contain three principal seeds")

    required_metrics = (
        "perturbed_violation_step_rate",
        "executed_violation_step_rate",
        "executed_constraint_violation_rate",
        "critical_violation_step_rate",
        "mean_reward",
        "success_rate",
        "intervention_rate",
        "mean_safety_correction_l2",
        "p95_safety_correction_l2",
        "executed_violation_delta_from_clean",
        "reward_delta_from_clean",
        "success_delta_from_clean",
        "selected_lower_than_perturbed_rate",
        "strict_lyapunov_decrease_rate",
        "lyapunov_nonincrease_rate",
        "environment_interface_adjustment_rate",
        "proposed_to_perturbed_gripper_semantic_change_rate",
        "perturbed_to_executed_gripper_semantic_change_rate",
    )

    result: dict[
        str,
        MetricSummary | NullableMetricSummary,
    ] = {}

    for metric in required_metrics:
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

    for metric in (
        "recovery_rate",
        "within_filter_violation_reduction",
    ):
        result[metric] = summarize_nullable_seed_values(
            {
                row.principal_seed: getattr(
                    row,
                    metric,
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

    def value(
        row: dict[str, object],
    ) -> float:
        metric_value = row[metric]

        if not isinstance(
            metric_value,
            dict,
        ):
            raise TypeError(f"{metric} must be an aggregate dict")

        aggregate_mean = metric_value.get("mean")

        if aggregate_mean is None:
            raise ValueError(f"{metric} has no defined aggregate mean")

        return float(aggregate_mean)

    if mode == "max":
        return max(
            rows,
            key=value,
        )

    return min(
        rows,
        key=value,
    )
