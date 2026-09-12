from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from q_vla_forge.data import Domain
from q_vla_forge.utils.reproducibility import DEFAULT_SEEDS


@dataclass(frozen=True)
class MetricSummary:
    """Mean, sample standard deviation, and raw values."""

    mean: float
    std: float
    values: tuple[float, ...]


@dataclass(frozen=True)
class DomainBaselineSummary:
    """Three-seed aggregate for one pilot domain."""

    domain: str
    seeds: tuple[int, ...]
    test_mse: MetricSummary
    test_mae: MetricSummary
    initial_validation_mse: MetricSummary
    best_validation_mse: MetricSummary
    final_validation_mse: MetricSummary
    mean_latency_ms: MetricSummary
    p95_latency_ms: MetricSummary
    action_mae: dict[str, MetricSummary]
    parameters: int
    fp32_model_size_bytes: int


@dataclass(frozen=True)
class MultiSeedBaselineSummary:
    """Cross-domain FP32 baseline validation summary."""

    method: str
    seeds: tuple[int, ...]
    driving: DomainBaselineSummary
    robotics: DomainBaselineSummary
    created_at: str


def summarize_values(
    values: Sequence[float],
) -> MetricSummary:
    """Calculate mean and sample standard deviation."""
    if not values:
        raise ValueError("values must not be empty")

    float_values = tuple(float(value) for value in values)

    std = statistics.stdev(float_values) if len(float_values) > 1 else 0.0

    return MetricSummary(
        mean=float(statistics.mean(float_values)),
        std=float(std),
        values=float_values,
    )


def _extract_metric(
    payloads: Sequence[dict[str, Any]],
    *keys: str,
) -> MetricSummary:
    values: list[float] = []

    for payload in payloads:
        value: Any = payload

        for key in keys:
            value = value[key]

        values.append(float(value))

    return summarize_values(values)


def _validate_payloads(
    payloads: Sequence[dict[str, Any]],
    expected_domain: Domain,
    expected_seeds: tuple[int, ...],
) -> None:
    if len(payloads) != len(expected_seeds):
        raise ValueError("payload count must match seed count")

    actual_seeds = tuple(int(payload["seed"]) for payload in payloads)

    if actual_seeds != expected_seeds:
        raise ValueError("payload seeds must match expected seed order")

    if any(payload["domain"] != expected_domain.value for payload in payloads):
        raise ValueError("payload domain does not match expected domain")

    if any(payload["method"] != "shared_vla_fp32" for payload in payloads):
        raise ValueError("all payloads must use shared_vla_fp32")

    parameter_counts = {int(payload["parameters"]) for payload in payloads}

    if len(parameter_counts) != 1:
        raise ValueError("parameter counts must match across seeds")

    model_sizes = {int(payload["fp32_model_size_bytes"]) for payload in payloads}

    if len(model_sizes) != 1:
        raise ValueError("model sizes must match across seeds")


def _domain_summary(
    payloads: Sequence[dict[str, Any]],
    domain: Domain,
    seeds: tuple[int, ...],
) -> DomainBaselineSummary:
    _validate_payloads(
        payloads,
        domain,
        seeds,
    )

    if domain == Domain.AUTONOMOUS_DRIVING:
        action_names = (
            "steering_mae",
            "acceleration_mae",
            "braking_mae",
        )
    elif domain == Domain.ROBOTICS:
        action_names = (
            "delta_x_mae",
            "delta_y_mae",
            "gripper_mae",
        )
    else:
        raise ValueError(f"unsupported domain: {domain}")

    action_mae = {
        name: _extract_metric(
            payloads,
            "test_metrics",
            name,
        )
        for name in action_names
    }

    first = payloads[0]

    return DomainBaselineSummary(
        domain=domain.value,
        seeds=seeds,
        test_mse=_extract_metric(
            payloads,
            "test_metrics",
            "test_mse",
        ),
        test_mae=_extract_metric(
            payloads,
            "test_metrics",
            "test_mae",
        ),
        initial_validation_mse=_extract_metric(
            payloads,
            "initial_validation_mse",
        ),
        best_validation_mse=_extract_metric(
            payloads,
            "best_validation_mse",
        ),
        final_validation_mse=_extract_metric(
            payloads,
            "final_validation_mse",
        ),
        mean_latency_ms=_extract_metric(
            payloads,
            "latency",
            "mean_ms",
        ),
        p95_latency_ms=_extract_metric(
            payloads,
            "latency",
            "p95_ms",
        ),
        action_mae=action_mae,
        parameters=int(first["parameters"]),
        fp32_model_size_bytes=int(first["fp32_model_size_bytes"]),
    )


def aggregate_baseline_payloads(
    driving_payloads: Sequence[dict[str, Any]],
    robotics_payloads: Sequence[dict[str, Any]],
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> MultiSeedBaselineSummary:
    """Aggregate independent baseline runs across domains."""
    if not seeds:
        raise ValueError("seeds must not be empty")

    driving = _domain_summary(
        driving_payloads,
        Domain.AUTONOMOUS_DRIVING,
        seeds,
    )

    robotics = _domain_summary(
        robotics_payloads,
        Domain.ROBOTICS,
        seeds,
    )

    if driving.parameters != robotics.parameters:
        raise ValueError("shared architecture parameter counts must match")

    return MultiSeedBaselineSummary(
        method="shared_vla_fp32",
        seeds=seeds,
        driving=driving,
        robotics=robotics,
        created_at=(datetime.now(UTC).isoformat()),
    )


def save_multiseed_summary(
    summary: MultiSeedBaselineSummary,
    output_path: Path,
) -> None:
    """Save aggregate evidence as JSON."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            asdict(summary),
            indent=2,
        ),
        encoding="utf-8",
    )
