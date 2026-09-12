"""Three-seed compression validation aggregation."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CompressionMetricSummary:
    """Mean, sample standard deviation, and raw metric values."""

    mean: float
    std: float
    values: tuple[float, ...]


@dataclass(frozen=True)
class CompressionMethodValidation:
    """Three-seed validation summary for one compression method."""

    method: str
    configuration: str
    seeds: tuple[int, ...]

    compression_ratio: CompressionMetricSummary
    storage_reduction_percent: CompressionMetricSummary
    parameter_reduction_percent: CompressionMetricSummary

    baseline_test_mse: CompressionMetricSummary
    compressed_test_mse: CompressionMetricSummary
    mse_change_percent: CompressionMetricSummary

    baseline_test_mae: CompressionMetricSummary
    compressed_test_mae: CompressionMetricSummary
    mae_change_percent: CompressionMetricSummary

    baseline_mean_latency_ms: CompressionMetricSummary
    compressed_mean_latency_ms: CompressionMetricSummary
    latency_change_percent: CompressionMetricSummary

    pilot_feasible_runs: int
    all_runs_pilot_feasible: bool


@dataclass(frozen=True)
class DomainCompressionValidation:
    """Three-seed compression evidence for one domain."""

    domain: str
    seeds: tuple[int, ...]

    minimum_compression_ratio: float
    maximum_mse_increase_percent: float

    int8: CompressionMethodValidation
    svd: CompressionMethodValidation
    tensor_network: CompressionMethodValidation


@dataclass(frozen=True)
class MultiSeedCompressionValidation:
    """Complete cross-domain compression validation."""

    seeds: tuple[int, ...]

    driving: DomainCompressionValidation
    robotics: DomainCompressionValidation


def summarize_metric(
    values: Sequence[float],
) -> CompressionMetricSummary:
    """Return mean and sample standard deviation for finite values."""
    if not values:
        raise ValueError("at least one value is required")

    converted = tuple(float(value) for value in values)

    if not all(math.isfinite(value) for value in converted):
        raise ValueError("all metric values must be finite")

    mean_value = statistics.mean(converted)

    std_value = statistics.stdev(converted) if len(converted) > 1 else 0.0

    return CompressionMetricSummary(
        mean=float(mean_value),
        std=float(std_value),
        values=converted,
    )


def _percent_change(
    baseline: float,
    compressed: float,
) -> float:
    """Return percentage change relative to a positive baseline."""
    if baseline <= 0.0:
        raise ValueError("baseline metric must be positive")

    return (compressed - baseline) / baseline * 100.0


def _load_payload(
    path: Path,
) -> dict[str, Any]:
    """Load one compression result payload."""
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_method_results(
    paths: Sequence[Path],
    *,
    expected_domain: str,
    expected_method: str,
    configuration: str,
    expected_seeds: Sequence[int],
    minimum_compression_ratio: float = 2.0,
    maximum_mse_increase_percent: float = 5.0,
) -> CompressionMethodValidation:
    """Aggregate paired per-seed compression result JSON files."""
    if not paths:
        raise ValueError("at least one result path is required")

    payloads = [_load_payload(path) for path in paths]

    seeds = tuple(int(payload["seed"]) for payload in payloads)

    expected_seed_tuple = tuple(int(seed) for seed in expected_seeds)

    if seeds != expected_seed_tuple:
        raise ValueError(
            f"unexpected seed order: {seeds}; " f"expected {expected_seed_tuple}"
        )

    for payload in payloads:
        if payload["domain"] != expected_domain:
            raise ValueError("unexpected domain")

        if payload["method"] != expected_method:
            raise ValueError("unexpected compression method")

    metrics = [payload["metrics"] for payload in payloads]

    compression_ratios = [
        float(metric["baseline_size_bytes"]) / float(metric["compressed_size_bytes"])
        for metric in metrics
    ]

    storage_reductions = [
        (
            1.0
            - (
                float(metric["compressed_size_bytes"])
                / float(metric["baseline_size_bytes"])
            )
        )
        * 100.0
        for metric in metrics
    ]

    parameter_reductions = [
        (
            1.0
            - (
                float(metric["compressed_parameters"])
                / float(metric["baseline_parameters"])
            )
        )
        * 100.0
        for metric in metrics
    ]

    baseline_mse = [float(metric["baseline_test_mse"]) for metric in metrics]

    compressed_mse = [float(metric["compressed_test_mse"]) for metric in metrics]

    mse_changes = [
        _percent_change(
            baseline,
            compressed,
        )
        for baseline, compressed in zip(
            baseline_mse,
            compressed_mse,
            strict=True,
        )
    ]

    baseline_mae = [float(metric["baseline_test_mae"]) for metric in metrics]

    compressed_mae = [float(metric["compressed_test_mae"]) for metric in metrics]

    mae_changes = [
        _percent_change(
            baseline,
            compressed,
        )
        for baseline, compressed in zip(
            baseline_mae,
            compressed_mae,
            strict=True,
        )
    ]

    baseline_latency = [float(metric["baseline_mean_latency_ms"]) for metric in metrics]

    compressed_latency = [
        float(metric["compressed_mean_latency_ms"]) for metric in metrics
    ]

    latency_changes = [
        _percent_change(
            baseline,
            compressed,
        )
        for baseline, compressed in zip(
            baseline_latency,
            compressed_latency,
            strict=True,
        )
    ]

    feasible = [
        (
            ratio >= minimum_compression_ratio
            and mse_change <= maximum_mse_increase_percent
        )
        for ratio, mse_change in zip(
            compression_ratios,
            mse_changes,
            strict=True,
        )
    ]

    return CompressionMethodValidation(
        method=expected_method,
        configuration=configuration,
        seeds=seeds,
        compression_ratio=(summarize_metric(compression_ratios)),
        storage_reduction_percent=(summarize_metric(storage_reductions)),
        parameter_reduction_percent=(summarize_metric(parameter_reductions)),
        baseline_test_mse=(summarize_metric(baseline_mse)),
        compressed_test_mse=(summarize_metric(compressed_mse)),
        mse_change_percent=(summarize_metric(mse_changes)),
        baseline_test_mae=(summarize_metric(baseline_mae)),
        compressed_test_mae=(summarize_metric(compressed_mae)),
        mae_change_percent=(summarize_metric(mae_changes)),
        baseline_mean_latency_ms=(summarize_metric(baseline_latency)),
        compressed_mean_latency_ms=(summarize_metric(compressed_latency)),
        latency_change_percent=(summarize_metric(latency_changes)),
        pilot_feasible_runs=sum(feasible),
        all_runs_pilot_feasible=all(feasible),
    )


def save_multiseed_compression_validation(
    validation: MultiSeedCompressionValidation,
    path: Path,
) -> None:
    """Save multi-seed compression validation evidence."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            asdict(validation),
            indent=2,
        ),
        encoding="utf-8",
    )
