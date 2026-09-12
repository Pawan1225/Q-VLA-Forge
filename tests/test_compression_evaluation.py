"""Tests for common compression evaluation helpers."""

from __future__ import annotations

import math

from q_vla_forge.compression import (
    build_compression_metrics,
)


def test_build_compression_metrics() -> None:
    metrics = build_compression_metrics(
        baseline_parameters=1000,
        compressed_parameters=1000,
        baseline_size_bytes=4000,
        compressed_size_bytes=1100,
        baseline_test_mse=0.20,
        compressed_test_mse=0.21,
        baseline_test_mae=0.10,
        compressed_test_mae=0.102,
        baseline_mean_latency_ms=10.0,
        compressed_mean_latency_ms=11.0,
        baseline_p95_latency_ms=12.0,
        compressed_p95_latency_ms=13.0,
        compression_time_seconds=0.5,
    )

    assert metrics.baseline_parameters == 1000
    assert metrics.compressed_parameters == 1000


def test_weight_quantization_has_zero_parameter_reduction() -> None:
    metrics = build_compression_metrics(
        baseline_parameters=1000,
        compressed_parameters=1000,
        baseline_size_bytes=4000,
        compressed_size_bytes=1000,
        baseline_test_mse=0.20,
        compressed_test_mse=0.20,
        baseline_test_mae=0.10,
        compressed_test_mae=0.10,
        baseline_mean_latency_ms=10.0,
        compressed_mean_latency_ms=10.0,
        baseline_p95_latency_ms=12.0,
        compressed_p95_latency_ms=12.0,
        compression_time_seconds=0.1,
    )

    assert metrics.parameter_reduction_percent == 0.0


def test_storage_compression_ratio() -> None:
    metrics = build_compression_metrics(
        baseline_parameters=1000,
        compressed_parameters=1000,
        baseline_size_bytes=4000,
        compressed_size_bytes=1000,
        baseline_test_mse=0.20,
        compressed_test_mse=0.20,
        baseline_test_mae=0.10,
        compressed_test_mae=0.10,
        baseline_mean_latency_ms=10.0,
        compressed_mean_latency_ms=10.0,
        baseline_p95_latency_ms=12.0,
        compressed_p95_latency_ms=12.0,
        compression_time_seconds=0.1,
    )

    assert math.isclose(
        metrics.compression_ratio,
        4.0,
    )


def test_positive_error_change_means_worse() -> None:
    metrics = build_compression_metrics(
        baseline_parameters=100,
        compressed_parameters=100,
        baseline_size_bytes=400,
        compressed_size_bytes=100,
        baseline_test_mse=0.20,
        compressed_test_mse=0.22,
        baseline_test_mae=0.10,
        compressed_test_mae=0.11,
        baseline_mean_latency_ms=1.0,
        compressed_mean_latency_ms=1.0,
        baseline_p95_latency_ms=1.0,
        compressed_p95_latency_ms=1.0,
        compression_time_seconds=0.1,
    )

    assert metrics.mse_change_percent > 0.0
    assert metrics.mae_change_percent > 0.0


def test_negative_error_change_means_improvement() -> None:
    metrics = build_compression_metrics(
        baseline_parameters=100,
        compressed_parameters=100,
        baseline_size_bytes=400,
        compressed_size_bytes=100,
        baseline_test_mse=0.20,
        compressed_test_mse=0.18,
        baseline_test_mae=0.10,
        compressed_test_mae=0.09,
        baseline_mean_latency_ms=1.0,
        compressed_mean_latency_ms=1.0,
        baseline_p95_latency_ms=1.0,
        compressed_p95_latency_ms=1.0,
        compression_time_seconds=0.1,
    )

    assert metrics.mse_change_percent < 0.0
    assert metrics.mae_change_percent < 0.0
