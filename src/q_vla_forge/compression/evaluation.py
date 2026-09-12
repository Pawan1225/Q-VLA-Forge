"""Helpers for building common Sprint 2 compression metrics."""

from __future__ import annotations

from q_vla_forge.compression.contracts import (
    CompressionMetrics,
)


def build_compression_metrics(
    *,
    baseline_parameters: int,
    compressed_parameters: int,
    baseline_size_bytes: int,
    compressed_size_bytes: int,
    baseline_test_mse: float,
    compressed_test_mse: float,
    baseline_test_mae: float,
    compressed_test_mae: float,
    baseline_mean_latency_ms: float,
    compressed_mean_latency_ms: float,
    baseline_p95_latency_ms: float,
    compressed_p95_latency_ms: float,
    compression_time_seconds: float,
) -> CompressionMetrics:
    """Build the shared compression metric contract."""
    return CompressionMetrics(
        baseline_parameters=(baseline_parameters),
        compressed_parameters=(compressed_parameters),
        baseline_size_bytes=(baseline_size_bytes),
        compressed_size_bytes=(compressed_size_bytes),
        baseline_test_mse=(baseline_test_mse),
        compressed_test_mse=(compressed_test_mse),
        baseline_test_mae=(baseline_test_mae),
        compressed_test_mae=(compressed_test_mae),
        baseline_mean_latency_ms=(baseline_mean_latency_ms),
        compressed_mean_latency_ms=(compressed_mean_latency_ms),
        baseline_p95_latency_ms=(baseline_p95_latency_ms),
        compressed_p95_latency_ms=(compressed_p95_latency_ms),
        compression_time_seconds=(compression_time_seconds),
    )
