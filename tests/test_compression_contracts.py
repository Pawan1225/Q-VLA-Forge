"""Tests for shared compression contracts."""

from __future__ import annotations

import math

import pytest

from q_vla_forge.compression import (
    CompressionMethod,
    CompressionMetrics,
    CompressionResult,
)
from q_vla_forge.data import Domain


def _metrics() -> CompressionMetrics:
    return CompressionMetrics(
        baseline_parameters=1000,
        compressed_parameters=500,
        baseline_size_bytes=4000,
        compressed_size_bytes=2000,
        baseline_test_mse=0.20,
        compressed_test_mse=0.21,
        baseline_test_mae=0.10,
        compressed_test_mae=0.105,
        baseline_mean_latency_ms=10.0,
        compressed_mean_latency_ms=8.0,
        baseline_p95_latency_ms=12.0,
        compressed_p95_latency_ms=9.0,
        compression_time_seconds=1.5,
    )


def test_compression_ratio() -> None:
    metrics = _metrics()

    assert metrics.compression_ratio == 2.0


def test_parameter_reduction_percent() -> None:
    metrics = _metrics()

    assert metrics.parameter_reduction_percent == 50.0


def test_storage_reduction_percent() -> None:
    metrics = _metrics()

    assert metrics.storage_reduction_percent == 50.0


def test_mse_change_percent() -> None:
    metrics = _metrics()

    assert math.isclose(
        metrics.mse_change_percent,
        5.0,
    )


def test_mae_change_percent() -> None:
    metrics = _metrics()

    assert math.isclose(
        metrics.mae_change_percent,
        5.0,
    )


def test_mean_latency_change_percent() -> None:
    metrics = _metrics()

    assert math.isclose(
        metrics.mean_latency_change_percent,
        -20.0,
    )


def test_p95_latency_change_percent() -> None:
    metrics = _metrics()

    assert math.isclose(
        metrics.p95_latency_change_percent,
        -25.0,
    )


def test_compression_result() -> None:
    result = CompressionResult(
        experiment_id="driving-int8-seed-42",
        domain=Domain.AUTONOMOUS_DRIVING,
        method=CompressionMethod.INT8,
        seed=42,
        metrics=_metrics(),
        configuration={
            "dtype": "int8",
        },
    )

    assert result.domain == Domain.AUTONOMOUS_DRIVING
    assert result.method == CompressionMethod.INT8
    assert result.seed == 42


def test_all_compression_methods_exist() -> None:
    assert CompressionMethod.INT8.value == "int8"
    assert CompressionMethod.SVD.value == "svd"
    assert CompressionMethod.TENSOR_TRAIN.value == "tensor_train"
    assert CompressionMethod.MPS.value == "mps"


def test_zero_baseline_error_and_zero_candidate() -> None:
    metrics = CompressionMetrics(
        baseline_parameters=10,
        compressed_parameters=5,
        baseline_size_bytes=40,
        compressed_size_bytes=20,
        baseline_test_mse=0.0,
        compressed_test_mse=0.0,
        baseline_test_mae=0.0,
        compressed_test_mae=0.0,
        baseline_mean_latency_ms=1.0,
        compressed_mean_latency_ms=1.0,
        baseline_p95_latency_ms=1.0,
        compressed_p95_latency_ms=1.0,
        compression_time_seconds=0.0,
    )

    assert metrics.mse_change_percent == 0.0
    assert metrics.mae_change_percent == 0.0


def test_zero_baseline_error_and_nonzero_candidate() -> None:
    metrics = CompressionMetrics(
        baseline_parameters=10,
        compressed_parameters=5,
        baseline_size_bytes=40,
        compressed_size_bytes=20,
        baseline_test_mse=0.0,
        compressed_test_mse=0.1,
        baseline_test_mae=0.0,
        compressed_test_mae=0.1,
        baseline_mean_latency_ms=1.0,
        compressed_mean_latency_ms=1.0,
        baseline_p95_latency_ms=1.0,
        compressed_p95_latency_ms=1.0,
        compression_time_seconds=0.0,
    )

    assert math.isinf(metrics.mse_change_percent)
    assert math.isinf(metrics.mae_change_percent)


def test_rejects_zero_baseline_parameters() -> None:
    with pytest.raises(
        ValueError,
        match="baseline_parameters",
    ):
        CompressionMetrics(
            baseline_parameters=0,
            compressed_parameters=5,
            baseline_size_bytes=40,
            compressed_size_bytes=20,
            baseline_test_mse=0.1,
            compressed_test_mse=0.1,
            baseline_test_mae=0.1,
            compressed_test_mae=0.1,
            baseline_mean_latency_ms=1.0,
            compressed_mean_latency_ms=1.0,
            baseline_p95_latency_ms=1.0,
            compressed_p95_latency_ms=1.0,
            compression_time_seconds=0.0,
        )


def test_rejects_zero_compressed_size() -> None:
    with pytest.raises(
        ValueError,
        match="compressed_size_bytes",
    ):
        CompressionMetrics(
            baseline_parameters=10,
            compressed_parameters=5,
            baseline_size_bytes=40,
            compressed_size_bytes=0,
            baseline_test_mse=0.1,
            compressed_test_mse=0.1,
            baseline_test_mae=0.1,
            compressed_test_mae=0.1,
            baseline_mean_latency_ms=1.0,
            compressed_mean_latency_ms=1.0,
            baseline_p95_latency_ms=1.0,
            compressed_p95_latency_ms=1.0,
            compression_time_seconds=0.0,
        )


def test_rejects_negative_compression_time() -> None:
    with pytest.raises(
        ValueError,
        match="compression_time_seconds",
    ):
        CompressionMetrics(
            baseline_parameters=10,
            compressed_parameters=5,
            baseline_size_bytes=40,
            compressed_size_bytes=20,
            baseline_test_mse=0.1,
            compressed_test_mse=0.1,
            baseline_test_mae=0.1,
            compressed_test_mae=0.1,
            baseline_mean_latency_ms=1.0,
            compressed_mean_latency_ms=1.0,
            baseline_p95_latency_ms=1.0,
            compressed_p95_latency_ms=1.0,
            compression_time_seconds=-1.0,
        )


def test_rejects_non_finite_metric() -> None:
    with pytest.raises(
        ValueError,
        match="compressed_test_mse",
    ):
        CompressionMetrics(
            baseline_parameters=10,
            compressed_parameters=5,
            baseline_size_bytes=40,
            compressed_size_bytes=20,
            baseline_test_mse=0.1,
            compressed_test_mse=math.inf,
            baseline_test_mae=0.1,
            compressed_test_mae=0.1,
            baseline_mean_latency_ms=1.0,
            compressed_mean_latency_ms=1.0,
            baseline_p95_latency_ms=1.0,
            compressed_p95_latency_ms=1.0,
            compression_time_seconds=0.0,
        )


def test_rejects_empty_experiment_id() -> None:
    with pytest.raises(
        ValueError,
        match="experiment_id",
    ):
        CompressionResult(
            experiment_id="",
            domain=Domain.ROBOTICS,
            method=CompressionMethod.SVD,
            seed=42,
            metrics=_metrics(),
        )


def test_rejects_negative_seed() -> None:
    with pytest.raises(
        ValueError,
        match="seed",
    ):
        CompressionResult(
            experiment_id="robotics-svd",
            domain=Domain.ROBOTICS,
            method=CompressionMethod.SVD,
            seed=-1,
            metrics=_metrics(),
        )
