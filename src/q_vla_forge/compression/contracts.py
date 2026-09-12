"""Shared contracts for Sprint 2 compression experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from q_vla_forge.data import Domain


class CompressionMethod(str, Enum):
    """Supported Sprint 2 compression families."""

    INT8 = "int8"
    SVD = "svd"
    TENSOR_TRAIN = "tensor_train"
    MPS = "mps"


@dataclass(frozen=True)
class CompressionMetrics:
    """Standard metrics used to compare compression against FP32."""

    baseline_parameters: int
    compressed_parameters: int

    baseline_size_bytes: int
    compressed_size_bytes: int

    baseline_test_mse: float
    compressed_test_mse: float

    baseline_test_mae: float
    compressed_test_mae: float

    baseline_mean_latency_ms: float
    compressed_mean_latency_ms: float

    baseline_p95_latency_ms: float
    compressed_p95_latency_ms: float

    compression_time_seconds: float

    def __post_init__(self) -> None:
        positive_integer_fields = {
            "baseline_parameters": self.baseline_parameters,
            "compressed_parameters": self.compressed_parameters,
            "baseline_size_bytes": self.baseline_size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
        }

        for name, value in positive_integer_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")

        non_negative_float_fields = {
            "baseline_test_mse": self.baseline_test_mse,
            "compressed_test_mse": self.compressed_test_mse,
            "baseline_test_mae": self.baseline_test_mae,
            "compressed_test_mae": self.compressed_test_mae,
            "baseline_mean_latency_ms": self.baseline_mean_latency_ms,
            "compressed_mean_latency_ms": self.compressed_mean_latency_ms,
            "baseline_p95_latency_ms": self.baseline_p95_latency_ms,
            "compressed_p95_latency_ms": self.compressed_p95_latency_ms,
            "compression_time_seconds": self.compression_time_seconds,
        }

        for name, float_value in non_negative_float_fields.items():
            if not math.isfinite(float_value):
                raise ValueError(f"{name} must be finite")

            if float_value < 0.0:
                raise ValueError(f"{name} must not be negative")

    @property
    def compression_ratio(self) -> float:
        """Return storage compression ratio versus FP32."""
        return self.baseline_size_bytes / self.compressed_size_bytes

    @property
    def parameter_reduction_percent(self) -> float:
        """Return effective parameter reduction percentage."""
        return (1.0 - self.compressed_parameters / self.baseline_parameters) * 100.0

    @property
    def storage_reduction_percent(self) -> float:
        """Return stored-byte reduction percentage."""
        return (1.0 - self.compressed_size_bytes / self.baseline_size_bytes) * 100.0

    @staticmethod
    def _relative_change_percent(
        baseline: float,
        candidate: float,
    ) -> float:
        """Return relative percentage change from baseline."""
        if baseline == 0.0:
            return 0.0 if candidate == 0.0 else math.inf

        return ((candidate - baseline) / baseline) * 100.0

    @property
    def mse_change_percent(self) -> float:
        """Return relative MSE change versus FP32."""
        return self._relative_change_percent(
            self.baseline_test_mse,
            self.compressed_test_mse,
        )

    @property
    def mae_change_percent(self) -> float:
        """Return relative MAE change versus FP32."""
        return self._relative_change_percent(
            self.baseline_test_mae,
            self.compressed_test_mae,
        )

    @property
    def mean_latency_change_percent(self) -> float:
        """Return relative mean-latency change versus FP32."""
        return self._relative_change_percent(
            self.baseline_mean_latency_ms,
            self.compressed_mean_latency_ms,
        )

    @property
    def p95_latency_change_percent(self) -> float:
        """Return relative P95-latency change versus FP32."""
        return self._relative_change_percent(
            self.baseline_p95_latency_ms,
            self.compressed_p95_latency_ms,
        )


@dataclass(frozen=True)
class CompressionResult:
    """Standardized record for one compression experiment."""

    experiment_id: str
    domain: Domain
    method: CompressionMethod
    seed: int
    metrics: CompressionMetrics
    configuration: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty")

        if self.seed < 0:
            raise ValueError("seed must not be negative")