"""Tests for compression experiment selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from q_vla_forge.evaluation.compression_selection import (
    CompressionCandidate,
    build_domain_compression_selection,
    load_compression_candidate,
    select_method_candidate,
)


def _candidate(
    *,
    experiment_id: str,
    method: str,
    ratio: float,
    mse_change: float,
    mae_change: float = 0.0,
) -> CompressionCandidate:
    """Create a normalized candidate for selection tests."""
    return CompressionCandidate(
        experiment_id=experiment_id,
        domain="autonomous_driving",
        method=method,
        seed=42,
        configuration_label=experiment_id,
        compression_ratio=ratio,
        storage_reduction_percent=(1.0 - 1.0 / ratio) * 100.0,
        parameter_reduction_percent=0.0,
        baseline_test_mse=1.0,
        compressed_test_mse=(1.0 + mse_change / 100.0),
        mse_change_percent=mse_change,
        baseline_test_mae=1.0,
        compressed_test_mae=(1.0 + mae_change / 100.0),
        mae_change_percent=mae_change,
        baseline_mean_latency_ms=1.0,
        compressed_mean_latency_ms=1.0,
        latency_change_percent=0.0,
        pilot_feasible=(ratio >= 2.0 and mse_change <= 5.0),
        source_file=(experiment_id + ".json"),
    )


def test_feasible_candidate_selected_over_infeasible() -> None:
    candidates = [
        _candidate(
            experiment_id="a",
            method="svd",
            ratio=3.0,
            mse_change=20.0,
        ),
        _candidate(
            experiment_id="b",
            method="svd",
            ratio=2.2,
            mse_change=4.0,
        ),
    ]

    result = select_method_candidate(
        "svd",
        candidates,
    )

    assert result.selected.experiment_id == "b"


def test_highest_compression_selected_among_feasible() -> None:
    candidates = [
        _candidate(
            experiment_id="a",
            method="svd",
            ratio=2.1,
            mse_change=1.0,
        ),
        _candidate(
            experiment_id="b",
            method="svd",
            ratio=2.8,
            mse_change=4.0,
        ),
    ]

    result = select_method_candidate(
        "svd",
        candidates,
    )

    assert result.selected.experiment_id == "b"


def test_lower_mse_breaks_equal_ratio_tie() -> None:
    candidates = [
        _candidate(
            experiment_id="a",
            method="tensor_train",
            ratio=2.5,
            mse_change=4.0,
        ),
        _candidate(
            experiment_id="b",
            method="tensor_train",
            ratio=2.5,
            mse_change=2.0,
        ),
    ]

    result = select_method_candidate(
        "tensor_train",
        candidates,
    )

    assert result.selected.experiment_id == "b"


def test_fallback_used_when_none_feasible() -> None:
    candidates = [
        _candidate(
            experiment_id="a",
            method="svd",
            ratio=1.8,
            mse_change=1.0,
        ),
        _candidate(
            experiment_id="b",
            method="svd",
            ratio=2.5,
            mse_change=50.0,
        ),
    ]

    result = select_method_candidate(
        "svd",
        candidates,
    )

    assert result.feasible_candidates == 0

    assert result.selected in candidates


def test_method_mismatch_rejected() -> None:
    candidates = [
        _candidate(
            experiment_id="a",
            method="svd",
            ratio=2.0,
            mse_change=1.0,
        )
    ]

    with pytest.raises(ValueError):
        select_method_candidate(
            "tensor_train",
            candidates,
        )


def test_empty_method_candidates_rejected() -> None:
    with pytest.raises(ValueError):
        select_method_candidate(
            "svd",
            [],
        )


def test_domain_selection_returns_three_methods() -> None:
    candidates = [
        _candidate(
            experiment_id="int8",
            method="int8",
            ratio=3.5,
            mse_change=1.0,
        ),
        _candidate(
            experiment_id="svd",
            method="svd",
            ratio=2.1,
            mse_change=2.0,
        ),
        _candidate(
            experiment_id="tt",
            method="tensor_train",
            ratio=2.3,
            mse_change=3.0,
        ),
    ]

    result = build_domain_compression_selection(
        candidates,
        domain="autonomous_driving",
        seed=42,
    )

    assert len(result.selected_experiment_ids) == 3


def test_domain_selection_rejects_missing_method() -> None:
    candidates = [
        _candidate(
            experiment_id="int8",
            method="int8",
            ratio=3.5,
            mse_change=1.0,
        )
    ]

    with pytest.raises(ValueError):
        build_domain_compression_selection(
            candidates,
            domain="autonomous_driving",
            seed=42,
        )


def test_load_real_shape_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experiment.json"

    payload = {
        "experiment_id": "driving-int8-seed-42",
        "domain": "autonomous_driving",
        "method": "int8",
        "seed": 42,
        "metrics": {
            "baseline_parameters": 1000,
            "compressed_parameters": 1000,
            "baseline_size_bytes": 4000,
            "compressed_size_bytes": 1100,
            "baseline_test_mse": 0.2,
            "compressed_test_mse": 0.204,
            "baseline_test_mae": 0.1,
            "compressed_test_mae": 0.101,
            "baseline_mean_latency_ms": 5.0,
            "compressed_mean_latency_ms": 5.2,
            "baseline_p95_latency_ms": 6.0,
            "compressed_p95_latency_ms": 6.1,
            "compression_time_seconds": 0.1,
        },
        "configuration": {"scheme": ("symmetric_per_tensor_weight_int8")},
    }

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    candidate = load_compression_candidate(path)

    assert candidate.method == "int8"

    assert candidate.configuration_label == "INT8"

    assert candidate.compression_ratio > 3.0


def test_negative_mse_change_is_allowed() -> None:
    candidate = _candidate(
        experiment_id="better",
        method="svd",
        ratio=2.1,
        mse_change=-2.0,
    )

    assert candidate.pilot_feasible


def test_candidate_source_file_retained(
    tmp_path: Path,
) -> None:
    path = tmp_path / "svd.json"

    payload = {
        "experiment_id": "svd-test",
        "domain": "autonomous_driving",
        "method": "svd",
        "seed": 42,
        "metrics": {
            "baseline_parameters": 1000,
            "compressed_parameters": 700,
            "baseline_size_bytes": 4000,
            "compressed_size_bytes": 2800,
            "baseline_test_mse": 1.0,
            "compressed_test_mse": 1.02,
            "baseline_test_mae": 1.0,
            "compressed_test_mae": 1.01,
            "baseline_mean_latency_ms": 1.0,
            "compressed_mean_latency_ms": 1.1,
            "baseline_p95_latency_ms": 1.2,
            "compressed_p95_latency_ms": 1.3,
            "compression_time_seconds": 0.1,
        },
        "configuration": {"rank_fraction": 0.5},
    }

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    candidate = load_compression_candidate(path)

    assert candidate.source_file == "svd.json"


def test_invalid_baseline_mse_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.json"

    payload = {
        "experiment_id": "bad",
        "domain": "autonomous_driving",
        "method": "int8",
        "seed": 42,
        "metrics": {
            "baseline_parameters": 100,
            "compressed_parameters": 100,
            "baseline_size_bytes": 400,
            "compressed_size_bytes": 100,
            "baseline_test_mse": 0.0,
            "compressed_test_mse": 0.1,
            "baseline_test_mae": 1.0,
            "compressed_test_mae": 1.0,
            "baseline_mean_latency_ms": 1.0,
            "compressed_mean_latency_ms": 1.0,
            "baseline_p95_latency_ms": 1.0,
            "compressed_p95_latency_ms": 1.0,
            "compression_time_seconds": 0.1,
        },
        "configuration": {},
    }

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_compression_candidate(path)


def test_robotics_domain_selection() -> None:
    candidates = [
        CompressionCandidate(
            experiment_id="robotics-int8",
            domain="robotics",
            method="int8",
            seed=42,
            configuration_label="INT8",
            compression_ratio=3.5,
            storage_reduction_percent=71.4,
            parameter_reduction_percent=0.0,
            baseline_test_mse=1.0,
            compressed_test_mse=1.01,
            mse_change_percent=1.0,
            baseline_test_mae=1.0,
            compressed_test_mae=1.01,
            mae_change_percent=1.0,
            baseline_mean_latency_ms=1.0,
            compressed_mean_latency_ms=1.0,
            latency_change_percent=0.0,
            pilot_feasible=True,
            source_file="robotics-int8.json",
        ),
        CompressionCandidate(
            experiment_id="robotics-svd",
            domain="robotics",
            method="svd",
            seed=42,
            configuration_label="SVD-50%",
            compression_ratio=2.1,
            storage_reduction_percent=52.4,
            parameter_reduction_percent=52.4,
            baseline_test_mse=1.0,
            compressed_test_mse=1.02,
            mse_change_percent=2.0,
            baseline_test_mae=1.0,
            compressed_test_mae=1.02,
            mae_change_percent=2.0,
            baseline_mean_latency_ms=1.0,
            compressed_mean_latency_ms=1.0,
            latency_change_percent=0.0,
            pilot_feasible=True,
            source_file="robotics-svd.json",
        ),
        CompressionCandidate(
            experiment_id="robotics-tt",
            domain="robotics",
            method="tensor_train",
            seed=42,
            configuration_label="TT-rank-4",
            compression_ratio=2.3,
            storage_reduction_percent=56.5,
            parameter_reduction_percent=56.5,
            baseline_test_mse=1.0,
            compressed_test_mse=1.03,
            mse_change_percent=3.0,
            baseline_test_mae=1.0,
            compressed_test_mae=1.03,
            mae_change_percent=3.0,
            baseline_mean_latency_ms=1.0,
            compressed_mean_latency_ms=1.0,
            latency_change_percent=0.0,
            pilot_feasible=True,
            source_file="robotics-tt.json",
        ),
    ]

    result = build_domain_compression_selection(
        candidates,
        domain="robotics",
        seed=42,
    )

    assert result.domain == "robotics"

    assert result.tensor_network.selected.configuration_label == "TT-rank-4"
