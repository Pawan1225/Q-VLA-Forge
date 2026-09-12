"""Tests for three-seed compression validation aggregation."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from q_vla_forge.evaluation.compression_validation import (
    aggregate_method_results,
    summarize_metric,
)


def test_summary_mean() -> None:
    summary = summarize_metric(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    assert summary.mean == pytest.approx(2.0)


def test_summary_uses_sample_standard_deviation() -> None:
    values = [
        1.0,
        2.0,
        4.0,
    ]

    summary = summarize_metric(values)

    assert summary.std == pytest.approx(statistics.stdev(values))


def test_single_value_has_zero_sd() -> None:
    summary = summarize_metric([3.0])

    assert summary.std == 0.0


def test_empty_metric_rejected() -> None:
    with pytest.raises(ValueError):
        summarize_metric([])


def _write_result(
    path: Path,
    *,
    seed: int,
    domain: str = "autonomous_driving",
    method: str = "int8",
    ratio: float = 4.0,
    mse_change_percent: float = 2.0,
) -> None:
    baseline_size = 4000

    compressed_size = int(baseline_size / ratio)

    baseline_mse = 1.0

    compressed_mse = baseline_mse * (1.0 + mse_change_percent / 100.0)

    payload = {
        "experiment_id": (f"test-{method}-{seed}"),
        "domain": domain,
        "method": method,
        "seed": seed,
        "metrics": {
            "baseline_parameters": 1000,
            "compressed_parameters": 500,
            "baseline_size_bytes": (baseline_size),
            "compressed_size_bytes": (compressed_size),
            "baseline_test_mse": (baseline_mse),
            "compressed_test_mse": (compressed_mse),
            "baseline_test_mae": 1.0,
            "compressed_test_mae": 1.01,
            "baseline_mean_latency_ms": 5.0,
            "compressed_mean_latency_ms": 5.1,
            "baseline_p95_latency_ms": 6.0,
            "compressed_p95_latency_ms": 6.1,
            "compression_time_seconds": 0.1,
        },
        "configuration": {},
    }

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_aggregate_three_seeds(
    tmp_path: Path,
) -> None:
    paths = []

    for seed in (
        42,
        123,
        456,
    ):
        path = tmp_path / f"result-{seed}.json"

        _write_result(
            path,
            seed=seed,
        )

        paths.append(path)

    result = aggregate_method_results(
        paths,
        expected_domain=("autonomous_driving"),
        expected_method="int8",
        configuration="INT8",
        expected_seeds=(
            42,
            123,
            456,
        ),
    )

    assert result.seeds == (
        42,
        123,
        456,
    )

    assert result.compression_ratio.mean == pytest.approx(4.0)


def test_all_runs_feasible(
    tmp_path: Path,
) -> None:
    paths = []

    for seed in (
        42,
        123,
        456,
    ):
        path = tmp_path / f"result-{seed}.json"

        _write_result(
            path,
            seed=seed,
            ratio=2.5,
            mse_change_percent=4.0,
        )

        paths.append(path)

    result = aggregate_method_results(
        paths,
        expected_domain=("autonomous_driving"),
        expected_method="int8",
        configuration="INT8",
        expected_seeds=(
            42,
            123,
            456,
        ),
    )

    assert result.pilot_feasible_runs == 3

    assert result.all_runs_pilot_feasible


def test_partial_feasibility(
    tmp_path: Path,
) -> None:
    paths = []

    changes = {
        42: 2.0,
        123: 4.0,
        456: 8.0,
    }

    for seed in (
        42,
        123,
        456,
    ):
        path = tmp_path / f"result-{seed}.json"

        _write_result(
            path,
            seed=seed,
            ratio=2.5,
            mse_change_percent=(changes[seed]),
        )

        paths.append(path)

    result = aggregate_method_results(
        paths,
        expected_domain=("autonomous_driving"),
        expected_method="int8",
        configuration="INT8",
        expected_seeds=(
            42,
            123,
            456,
        ),
    )

    assert result.pilot_feasible_runs == 2

    assert not (result.all_runs_pilot_feasible)


def test_wrong_seed_order_rejected(
    tmp_path: Path,
) -> None:
    paths = []

    for seed in (
        123,
        42,
        456,
    ):
        path = tmp_path / f"result-{seed}.json"

        _write_result(
            path,
            seed=seed,
        )

        paths.append(path)

    with pytest.raises(ValueError):
        aggregate_method_results(
            paths,
            expected_domain=("autonomous_driving"),
            expected_method="int8",
            configuration="INT8",
            expected_seeds=(
                42,
                123,
                456,
            ),
        )


def test_wrong_domain_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.json"

    _write_result(
        path,
        seed=42,
        domain="robotics",
    )

    with pytest.raises(ValueError):
        aggregate_method_results(
            [path],
            expected_domain=("autonomous_driving"),
            expected_method="int8",
            configuration="INT8",
            expected_seeds=(42,),
        )


def test_wrong_method_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.json"

    _write_result(
        path,
        seed=42,
        method="svd",
    )

    with pytest.raises(ValueError):
        aggregate_method_results(
            [path],
            expected_domain=("autonomous_driving"),
            expected_method="int8",
            configuration="INT8",
            expected_seeds=(42,),
        )
