"""Tests for frozen Sprint 1 compression references."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from q_vla_forge.compression import (
    load_frozen_baseline,
    load_seed_baseline,
)
from q_vla_forge.data import Domain


def _write_json(
    path: Path,
    payload: dict[str, object],
) -> None:
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _prepare_baseline(
    results_dir: Path,
) -> None:
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_json(
        results_dir / "sprint1-baseline-manifest.json",
        {
            "project": "Q-VLA Forge",
            "baseline_name": "shared_vla_fp32",
            "baseline_version": "1.0",
            "seeds": [
                42,
                123,
                456,
            ],
            "parameters": 1000,
            "fp32_model_size_bytes": 4000,
        },
    )

    _write_json(
        results_dir / "baseline-validation-summary.json",
        {
            "method": "shared_vla_fp32",
            "seeds": [
                42,
                123,
                456,
            ],
            "driving": {
                "domain": "autonomous_driving",
                "parameters": 1000,
                "fp32_model_size_bytes": 4000,
                "test_mse": {
                    "mean": 0.20,
                    "std": 0.01,
                },
                "test_mae": {
                    "mean": 0.10,
                    "std": 0.01,
                },
                "mean_latency_ms": {
                    "mean": 10.0,
                    "std": 1.0,
                },
                "p95_latency_ms": {
                    "mean": 12.0,
                    "std": 1.0,
                },
            },
            "robotics": {
                "domain": "robotics",
                "parameters": 1000,
                "fp32_model_size_bytes": 4000,
                "test_mse": {
                    "mean": 0.30,
                    "std": 0.02,
                },
                "test_mae": {
                    "mean": 0.15,
                    "std": 0.01,
                },
                "mean_latency_ms": {
                    "mean": 11.0,
                    "std": 1.0,
                },
                "p95_latency_ms": {
                    "mean": 13.0,
                    "std": 1.0,
                },
            },
        },
    )

    for seed in (
        42,
        123,
        456,
    ):
        _write_json(
            results_dir / f"driving-baseline-seed-{seed}.json",
            {
                "experiment_id": (f"driving-baseline-seed-{seed}"),
                "domain": "autonomous_driving",
                "method": "shared_vla_fp32",
                "seed": seed,
                "parameters": 1000,
                "fp32_model_size_bytes": 4000,
                "test_metrics": {
                    "test_mse": 0.20,
                    "test_mae": 0.10,
                },
                "latency": {
                    "mean_ms": 10.0,
                    "p95_ms": 12.0,
                },
            },
        )

        _write_json(
            results_dir / f"robotics-baseline-seed-{seed}.json",
            {
                "experiment_id": (f"robotics-baseline-seed-{seed}"),
                "domain": "robotics",
                "method": "shared_vla_fp32",
                "seed": seed,
                "parameters": 1000,
                "fp32_model_size_bytes": 4000,
                "test_metrics": {
                    "test_mse": 0.30,
                    "test_mae": 0.15,
                },
                "latency": {
                    "mean_ms": 11.0,
                    "p95_ms": 13.0,
                },
            },
        )


def test_load_frozen_baseline(
    tmp_path: Path,
) -> None:
    _prepare_baseline(tmp_path)

    baseline = load_frozen_baseline(tmp_path)

    assert baseline.baseline_name == "shared_vla_fp32"
    assert baseline.baseline_version == "1.0"
    assert baseline.seeds == (
        42,
        123,
        456,
    )
    assert baseline.parameters == 1000
    assert baseline.fp32_model_size_bytes == 4000
    assert baseline.driving_test_mse_mean == 0.20
    assert baseline.robotics_test_mse_mean == 0.30


def test_load_driving_seed_baseline(
    tmp_path: Path,
) -> None:
    _prepare_baseline(tmp_path)

    baseline = load_seed_baseline(
        tmp_path,
        Domain.AUTONOMOUS_DRIVING,
        42,
    )

    assert baseline.domain == Domain.AUTONOMOUS_DRIVING
    assert baseline.seed == 42
    assert baseline.parameters == 1000
    assert baseline.test_mse == 0.20
    assert baseline.test_mae == 0.10
    assert baseline.mean_latency_ms == 10.0
    assert baseline.p95_latency_ms == 12.0


def test_load_robotics_seed_baseline(
    tmp_path: Path,
) -> None:
    _prepare_baseline(tmp_path)

    baseline = load_seed_baseline(
        tmp_path,
        Domain.ROBOTICS,
        123,
    )

    assert baseline.domain == Domain.ROBOTICS
    assert baseline.seed == 123
    assert baseline.test_mse == 0.30
    assert baseline.test_mae == 0.15
    assert baseline.mean_latency_ms == 11.0
    assert baseline.p95_latency_ms == 13.0


def test_rejects_unsupported_seed(
    tmp_path: Path,
) -> None:
    _prepare_baseline(tmp_path)

    with pytest.raises(
        ValueError,
        match="unsupported baseline seed",
    ):
        load_seed_baseline(
            tmp_path,
            Domain.ROBOTICS,
            999,
        )


def test_rejects_manifest_seed_mismatch(
    tmp_path: Path,
) -> None:
    _prepare_baseline(tmp_path)

    manifest_path = tmp_path / "sprint1-baseline-manifest.json"

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["seeds"] = [
        42,
        123,
    ]

    _write_json(
        manifest_path,
        payload,
    )

    with pytest.raises(
        ValueError,
        match="manifest seeds",
    ):
        load_frozen_baseline(tmp_path)


def test_rejects_summary_method_mismatch(
    tmp_path: Path,
) -> None:
    _prepare_baseline(tmp_path)

    summary_path = tmp_path / "baseline-validation-summary.json"

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["method"] = "wrong_method"

    _write_json(
        summary_path,
        payload,
    )

    with pytest.raises(
        ValueError,
        match="summary method",
    ):
        load_frozen_baseline(tmp_path)


def test_rejects_cross_domain_parameter_mismatch(
    tmp_path: Path,
) -> None:
    _prepare_baseline(tmp_path)

    summary_path = tmp_path / "baseline-validation-summary.json"

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["robotics"]["parameters"] = 999

    _write_json(
        summary_path,
        payload,
    )

    with pytest.raises(
        ValueError,
        match="parameter counts",
    ):
        load_frozen_baseline(tmp_path)


def test_rejects_manifest_parameter_mismatch(
    tmp_path: Path,
) -> None:
    _prepare_baseline(tmp_path)

    manifest_path = tmp_path / "sprint1-baseline-manifest.json"

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["parameters"] = 999

    _write_json(
        manifest_path,
        payload,
    )

    with pytest.raises(
        ValueError,
        match="parameter counts differ",
    ):
        load_frozen_baseline(tmp_path)


def test_rejects_seed_payload_domain_mismatch(
    tmp_path: Path,
) -> None:
    _prepare_baseline(tmp_path)

    path = tmp_path / "driving-baseline-seed-42.json"

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["domain"] = "robotics"

    _write_json(
        path,
        payload,
    )

    with pytest.raises(
        ValueError,
        match="domain mismatch",
    ):
        load_seed_baseline(
            tmp_path,
            Domain.AUTONOMOUS_DRIVING,
            42,
        )


def test_rejects_seed_payload_method_mismatch(
    tmp_path: Path,
) -> None:
    _prepare_baseline(tmp_path)

    path = tmp_path / "robotics-baseline-seed-42.json"

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["method"] = "int8"

    _write_json(
        path,
        payload,
    )

    with pytest.raises(
        ValueError,
        match="unexpected baseline method",
    ):
        load_seed_baseline(
            tmp_path,
            Domain.ROBOTICS,
            42,
        )


def test_missing_manifest_rejected(
    tmp_path: Path,
) -> None:
    _prepare_baseline(tmp_path)

    (tmp_path / "sprint1-baseline-manifest.json").unlink()

    with pytest.raises(
        FileNotFoundError,
        match="required baseline file",
    ):
        load_frozen_baseline(tmp_path)
