from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from q_vla_forge.evaluation.multiseed import (
    aggregate_baseline_payloads,
    save_multiseed_summary,
    summarize_values,
)

SEEDS = (
    42,
    123,
    456,
)


def _payload(
    *,
    domain: str,
    seed: int,
    test_mse: float,
    test_mae: float,
) -> dict[str, Any]:
    if domain == "autonomous_driving":
        action_metrics = {
            "steering_mae": 0.10,
            "acceleration_mae": 0.20,
            "braking_mae": 0.30,
        }
    else:
        action_metrics = {
            "delta_x_mae": 0.10,
            "delta_y_mae": 0.20,
            "gripper_mae": 0.30,
        }

    return {
        "experiment_id": f"{domain}-{seed}",
        "domain": domain,
        "method": "shared_vla_fp32",
        "seed": seed,
        "parameters": 1000,
        "fp32_model_size_bytes": 4000,
        "initial_validation_mse": 0.8,
        "best_validation_mse": 0.2,
        "final_validation_mse": 0.25,
        "test_metrics": {
            "test_mse": test_mse,
            "test_mae": test_mae,
            **action_metrics,
        },
        "latency": {
            "mean_ms": 10.0 + seed / 100.0,
            "p95_ms": 12.0 + seed / 100.0,
            "samples": 100,
        },
    }


def test_summarize_values() -> None:
    result = summarize_values(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    assert result.mean == 2.0
    assert result.std == 1.0

    assert result.values == (
        1.0,
        2.0,
        3.0,
    )


def test_single_value_has_zero_std() -> None:
    result = summarize_values([5.0])

    assert result.mean == 5.0
    assert result.std == 0.0


def test_summarize_rejects_empty_values() -> None:
    with pytest.raises(ValueError):
        summarize_values([])


def test_aggregate_three_seed_baselines() -> None:
    driving = [
        _payload(
            domain="autonomous_driving",
            seed=seed,
            test_mse=value,
            test_mae=value / 2.0,
        )
        for seed, value in zip(
            SEEDS,
            (
                0.30,
                0.20,
                0.10,
            ),
            strict=True,
        )
    ]

    robotics = [
        _payload(
            domain="robotics",
            seed=seed,
            test_mse=value,
            test_mae=value / 2.0,
        )
        for seed, value in zip(
            SEEDS,
            (
                0.60,
                0.50,
                0.40,
            ),
            strict=True,
        )
    ]

    result = aggregate_baseline_payloads(
        driving,
        robotics,
        SEEDS,
    )

    assert result.seeds == SEEDS

    assert math.isclose(
        result.driving.test_mse.mean,
        0.20,
    )

    assert math.isclose(
        result.robotics.test_mse.mean,
        0.50,
    )

    assert result.driving.parameters == 1000

    assert result.robotics.parameters == 1000


def test_action_metrics_are_aggregated() -> None:
    driving = [
        _payload(
            domain="autonomous_driving",
            seed=seed,
            test_mse=0.2,
            test_mae=0.1,
        )
        for seed in SEEDS
    ]

    robotics = [
        _payload(
            domain="robotics",
            seed=seed,
            test_mse=0.3,
            test_mae=0.2,
        )
        for seed in SEEDS
    ]

    result = aggregate_baseline_payloads(
        driving,
        robotics,
        SEEDS,
    )

    assert "steering_mae" in result.driving.action_mae

    assert "gripper_mae" in result.robotics.action_mae


def test_aggregate_rejects_wrong_seed_order() -> None:
    driving = [
        _payload(
            domain="autonomous_driving",
            seed=seed,
            test_mse=0.2,
            test_mae=0.1,
        )
        for seed in (
            123,
            42,
            456,
        )
    ]

    robotics = [
        _payload(
            domain="robotics",
            seed=seed,
            test_mse=0.3,
            test_mae=0.2,
        )
        for seed in SEEDS
    ]

    with pytest.raises(ValueError):
        aggregate_baseline_payloads(
            driving,
            robotics,
            SEEDS,
        )


def test_aggregate_rejects_domain_mismatch() -> None:
    driving = [
        _payload(
            domain="robotics",
            seed=seed,
            test_mse=0.2,
            test_mae=0.1,
        )
        for seed in SEEDS
    ]

    robotics = [
        _payload(
            domain="robotics",
            seed=seed,
            test_mse=0.3,
            test_mae=0.2,
        )
        for seed in SEEDS
    ]

    with pytest.raises(ValueError):
        aggregate_baseline_payloads(
            driving,
            robotics,
            SEEDS,
        )


def test_summary_saves_json(
    tmp_path: Path,
) -> None:
    driving = [
        _payload(
            domain="autonomous_driving",
            seed=seed,
            test_mse=0.2,
            test_mae=0.1,
        )
        for seed in SEEDS
    ]

    robotics = [
        _payload(
            domain="robotics",
            seed=seed,
            test_mse=0.3,
            test_mae=0.2,
        )
        for seed in SEEDS
    ]

    summary = aggregate_baseline_payloads(
        driving,
        robotics,
        SEEDS,
    )

    output = tmp_path / "summary.json"

    save_multiseed_summary(
        summary,
        output,
    )

    assert output.exists()

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["seeds"] == [
        42,
        123,
        456,
    ]

    assert payload["method"] == "shared_vla_fp32"
