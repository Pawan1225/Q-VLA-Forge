from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from q_vla_forge.data import (
    Domain,
    SyntheticDrivingDataset,
    SyntheticRoboticsDataset,
)
from q_vla_forge.evaluation.driving_baseline import (
    evaluate_driving_model,
    measure_driving_latency,
    run_driving_baseline,
)
from q_vla_forge.models import SharedVLAModel
from q_vla_forge.utils.reproducibility import set_seed


def test_driving_evaluation_returns_finite_metrics() -> None:
    set_seed(42)

    model = SharedVLAModel()

    dataset = SyntheticDrivingDataset(
        size=8,
        seed=42,
    )

    result = evaluate_driving_model(
        model,
        list(dataset),
        batch_size=4,
    )

    assert math.isfinite(result.test_mse)
    assert math.isfinite(result.test_mae)
    assert math.isfinite(result.steering_mae)
    assert math.isfinite(result.acceleration_mae)
    assert math.isfinite(result.braking_mae)


def test_driving_evaluation_rejects_robotics() -> None:
    model = SharedVLAModel()

    dataset = SyntheticRoboticsDataset(
        size=4,
        seed=42,
    )

    with pytest.raises(ValueError):
        evaluate_driving_model(
            model,
            list(dataset),
        )


def test_latency_returns_positive_values() -> None:
    set_seed(42)

    model = SharedVLAModel()

    sample = SyntheticDrivingDataset(
        size=1,
        seed=42,
    )[0]

    result = measure_driving_latency(
        model=model,
        sample=sample,
        warmup_runs=1,
        measured_runs=3,
    )

    assert result.mean_ms > 0.0
    assert result.p95_ms > 0.0
    assert result.samples == 3


def test_latency_rejects_robotics_sample() -> None:
    model = SharedVLAModel()

    sample = SyntheticRoboticsDataset(
        size=1,
        seed=42,
    )[0]

    with pytest.raises(ValueError):
        measure_driving_latency(
            model=model,
            sample=sample,
        )


def test_baseline_creates_result() -> None:
    result = run_driving_baseline(
        seed=42,
        train_size=16,
        validation_size=8,
        test_size=8,
        epochs=1,
        batch_size=8,
    )

    assert result.seed == 42
    assert result.domain == Domain.AUTONOMOUS_DRIVING.value
    assert result.method == "shared_vla_fp32"
    assert result.parameters > 0
    assert result.fp32_model_size_bytes == result.parameters * 4
    assert math.isfinite(result.test_metrics.test_mse)


def test_baseline_saves_json(
    tmp_path: Path,
) -> None:
    output = tmp_path / "baseline.json"

    result = run_driving_baseline(
        seed=42,
        train_size=16,
        validation_size=8,
        test_size=8,
        epochs=1,
        batch_size=8,
        output_path=output,
    )

    assert output.exists()

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["experiment_id"] == result.experiment_id
    assert payload["seed"] == 42
    assert payload["domain"] == "autonomous_driving"


def test_baseline_rejects_invalid_train_size() -> None:
    with pytest.raises(ValueError):
        run_driving_baseline(
            train_size=0,
        )


def test_baseline_rejects_invalid_validation_size() -> None:
    with pytest.raises(ValueError):
        run_driving_baseline(
            validation_size=0,
        )


def test_baseline_rejects_invalid_test_size() -> None:
    with pytest.raises(ValueError):
        run_driving_baseline(
            test_size=0,
        )
