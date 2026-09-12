from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from q_vla_forge.data import (
    Domain,
    SyntheticDrivingDataset,
    TaskSample,
)
from q_vla_forge.models import SharedVLAModel
from q_vla_forge.training import (
    TrainingConfig,
    TrainingResult,
    build_tensor_batch,
    train_supervised,
)
from q_vla_forge.utils.reproducibility import set_seed


@dataclass(frozen=True)
class DrivingEvaluationMetrics:
    """Evaluation metrics for the driving baseline."""

    test_mse: float
    test_mae: float
    steering_mae: float
    acceleration_mae: float
    braking_mae: float


@dataclass(frozen=True)
class LatencyMetrics:
    """CPU inference latency statistics."""

    mean_ms: float
    p95_ms: float
    samples: int


@dataclass(frozen=True)
class DrivingBaselineResult:
    """Complete evidence record for one driving baseline run."""

    experiment_id: str
    domain: str
    method: str
    seed: int
    train_size: int
    validation_size: int
    test_size: int
    parameters: int
    fp32_model_size_bytes: int
    initial_validation_mse: float
    best_validation_mse: float
    final_validation_mse: float
    test_metrics: DrivingEvaluationMetrics
    latency: LatencyMetrics
    training_history: tuple[dict[str, float | int], ...]
    created_at: str


def evaluate_driving_model(
    model: SharedVLAModel,
    samples: list[TaskSample],
    batch_size: int = 64,
    device: torch.device | None = None,
) -> DrivingEvaluationMetrics:
    """Evaluate action prediction quality on driving samples."""
    if not samples:
        raise ValueError("samples must not be empty")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    if any(sample.domain != Domain.AUTONOMOUS_DRIVING for sample in samples):
        raise ValueError("all samples must belong to autonomous driving")

    active_device = device if device is not None else torch.device("cpu")

    model.to(active_device)
    model.eval()

    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(
            0,
            len(samples),
            batch_size,
        ):
            batch_samples = samples[start : start + batch_size]

            batch = build_tensor_batch(
                batch_samples,
                active_device,
            )

            prediction = model(
                batch.visual,
                batch.state,
                batch.language_goals,
                Domain.AUTONOMOUS_DRIVING,
            )

            predictions.append(prediction.cpu())

            targets.append(batch.targets.cpu())

    prediction_tensor = torch.cat(
        predictions,
        dim=0,
    )

    target_tensor = torch.cat(
        targets,
        dim=0,
    )

    error = prediction_tensor - target_tensor
    absolute_error = error.abs()

    mse = error.square().mean().item()
    mae = absolute_error.mean().item()

    per_action_mae = absolute_error.mean(dim=0)

    return DrivingEvaluationMetrics(
        test_mse=float(mse),
        test_mae=float(mae),
        steering_mae=float(per_action_mae[0].item()),
        acceleration_mae=float(per_action_mae[1].item()),
        braking_mae=float(per_action_mae[2].item()),
    )


def measure_driving_latency(
    model: SharedVLAModel,
    sample: TaskSample,
    warmup_runs: int = 10,
    measured_runs: int = 100,
    device: torch.device | None = None,
) -> LatencyMetrics:
    """Measure single-sample forward-pass latency."""
    if sample.domain != Domain.AUTONOMOUS_DRIVING:
        raise ValueError("sample must belong to autonomous driving")

    if warmup_runs < 0:
        raise ValueError("warmup_runs must not be negative")

    if measured_runs <= 0:
        raise ValueError("measured_runs must be greater than zero")

    active_device = device if device is not None else torch.device("cpu")

    model.to(active_device)
    model.eval()

    batch = build_tensor_batch(
        [sample],
        active_device,
    )

    with torch.no_grad():
        for _ in range(warmup_runs):
            model(
                batch.visual,
                batch.state,
                batch.language_goals,
                Domain.AUTONOMOUS_DRIVING,
            )

        durations_ms: list[float] = []

        for _ in range(measured_runs):
            start = time.perf_counter()

            model(
                batch.visual,
                batch.state,
                batch.language_goals,
                Domain.AUTONOMOUS_DRIVING,
            )

            duration_ms = (time.perf_counter() - start) * 1000.0

            durations_ms.append(duration_ms)

    sorted_durations = sorted(durations_ms)

    p95_index = min(
        len(sorted_durations) - 1,
        int(0.95 * len(sorted_durations)),
    )

    return LatencyMetrics(
        mean_ms=float(statistics.mean(durations_ms)),
        p95_ms=float(sorted_durations[p95_index]),
        samples=measured_runs,
    )


def _training_history(
    training_result: TrainingResult,
) -> tuple[
    dict[str, float | int],
    ...,
]:
    return tuple(
        {
            "epoch": item.epoch,
            "train_loss": item.train_loss,
            "validation_loss": item.validation_loss,
            "learning_rate": item.learning_rate,
        }
        for item in training_result.history
    )


def run_driving_baseline(
    *,
    seed: int = 42,
    train_size: int = 512,
    validation_size: int = 128,
    test_size: int = 128,
    epochs: int = 20,
    batch_size: int = 32,
    output_path: Path | None = None,
) -> DrivingBaselineResult:
    """Train, evaluate, and optionally save the driving baseline."""
    if train_size <= 0:
        raise ValueError("train_size must be greater than zero")

    if validation_size <= 0:
        raise ValueError("validation_size must be greater than zero")

    if test_size <= 0:
        raise ValueError("test_size must be greater than zero")

    set_seed(seed)

    train_dataset = SyntheticDrivingDataset(
        size=train_size,
        seed=seed,
    )

    validation_dataset = SyntheticDrivingDataset(
        size=validation_size,
        seed=seed + 1000,
    )

    test_dataset = SyntheticDrivingDataset(
        size=test_size,
        seed=seed + 2000,
    )

    train_samples = list(train_dataset)

    validation_samples = list(validation_dataset)

    test_samples = list(test_dataset)

    model = SharedVLAModel()

    training_result = train_supervised(
        model=model,
        train_samples=train_samples,
        validation_samples=validation_samples,
        domain=Domain.AUTONOMOUS_DRIVING,
        config=TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=1e-3,
            weight_decay=1e-4,
            min_learning_rate=1e-5,
            seed=seed,
        ),
    )

    test_metrics = evaluate_driving_model(
        model=model,
        samples=test_samples,
        batch_size=batch_size,
    )

    latency = measure_driving_latency(
        model=model,
        sample=test_samples[0],
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    result = DrivingBaselineResult(
        experiment_id=(f"driving-baseline-seed-{seed}"),
        domain=(Domain.AUTONOMOUS_DRIVING.value),
        method="shared_vla_fp32",
        seed=seed,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        parameters=parameter_count,
        fp32_model_size_bytes=(parameter_count * 4),
        initial_validation_mse=(training_result.initial_validation_loss),
        best_validation_mse=(training_result.best_validation_loss),
        final_validation_mse=(training_result.final_validation_loss),
        test_metrics=test_metrics,
        latency=latency,
        training_history=(_training_history(training_result)),
        created_at=(datetime.now(UTC).isoformat()),
    )

    if output_path is not None:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                asdict(result),
                indent=2,
            ),
            encoding="utf-8",
        )

    return result
