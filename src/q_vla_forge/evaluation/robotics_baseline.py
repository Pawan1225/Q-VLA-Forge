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
    SyntheticRoboticsDataset,
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
class RoboticsEvaluationMetrics:
    """Evaluation metrics for the robotics baseline."""

    test_mse: float
    test_mae: float
    delta_x_mae: float
    delta_y_mae: float
    gripper_mae: float


@dataclass(frozen=True)
class RoboticsLatencyMetrics:
    """CPU inference latency statistics for robotics."""

    mean_ms: float
    p95_ms: float
    samples: int


@dataclass(frozen=True)
class RoboticsBaselineResult:
    """Complete evidence record for one robotics baseline run."""

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
    test_metrics: RoboticsEvaluationMetrics
    latency: RoboticsLatencyMetrics
    training_history: tuple[dict[str, float | int], ...]
    created_at: str


def evaluate_robotics_model(
    model: SharedVLAModel,
    samples: list[TaskSample],
    batch_size: int = 64,
    device: torch.device | None = None,
) -> RoboticsEvaluationMetrics:
    """Evaluate action prediction quality on robotics samples."""
    if not samples:
        raise ValueError("samples must not be empty")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    if any(sample.domain != Domain.ROBOTICS for sample in samples):
        raise ValueError("all samples must belong to robotics")

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
                Domain.ROBOTICS,
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

    return RoboticsEvaluationMetrics(
        test_mse=float(mse),
        test_mae=float(mae),
        delta_x_mae=float(per_action_mae[0].item()),
        delta_y_mae=float(per_action_mae[1].item()),
        gripper_mae=float(per_action_mae[2].item()),
    )


def measure_robotics_latency(
    model: SharedVLAModel,
    sample: TaskSample,
    warmup_runs: int = 10,
    measured_runs: int = 100,
    device: torch.device | None = None,
) -> RoboticsLatencyMetrics:
    """Measure single-sample robotics inference latency."""
    if sample.domain != Domain.ROBOTICS:
        raise ValueError("sample must belong to robotics")

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
                Domain.ROBOTICS,
            )

        durations_ms: list[float] = []

        for _ in range(measured_runs):
            start = time.perf_counter()

            model(
                batch.visual,
                batch.state,
                batch.language_goals,
                Domain.ROBOTICS,
            )

            duration_ms = (time.perf_counter() - start) * 1000.0

            durations_ms.append(duration_ms)

    sorted_durations = sorted(durations_ms)

    p95_index = min(
        len(sorted_durations) - 1,
        int(0.95 * len(sorted_durations)),
    )

    return RoboticsLatencyMetrics(
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
    """Convert training history to JSON-ready dictionaries."""
    return tuple(
        {
            "epoch": item.epoch,
            "train_loss": item.train_loss,
            "validation_loss": item.validation_loss,
            "learning_rate": item.learning_rate,
        }
        for item in training_result.history
    )


def run_robotics_baseline(
    *,
    seed: int = 42,
    train_size: int = 512,
    validation_size: int = 128,
    test_size: int = 128,
    epochs: int = 20,
    batch_size: int = 32,
    output_path: Path | None = None,
) -> RoboticsBaselineResult:
    """Train, evaluate, and optionally save the robotics baseline."""
    if train_size <= 0:
        raise ValueError("train_size must be greater than zero")

    if validation_size <= 0:
        raise ValueError("validation_size must be greater than zero")

    if test_size <= 0:
        raise ValueError("test_size must be greater than zero")

    set_seed(seed)

    train_dataset = SyntheticRoboticsDataset(
        size=train_size,
        seed=seed,
    )

    validation_dataset = SyntheticRoboticsDataset(
        size=validation_size,
        seed=seed + 1000,
    )

    test_dataset = SyntheticRoboticsDataset(
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
        domain=Domain.ROBOTICS,
        config=TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=1e-3,
            weight_decay=1e-4,
            min_learning_rate=1e-5,
            seed=seed,
        ),
    )

    test_metrics = evaluate_robotics_model(
        model=model,
        samples=test_samples,
        batch_size=batch_size,
    )

    latency = measure_robotics_latency(
        model=model,
        sample=test_samples[0],
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    result = RoboticsBaselineResult(
        experiment_id=(f"robotics-baseline-seed-{seed}"),
        domain=Domain.ROBOTICS.value,
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
