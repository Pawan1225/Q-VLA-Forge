"""Run seed-42 INT8 compression experiments for both pilot domains."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from q_vla_forge.compression import (
    CompressionMethod,
    CompressionResult,
    Int8QuantizationReport,
    build_compression_metrics,
    load_seed_baseline,
    quantize_model_weights_int8,
)
from q_vla_forge.data import (
    Domain,
    SyntheticDrivingDataset,
    SyntheticRoboticsDataset,
    TaskSample,
)
from q_vla_forge.evaluation.driving_baseline import (
    evaluate_driving_model,
    measure_driving_latency,
)
from q_vla_forge.evaluation.robotics_baseline import (
    evaluate_robotics_model,
    measure_robotics_latency,
)
from q_vla_forge.models import SharedVLAModel
from q_vla_forge.training import (
    TrainingConfig,
    train_supervised,
)
from q_vla_forge.utils.reproducibility import set_seed

RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "compression"

SEED = 42

TRAIN_SIZE = 512
VALIDATION_SIZE = 128
TEST_SIZE = 128

EPOCHS = 20
BATCH_SIZE = 32


def _training_config() -> TrainingConfig:
    """Return the frozen Sprint 1 supervised-training configuration."""
    return TrainingConfig(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=1e-3,
        weight_decay=1e-4,
        min_learning_rate=1e-5,
        seed=SEED,
    )


def _train_driving() -> tuple[
    SharedVLAModel,
    list[TaskSample],
]:
    """Reproduce the Sprint 1 driving seed-42 model."""
    set_seed(SEED)

    train_samples = list(
        SyntheticDrivingDataset(
            size=TRAIN_SIZE,
            seed=SEED,
        )
    )

    validation_samples = list(
        SyntheticDrivingDataset(
            size=VALIDATION_SIZE,
            seed=SEED + 1000,
        )
    )

    test_samples = list(
        SyntheticDrivingDataset(
            size=TEST_SIZE,
            seed=SEED + 2000,
        )
    )

    model = SharedVLAModel()

    train_supervised(
        model=model,
        train_samples=train_samples,
        validation_samples=validation_samples,
        domain=Domain.AUTONOMOUS_DRIVING,
        config=_training_config(),
    )

    return model, test_samples


def _train_robotics() -> tuple[
    SharedVLAModel,
    list[TaskSample],
]:
    """Reproduce the Sprint 1 robotics seed-42 model."""
    set_seed(SEED)

    train_samples = list(
        SyntheticRoboticsDataset(
            size=TRAIN_SIZE,
            seed=SEED,
        )
    )

    validation_samples = list(
        SyntheticRoboticsDataset(
            size=VALIDATION_SIZE,
            seed=SEED + 1000,
        )
    )

    test_samples = list(
        SyntheticRoboticsDataset(
            size=TEST_SIZE,
            seed=SEED + 2000,
        )
    )

    model = SharedVLAModel()

    train_supervised(
        model=model,
        train_samples=train_samples,
        validation_samples=validation_samples,
        domain=Domain.ROBOTICS,
        config=_training_config(),
    )

    return model, test_samples


def _verify_reproduced_metric(
    *,
    label: str,
    frozen: float,
    reproduced: float,
) -> None:
    """Require the retrained FP32 reference to match Sprint 1."""
    if not math.isclose(
        frozen,
        reproduced,
        rel_tol=1e-7,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            f"{label} reproduction mismatch: "
            f"frozen={frozen}, reproduced={reproduced}"
        )


def _quantization_payload(
    report: Int8QuantizationReport,
) -> dict[str, Any]:
    """Serialize the INT8 storage report."""
    return {
        "total_parameters": report.total_parameters,
        "quantized_weight_parameters": (report.quantized_weight_parameters),
        "untouched_parameters": report.untouched_parameters,
        "baseline_size_bytes": report.baseline_size_bytes,
        "compressed_size_bytes": report.compressed_size_bytes,
        "compression_ratio": report.compression_ratio,
        "storage_reduction_percent": (report.storage_reduction_percent),
        "quantized_tensor_count": len(report.quantized_tensors),
        "quantized_tensors": [
            {
                "name": tensor.name,
                "shape": list(tensor.shape),
                "elements": tensor.elements,
                "original_bytes": tensor.original_bytes,
                "quantized_bytes": tensor.quantized_bytes,
                "scale": tensor.scale,
            }
            for tensor in report.quantized_tensors
        ],
    }


def _result_payload(
    result: CompressionResult,
    report: Int8QuantizationReport,
) -> dict[str, Any]:
    """Serialize a compression experiment result."""
    metrics = result.metrics

    return {
        "experiment_id": result.experiment_id,
        "domain": result.domain.value,
        "method": result.method.value,
        "seed": result.seed,
        "metrics": {
            "baseline_parameters": (metrics.baseline_parameters),
            "compressed_parameters": (metrics.compressed_parameters),
            "baseline_size_bytes": (metrics.baseline_size_bytes),
            "compressed_size_bytes": (metrics.compressed_size_bytes),
            "compression_ratio": (metrics.compression_ratio),
            "parameter_reduction_percent": (metrics.parameter_reduction_percent),
            "storage_reduction_percent": (metrics.storage_reduction_percent),
            "baseline_test_mse": (metrics.baseline_test_mse),
            "compressed_test_mse": (metrics.compressed_test_mse),
            "mse_change_percent": (metrics.mse_change_percent),
            "baseline_test_mae": (metrics.baseline_test_mae),
            "compressed_test_mae": (metrics.compressed_test_mae),
            "mae_change_percent": (metrics.mae_change_percent),
            "baseline_mean_latency_ms": (metrics.baseline_mean_latency_ms),
            "compressed_mean_latency_ms": (metrics.compressed_mean_latency_ms),
            "mean_latency_change_percent": (metrics.mean_latency_change_percent),
            "baseline_p95_latency_ms": (metrics.baseline_p95_latency_ms),
            "compressed_p95_latency_ms": (metrics.compressed_p95_latency_ms),
            "p95_latency_change_percent": (metrics.p95_latency_change_percent),
            "compression_time_seconds": (metrics.compression_time_seconds),
        },
        "configuration": result.configuration,
        "quantization": _quantization_payload(report),
        "notes": result.notes,
    }


def _save_result(
    result: CompressionResult,
    report: Int8QuantizationReport,
    path: Path,
) -> None:
    """Write one INT8 experiment evidence artifact."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            _result_payload(
                result,
                report,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


def run_driving() -> None:
    """Run seed-42 driving INT8 compression."""
    print()
    print("===== Driving INT8 =====")

    frozen = load_seed_baseline(
        RESULTS_DIR,
        Domain.AUTONOMOUS_DRIVING,
        SEED,
    )

    model, test_samples = _train_driving()

    reproduced = evaluate_driving_model(
        model=model,
        samples=test_samples,
        batch_size=BATCH_SIZE,
    )

    print(
        "Frozen FP32 MSE:",
        frozen.test_mse,
    )
    print(
        "Reproduced FP32 MSE:",
        reproduced.test_mse,
    )
    print(
        "Frozen FP32 MAE:",
        frozen.test_mae,
    )
    print(
        "Reproduced FP32 MAE:",
        reproduced.test_mae,
    )

    _verify_reproduced_metric(
        label="driving MSE",
        frozen=frozen.test_mse,
        reproduced=reproduced.test_mse,
    )
    _verify_reproduced_metric(
        label="driving MAE",
        frozen=frozen.test_mae,
        reproduced=reproduced.test_mae,
    )

    start = time.perf_counter()

    compressed_model, report = quantize_model_weights_int8(model)

    compression_seconds = time.perf_counter() - start

    compressed_quality = evaluate_driving_model(
        model=compressed_model,
        samples=test_samples,
        batch_size=BATCH_SIZE,
    )

    compressed_latency = measure_driving_latency(
        model=compressed_model,
        sample=test_samples[0],
    )

    metrics = build_compression_metrics(
        baseline_parameters=frozen.parameters,
        compressed_parameters=report.total_parameters,
        baseline_size_bytes=(frozen.fp32_model_size_bytes),
        compressed_size_bytes=(report.compressed_size_bytes),
        baseline_test_mse=frozen.test_mse,
        compressed_test_mse=(compressed_quality.test_mse),
        baseline_test_mae=frozen.test_mae,
        compressed_test_mae=(compressed_quality.test_mae),
        baseline_mean_latency_ms=(frozen.mean_latency_ms),
        compressed_mean_latency_ms=(compressed_latency.mean_ms),
        baseline_p95_latency_ms=(frozen.p95_latency_ms),
        compressed_p95_latency_ms=(compressed_latency.p95_ms),
        compression_time_seconds=(compression_seconds),
    )

    result = CompressionResult(
        experiment_id="driving-int8-seed-42",
        domain=Domain.AUTONOMOUS_DRIVING,
        method=CompressionMethod.INT8,
        seed=SEED,
        metrics=metrics,
        configuration={
            "scheme": "symmetric_per_tensor_weight_int8",
            "inference": "fp32_dequantized_weights",
            "quantized_module_types": [
                "Conv2d",
                "Embedding",
                "Linear",
            ],
            "bias_dtype": "fp32",
            "normalization_dtype": "fp32",
        },
        notes=(
            "Weight-storage INT8 simulation; inference "
            "uses FP32-dequantized weights."
        ),
    )

    output_path = OUTPUT_DIR / "driving-int8-seed-42.json"

    _save_result(
        result,
        report,
        output_path,
    )

    print(
        "Compression ratio:",
        f"{metrics.compression_ratio:.4f}x",
    )
    print(
        "Storage reduction:",
        f"{metrics.storage_reduction_percent:.4f}%",
    )
    print(
        "Compressed MSE:",
        compressed_quality.test_mse,
    )
    print(
        "MSE change:",
        f"{metrics.mse_change_percent:.4f}%",
    )
    print(
        "Compressed MAE:",
        compressed_quality.test_mae,
    )
    print(
        "MAE change:",
        f"{metrics.mae_change_percent:.4f}%",
    )
    print(
        "Dequantized mean latency:",
        f"{compressed_latency.mean_ms:.4f} ms",
    )
    print(
        "Saved:",
        output_path,
    )


def run_robotics() -> None:
    """Run seed-42 robotics INT8 compression."""
    print()
    print("===== Robotics INT8 =====")

    frozen = load_seed_baseline(
        RESULTS_DIR,
        Domain.ROBOTICS,
        SEED,
    )

    model, test_samples = _train_robotics()

    reproduced = evaluate_robotics_model(
        model=model,
        samples=test_samples,
        batch_size=BATCH_SIZE,
    )

    print(
        "Frozen FP32 MSE:",
        frozen.test_mse,
    )
    print(
        "Reproduced FP32 MSE:",
        reproduced.test_mse,
    )
    print(
        "Frozen FP32 MAE:",
        frozen.test_mae,
    )
    print(
        "Reproduced FP32 MAE:",
        reproduced.test_mae,
    )

    _verify_reproduced_metric(
        label="robotics MSE",
        frozen=frozen.test_mse,
        reproduced=reproduced.test_mse,
    )
    _verify_reproduced_metric(
        label="robotics MAE",
        frozen=frozen.test_mae,
        reproduced=reproduced.test_mae,
    )

    start = time.perf_counter()

    compressed_model, report = quantize_model_weights_int8(model)

    compression_seconds = time.perf_counter() - start

    compressed_quality = evaluate_robotics_model(
        model=compressed_model,
        samples=test_samples,
        batch_size=BATCH_SIZE,
    )

    compressed_latency = measure_robotics_latency(
        model=compressed_model,
        sample=test_samples[0],
    )

    metrics = build_compression_metrics(
        baseline_parameters=frozen.parameters,
        compressed_parameters=report.total_parameters,
        baseline_size_bytes=(frozen.fp32_model_size_bytes),
        compressed_size_bytes=(report.compressed_size_bytes),
        baseline_test_mse=frozen.test_mse,
        compressed_test_mse=(compressed_quality.test_mse),
        baseline_test_mae=frozen.test_mae,
        compressed_test_mae=(compressed_quality.test_mae),
        baseline_mean_latency_ms=(frozen.mean_latency_ms),
        compressed_mean_latency_ms=(compressed_latency.mean_ms),
        baseline_p95_latency_ms=(frozen.p95_latency_ms),
        compressed_p95_latency_ms=(compressed_latency.p95_ms),
        compression_time_seconds=(compression_seconds),
    )

    result = CompressionResult(
        experiment_id="robotics-int8-seed-42",
        domain=Domain.ROBOTICS,
        method=CompressionMethod.INT8,
        seed=SEED,
        metrics=metrics,
        configuration={
            "scheme": "symmetric_per_tensor_weight_int8",
            "inference": "fp32_dequantized_weights",
            "quantized_module_types": [
                "Conv2d",
                "Embedding",
                "Linear",
            ],
            "bias_dtype": "fp32",
            "normalization_dtype": "fp32",
        },
        notes=(
            "Weight-storage INT8 simulation; inference "
            "uses FP32-dequantized weights."
        ),
    )

    output_path = OUTPUT_DIR / "robotics-int8-seed-42.json"

    _save_result(
        result,
        report,
        output_path,
    )

    print(
        "Compression ratio:",
        f"{metrics.compression_ratio:.4f}x",
    )
    print(
        "Storage reduction:",
        f"{metrics.storage_reduction_percent:.4f}%",
    )
    print(
        "Compressed MSE:",
        compressed_quality.test_mse,
    )
    print(
        "MSE change:",
        f"{metrics.mse_change_percent:.4f}%",
    )
    print(
        "Compressed MAE:",
        compressed_quality.test_mae,
    )
    print(
        "MAE change:",
        f"{metrics.mae_change_percent:.4f}%",
    )
    print(
        "Dequantized mean latency:",
        f"{compressed_latency.mean_ms:.4f} ms",
    )
    print(
        "Saved:",
        output_path,
    )


def main() -> None:
    """Run the complete seed-42 INT8 baseline."""
    print()
    print("===== Q-VLA Forge " "Sprint 2.3 INT8 Baseline =====")

    run_driving()
    run_robotics()

    print()
    print("===== INT8 BASELINE COMPLETE =====")


if __name__ == "__main__":
    main()
