"""Run seed-42 SVD compression experiments for both pilot domains."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from q_vla_forge.compression import (
    CompressionMethod,
    CompressionResult,
    SVDCompressionReport,
    build_compression_metrics,
    compress_model_svd,
    load_seed_baseline,
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

RANK_FRACTIONS = (
    0.25,
    0.50,
    0.75,
)


def _training_config() -> TrainingConfig:
    """Return the frozen Sprint 1 training configuration."""
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


def _svd_report_payload(
    report: SVDCompressionReport,
) -> dict[str, Any]:
    """Serialize SVD compression structure."""
    return {
        "total_parameters": report.total_parameters,
        "compressed_layer_weight_parameters": (
            report.compressed_layer_weight_parameters
        ),
        "effective_stored_parameters": (report.effective_stored_parameters),
        "baseline_size_bytes": report.baseline_size_bytes,
        "compressed_size_bytes": report.compressed_size_bytes,
        "compression_ratio": report.compression_ratio,
        "storage_reduction_percent": (report.storage_reduction_percent),
        "parameter_reduction_percent": (report.parameter_reduction_percent),
        "layers": [
            {
                "name": layer.name,
                "input_dim": layer.input_dim,
                "output_dim": layer.output_dim,
                "rank": layer.rank,
                "maximum_rank": layer.maximum_rank,
                "profitable_max_rank": (layer.profitable_max_rank),
                "original_parameters": (layer.original_parameters),
                "compressed_parameters": (layer.compressed_parameters),
                "original_bytes": layer.original_bytes,
                "compressed_bytes": layer.compressed_bytes,
                "compression_ratio": layer.compression_ratio,
                "relative_reconstruction_error": (layer.relative_reconstruction_error),
            }
            for layer in report.layers
        ],
    }


def _result_payload(
    result: CompressionResult,
    report: SVDCompressionReport,
) -> dict[str, Any]:
    """Serialize one SVD experiment result."""
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
        "svd": _svd_report_payload(report),
        "notes": result.notes,
    }


def _save_result(
    result: CompressionResult,
    report: SVDCompressionReport,
    path: Path,
) -> None:
    """Save one SVD evidence artifact."""
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


def _run_domain(
    *,
    domain: Domain,
    model: SharedVLAModel,
    samples: list[TaskSample],
    evaluate: Callable[..., Any],
    measure_latency: Callable[..., Any],
    output_prefix: str,
) -> None:
    """Evaluate all SVD rank fractions for one domain."""
    frozen = load_seed_baseline(
        RESULTS_DIR,
        domain,
        SEED,
    )

    reproduced = evaluate(
        model=model,
        samples=samples,
        batch_size=BATCH_SIZE,
    )

    print()
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
        label=f"{output_prefix} MSE",
        frozen=frozen.test_mse,
        reproduced=reproduced.test_mse,
    )

    _verify_reproduced_metric(
        label=f"{output_prefix} MAE",
        frozen=frozen.test_mae,
        reproduced=reproduced.test_mae,
    )

    for rank_fraction in RANK_FRACTIONS:
        print()
        print(
            "Rank fraction:",
            rank_fraction,
        )

        start = time.perf_counter()

        compressed_model, report = compress_model_svd(
            model,
            rank_fraction=rank_fraction,
        )

        compression_seconds = time.perf_counter() - start

        compressed_quality = evaluate(
            model=compressed_model,
            samples=samples,
            batch_size=BATCH_SIZE,
        )

        compressed_latency = measure_latency(
            model=compressed_model,
            sample=samples[0],
        )

        metrics = build_compression_metrics(
            baseline_parameters=frozen.parameters,
            compressed_parameters=(report.effective_stored_parameters),
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

        percent = int(rank_fraction * 100)

        experiment_id = f"{output_prefix}-" f"svd-rf{percent}-" f"seed-{SEED}"

        result = CompressionResult(
            experiment_id=experiment_id,
            domain=domain,
            method=CompressionMethod.SVD,
            seed=SEED,
            metrics=metrics,
            configuration={
                "rank_fraction": rank_fraction,
                "rank_reference": ("maximum_profitable_rank"),
                "minimum_weight_parameters": 1024,
                "inference": ("fp32_reconstructed_dense_weights"),
                "selected_layer_count": len(report.layers),
            },
            notes=(
                "Post-training truncated SVD. Effective "
                "storage is estimated from U, singular-value, "
                "and Vh factors; inference uses reconstructed "
                "FP32 dense weights."
            ),
        )

        output_path = OUTPUT_DIR / f"{experiment_id}.json"

        _save_result(
            result,
            report,
            output_path,
        )

        print(
            "Effective parameters:",
            report.effective_stored_parameters,
        )
        print(
            "Compression ratio:",
            f"{metrics.compression_ratio:.4f}x",
        )
        print(
            "Parameter reduction:",
            f"{metrics.parameter_reduction_percent:.4f}%",
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
            "Dense reconstructed mean latency:",
            f"{compressed_latency.mean_ms:.4f} ms",
        )
        print(
            "Saved:",
            output_path,
        )


def main() -> None:
    """Run the complete seed-42 SVD baseline."""
    print()
    print("===== Q-VLA Forge " "Sprint 2.4 SVD Baseline =====")

    print()
    print("===== Driving =====")

    driving_model, driving_samples = _train_driving()

    _run_domain(
        domain=Domain.AUTONOMOUS_DRIVING,
        model=driving_model,
        samples=driving_samples,
        evaluate=evaluate_driving_model,
        measure_latency=measure_driving_latency,
        output_prefix="driving",
    )

    print()
    print("===== Robotics =====")

    robotics_model, robotics_samples = _train_robotics()

    _run_domain(
        domain=Domain.ROBOTICS,
        model=robotics_model,
        samples=robotics_samples,
        evaluate=evaluate_robotics_model,
        measure_latency=measure_robotics_latency,
        output_prefix="robotics",
    )

    print()
    print("===== SVD BASELINE COMPLETE =====")


if __name__ == "__main__":
    main()
