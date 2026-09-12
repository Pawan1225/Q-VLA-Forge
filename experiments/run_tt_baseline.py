"""Run seed-42 Tensor Train compression experiments for both pilot domains."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from q_vla_forge.compression import (
    CompressionMethod,
    CompressionResult,
    TTCompressionReport,
    build_compression_metrics,
    compress_model_tt,
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

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MIN_LEARNING_RATE = 1e-5

TT_RANKS = (
    2,
    4,
    8,
)

TENSOR_ORDER = 3
MINIMUM_WEIGHT_PARAMETERS = 1024


def _training_config() -> TrainingConfig:
    """Return the frozen Sprint 1 training configuration."""
    return TrainingConfig(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        min_learning_rate=MIN_LEARNING_RATE,
        seed=SEED,
    )


def _train_driving() -> tuple[
    SharedVLAModel,
    list[TaskSample],
]:
    """Train the frozen seed-42 autonomous-driving baseline."""
    set_seed(SEED)

    train_dataset = SyntheticDrivingDataset(
        size=TRAIN_SIZE,
        seed=SEED,
    )

    validation_dataset = SyntheticDrivingDataset(
        size=VALIDATION_SIZE,
        seed=SEED + 1000,
    )

    test_dataset = SyntheticDrivingDataset(
        size=TEST_SIZE,
        seed=SEED + 2000,
    )

    model = SharedVLAModel()

    train_supervised(
        model,
        list(train_dataset),
        list(validation_dataset),
        Domain.AUTONOMOUS_DRIVING,
        _training_config(),
    )

    return (
        model,
        list(test_dataset),
    )


def _train_robotics() -> tuple[
    SharedVLAModel,
    list[TaskSample],
]:
    """Train the frozen seed-42 robotics baseline."""
    set_seed(SEED)

    train_dataset = SyntheticRoboticsDataset(
        size=TRAIN_SIZE,
        seed=SEED,
    )

    validation_dataset = SyntheticRoboticsDataset(
        size=VALIDATION_SIZE,
        seed=SEED + 1000,
    )

    test_dataset = SyntheticRoboticsDataset(
        size=TEST_SIZE,
        seed=SEED + 2000,
    )

    model = SharedVLAModel()

    train_supervised(
        model,
        list(train_dataset),
        list(validation_dataset),
        Domain.ROBOTICS,
        _training_config(),
    )

    return (
        model,
        list(test_dataset),
    )


def _serialize_tt_report(
    report: TTCompressionReport,
) -> dict[str, Any]:
    """Serialize Tensor Train structure for experiment evidence."""
    return {
        "requested_rank": report.requested_rank,
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
                "row_factors": list(layer.row_factors),
                "column_factors": list(layer.column_factors),
                "tensor_shape": list(layer.tensor_shape),
                "requested_rank": layer.requested_rank,
                "actual_ranks": list(layer.actual_ranks),
                "original_parameters": (layer.original_parameters),
                "compressed_parameters": (layer.compressed_parameters),
                "original_bytes": layer.original_bytes,
                "compressed_bytes": layer.compressed_bytes,
                "compression_ratio": (layer.compression_ratio),
                "relative_reconstruction_error": (layer.relative_reconstruction_error),
            }
            for layer in report.layers
        ],
    }


def _save_result(
    result: CompressionResult,
    report: TTCompressionReport,
    path: Path,
) -> None:
    """Persist one Tensor Train experiment result."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = asdict(result)

    payload["domain"] = result.domain.value
    payload["method"] = result.method.value
    payload["tt"] = _serialize_tt_report(report)

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def _assert_baseline_matches(
    *,
    frozen_mse: float,
    reproduced_mse: float,
    frozen_mae: float,
    reproduced_mae: float,
) -> None:
    """Verify frozen-baseline reproduction before compression."""
    if not math.isclose(
        frozen_mse,
        reproduced_mse,
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise RuntimeError(
            "reproduced baseline MSE does not match " "the frozen Sprint 1 reference"
        )

    if not math.isclose(
        frozen_mae,
        reproduced_mae,
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise RuntimeError(
            "reproduced baseline MAE does not match " "the frozen Sprint 1 reference"
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
    """Run Tensor Train compression sweeps for one domain."""
    frozen = load_seed_baseline(
        RESULTS_DIR,
        domain,
        SEED,
    )

    local_fp32 = evaluate(
        model,
        samples,
    )

    print()
    print(
        "Frozen FP32 MSE:",
        frozen.test_mse,
    )

    print(
        "Reproduced FP32 MSE:",
        local_fp32.test_mse,
    )

    print(
        "Frozen FP32 MAE:",
        frozen.test_mae,
    )

    print(
        "Reproduced FP32 MAE:",
        local_fp32.test_mae,
    )

    _assert_baseline_matches(
        frozen_mse=frozen.test_mse,
        reproduced_mse=local_fp32.test_mse,
        frozen_mae=frozen.test_mae,
        reproduced_mae=local_fp32.test_mae,
    )

    for rank in TT_RANKS:
        print()
        print(
            "TT rank:",
            rank,
        )

        start = time.perf_counter()

        (
            compressed_model,
            report,
        ) = compress_model_tt(
            model,
            max_rank=rank,
            minimum_weight_parameters=(MINIMUM_WEIGHT_PARAMETERS),
            tensor_order=TENSOR_ORDER,
        )

        compression_seconds = time.perf_counter() - start

        compressed_metrics = evaluate(
            compressed_model,
            samples,
        )

        compressed_latency = measure_latency(
            compressed_model,
            samples[0],
        )

        metrics = build_compression_metrics(
            baseline_parameters=(frozen.parameters),
            compressed_parameters=(report.effective_stored_parameters),
            baseline_size_bytes=(frozen.fp32_model_size_bytes),
            compressed_size_bytes=(report.compressed_size_bytes),
            baseline_test_mse=(frozen.test_mse),
            compressed_test_mse=(compressed_metrics.test_mse),
            baseline_test_mae=(frozen.test_mae),
            compressed_test_mae=(compressed_metrics.test_mae),
            baseline_mean_latency_ms=(frozen.mean_latency_ms),
            compressed_mean_latency_ms=(compressed_latency.mean_ms),
            baseline_p95_latency_ms=(frozen.p95_latency_ms),
            compressed_p95_latency_ms=(compressed_latency.p95_ms),
            compression_time_seconds=(compression_seconds),
        )

        experiment_id = f"{output_prefix}-" f"tt-rank-{rank}-" f"seed-{SEED}"

        result = CompressionResult(
            experiment_id=experiment_id,
            domain=domain,
            method=CompressionMethod.TENSOR_TRAIN,
            seed=SEED,
            metrics=metrics,
            configuration={
                "algorithm": "tt_svd",
                "tensor_order": TENSOR_ORDER,
                "requested_max_rank": rank,
                "minimum_weight_parameters": (MINIMUM_WEIGHT_PARAMETERS),
                "inference": ("fp32_reconstructed_weights"),
                "native_tt_inference": False,
            },
            notes=(
                "Quantum-inspired Tensor Train compression "
                "using TT-SVD. Effective storage is measured "
                "from TT cores. Runtime uses reconstructed "
                "FP32 dense weights rather than native TT "
                "core execution."
            ),
        )

        output = OUTPUT_DIR / f"{experiment_id}.json"

        _save_result(
            result,
            report,
            output,
        )

        print(
            "Selected layers:",
            len(report.layers),
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
            compressed_metrics.test_mse,
        )

        print(
            "MSE change:",
            f"{metrics.mse_change_percent:.4f}%",
        )

        print(
            "Compressed MAE:",
            compressed_metrics.test_mae,
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
            output,
        )


def main() -> None:
    """Run Tensor Train compression experiments."""
    print()
    print("===== Q-VLA Forge Sprint 2.5 " "TT / TT-SVD Baseline =====")

    print()
    print("===== Driving =====")

    (
        driving_model,
        driving_samples,
    ) = _train_driving()

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

    (
        robotics_model,
        robotics_samples,
    ) = _train_robotics()

    _run_domain(
        domain=Domain.ROBOTICS,
        model=robotics_model,
        samples=robotics_samples,
        evaluate=evaluate_robotics_model,
        measure_latency=measure_robotics_latency,
        output_prefix="robotics",
    )

    print()
    print("===== TT / TT-SVD BASELINE COMPLETE =====")


if __name__ == "__main__":
    main()
