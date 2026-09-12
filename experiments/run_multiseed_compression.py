"""Run paired three-seed compression validation."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from q_vla_forge.compression import (
    CompressionMethod,
    CompressionResult,
    build_compression_metrics,
    compress_model_svd,
    compress_model_tt,
    quantize_model_weights_int8,
)
from q_vla_forge.data import (
    Domain,
    SyntheticDrivingDataset,
    SyntheticRoboticsDataset,
    TaskSample,
)
from q_vla_forge.evaluation import (
    DomainCompressionValidation,
    MultiSeedCompressionValidation,
    aggregate_method_results,
    save_multiseed_compression_validation,
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
from q_vla_forge.utils.reproducibility import (
    DEFAULT_SEEDS,
    set_seed,
)

RESULTS_DIR = Path("results")

COMPRESSION_DIR = RESULTS_DIR / "compression"

VALIDATION_DIR = COMPRESSION_DIR / "validation"

DRIVING_SELECTION_PATH = COMPRESSION_DIR / "driving-compression-selection.json"

ROBOTICS_SELECTION_PATH = COMPRESSION_DIR / "robotics-compression-selection.json"

SUMMARY_PATH = COMPRESSION_DIR / "compression-validation-summary.json"

TRAIN_SIZE = 512
VALIDATION_SIZE = 128
TEST_SIZE = 128

EPOCHS = 20
BATCH_SIZE = 32

MINIMUM_COMPRESSION_RATIO = 2.0
MAXIMUM_MSE_INCREASE_PERCENT = 5.0


def _load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one JSON artifact."""
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))


def _training_config(
    seed: int,
) -> TrainingConfig:
    """Build the frozen Sprint 1 training configuration."""
    return TrainingConfig(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=1e-3,
        weight_decay=1e-4,
        min_learning_rate=1e-5,
        seed=seed,
    )


def _selection_configuration(
    selection: dict[str, Any],
    key: str,
) -> tuple[
    str,
    dict[str, Any],
]:
    """Load the frozen configuration selected during seed-42 screening."""
    selected = selection[key]["selected"]

    source_path = COMPRESSION_DIR / selected["source_file"]

    source = _load_json(source_path)

    return (
        str(selected["configuration_label"]),
        source["configuration"],
    )


def _save_result(
    result: CompressionResult,
    path: Path,
) -> None:
    """Save one per-seed compression result."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = asdict(result)

    payload["method"] = result.method.value

    payload["domain"] = (
        result.domain.value
        if hasattr(
            result.domain,
            "value",
        )
        else result.domain
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def _train_domain(
    *,
    domain: Domain,
    seed: int,
) -> tuple[
    SharedVLAModel,
    list[TaskSample],
]:
    """Train one FP32 baseline model for one domain and seed."""
    set_seed(seed)

    if domain == Domain.AUTONOMOUS_DRIVING:
        dataset_class = SyntheticDrivingDataset
    elif domain == Domain.ROBOTICS:
        dataset_class = SyntheticRoboticsDataset
    else:
        raise ValueError(f"unsupported domain: {domain}")

    train_dataset = dataset_class(
        size=TRAIN_SIZE,
        seed=seed,
    )

    validation_dataset = dataset_class(
        size=VALIDATION_SIZE,
        seed=seed + 1000,
    )

    test_dataset = dataset_class(
        size=TEST_SIZE,
        seed=seed + 2000,
    )

    model = SharedVLAModel()

    train_supervised(
        model,
        list(train_dataset),
        list(validation_dataset),
        domain,
        _training_config(seed),
    )

    return (
        model,
        list(test_dataset),
    )


def _evaluate_domain(
    *,
    domain: Domain,
) -> tuple[
    Callable[..., Any],
    Callable[..., Any],
]:
    """Return evaluation and latency functions for a domain."""
    if domain == Domain.AUTONOMOUS_DRIVING:
        return (
            evaluate_driving_model,
            measure_driving_latency,
        )

    if domain == Domain.ROBOTICS:
        return (
            evaluate_robotics_model,
            measure_robotics_latency,
        )

    raise ValueError(f"unsupported domain: {domain}")


def _build_result(
    *,
    experiment_id: str,
    domain: Domain,
    method: CompressionMethod,
    seed: int,
    configuration: dict[str, Any],
    baseline_metrics: Any,
    baseline_latency: Any,
    compressed_metrics: Any,
    compressed_latency: Any,
    baseline_parameters: int,
    compressed_parameters: int,
    baseline_size_bytes: int,
    compressed_size_bytes: int,
    compression_seconds: float,
    notes: str,
) -> CompressionResult:
    """Construct a normalized compression result."""
    metrics = build_compression_metrics(
        baseline_parameters=(baseline_parameters),
        compressed_parameters=(compressed_parameters),
        baseline_size_bytes=(baseline_size_bytes),
        compressed_size_bytes=(compressed_size_bytes),
        baseline_test_mse=(baseline_metrics.test_mse),
        compressed_test_mse=(compressed_metrics.test_mse),
        baseline_test_mae=(baseline_metrics.test_mae),
        compressed_test_mae=(compressed_metrics.test_mae),
        baseline_mean_latency_ms=(baseline_latency.mean_ms),
        compressed_mean_latency_ms=(compressed_latency.mean_ms),
        baseline_p95_latency_ms=(baseline_latency.p95_ms),
        compressed_p95_latency_ms=(compressed_latency.p95_ms),
        compression_time_seconds=(compression_seconds),
    )

    return CompressionResult(
        experiment_id=(experiment_id),
        domain=domain,
        method=method,
        seed=seed,
        metrics=metrics,
        configuration=configuration,
        notes=notes,
    )


def _run_seed(
    *,
    domain: Domain,
    seed: int,
    svd_label: str,
    svd_configuration: dict[str, Any],
    tt_label: str,
    tt_configuration: dict[str, Any],
) -> dict[str, Path]:
    """Train one baseline and evaluate all selected compression methods."""
    print()
    print("========================================")

    print(
        "Domain:",
        domain.value,
    )

    print(
        "Seed:",
        seed,
    )

    model, samples = _train_domain(
        domain=domain,
        seed=seed,
    )

    (
        evaluate,
        measure_latency,
    ) = _evaluate_domain(domain=domain)

    baseline_metrics = evaluate(
        model,
        samples,
    )

    baseline_latency = measure_latency(
        model,
        samples[0],
    )

    baseline_parameters = sum(parameter.numel() for parameter in model.parameters())

    baseline_size_bytes = baseline_parameters * 4

    prefix = "driving" if domain == Domain.AUTONOMOUS_DRIVING else "robotics"

    outputs: dict[
        str,
        Path,
    ] = {}

    # INT8
    start = time.perf_counter()

    (
        int8_model,
        int8_report,
    ) = quantize_model_weights_int8(model)

    compression_seconds = time.perf_counter() - start

    int8_metrics = evaluate(
        int8_model,
        samples,
    )

    int8_latency = measure_latency(
        int8_model,
        samples[0],
    )

    int8_result = _build_result(
        experiment_id=(f"{prefix}-validation-" f"int8-seed-{seed}"),
        domain=domain,
        method=CompressionMethod.INT8,
        seed=seed,
        configuration={
            "label": "INT8",
            "scheme": ("symmetric_per_tensor_weight_int8"),
            "inference": ("fp32_dequantized_weights"),
        },
        baseline_metrics=(baseline_metrics),
        baseline_latency=(baseline_latency),
        compressed_metrics=(int8_metrics),
        compressed_latency=(int8_latency),
        baseline_parameters=(baseline_parameters),
        compressed_parameters=(baseline_parameters),
        baseline_size_bytes=(baseline_size_bytes),
        compressed_size_bytes=(int8_report.compressed_size_bytes),
        compression_seconds=(compression_seconds),
        notes=(
            "Three-seed validation of classical "
            "symmetric per-tensor INT8 weight "
            "quantization with FP32 dequantized inference."
        ),
    )

    int8_path = VALIDATION_DIR / (f"{prefix}-int8-" f"seed-{seed}.json")

    _save_result(
        int8_result,
        int8_path,
    )

    outputs["int8"] = int8_path

    # SVD
    rank_fraction = float(svd_configuration["rank_fraction"])

    minimum_weight_parameters = int(
        svd_configuration.get(
            "minimum_weight_parameters",
            1024,
        )
    )

    start = time.perf_counter()

    (
        svd_model,
        svd_report,
    ) = compress_model_svd(
        model,
        rank_fraction=(rank_fraction),
        minimum_weight_parameters=(minimum_weight_parameters),
    )

    compression_seconds = time.perf_counter() - start

    svd_metrics = evaluate(
        svd_model,
        samples,
    )

    svd_latency = measure_latency(
        svd_model,
        samples[0],
    )

    svd_result = _build_result(
        experiment_id=(f"{prefix}-validation-" f"svd-seed-{seed}"),
        domain=domain,
        method=CompressionMethod.SVD,
        seed=seed,
        configuration={
            "label": svd_label,
            "rank_fraction": (rank_fraction),
            "rank_reference": ("maximum_profitable_rank"),
            "minimum_weight_parameters": (minimum_weight_parameters),
            "inference": ("fp32_reconstructed_weights"),
        },
        baseline_metrics=(baseline_metrics),
        baseline_latency=(baseline_latency),
        compressed_metrics=(svd_metrics),
        compressed_latency=(svd_latency),
        baseline_parameters=(baseline_parameters),
        compressed_parameters=(svd_report.effective_stored_parameters),
        baseline_size_bytes=(baseline_size_bytes),
        compressed_size_bytes=(svd_report.compressed_size_bytes),
        compression_seconds=(compression_seconds),
        notes=(
            "Three-seed validation of the frozen " "post-training SVD configuration."
        ),
    )

    svd_path = VALIDATION_DIR / (f"{prefix}-svd-" f"seed-{seed}.json")

    _save_result(
        svd_result,
        svd_path,
    )

    outputs["svd"] = svd_path

    # TT / MPS
    tt_rank = int(tt_configuration["requested_max_rank"])

    tensor_order = int(
        tt_configuration.get(
            "tensor_order",
            3,
        )
    )

    tt_minimum_weight_parameters = int(
        tt_configuration.get(
            "minimum_weight_parameters",
            1024,
        )
    )

    start = time.perf_counter()

    (
        tt_model,
        tt_report,
    ) = compress_model_tt(
        model,
        max_rank=(tt_rank),
        minimum_weight_parameters=(tt_minimum_weight_parameters),
        tensor_order=(tensor_order),
    )

    compression_seconds = time.perf_counter() - start

    tt_metrics = evaluate(
        tt_model,
        samples,
    )

    tt_latency = measure_latency(
        tt_model,
        samples[0],
    )

    tt_result = _build_result(
        experiment_id=(f"{prefix}-validation-" f"tt-seed-{seed}"),
        domain=domain,
        method=(CompressionMethod.TENSOR_TRAIN),
        seed=seed,
        configuration={
            "label": tt_label,
            "algorithm": "tt_svd",
            "representation": ("tt_open_boundary_mps"),
            "requested_max_rank": (tt_rank),
            "tensor_order": (tensor_order),
            "minimum_weight_parameters": (tt_minimum_weight_parameters),
            "inference": ("fp32_reconstructed_weights"),
        },
        baseline_metrics=(baseline_metrics),
        baseline_latency=(baseline_latency),
        compressed_metrics=(tt_metrics),
        compressed_latency=(tt_latency),
        baseline_parameters=(baseline_parameters),
        compressed_parameters=(tt_report.effective_stored_parameters),
        baseline_size_bytes=(baseline_size_bytes),
        compressed_size_bytes=(tt_report.compressed_size_bytes),
        compression_seconds=(compression_seconds),
        notes=(
            "Three-seed validation of the frozen "
            "quantum-inspired TT/MPS configuration "
            "constructed with TT-SVD."
        ),
    )

    tt_path = VALIDATION_DIR / (f"{prefix}-tt-" f"seed-{seed}.json")

    _save_result(
        tt_result,
        tt_path,
    )

    outputs["tensor_train"] = tt_path

    print()
    print(f"{'Method':12}" f"{'Ratio':>10}" f"{'MSE Δ':>12}" f"{'MAE Δ':>12}")

    print("-" * 46)

    for name, result in (
        (
            "INT8",
            int8_result,
        ),
        (
            svd_label,
            svd_result,
        ),
        (
            tt_label,
            tt_result,
        ),
    ):
        print(
            f"{name:12}"
            f"{result.metrics.compression_ratio:>9.3f}x"
            f"{result.metrics.mse_change_percent:>11.3f}%"
            f"{result.metrics.mae_change_percent:>11.3f}%"
        )

    return outputs


def _aggregate_domain(
    *,
    domain: Domain,
    prefix: str,
    svd_label: str,
    tt_label: str,
) -> DomainCompressionValidation:
    """Aggregate the three frozen-seed compression runs."""
    seeds = tuple(int(seed) for seed in DEFAULT_SEEDS)

    int8_paths = [VALIDATION_DIR / f"{prefix}-int8-seed-{seed}.json" for seed in seeds]

    svd_paths = [VALIDATION_DIR / f"{prefix}-svd-seed-{seed}.json" for seed in seeds]

    tt_paths = [VALIDATION_DIR / f"{prefix}-tt-seed-{seed}.json" for seed in seeds]

    int8 = aggregate_method_results(
        int8_paths,
        expected_domain=(domain.value),
        expected_method="int8",
        configuration="INT8",
        expected_seeds=seeds,
        minimum_compression_ratio=(MINIMUM_COMPRESSION_RATIO),
        maximum_mse_increase_percent=(MAXIMUM_MSE_INCREASE_PERCENT),
    )

    svd = aggregate_method_results(
        svd_paths,
        expected_domain=(domain.value),
        expected_method="svd",
        configuration=svd_label,
        expected_seeds=seeds,
        minimum_compression_ratio=(MINIMUM_COMPRESSION_RATIO),
        maximum_mse_increase_percent=(MAXIMUM_MSE_INCREASE_PERCENT),
    )

    tensor_network = aggregate_method_results(
        tt_paths,
        expected_domain=(domain.value),
        expected_method=("tensor_train"),
        configuration=tt_label,
        expected_seeds=seeds,
        minimum_compression_ratio=(MINIMUM_COMPRESSION_RATIO),
        maximum_mse_increase_percent=(MAXIMUM_MSE_INCREASE_PERCENT),
    )

    return DomainCompressionValidation(
        domain=(domain.value),
        seeds=seeds,
        minimum_compression_ratio=(MINIMUM_COMPRESSION_RATIO),
        maximum_mse_increase_percent=(MAXIMUM_MSE_INCREASE_PERCENT),
        int8=int8,
        svd=svd,
        tensor_network=(tensor_network),
    )


def _print_summary(
    validation: DomainCompressionValidation,
) -> None:
    """Print compact proposal-oriented mean ± SD results."""
    print()
    print("========================================")

    print(validation.domain.upper())

    print("========================================")

    print(f"{'Method':14}" f"{'Compression':>22}" f"{'MSE Δ':>22}" f"{'Feasible':>12}")

    print("-" * 72)

    for method in (
        validation.int8,
        validation.svd,
        validation.tensor_network,
    ):
        compression = (
            f"{method.compression_ratio.mean:.3f}"
            f" ± "
            f"{method.compression_ratio.std:.3f}x"
        )

        mse = (
            f"{method.mse_change_percent.mean:.3f}"
            f" ± "
            f"{method.mse_change_percent.std:.3f}%"
        )

        feasible = f"{method.pilot_feasible_runs}/3"

        print(
            f"{method.configuration:14}"
            f"{compression:>22}"
            f"{mse:>22}"
            f"{feasible:>12}"
        )


def main() -> None:
    """Run three-seed validation for both domains."""
    driving_selection = _load_json(DRIVING_SELECTION_PATH)

    robotics_selection = _load_json(ROBOTICS_SELECTION_PATH)

    (
        driving_svd_label,
        driving_svd_configuration,
    ) = _selection_configuration(
        driving_selection,
        "svd",
    )

    (
        driving_tt_label,
        driving_tt_configuration,
    ) = _selection_configuration(
        driving_selection,
        "tensor_network",
    )

    (
        robotics_svd_label,
        robotics_svd_configuration,
    ) = _selection_configuration(
        robotics_selection,
        "svd",
    )

    (
        robotics_tt_label,
        robotics_tt_configuration,
    ) = _selection_configuration(
        robotics_selection,
        "tensor_network",
    )

    print()
    print("===== Frozen Configurations =====")

    print(
        "Driving SVD:",
        driving_svd_label,
    )

    print(
        "Driving TT/MPS:",
        driving_tt_label,
    )

    print(
        "Robotics SVD:",
        robotics_svd_label,
    )

    print(
        "Robotics TT/MPS:",
        robotics_tt_label,
    )

    for seed in DEFAULT_SEEDS:
        _run_seed(
            domain=(Domain.AUTONOMOUS_DRIVING),
            seed=int(seed),
            svd_label=(driving_svd_label),
            svd_configuration=(driving_svd_configuration),
            tt_label=(driving_tt_label),
            tt_configuration=(driving_tt_configuration),
        )

        _run_seed(
            domain=(Domain.ROBOTICS),
            seed=int(seed),
            svd_label=(robotics_svd_label),
            svd_configuration=(robotics_svd_configuration),
            tt_label=(robotics_tt_label),
            tt_configuration=(robotics_tt_configuration),
        )

    driving = _aggregate_domain(
        domain=(Domain.AUTONOMOUS_DRIVING),
        prefix="driving",
        svd_label=(driving_svd_label),
        tt_label=(driving_tt_label),
    )

    robotics = _aggregate_domain(
        domain=(Domain.ROBOTICS),
        prefix="robotics",
        svd_label=(robotics_svd_label),
        tt_label=(robotics_tt_label),
    )

    validation = MultiSeedCompressionValidation(
        seeds=tuple(int(seed) for seed in DEFAULT_SEEDS),
        driving=driving,
        robotics=robotics,
    )

    save_multiseed_compression_validation(
        validation,
        SUMMARY_PATH,
    )

    _print_summary(driving)

    _print_summary(robotics)

    print()
    print(
        "Saved:",
        SUMMARY_PATH,
    )


if __name__ == "__main__":
    main()
