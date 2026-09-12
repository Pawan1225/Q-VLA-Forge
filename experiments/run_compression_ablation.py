from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from q_vla_forge.compression import (
    CompressionTarget,
    compress_model_svd,
    compress_model_tt,
    selected_linear_layer_names,
)
from q_vla_forge.data import (
    Domain,
    SyntheticDrivingDataset,
    SyntheticRoboticsDataset,
)
from q_vla_forge.evaluation.driving_baseline import (
    evaluate_driving_model,
)
from q_vla_forge.evaluation.robotics_baseline import (
    evaluate_robotics_model,
)
from q_vla_forge.models import SharedVLAModel
from q_vla_forge.training import (
    TrainingConfig,
    train_supervised,
)
from q_vla_forge.utils.reproducibility import (
    set_seed,
)

COMPRESSION_DIR = Path("results") / "compression"
ABLATION_DIR = COMPRESSION_DIR / "ablation"

DRIVING_SELECTION = COMPRESSION_DIR / "driving-compression-selection.json"

ROBOTICS_SELECTION = COMPRESSION_DIR / "robotics-compression-selection.json"

SEED = 42

TRAIN_SIZE = 512
VALIDATION_SIZE = 128
TEST_SIZE = 128

TARGETS = (
    CompressionTarget.FUSION,
    CompressionTarget.LATENT,
    CompressionTarget.ACTION,
    CompressionTarget.FUSION_LATENT,
    CompressionTarget.ALL,
)


def _load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))


def _selected_configuration(
    selection: dict[str, Any],
    key: str,
) -> tuple[str, dict[str, Any]]:
    selected = selection[key]["selected"]

    source = _load_json(COMPRESSION_DIR / selected["source_file"])

    return (
        str(selected["configuration_label"]),
        source["configuration"],
    )


def _train(
    domain: Domain,
) -> tuple[
    SharedVLAModel,
    list,
]:
    set_seed(SEED)

    dataset_class = (
        SyntheticDrivingDataset
        if domain == Domain.AUTONOMOUS_DRIVING
        else SyntheticRoboticsDataset
    )

    train_dataset = dataset_class(
        size=TRAIN_SIZE,
        seed=SEED,
    )

    validation_dataset = dataset_class(
        size=VALIDATION_SIZE,
        seed=SEED + 1000,
    )

    test_dataset = dataset_class(
        size=TEST_SIZE,
        seed=SEED + 2000,
    )

    model = SharedVLAModel()

    config = TrainingConfig(
        epochs=20,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        min_learning_rate=1e-5,
        seed=SEED,
    )

    train_supervised(
        model,
        list(train_dataset),
        list(validation_dataset),
        domain,
        config,
    )

    return (
        model,
        list(test_dataset),
    )


def _evaluate(
    domain: Domain,
    model: SharedVLAModel,
    samples: list,
):
    if domain == Domain.AUTONOMOUS_DRIVING:
        return evaluate_driving_model(
            model,
            samples,
        )

    return evaluate_robotics_model(
        model,
        samples,
    )


def _percent_change(
    baseline: float,
    compressed: float,
) -> float:
    if baseline <= 0.0:
        raise ValueError("baseline must be positive")

    return (compressed - baseline) / baseline * 100.0


def _run_domain(
    domain: Domain,
    selection_path: Path,
) -> list[dict[str, Any]]:
    selection = _load_json(selection_path)

    (
        svd_label,
        svd_config,
    ) = _selected_configuration(
        selection,
        "svd",
    )

    (
        tt_label,
        tt_config,
    ) = _selected_configuration(
        selection,
        "tensor_network",
    )

    model, test_samples = _train(domain)

    baseline = _evaluate(
        domain,
        model,
        test_samples,
    )

    baseline_parameters = sum(parameter.numel() for parameter in model.parameters())

    baseline_bytes = baseline_parameters * 4

    results: list[dict[str, Any]] = []

    for target in TARGETS:
        selected_layers = set(
            selected_linear_layer_names(
                model,
                target,
                minimum_weight_parameters=1024,
            )
        )

        # -----------------------------
        # SVD
        # -----------------------------

        (
            svd_model,
            svd_report,
        ) = compress_model_svd(
            model,
            rank_fraction=float(svd_config["rank_fraction"]),
            minimum_weight_parameters=int(
                svd_config.get(
                    "minimum_weight_parameters",
                    1024,
                )
            ),
            selected_layer_names=(selected_layers),
        )

        svd_metrics = _evaluate(
            domain,
            svd_model,
            test_samples,
        )

        results.append(
            {
                "domain": domain.value,
                "seed": SEED,
                "method": "svd",
                "configuration": svd_label,
                "target": target.value,
                "selected_layers": sorted(selected_layers),
                "compressed_layers": [layer.name for layer in svd_report.layers],
                "baseline_parameters": (baseline_parameters),
                "effective_stored_parameters": (svd_report.effective_stored_parameters),
                "baseline_size_bytes": (baseline_bytes),
                "compressed_size_bytes": (svd_report.compressed_size_bytes),
                "compression_ratio": (
                    baseline_bytes / svd_report.compressed_size_bytes
                ),
                "baseline_test_mse": (baseline.test_mse),
                "compressed_test_mse": (svd_metrics.test_mse),
                "mse_change_percent": (
                    _percent_change(
                        baseline.test_mse,
                        svd_metrics.test_mse,
                    )
                ),
                "baseline_test_mae": (baseline.test_mae),
                "compressed_test_mae": (svd_metrics.test_mae),
                "mae_change_percent": (
                    _percent_change(
                        baseline.test_mae,
                        svd_metrics.test_mae,
                    )
                ),
            }
        )

        # -----------------------------
        # TT / MPS
        # -----------------------------

        (
            tt_model,
            tt_report,
        ) = compress_model_tt(
            model,
            max_rank=int(tt_config["requested_max_rank"]),
            minimum_weight_parameters=int(
                tt_config.get(
                    "minimum_weight_parameters",
                    1024,
                )
            ),
            tensor_order=int(
                tt_config.get(
                    "tensor_order",
                    3,
                )
            ),
            selected_layer_names=(selected_layers),
        )

        tt_metrics = _evaluate(
            domain,
            tt_model,
            test_samples,
        )

        results.append(
            {
                "domain": domain.value,
                "seed": SEED,
                "method": "tensor_train",
                "representation": ("open_boundary_mps"),
                "configuration": tt_label,
                "target": target.value,
                "selected_layers": sorted(selected_layers),
                "compressed_layers": [layer.name for layer in tt_report.layers],
                "baseline_parameters": (baseline_parameters),
                "effective_stored_parameters": (tt_report.effective_stored_parameters),
                "baseline_size_bytes": (baseline_bytes),
                "compressed_size_bytes": (tt_report.compressed_size_bytes),
                "compression_ratio": (baseline_bytes / tt_report.compressed_size_bytes),
                "baseline_test_mse": (baseline.test_mse),
                "compressed_test_mse": (tt_metrics.test_mse),
                "mse_change_percent": (
                    _percent_change(
                        baseline.test_mse,
                        tt_metrics.test_mse,
                    )
                ),
                "baseline_test_mae": (baseline.test_mae),
                "compressed_test_mae": (tt_metrics.test_mae),
                "mae_change_percent": (
                    _percent_change(
                        baseline.test_mae,
                        tt_metrics.test_mae,
                    )
                ),
            }
        )

    return results


def main() -> None:
    ABLATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    results.extend(
        _run_domain(
            Domain.AUTONOMOUS_DRIVING,
            DRIVING_SELECTION,
        )
    )

    results.extend(
        _run_domain(
            Domain.ROBOTICS,
            ROBOTICS_SELECTION,
        )
    )

    if len(results) != 20:
        raise RuntimeError(f"expected 20 ablation points, " f"found {len(results)}")

    output = ABLATION_DIR / "compression-ablation-seed-42.json"

    output.write_text(
        json.dumps(
            {
                "seed": SEED,
                "points": results,
                "notes": (
                    "Architectural target ablation. "
                    "SVD is the classical low-rank comparator; "
                    "TT/MPS is the quantum-inspired tensor-network "
                    "method. Only profitable eligible layers are "
                    "counted as compressed."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("===== Compression Ablation =====")

    print(
        f"{'Domain':22}"
        f"{'Method':16}"
        f"{'Target':18}"
        f"{'Ratio':>10}"
        f"{'MSE Δ':>12}"
    )

    print("-" * 78)

    for result in results:
        print(
            f"{result['domain']:22}"
            f"{result['method']:16}"
            f"{result['target']:18}"
            f"{result['compression_ratio']:>9.3f}x"
            f"{result['mse_change_percent']:>11.3f}%"
        )

    print()

    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()
